# -*- coding: utf-8 -*-
"""Stage `index` (phase D③): tantivy + date-sorted memmap vectors over the four upstream stages (§9.1).

Three indexes under <derived>/index/ (read side and schemas in index/search.py):
  papers      every active registry paper with a first public date — title (boosted) + abstract, tier flags
              (t0 always / t1 has abstract / t2 has a parsed full text / ft has a careful-or-Sciverse text),
              day_hi = first_hi
  passages    the canonical reading's sentences per paper (extract.reading.full_text — tier priority
              sciverse -> mineru -> grobid, delta sentences appended), each at its own sentence date (v2.5)
  statements  every extract statement (self + other): filters kind/facet/role/epistemic/about/speaker,
              body = text + quote, day_hi = the statement's date

Vectors (§9.1 v1 scope): `papers` = the extract-scope papers with title+abstract; `statements` = the self
statements. Rows live in vec_rows (the incremental embedding cache — an item is re-embedded only when its
inputs changed) and are exported after every build to the date-sorted memmap files search.py reads.

Incrementality: the papers pass keeps its own streaming fingerprint table (papers_fp — 1.19M items must not
materialise in RAM); passages/statements/vectors are dfc work passes keyed by paper. tantivy re-indexing is
delete-then-add in chunks with a commit BETWEEN the two (a delete and an add of the same term inside one
commit is engine-undefined — the new doc can die with the old). A SCHEMA_V bump or --rebuild wipes the
sidecar directory and every bookkeeping row; a missing dir or missing bookkeeping resets the other side, so
the two can never drift into duplicates or phantoms.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..core import paths
from ..dfc import store
from ..documents.build import Documents
from ..extract import reading as RD
from ..extract.schema import FACETS
from .search import Index, day_int, index_dir, make_schema, vectors_dir  # noqa: F401  (Index re-export: tools/api)

SCHEMA_V = 3          # bump on any tantivy-schema or vector-layout change: reopens every item
CHUNK = 500           # papers per delete-commit / add-commit cycle (tantivy + work passes)
EMBED_BATCH = 256     # texts per embed call (items are never split across calls)
EXPORT_CHUNK = 16384

DDL = """CREATE TABLE IF NOT EXISTS vec_rows(name TEXT, rowkey TEXT, day_hi INT, owner TEXT, facet INT,
  item TEXT, vec BLOB, PRIMARY KEY(name, rowkey));
CREATE INDEX IF NOT EXISTS ix_vec_item ON vec_rows(name, item);
CREATE TABLE IF NOT EXISTS vec_meta(name TEXT PRIMARY KEY, model TEXT, dim INT);
CREATE TABLE IF NOT EXISTS papers_fp(paper_id TEXT PRIMARY KEY, fp TEXT);"""


def build(dense=("papers", "statements"), embed=None, model_label: str | None = None, rebuild: bool = False,
          log=print) -> dict:
    """`dense`: which vector stores to build. `embed`: callable(list[str]) -> list[vector] (default: the local
    embedding service); `model_label`: recorded in the vector headers — the query side refuses a different
    model. The index is a pure function of its upstreams + the registry, so a rebuild is always safe."""
    dense = tuple(sorted(set(dense)))
    if dense:
        if embed is None:
            from ..llm.embedding import embed_local, model_label as _ml
            embed = embed_local
            model_label = model_label or _ml()
        elif model_label is None:
            raise ValueError("model_label is required when embed is injected")
    params = {"v": SCHEMA_V, "dense": list(dense), "embed_model": model_label if dense else None}
    idir = index_dir()
    m = store.read_manifest("index")
    bump = bool(m) and (m.get("params") or {}).get("v") != SCHEMA_V
    with store.Run("index", params, rebuild=rebuild) as run:
        if rebuild or bump:            # only once the lock and the upstream check passed — a refused start
            shutil.rmtree(idir, ignore_errors=True)   # must not wipe the sidecar a server may be reading
        if bump:                       # rebuild=True already starts from an empty file
            run.con.executescript(DDL)
            run.con.execute("DELETE FROM _work")
            run.con.execute("DELETE FROM papers_fp")
            run.con.execute("DELETE FROM vec_rows")
            run.con.execute("DELETE FROM vec_meta")
            run.con.commit()
        return _build(run, idir, dense, embed, model_label, log)


def _sync_dir(d: Path, con, pass_name: str | None = None, table: str | None = None) -> None:
    """Keep a tantivy directory and its sqlite bookkeeping consistent: whichever side is missing resets the
    other, so the next pass re-does every item into an empty index (never duplicates, never phantoms)."""
    d.mkdir(parents=True, exist_ok=True)
    dir_ok = (d / "meta.json").exists()          # tantivy's own meta: a committed index exists
    if table:
        book = con.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None
    else:
        book = con.execute("SELECT 1 FROM _work WHERE pass=? LIMIT 1", (pass_name,)).fetchone() is not None
    if dir_ok and not book:
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    elif book and not dir_ok:
        if table:
            con.execute(f"DELETE FROM {table}")
        else:
            con.execute("DELETE FROM _work WHERE pass=?", (pass_name,))
        con.commit()


def _open_writer(tantivy, d: Path, schema, threads: int = 1):
    d.mkdir(parents=True, exist_ok=True)
    ix = tantivy.Index(schema, path=str(d))
    return ix, ix.writer(heap_size=512_000_000, num_threads=threads)


def _commit(w, tries: int = 40) -> None:
    """A tantivy commit ends in a meta.json rename, which on Windows fails with access-denied while a scanner
    (Defender / the indexer, busy under heavy disk load) holds a transient handle. Measured 10-07: the failed
    commit keeps the writer's queue intact and a retry after the handle is released lands without loss — so
    access-denied retries, any other IO error propagates."""
    for i in range(tries):
        try:
            w.commit()
            return
        except ValueError as e:
            if "os error 5" not in str(e) or i == tries - 1:
                raise
            time.sleep(0.05)


# ---- papers: streaming diff against papers_fp (1.19M items must not materialise)

def _papers_pass(run, idir, reg, t2_set, ft_set, log) -> int:
    import tantivy
    d = idir / "papers"
    con = run.con
    _sync_dir(d, con, table="papers_fp")
    schema = make_schema("papers")

    def new_rows():
        recs = reg.execute("SELECT paper_id, title, abstract FROM records ORDER BY paper_id, source='arxiv'")
        nxt = next(recs, None)
        for pid, hi, ptitle in reg.execute("SELECT paper_id, first_hi, title FROM papers "
                                           "WHERE status='active' AND first_hi IS NOT NULL ORDER BY paper_id"):
            title, abstract = ptitle or "", ""
            while nxt is not None and nxt[0] < pid:
                nxt = next(recs, None)
            while nxt is not None and nxt[0] == pid:      # the arxiv row sorts last and wins
                title = nxt[1] or title
                abstract = nxt[2] or abstract
                nxt = next(recs, None)
            t1, t2, ft = bool(abstract), pid in t2_set, pid in ft_set
            yield pid, hi, title, abstract, (t1, t2, ft), store.sha(hi, title, abstract, t1, t2, ft)

    ix, w = _open_writer(tantivy, d, schema, threads=2)
    add_fp: dict[str, str] = {}
    deleted: list[str] = []
    tiers = Counter()
    n = 0
    old = con.execute("SELECT paper_id, fp FROM papers_fp ORDER BY paper_id")
    o = old.fetchone()
    for pid, hi, title, abstract, tg, fp in new_rows():
        n += 1
        tiers.update(t for t, on in zip(("t1", "t2", "ft"), tg) if on)
        while o is not None and o[0] < pid:
            w.delete_documents("pid", o[0])
            deleted.append(o[0])
            o = old.fetchone()
        if o is not None and o[0] == pid:
            same = o[1] == fp
            o = old.fetchone()
            if same:
                continue
            w.delete_documents("pid", pid)
        add_fp[pid] = fp
    while o is not None:
        w.delete_documents("pid", o[0])
        deleted.append(o[0])
        o = old.fetchone()
    _commit(w)                                            # deletes land before any add of the same pid
    n_add = 0
    for pid, hi, title, abstract, tg, fp in new_rows():
        if pid not in add_fp:
            continue
        t1, t2, ft = tg
        toks = " ".join(t for t, on in (("t0", True), ("t1", t1), ("t2", t2), ("ft", ft)) if on)
        w.add_document(tantivy.Document(pid=pid, title=title or "", abstract=abstract or "", tier=toks,
                                        day_hi=day_int(hi)))
        n_add += 1
        if n_add % 100_000 == 0:
            _commit(w)
            log(f"[index] papers added {n_add:,}/{len(add_fp):,}")
    _commit(w)
    gone = [p for p in deleted if p not in add_fp]
    con.executemany("INSERT OR REPLACE INTO papers_fp VALUES (?,?)", list(add_fp.items()))
    con.executemany("DELETE FROM papers_fp WHERE paper_id=?", [(p,) for p in gone])
    con.commit()
    ix.reload()
    log(f"[index] papers: {n:,} active ({n_add:,} re-indexed, {len(gone):,} removed); tiers {dict(tiers)}")
    return n


# ---- passages + statements: dfc work passes, chunked delete-commit / add-commit

def _passages_pass(run, idir, D, log) -> int:
    import tantivy
    d = idir / "passages"
    con = run.con
    _sync_dir(d, con, "passages")
    schema = make_schema("passages")
    doc_keys = store.item_keys("documents", "docs")
    delta_keys = store.item_keys("documents", "delta")
    sv_keys = store.item_keys("documents", "sv")
    vers: dict[str, list] = {}
    for p_, v_ in D.con.execute("SELECT paper_id, version FROM docs"):
        vers.setdefault(p_, []).append(v_)
    fps = {}
    for pid, vs in vers.items():
        bv = 1 if 1 in vs else (0 if 0 in vs else max(vs))
        lv = max(vs)
        fps[pid] = store.sha(doc_keys.get(f"{pid}@v{bv}"), doc_keys.get(f"{pid}@v{lv}") if lv != bv else None,
                             delta_keys.get(pid), sv_keys.get(pid))
    wp = run.work("passages")
    todo = sorted(set(wp.todo([(p, f) for p, f in sorted(fps.items())])))
    log(f"[index] passages: {len(todo):,} of {len(fps):,} papers")
    ix, w = _open_writer(tantivy, d, schema)
    skipped_undated = 0

    def process(items):
        nonlocal skipped_undated
        for pid in items:
            w.delete_documents("pid", pid)
        _commit(w)
        ok = []
        for pid in items:
            try:
                full = RD.full_text(D, pid)
                if full:
                    sec = {u["uid"]: (u.get("section") or "") for u in full["units"]}
                    for s in full["sentences"]:
                        txt = (s.get("text") or "").strip()
                        if not txt:
                            continue
                        if not s.get("date"):
                            skipped_undated += 1
                            continue
                        w.add_document(tantivy.Document(
                            pid=pid, uid=s["sid"], section=sec.get(s.get("unit"), ""),
                            kind=("delta" if s.get("in_delta") else full["source"]),
                            body=txt, day_hi=day_int(s["date"])))
                ok.append(pid)
            except Exception as e:
                wp.fail(pid, f"{type(e).__name__}: {e}")
        _commit(w)
        wp.ok_many(ok)

    for i in range(0, len(todo), CHUNK):
        process(todo[i:i + CHUNK])
    _commit(w)
    wp.sweep(sorted(fps), lambda pid: w.delete_documents("pid", pid))
    _commit(w)
    ix.reload()
    n = ix.searcher().num_docs
    log(f"[index] passages: {n:,} sentences ({skipped_undated} undated skipped)")
    return n


def _statements_pass(run, idir, ext, log) -> int:
    import tantivy
    d = idir / "statements"
    con = run.con
    _sync_dir(d, con, "statements")
    schema = make_schema("statements")
    keys = [store.item_keys("extract", p) for p in ("t1", "t2", "results", "other")]
    items = [r[0] for r in ext.execute("SELECT DISTINCT item FROM statements")]
    ws = run.work("statements")
    fps = {it: store.sha(*(k.get(it) for k in keys)) for it in items}
    todo = set(ws.todo([(i, fps[i]) for i in sorted(items)]))
    log(f"[index] statements: {len(todo):,} of {len(items):,} items")
    ix, w = _open_writer(tantivy, d, schema)
    q = ("SELECT item, id, speaker, date, kind, about, role, facet, epistemic, text, quote FROM statements "
         "ORDER BY item, id")

    def process(chunk_rows, chunk_items):
        for it in chunk_items:
            w.delete_documents("item", it)
        _commit(w)
        for it, sid, speaker, date, kind, about, role, facet, ep, text, quote in chunk_rows:
            w.add_document(tantivy.Document(
                sid=int(sid), item=it, speaker=speaker or "", about=about or "", kind=kind or "",
                facet=facet or "", role=role or "", epistemic=ep or "", body=f"{text} {quote}",
                day_hi=day_int(date)))
        _commit(w)
        ws.ok_many(chunk_items)

    cur = ext.execute(q)
    buf_rows, buf_items = [], []
    for row in cur:
        it = row[0]
        if it not in todo:
            continue
        if buf_items and buf_items[-1] != it:
            if len(buf_items) >= CHUNK:
                process(buf_rows, buf_items)
                buf_rows, buf_items = [], []
        if not buf_items or buf_items[-1] != it:
            buf_items.append(it)
        buf_rows.append(row)
    if buf_items:
        process(buf_rows, buf_items)
    ws.sweep(items, lambda it: w.delete_documents("item", it))
    _commit(w)
    ix.reload()
    n = ix.searcher().num_docs
    log(f"[index] statements: {n:,} docs")
    return n


# ---- vectors: incremental embedding cache (vec_rows) + date-sorted memmap export

def _vec_groups_papers(ext, reg, label):
    for (pid,) in ext.execute("SELECT paper_id FROM scope ORDER BY paper_id"):
        title, abstract = "", ""
        for t, a in reg.execute("SELECT title, abstract FROM records WHERE paper_id=? ORDER BY source='arxiv'",
                                (pid,)):
            title, abstract = t or title, a or abstract
        r = reg.execute("SELECT first_hi FROM papers WHERE paper_id=?", (pid,)).fetchone()
        hi = r[0] if r else None
        if not (title and abstract and hi):
            continue
        text = f"{title}. {abstract[:1500]}"
        yield pid, store.sha(label, hi, text), [(pid, day_int(hi), pid, -1, text)]


def _vec_groups_statements(ext, label):
    q = ("SELECT item, id, date, about, facet, text, quote FROM statements WHERE kind='self' "
         "ORDER BY item, id")
    item, rows = None, []
    for it, sid, date, about, facet, text, quote in ext.execute(q):
        if it != item:
            if item is not None:
                yield item, store.sha(label, rows), rows
            item, rows = it, []
        fc = FACETS.index(facet) if facet in FACETS else -1
        rows.append((str(sid), day_int(date), about or it, fc, f"{text} {quote}"[:1500]))
    if item is not None:
        yield item, store.sha(label, rows), rows


def _vec_pass(run, name, ext, reg, embed, label, log) -> int:
    con = run.con
    w = run.work(f"vec_{name}")
    groups = (_vec_groups_papers(ext, reg, label) if name == "papers"
              else _vec_groups_statements(ext, label))
    todo, all_items, batch = set(), [], []
    for item, fp, _rows in groups:
        all_items.append(item)
        batch.append((item, fp))
        if len(batch) >= 4096:
            todo |= set(w.todo(batch))
            batch = []
    todo |= set(w.todo(batch))
    log(f"[index] vec_{name}: {len(todo):,} of {len(all_items):,} items to embed")

    buf: list[tuple[str, list]] = []        # [(item, rows)] — whole items only
    n_buf = 0
    stats = Counter()

    def flush():
        nonlocal n_buf
        if not buf:
            return
        texts = [r[4] for _it, rows in buf for r in rows]
        try:
            vecs = embed(texts)
            if len(vecs) != len(texts):
                raise RuntimeError(f"embed returned {len(vecs)}/{len(texts)}")
        except Exception as e:
            w.fail_many([it for it, _ in buf], f"embed: {type(e).__name__}: {e}")
            buf.clear()
            n_buf = 0
            return
        dim = len(vecs[0])
        with run.lock:
            r = con.execute("SELECT dim FROM vec_meta WHERE name=?", (name,)).fetchone()
            if r is None:
                con.execute("INSERT INTO vec_meta VALUES (?,?,?)", (name, label, dim))
            elif r[0] != dim:
                w.fail_many([it for it, _ in buf], f"embedding dim {dim} != recorded {r[0]}")
                buf.clear()
                n_buf = 0
                return
            rows_db = []
            for it, rows in buf:
                con.execute("DELETE FROM vec_rows WHERE name=? AND item=?", (name, it))
                for rk, dh, own, fc, _t in rows:
                    rows_db.append([name, rk, dh, own, fc, it, None])
            for row, v in zip(rows_db, vecs):
                row[6] = np.asarray(v, dtype="<f4").tobytes()
            con.executemany("INSERT OR REPLACE INTO vec_rows VALUES (?,?,?,?,?,?,?)", rows_db)
            w.ok_many([it for it, _ in buf])
            con.commit()
        stats["embedded"] += len(texts)
        buf.clear()
        n_buf = 0

    for item, _fp, rows in (_vec_groups_papers(ext, reg, label) if name == "papers"
                            else _vec_groups_statements(ext, label)):
        if item not in todo:
            continue
        buf.append((item, rows))
        n_buf += len(rows)
        if n_buf >= EMBED_BATCH:
            flush()
    flush()
    w.sweep(all_items, lambda it: con.execute("DELETE FROM vec_rows WHERE name=? AND item=?", (name, it)))
    con.commit()
    log(f"[index] vec_{name}: {stats['embedded']:,} vectors embedded")
    return con.execute("SELECT count(*) FROM vec_rows WHERE name=?", (name,)).fetchone()[0]


def _export(name: str, idir: Path, con, label, log) -> None:
    """vec_rows -> the date-sorted memmap files (search.py): normalised floats, per-dimension-mean sign codes,
    aligned days/keys/owners[/facets], atomic replaces."""
    d = vectors_dir(name, idir)
    d.mkdir(parents=True, exist_ok=True)
    n, dim = con.execute("SELECT count(*), max(length(vec)) FROM vec_rows WHERE name=?", (name,)).fetchone()
    dim = (dim or 0) // 4
    files = ("float32.dat", "codes.u8", "days.i32", "keys.txt", "owners.txt", "mean.f32", "facets.u8")
    meta = {"name": name, "model": label, "dim": dim, "n": n, "has_facets": False,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if not n:
        for f in files:
            (d / f).unlink(missing_ok=True)
    else:
        cur = con.execute("SELECT rowkey, day_hi, owner, facet, vec FROM vec_rows WHERE name=? "
                          "ORDER BY day_hi, rowkey", (name,))
        days = np.empty(n, dtype="int32")
        facets = np.empty(n, dtype="uint8")
        keys, owners = [], []
        total = np.zeros(dim, dtype="float64")
        has_facets = False
        with open(d / "float32.dat.tmp", "wb") as f:
            for i, (rk, dh, own, fc, vec) in enumerate(cur):
                v = np.frombuffer(vec, dtype="<f4")
                nrm = float(np.linalg.norm(v))
                if nrm:
                    v = v / nrm
                f.write(v.tobytes())
                total += v
                keys.append(rk)
                owners.append(own or "")
                days[i] = dh
                facets[i] = 255 if fc is None or fc < 0 else fc
                has_facets |= bool(fc is not None and fc >= 0)
        mean = (total / n).astype("<f4")
        with open(d / "float32.dat.tmp", "rb") as f, open(d / "codes.u8.tmp", "wb") as g:
            while True:
                chunk = np.fromfile(f, dtype="<f4", count=EXPORT_CHUNK * dim)
                if chunk.size == 0:
                    break
                g.write(np.packbits(chunk.reshape(-1, dim) > mean, axis=1).tobytes())
        days.tofile(d / "days.i32.tmp")
        mean.tofile(d / "mean.f32.tmp")
        if has_facets:
            facets.tofile(d / "facets.u8.tmp")
        (d / "keys.txt.tmp").write_text("\n".join(keys), encoding="utf-8")
        (d / "owners.txt.tmp").write_text("\n".join(owners), encoding="utf-8")
        meta["has_facets"] = has_facets
        moves = [("float32.dat.tmp", "float32.dat"), ("codes.u8.tmp", "codes.u8"), ("days.i32.tmp", "days.i32"),
                 ("mean.f32.tmp", "mean.f32"), ("keys.txt.tmp", "keys.txt"), ("owners.txt.tmp", "owners.txt")]
        if has_facets:
            moves.append(("facets.u8.tmp", "facets.u8"))
        else:
            (d / "facets.u8").unlink(missing_ok=True)
        for src, dst in moves:
            for _ in range(40):                      # a reader's memmap may hold the target open (Windows)
                try:
                    os.replace(d / src, d / dst)
                    break
                except PermissionError:
                    time.sleep(0.05)
            else:
                os.replace(d / src, d / dst)
    (d / "meta.json.tmp").write_text(json.dumps(meta), encoding="utf-8")
    os.replace(d / "meta.json.tmp", d / "meta.json")
    log(f"[index] vectors {name}: n={n:,} dim={dim}")


def _build(run, idir: Path, dense, embed, label, log) -> dict:
    con = run.con
    con.executescript(DDL)
    reg = store.read_only(paths.library() / "registry.sqlite")
    ext = store.read_only(store.db_path("extract"))
    D = Documents()
    try:
        t2_set = {r[0] for r in D.con.execute("SELECT DISTINCT paper_id FROM docs")}
        ft_set = {r[0] for r in D.con.execute("SELECT DISTINCT paper_id FROM docs WHERE careful_z IS NOT NULL")}
        ft_set |= {r[0] for r in D.con.execute("SELECT paper_id FROM sv")}
        counts = {"papers": _papers_pass(run, idir, reg, t2_set, ft_set, log),
                  "passages": _passages_pass(run, idir, D, log),
                  "statements": _statements_pass(run, idir, ext, log)}
        for name in ("papers", "statements"):
            if name in dense:
                counts[f"vec_{name}"] = _vec_pass(run, name, ext, reg, embed, label, log)
                _export(name, idir, con, label, log)
            else:                       # dropped from the dense config: free the cache and the files
                con.execute("DELETE FROM vec_rows WHERE name=?", (name,))
                con.execute("DELETE FROM vec_meta WHERE name=?", (name,))
                con.commit()
                shutil.rmtree(vectors_dir(name, idir), ignore_errors=True)
        run.finish(counts)
        return counts
    finally:
        reg.close()
        ext.close()
        D.close()


def db_file() -> Path:
    return store.db_path("index")
