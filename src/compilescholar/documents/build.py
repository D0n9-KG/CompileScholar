# -*- coding: utf-8 -*-
"""Stage `documents` (phase C-1): the library's parsed papers, materialised per version for every later stage
(INTEGRATED-SYSTEM-1005 v2.5; the stage moved from arXiv-id-keyed Markdown/HTML sources onto the registry).

One docs row per (paper, version) that has an ok parse in data/library/registry.sqlite, built by documents.assemble:
  key        "<paper_id>@v<N>" — N = arXiv version, 0 = version of record
  text_date  that version's date; everything a later stage reads from the row carries it (v2.5: content is dated
             by the earliest version it appears in — v1 rows give v1 content the v1 date, delta rows give the
             latest version's additions the latest version's date)
  fast_z     zlib JSON of the GROBID tier (documents.tei Doc dict: units, sentences, entries, cites)
  careful_z  zlib JSON of the MinerU tier (units with table HTML + references), when the paper was deep-parsed

One deltas row per multi-version paper whose v1 and latest version are both materialised: versions.delta — the
sentences, revisions, bibliography entries and citation pairs only the latest version has.

Work items (no LLM anywhere in this stage):
  docs pass   item = key, fingerprint = (chosen asset's sha256, its parses rows incl. created_at, text_date) —
              a re-parsed PDF or a corrected version date re-opens exactly that item;
  delta pass  item = paper_id, fingerprint = the two docs work keys.
Assets that disappeared (a merge moved them to the survivor) are swept with their rows.

Legacy: citations / extract / index / tools / grow still import Documents from here expecting the pre-C arXiv-id
API; they are rebuilt onto this contract in C②–C④ and phase D (their stages are stale and unrunnable until then)."""
from __future__ import annotations

import json
import zlib

from ..core import paths
from ..dfc import store
from . import assemble as A
from . import versions as VV

DDL = """CREATE TABLE IF NOT EXISTS docs(
  key TEXT PRIMARY KEY, paper_id TEXT, version INT, sha256 TEXT, source TEXT,
  text_date TEXT, date_precision TEXT,
  n_units INT, n_sentences INT, n_entries INT, n_cites INT, n_tables INT,
  fast_z BLOB, careful_z BLOB);
CREATE INDEX IF NOT EXISTS ix_docs_paper ON docs(paper_id, version);
CREATE TABLE IF NOT EXISTS deltas(
  paper_id TEXT PRIMARY KEY, v1_key TEXT, latest_key TEXT,
  n_new INT, n_revised INT, n_new_entries INT, n_cites INT, delta_z BLOB);"""


def _z(obj) -> bytes:
    return zlib.compress(json.dumps(obj, ensure_ascii=False).encode("utf-8"), 6)


def _uz(b) -> dict | None:
    return json.loads(zlib.decompress(b).decode("utf-8")) if b else None


def _registry():
    return store.read_only(paths.library() / "registry.sqlite")


def _items(reg) -> tuple[list[tuple], dict]:
    """[(key, paper_id, version)] and {key: fingerprint}: one asset per (paper, version) — the same preference
    assemble.get uses (non-scihub channel first) — restricted to (paper, version)s with at least one ok parse."""
    parses: dict[str, list] = {}
    for sha, parser, pv, status, created in reg.execute(
            "SELECT sha256, parser, parser_version, status, created_at FROM parses ORDER BY sha256, parser"):
        parses.setdefault(sha, []).append((parser, pv, status, created))
    dates: dict[tuple, tuple] = {}
    for pid, ver, hi, pr in reg.execute("SELECT paper_id, version, hi, precision FROM dates "
                                        "WHERE kind='arxiv_v' ORDER BY paper_id, version, hi"):
        dates.setdefault((pid, ver), (hi, pr))
    out, fps, seen = [], {}, set()
    for pid, ver, sha in reg.execute("SELECT paper_id, version, sha256 FROM assets "
                                     "ORDER BY paper_id, version, channel='scihub_local'"):
        key = f"{pid}@v{ver}"
        if key in seen:
            continue
        seen.add(key)
        prows = tuple(parses.get(sha, ()))
        if not any(st == "ok" for _, _, st, _ in prows):
            continue
        td = dates.get((pid, ver)) or (A.text_date(reg, pid, ver) if ver == 0 else None)
        out.append((key, pid, ver))
        fps[key] = store.sha(sha, prows, td)
    return out, fps


def _materialize(reg, con, lock, key: str, pid: str, version: int, counter: dict) -> None:
    d = A.get(reg, pid, version)
    if d is None:
        raise LookupError("asset disappeared (merged away?)")
    fast, careful = d["fast"], d["careful"]
    td = d["text_date"] or (None, None)
    with lock:
        con.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (key, pid, version, d["sha256"], d["source"], td[0], td[1],
                     len(fast["units"]) if fast else 0, len(fast["sentences"]) if fast else 0,
                     len(fast["entries"]) if fast else 0, len(fast["cites"]) if fast else 0,
                     sum(u["kind"] == "table" for u in careful["units"]) if careful else 0,
                     _z(fast) if fast else None, _z(careful) if careful else None))
        counter["written"] += 1
        if counter["written"] % 500 == 0:
            con.commit()


def _delta_row(rconn, pid: str, v1_key: str, latest_key: str):
    rows = dict(rconn.execute("SELECT key, fast_z FROM docs WHERE key IN (?,?)", (v1_key, latest_key)).fetchall())
    v1, latest = _uz(rows.get(v1_key)), _uz(rows.get(latest_key))
    if not v1 or not latest:                      # an end without a fast tier has no sentence-level delta
        return None
    d = VV.delta(v1, latest)
    return (pid, v1_key, latest_key, len(d["sentences"]), len(d["revised"]), len(d["entries"]), len(d["cites"]), _z(d))


def build(workers: int | None = None, rebuild: bool = False, log=print) -> dict:
    """Materialise every parsed library asset. Incremental and question-blind: an item is redone only when its
    PDF was re-parsed, its version date changed, or this stage's code changed."""
    workers = workers or 8                        # CPU-bound (lxml + rapidfuzz), not an LLM stage
    params = {"v": 2}
    with store.Run("documents", params, rebuild=rebuild) as run:
        con = run.con
        con.executescript(DDL)
        if "key" not in {r[1] for r in con.execute("PRAGMA table_info(docs)")}:
            raise RuntimeError("documents.sqlite has the pre-C (arXiv-id) schema; build it with --rebuild")
        reg = _registry()
        try:
            items, fps = _items(reg)
            keys = [k for k, _, _ in items]
            w_docs = run.work("docs")
            todo = set(w_docs.todo([(k, fps[k]) for k in keys]))
            log(f"[documents] {len(todo):,} of {len(keys):,} version docs to materialise")
            counter = {"written": 0}

            def one(triple):
                key, pid, version = triple
                try:
                    _materialize(reg, con, run.lock, key, pid, version, counter)
                    w_docs.ok(key)
                except Exception as e:            # a broken parse product is this item's failure, not the stage's
                    w_docs.fail(key, f"{type(e).__name__}: {e}")

            store.parallel(one, [t for t in items if t[0] in todo], workers, log=log, every=2000, label="documents")
            with run.lock:
                con.commit()
            w_docs.sweep(keys, lambda k: con.execute("DELETE FROM docs WHERE key=?", (k,)))

            w_delta = run.work("delta")
            pairs = [(pid, f"{pid}@v1", f"{pid}@v{mx}") for pid, mx in con.execute(
                "SELECT paper_id, MAX(version) FROM docs GROUP BY paper_id "
                "HAVING MAX(version) > 1 AND SUM(version = 1) > 0")]
            dtodo = set(w_delta.todo([(pid, store.sha(w_docs.key(a), w_docs.key(b))) for pid, a, b in pairs]))
            rconn = store.read_only(run.path)     # per-thread reads of this run's file (rebuild writes a .new)
            dcounter = {"written": 0}

            def one_delta(triple):
                pid = triple[0]
                try:
                    row = _delta_row(rconn, *triple)
                    with run.lock:
                        if row:
                            con.execute("INSERT OR REPLACE INTO deltas VALUES (?,?,?,?,?,?,?,?)", row)
                        else:
                            con.execute("DELETE FROM deltas WHERE paper_id=?", (pid,))
                        dcounter["written"] += 1
                        if dcounter["written"] % 200 == 0:
                            con.commit()
                    w_delta.ok(pid)
                except Exception as e:
                    w_delta.fail(pid, f"{type(e).__name__}: {e}")

            store.parallel(one_delta, [t for t in pairs if t[0] in dtodo], workers, log=log, every=5000,
                           label="documents:delta")
            with run.lock:
                con.commit()
            w_delta.sweep([p for p, _, _ in pairs], lambda p: con.execute("DELETE FROM deltas WHERE paper_id=?", (p,)))
            rconn.close()

            q = lambda s: con.execute(s).fetchone()[0] or 0        # noqa: E731
            counts = {"docs": q("SELECT count(*) FROM docs"), "deltas": q("SELECT count(*) FROM deltas"),
                      "sentences": q("SELECT sum(n_sentences) FROM docs"), "cites": q("SELECT sum(n_cites) FROM docs"),
                      "entries": q("SELECT sum(n_entries) FROM docs"),
                      "careful": q("SELECT count(*) FROM docs WHERE careful_z IS NOT NULL")}
            run.finish(counts)
        finally:
            reg.close()
    return counts


class Documents:
    """Read-only access to the materialised documents, by key ("<paper_id>@v<N>") or paper_id (all later stages
    use this; the pre-C arXiv-id API of the same class is gone — see the module docstring)."""

    def __init__(self):
        self.con = store.connect("documents", readonly=True)

    def keys(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT key FROM docs ORDER BY key")]

    def papers(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT DISTINCT paper_id FROM docs ORDER BY paper_id")]

    def versions(self, paper_id: str) -> list[int]:
        return [r[0] for r in self.con.execute("SELECT version FROM docs WHERE paper_id=? ORDER BY version",
                                               (paper_id,))]

    def get(self, key: str) -> dict | None:
        r = self.con.execute("SELECT paper_id, version, sha256, source, text_date, date_precision, fast_z, careful_z "
                             "FROM docs WHERE key=?", (key,)).fetchone()
        if not r:
            return None
        return {"key": key, "paper_id": r[0], "version": r[1], "sha256": r[2], "source": r[3],
                "text_date": r[4], "date_precision": r[5], "fast": _uz(r[6]), "careful": _uz(r[7])}

    def delta(self, paper_id: str) -> dict | None:
        r = self.con.execute("SELECT v1_key, latest_key, delta_z FROM deltas WHERE paper_id=?",
                             (paper_id,)).fetchone()
        return None if not r else {"v1_key": r[0], "latest_key": r[1], **_uz(r[2])}

    def close(self) -> None:
        self.con.close()
