# -*- coding: utf-8 -*-
"""Stage `index` (L5): multi-granularity retrieval over the store, every search filtered by as_of.

Three indexes, each a SQLite FTS5 table (BM25) plus an optional dense matrix (local qwen3-embedding-8b, same model for
documents and queries; never mixed):
  papers      title + abstract of every paper in the papers stage that the system knows something about (in the
              extract scope, or having a full text, or cited in the citations stage) — paper-level "find"
  passages    body paragraphs of every document (documents stage) — "read" and evidence search
  statements  extracted statements (self + other) — evidence at sentence level, already typed and dated
Fusion is Reciprocal Rank Fusion (k = 60) of BM25 and dense ranks, then a per-paper cap (the old kb.index.HybridIndex
behaviour, kept). The date of every row is stored so the filter `date <= as_of` happens inside the query.

Dense vectors are optional per index (`dense=("papers", "statements")`); without them search is BM25 only and says so.
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from ..dfc import store
from ..documents.build import Documents

DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(arxiv_id UNINDEXED, date UNINDEXED, title, body);
CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(uid UNINDEXED, arxiv_id UNINDEXED, date UNINDEXED,
                                                          section UNINDEXED, body);
CREATE VIRTUAL TABLE IF NOT EXISTS statements_fts USING fts5(sid UNINDEXED, speaker UNINDEXED, about UNINDEXED,
                                                            date UNINDEXED, kind UNINDEXED, facet UNINDEXED, body);
CREATE TABLE IF NOT EXISTS dense(name TEXT, rowkey TEXT, date TEXT, vec BLOB);
"""
RRF_K = 60
_TOK = re.compile(r"[A-Za-z0-9]+")
_STOP = set("a an the of in on for to and or with by from as is are was were be this that we our it its".split())


def fts_query(q: str) -> str:
    """Free text -> a safe FTS5 OR-query of quoted terms."""
    toks = [t for t in _TOK.findall(q.lower()) if t not in _STOP and len(t) > 1]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(toks)) or '""'


def _scope_ids(pap, cit, ext) -> set[str]:
    ids = {r[0] for r in Documents().con.execute("SELECT arxiv_id FROM docs")}
    ids |= {r[0][6:] for r in cit.execute("SELECT DISTINCT cited FROM cites WHERE cited LIKE 'paper:%'")}
    ids |= {r[0] for r in ext.execute("SELECT arxiv_id FROM scope")}
    return ids


def build(dense: tuple[str, ...] = (), embed=None, log=print) -> dict:
    store.require_fresh("papers", "documents", "citations", "extract")
    pap = store.connect("papers", readonly=True)
    cit = store.connect("citations", readonly=True)
    ext = store.connect("extract", readonly=True)
    con = store.connect("index")
    for t in ("papers_fts", "passages_fts", "statements_fts", "dense"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    con.executescript(DDL)
    ids = _scope_ids(pap, cit, ext)
    rows = []
    for aid in sorted(ids):
        r = pap.execute("SELECT v1_date, title, abstract FROM papers WHERE arxiv_id=?", (aid,)).fetchone()
        if r and r[0]:
            rows.append((aid, r[0], r[1] or "", r[2] or ""))
    con.executemany("INSERT INTO papers_fts VALUES (?,?,?,?)", rows)
    log(f"[index] papers {len(rows)}")
    docs = Documents()
    n_pass = 0
    for aid in docs.ids():
        d = docs.get(aid)
        batch = [(u.uid, aid, d["v1_date"], u.section, u.text) for u in d["units"] if u.kind in ("para", "caption")]
        con.executemany("INSERT INTO passages_fts VALUES (?,?,?,?,?)", batch)
        n_pass += len(batch)
    log(f"[index] passages {n_pass}")
    st = ext.execute("SELECT id, speaker, about, date, kind, facet, text, quote FROM statements").fetchall()
    con.executemany("INSERT INTO statements_fts VALUES (?,?,?,?,?,?,?)",
                    [(i, sp, ab, dt, k, f, f"{tx} {q}") for i, sp, ab, dt, k, f, tx, q in st])
    log(f"[index] statements {len(st)}")
    con.commit()
    counts = {"papers": len(rows), "passages": n_pass, "statements": len(st)}
    if dense:
        import numpy as np
        from ..llm.embedding import embed_local
        embed = embed or embed_local
        for name in dense:
            if name == "papers":
                items = [(a, dt, f"{t}. {ab[:1500]}") for a, dt, t, ab in rows]
            elif name == "statements":
                items = [(str(i), dt, f"{tx} {q}"[:1500]) for i, sp, ab, dt, k, f, tx, q in st]
            else:
                raise ValueError(f"dense vectors are not built for {name!r}")
            for j in range(0, len(items), 512):
                chunk = items[j:j + 512]
                vecs = embed([t for _, _, t in chunk])
                con.executemany("INSERT INTO dense VALUES (?,?,?,?)",
                                [(name, k, dt, np.asarray(v, dtype="float32").tobytes())
                                 for (k, dt, _), v in zip(chunk, vecs)])
                con.commit()
            counts[f"dense_{name}"] = len(items)
            log(f"[index] dense {name} {len(items)}")
    con.close()
    store.write_manifest("index", {"dense": list(dense)}, counts)
    return counts


class Index:
    """Read side. Every search takes as_of and never returns a row dated after it."""

    def __init__(self, embed=None):
        self.con = store.connect("index", readonly=True)
        self.embed = embed
        self._mats: dict[str, tuple] = {}

    def _dense(self, name: str, query: str, as_of: str, pool: int) -> list[str]:
        if self.embed is None:
            return []
        import numpy as np
        if name not in self._mats:
            rows = self.con.execute("SELECT rowkey, date, vec FROM dense WHERE name=?", (name,)).fetchall()
            if not rows:
                self._mats[name] = None
            else:
                M = np.stack([np.frombuffer(v, dtype="float32") for _, _, v in rows])
                M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
                self._mats[name] = ([k for k, _, _ in rows], np.array([d for _, d, _ in rows]), M)
        m = self._mats[name]
        if m is None:
            return []
        keys, dates, M = m
        qv = np.asarray(self.embed([query])[0], dtype="float32")
        qv /= (np.linalg.norm(qv) + 1e-8)
        sims = M @ qv
        sims[dates > as_of] = -9
        return [keys[i] for i in np.argsort(-sims)[:pool] if sims[i] > -9]

    @staticmethod
    def _fuse(*ranks: list[str]) -> list[str]:
        sc = defaultdict(float)
        for r in ranks:
            for i, k in enumerate(r):
                sc[k] += 1.0 / (RRF_K + i + 1)
        return [k for k, _ in sorted(sc.items(), key=lambda kv: -kv[1])]

    def papers(self, query: str, as_of: str, k: int = 20, pool: int = 200) -> list[str]:
        bm = [r[0] for r in self.con.execute(
            "SELECT arxiv_id FROM papers_fts WHERE papers_fts MATCH ? AND date <= ? ORDER BY bm25(papers_fts, 0, 0, 3, 1) "
            "LIMIT ?", (fts_query(query), as_of, pool))]
        return [f"paper:{a}" for a in self._fuse(bm, self._dense("papers", query, as_of, pool))[:k]]

    def passages(self, query: str, as_of: str, k: int = 10, paper: str | None = None, per_paper: int = 2) -> list[dict]:
        q = "SELECT uid, arxiv_id, date, section, body FROM passages_fts WHERE passages_fts MATCH ? AND date <= ?"
        a = [fts_query(query), as_of]
        if paper:
            q += " AND arxiv_id = ?"
            a.append(paper.removeprefix("paper:"))
        out, per = [], Counter()
        for uid, aid, date, sec, body in self.con.execute(q + " ORDER BY bm25(passages_fts) LIMIT 400", a):
            if per[aid] >= per_paper and not paper:
                continue
            per[aid] += 1
            out.append({"uid": uid, "paper": f"paper:{aid}", "date": date, "section": sec, "text": body})
            if len(out) >= k:
                break
        return out

    def statements(self, query: str, as_of: str, k: int = 20, kind: str | None = None, facet: str | None = None,
                   per_paper: int = 2) -> list[int]:
        q = "SELECT sid, about FROM statements_fts WHERE statements_fts MATCH ? AND date <= ?"
        a = [fts_query(query), as_of]
        for col, v in (("kind", kind), ("facet", facet)):
            if v:
                q += f" AND {col} = ?"
                a.append(v)
        bm = [(str(s), ab) for s, ab in self.con.execute(q + " ORDER BY bm25(statements_fts) LIMIT 400", a)]
        about = dict(bm)
        fused = self._fuse([s for s, _ in bm], self._dense("statements", query, as_of, 400))
        out, per = [], Counter()
        for s in fused:
            ab = about.get(s)
            if ab is not None and per[ab] >= per_paper:
                continue
            per[ab] += 1
            out.append(int(s))
            if len(out) >= k:
                break
        return out


def db_file() -> Path:
    return store.db_path("index")


def connect_rw() -> sqlite3.Connection:
    return store.connect("index")
