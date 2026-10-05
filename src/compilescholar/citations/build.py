# -*- coding: utf-8 -*-
"""Stage `citations` (L1b): full text -> citation sentences -> cited paper ids, written to data/dfc/citations.sqlite.

Input: the `documents` stage (every full text we hold, Markdown or arXiv HTML, keyed by arxiv_id); raw files are not
read here. Runs over ALL documents — no selection (zero LLM, ~0.02 s/paper; a selection would make the citation
graph depend on whatever the selection was based on, which is how benchmark gold could leak in).
Every citing paper exists in the papers stage (its v1_date dates every sentence it contributes).

Tables
  docs(arxiv_id PK, source, style, n_entries, n_sentences, n_resolved)
  entries(citing, key, raw, cited, method, title, year)            -- one row per bibliography entry actually cited
  sentences(id PK, citing, date, sentence, keys)                   -- keys = JSON list of entry keys in order
  cites(sentence_id, citing, date, key, cited, n_group)            -- one row per (sentence, cited entry)
cited = "paper:<arxiv_id>" when resolved, else "stub:<norm_title>" (never empty).
Params recorded in the manifest: which sources, which papers (selection), resolver version.
"""
from __future__ import annotations

import json

from ..corpus.papers import Papers
from ..dfc import store
from ..documents.build import Documents
from . import html as H
from . import markdown as M
from .resolve import EntryResolver

DDL = """
CREATE TABLE IF NOT EXISTS docs(arxiv_id TEXT PRIMARY KEY, source TEXT, style TEXT, n_entries INT, n_sentences INT,
                                n_resolved INT);
CREATE TABLE IF NOT EXISTS entries(citing TEXT, key TEXT, raw TEXT, cited TEXT, method TEXT, title TEXT, year INT,
                                   PRIMARY KEY(citing, key));
CREATE TABLE IF NOT EXISTS sentences(id INTEGER PRIMARY KEY, citing TEXT, date TEXT, sentence TEXT, keys TEXT);
CREATE TABLE IF NOT EXISTS cites(sentence_id INT, citing TEXT, date TEXT, key TEXT, cited TEXT, n_group INT);
CREATE INDEX IF NOT EXISTS ix_cites_cited ON cites(cited, date);
CREATE INDEX IF NOT EXISTS ix_cites_citing ON cites(citing);
"""


def _key(k) -> str:
    return json.dumps(k, ensure_ascii=False) if not isinstance(k, str) else k


def _write_doc(con, papers: Papers, resolver: EntryResolver, aid: str, source: str, bib, cs) -> dict:
    v1 = papers.v1_date(aid)
    if not v1:
        return {"skipped": "not in papers"}
    used = {}
    for c in cs:
        used.setdefault(c.key, None)
    n_res = 0
    for k in used:
        e = resolver.resolve(bib.entries[k]["raw"])
        cited = f"paper:{e.arxiv_id}" if e.arxiv_id else f"stub:{e.stub_key}"
        n_res += bool(e.arxiv_id)
        used[k] = cited
        con.execute("INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?)",
                    (aid, _key(k), bib.entries[k]["raw"], cited, e.method, e.title, e.year))
    seen = {}
    for c in cs:
        sid = seen.get(c.sentence)
        if sid is None:
            cur = con.execute("INSERT INTO sentences(citing, date, sentence, keys) VALUES (?,?,?,?)",
                              (aid, v1, c.sentence, json.dumps([_key(k) for k in c.group], ensure_ascii=False)))
            sid = seen[c.sentence] = cur.lastrowid
        con.execute("INSERT INTO cites VALUES (?,?,?,?,?,?)", (sid, aid, v1, _key(c.key), used[c.key], c.n_keys))
    con.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?)",
                (aid, source, bib.style, len(bib.entries), len(seen), n_res))
    return {"sentences": len(seen), "entries": len(used), "resolved": n_res}


def build(log=print, report_every: int = 5000) -> dict:
    """Every document in the documents stage; a paper already in `docs` is skipped (resumable)."""
    store.require_fresh("papers", "documents")
    papers = Papers()
    resolver = EntryResolver(papers)
    documents = Documents()
    con = store.connect("citations")
    con.executescript(DDL)
    done = {r[0] for r in con.execute("SELECT arxiv_id FROM docs")}
    n = {"docs": 0, "no_bib": 0}
    for i, aid in enumerate(documents.ids()):
        if aid in done:
            continue
        d = documents.get(aid)
        parsed = H.parse(d["raw"]) if d["source"] == "arxiv_html" else M.citation_sentences(d["raw"])
        if parsed is None or parsed[0] is None or not parsed[1]:
            n["no_bib"] += 1
            continue
        _write_doc(con, papers, resolver, aid, d["source"], *parsed)
        n["docs"] += 1
        if (i + 1) % report_every == 0:
            con.commit()
            log(f"[citations] {i + 1} documents: {n}")
    con.commit()
    counts = {k: con.execute(f"SELECT count(*) FROM {k}").fetchone()[0] for k in ("docs", "entries", "sentences",
                                                                                 "cites")}
    counts["cites_resolved"] = con.execute("SELECT count(*) FROM cites WHERE cited LIKE 'paper:%'").fetchone()[0]
    counts["no_bib"] = n["no_bib"]
    con.close()
    store.write_manifest("citations", {"input": "documents stage, all"}, counts)
    return {**n, **counts}
