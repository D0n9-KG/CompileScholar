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
CREATE INDEX IF NOT EXISTS ix_cites_sid ON cites(sentence_id);
CREATE INDEX IF NOT EXISTS ix_sent_citing ON sentences(citing);
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


def _delete(con, aid: str) -> None:
    for t, col in (("cites", "citing"), ("sentences", "citing"), ("entries", "citing"), ("docs", "arxiv_id")):
        con.execute(f"DELETE FROM {t} WHERE {col}=?", (aid,))


def build(rebuild: bool = False, log=print, report_every: int = 5000) -> dict:
    """Every document in the documents stage. Incremental: a citing paper is redone when its document changed or the
    papers stage has new data (an entry that was a stub may now resolve); papers that left documents are removed."""
    params = {"input": "documents stage, all"}
    with store.Run("citations", params, rebuild=rebuild) as run:
        papers = Papers()
        resolver = EntryResolver(papers)
        documents = Documents()
        con = run.con
        con.executescript(DDL)
        doc_keys = store.item_keys("documents", "documents")
        pfp = run.fingerprint("papers")
        w = run.work("citations")
        todo = w.todo((aid, f"{k}|{pfp}") for aid, k in doc_keys.items())
        n = {"docs": 0, "no_bib": 0}
        for i, aid in enumerate(todo):
            _delete(con, aid)
            d = documents.get(aid)
            if d is None:            # an ok document item with nothing stored (not in papers)
                w.ok(aid)
                continue
            try:
                parsed = H.parse(d["raw"]) if d["source"] == "arxiv_html" else M.citation_sentences(d["raw"])
                if parsed is None or parsed[0] is None or not parsed[1]:
                    n["no_bib"] += 1
                else:
                    _write_doc(con, papers, resolver, aid, d["source"], *parsed)
                    n["docs"] += 1
            except Exception as e:
                _delete(con, aid)
                w.fail(aid, f"{type(e).__name__}: {e}")
                continue
            w.ok(aid)
            if (i + 1) % report_every == 0:
                con.commit()
                log(f"[citations] {i + 1}/{len(todo)} documents: {n}")
        con.commit()
        n["removed"] = w.sweep(doc_keys, lambda aid: _delete(con, aid))
        counts = {k: con.execute(f"SELECT count(*) FROM {k}").fetchone()[0] for k in ("docs", "entries", "sentences",
                                                                                     "cites")}
        counts["cites_resolved"] = con.execute("SELECT count(*) FROM cites WHERE cited LIKE 'paper:%'").fetchone()[0]
        run.finish(counts)
    return {**n, **counts}
