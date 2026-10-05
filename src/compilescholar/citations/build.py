# -*- coding: utf-8 -*-
"""Stage `citations` (L1): full text -> citation sentences -> cited paper ids, written to data/dfc/citations.sqlite.

Inputs (read here and nowhere downstream):
  - IdeaForecastBench Markdown (data/external/ideaforecast/*.parquet; arxiv_id, text)
  - arXiv HTML pages cached under cache/arxiv_html/<id>.html (fetched by sources.arxiv_html, 15 s/request)
Every citing paper must exist in the papers stage (its v1_date dates every sentence it contributes).

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
from pathlib import Path

import pyarrow.parquet as pq

from ..core import paths
from ..corpus.papers import Papers
from ..dfc import store
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


def ideaforecast_dir() -> Path:
    return paths.data() / "external" / "ideaforecast"


def html_cache() -> Path:
    from ..sources.arxiv_html import cache_dir
    return cache_dir()


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


def build(selection: set[str] | None = None, sources=("ideaforecast", "html"), log=print) -> dict:
    """Process every paper in `selection` (None = everything available) from the configured sources; a paper already
    in `docs` is skipped (resumable). Markdown first; HTML only for papers without usable Markdown."""
    store.require_fresh("papers")
    papers = Papers()
    resolver = EntryResolver(papers)
    con = store.connect("citations")
    con.executescript(DDL)
    done = {r[0] for r in con.execute("SELECT arxiv_id FROM docs")}
    n = {"docs": 0, "no_bib": 0, "not_in_papers": 0}
    if "ideaforecast" in sources:
        for f in sorted(ideaforecast_dir().glob("*.parquet")):
            for r in pq.read_table(f, columns=["arxiv_id", "text"]).to_pylist():
                aid = r["arxiv_id"]
                if aid in done or (selection is not None and aid not in selection):
                    continue
                bib, cs = M.citation_sentences(r["text"])
                if bib is None or not cs:
                    n["no_bib"] += 1
                    continue
                res = _write_doc(con, papers, resolver, aid, "ideaforecast_md", bib, cs)
                if "skipped" in res:
                    n["not_in_papers"] += 1
                    continue
                done.add(aid)
                n["docs"] += 1
            con.commit()
            log(f"[citations] {f.name}: {n}")
    if "html" in sources and html_cache().exists():
        for p in sorted(html_cache().glob("*.html")):
            aid = p.stem
            if aid in done or (selection is not None and aid not in selection):
                continue
            parsed = H.parse(p.read_text(encoding="utf-8", errors="replace"))
            if parsed is None or not parsed[1]:
                n["no_bib"] += 1
                continue
            res = _write_doc(con, papers, resolver, aid, "arxiv_html", *parsed)
            if "skipped" in res:
                n["not_in_papers"] += 1
                continue
            done.add(aid)
            n["docs"] += 1
        con.commit()
    counts = {k: con.execute(f"SELECT count(*) FROM {k}").fetchone()[0] for k in ("docs", "entries", "sentences",
                                                                                 "cites")}
    counts["cites_resolved"] = con.execute("SELECT count(*) FROM cites WHERE cited LIKE 'paper:%'").fetchone()[0]
    con.close()
    store.write_manifest("citations", {"sources": list(sources),
                                       "selection": None if selection is None else sorted(selection)[:5] +
                                       [f"... {len(selection)} total"]}, counts)
    return {**n, **counts}
