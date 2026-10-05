# -*- coding: utf-8 -*-
"""Stage `documents` (L1a): every full text we hold, normalised into one random-access store, so that each later stage
reads a paper's text and units by id (no stage downstream touches parquet / HTML files).

Sources (raw inputs are read here and in `papers` only):
  - IdeaForecastBench Markdown: data/external/ideaforecast/*.parquet (arxiv_id, text), MinerU rendering
  - arXiv HTML cache: cache/arxiv_html/<id>.html (sources.arxiv_html), LaTeXML rendering
Markdown is preferred when both exist (same paper, already-converted text, tables as <table>).

Table docs(arxiv_id PK, source, v1_date, n_chars, n_units, n_tables, raw_z BLOB, units_z BLOB)
  raw_z   zlib(source text): Markdown, or the HTML page
  units_z zlib(JSON list of documents.units.Unit as dicts)
Only papers present in the papers stage are stored (their v1_date dates everything derived from them)."""
from __future__ import annotations

import json
import zlib
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq

from ..core import paths
from ..corpus.papers import Papers
from ..dfc import store
from . import units as U

DDL = """CREATE TABLE IF NOT EXISTS docs(arxiv_id TEXT PRIMARY KEY, source TEXT, v1_date TEXT, n_chars INT,
  n_units INT, n_tables INT, raw_z BLOB, units_z BLOB);"""


def ideaforecast_dir() -> Path:
    return paths.data() / "external" / "ideaforecast"


def html_cache() -> Path:
    from ..sources.arxiv_html import cache_dir
    return cache_dir()


def _put(con, aid: str, source: str, v1: str, raw: str, units: list) -> None:
    con.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?,?,?)",
                (aid, source, v1, len(raw), len(units), sum(u.kind == "table" for u in units),
                 zlib.compress(raw.encode("utf-8"), 6),
                 zlib.compress(json.dumps([asdict(u) for u in units], ensure_ascii=False).encode("utf-8"), 6)))


def build(sources=("ideaforecast", "html"), log=print) -> dict:
    """Store every available full text (no selection: L1 is cheap and question-blind). Resumable."""
    store.require_fresh("papers")
    papers = Papers()
    con = store.connect("documents")
    con.executescript(DDL)
    done = {r[0] for r in con.execute("SELECT arxiv_id FROM docs")}
    n = {"stored": 0, "not_in_papers": 0}
    if "ideaforecast" in sources:
        for f in sorted(ideaforecast_dir().glob("*.parquet")):
            for r in pq.read_table(f, columns=["arxiv_id", "text"]).to_pylist():
                aid = r["arxiv_id"]
                if aid in done:
                    continue
                v1 = papers.v1_date(aid)
                if not v1:
                    n["not_in_papers"] += 1
                    continue
                _put(con, aid, "ideaforecast_md", v1, r["text"], U.from_markdown(aid, r["text"]))
                done.add(aid)
                n["stored"] += 1
            con.commit()
            log(f"[documents] {f.name}: {n}")
    if "html" in sources and html_cache().exists():
        for p in sorted(html_cache().glob("*.html")):
            aid = p.stem
            if aid in done:
                continue
            v1 = papers.v1_date(aid)
            if not v1:
                n["not_in_papers"] += 1
                continue
            page = p.read_text(encoding="utf-8", errors="replace")
            _put(con, aid, "arxiv_html", v1, page, U.from_html(aid, page))
            done.add(aid)
            n["stored"] += 1
        con.commit()
    counts = {"docs": con.execute("SELECT count(*) FROM docs").fetchone()[0],
              "units": con.execute("SELECT sum(n_units) FROM docs").fetchone()[0] or 0,
              "tables": con.execute("SELECT sum(n_tables) FROM docs").fetchone()[0] or 0}
    con.close()
    store.write_manifest("documents", {"sources": list(sources)}, counts)
    return {**n, **counts}


class Documents:
    """Read-only access by paper id (all later stages use this)."""

    def __init__(self):
        self.con = store.connect("documents", readonly=True)

    def ids(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT arxiv_id FROM docs")]

    def get(self, arxiv_id: str) -> dict | None:
        r = self.con.execute("SELECT source, v1_date, raw_z, units_z FROM docs WHERE arxiv_id=?", (arxiv_id,)).fetchone()
        if not r:
            return None
        return {"arxiv_id": arxiv_id, "source": r[0], "v1_date": r[1],
                "raw": zlib.decompress(r[2]).decode("utf-8"),
                "units": [U.Unit(**d) for d in json.loads(zlib.decompress(r[3]).decode("utf-8"))]}

    def iter(self, ids=None):
        q = "SELECT arxiv_id FROM docs" + ("" if ids is None else "")
        for (aid,) in self.con.execute(q).fetchall():
            if ids is None or aid in ids:
                yield self.get(aid)
