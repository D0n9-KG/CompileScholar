# -*- coding: utf-8 -*-
"""One paper version's document, assembled from the library: get(con, paper_id, version) -> dict.

  key          "<paper_id>@v<N>" (N = arXiv version, 0 = version of record)
  text_date    that version's date (the date every statement and citation sentence read from it carries;
               INTEGRATED-SYSTEM-1005 §2.1) — the arXiv vN date, or the journal online / issued date for version 0
  source       asset channel the text came from (scihub_local is flagged so CS-main and published data can exclude it)
  fast         documents.tei.Doc as a dict (title, abstract, units, sentences, entries, cites) — every acquired PDF
  careful      documents.mineru units + references, when the PDF is in the deep subset (else None)
The derived `documents` stage builds from this when the derived stages move to paper_id keys (phase C-1, together
with the citations rewrite); until then the stage keeps its legacy inputs for the frozen paths."""
from __future__ import annotations

import sqlite3

from ..library import identity
from . import mineru as MI
from . import parse as P
from . import tei as TE


def text_date(con: sqlite3.Connection, pid: str, version: int) -> tuple[str, str] | None:
    """(date_hi, precision) of a paper version's text."""
    if version:
        r = con.execute("SELECT hi, precision FROM dates WHERE paper_id=? AND kind='arxiv_v' AND version=? "
                        "ORDER BY hi LIMIT 1", (pid, version)).fetchone()
    else:
        r = con.execute("SELECT hi, precision FROM dates WHERE paper_id=? AND kind IN ('online','issued') "
                        "ORDER BY hi LIMIT 1", (pid,)).fetchone()
        r = r or con.execute("SELECT hi, precision FROM dates WHERE paper_id=? AND kind IN ('published','venue_year') "
                             "ORDER BY hi DESC LIMIT 1", (pid,)).fetchone()    # secondary: the later is leak-safe
    return tuple(r) if r else None


def versions(con: sqlite3.Connection, pid: str) -> list[int]:
    return [v for (v,) in con.execute("SELECT DISTINCT version FROM assets WHERE paper_id=? ORDER BY version",
                                      (identity.canonical(con, pid),))]


def get(con: sqlite3.Connection, pid: str, version: int) -> dict | None:
    pid = identity.canonical(con, pid)
    a = con.execute("SELECT channel, sha256 FROM assets WHERE paper_id=? AND version=? ORDER BY channel='scihub_local'"
                    " LIMIT 1", (pid, version)).fetchone()
    if not a:
        return None
    channel, sha = a
    parses = dict(con.execute("SELECT parser, status FROM parses WHERE sha256=?", (sha,)).fetchall())
    key = f"{pid}@v{version}"
    out = {"key": key, "paper_id": pid, "version": version, "sha256": sha, "source": channel,
           "text_date": text_date(con, pid, version), "fast": None, "careful": None}
    if parses.get("grobid") == "ok":
        tei, links = P.load_fast(sha)
        out["fast"] = TE.parse(tei, links, key).to_dict()
    if parses.get("mineru") == "ok":
        us, refs = MI.units(MI.load(P.product_path("mineru", sha, "mineru.zip")), key)
        out["careful"] = MI.to_dict(us, refs)
    return out
