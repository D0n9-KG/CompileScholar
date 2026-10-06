# -*- coding: utf-8 -*-
"""sci-evo-extract's registry (490 papers, mostly granular flow) -> the registry, cleaned
(FITNESS-LIBRARY-ACQUIRE "标识污染"; INTEGRATED-SYSTEM-1005 §3).

Kept identifiers: the user-entered DOI (source=user_input; a 10.48550/arXiv DOI becomes an arXiv id), the arXiv id
from arXiv metadata, the OpenAlex work id when the paper has exactly one. Dropped: every DOI that came from a search
candidate (crossref / openalex / sciverse sources — 13 papers carried 2-12 DOIs of other works this way), the
`crossref` / `local_doi_archive` / `sciverse` pseudo-identifiers (pointers, not names). Skipped papers: archived ones
and test fixtures (the reserved 10.1000/ test prefix: E2E / smoke runs). A paper with no DOI and no arXiv id keeps its
title and year (paper_id title:...), and its title is matched later through the queue like any other.
Dates: sci-evo's published_date / year are unverified (year_source varies) and enter as kind=published (secondary).
Assets (PDF, MinerU outputs) are not migrated here: acquire re-registers them with identity checks (phase B-6);
`asset_map()` lists them for that step."""
from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from ..core import ids, paths
from ..core.asof import Date
from . import identity, store

TEST_DOI_PREFIX = "10.1000/"


def registry_path() -> Path:
    return paths.resource("scievo_registry")


def _rows(src: sqlite3.Connection):
    idents = defaultdict(list)
    for pid, scheme, value, source in src.execute("SELECT paper_id, scheme, value, source FROM paper_identifiers"):
        idents[pid].append((scheme, value, source))
    for pid, ndoi, title, year, venue, archived, pdate in src.execute(
            "SELECT paper_id, normalized_doi, title, year, venue, archived, published_date FROM papers"):
        yield pid, ndoi, title, year, venue, archived, pdate, idents[pid]


def clean(pid, ndoi, title, year, venue, archived, pdate, idents) -> tuple[identity.Incoming | None, str]:
    """(incoming record, reason) — reason names why a paper is skipped or what was dropped."""
    if archived:
        return None, "archived"
    user = [v for s, v, src in idents if s == "doi" and src == "user_input"] or ([ndoi] if ndoi else [])
    if any((ids.normalize_doi(v) or "").startswith(TEST_DOI_PREFIX) for v in user):
        return None, "test fixture"
    out = []
    for v in user:
        d, ax = ids.normalize_doi(v), ids.arxiv_from_doi(v)
        if d:
            out.append(("doi", d, "self"))
        elif ax:
            out.append(("arxiv", ax, "self"))
    for s, v, src in idents:
        if s == "arxiv" and src == "arxiv" and ids.normalize_arxiv(v):
            out.append(("arxiv", ids.normalize_arxiv(v), "self"))
    oa = {v for s, v, _ in idents if s == "openalex"}
    if len(oa) == 1:
        out.append(("openalex", oa.pop(), "self"))
    out = list(dict.fromkeys(out))
    # one value per scheme: two arXiv ids or two DOIs on one sci-evo paper are pollution, not versions
    seen, kept = set(), []
    for s, v, r in out:
        if s in seen:
            continue
        seen.add(s)
        kept.append((s, v, r))
    kept.append(("scievo", pid, "self"))
    kept.sort(key=lambda x: {"doi": 0, "arxiv": 1, "openalex": 2, "scievo": 3}[x[0]])
    dropped = len([1 for s, *_ in idents if s == "doi"]) - len([1 for s, *_ in kept if s == "doi"])
    d = Date.parse(pdate) if pdate else (Date.parse(int(year)) if year else None)
    return identity.Incoming(source="scievo", ids=kept, title=title or "", venue=venue or "",
                             dates=[("published", 0, d)] if d else []), f"dropped {dropped} doi"


def run(log=print, path=None) -> dict:
    src = sqlite3.connect(f"file:{registry_path().as_posix()}?mode=ro", uri=True)
    t0 = time.time()
    counts = defaultdict(int)
    with store.lock(path):
        con = store.connect(path)
        w = identity.Writer(con, "scievo")
        for row in _rows(src):
            inc, why = clean(*row)
            if inc is None:
                counts[f"skipped: {why}"] += 1
                continue
            counts["papers"] += 1
            counts["dropped_dois"] += int(why.split()[1])
            w.add(inc)
        counts.update(w.close())
        counts["undated"] = identity.recompute_first_public(con)
        con.execute("INSERT INTO imports(source, inputs, counts, started_at, finished_at) VALUES (?,?,?,?,?)",
                    ("scievo", json.dumps({"registry": str(registry_path())}), json.dumps(counts),
                     time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)), time.strftime("%Y-%m-%dT%H:%M:%S")))
        con.commit()
        con.close()
    src.close()
    log(f"[library] scievo: {dict(counts)}")
    return dict(counts)


def asset_map() -> list[dict]:
    """sci-evo PDF assets (paper_id in sci-evo, sha256, source_kind, path relative to the sci-evo library root), for
    acquire to re-register with identity checks."""
    src = sqlite3.connect(f"file:{registry_path().as_posix()}?mode=ro", uri=True)
    rows = [dict(zip(("scievo_id", "sha256", "source_kind", "local_path"), r)) for r in src.execute(
        "SELECT paper_id, sha256, source_kind, local_path FROM pdf_assets WHERE status='ready'")]
    src.close()
    return rows
