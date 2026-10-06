# -*- coding: utf-8 -*-
"""arXiv metadata -> the registry: every arXiv paper in scope, with every version's date.

Sources: the local OAI snapshot (one JSON per line; `versions` lists each version's created date) and the OAI-PMH
harvest (sources.arxiv_oai, format 2: version_dates). Scope: CS by default (categories include cs.* or one of
EXTRA_CATS, cross-listings count), or every record (scope="all"). Per paper:
  identifiers  arxiv:<id> (self); each DOI on the record (published_version — a multi-DOI field is split);
               an arXiv DOI in the doi field is ignored (it names the same record)
  dates        arxiv_v, version N, day precision, one row per version
  records      source=arxiv, version = the latest version (arXiv metadata is the latest version's title / abstract)
  authors      source=arxiv
The snapshot and the harvest overlap; the harvest copy (newer) wins for the record and adds versions."""
from __future__ import annotations

import json
import re
import time

from ..core import ids
from ..core.asof import Date
from ..sources import arxiv_oai
from ..sources.arxiv_snapshot import snapshot_path
from . import identity, store

EXTRA_CATS = ("stat.ML", "eess.AS", "eess.IV", "eess.SP")
OTHER_SETS = ("physics", "math", "eess", "stat", "q-bio", "q-fin", "econ")   # OAI sets besides cs


def in_scope(categories: list[str], scope: str) -> bool:
    return scope == "all" or any(c.startswith("cs.") or c in EXTRA_CATS for c in categories)


def _dois(field: str | None) -> list[str]:
    out = []
    for part in re.split(r"[\s,;]+", field or ""):
        d = ids.normalize_doi(part)
        if d and d not in out:
            out.append(d)
    return out


def _incoming(aid: str, vdates: dict, title: str, abstract: str, authors: list, cats: list, doi_field) -> \
        identity.Incoming | None:
    aid = ids.normalize_arxiv(aid)
    if not aid:
        return None
    dates, latest = [], 0
    for v, d in vdates.items():
        m = re.fullmatch(r"v(\d+)", v or "")
        if m and d:
            n = int(m.group(1))
            dates.append(("arxiv_v", n, Date.parse(d)))
            latest = max(latest, n)
    hi = next((d.hi_iso for k, n, d in dates if n == latest and d), None)
    return identity.Incoming(
        source="arxiv", ids=[("arxiv", aid, "self")] + [("doi", d, "published_version") for d in _dois(doi_field)],
        title=re.sub(r"\s+", " ", title or "").strip(), abstract=re.sub(r"\s+", " ", abstract or "").strip(),
        categories=cats, version=latest or None, version_hi=hi, dates=dates, authors=authors)


def _snapshot(scope: str, only: set | None = None):
    with open(snapshot_path(), "rb") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            cats = (r.get("categories") or "").split()
            if only is not None:
                if ids.normalize_arxiv(r.get("id")) not in only:
                    continue
            elif not in_scope(cats, scope):
                continue
            vd = {v.get("version"): arxiv_oai.rfc2822_day(v.get("created") or "") for v in r.get("versions") or []}
            authors = [{"name": " ".join(x for x in (a[1], a[0]) if x).strip(), "surname": a[0]}
                       for a in r.get("authors_parsed") or [] if a and a[0]]
            doi = r.get("doi")
            inc = _incoming(r.get("id"), vd, r.get("title"), r.get("abstract"), authors, cats,
                            doi if doi and doi != "None" else None)
            if inc:
                yield inc


def _harvest(scope: str, sets=("cs",), only=None):
    """Harvested OAI records of `sets`; only (a callable returning a set of ids, evaluated when the pass starts)
    restricts them to those ids."""
    want = only() if only else None
    for r in (x for s in sets for x in arxiv_oai.iter_records(s)):
        cats = r.get("categories") or []
        if "version_dates" not in r:
            continue
        if want is not None:
            if ids.normalize_arxiv(r["arxiv_id"]) not in want:
                continue
        elif not in_scope(cats, scope):
            continue
        inc = _incoming(r["arxiv_id"], r["version_dates"], r.get("title"), r.get("abstract"),
                        arxiv_oai.author_names(r.get("authors_raw") or ""), cats, r.get("doi"))
        if inc:
            yield inc


def missing_version_dates(con) -> set:
    """arXiv ids in the registry (e.g. a benchmark's non-CS members) with no arXiv version date yet."""
    return {v for (v,) in con.execute(
        "SELECT i.value FROM identifiers i WHERE i.scheme='arxiv' AND NOT EXISTS (SELECT 1 FROM dates d "
        "WHERE d.paper_id = i.paper_id AND d.kind = 'arxiv_v')")}


def run(scope: str = "cs", log=print, path=None, fill_members: bool = False) -> dict:
    """scope=cs|all imports every record in scope. fill_members=True instead fills the arXiv ids already in the
    registry without version dates (out-of-scope members of a benchmark): from the snapshot, then from the OAI
    harvest of the other sets (OTHER_SETS; experiments/corpus/harvest_oai.py <set>) — their dates and records only."""
    t0 = time.time()
    with store.lock(path):
        con = store.connect(path)
        out = {}
        if fill_members:
            want = missing_version_dates(con)
            log(f"[library] arxiv ids without version dates: {len(want):,}")
            passes = (("snapshot_members", _snapshot("all", only=want)),
                      ("oai_members", _harvest("all", sets=OTHER_SETS, only=lambda: missing_version_dates(con))))
        else:
            passes = (("snapshot", _snapshot(scope)), ("oai", _harvest(scope)))
        for name, it in passes:
            w = identity.Writer(con, "arxiv")
            for i, inc in enumerate(it, 1):
                w.add(inc)
                if i % 200000 == 0:
                    log(f"[library] arxiv {name}: {i:,} ({time.time() - t0:.0f}s)")
            out[name] = w.close()
            log(f"[library] arxiv {name}: {out[name]}")
        undated = identity.recompute_first_public(con)
        snap = snapshot_path()
        st = snap.stat()
        con.execute("INSERT INTO imports(source, inputs, counts, started_at, finished_at) VALUES (?,?,?,?,?)",
                    ("arxiv", json.dumps({"scope": "members" if fill_members else scope,
                                          "snapshot": [str(snap), st.st_size, int(st.st_mtime)],
                                          "oai_windows": len(list(arxiv_oai.out_dir().glob("*.jsonl")))}),
                     json.dumps({**out, "undated": undated}), time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                     time.strftime("%Y-%m-%dT%H:%M:%S")))
        con.commit()
        con.close()
    return {**out, "undated": undated, "seconds": round(time.time() - t0)}
