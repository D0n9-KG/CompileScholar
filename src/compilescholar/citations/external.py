# -*- coding: utf-8 -*-
"""Stage 2 of the entry-resolution cascade (INTEGRATED-SYSTEM-1005 §7.2), as an offline batch: the stub entries the
citations build left -> the outside world -> the registry. The build itself stays deterministic and offline; after
a pass here, `build citations` re-opens every item (the registry generation digest moved) and stages 1/2/3 find
what this pass created.

Three lanes, per stub group, most-cited first:
  A   the entry carries a DOI the registry does not know -> Crossref works/<doi> (library.import_crossref) ->
      the design's "metadata-only paper" (no assets). Stage 1 then resolves the entry by its own DOI.
  B   Crossref query.bibliographic on the raw entry text — the only path for entries whose title slot holds a
      venue abbreviation (APS / ACS / Nature style: authors + journal + volume + page + year). Acceptance:
        B1  hit title_key == the group's title_key, years within 1  -> the created paper alone is enough
            (stage 3 finds it by title);
        B2  score >= SCORE_MIN, hit year within 1 of the entry year, and a hit surname (len >= 3) occurs in the
            raw text -> the mangled entry cannot find the paper by title or DOI, so the raw-text -> paper_id
            mapping is remembered in registry.ref_resolutions; Resolver stage 2 reads it (method external_ref).
  C   OpenAlex title.search, batched 6 titles per request — the literature Crossref does not cover (NeurIPS /
      Curran, ICLR / OpenReview, tech reports). Acceptance: title_key equality + year within 1.

Identity discipline: a wrong accept poisons the registry, so everything that fails validation stays a stub, and
same-work duplicates created here are the merge queue's job afterwards (propose-titles + adjudicate, as
everywhere else). Measured on real stubs: a correct bibliographic hit scores ~84, wrong hits <= 36.

Re-running is cheap and converges: every HTTP response is cached (sources.http), raws already in
ref_resolutions are skipped, and DOIs/identifiers already in the registry make Writer.add a match, not a
duplicate. `limit` caps the groups one run processes.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..core import ids, paths
from ..core.asof import Date
from ..library import identity, import_crossref, store
from ..sources import http

SCORE_MIN = 55.0        # measured: correct hits ~84, wrong <= 36
_BAD_TYPES = {"component", "journal-issue", "journal-volume", "journal", "proceedings-series"}
_CROSSREF_SELECT = "DOI,title,issued,author,type,score"
_OA_SELECT = "id,display_name,publication_year,publication_date,doi,abstract_inverted_index,authorships"


def raw_sha(raw: str) -> str:
    """The key Resolver stage 2 and this pass share for one raw bibliography text."""
    return hashlib.sha1(re.sub(r"\s+", " ", (raw or "").strip().lower()).encode()).hexdigest()


@dataclass
class Group:
    key: str                                    # 'title:<title_key>' | 'raw:<sha>'
    title: str = ""                             # longest title seen ('' when the entries have none)
    n: int = 0                                  # stub entry rows covered (priority)
    entries: dict = field(default_factory=dict)  # raw_sha -> (raw, year, doi|None)
    dois: list = field(default_factory=list)


@dataclass
class Plan:
    inc: identity.Incoming
    sha: str | None                             # raw_sha to remember in ref_resolutions (lane B2 only)
    method: str
    doi: str | None


def _rep(g: Group):
    return max(g.entries.values(), key=lambda e: len(e[0]))


def _has_table(con, name: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _year_ok(a, b, window: int = 1) -> bool:
    if not a or not b:
        return True
    return abs(int(a) - int(b)) <= window


def _hit_year(it: dict):
    p = ((it.get("issued") or {}).get("date-parts") or [[None]])[0]
    return int(p[0]) if p and p[0] else None


def _surname_in(it: dict, raw: str) -> bool:
    rl = raw.lower()
    for a in it.get("author") or []:
        fam = (a.get("family") or "").strip().lower()
        if len(fam) >= 3 and fam.isalpha() and re.search(rf"\b{re.escape(fam)}\b", rl):
            return True
    return False


def stub_groups(cit, reg, stats: Counter) -> list[Group]:
    """Stub entries from the citations store, deduped: real titles (>= 4 words) share one group (one query for
    every raw spelling of the same work); everything else groups by raw text, because entries whose title slot
    holds a venue name are different works under one 'title'."""
    resolved = {s for (s,) in reg.execute("SELECT raw_sha FROM ref_resolutions")} if _has_table(reg, "ref_resolutions") else set()
    groups: dict[str, Group] = {}
    for cited, raw, title, year, doi in cit.execute(
            "SELECT cited, raw, title, year, doi FROM entries WHERE method='stub'"):
        raw = (raw or "").strip()
        if not raw:
            stats["no_raw"] += 1
            continue
        sha = raw_sha(raw)
        if sha in resolved:
            stats["already"] += 1
            continue
        t = (title or "").strip()
        tk = ids.title_key(t)
        gk = f"title:{tk}" if len(t.split()) >= 4 and tk else f"raw:{sha}"
        g = groups.get(gk)
        if g is None:
            g = groups[gk] = Group(key=gk)
        g.n += 1
        if len(t) > len(g.title):
            g.title = t
        d = ids.normalize_doi(doi) if doi else None
        g.entries.setdefault(sha, (raw, year, d))
        if d and d not in g.dois:
            g.dois.append(d)
    out = sorted(groups.values(), key=lambda g: (-g.n, g.key))
    stats["groups"] = len(out)
    stats["stub_entries"] = sum(g.n for g in out)
    return out


def _crossref_match(g: Group, rep, stats: Counter):
    """query.bibliographic on the representative raw text -> (hit, 'b1'|'b2') or (None, None)."""
    r = http.get("crossref", "https://api.crossref.org/works", params={
        "query.bibliographic": rep[0][:300], "rows": "3", "select": _CROSSREF_SELECT})
    if not r.ok:
        stats["b_http_miss"] += 1
        return None, None
    items = [it for it in ((r.json().get("message") or {}).get("items") or [])
             if it.get("DOI") and (it.get("type") or "") not in _BAD_TYPES]
    if not items:
        stats["b_no_hits"] += 1
        return None, None
    tk_g = ids.title_key(g.title) if g.title else ""
    best = None
    for it in items:
        title = import_crossref.strip_markup(" ".join(it.get("title") or []))
        year = _hit_year(it)
        if tk_g and ids.title_key(title) == tk_g and _year_ok(year, rep[1]):
            return it, "b1"
        if float(it.get("score") or 0.0) >= SCORE_MIN and _year_ok(year, rep[1]) and _surname_in(it, rep[0]):
            if best is None or float(it.get("score") or 0) > float(best.get("score") or 0):
                best = it
    if best is not None:
        return best, "b2"
    stats["b_reject"] += 1
    return None, None


def _resolve_ab(g: Group, stats: Counter) -> Plan | None:
    rep = _rep(g)
    for doi in g.dois:
        inc = import_crossref.record(doi)
        if inc is not None:
            stats["lane_a"] += 1
            return Plan(inc, None, "crossref_doi", doi)
        stats["lane_a_404"] += 1
    hit, basis = _crossref_match(g, rep, stats)
    if hit is None:
        return None
    doi = ids.normalize_doi(hit["DOI"])
    inc = import_crossref.record(doi) if doi else None
    if inc is None:
        stats["b_record_404"] += 1
        return None
    stats[f"lane_{basis}"] += 1
    return Plan(inc, raw_sha(rep[0]) if basis == "b2" else None, f"crossref_{basis}", doi)


def _invert(inv: dict | None) -> str:
    if not inv:
        return ""
    return " ".join(w for _, w in sorted((p, w) for w, ps in inv.items() for p in ps))


def _oa_plan(hit: dict, stats: Counter) -> Plan | None:
    doi = ids.normalize_doi((hit.get("doi") or "").replace("https://doi.org/", ""))
    oa = (hit.get("id") or "").split("/")[-1]
    idl = [("doi", doi, "self")] if doi else ([("openalex", oa, "self")] if oa else None)
    title = (hit.get("display_name") or "").strip()
    if not idl or not title:
        stats["c_bad_hit"] += 1
        return None
    d = Date.parse(hit.get("publication_date") or (str(hit["publication_year"]) if hit.get("publication_year") else ""))
    authors = []
    for a in (hit.get("authorships") or [])[:30]:
        n = ((a.get("author") or {}).get("display_name") or "").strip()
        if n:
            authors.append({"name": n, "surname": n.split()[-1].strip(".,")})
    inc = identity.Incoming(source="openalex", ids=idl, title=title,
                            abstract=_invert(hit.get("abstract_inverted_index")),
                            dates=[("issued", 0, d)] if d is not None else [], authors=authors)
    return Plan(inc, None, "openalex_title", doi)


def _lane_c(groups: list[Group], stats: Counter) -> list[Plan]:
    plans = []
    for i in range(0, len(groups), 6):
        chunk = groups[i:i + 6]
        q = "|".join(re.sub(r"[,:|()\"?!]", " ", g.title)[:200] for g in chunk)
        r = http.get("openalex", "https://api.openalex.org/works", params={
            "filter": f"title.search:{q}", "per-page": "50", "select": _OA_SELECT})
        if not r.ok:
            stats["c_http_miss"] += len(chunk)
            continue
        results = (r.json() or {}).get("results") or []
        for g in chunk:
            rep = _rep(g)
            tk = ids.title_key(g.title)
            cands = [x for x in results
                     if ids.title_key(x.get("display_name")) == tk and _year_ok(x.get("publication_year"), rep[1])]
            if not cands:
                stats["c_miss"] += 1
                continue
            hit = next((x for x in cands if x.get("doi")), cands[0])
            p = _oa_plan(hit, stats)
            if p is not None:
                plans.append(p)
                stats["lane_c"] += 1
    return plans


def _write(plans: list[Plan], stats: Counter, reg_path, log) -> None:
    """Fetch first (done by the caller), then write under the registry lock — import_crossref.run's discipline."""
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S")
    with store.lock(reg_path):
        con = store.connect(reg_path)
        try:
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            ref_rows = []
            for src in ("crossref", "openalex"):
                sub = [p for p in plans if p.inc.source == src]
                if not sub:
                    continue
                w = identity.Writer(con, src)
                for p in sub:
                    pid = w.add(p.inc)
                    if p.sha:
                        ref_rows.append((p.sha, pid, p.method, p.doi, now))
                for k, v in w.close().items():
                    stats[f"writer_{k}"] += v
            con.executemany("INSERT OR REPLACE INTO ref_resolutions VALUES (?,?,?,?,?)", ref_rows)
            stats["ref_rows"] = len(ref_rows)
            stats["undated"] = identity.recompute_first_public(con)
            con.execute("INSERT INTO imports(source, inputs, counts, started_at, finished_at) VALUES (?,?,?,?,?)",
                        ("external_stubs", json.dumps({"plans": len(plans)}), json.dumps(dict(stats), default=str),
                         t0, time.strftime("%Y-%m-%dT%H:%M:%S")))
            con.commit()
        finally:
            con.close()


def run(limit: int | None = None, log=print, cit_path=None, reg_path=None) -> dict:
    """One external-resolution pass over the citations store's stubs. Returns stats; the registry (papers,
    identifiers, ref_resolutions) is the real output. Afterwards: `build citations` re-resolves, and
    `library propose-titles` + `library adjudicate` merge the same-work duplicates this pass may have created."""
    stats: Counter = Counter()
    cit_path = Path(cit_path) if cit_path else paths.derived() / "citations.sqlite"
    if not cit_path.exists():
        log("[citations] external: no citations store yet")
        return {"error": "no citations store"}
    cit = sqlite3.connect(f"file:{cit_path.as_posix()}?mode=ro", uri=True)
    reg = sqlite3.connect(f"file:{(Path(reg_path) if reg_path else store.db_path()).as_posix()}?mode=ro", uri=True)
    try:
        if not _has_table(cit, "entries"):
            log("[citations] external: citations store has no entries table")
            return {"error": "no entries table"}
        groups = stub_groups(cit, reg, stats)
    finally:
        cit.close()
        reg.close()
    if limit:
        groups = groups[:limit]
    plans, lane_c = [], []
    for g in groups:
        try:
            p = _resolve_ab(g, stats)
        except http.Transient:
            stats["transient"] += 1
            continue
        if p is not None:
            plans.append(p)
        elif g.key.startswith("title:") and g.title:
            lane_c.append(g)
        else:
            stats["no_path"] += 1
    try:
        plans += _lane_c(lane_c, stats)
    except http.Transient:
        stats["transient"] += 1
    if plans:
        _write(plans, stats, reg_path, log)
    log(f"[citations] external stub resolution: {json.dumps(dict(stats), default=str)}")
    return dict(stats)
