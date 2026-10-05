# -*- coding: utf-8 -*-
"""L6 tools: the only interface consumers (agents via MCP, the built-in answer pipeline, benchmark adapters) use.

Five groups (DESIGN-LITERATURE-LAYER §5). Every tool takes as_of (YYYY-MM-DD, inclusive) and never returns anything
dated after it; results are compact and budgeted (MAX_ITEMS, text truncated); every item carries ids and verbatim
quotes. Ids are always 'paper:<arxiv_id>' visible at as_of, or 'stub:<title>' flagged as such.

  find      search_papers(query)            paper-level hybrid search (L5 index)
            citations_of(paper) / references_of(paper)
            what_is_missing(topic)          boundary: works cited around a topic that the corpus has not read
  read      paper_card(paper)               self description: contributions, proposals, method, findings, results,
                                            setting, limitations (single-paper result units only)
            read(paper, section?)           passages of a paper's text
  evidence  find_evidence(claim)            verbatim sentences (passages + statements) that bear on a claim
  field     field_map(topic)                families around a topic, members, name candidates, family facts, boundary
            paper_profile(paper)            self vs. field description, reception shift, events, lineage, evidence
            closest_prior(idea)             existing works closest to an idea, each with own contribution + reception
            baselines_for(problem)          works the field uses as baselines for a problem (qualitative)
            open_issues(topic)              limitations the field states, with support set and status
            frontier(topic)                 newest works; works becoming components / baselines; superseded works
  compare   compared_with(paper)            qualitative comparison graph (who compared against it, outcomes); no
                                            cross-paper numbers (user ruling 10-05)
"""
from __future__ import annotations

import re
import threading

from ..cognition import comparisons as C
from ..cognition import facts as FA
from ..cognition import families as F
from ..cognition import lineage as L
from ..cognition import profiles as P
from ..cognition import shifts as SH
from ..cognition.asof import AsOf
from ..cognition.identity import Identity

__all__ = ["AsOf"]   # re-exported for in-process consumers (the answer pipeline reads abstracts through it)
MAX_ITEMS = 8
TXT = 300
_lock = threading.Lock()
_index = [None]


def _idx():
    with _lock:
        if _index[0] is None:
            from ..index.build import Index
            try:
                from ..llm.embedding import embed_local
                emb = embed_local
            except Exception:  # pragma: no cover
                emb = None
            _index[0] = Index(embed=emb)
        return _index[0]


def set_index(index) -> None:
    """Inject an index (tests, BM25-only runs)."""
    _index[0] = index


def _t(s, n=TXT):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n - 1] + "…"


def _pid(paper: str) -> str:
    return paper if paper.startswith(("paper:", "stub:")) else f"paper:{paper}"


def _brief(view: AsOf, obj: str) -> dict:
    pr = P.profile(view, obj)
    paper = pr["paper"] or {}
    own = next((s["text"] for s in pr["self"] if s["facet"] == "contribution"), "") or _t(paper.get("abstract"), 200)
    rec = pr["reception"]
    return {"id": obj, "title": paper.get("title"), "date": paper.get("v1_date"), "self": _t(own, 220),
            "field_says": [_t(d["text"], 160) for d in rec["descriptions"][-3:]],
            "used_as": rec["function_share"], "categories": [c for c, _ in rec["categories"][:3]],
            "n_citing": rec["n_citing"]}


# ------------------------------------------------------------------ find
def search_papers(query: str, as_of: str, k: int = MAX_ITEMS) -> list[dict]:
    view = AsOf(as_of)
    return [_brief(view, o) for o in _idx().papers(query, as_of, k)]


def citations_of(paper: str, as_of: str, k: int = 20) -> list[dict]:
    view, obj = AsOf(as_of), _pid(paper)
    rows = {}
    for citing, date, sid in view.cited_by(obj):
        rows.setdefault(citing, date)
    return [{"id": f"paper:{c}", "date": d, "title": (view.paper(c) or {}).get("title")}
            for c, d in sorted(rows.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def references_of(paper: str, as_of: str) -> list[dict]:
    view = AsOf(as_of)
    out = []
    for r in view.references(_pid(paper)[6:]):
        if r.startswith("paper:"):
            p = view.paper(r[6:])
            if p:
                out.append({"id": r, "title": p["title"], "date": p["v1_date"]})
        else:
            out.append({"id": r, "title": r[5:], "stub": True})
    return out


def what_is_missing(topic: str, as_of: str) -> dict:
    """Works cited by the papers around a topic that the corpus has no text for, ranked by how many of those papers
    cite them; plus stubs (cited works we cannot even resolve to an arXiv id)."""
    view = AsOf(as_of)
    from ..documents.build import Documents
    have = set(Documents().ids())
    cnt, stubs = {}, {}
    for o in _idx().papers(topic, as_of, 30):
        for r in view.references(o[6:]):
            if r.startswith("paper:") and r[6:] not in have and view.visible(r):
                cnt[r] = cnt.get(r, 0) + 1
            elif r.startswith("stub:"):
                stubs[r] = stubs.get(r, 0) + 1
    unread = sorted(cnt.items(), key=lambda kv: -kv[1])[:MAX_ITEMS]
    return {"topic": topic, "as_of": as_of,
            "known_unread": [{"id": o, "title": (view.paper(o[6:]) or {}).get("title"), "cited_by_n": n}
                             for o, n in unread],
            "unresolved_cited": [{"title": s[5:], "cited_by_n": n}
                                 for s, n in sorted(stubs.items(), key=lambda kv: -kv[1])[:MAX_ITEMS]]}


# ------------------------------------------------------------------ read
def paper_card(paper: str, as_of: str) -> dict:
    view, obj = AsOf(as_of), _pid(paper)
    p = view.paper(obj[6:]) if obj.startswith("paper:") else None
    if obj.startswith("paper:") and not p:
        return {"id": obj, "as_of": as_of, "error": "not published by as_of"}
    selfs = view.statements(about=obj, kind="self")
    by = lambda f, role=None: [{"text": _t(s["text"], 220), "quote": _t(s["quote"], 260)}
                               for s in selfs if s["facet"] == f and (role is None or s["role"] == role)]
    results = [s for s in selfs if s["facet"] == "result" and (s["meta"] or {}).get("pass") == "results"]
    setting = next((s["meta"] for s in selfs if s["facet"] == "setting"), {})
    return {"id": obj, "as_of": as_of, "title": (p or {}).get("title"), "date": (p or {}).get("v1_date"),
            "contributions": by("contribution", "proposes")[:MAX_ITEMS],
            "proposes": [{"name": s["meta"].get("name"), "artefact": s["meta"].get("artefact"),
                          "quote": _t(s["quote"], 260)} for s in selfs if (s["meta"] or {}).get("name")],
            "method": by("method")[:MAX_ITEMS],
            "findings": [{"text": _t(s["text"], 220), "quote": _t(s["quote"], 260)} for s in selfs
                         if s["facet"] == "result" and (s["meta"] or {}).get("pass") != "results"][:MAX_ITEMS],
            "result_units": [{"method": s["meta"].get("method"), "dataset": s["meta"].get("dataset"),
                              "metric": s["meta"].get("metric"), "value": s["meta"].get("value"),
                              "unit": s["meta"].get("unit"), "row": _t(s["quote"], 200)} for s in results[:40]],
            "result_units_scope": "values from this paper's own tables only; not comparable across papers",
            "setting": {k: v for k, v in setting.items() if k != "pass"},
            "limitations": by("limitation")[:MAX_ITEMS],
            "references": len(view.references(obj[6:])) if obj.startswith("paper:") else 0}


def read(paper: str, as_of: str, section: str | None = None, query: str | None = None, k: int = 6) -> list[dict]:
    obj = _pid(paper)
    if not AsOf(as_of).visible(obj):
        return []
    if query:
        return _idx().passages(query, as_of, k, paper=obj)
    from ..documents.build import Documents
    d = Documents().get(obj[6:])
    if not d:
        return []
    us = [u for u in d["units"] if u.kind in ("para", "caption") and (section is None or section.lower() in u.section.lower())]
    return [{"uid": u.uid, "section": u.section, "text": _t(u.text, 1200)} for u in us[:k]]


# ------------------------------------------------------------------ evidence
def find_evidence(claim: str, as_of: str, k: int = MAX_ITEMS) -> list[dict]:
    view = AsOf(as_of)
    ids = _idx().statements(claim, as_of, k)
    rows = {r["id"]: r for r in view.statements_by_id(ids)}
    out = [{"kind": rows[i]["kind"], "by": f"paper:{rows[i]['speaker']}", "about": rows[i]["about"],
            "date": rows[i]["date"], "facet": rows[i]["facet"], "text": _t(rows[i]["text"], 200),
            "quote": _t(rows[i]["quote"], 300)} for i in ids if i in rows]
    for p in _idx().passages(claim, as_of, max(0, k - len(out))):
        out.append({"kind": "passage", "by": p["paper"], "date": p["date"], "section": p["section"],
                    "quote": _t(p["text"], 300)})
    return out[:k]


# ------------------------------------------------------------------ field (diachronic field cognition)
def _scope(view: AsOf, topic: str) -> tuple[list[str], set[str]]:
    seeds = _idx().papers(topic, view.T, 40)
    scope = set(seeds)
    for sd in seeds:
        scope |= {r for r in view.references(sd[6:]) if r.startswith("paper:") and view.visible(r)}
    return seeds, scope


def field_map(topic: str, as_of: str) -> dict:
    view = AsOf(as_of)
    seeds, scope = _scope(view, topic)
    out = []
    for f in F.families(view, scope)[:MAX_ITEMS]:
        fx = FA.facts(view, f["members"][:12])
        out.append({"name_candidates": [c for c, _ in f["name_candidates"]],
                    "members": [_brief(view, m) for m in f["members"][:6]], "n_members": len(f["members"]),
                    "n_citing": f["n_citing"],
                    "facts": [{k: x[k] for k in ("facet", "text", "status", "n_independent", "first_seen")}
                              for x in fx[:4]]})
    stubs = {}
    for sd in seeds:
        for r in view.references(sd[6:]):
            if r.startswith("stub:"):
                stubs[r] = stubs.get(r, 0) + 1
    return {"topic": topic, "as_of": as_of, "families": out,
            "boundary": {"cited_not_resolved": len(stubs),
                         "most_cited_unresolved": [s[5:] for s, _ in sorted(stubs.items(), key=lambda kv: -kv[1])[:5]]}}


def paper_profile(paper: str, as_of: str) -> dict:
    view, obj = AsOf(as_of), _pid(paper)
    if obj.startswith("paper:") and not view.visible(obj):
        return {"id": obj, "as_of": as_of, "error": "not published by as_of"}
    pr = P.profile(view, obj)
    rec = pr["reception"]
    ident = Identity(view)
    lin = L.lineage_of(view, obj, ident=ident)
    edges = L.edges(view, ident)["edges"]
    return {"id": obj, "as_of": as_of, "title": (pr["paper"] or {}).get("title"),
            "self": [{"facet": s["facet"], "text": _t(s["text"], 200), "quote": _t(s["quote"], 240)}
                     for s in pr["self"] if s["facet"] in ("contribution", "limitation")][:MAX_ITEMS],
            "field": {"n_citing": rec["n_citing"], "first": rec["first"], "last": rec["last"],
                      "used_as": rec["function_share"], "relation": rec["relation_share"],
                      "categories": rec["categories"], "limitations": rec["limitations"][:5],
                      "descriptions": [{"date": d["date"], "by": f"paper:{d['citing']}", "text": _t(d["text"], 200),
                                        "quote": _t(d["quote"], 260)} for d in rec["descriptions"]]},
            "shift": pr["shift"], "events": SH.events(view, obj, edges),
            "lineage": {"parents": [{"id": e["parent"], "relation": e["relation"], "self": len(e["self"]),
                                     "third": len(e["third"])} for e in lin["parents"][:MAX_ITEMS]],
                        "children": [{"id": e["child"], "relation": e["relation"], "self": len(e["self"]),
                                      "third": len(e["third"])} for e in lin["children"][:MAX_ITEMS]]},
            "known_as": ident.aliases(obj)[:5], "evidence": pr["evidence"]}


def closest_prior(idea: str, as_of: str, k: int = MAX_ITEMS) -> list[dict]:
    view = AsOf(as_of)
    return [_brief(view, o) for o in _idx().papers(idea, as_of, k)]


def baselines_for(problem: str, as_of: str, k: int = MAX_ITEMS) -> list[dict]:
    view = AsOf(as_of)
    seeds = _idx().papers(problem, as_of, 30)
    out = []
    for obj, n, ps in C.baselines_in(view, seeds)[:k]:
        b = _brief(view, obj) if obj.startswith("paper:") else {"id": obj, "title": obj[5:], "stub": True}
        out.append({**b, "used_as_baseline_by": n, "in_papers": sorted(ps)[:5]})
    return out


def open_issues(topic: str, as_of: str) -> list[dict]:
    view = AsOf(as_of)
    _, scope = _scope(view, topic)
    fx = FA.facts(view, sorted(scope)[:200], facets=("limitation",))
    return [{"limitation": _t(x["text"], 200), "status": x["status"], "n_independent": x["n_independent"],
             "about": x["members"][:5], "first_seen": x["first_seen"], "quote": _t(x["support"][0]["quote"], 240)}
            for x in fx[:MAX_ITEMS]]


def frontier(topic: str, as_of: str) -> dict:
    view = AsOf(as_of)
    cands = _idx().papers(topic, as_of, 40)
    briefs = [_brief(view, o) for o in cands]
    edges = L.edges(view)["edges"]
    turning, superseded = [], []
    for o in cands:
        for ev in SH.events(view, o, edges):
            if ev["type"] in ("became_component", "became_baseline"):
                turning.append({"id": o, "event": ev["type"], "since": ev["date"]})
            if ev["type"] == "superseded":
                superseded.append({"id": o, "by": ev["by"], "since": ev["date"]})
    return {"topic": topic, "as_of": as_of,
            "newest": sorted(briefs, key=lambda b: b["date"] or "", reverse=True)[:MAX_ITEMS],
            "becoming_standard": turning[:MAX_ITEMS], "superseded": superseded[:MAX_ITEMS]}


# ------------------------------------------------------------------ compare (qualitative only)
def compared_with(paper: str, as_of: str) -> dict:
    view, obj = AsOf(as_of), _pid(paper)
    g = C.compared_with(view, obj)
    g["rows"] = [{**r, "quote": _t(r["quote"], 240)} for r in g["rows"][-MAX_ITEMS:]]
    return g


TOOLS = {f.__name__: f for f in (search_papers, citations_of, references_of, what_is_missing, paper_card, read,
                                 find_evidence, field_map, paper_profile, closest_prior, baselines_for, open_issues,
                                 frontier, compared_with)}
