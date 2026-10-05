# -*- coding: utf-8 -*-
"""L5 tools: the only interface consumers (agents, the answer pipeline, benchmark adapters) use to read field
cognition. Every tool takes as_of (YYYY-MM-DD, inclusive); results are compact, budgeted (MAX_ITEMS, text truncated),
and every item carries the ids and verbatim quotes it rests on. Ids returned are always 'paper:<arxiv_id>' present in
the papers table at as_of, or 'stub:<title>' flagged as such.

Tools
  paper_profile(paper, as_of)          self description vs. how the field describes it, shift, limitations, evidence
  closest_prior(idea, as_of, k)        existing works closest to an idea, each with its own contribution and how the
                                       field describes it (retrieval over papers visible at as_of)
  baselines_for(problem, as_of, k)     works the field uses as baselines / points of comparison for a problem
  field_map(topic, as_of)              families of works around a topic, members, name candidates, boundary counts
  open_issues(topic, as_of)            limitations the field states about works on a topic, with support counts
  frontier(topic, as_of)               newest works on a topic, works turning into components / baselines

Retrieval (`search`) is BM25 over title+abstract of papers with v1_date <= as_of (rank_bm25-free implementation over
an inverted index built lazily per process); it only selects candidates — every judgement comes from cognition."""
from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict

from ..cognition import families as F
from ..cognition import profiles as P
from ..cognition.asof import AsOf
from ..dfc import store

MAX_ITEMS = 8
TXT = 300
_TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("a an the of in on for to and or with by from as is are was were be this that we our these it its via using "
           "based".split())


def _tok(s: str) -> list[str]:
    return [w for w in _TOKEN.findall((s or "").lower()) if w not in STOP and len(w) > 1]


class _Index:
    """BM25 over title + abstract of the papers that have statements (the compiled scope), with v1 dates."""
    _inst = None
    _lock = threading.Lock()

    def __init__(self):
        ext = store.connect("extract", readonly=True)
        pap = store.connect("papers", readonly=True)
        ids = {r[0][6:] for r in ext.execute("SELECT DISTINCT about FROM statements WHERE about LIKE 'paper:%'")}
        ids |= {r[0] for r in ext.execute("SELECT DISTINCT speaker FROM statements")}
        self.docs, self.dates, self.post = [], [], defaultdict(list)
        for aid in sorted(ids):
            r = pap.execute("SELECT title, abstract, v1_date FROM papers WHERE arxiv_id=?", (aid,)).fetchone()
            if not r:
                continue
            toks = _tok(f"{r[0]} {r[0]} {r[1]}")
            i = len(self.docs)
            self.docs.append((aid, len(toks)))
            self.dates.append(r[2])
            for w, c in Counter(toks).items():
                self.post[w].append((i, c))
        self.avg = sum(n for _, n in self.docs) / max(1, len(self.docs))

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._inst is None:
                cls._inst = cls()
            return cls._inst

    def search(self, query: str, as_of: str, k: int = 20) -> list[tuple[str, float]]:
        N = len(self.docs)
        sc = defaultdict(float)
        for w in set(_tok(query)):
            post = self.post.get(w)
            if not post:
                continue
            idf = math.log(1 + (N - len(post) + 0.5) / (len(post) + 0.5))
            for i, c in post:
                if self.dates[i] > as_of:
                    continue
                n = self.docs[i][1]
                sc[i] += idf * c * 2.2 / (c + 1.2 * (0.25 + 0.75 * n / self.avg))
        return [(self.docs[i][0], s) for i, s in sorted(sc.items(), key=lambda kv: -kv[1])[:k]]


def search(query: str, as_of: str, k: int = 20) -> list[str]:
    return [f"paper:{a}" for a, _ in _Index.get().search(query, as_of, k)]


def _t(s, n=TXT):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n - 1] + "…"


def _brief(view: AsOf, obj: str) -> dict:
    pr = P.profile(view, obj)
    paper = pr["paper"] or {}
    own = next((s["text"] for s in pr["self"] if s["facet"] == "contribution"), "") or _t(paper.get("abstract"), 200)
    rec = pr["reception"]
    return {"id": obj, "title": paper.get("title"), "date": paper.get("v1_date"), "self": _t(own, 220),
            "field_says": [_t(d["text"], 160) for d in rec["descriptions"][-3:]],
            "used_as": rec["function_share"], "categories": [c for c, _ in rec["categories"][:3]],
            "n_citing": rec["n_citing"]}


def paper_profile(paper: str, as_of: str) -> dict:
    obj = paper if paper.startswith(("paper:", "stub:")) else f"paper:{paper}"
    view = AsOf(as_of)
    if obj.startswith("paper:") and not view.visible(obj):
        return {"id": obj, "as_of": as_of, "error": "not published by as_of"}
    pr = P.profile(view, obj)
    rec = pr["reception"]
    return {"id": obj, "as_of": as_of, "title": (pr["paper"] or {}).get("title"),
            "self": [{"facet": s["facet"], "text": _t(s["text"], 200), "quote": _t(s["quote"], 240)}
                     for s in pr["self"][:MAX_ITEMS]],
            "field": {"n_citing": rec["n_citing"], "first": rec["first"], "last": rec["last"],
                      "used_as": rec["function_share"], "relation": rec["relation_share"],
                      "categories": rec["categories"], "limitations": rec["limitations"][:5],
                      "descriptions": [{"date": d["date"], "by": f"paper:{d['citing']}", "text": _t(d["text"], 200),
                                        "quote": _t(d["quote"], 260)} for d in rec["descriptions"]]},
            "shift": pr["shift"], "evidence": pr["evidence"]}


def closest_prior(idea: str, as_of: str, k: int = MAX_ITEMS) -> list[dict]:
    view = AsOf(as_of)
    return [_brief(view, o) for o in search(idea, as_of, k)]


def baselines_for(problem: str, as_of: str, k: int = MAX_ITEMS) -> list[dict]:
    """Candidates from retrieval, then everything the field has used as a baseline/contrast in papers about the
    problem; ranked by how many distinct papers used it that way."""
    view = AsOf(as_of)
    seeds = search(problem, as_of, 30)
    count, by = Counter(), defaultdict(set)
    for sd in seeds:
        for s in view.statements(speaker=sd[6:], kind="other"):
            if s["function"] in ("baseline", "contrast") or s["role"] == "compares":
                count[s["about"]] += 1
                by[s["about"]].add(sd)
    out = []
    for obj, n in count.most_common(k):
        b = _brief(view, obj) if obj.startswith("paper:") else {"id": obj, "title": obj[5:], "stub": True}
        out.append({**b, "used_as_baseline_by": n, "in_papers": sorted(by[obj])[:5]})
    return out


def field_map(topic: str, as_of: str) -> dict:
    view = AsOf(as_of)
    seeds = set(search(topic, as_of, 40))
    scope = set(seeds)
    for sd in list(seeds):
        scope |= {r for r in view.references(sd[6:]) if r.startswith("paper:") and view.visible(r)}
    fams = F.families(view, scope)[:MAX_ITEMS]
    out = []
    for f in fams:
        out.append({"name_candidates": [c for c, _ in f["name_candidates"]],
                    "members": [_brief(view, m) for m in f["members"][:6]],
                    "n_members": len(f["members"]), "n_citing": f["n_citing"]})
    boundary = Counter(r for sd in seeds for r in view.references(sd[6:]) if r.startswith("stub:"))
    return {"topic": topic, "as_of": as_of, "families": out,
            "boundary": {"cited_not_in_corpus": len(boundary),
                         "most_cited_unknown": [o[5:] for o, _ in boundary.most_common(5)]}}


def open_issues(topic: str, as_of: str) -> list[dict]:
    view = AsOf(as_of)
    lim = defaultdict(lambda: {"by": set(), "about": set(), "quote": ""})
    for sd in search(topic, as_of, 30):
        for s in view.statements(about=sd, facets=("limitation",)):
            key = _t(s["text"], 160).lower()
            lim[key]["by"].add(s["speaker"])
            lim[key]["about"].add(s["about"])
            lim[key]["quote"] = lim[key]["quote"] or _t(s["quote"], 240)
    ranked = sorted(lim.items(), key=lambda kv: -len(kv[1]["by"]))[:MAX_ITEMS]
    return [{"limitation": k, "n_papers_stating": len(v["by"]), "about": sorted(v["about"])[:5],
             "quote": v["quote"]} for k, v in ranked]


def frontier(topic: str, as_of: str) -> dict:
    view = AsOf(as_of)
    cands = search(topic, as_of, 40)
    briefs = [_brief(view, o) for o in cands]
    newest = sorted(briefs, key=lambda b: b["date"] or "", reverse=True)[:MAX_ITEMS]
    turning = []
    for o in cands:
        pr = P.profile(view, o)
        if pr["shift"] and pr["shift"]["late"]["lineage"] < pr["shift"]["early"]["lineage"]:
            turning.append({"id": o, "title": (pr["paper"] or {}).get("title"), "early": pr["shift"]["early"],
                            "late": pr["shift"]["late"], "since": pr["shift"]["split_date"]})
    return {"topic": topic, "as_of": as_of, "newest": newest, "becoming_standard": turning[:MAX_ITEMS]}
