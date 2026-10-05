# -*- coding: utf-8 -*-
"""Family-level facts at T (L4, N5): what the field says about a family of works, with its support set.

A fact groups statements about members of one family (cognition.families) that state the same thing. Grouping is
deterministic: statements are clustered by content-word overlap of their text (Jaccard >= SIM) within one facet;
the LLM is not used to decide what agrees with what. For each fact:
  support       statements (speaker, about, date, quote) behind it
  n_papers      distinct speaking papers
  n_independent speaking papers after merging papers that share >= half of their author surnames (same group)
  members       distinct family members the fact is stated about
  status        computed from the support set (DESIGN-CROSSPAPER §2.3, pre-registered rules):
                  consensus     n_independent >= 3 and stated about >= 2 members
                  established   n_independent >= 2
                  single-source otherwise
                and "contested" when a statement about the same member states the opposite (negation / criticizes)
                — only for facets where a contradiction is meaningful (limitation vs. a contribution claim is not one)
  first_seen    date of the earliest support statement (re-running at earlier T shows when the fact formed)
Pure function of AsOf(T)."""
from __future__ import annotations

import json
import re
from collections import defaultdict

from .asof import AsOf

SIM = 0.5
STOP = set("a an the of in on for to and or with by from as is are was were be been this that these those it its "
           "their they we our which such via using use used based into than also can may method methods model models "
           "approach approaches work works paper papers".split())
NEG = re.compile(r"(?i)\b(not|no|fails?|cannot|unable|without|lack\w*|struggl\w*|poor\w*|limited)\b")


def _w(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP and len(w) > 2}


def _jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _authors(view: AsOf, speaker: str) -> set[str]:
    p = view.paper(speaker)
    return {a.lower() for a in (p or {}).get("authors") or []}


def independent(view: AsOf, speakers: set[str]) -> int:
    """Greedy merge of speakers sharing >= half of the smaller author list (same group counts once)."""
    groups: list[set[str]] = []
    for s in sorted(speakers):
        a = _authors(view, s)
        for g in groups:
            if a and g and len(a & g) * 2 >= min(len(a), len(g)):
                g |= a
                break
        else:
            groups.append(set(a) or {s})
    return len(groups)


def facts(view: AsOf, members: list[str], facets=("limitation", "method", "result", "categorization")) -> list[dict]:
    stmts = []
    for m in members:
        for s in view.statements(about=m):
            if s["facet"] in facets and s["text"] and not s["text"].startswith("("):
                stmts.append(s)
    clusters: list[dict] = []
    for s in sorted(stmts, key=lambda s: s["date"]):
        w = _w(s["text"])
        best = max((c for c in clusters if c["facet"] == s["facet"]), key=lambda c: _jac(w, c["words"]), default=None)
        if best is not None and _jac(w, best["words"]) >= SIM:
            best["support"].append(s)
            best["words"] |= w
        else:
            clusters.append({"facet": s["facet"], "words": set(w), "support": [s]})
    out = []
    for c in clusters:
        sup = c["support"]
        speakers = {s["speaker"] for s in sup}
        mem = {s["about"] for s in sup}
        n_ind = independent(view, speakers)
        status = "consensus" if n_ind >= 3 and len(mem) >= 2 else "established" if n_ind >= 2 else "single-source"
        if c["facet"] in ("method", "result"):
            neg = {s["about"] for s in sup if NEG.search(s["text"])}
            pos = {s["about"] for s in sup if not NEG.search(s["text"])}
            if neg & pos:
                status = "contested"
        rep = max(sup, key=lambda s: (s["kind"] == "other", len(s["text"])))
        out.append({"facet": c["facet"], "text": rep["text"], "status": status, "n_papers": len(speakers),
                    "n_independent": n_ind, "members": sorted(mem), "first_seen": sup[0]["date"],
                    "support": [{"by": f"paper:{s['speaker']}", "about": s["about"], "date": s["date"],
                                 "kind": s["kind"], "quote": s["quote"]} for s in sup[:8]]})
    return sorted(out, key=lambda f: (-f["n_independent"], f["first_seen"]))


def dump(f: dict) -> str:
    return json.dumps(f, ensure_ascii=False)
