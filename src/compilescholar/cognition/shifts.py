# -*- coding: utf-8 -*-
"""Reception-shift events at T (L4, N6): dated changes in how the field treats a work, detected from its statements.

Event types (each with a date and the statements that show it; pre-registered thresholds):
  became_component   share of {uses, tool, data, metric} roles/functions in the later half >= COMP and at least
                     COMP_GAIN above the earlier half (P1 signal: lineage start -> default component)
  became_baseline    share of {baseline, contrast, compares} in the later half >= BASE and >= BASE_GAIN above earlier
  limitation_exposed first date at which >= 2 independent papers state a limitation of the work
  superseded         a later work has a `replaces` edge to it and, after that edge's first date, the work is used as
                     a baseline / contrast more than it is extended (needs lineage edges)
  recategorized      the most frequent category phrase of the later half differs from the earlier half's and both
                     are supported by >= 2 statements
Works with fewer than MIN_N reception statements get no events (not enough evidence; the profile says so).
Pure function of AsOf(T): computing at an earlier T can only show events whose evidence existed by then."""
from __future__ import annotations

from collections import Counter

from .asof import AsOf
from .facts import independent

MIN_N = 8
COMP, COMP_GAIN = 0.5, 0.2
BASE, BASE_GAIN = 0.4, 0.2
COMPONENT = {"uses"}
COMPONENT_FN = {"tool", "data", "metric"}
BASELINE_FN = {"baseline", "contrast"}


def _share(rows, pred) -> float:
    return sum(1 for r in rows if pred(r)) / len(rows) if rows else 0.0


def _cat(s):
    return " ".join((s["meta"].get("category") or "").lower().replace("-", " ").split())


def events(view: AsOf, obj: str, lineage_edges: list[dict] | None = None) -> list[dict]:
    others = sorted(view.statements(about=obj, kind="other"), key=lambda s: s["date"])
    per = {}
    for s in others:
        per.setdefault((s["speaker"], s["meta"].get("sentence_id")), s)
    votes = sorted(per.values(), key=lambda s: s["date"])
    out = []
    if len(votes) >= MIN_N:
        mid = len(votes) // 2
        early, late = votes[:mid], votes[mid:]
        comp = lambda s: s["role"] in COMPONENT or s["function"] in COMPONENT_FN
        base = lambda s: s["function"] in BASELINE_FN or s["role"] == "compares"
        ce, cl = _share(early, comp), _share(late, comp)
        if cl >= COMP and cl - ce >= COMP_GAIN:
            out.append({"type": "became_component", "date": late[0]["date"], "early": round(ce, 3),
                        "late": round(cl, 3), "evidence": [s["quote"] for s in late if comp(s)][:3]})
        be, bl = _share(early, base), _share(late, base)
        if bl >= BASE and bl - be >= BASE_GAIN:
            out.append({"type": "became_baseline", "date": late[0]["date"], "early": round(be, 3),
                        "late": round(bl, 3), "evidence": [s["quote"] for s in late if base(s)][:3]})
        ec = Counter(_cat(s) for s in early if _cat(s))
        lc = Counter(_cat(s) for s in late if _cat(s))
        if ec and lc:
            (e1, en), (l1, ln) = ec.most_common(1)[0], lc.most_common(1)[0]
            if e1 != l1 and en >= 2 and ln >= 2:
                out.append({"type": "recategorized", "date": late[0]["date"], "from": e1, "to": l1,
                            "evidence": [s["quote"] for s in late if _cat(s) == l1][:3]})
    # limitation exposed: first date with >= 2 independent stating papers
    seen = set()
    for s in others:
        if s["facet"] == "limitation":
            seen.add(s["speaker"])
            if independent(view, seen) >= 2:
                out.append({"type": "limitation_exposed", "date": s["date"],
                            "evidence": [x["quote"] for x in others if x["facet"] == "limitation"][:3]})
                break
    # superseded
    for e in lineage_edges or []:
        if e["parent"] == obj and e["relation"] == "replaces":
            first = min(x["date"] for x in e["self"] + e["third"])
            after = [s for s in votes if s["date"] >= first]
            if after and _share(after, lambda s: s["function"] in BASELINE_FN) > _share(
                    after, lambda s: s["role"] in ("extends", "improves", "adapts")):
                out.append({"type": "superseded", "date": first, "by": e["child"],
                            "evidence": [x["quote"] for x in e["self"] + e["third"]][:2]})
                break
    return sorted(out, key=lambda ev: ev["date"])
