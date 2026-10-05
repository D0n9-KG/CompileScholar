# -*- coding: utf-8 -*-
"""Qualitative comparison graph at T (L4, N7). User ruling 10-05: no cross-paper numeric tables.

An edge (citing paper -> compared work) exists when a citing paper uses the work as a baseline / point of contrast
(other-statement function in {baseline, contrast} or role == compares). Each edge carries the sentence and, when the
sentence itself states it, the qualitative outcome (citing_better / cited_better / mixed). Numbers are never read or
aligned here; single-paper result units stay in that paper's card.
Pure function of AsOf(T)."""
from __future__ import annotations

from collections import Counter, defaultdict

from .asof import AsOf


def compared_with(view: AsOf, obj: str) -> dict:
    """Who used obj as a baseline / contrast, with outcomes, over time."""
    rows = []
    for s in view.statements(about=obj, kind="other"):
        if s["function"] in ("baseline", "contrast") or s["role"] == "compares":
            rows.append({"by": f"paper:{s['speaker']}", "date": s["date"], "function": s["function"],
                         "outcome": s["meta"].get("outcome"), "quote": s["quote"]})
    rows.sort(key=lambda r: r["date"])
    seen, uniq = set(), []
    for r in rows:
        if (r["by"], r["quote"]) not in seen:
            seen.add((r["by"], r["quote"]))
            uniq.append(r)
    out = Counter(r["outcome"] for r in uniq if r["outcome"])
    return {"object": obj, "as_of": view.T, "n_compared_by": len({r["by"] for r in uniq}),
            "outcomes": dict(out), "first": uniq[0]["date"] if uniq else None, "rows": uniq}


def baselines_in(view: AsOf, papers: list[str]) -> list[tuple[str, int, set[str]]]:
    """Works used as baselines / contrast by the given (citing) papers: (object, n_papers, papers)."""
    by = defaultdict(set)
    for p in papers:
        for s in view.statements(speaker=p.removeprefix("paper:"), kind="other"):
            if (s["function"] in ("baseline", "contrast") or s["role"] == "compares") and view.visible(s["about"]):
                by[s["about"]].add(p)
    return sorted(((o, len(ps), ps) for o, ps in by.items()), key=lambda x: (-x[1], x[0]))
