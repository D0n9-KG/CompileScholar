# -*- coding: utf-8 -*-
"""Qualitative comparison graph at T (phase D). User ruling 10-05: no cross-paper numeric tables — numbers
stay inside the paper that reported them (its card); across papers only the qualitative outcome travels.

compared_with is a bounded per-paper tool (§9.2: compare stays computed online): it reads the other-pass
statements about the paper (function baseline/contrast or role compares) through AsOf, so the cutoff is the
statement's text-version date. The materialised comparison_edge table (cognition.build) is the outcome-meta
subset at field scale — outcome_edges() reads it for statistics and cross-paper sweeps without touching the
extract store per paper."""
from __future__ import annotations

from collections import Counter

from .asof import AsOf


def compared_with(view: AsOf, pid: str) -> dict:
    """Who used `pid` as a baseline / point of comparison, with the sentence's own qualitative outcome."""
    rows = []
    for s in view.statements(about=pid, kind="other"):
        if s["function"] in ("baseline", "contrast") or s["role"] == "compares":
            rows.append({"by": s["speaker"], "date": s["date"], "function": s["function"],
                         "outcome": (s.get("meta") or {}).get("outcome"), "quote": s["quote"]})
    rows.sort(key=lambda r: (r["date"], r["by"]))
    seen, uniq = set(), []
    for r in rows:
        if (r["by"], r["quote"]) not in seen:
            seen.add((r["by"], r["quote"]))
            uniq.append(r)
    return {"object": pid, "as_of": view.T, "n_compared_by": len({r["by"] for r in uniq}),
            "outcomes": dict(Counter(r["outcome"] for r in uniq if r["outcome"])),
            "first": uniq[0]["date"] if uniq else None, "rows": uniq}


def outcome_edges(view: AsOf, pid: str | None = None) -> list[dict]:
    """The materialised outcome edges (a=citing, b=cited) dated <= T — field-scale comparison statistics."""
    if view.cog is None:
        return []
    q = "SELECT stmt_id, a, b, outcome, date FROM comparison_edge WHERE date<=?"
    a: list = [view.T]
    if pid is not None:
        q += " AND (a=? OR b=?)"
        a += [pid, pid]
    return [{"stmt_id": r[0], "citing": r[1], "cited": r[2], "outcome": r[3], "date": r[4]}
            for r in view.cog.execute(q + " ORDER BY date", a)]


def baselines_in(view: AsOf, papers: list[str]) -> list[tuple[str, int, set[str]]]:
    """Works the given (citing) papers use as baselines / points of contrast: (object, n_papers, papers)."""
    by: dict[str, set] = {}
    for p in papers:
        for s in view.statements(speaker=p, kind="other"):
            if (s["function"] in ("baseline", "contrast") or s["role"] == "compares") and view.visible(s["about"]):
                by.setdefault(s["about"], set()).add(p)
    return sorted(((o, len(ps), ps) for o, ps in by.items()), key=lambda x: (-x[1], x[0]))
