# -*- coding: utf-8 -*-
"""Lineage at T (phase D reader): reads the materialised lineage_edge / lineage_hyper tables.

The build assembles edges from three assertion sources (a self statement's mentions, an other statement's
role, a third-party builds_on), drops time-inconsistent claims (a child cannot extend a parent that did not
exist yet) and dates every edge with the §2.4 effective date valid_from = max(statement date, both papers'
first public dates) — so an edge is visible at T exactly when BOTH works were public and the claiming
sentence's text version existed. kind: 'self' = the child paper's own claim; 'third' = a third paper's
assertion (the two are never merged into one count). n-ary combines live in lineage_hyper (one row per
(child, member) of the same sentence). Assertion quotes are fetched from the extract store on demand —
an edge keeps statement ids, not text."""
from __future__ import annotations

from .asof import AsOf


def edges(view: AsOf, pid: str | None = None) -> list[dict]:
    """(child, parent, relation) edges valid at T, aggregated over their assertions, earliest-valid first."""
    if view.cog is None:
        return []
    q = ("SELECT stmt_id, child, parent, relation, speaker, date, valid_from, kind FROM lineage_edge "
         "WHERE valid_from<=?")
    a: list = [view.T]
    if pid is not None:
        q += " AND (child=? OR parent=?)"
        a += [pid, pid]
    agg: dict[tuple, dict] = {}
    for sid, ch, pa, rel, sp, date, vf, kind in view.cog.execute(q + " ORDER BY date", a):
        e = agg.get((ch, pa, rel))
        if e is None:
            e = agg[(ch, pa, rel)] = {"child": ch, "parent": pa, "relation": rel, "valid_from": vf,
                                      "assertions": []}
        e["valid_from"] = min(e["valid_from"], vf)
        e["assertions"].append({"by": sp, "date": date, "kind": kind, "stmt_id": sid})
    return sorted(agg.values(), key=lambda e: (e["valid_from"], e["child"], e["parent"], e["relation"]))


def quotes(view: AsOf, edge: dict, k: int = 3) -> list[dict]:
    """The verbatim quotes behind an edge's first k assertions (bounded extract-store lookup)."""
    ids = [a["stmt_id"] for a in edge["assertions"][:k] if a.get("stmt_id") is not None]
    by_id = {s["id"]: s for s in view.statements_by_id(ids)}
    return [{"by": a["by"], "date": a["date"], "kind": a["kind"], "quote": by_id[a["stmt_id"]]["quote"]}
            for a in edge["assertions"][:k] if a["stmt_id"] in by_id]


def combines(view: AsOf, pid: str | None = None) -> list[dict]:
    """The n-ary 'combines' hyperedges valid at T, grouped back per asserting sentence."""
    if view.cog is None:
        return []
    q = "SELECT stmt_id, child, member, relation, speaker, date, valid_from FROM lineage_hyper WHERE valid_from<=?"
    a: list = [view.T]
    if pid is not None:
        q += " AND (child=? OR member=?)"
        a += [pid, pid]
    groups: dict = {}
    for sid, ch, m, rel, sp, date, vf in view.cog.execute(q + " ORDER BY date", a):
        g = groups.setdefault(sid, {"child": ch, "relation": rel, "members": [], "by": sp, "date": date,
                                    "valid_from": vf})
        g["members"].append(m)
    return [dict(g, members=sorted(set(g["members"]))) for g in groups.values()]


def lineage_of(view: AsOf, pid: str) -> dict:
    es = edges(view, pid)
    return {"paper": pid, "as_of": view.T,
            "parents": [e for e in es if e["child"] == pid],
            "children": [e for e in es if e["parent"] == pid],
            "combines": combines(view, pid)}
