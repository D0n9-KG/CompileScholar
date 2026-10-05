# -*- coding: utf-8 -*-
"""Lineage edges at T (L4, N5): paper -> paper, typed with the one relation vocabulary, every edge carrying who
asserted it, when, and the sentence.

Two kinds of assertion, kept apart (never merged into one count):
  self   the citing paper says its own work extends / improves / replaces / adapts / combines the cited work
         (other-pass statement with role in LINEAGE about the cited paper, speaker = citing paper)
  third  a sentence in a third paper says cited work A extends / improves / ... work B (other-pass meta.builds_on,
         B resolved through Identity, or B is another work cited in the same sentence whose name matches)
Edge = (child, parent, relation): child is the newer work. Time consistency: an edge whose child's v1_date precedes
its parent's is dropped and counted (cannot extend something that did not exist yet). n-ary "combines" keeps every
parent cited in the same sentence as one hyperedge record (group id = sentence id).
Pure function of AsOf(T)."""
from __future__ import annotations

from collections import defaultdict

from ..extract.schema import LINEAGE
from .asof import AsOf
from .identity import Identity, norm_name


def edges(view: AsOf, ident: Identity | None = None) -> dict:
    ident = ident or Identity(view)
    out: dict[tuple[str, str, str], dict] = {}
    dropped = 0
    combines: dict[tuple[str, int], set[str]] = defaultdict(set)

    def add(child, parent, rel, kind, s):
        nonlocal dropped
        if child == parent or not child.startswith("paper:") or not parent.startswith("paper:"):
            return
        pc, pp = view.paper(child[6:]), view.paper(parent[6:])
        if not pc or not pp:
            return
        if pc["v1_date"] < pp["v1_date"]:
            dropped += 1
            return
        e = out.setdefault((child, parent, rel), {"child": child, "parent": parent, "relation": rel,
                                                 "self": [], "third": []})
        e[kind].append({"by": f"paper:{s['speaker']}", "date": s["date"], "quote": s["quote"]})

    for s in view.statements(kind="other"):
        me = f"paper:{s['speaker']}"
        if s["role"] in LINEAGE:
            add(me, s["about"], s["role"], "self", s)
            if s["role"] == "combines":
                combines[(me, s["meta"].get("sentence_id"))].add(s["about"])
        bo, rel = s["meta"].get("builds_on"), s["meta"].get("builds_on_relation")
        if bo and rel in LINEAGE:
            parent = ident.resolve(bo)
            if parent is None:  # fall back: a co-cited work in the same sentence whose own name matches
                for g in s["group"]:
                    if g != s["about"] and norm_name(bo) in {norm_name(a) for a in ident.aliases(g)}:
                        parent = g
                        break
            if parent:
                add(s["about"], parent, rel, "third", s)
    hyper = [{"child": c, "parents": sorted(ps), "sentence_id": sid} for (c, sid), ps in combines.items() if len(ps) >= 2]
    return {"edges": list(out.values()), "combines": hyper, "dropped_time_inconsistent": dropped}


def lineage_of(view: AsOf, paper: str, direction: str = "both", ident: Identity | None = None) -> dict:
    E = edges(view, ident)
    up = [e for e in E["edges"] if e["child"] == paper]
    down = [e for e in E["edges"] if e["parent"] == paper]
    return {"paper": paper, "as_of": view.T,
            "parents": up if direction in ("both", "up") else [],
            "children": down if direction in ("both", "down") else [],
            "combines": [h for h in E["combines"] if h["child"] == paper or paper in h["parents"]]}
