# -*- coding: utf-8 -*-
"""Paper profile at T (phase D): what a work says about itself vs. how the field has treated it up to T.

§9.2 splits the sources: the heavy field-level cognition is READ FROM THE MATERIALISED TABLES (reception
daily counts, canonical categories, approved shift events, lineage edges, family facts), while the bounded
per-paper part — the self description and the selection of representative other-statements — stays computed
online from the extract store through AsOf. One vote per (citing paper, sentence): a sentence that produced
both a limitation and a categorization statement must not count twice in the shares.

A paper with no reception yet (common for work under a year old) still gets its self part and zeros, so a
consumer can fall back on the self description and the paper's references."""
from __future__ import annotations

from collections import Counter

from . import shifts as SH
from .asof import AsOf

K_DESC = 8
LINEAGE = {"extends", "improves", "replaces", "adapts", "combines"}


def _shares(vals: list) -> dict:
    c = Counter(v for v in vals if v)
    n = sum(c.values())
    return {k: round(v / n, 3) for k, v in c.most_common()} if n else {}


def reception_counts(view: AsOf, pid: str) -> dict:
    """The materialised daily citation counts (reception_daily, full counts — never truncated by the other
    pass's sampled subset): total, first/last day, and the latest 24 months as a series."""
    if view.cog is None:
        return {"n_cites": 0, "first": None, "last": None, "monthly": []}
    rows = view.cog.execute("SELECT day, n FROM reception_daily WHERE cited=? AND day<=? ORDER BY day",
                            (pid, view.T)).fetchall()
    monthly: dict = {}
    for d, n in rows:
        monthly[d[:7]] = monthly.get(d[:7], 0) + n
    months = sorted(monthly)[-24:]
    return {"n_cites": sum(n for _, n in rows), "first": rows[0][0] if rows else None,
            "last": rows[-1][0] if rows else None,
            "monthly": [{"month": m, "n": monthly[m]} for m in months]}


def profile(view: AsOf, pid: str) -> dict:
    paper = view.paper(pid)
    selfs = view.statements(about=pid, kind="self")
    others = sorted(view.statements(about=pid, kind="other"), key=lambda s: s["date"])
    per_sentence: dict = {}
    for s in others:
        per_sentence.setdefault((s["speaker"], (s.get("loc") or {}).get("sent_id")), s)
    votes = sorted(per_sentence.values(), key=lambda s: s["date"])
    canon = dict(view.cog.execute("SELECT phrase, canonical FROM category_canon")) if view.cog is not None else {}
    cats = Counter()
    for s in votes:
        c = (s.get("meta") or {}).get("category")
        if c and c.strip():
            cats[canon.get(c.strip(), c.strip().lower())] += 1
    lims: dict = {}
    for s in others:
        if s["facet"] == "limitation" and s["text"].strip():
            lims.setdefault(s["text"].strip(), []).append(s["speaker"])
    described = [s for s in votes if s["text"] and not s["text"].startswith("(")]
    if len(described) > K_DESC:            # spread over the reception period, oldest first
        step = len(described) / K_DESC
        described = [described[int(i * step)] for i in range(K_DESC)]
    rec = reception_counts(view, pid)
    reception = {
        **rec,
        "n_statements": len(votes),
        "n_citing": len({s["speaker"] for s in votes}),
        "function_share": _shares([s["function"] for s in votes]),
        "relation_share": _shares([s["role"] for s in votes]),
        "lineage_share": round(sum(s["role"] in LINEAGE for s in votes) / len(votes), 3) if votes else None,
        "categories": cats.most_common(5),
        "limitations": [{"text": t, "citing": sorted(set(c))[:5], "n": len(set(c))}
                        for t, c in sorted(lims.items(), key=lambda kv: -len(set(kv[1])))[:6]],
        "descriptions": [{"date": s["date"], "citing": s["speaker"], "text": s["text"],
                          "function": s["function"], "quote": s["quote"]} for s in described],
    }
    return {"object": pid, "as_of": view.T, "paper": paper,
            "self": [{"facet": s["facet"], "text": s["text"], "quote": s["quote"], "date": s["date"],
                      "epistemic": s["epistemic"], "condition": s["condition"],
                      "name": (s.get("meta") or {}).get("name")} for s in selfs],
            "reception": reception,
            "shifts": SH.events(view, pid),
            "evidence": {"n_cites": rec["n_cites"], "n_citing": reception["n_citing"],
                         "months": len({s["date"][:7] for s in votes}), "has_self": bool(selfs)}}
