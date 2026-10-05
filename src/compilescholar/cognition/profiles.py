# -*- coding: utf-8 -*-
"""Paper profile at T (L4): what a work says about itself vs. how the field has described it up to T.

Pure function of AsOf(T) — no LLM, no raw text. Fields:
  self          the paper's own contributions / proposals / stated limitations (kind=self, about=paper)
  reception     aggregated other-statements about the paper with date <= T:
                  n_statements, n_citing (distinct citing papers), first/last citing date,
                  function_share  {basis, baseline, background, tool, data, metric, contrast} shares,
                  relation_share  {extends, improves, replaces, adapts, combines, uses, compares, criticizes, ...},
                  categories      most frequent categories the field puts it in (with counts),
                  limitations     limitations other papers state about it (deduplicated, with citing ids),
                  descriptions    up to K described statements, spread over time (newest last)
  shift         reception split at the midpoint of the citing period: early vs late function/relation shares
                (the P1 signal: lineage -> component/baseline); None when < MIN_SHIFT statements
  evidence      counts that tell a consumer how much this profile rests on (n_citing, months covered)
A paper with no reception yet (common for work under a year old) still gets its self part and evidence=0, so a
consumer can fall back on the self description and the paper's references (DESIGN §1 principle 5)."""
from __future__ import annotations

from collections import Counter

from .asof import AsOf

K_DESC = 8
MIN_SHIFT = 8
LINEAGE = {"extends", "improves", "replaces", "adapts", "combines"}


def _shares(vals: list[str]) -> dict[str, float]:
    c = Counter(v for v in vals if v)
    n = sum(c.values())
    return {k: round(v / n, 3) for k, v in c.most_common()} if n else {}


def _norm_cat(s: str) -> str:
    return " ".join((s or "").lower().replace("-", " ").split())


def profile(view: AsOf, obj: str) -> dict:
    selfs = view.statements(about=obj, kind="self") if obj.startswith("paper:") else []
    others = sorted(view.statements(about=obj, kind="other"), key=lambda s: s["date"])
    # one vote per (citing paper, sentence) for shares — a sentence that spawned a limitation statement and a
    # description statement must not count twice
    per_sentence = {}
    for s in others:
        per_sentence.setdefault((s["speaker"], s["meta"].get("sentence_id")), s)
    votes = sorted(per_sentence.values(), key=lambda s: s["date"])
    cats = Counter(_norm_cat(s["meta"].get("category")) for s in votes if s["meta"].get("category"))
    lims = {}
    for s in others:
        if s["facet"] == "limitation":
            lims.setdefault(s["text"].strip(), []).append(s["speaker"])
    described = [s for s in votes if s["meta"].get("described")]
    if len(described) > K_DESC:  # spread over time
        step = len(described) / K_DESC
        described = [described[int(i * step)] for i in range(K_DESC)]
    reception = {
        "n_statements": len(votes),
        "n_citing": len({s["speaker"] for s in votes}),
        "first": votes[0]["date"] if votes else None,
        "last": votes[-1]["date"] if votes else None,
        "function_share": _shares([s["function"] for s in votes]),
        "relation_share": _shares([s["role"] for s in votes]),
        "lineage_share": round(sum(s["role"] in LINEAGE for s in votes) / len(votes), 3) if votes else None,
        "categories": cats.most_common(5),
        "limitations": [{"text": t, "citing": sorted(set(c))[:5], "n": len(set(c))}
                        for t, c in sorted(lims.items(), key=lambda kv: -len(set(kv[1])))[:6]],
        "descriptions": [{"date": s["date"], "citing": s["speaker"], "text": s["text"], "function": s["function"],
                          "quote": s["quote"]} for s in described],
    }
    shift = None
    if len(votes) >= MIN_SHIFT:
        mid = len(votes) // 2
        early, late = votes[:mid], votes[mid:]
        shift = {"split_date": late[0]["date"],
                 "early": {"function": _shares([s["function"] for s in early]),
                           "lineage": round(sum(s["role"] in LINEAGE for s in early) / len(early), 3)},
                 "late": {"function": _shares([s["function"] for s in late]),
                          "lineage": round(sum(s["role"] in LINEAGE for s in late) / len(late), 3)}}
    months = {s["date"][:7] for s in votes}
    return {"object": obj, "as_of": view.T,
            "paper": view.paper(obj[6:]) if obj.startswith("paper:") else None,
            "self": [{"facet": s["facet"], "text": s["text"], "quote": s["quote"], "name": s["meta"].get("name")}
                     for s in selfs],
            "reception": reception, "shift": shift,
            "evidence": {"n_citing": reception["n_citing"], "months": len(months), "has_self": bool(selfs)}}
