# -*- coding: utf-8 -*-
"""W1-6: citation-format-aligned variants of existing CS2 judge inputs (no regeneration; only snippets change).

CS2 citation facets check each claim against the snippets attached to its citations, and our snippets are much longer
than the harness's (median ~1,140 vs ~240 characters). These variants separate "grounded citations" from "longer
snippets" (pre-registered reading rule: PREREG §4):

  trim   : every snippet replaced by the 1–2 sentences most lexically similar to the sentences citing it, capped at
           `cap` characters (default 240 = harness median). Deterministic (token-overlap ranking, no LLM).
  title  : snippets removed; citations keep only their titles (the official scorer's title-only tier).
  expand : (for arms with short snippets) each snippet replaced by the full retrieved abstract when one is known for
           that citation (`abstracts` map: citation title -> text); unknown ones unchanged.

Text, citation ids and titles are never modified.
"""
from __future__ import annotations

import copy
import re

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")
_TOK = re.compile(r"[a-z0-9]+")
_STOP = frozenset("the a an of and or to in for on with by is are was were be as that this these those from at its "
                  "their it into via using we our can".split())


def _toks(s: str) -> set[str]:
    return {t for t in _TOK.findall((s or "").lower()) if t not in _STOP and len(t) > 2}


def citing_sentences(text: str, cid: str) -> list[str]:
    return [s for s in _SENT.split(text or "") if cid in s]


def trim_snippet(snippet: str, context: str, cap: int = 240) -> str:
    """The 1-2 snippet sentences with the highest token overlap with the citing context, in original order, cut at
    `cap` characters on a word boundary."""
    sents = [s.strip() for s in _SENT.split(snippet or "") if s.strip()]
    if not sents:
        return ""
    ctx = _toks(context)
    ranked = sorted(range(len(sents)), key=lambda i: (-len(_toks(sents[i]) & ctx), i))
    keep = sorted(ranked[:2])
    out = " ".join(sents[i] for i in keep)
    if len(out) <= cap:
        return out
    cut = out[:cap]
    return cut[: cut.rfind(" ")] if " " in cut else cut


def variant(rows: list[dict], kind: str, cap: int = 240, abstracts: dict[str, str] | None = None) -> list[dict]:
    out = copy.deepcopy(rows)
    for r in out:
        for sec in r.get("sections") or []:
            for c in sec.get("citations") or []:
                if kind == "title":
                    c["snippets"] = []
                elif kind == "trim":
                    ctx = " ".join(citing_sentences(sec.get("text") or "", c.get("id") or ""))
                    c["snippets"] = [t for t in (trim_snippet(s, ctx, cap) for s in c.get("snippets") or []) if t]
                elif kind == "expand":
                    ab = (abstracts or {}).get((c.get("title") or "").strip().lower())
                    if ab:
                        c["snippets"] = [ab]
                else:
                    raise ValueError(kind)
    return out


def snippet_stats(rows: list[dict]) -> dict:
    import statistics as st
    lens = [sum(len(s) for s in c.get("snippets") or []) for r in rows for sec in r.get("sections") or []
            for c in sec.get("citations") or []]
    return {"citations": len(lens), "median_chars": st.median(lens) if lens else 0,
            "with_snippet": sum(1 for x in lens if x > 0)}
