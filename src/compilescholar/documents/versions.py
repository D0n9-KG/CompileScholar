# -*- coding: utf-8 -*-
"""Version awareness (INTEGRATED-SYSTEM-1005 §2.3): v1 is read in full; a later version contributes only what v1 does
not have, dated by that version's date. Intermediate versions are skipped (their additions land on the latest
version's date: later, never earlier, so nothing leaks).

delta(v1_doc, latest_doc) -> {"sentences": [...], "entries": [...], "cites": [...], "revised": [...]}
  sentences  latest-version sentences with no counterpart in v1: best token-set match against every v1 sentence
             < NEW_BELOW (rapidfuzz, on core.ids.norm_title). Exact matching is not enough: measured on 74 two-version
             cs papers, 48 % (median) of the latest version's sentences are not verbatim in v1, but only 13 % have no
             v1 sentence scoring >= 85 — re-flowed lines, re-segmentation and small edits make up the rest.
  revised    sentences whose best match is in [NEW_BELOW, SAME_ABOVE): edited sentences; whether one states a new
             claim (a new number, a changed limitation) is for the LLM in the extract stage, not decided here
  entries    bibliography entries of the latest version with no v1 entry of the same title_key (new references)
  cites      the latest version's citation pairs whose sentence is new or revised, or whose entry is new
Deterministic; the units of the latest version are not diffed (the extract stage reads the delta's sentences)."""
from __future__ import annotations

from rapidfuzz import fuzz, process

from ..core import ids

NEW_BELOW = 85
SAME_ABOVE = 98


def _entry_key(e: dict) -> str:
    return ids.title_key(e.get("title") or "") or ids.title_key((e.get("raw") or "")[:120])


def delta(v1: dict, latest: dict) -> dict:
    a = [ids.norm_title(s["text"]) for s in v1["sentences"]]
    sa = set(a)
    new, revised = [], []
    for s in latest["sentences"]:
        t = ids.norm_title(s["text"])
        if not t or t in sa:
            continue
        m = process.extractOne(t, a, scorer=fuzz.token_set_ratio) if a else None
        score = m[1] if m else 0
        if score < NEW_BELOW:
            new.append({**s, "match": round(score)})
        elif score < SAME_ABOVE:
            revised.append({**s, "match": round(score), "v1_text": v1["sentences"][m[2]]["text"]})
    old_keys = {_entry_key(e) for e in v1["entries"].values()}
    new_entries = {k: e for k, e in latest["entries"].items() if _entry_key(e) not in old_keys}
    touched = {s["sid"] for s in new + revised}
    cites = [c for c in latest["cites"] if c[0] in touched or c[1] in new_entries]
    return {"sentences": new, "revised": revised, "entries": new_entries, "cites": cites}
