# -*- coding: utf-8 -*-
"""Other pass (L2b): what a citing paper says about each work it cites, and how its own work relates to it.

Input: citation sentences of ONE citing paper from the citations stage (sentence text + cited entry title), batched
up to BATCH pairs per LLM call. One output per (sentence, cited entry) pair:
  function   citation function (schema.FUNCTIONS)
  relation   the citing paper's work -> cited work (schema.OTHER_RELATIONS)
  about      one sentence: what the sentence says the cited work is / does (None = nothing specific)
  facet      what kind of content `about` carries (schema.FACETS)
  category   the class / family the sentence puts the cited work in (None if none)
  limitation the cited work's weakness the sentence states (None if none)
Validation is deterministic: closed vocabularies; `about`, `category`, `limitation` must be lexically supported by
the sentence (>= SUPPORT of their content words occur in it); otherwise the field is dropped (the pair is kept with
its function/relation). The quote of every statement is the citation sentence itself, verbatim from L1."""
from __future__ import annotations

import hashlib
import re

from ..llm.client import call_local
from ..llm.jsonparse import parse_json_response
from .schema import FACETS, FUNCTIONS, OTHER_RELATIONS, Statement

BATCH = 24
SUPPORT = 0.6
MODEL = "Qwen3.8-27B"
PROMPT = """You read sentences from ONE research paper (the citing paper, first posted {date}) in which it cites other
works. For each numbered (sentence, cited work) pair below, report what THIS sentence says about the cited work and how
the citing paper's own work relates to it. Use only what the sentence says; do not use outside knowledge.

{pairs}

For every pair return an object with:
- "id": the pair number
- "function": the role of the citation: background | basis | baseline | contrast | data | tool | metric
- "relation": how the citing paper's own work relates to the cited work: extends | improves | replaces | adapts |
  combines | uses | compares | background | criticizes
- "about": one short sentence on what the sentence says the cited work is or does, in the sentence's own words; null if
  the sentence says nothing specific about it (e.g. a bare list of citations)
- "facet": what "about" describes: contribution | method | result | limitation | setting | categorization
- "category": the class or family of methods the sentence puts the cited work in, in the sentence's words (e.g.
  "parameter-efficient fine-tuning methods"); null if none
- "limitation": the weakness or shortcoming of the cited work that the sentence states; null if none

Return JSON only: {{"pairs": [{{"id": 1, "function": "...", "relation": "...", "about": "...", "facet": "...",
"category": null, "limitation": null}}]}}"""

PROMPT_SHA = hashlib.sha256(PROMPT.encode()).hexdigest()[:16]
STOP = set("a an the of in on for to and or with by from as is are was were be been this that these those it its "
           "their they we our which such via using use used based into than also can may".split())


def _words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP and len(w) > 1]


def supported(field: str | None, sentence: str) -> bool:
    w = _words(field or "")
    if not w:
        return False
    sent = set(_words(sentence))
    return sum(x in sent for x in w) / len(w) >= SUPPORT


def _pairs_block(items: list[dict]) -> str:
    return "\n".join(f'[{i + 1}] cited work: "{(it["title"] or it["raw"][:160])[:200]}"\n    sentence: "{it["sentence"][:900]}"'
                     for i, it in enumerate(items))


def run_batch(citing: str, date: str, items: list[dict], chat=call_local) -> tuple[list[Statement], dict]:
    """items: [{sentence_id, sentence, cited, title, raw, group}] -> statements (one per pair that parsed) + stats."""
    raw = chat(PROMPT.format(date=date, pairs=_pairs_block(items)), model=MODEL, max_tokens=180 * len(items) + 200,
               temperature=0.0, enable_thinking=False)
    obj = parse_json_response(raw or "")
    got = {}
    for o in (obj or {}).get("pairs") or [] if isinstance(obj, dict) else []:
        if isinstance(o, dict) and str(o.get("id", "")).isdigit():
            got[int(o["id"])] = o
    out, st = [], {"pairs": len(items), "parsed": 0, "dropped_about": 0, "bad_vocab": 0}
    for i, it in enumerate(items):
        o = got.get(i + 1)
        if not o:
            continue
        rel, fn, facet = o.get("relation"), o.get("function"), o.get("facet")
        if rel not in OTHER_RELATIONS or fn not in FUNCTIONS:
            st["bad_vocab"] += 1
            continue
        about = o.get("about") if isinstance(o.get("about"), str) else None
        if about and not supported(about, it["sentence"]):
            about, st["dropped_about"] = None, st["dropped_about"] + 1
        cat = o.get("category") if isinstance(o.get("category"), str) and supported(o.get("category"), it["sentence"]) else None
        lim = o.get("limitation") if isinstance(o.get("limitation"), str) and supported(o.get("limitation"), it["sentence"]) else None
        facet = facet if facet in FACETS else ("limitation" if lim else "categorization" if cat else "contribution")
        if lim and facet != "limitation":
            out.append(Statement(speaker=citing, date=date, kind="other", about=it["cited"], role=rel,
                                 facet="limitation", text=lim, quote=it["sentence"], group=tuple(it["group"]),
                                 function=fn, meta={"sentence_id": it["sentence_id"]}))
        out.append(Statement(speaker=citing, date=date, kind="other", about=it["cited"], role=rel, facet=facet,
                             text=about or lim or cat or "(cited without description)", quote=it["sentence"],
                             group=tuple(it["group"]), function=fn,
                             meta={"sentence_id": it["sentence_id"], "category": cat, "limitation": lim,
                                   "described": bool(about)}))
        st["parsed"] += 1
    return out, st
