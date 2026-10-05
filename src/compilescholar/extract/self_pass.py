# -*- coding: utf-8 -*-
"""Self pass (L2a): what a paper says about its own work, from its abstract and introduction.

One LLM call per paper. Output, all with a verbatim quote that must occur in the input text (whitespace-normalized):
  contributions  what the paper does (role=proposes, facet=contribution)
  proposes       named artefacts the paper introduces: method / model / dataset / benchmark / metric / system
                 (role=proposes, facet=contribution, meta={name, aliases, artefact}); the quote must read as a proposal
                 (compile.skeleton.proposes.validate rules: proposal wording, not a reliance statement)
  limitations    weaknesses / open issues the paper states about itself (role=describes, facet=limitation)
  setting        tasks / datasets / metrics the paper works with (role=describes, facet=setting, meta lists)
The former abstract-only coarse extractor (compile.state.coarse) and the proposer extractor (compile.skeleton.proposes)
are subsumed here; their validation rules are reused, not duplicated."""
from __future__ import annotations

import hashlib
import re

from ..compile.skeleton import proposes as P
from ..llm.client import call_local
from ..llm.jsonparse import parse_json_response
from .schema import Statement

MODEL = "Qwen3.8-27B"
INTRO_CHARS = 7000
PROMPT = """Below are the title, abstract and the beginning of the introduction of ONE research paper. Extract what the
paper says about ITS OWN work. Every item must carry "quote": one sentence copied verbatim from the text below.

Title: {title}
Text:
{text}

Return JSON only:
{{"contributions": [{{"text": "<one sentence: something this paper does or claims>", "quote": "..."}}],
  "proposes": [{{"name": "<name the paper gives it; acronyms as written>", "aliases": ["..."],
                 "artefact": "method|model|dataset|benchmark|metric|system|framework", "quote": "..."}}],
  "limitations": [{{"text": "<a weakness, assumption or open issue the paper states about its own work>", "quote": "..."}}],
  "setting": {{"tasks": ["..."], "datasets": ["..."], "metrics": ["..."]}}}}
Rules: list in "proposes" only things this paper introduces as new, not things it uses or builds on. 1-5 contributions.
Empty lists are fine. No outside knowledge."""
PROMPT_SHA = hashlib.sha256(PROMPT.encode()).hexdigest()[:16]


def _n(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def in_text(quote: str, text: str) -> bool:
    q = _n(quote).lower()
    return len(q) >= 20 and q in _n(text).lower()


def input_text(title: str, abstract: str, body: str | None) -> str:
    """Abstract + the first INTRO_CHARS of the body (full text when we have it)."""
    parts = [f"Abstract: {abstract.strip()}"] if abstract else []
    if body:
        parts.append(body[:INTRO_CHARS])
    return "\n\n".join(parts)


def run(arxiv_id: str, date: str, title: str, abstract: str, body: str | None = None,
        chat=call_local) -> tuple[list[Statement], dict]:
    text = input_text(title, abstract, body)
    st = {"items": 0, "kept": 0, "quote_rejected": 0}
    if len(text) < 200:
        return [], st
    raw = chat(PROMPT.format(title=title[:300], text=text), model=MODEL, max_tokens=2500, temperature=0.0,
               enable_thinking=False)
    obj = parse_json_response(raw or "")
    if not isinstance(obj, dict):
        return [], {**st, "parse_failed": 1}
    me = f"paper:{arxiv_id}"
    out = []

    def keep(stmt: Statement, quote: str):
        st["items"] += 1
        if not in_text(quote, text):
            st["quote_rejected"] += 1
            return
        st["kept"] += 1
        out.append(stmt)

    for c in obj.get("contributions") or []:
        if isinstance(c, dict) and c.get("text") and c.get("quote"):
            keep(Statement(arxiv_id, date, "self", me, "proposes", "contribution", _n(c["text"]), _n(c["quote"])),
                 c["quote"])
    for p in obj.get("proposes") or []:
        if not isinstance(p, dict):
            continue
        v = P.validate({"name": p.get("name"), "aliases": p.get("aliases"), "evidence": p.get("quote")}, text)
        st["items"] += 1
        if not v:
            st["quote_rejected"] += 1
            continue
        st["kept"] += 1
        out.append(Statement(arxiv_id, date, "self", me, "proposes", "contribution", f"proposes {v['method']}",
                             v["evidence"], meta={"name": v["method"], "aliases": v["aliases"],
                                                  "artefact": p.get("artefact"), "generic": v["generic"]}))
    for l in obj.get("limitations") or []:
        if isinstance(l, dict) and l.get("text") and l.get("quote"):
            keep(Statement(arxiv_id, date, "self", me, "describes", "limitation", _n(l["text"]), _n(l["quote"])),
                 l["quote"])
    s = obj.get("setting") if isinstance(obj.get("setting"), dict) else {}
    lists = {k: [_n(str(x)) for x in s.get(k) or [] if _n(str(x)) and _n(str(x)).lower() in text.lower()]
             for k in ("tasks", "datasets", "metrics")}
    if any(lists.values()):
        out.append(Statement(arxiv_id, date, "self", me, "describes", "setting",
                             "; ".join(f"{k}: {', '.join(v)}" for k, v in lists.items() if v),
                             (abstract or text)[:300].strip(), meta=lists))
    return out, st
