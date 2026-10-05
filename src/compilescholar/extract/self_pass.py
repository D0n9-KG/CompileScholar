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
PROMPT = """Below are the title, abstract and (when available) the introduction of ONE research paper. Extract what the
paper says about ITS OWN work. Every item must carry "quote": one sentence copied verbatim from the text below.

Title: {title}
Text:
{text}

Return JSON only:
{{"contributions": [{{"text": "<one sentence: something this paper does or claims>", "quote": "..."}}],
  "proposes": [{{"name": "<name the paper gives it; acronyms as written>", "aliases": ["..."],
                 "artefact": "method|model|dataset|benchmark|metric|system|framework", "quote": "..."}}],
  "findings": [{{"text": "<a result or observation the paper reports, in words>", "quote": "..."}}],
  "limitations": [{{"text": "<a weakness, assumption or open issue the paper states about its own work>", "quote": "..."}}],
  "setting": {{"tasks": ["..."], "datasets": ["..."], "metrics": ["..."]}}}}
Rules: list in "proposes" only things this paper introduces as new, not things it uses or builds on. 1-5 contributions,
0-5 findings. Empty lists are fine. No outside knowledge."""
PROMPT_SHA = hashlib.sha256(PROMPT.encode()).hexdigest()[:16]

# T2 (deep) second call, on the method and experiment sections
DEEP_PROMPT = """Below are the method and experiment sections of ONE research paper titled "{title}". Extract how its
method works and what its experiments show. Every item must carry "quote": one sentence copied verbatim from the text.

Text:
{text}

Return JSON only:
{{"method": [{{"text": "<one sentence: a key component, step or design choice of the paper's method>", "quote": "..."}}],
  "findings": [{{"text": "<a result, comparison or ablation outcome the paper reports, in words>", "quote": "..."}}],
  "limitations": [{{"text": "<a weakness or failure case the paper reports>", "quote": "..."}}],
  "setting": {{"tasks": ["..."], "datasets": ["..."], "metrics": ["..."], "baselines": ["..."]}}}}
Rules: 2-8 method items, 0-8 findings. Only what this text says. No outside knowledge."""
DEEP_SHA = hashlib.sha256(DEEP_PROMPT.encode()).hexdigest()[:16]


def _n(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def in_text(quote: str, text: str) -> bool:
    q = _n(quote).lower()
    return len(q) >= 20 and q in _n(text).lower()


def input_text(title: str, abstract: str, body: str | None) -> str:
    """Abstract + (T2) the abstract-and-introduction text from documents.units, capped at INTRO_CHARS."""
    parts = [f"Abstract: {abstract.strip()}"] if abstract else []
    if body:
        parts.append(body[:INTRO_CHARS])
    return "\n\n".join(parts)


def _setting(s: dict, text: str, keys) -> dict:
    """Keep only setting names that literally occur in the input text."""
    s = s if isinstance(s, dict) else {}
    return {k: [_n(str(x)) for x in s.get(k) or [] if _n(str(x)) and _n(str(x)).lower() in text.lower()] for k in keys}


def run(arxiv_id: str, date: str, title: str, abstract: str, body: str | None = None,
        chat=call_local, method_text: str | None = None) -> tuple[list[Statement], dict]:
    """T1 (abstract only): body=None, method_text=None. T2 (deep): body = abstract-and-introduction text,
    method_text = method-and-experiments text (documents.units); a second call extracts method components and
    experimental findings from it. Every quote is checked against the text of the call that produced it."""
    text = input_text(title, abstract, body)
    st = {"items": 0, "kept": 0, "quote_rejected": 0}
    if len(text) < 200:
        return [], st
    me = f"paper:{arxiv_id}"
    out = []

    def keep(stmt: Statement, quote: str, src: str):
        st["items"] += 1
        if not in_text(quote, src):
            st["quote_rejected"] += 1
            return
        st["kept"] += 1
        out.append(stmt)

    def items(obj, key, role, facet, src, tag):
        for c in obj.get(key) or []:
            if isinstance(c, dict) and c.get("text") and c.get("quote"):
                keep(Statement(arxiv_id, date, "self", me, role, facet, _n(c["text"]), _n(c["quote"]),
                               meta={"pass": tag}), c["quote"], src)

    raw = chat(PROMPT.format(title=title[:300], text=text), model=MODEL, max_tokens=3000, temperature=0.0,
               enable_thinking=False)
    obj = parse_json_response(raw or "")
    if not isinstance(obj, dict):
        return [], {**st, "parse_failed": 1}
    items(obj, "contributions", "proposes", "contribution", text, "self")
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
                                                  "artefact": p.get("artefact"), "generic": v["generic"],
                                                  "pass": "self"}))
    items(obj, "findings", "describes", "result", text, "self")
    items(obj, "limitations", "describes", "limitation", text, "self")
    lists = _setting(obj.get("setting"), text, ("tasks", "datasets", "metrics"))

    if method_text and len(method_text) >= 300:
        raw = chat(DEEP_PROMPT.format(title=title[:300], text=method_text), model=MODEL, max_tokens=3500,
                   temperature=0.0, enable_thinking=False)
        deep = parse_json_response(raw or "")
        if isinstance(deep, dict):
            items(deep, "method", "describes", "method", method_text, "deep")
            items(deep, "findings", "describes", "result", method_text, "deep")
            items(deep, "limitations", "describes", "limitation", method_text, "deep")
            d = _setting(deep.get("setting"), method_text, ("tasks", "datasets", "metrics", "baselines"))
            for k, v in d.items():
                lists[k] = list(dict.fromkeys((lists.get(k) or []) + v))
        else:
            st["deep_parse_failed"] = 1
    if any(lists.values()):
        out.append(Statement(arxiv_id, date, "self", me, "describes", "setting",
                             "; ".join(f"{k}: {', '.join(v)}" for k, v in lists.items() if v),
                             (abstract or text)[:300].strip(), meta={**lists, "pass": "self"}))
    return out, st
