# -*- coding: utf-8 -*-
"""W2 P0-3: which method does each (non-survey) paper propose?

Self-report only, from the paper's own abstract: one LLM call per paper returns the methods the paper itself proposes
(name, aliases, a verbatim sentence from the abstract that says so). Validation is deterministic:
  - the evidence sentence must occur in the abstract (whitespace-normalized);
  - the method name (or an alias) must occur in that sentence;
  - names on the generic list (single common nouns / umbrella terms) are kept but flagged generic=True and never used
    to ground lineage endpoints.
Methods the paper merely uses or builds on are excluded by the prompt and by the evidence check (the sentence must be a
proposal statement: "we propose / introduce / present / develop / design ..." or "<Name>, a new ...").

Output rows: {"paper_id", "method", "aliases", "evidence", "generic"}.
"""
from __future__ import annotations

import json
import re

from ...llm.client import call_local
from ...llm.jsonparse import parse_json_response

PROMPT = """Below is the abstract of ONE research paper. List the methods, models, algorithms, frameworks, datasets or
benchmarks that THIS paper itself proposes (introduces as new). Do NOT list things the paper only uses, compares
against, or builds on.

For each item give:
- "name": the name the paper gives it (keep acronyms as written, e.g. "LoRA"); if the paper gives no name, a short
  descriptive name taken from the abstract's own words
- "aliases": other forms used in the abstract (expanded acronym, etc.), possibly empty
- "evidence": ONE sentence copied verbatim from the abstract that states the paper proposes it

If the paper proposes nothing new (e.g. it is a study, survey or analysis), return an empty list.

Title: {title}
Abstract: {abstract}

Return JSON only: {{"proposes": [{{"name": "...", "aliases": ["..."], "evidence": "..."}}]}}"""

PROPOSAL = re.compile(r"\b(we|this (?:paper|work|article)|our|the authors?)\b[^.]{0,80}?\b(propos|introduc|present|develop|"
                      r"design|describ|creat|releas|construct|formulat)\w*|\b(a|an) (?:novel|new)\b|\bnamed\b|\bcalled\b",
                      re.I)
# reliance statements: the sentence is about something the paper uses / extends, not proposes
RELIANCE = re.compile(r"\b(build[s]?|built|based|relies|rely|relying|extend[s]?|extending|follow[s]?|adopt[s]?|"
                      r"leverag\w+|us(?:e|es|ing))\s+(?:up)?on\b|\b(us(?:e|es|ing)|employ\w*|adopt\w*)\s+(?:the\s)?[A-Z]",
                      re.I)
GENERIC = {"transformer", "transformers", "gnn", "cnn", "rnn", "lstm", "gan", "vae", "llm", "llms", "bert", "model",
           "framework", "method", "approach", "algorithm", "network", "dataset", "benchmark", "system", "pipeline",
           "attention", "diffusion model", "language model", "neural network", "deep learning", "reinforcement learning"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def validate(item: dict, abstract: str) -> dict | None:
    name = _norm(str(item.get("name") or ""))
    ev = _norm(str(item.get("evidence") or ""))
    aliases = [_norm(str(a)) for a in item.get("aliases") or [] if _norm(str(a))]
    if not name or not ev or len(ev) < 20:
        return None
    ab = _norm(abstract)
    if ev.lower() not in ab.lower():
        return None
    if not PROPOSAL.search(ev):
        return None
    if RELIANCE.search(ev) and not re.search(r"\b(propos|introduc|present)\w*", ev, re.I):
        return None
    return {"method": name, "aliases": aliases, "evidence": ev, "generic": name.lower() in GENERIC}


def extract(paper_id: str, title: str, abstract: str, chat=call_local, model: str = "Qwen3.8-27B") -> list[dict]:
    if not (abstract or "").strip():
        return []
    raw = chat(PROMPT.format(title=title, abstract=abstract[:4000]), model=model, max_tokens=1200, temperature=0.0,
               enable_thinking=False)
    obj = parse_json_response(raw or "")
    items = (obj or {}).get("proposes") if isinstance(obj, dict) else None
    out = []
    for it in items or []:
        if isinstance(it, dict):
            v = validate(it, abstract)
            if v:
                out.append({"paper_id": paper_id, **v})
    return out
