"""Query understanding layer (P4-A A3, 2026-09-24).

The gap this closes (RETRIEVAL-MODULE-DESIGN.md §一.3): discover_by_query
and discover_tiered take a bare keyword string — a natural-language
question ("which methods improve LNP stability and how is it measured?")
retrieves poorly on every source. One small LLM call turns it into:

  - intent: find_papers | find_facts | survey_scan | trace_lineage
  - domain hint: cs | bio | physics | default (routes the tier preset)
  - 2-4 sub-queries, each a keyword group tuned for metadata search
    (title/abstract vocabulary, not question prose)
  - per-subquery source preference ordering (advisory; the circuit may
    override)

Design constraints:
  - ONE call, small output (~200 tokens) — the <2s budget for the fast
    path. Thinking disabled when the backend supports it.
  - Pluggable chat callable: the caller injects `chat(prompt) -> str`.
    Default: OpenAI-compatible client pointed at LOCAL_BASE_URL /
    LOCAL_API_KEY / LOCAL_MODEL env (our GPUStack box). CompileScholar
    injects kb_infra.call_local so calls land in the shared ledger and
    semaphore.
  - Deterministic fallback: on any LLM failure the original query passes
    through unchanged with intent=unknown — degradation never blocks
    retrieval.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable

UNDERSTAND_PROMPT = """You turn a researcher's question into library-search queries.

Question: {question}

Output strict JSON (no prose, no markdown):
{{"intent": "find_papers|find_facts|survey_scan|trace_lineage",
 "domain": "cs|bio|physics|default",
 "subqueries": [{{"q": "<keyword-group for title/abstract search: nouns and method names only, no question words>", "prefer": ["<source>", "<source>"]}}]}}

Rules:
- 2-4 subqueries; each covers one distinct angle of the question
  (method names, measured properties, material/system names).
- q must be keyword-style ("lipid nanoparticle stability assay"), never
  a sentence.
- ANGLE DIVERSITY IS MANDATORY: subquery 1 = the question's own
  distinctive/rare terms kept verbatim (specific technique names,
  materials, phenomena — e.g. a question mentioning "glycosylation"
  must have "glycosylation" in a subquery); the remaining subqueries
  paraphrase into broader vocabulary. Rare terms are high-precision
  retrieval handles — do NOT distill them away.
- prefer lists 1-2 keyword sources ordered by fit, from: arxiv, s2,
  openalex, crossref (keyword metadata channels; the semantic channel
  always leads and is NOT listed here).
- domain routes the keyword order: cs -> arxiv first; bio -> openalex
  first; physics -> arxiv/openalex."""


@dataclass(frozen=True)
class UnderstoodQuery:
    original: str
    intent: str = "unknown"
    domain: str = "default"
    subqueries: list[dict] = field(default_factory=list)
    llm_used: bool = False


_VALID_INTENTS = {"find_papers", "find_facts", "survey_scan", "trace_lineage"}
_VALID_DOMAINS = {"cs", "bio", "physics", "default"}
_VALID_SOURCES = {"arxiv", "s2", "openalex", "crossref"}


def _default_chat(prompt: str) -> str:
    """OpenAI-compatible chat against the local GPUStack box (env-driven)."""
    import urllib.error
    import urllib.request

    base = (os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1")
            .rstrip("/"))
    key = os.environ.get("LOCAL_API_KEY", "local")
    model = os.environ.get("LOCAL_MODEL", "Qwen3.8-27B")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        base + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read())
    return payload["choices"][0]["message"]["content"] or ""


def _parse(raw: str) -> UnderstoodQuery | None:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    subs = []
    for sq in obj.get("subqueries") or []:
        if not isinstance(sq, dict) or not (sq.get("q") or "").strip():
            continue
        prefer = [s for s in (sq.get("prefer") or [])
                  if str(s).lower() in _VALID_SOURCES]
        subs.append({"q": str(sq["q"]).strip()[:200],
                     "prefer": [p.lower() for p in prefer]})
    if not subs:
        return None
    return UnderstoodQuery(
        original="",
        intent=obj.get("intent") if obj.get("intent") in _VALID_INTENTS
        else "find_papers",
        domain=obj.get("domain") if obj.get("domain") in _VALID_DOMAINS
        else "default",
        subqueries=subs[:4],
        llm_used=True,
    )


def understand_query(question: str,
                     chat: Callable[[str], str] | None = None,
                     ) -> UnderstoodQuery:
    """Question -> structured sub-queries. Never raises: on any failure the
    original question survives as a single passthrough sub-query."""
    q = (question or "").strip()
    if not q:
        return UnderstoodQuery(original="")
    fn = chat or _default_chat
    try:
        raw = fn(UNDERSTAND_PROMPT.replace("{question}", q))
        parsed = _parse(raw)
        if parsed is not None:
            return UnderstoodQuery(original=q, intent=parsed.intent,
                                   domain=parsed.domain,
                                   subqueries=parsed.subqueries,
                                   llm_used=True)
    except Exception:
        pass
    return UnderstoodQuery(original=q, intent="unknown", domain="default",
                           subqueries=[{"q": q, "prefer": []}])
