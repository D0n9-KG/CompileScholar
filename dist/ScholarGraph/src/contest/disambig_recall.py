# -*- coding: utf-8 -*-
"""Term-of-art field disambiguation expansion (GOAL 2026-08-29 深夜).

Diagnosed failure (AutoScholar q1): queries are context-free paraphrases of
citation context — "hybrid architectures in reconstruction-based techniques"
MEANS anomaly detection to the citing author (reconstruction-based AD is a
standard family), but retrieves image-reconstruction papers literally. The
gold's field is invisible from the query text; both S2 AND Google verbatim
land in the wrong world (measured).

Fix: one LLM call enumerates the fields where the query's core phrase is a
term-of-art, each with the field's own short search vocabulary. Those queries
join the recall set — the pool gains one facet per plausible field reading.

BLIND by construction: the disambiguation prompt sees the user query only.
Field knowledge is model prior, same as any query rewrite. Disk-cached.
"""
import json, os

from granular_agent.llm_client import parse_json_response
from contest.cost_ledger import ledger

_CACHE_DIR = os.path.join(".research_tmp", "contest_survey", "_intent_cache")


def _qkey(query: str) -> str:
    import hashlib
    return hashlib.md5(query.encode()).hexdigest()[:12]


def disambiguation_queries(query: str, max_fields: int = 3,
                            max_queries: int = 6) -> list[str]:
    """Field-disambiguated short queries for the recall set (cached)."""
    key = _qkey(query)
    cpath = os.path.join(_CACHE_DIR, f"disambig_{key}.json")
    if os.path.exists(cpath):
        try:
            return json.loads(open(cpath, encoding="utf-8").read())
        except Exception:
            pass
    p = ("An academic search request follows. Some phrase in it may be a "
         "TERM-OF-ART: words that carry a specific technical meaning inside "
         "certain research fields, different from their everyday reading. A "
         "searcher using only the everyday reading retrieves the wrong "
         "field's papers.\n\n"
         f"Request: «{query}»\n\n"
         "1. Identify the phrase(s) in the request that could be field "
         "jargon.\n"
         "2. For each, list up to 3 DISTINCT fields where it is a "
         "term-of-art, PREFERRING fields common in ML/CS survey papers "
         "(the request likely paraphrases a citation context from one).\n"
         "3. For each field, write 1-2 SHORT keyword queries (2-3 content "
         "tokens, AND-matched) in THAT field's own vocabulary — the words "
         "its authors put in titles.\n"
         'Output JSON: {"queries": ["...", ...]} (max 8 queries total)')
    raw = ledger.llm("Qwen3.8-Max", p, max_tokens=500, enable_thinking=False)
    if raw is None:
        return []                  # transient failure: do NOT cache empty
    obj = parse_json_response(raw) or {}
    qs = [q.strip() for q in (obj.get("queries") or []) if q and q.strip()]
    qs = qs[:max_queries]
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(qs, f, ensure_ascii=False)
    except Exception:
        pass
    return qs
