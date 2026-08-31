# -*- coding: utf-8 -*-
"""Schema intent recall — mechanisms ① (intent parse) and ③ (slot-fill
recall) of the hypergraph precision-recall direction (GOAL 2026-08-29).

① intent parse: user query -> frozen-schema patterns + slot fillers
   (LLM schema-in-context over the 71 tbox patterns; deterministic disk cache).
③ slot filling: matched patterns -> hyperedges in the domain graph whose
   pattern_type matches; the hyperedge's role concepts + provenance papers
   (with evidence) are the recall candidates (form (b), in-graph).
Form (a) — the blind one that can reach gold OUTSIDE the graph: the matched
patterns' semantics drive short expansion queries for the S2 bulk engine.

BLIND by construction: expansion queries are generated from the user query +
frozen schema only. Gold content never enters query generation (the gold-title
probes in .research_tmp/diag_source_vs_query.py were diagnostics, not pipeline).

Validation criteria pre-registered in .research_tmp/goal_log_0829.md §2.
"""
import hashlib, json, os

from granular_agent.llm_client import parse_json_response
from contest.cost_ledger import ledger

GRAPH_DIR = os.path.join("data", "graphs", "llm_domain")
_CACHE_DIR = os.path.join(".research_tmp", "contest_survey", "_intent_cache")


def _cache_get(name: str, key: str):
    path = os.path.join(_CACHE_DIR, f"{name}_{key}.json")
    if os.path.exists(path):
        try:
            return json.loads(open(path, encoding="utf-8").read())
        except Exception:
            return None
    return None


def _cache_put(name: str, key: str, obj):
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(os.path.join(_CACHE_DIR, f"{name}_{key}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    except Exception:
        pass


def _qkey(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()[:12]


def load_frozen_schema(graph_dir: str = GRAPH_DIR) -> list[dict]:
    """Frozen tbox patterns: [{pattern_id, description, roles, family}]."""
    cached = getattr(load_frozen_schema, "_s", None)
    if cached is not None:
        return cached
    kb = json.load(open(os.path.join(graph_dir, "kb.json"), encoding="utf-8"))
    pats = kb["tbox"]["patterns"]
    items = pats.values() if isinstance(pats, dict) else pats
    out = []
    for p in items:
        if p.get("deprecated"):
            continue
        out.append({"pattern_id": p["pattern_id"],
                    "description": p.get("description", ""),
                    "roles": [r["role"] for r in p.get("role_slots", [])][:6],
                    "family": p.get("family", "")})
    load_frozen_schema._s = out
    return out


def _schema_lines(schema: list[dict]) -> str:
    return "\n".join(f"- {p['pattern_id']}: {p['description']} "
                     f"[roles: {', '.join(p['roles'])}]" for p in schema)


def intent_parse(query: str, max_patterns: int = 3,
                 schema: list[dict] | None = None) -> list[dict]:
    """① query -> [{pattern_id, intent, slots}] against the frozen schema."""
    key = _qkey(query)
    hit = _cache_get("parse", key)
    if hit is not None:
        return hit
    schema = schema or load_frozen_schema()
    p = ("An academic paper search request must be parsed into a frozen "
         "knowledge-graph schema. Schema patterns (relation types with roles):\n"
         + _schema_lines(schema) + "\n\n"
         f"Search request: «{query}»\n\n"
         f"Pick up to {max_patterns} patterns capturing the request's INTENT — "
         "the relation(s) the researcher actually cares about (e.g. a method "
         "IMPROVES another, a technique APPLIES_TO a task, X COMPARED with Y), "
         "not just topic keywords. Fill each pattern's roles with short phrases "
         "from the request (empty string if the request leaves it open).\n"
         'Output JSON: {"matches": [{"pattern_id": "...", "intent": "<one '
         'sentence: what relation the seeker wants>", "slots": {"<role>": '
         '"<phrase or empty>"}}]}')
    raw = ledger.llm("DeepSeek-V4-Flash", p, max_tokens=500, enable_thinking=False)
    if raw is None:
        return []          # transient LLM failure: do NOT cache the empty
    obj = parse_json_response(raw) or {}
    valid = {s["pattern_id"] for s in schema}
    matches = [m for m in (obj.get("matches") or [])
               if m.get("pattern_id") in valid][:max_patterns]
    _cache_put("parse", key, matches)
    return matches


def schema_expansion_queries(query: str, max_patterns: int = 3,
                             per_pattern: int = 2) -> list[str]:
    """Form (a), blind: expansion queries derived from the matched patterns'
    semantics + the request's topic. Short token queries — the S2 bulk engine
    AND-matches all tokens over title+abstract (pipeline-measured: 2-3 token
    queries recall best, 6+ over-constrain)."""
    matches = intent_parse(query, max_patterns=max_patterns)
    if not matches:
        return []
    key = _qkey(query)
    hit = _cache_get("expandv4", key)          # v4: + family-level broad query (④)
    if hit is not None:
        return hit
    schema = {s["pattern_id"]: s for s in load_frozen_schema()}
    lines = []
    for m in matches:
        s = schema.get(m.get("pattern_id"), {})
        slots = "; ".join(f"{k}={v}" for k, v in (m.get("slots") or {}).items() if v)
        lines.append(f"- pattern {m['pattern_id']} ({s.get('description', '')}); "
                     f"intent: {m.get('intent', '')}; slots: {slots or 'none'}")
    p = ("A paper search engine AND-matches ALL tokens of a query against "
         "title+abstract — every extra token SHRINKS recall. For the request "
         "below, schema patterns were matched. Write keyword queries from the "
         "VOCABULARY THE TARGET PAPERS' AUTHORS use in their titles/abstracts.\n\n"
         f"Request: «{query}»\nMatched patterns:\n" + "\n".join(lines) + "\n\n"
         f"Write up to {per_pattern} ENTITY queries PER matched pattern, PLUS "
         "ONE narrowing variant per pattern. Hard rules:\n"
         "1. 2-3 content tokens MAX per query (4+ over-constrains AND matching).\n"
         "2. NEVER use relation verbs as tokens (improves, outperforms, relies, "
         "followed by, after...) — papers are found by their ENTITY vocabulary: "
         "task names, method/model names, dataset/domain nouns.\n"
         "3. Use the field's own nouns, including adjacent-field synonyms a "
         "seeker's phrasing may miss (e.g. climate prediction -> weather "
         "forecasting; text classification -> NLP; robot control -> motion "
         "planning) — this is what the pattern's ROLES are filled with in real "
         "papers.\n"
         "4. NARROWING VARIANT: the pattern's core entity phrase + exactly one "
         "of 'survey' / 'review' / 'benchmark' (pick the one authors of such "
         "papers actually use). Expert reading lists favor representative "
         "works; this token collapses huge match sets onto them.\n"
         "5. Stay on the request's CENTRAL topic entities — ignore its example "
         "scenarios and edge conditions.\n"
         "6. No question phrasing, no stopwords. Named methods verbatim.\n"
         "7. FINALLY add exactly ONE broad family-level query (2-3 tokens) "
         "covering the shared research theme of ALL matched patterns together "
         "(mechanism ④: the coarser granularity of the pattern families).\n"
         'Output JSON: {"queries": ["...", ...]}')
    raw = ledger.llm("DeepSeek-V4-Flash", p, max_tokens=500, enable_thinking=False)
    if raw is None:
        return []          # transient LLM failure: do NOT cache the empty
    obj = parse_json_response(raw) or {}
    qs = [q.strip() for q in (obj.get("queries") or [])
          if q and q.strip()][:max_patterns * per_pattern + max_patterns + 2]
    _cache_put("expandv4", key, qs)
    return qs


def graph_direct_hits(pattern_ids: list[str], graph_dir: str = GRAPH_DIR,
                      max_edges: int = 40) -> list[dict]:
    """③ form (b): hyperedges of the matched pattern types -> role concepts +
    provenance papers (evidence included). In-graph only; informational until
    the graph covers more of the benchmark domain."""
    kb = json.load(open(os.path.join(graph_dir, "kb.json"), encoding="utf-8"))
    con = kb["abox"]["concepts"]
    pm = kb.get("_paper_meta", {})
    wanted = set(pattern_ids)
    out = []
    hg = kb["abox"]["hyperedges"]
    edges = hg.values() if isinstance(hg, dict) else hg
    for h in edges:
        if h.get("kind") == "cites" or h.get("pattern_type") not in wanted:
            continue
        roles = {}
        for nid, role in zip(h.get("node_ids", []), h.get("node_roles", [])):
            c = con.get(nid) if isinstance(con, dict) else None
            if not c or c.get("type") == "PAPER":
                continue
            sv = c.get("surface_variants") or []
            roles.setdefault(role, sv[0].get("surface", nid) if sv else nid)
        prov = h.get("provenance") or []
        papers = []
        for pr in prov:
            meta = pm.get(pr.get("paper_id", ""), {})
            if meta.get("title"):
                papers.append({"paper_id": pr["paper_id"],
                               "title": meta["title"],
                               "year": meta.get("year", ""),
                               "evidence": pr.get("evidence", "")[:200]})
        out.append({"he_id": h.get("he_id"), "pattern_type": h["pattern_type"],
                    "roles": roles, "papers": papers})
        if len(out) >= max_edges:
            break
    return out
