# -*- coding: utf-8 -*-
"""Hypergraph-assisted query expansion (form A, 2026-08-28).

Uses the evolved concept graph's surface_variants — the cross-paper-merged
method-name synonyms (DQN / deep Q-network / deep Q learning are ONE concept
node after 86 merges) — as a query expansion dictionary. This is the
retrieval-layer differentiation #1: the graph participates in recall BEFORE
any candidate exists, not just in reranking.

Mechanism (deterministic, graph-side only — no LLM):
  user query -> tokenize -> match against concept surfaces (longest-match)
  -> for each matched concept, inject its OTHER surfaces into the rewrite
  pool -> recall queries generated from the expanded term set.

The expanded variants are things keyword recall would otherwise miss: the
graph knows 'Double Q-learning' and 'clipping the error term' live near DQN
because six papers' usage merged them.
"""
import json, os, re

_GRAPH_PATH = os.path.join(".research_tmp", "runs", "kernel_v2",
                           "A2M_EVO1_PPR_65EB3FEB4B1B", "concept_graph.json")

_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with",
         "how", "can", "be", "is", "are", "some", "me", "my", "please",
         "provide", "give", "show", "recent", "latest", "past", "paper",
         "papers", "research", "work", "works", "study", "studies", "using",
         "use", "used", "based", "improve", "improving", "improves"}

# closed-class generic surfaces: a single common word that names no method.
# As a MATCH seed it's noise ('network' in 'neural network architecture'),
# as a VARIANT it's a downgrade ('reinforcement learning' -> 'learning').
# (Same discipline as the aligner noise gate.)
_GENERIC_SINGLE = {"learning", "network", "networks", "model", "models",
                   "method", "algorithm", "training", "agent", "policy",
                   "value", "values", "data", "function", "gradient",
                   "architecture", "layer", "layers", "loss"}


def _is_generic_surface(s: str) -> bool:
    t = _tokens(s)
    return len(t) == 1 and next(iter(t)) in _GENERIC_SINGLE


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()


def _tokens(s: str) -> set[str]:
    return {t for t in _norm(s).split() if t and t not in _STOP and len(t) > 2}


def load_concept_dict(graph_path: str = _GRAPH_PATH) -> list[dict]:
    """[{surfaces: set, type, canonical}] from the accumulated concept graph.
    Only concepts with >=2 distinct surfaces are useful as a dictionary."""
    g = json.load(open(graph_path, encoding="utf-8"))
    out = []
    for cid, c in g["concepts"].items():
        surfaces = list({(v.get("surface") or "").strip()
                         for v in c.get("surface_variants", [])
                         if (v.get("surface") or "").strip()})
        # noise gate (same discipline as aligner): figure/table refs, single
        # letters, closed-class generics never become expansion seeds
        keep = [s for s in surfaces
                if len(s) >= 4 and not re.match(r"^(fig|table|eq)", s.lower())
                and len(_tokens(s)) >= 1 and not _is_generic_surface(s)
                and "$" not in s and "\\" not in s]
        if len(keep) >= 2:
            out.append({"surfaces": keep, "type": c.get("type", ""),
                        "canonical": c.get("canonical_name") or keep[0]})
    return out


def expand_query(query: str, concept_dict: list[dict] | None = None,
                 max_concepts: int = 5) -> dict:
    """Match query tokens against concept surfaces; return matched concepts
    with their variant expansions. Longest-surface-match wins (a matched
    multi-word surface suppresses single-token matches inside it)."""
    cd = concept_dict if concept_dict is not None else load_concept_dict()
    qnorm = _norm(query)
    qtoks = _tokens(query)
    matched = []
    used_spans: list[tuple[int, int]] = []
    # sort candidate surfaces longest-first; a surface matches if all its
    # tokens appear in the query (order-insensitive containment)
    ranked = sorted(((s, c) for c in cd for s in c["surfaces"]),
                    key=lambda t: -len(t[0]))
    for surface, concept in ranked:
        stoks = _tokens(surface)
        if not stoks or not stoks <= qtoks or _is_generic_surface(surface):
            continue
        if any(concept["canonical"] == m["canonical"] for m in matched):
            continue
        matched.append({"canonical": concept["canonical"],
                        "type": concept["type"],
                        "matched_surface": surface,
                        "variants": [s for s in concept["surfaces"]
                                     if s != surface][:6]})
        if len(matched) >= max_concepts:
            break
    expansions = []
    for m in matched:
        for v in m["variants"]:
            if v not in expansions:
                expansions.append(v)
    return {"matched": matched, "expansion_terms": expansions}


def expanded_rewrites(query: str, base_rewrites: list[str],
                      concept_dict: list[dict] | None = None) -> list[str]:
    """Inject graph expansions into the LLM's base rewrites: append rewrite
    variants that swap the matched surface for a graph variant. Returns the
    combined query list (base first — expansions are supplements, not
    replacements)."""
    r = expand_query(query, concept_dict)
    out = list(base_rewrites)
    for m in r["matched"]:
        for v in m["variants"][:3]:
            # swap the matched surface for this variant in each base rewrite
            # that contains it (case-insensitive)
            for br in base_rewrites:
                if re.search(re.escape(m["matched_surface"]), br, re.I):
                    swapped = re.sub(re.escape(m["matched_surface"]), v, br,
                                     flags=re.I)
                    if swapped not in out:
                        out.append(swapped)
    return out
