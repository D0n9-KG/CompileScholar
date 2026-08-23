"""Deep meta-hypergraph self-evolution: trigger -> probe -> validate -> apply.

The closed loop that makes hypergraph_schema.py's data structure actually
self-evolve DURING extraction. This is the core of the deep-self-evolution
design (docs/dataset-design/SELF-EVOLUTION-v2.md); hypergraph_schema.py only
provides the data structure + the 4 evolution operations, NOT this loop.

Open problems addressed (see goal condition):

P1 trigger (schema-gap vs extraction-error): a single validate() failure is
NOT enough — could be an LLM extraction error to discard. We use a mismatch
SIGNATURE = (pattern_type, role-tuple, reason). Recurrence of the same
signature across >=2 DAG nodes = strong structural-gap signal. A single
isolated failure is still sent to the probe, but must clear the evidence +
near-duplicate gate; an extraction error usually has no evidence span that
supports a new structure, so validate rejects it. cross_node recurrence is
reported as a SCORE (not a hard gate) — mirroring gap_discovery.score_gap,
so forward propagation can fire intra-paper.

P2 structural evidence anchoring: every meta-structure change carries a
verbatim evidence_span. add_meta_node needs a span proving the new type
exists; add_pattern's instance anchor is the failing hyperedge's own
evidence_span + a natural-language description. No span -> reject.

P3 bloat control: deterministic near-duplicate gate (token Jaccard >=0.5,
reusing grounding._tokens) on new meta-node types vs existing types, and on
new pattern (role-structure + pattern_type token) vs existing patterns.
Merge only when the probe explicitly proposes it.

P4 forward propagation: the caller (hypergraph_extractor) re-fetches
meta.to_prompt() for downstream DAG nodes after a successful apply. The
mechanism mirrors chained_extractor's schema_dirty flag.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from granular_agent.hypergraph_schema import (
    MetaHypergraph, MetaHyperedgePattern, Hyperedge, InstanceHypergraph,
    TOP_LEVEL_FAMILIES, QUALIFIER_REGISTRY,
)
from granular_agent.llm_client import call_llm, call_paratera, parse_json_response
from granular_agent.llm_client import embed_batch, cosine_sim
from granular_agent.grounding import _tokens


# ---------------------------------------------------------------------------
# P1: trigger — mismatch signature + cross-node recurrence
# ---------------------------------------------------------------------------

def mismatch_signature(he: Hyperedge, reason: str) -> tuple:
    """Structural signature of a validate() failure, by SHAPE not by name.

    pattern_type is LLM-named and almost never recurs across nodes, so
    including it would fragment signatures and make cross-node recurrence
    always 1 — deadlocking the P1 stability signal. Instead we key on
    (role-tuple, qualifier-key-set, reason): two failures with the same
    shape are the same structural gap regardless of the surface text or the
    name the LLM happened to give the pattern.
    """
    return (tuple(he.node_roles), frozenset(he.qualifiers.keys()), reason)


class EvolutionTrigger:
    """Records validate() failures and decides when to fire the probe.

    P1: cross-node recurrence is the strong signal, but we do NOT hard-gate
    on it (single failures with real evidence still probe, gated by
    validate_proposal). This keeps forward propagation alive intra-paper.
    """

    def __init__(self):
        # signature -> {node_id: count}
        self.seen: dict[tuple, dict[str, int]] = {}

    def record(self, he: Hyperedge, reason: str, node_id: str) -> tuple:
        """Record one failure. Returns (signature, cross_node_count)."""
        sig = mismatch_signature(he, reason)
        nodes = self.seen.setdefault(sig, {})
        nodes[node_id] = nodes.get(node_id, 0) + 1
        return sig, len(nodes)

    def cross_node_count(self, sig: tuple) -> int:
        return len(self.seen.get(sig, {}))

    def cumulative_count(self, sig: tuple) -> int:
        """Total failure count for a signature (across all nodes/papers).
        RETUNE (gate v2): cross_node>=2 alone is too strict for small corpora
        (B_MIN2 couldn't grow claim/measure). A gap recurring 3+ times total
        (even at 1 node each across papers) is also a real schema gap. Accept
        on cumulative >= threshold too. Signature UNCHANGED (role-tuple) —
        previous attempt to also relax signature was too coarse (merged
        different gaps, made it worse), reverted."""
        nodes = self.seen.get(sig, {})
        return sum(nodes.values())


# ---------------------------------------------------------------------------
# P2/P3: probe -> validate -> apply
# ---------------------------------------------------------------------------

EVOLUTION_PROBE_PROMPT = """You are the schema-evolution probe for a knowledge-hypergraph extractor.

A hyperedge extracted from a {domain} paper FAILED validation against the current meta-hypergraph (the schema). Decide whether the SCHEMA should evolve to accommodate it, or whether this is just an extraction error to discard.

Current meta-hypergraph (the schema):
{meta_prompt}

Failing hyperedges (instance level — these are what the extractor produced but the schema rejected):
{failing_hes}

For each distinct structural mismatch you judge to be a genuine schema gap (not an extraction error), propose ONE evolution operation. Choose from EXACTLY these 5:
1. add_meta_node      — add a new node TYPE (e.g. a domain entity category the schema lacks). Needs: type_id (UPPER_SNAKE), description.
2. add_pattern        — add a new hyperedge PATTERN (a relation the schema lacks, connecting existing or new types). Needs: pattern_id, description, role_slots (list of {{"role":"...","type":"..."}}), allowed_qualifiers (list — reuse registry keys or propose new), family (an existing family that fits, OR a new family name if none fits).
3. add_subclass       — declare an existing-or-new type as a specialization of another. Needs: sub (type_id), sup (existing type_id).
4. split_meta_node     — split an existing type into a subclass. Needs: type_id (existing), new_sub (new type_id).
5. merge_meta_nodes   — merge two near-duplicate types. Needs: a, b (existing type_ids), into (a or b).

BOUNDED OPERATION (Constitutional guardrail — divergence is prevented by the
conservative gate + merge/retire, NOT by locking families/qualifiers):
- Families are GROWABLE: the seed families {families} are a STARTING skeleton, not a ceiling. If a relation genuinely doesn't fit any existing family, propose a new family name (the first pattern of a new family auto-becomes its root). Prefer fitting an existing family; only invent a new one when no existing family matches.
- Qualifier keys are EXTENSIBLE: the registry {qualifier_keys} covers common cases. You may reuse these, OR propose a new qualifier key if the relation carries a context dimension none of the existing keys capture (e.g. confidence_level, assumption_scope). New keys should be generalizable, not paper-specific.

Rules:
- evidence_span MUST be a verbatim phrase copied from the failing hyperedges' evidence above (or empty if none supports it). A proposal with NO supporting evidence will be rejected.
- Only propose a structural change if the failing hyperedge genuinely cannot fit the current schema without it. If the hyperedge is just wrong/garbage (extraction error), propose NOTHING for it.
- Prefer add_pattern / add_subclass over add_meta_node when the gap is about a RELATION or a specialization, not a new entity category.
- REDUNDANCY CONTROL (critical — schema-free induction): do NOT propose a new
  pattern for every surface variation of a relation. If a failing edge differs
  from an EXISTING pattern only in verb phrasing or granularity (e.g.
  "outperforms" vs "is better than" vs "superior to"), reuse the existing
  pattern — express the nuance as a qualifier (e.g. relation_type/evidence_strength),
  NOT a new pattern. Prefer FEWER, more general patterns; only add a new pattern
  when the relation is STRUCTURALLY or SEMANTICALLY distinct from every existing
  one. Reuse an existing pattern's role-structure when the roles are compatible
  (same participants in same roles) — vary only via qualifier, not new role slots.
- ACTIVELY consider split/merge when the schema has grown redundant or conflated:
  * split_meta_node: if an existing type's instances appear in MULTIPLE distinct contexts that should be distinguished (e.g. METHOD used for both a model architecture and a training procedure, or RESULT for both a metric value and a qualitative finding).
  * merge_meta_nodes: if two existing types are semantically near-duplicate and should be unified into one.
- role_slots types must use existing node types from the schema above (or a type you propose in the SAME batch via add_meta_node).

Output ONLY a JSON array of proposals (empty array if none):
[{{"op":"add_meta_node","type_id":"...","description":"...","evidence_span":"...","rationale":"..."}}
 {{"op":"add_pattern","pattern_id":"...","description":"...","role_slots":[{{"role":"...","type":"..."}}],"allowed_qualifiers":[...],"family":"...","evidence_span":"...","rationale":"..."}}
 {{"op":"add_subclass","sub":"...","sup":"...","evidence_span":"...","rationale":"..."}}
 {{"op":"split_meta_node","type_id":"...","new_sub":"...","evidence_span":"...","rationale":"..."}}
 {{"op":"merge_meta_nodes","a":"...","b":"...","into":"...","evidence_span":"...","rationale":"..."}}]"""


def evolution_probe(failing_hes: list[Hyperedge], meta: MetaHypergraph,
                    paper_id: str, domain: str = "granular flow physics",
                    llm: str = "deepseek", instance: InstanceHypergraph | None = None) -> list[dict]:
    """LLM probe: look at failing hyperedges + current schema -> constrained proposals.

    Constrained proposal space (5 ops) keeps the search tractable (P1 convergence).
    Each proposal MUST carry an evidence_span (P2). Returns raw proposals; they are
    then filtered by validate_proposal (P3 near-duplicate gate + LLM distinctness).
    """
    if not failing_hes:
        return []
    # Render failing hyperedges compactly (node surfaces + roles + qualifiers + evidence)
    he_lines = []
    for i, he in enumerate(failing_hes[:12]):  # cap to keep prompt bounded
        nodes_str = []
        if instance is not None:
            for nid, role in zip(he.node_ids, he.node_roles):
                n = instance.nodes.get(nid)
                nodes_str.append(f"{role}:{n.surface if n else nid}({','.join(n.labels) if n else '?'})")
        else:
            nodes_str = [f"{r}:{nid}" for nid, r in zip(he.node_ids, he.node_roles)]
        quals = ",".join(f"{k}={v}" for k, v in he.qualifiers.items()) or "-"
        he_lines.append(
            f"  HE{i}: pattern_type={he.pattern_type}; nodes=[{', '.join(nodes_str)}]; qualifiers={quals}; evidence=\"{he.evidence_span}\""
        )
    failing_str = "\n".join(he_lines) or "  (none)"

    prompt = EVOLUTION_PROBE_PROMPT.format(
        domain=domain, meta_prompt=meta.to_prompt(), failing_hes=failing_str,
        families=", ".join(TOP_LEVEL_FAMILIES),
        qualifier_keys=", ".join(QUALIFIER_REGISTRY.keys()),
    )
    raw = _call(prompt, llm, max_tokens=2500)
    parsed = parse_json_response(raw)
    if not isinstance(parsed, list):
        return []
    # keep only dicts with an op + evidence_span
    out = []
    for p in parsed:
        if isinstance(p, dict) and p.get("op") and p.get("evidence_span", "").strip():
            p["paper_id"] = paper_id
            out.append(p)
    return out


def _call(prompt: str, llm: str, max_tokens: int = 4000) -> str | None:
    if llm == "deepseek":
        return call_llm(prompt, model="deepseek-chat", max_tokens=max_tokens)
    return call_paratera(prompt, model=llm, max_tokens=max_tokens)


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def _singularize_token(t: str) -> str:
    """Crude singularization so MATERIALS/MATERIAL are caught as near-dup.
    _tokens() does exact-token intersection, which misses plural variants."""
    if t.endswith("ies") and len(t) > 3:
        return t[:-3] + "y"
    if t.endswith("es") and len(t) > 2:
        return t[:-2]
    if t.endswith("s") and not t.endswith("ss") and len(t) > 1:
        return t[:-1]
    return t


def _near_dup(a: str, b: str) -> bool:
    """Near-duplicate for type/pattern names: token Jaccard >=0.5, OR the
    singularized-stem token Jaccard >=0.5 (catches plural/case variants that
    exact-token matching misses)."""
    if _jaccard(a, b) >= 0.5:
        return True
    ta = {_singularize_token(t) for t in _tokens(a)}
    tb = {_singularize_token(t) for t in _tokens(b)}
    if ta and tb:
        return len(ta & tb) / max(1, len(ta | tb)) >= 0.5
    return False


# ---- P3 semantic dedup (embedding) for pattern proposals ----
# Token matching misses near-synonyms like depends_on_property ~ affects_property
# (different tokens, same relation). Embedding cosine catches them.

SEMANTIC_DUP_THRESHOLD = 0.85


def _pattern_text(pid: str, description: str, role_slots: list, quals: list) -> str:
    """Text for pattern embedding (used by semantic-dedup + merge). The
    DESCRIPTION leads + is repeated to dominate the embedding over the
    surface id-name (which is LLM-named and noisy). role-structure is
    included for structure-aware similarity. id-name is de-emphasized."""
    roles = ",".join(f"{s.get('role','')}:{s.get('type','')}" for s in (role_slots or []))
    desc = description or ""
    return f"{desc} | {desc} | {roles} quals={','.join(quals or [])} [{pid}]"


def _ensure_pattern_embeds(meta: MetaHypergraph) -> dict:
    """Lazily compute + cache (on the meta object) embeddings for all existing
    patterns. Returns pattern_id -> embedding. Network failures degrade to
    whatever embeddings are available (honest degrade — token gate still runs)."""
    cache = getattr(meta, "_pat_embed_cache", None)
    if cache is None:
        cache = {}
        setattr(meta, "_pat_embed_cache", cache)
    missing = [pid for pid in meta.patterns if pid not in cache]
    if missing:
        texts = [_pattern_text(pid, meta.patterns[pid].description,
                               meta.patterns[pid].role_slots,
                               meta.patterns[pid].allowed_qualifiers)
                 for pid in missing]
        try:
            embs = _embed_texts_robust(texts) if texts else []
        except Exception:
            embs = []  # SSL disconnect etc -> degrade, token gate still runs
        if embs is None:
            embs = []
        for pid, e in zip(missing, embs):
            cache[pid] = e
    return cache


def _role_sig_from_slots(role_slots: list) -> tuple:
    """Structural signature of a proposed pattern's role-slots:
    (role-sequence, type-sequence). Used to gate semantic dedup to
    same-structure patterns only — different structure => genuinely
    different relation, not a dup even if semantically near."""
    rs = role_slots or []
    return (tuple(s.get("role", "") for s in rs),
            tuple(s.get("type", "") for s in rs))


def _semantic_near_dup_pattern(pid: str, description: str, role_slots: list,
                               quals: list, meta: MetaHypergraph,
                               threshold: float = SEMANTIC_DUP_THRESHOLD) -> tuple | None:
    """Return (existing_pattern_id, cosine) if new pattern semantically
    duplicates an existing one above threshold, else None. Degrades to None
    (skip semantic check) if embeddings unavailable (network/empty).

    STRUCTURE-GATED: only compares against existing patterns with the SAME
    role-structure (role/type tuple). A pattern with different structure is a
    genuinely different relation even if semantically near — rejecting it was
    over-strict (gate audit found ~26% over-rejection from this). Semantic
    dedup now catches only same-structure near-synonyms (true bloat)."""
    cache = _ensure_pattern_embeds(meta)
    if not cache:
        return None  # no existing patterns or embed unavailable
    new_emb = _embed_texts_robust([_pattern_text(pid, description, role_slots, quals)])
    if not new_emb:
        return None  # both embedding providers failed -> honest degrade
    ne = new_emb[0]
    new_sig = _role_sig_from_slots(role_slots)
    best_pid, best_s = None, 0.0
    for ex_pid, ex_pat in meta.patterns.items():
        if _role_sig(ex_pat) != new_sig:
            continue  # different structure -> not a dup candidate
        e = cache.get(ex_pid)
        if not e:
            continue
        s = cosine_sim(ne, e)
        if s > best_s:
            best_s, best_pid = s, ex_pid
    if best_pid is not None and best_s >= threshold:
        return (best_pid, round(best_s, 3))
    return None


def _role_sig(pat: MetaHyperedgePattern) -> tuple:
    """Structural signature of a pattern: (role-sequence, type-sequence)."""
    return (tuple(s.get("role", "") for s in pat.role_slots),
            tuple(s.get("type", "") for s in pat.role_slots))


def validate_proposal(proposal: dict, meta: MetaHypergraph,
                      domain: str = "granular flow physics",
                      llm: str = "deepseek") -> dict:
    """P2+P3 gate: evidence required + deterministic near-duplicate rejection +
    LLM distinctness check. Returns {valid, reason, corrected}."""
    op = proposal.get("op", "")
    evidence = (proposal.get("evidence_span", "") or "").strip()
    if not evidence:
        return {"valid": False, "reason": "no verbatim evidence span", "proposal": proposal}

    # ---- P3: deterministic near-duplicate gate (non-LLM) ----
    if op == "add_meta_node":
        tid = (proposal.get("type_id", "") or "").strip()
        if not tid:
            return {"valid": False, "reason": "empty type_id"}
        for ex in meta.types():
            if _near_dup(tid, ex):
                return {"valid": False, "reason": f"near-duplicate type '{ex}'",
                        "suggested_alternative": ex, "proposal": proposal}
    elif op == "add_pattern":
        pid = (proposal.get("pattern_id", "") or "").strip()
        # REDESIGN v2: family is GROWABLE (no longer rejected if not in the
        # 6 seed families). A new family's first pattern auto-becomes its
        # root via add_pattern. Divergence is gated by conservative gate
        # (cross_node>=2) + merge/retire, not by locking families.
        # qualifier keys are EXTENSIBLE: a proposal may carry keys not in
        # QUALIFIER_REGISTRY (new context dimensions). They get registered
        # dynamically. Enum-value checks still apply to KNOWN enum keys.
        new_sig = (tuple(s.get("role", "") for s in proposal.get("role_slots", [])),
                   tuple(s.get("type", "") for s in proposal.get("role_slots", [])))
        for ex_pid, ex_pat in meta.patterns.items():
            # reject if same role-structure AND pattern_id token-near-duplicate
            if _role_sig(ex_pat) == new_sig and _near_dup(pid, ex_pid):
                return {"valid": False, "reason": f"near-duplicate pattern '{ex_pid}'",
                        "suggested_alternative": ex_pid, "proposal": proposal}
        # P3 semantic dedup (catches near-synonyms token matching misses)
        sem = _semantic_near_dup_pattern(pid, proposal.get("description", ""),
                                          proposal.get("role_slots", []),
                                          proposal.get("allowed_qualifiers", []), meta)
        if sem:
            return {"valid": False, "reason": f"semantic near-dup of pattern '{sem[0]}' (cosine {sem[1]})",
                    "suggested_alternative": sem[0], "proposal": proposal}
    elif op == "add_subclass":
        sub, sup = proposal.get("sub", ""), proposal.get("sup", "")
        if not sub or not sup:
            return {"valid": False, "reason": "missing sub/sup"}
        if sup not in meta.meta_nodes:
            return {"valid": False, "reason": f"sup '{sup}' not in schema"}
        if meta.is_subtype(sub, sup):
            return {"valid": False, "reason": f"{sub} already subclass of {sup}"}
    elif op == "split_meta_node":
        tid = proposal.get("type_id", "")
        if tid not in meta.meta_nodes:
            return {"valid": False, "reason": f"{tid} not in schema"}
    elif op == "merge_meta_nodes":
        a, b = proposal.get("a", ""), proposal.get("b", "")
        if a not in meta.meta_nodes or b not in meta.meta_nodes:
            return {"valid": False, "reason": "merge targets not in schema"}
        if a == b:
            return {"valid": False, "reason": "merge a==b"}
    else:
        return {"valid": False, "reason": f"unknown op '{op}'"}

    # ---- LLM distinctness / necessity check (one call, batched per proposal) ----
    # Keep this cheap: only fire if the deterministic gate passed.
    check_prompt = f"""A schema-evolution proposal for a {domain} knowledge hypergraph. Judge if it is a DISTINCT, generalizable structural addition.

Current schema:
{meta.to_prompt()}

Proposal:
{proposal}

Verbatim evidence span from the paper:
"{evidence}"

DUPLICATE RULE (critical): a proposal is a near-duplicate ONLY if an existing pattern has the SAME role-structure (same number of roles AND matching types). A proposal with a DIFFERENT role-structure is a DISTINCT relation — e.g. a multi-input dependency (one output depends on 3+ inputs) is NOT a duplicate of a single-input dependency pattern, even if semantically similar. Do NOT reject different-structure proposals as duplicates.

Reject ONLY if:
- the span doesn't support the proposed structure, OR
- the proposal is a pure numeric value / single equation / modeling-method NAME (NOT a relation between entities), OR
- an existing pattern with the SAME role-structure already covers it, OR
- it's too paper-specific to generalize.

IMPORTANT — do NOT conflate these two:
  * REJECT: "X is defined as 0.5" / "the learning rate is 0.4" (a value assignment, no relation).
  * ACCEPT: "Informer is the model composed of ProbSparse attention and a distilling layer" / "X is composed of Y and Z" / "a relation is universal if it holds under any input distribution" — these are DEFINITIONAL/COMPOSITIONAL RELATIONS (one entity is defined BY / composed OF others). They ARE relations between entities and should be accepted as new patterns (e.g. defines_composition, identified_with), not rejected as "definitions".

Accept if the span supports a distinct, generalizable relation the schema lacks (including multi-input/joint dependencies AND definitional/compositional relations).

Output ONLY JSON: {{"valid": true/false, "reason": "one short sentence citing the span and the role-structure judgment"}}"""
    raw = _call(check_prompt, llm, max_tokens=300)
    res = parse_json_response(raw)
    if not isinstance(res, dict):
        return {"valid": False, "reason": "LLM check unparseable", "proposal": proposal}
    return {"valid": bool(res.get("valid", False)),
            "reason": res.get("reason", ""), "proposal": proposal}


def apply_proposal(meta: MetaHypergraph, proposal: dict, paper_id: str) -> str | None:
    """Apply one validated proposal to the meta-hypergraph. Returns new version or None."""
    op = proposal.get("op", "")
    evidence = proposal.get("evidence_span", "")
    if op == "add_meta_node":
        return meta.add_meta_node(proposal["type_id"], proposal.get("description", ""),
                                  evidence=evidence, paper_id=paper_id)
    if op == "add_pattern":
        pat = MetaHyperedgePattern(
            pattern_id=proposal["pattern_id"], description=proposal.get("description", ""),
            role_slots=proposal.get("role_slots", []),
            allowed_qualifiers=proposal.get("allowed_qualifiers", []),
            family=proposal.get("family", ""))
        return meta.add_pattern(pat, evidence=evidence, paper_id=paper_id)
    if op == "add_subclass":
        return meta.add_subclass(proposal["sub"], proposal["sup"],
                                 evidence=evidence, paper_id=paper_id)
    if op == "split_meta_node":
        return meta.split_meta_node(proposal["type_id"], proposal["new_sub"],
                                    evidence=evidence, paper_id=paper_id)
    if op == "merge_meta_nodes":
        return meta.merge_meta_nodes(proposal["a"], proposal["b"],
                                     into=proposal.get("into"),
                                     evidence=evidence, paper_id=paper_id)
    return None


def run_evolution_loop(meta: MetaHypergraph, failing_hes: list[Hyperedge],
                       trigger: EvolutionTrigger, node_id: str, paper_id: str,
                       domain: str = "granular flow physics", llm: str = "deepseek",
                       instance: InstanceHypergraph | None = None) -> tuple[list[dict], int]:
    """One iteration of the closed loop for the failures accumulated at a node.

    1. record each failure -> cross_node counts (P1)
    2. probe: LLM proposes structural changes for the failing hyperedges (P2)
    3. validate_proposal: evidence + near-dup gate + LLM distinctness (P2/P3)
    4. apply_proposal: mutate meta-hypergraph (P4 forward propagation via to_prompt)

    Returns (evolutions, n_calls). evolutions records cross_node score + op + evidence.
    """
    evolutions: list[dict] = []
    n_calls = 0
    if not failing_hes:
        return evolutions, 0
    # P1: record all failures (cross-node recurrence accounting)
    sigs = {}
    for he, reason in failing_hes:  # list of (he, reason) tuples expected
        sig, cross = trigger.record(he, reason, node_id)
        sigs.setdefault(sig, (he, cross, reason))
    # The probe sees the WHOLE batch, so each proposal inherits the batch's
    # cross-node recurrence (the max across the triggering failures). The
    # previous version keyed the lookup on (op, type_id) which lives in a
    # different signature space than record() -> cross_node was always 1.
    # NOTE: a precise proposal->triggering-failure link is a known refinement;
    # the batch-max is correct (never understates) and fixes the dead signal.
    batch_cross = max((trigger.cross_node_count(sig) for sig in sigs), default=1)
    # RETUNE (gate v2): accept growth if cross_node>=2 OR cumulative>=3.
    # Signature kept (role-tuple) — only the threshold relaxes.
    batch_cumulative = max((trigger.cumulative_count(sig) for sig in sigs), default=1)
    # Probe on the distinct failing hyperedges (dedup by signature)
    distinct_hes = [he for (he, _, _) in sigs.values()]
    proposals = evolution_probe(distinct_hes, meta, paper_id, domain=domain,
                                llm=llm, instance=instance)
    n_calls += 1
    for p in proposals:
        v = validate_proposal(p, meta, domain=domain, llm=llm)
        n_calls += 1
        if not v.get("valid"):
            evolutions.append({"op": p.get("op"), "rejected": True,
                               "reason": v.get("reason", ""), "evidence": p.get("evidence_span", ""),
                               "proposal": p,  # full proposal for over-rejection audit
                               "suggested_alternative": v.get("suggested_alternative", "")})
            continue
        # CONSERVATIVE GATE (AgentCAT "new labels only when strictly necessary" —
        # the convergence mechanism the承重-design specifies, ON TOP of the
        # bounded-op family gate). A GROWTH op (add_pattern / add_meta_node /
        # add_subclass) is accepted ONLY when the structural gap RECURRED across
        # >= CONSERVATIVE_CROSS_NODE nodes (batch_cross). A single-node gap is
        # rejected as "needs recurrence" — if it is a real schema gap (not a
        # one-off extraction error) it will recur at another node/paper and be
        # accepted then. This is the direct cure for the monotonic-pattern-growth
        # divergence (iteration 2: 15->30->49->68 with no plateau): without it
        # the proposer adds a new within-family pattern for every one-off edge
        # that fails validate, even when the failure is an extraction artifact.
        # Repair ops (split/merge — already triggered by instance clustering,
        # not by a single failure) are NOT gated; only growth is.
        op = p.get("op", "")
        is_growth = op in ("add_pattern", "add_meta_node", "add_subclass")
        # RETUNE (gate v2): accept if cross_node>=2 OR cumulative>=8.
        gate_pass = batch_cross >= CONSERVATIVE_CROSS_NODE or batch_cumulative >= CONSERVATIVE_CUMULATIVE
        # schema-bloat cap: once MAX_ACTIVE_PATTERNS reached, reject all growth
        # so the extractor's schema prompt stays bounded (extraction quality
        # degrades when the schema balloons — see CONSERVATIVE_CUMULATIVE note).
        if is_growth and len(meta.active_patterns()) >= MAX_ACTIVE_PATTERNS:
            evolutions.append({"op": op, "rejected": True,
                               "reason": f"schema cap: active_patterns={len(meta.active_patterns())} >= {MAX_ACTIVE_PATTERNS}",
                               "evidence": p.get("evidence_span", ""),
                               "proposal": p, "cross_node": batch_cross,
                               "cumulative": batch_cumulative,
                               "node_id": node_id, "paper_id": paper_id})
            continue
        if is_growth and not gate_pass:
            evolutions.append({"op": op, "rejected": True,
                               "reason": f"conservative gate: cross_node={batch_cross} < {CONSERVATIVE_CROSS_NODE} AND cumulative={batch_cumulative} < {CONSERVATIVE_CUMULATIVE}",
                               "evidence": p.get("evidence_span", ""),
                               "proposal": p, "cross_node": batch_cross,
                               "cumulative": batch_cumulative,
                               "node_id": node_id, "paper_id": paper_id})
            continue
        new_ver = apply_proposal(meta, p, paper_id)
        if new_ver:
            evolutions.append({"op": op, "version": new_ver,
                               "type_id": p.get("type_id") or p.get("pattern_id") or p.get("sub") or p.get("new_sub"),
                               "evidence": p.get("evidence_span", ""),
                               "rationale": p.get("rationale", ""),
                               "cross_node": batch_cross, "node_id": node_id,
                               "paper_id": paper_id})
    return evolutions, n_calls


# Conservative-gate threshold: a growth proposal is accepted only when the
# structural gap recurred across >= this many DAG nodes (cross_node). 2 = the
# gap showed up at 2+ independent positions, which is the P1 signal that
# distinguishes a schema gap from a one-off extraction error. This is the
# "strictly necessary" bar (AgentCAT conservative policy) expressed as the
# already-computed cross_node recurrence rather than an LLM judgment (A4
# circularity preserved — the gate is deterministic).
CONSERVATIVE_CROSS_NODE = 2
# RETUNE (gate v3): cumulative threshold. v2=3 let schema balloon (6->237 on
# 30 papers) which bloated the prompt and degraded extraction. v3 raised to 8
# as a backstop. NOW (v4): retrieval-based schema injection (see
# hypergraph_extractor._retrieved_schema_prompt) keeps the prompt bounded
# regardless of schema size, so the gate can relax back to let the schema
# evolve fully. Kept at 5 (between v2's looseness and v3's tightness) as a
# mild anti-noise guard — a pattern must recur across >=5 cumulative failures
# to be added, filtering one-off LLM naming noise.
CONSERVATIVE_CUMULATIVE = 5
# No hard cap — retrieval injection handles prompt size. Cap removed so the
# schema can grow as the corpus demands (DIAL-KG style: grow schema, retrieve
# relevant subset per chunk).
MAX_ACTIVE_PATTERNS = 10**9


def mismatch_signature_for_proposal(p: dict) -> tuple:
    """Structural signature for a proposal, in the SAME space as
    mismatch_signature() (role-tuple + qualifier-keys) so cross-node lookup
    is meaningful. Used for reporting recurrence on accepted evolutions."""
    role_slots = p.get("role_slots", []) or []
    roles = tuple(s.get("role", "") for s in role_slots)
    quals = frozenset(p.get("allowed_qualifiers", []) or [])
    return (roles, quals, "no-matching-meta-pattern")


# ===========================================================================
# Pattern-level SPLIT trigger (the headline operation).
#
# Deterministic (NOT LLM-judged) — breaks the A4 LLM-judge circularity:
# proposer=judge=auditor=deepseek. The decision to split is made by instance
# CLUSTERING, not by asking the LLM "should I split?". The LLM is used ONLY
# to NAME the resulting sub-patterns (a labeling task, not a judgment task).
#
# Trigger: for each active pattern, gather its instance hyperedges from one
# (or accumulated) paper(s), cluster them by edge-semantic signature. If the
# edges fall into >=2 clusters each of size >= MIN_CLUSTER, the pattern is
# over-wide -> split.
#
# Feature source (three-tier, honest degrade):
#   1. embedding cosine single-link (preferred, Paratera GLM-Embedding)
#   2. qualifier-value discrete clustering (fallback when embedding 429s —
#      the dependency_type qualifier is already a discrete enum, which is
#      the MOST reliable signal for dependency_relation, the P-E1 testbed)
#   3. token Jaccard single-link (last resort)
# Which tier was used is REPORTED (never silently substituted).
# ===========================================================================

MIN_CLUSTER = 3                 # a cluster needs >=3 instances to be a real sub-pattern
SPLIT_COS_THRESHOLD = 0.55      # edges with cosine >= this are same cluster (single-link)
SPLIT_TOKEN_THRESHOLD = 0.35    # token-Jaccard fallback threshold


def _edge_cluster_text(he: Hyperedge, instance: InstanceHypergraph) -> str:
    """Text signature of an instance hyperedge FOR CLUSTERING. Captures the
    RELATION's semantics, deliberately excluding the entity surfaces (which
    share domain vocabulary — stress/shear/strain — and would collapse every
    edge of a domain pattern into one cluster, defeating split).

    Strategy: a catch-all pattern's distinguishing signal is HOW it relates
    its nodes, not WHAT it relates. We surface that as:
      role:value | role:value | <relation-qualifier values>
    where the relation qualifiers (dependency_type, relation_type, ...)
    carry the verb/relation phrase. Falls back to the evidence_span's first
    clause if no relation qualifiers (so the verb phrase still drives it)."""
    rel_qualifier_keys = ("dependency_type", "relation_type", "function_form",
                          "condition", "dependency")
    parts = []
    for nid, role in zip(he.node_ids, he.node_roles):
        n = instance.nodes.get(nid)
        parts.append(f"{role}:{n.surface if n else nid}")
    rel_vals = [str(he.qualifiers[k]) for k in rel_qualifier_keys if k in he.qualifiers]
    if rel_vals:
        # relation phrase is the PRIMARY clustering signal — put it first +
        # repeat so it dominates the embedding over the entity surfaces
        rel_phrase = " ".join(rel_vals)
        return f"{rel_phrase} | {rel_phrase} | " + " ".join(parts)
    # no relation qualifier: use the evidence_span verb phrase (first 100 chars,
    # which usually contains the relation verb), entity surfaces de-emphasized
    return (he.evidence_span[:100] or " ".join(parts))


def _embed_texts_robust(texts: list[str]) -> list[list[float]] | None:
    """Embedding with provider fallback: try Paratera GLM-Embedding first,
    fall back to CST qwen3-embedding:8b on 429/network error (Paratera's
    embedding endpoint rate-limits hard; CST is a separate OpenAI-compatible
    uni-api that doesn't share the limit). Returns None only if BOTH fail —
    caller then degrades to qualifier/token clustering (honest, reported)."""
    if not texts:
        return []
    # tier 1: Paratera GLM-Embedding
    try:
        from granular_agent.llm_client import ENV as _ENV
        embs = embed_batch(texts)
        if embs and len(embs) == len(texts):
            return embs
    except Exception:
        pass  # 429 / SSL -> fall through
    # tier 2: CST qwen3-embedding:8b (separate provider, separate rate limit)
    try:
        from granular_agent.llm_client import ENV as _ENV, _CTX
        import urllib.request, json
        key = _ENV.get("CST_API_KEY", "")
        base = (_ENV.get("CST_BASE_URL", "") or "").rstrip("/")
        if key and base:
            body = json.dumps({"model": "qwen3-embedding:8b", "input": texts}).encode()
            req = urllib.request.Request(
                base + "/embeddings", data=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            raw = urllib.request.urlopen(req, context=_CTX, timeout=60).read()
            data = json.loads(raw).get("data", [])
            data.sort(key=lambda x: x.get("index", 0))
            embs = [d["embedding"] for d in data]
            if len(embs) == len(texts):
                return embs
    except Exception:
        pass
    return None  # both providers failed -> honest degrade


def _embedding_clusters(texts: list[str], threshold: float) -> list[list[int]] | None:
    """Single-link clustering by embedding cosine. Returns None if embeddings
    unavailable (network/429) — caller falls back. Single-link: two edges are
    same cluster if cosine >= threshold (transitive closure)."""
    if len(texts) < 2:
        return [[0]] if texts else []
    embs = _embed_texts_robust(texts)
    if embs is None or len(embs) != len(texts):
        return None  # both embedding providers failed -> honest degrade
    n = len(texts)
    # union-find single-link
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for i in range(n):
        for j in range(i + 1, n):
            if cosine_sim(embs[i], embs[j]) >= threshold:
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _token_clusters(texts: list[str], threshold: float) -> list[list[int]]:
    """Single-link clustering by singularized-token Jaccard (last-resort
    fallback when both embedding and qualifier-value are unusable)."""
    n = len(texts)
    toks = [{_singularize_token(t) for t in _tokens(tx)} for tx in texts]
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for i in range(n):
        for j in range(i + 1, n):
            if toks[i] and toks[j]:
                jacc = len(toks[i] & toks[j]) / max(1, len(toks[i] | toks[j]))
                if jacc >= threshold:
                    union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _dependency_context_clusters(edges: list[Hyperedge], instance: InstanceHypergraph,
                                  meta: MetaHypergraph) -> list[list[int]] | None:
    """Cluster a pattern's instance edges by their DEPENDENCY CONTEXT (REDESIGN
    v2 step 5 — the new split dimension). The context of an edge = the SET of
    OTHER pattern families whose edges share at least one node with this edge.

    Two edges in the same cluster reference entities that co-occur with the
    same family mix — e.g. a measure edge whose node also appears in a
    constitutive_law edge vs one whose node appears in a claim edge. This
    catches an over-wide pattern that BEHAVES DIFFERENTLY in different relational
    contexts (the design's split-by-dependency-context case), which qualifier /
    embedding clustering miss when the edge text is similar but the surrounding
    graph structure differs.

    Deterministic + structural (graph reachability, no LLM). Returns None when
    every edge has the same (often empty) context — no split signal — so the
    caller falls through to embedding. Cluster sizes are gated downstream by
    MIN_CLUSTER (same as other tiers)."""
    # build nid -> set of families (across the WHOLE instance, so cross-pattern
    # context is captured, not just within this pattern's edges)
    node_fams: dict[str, set[str]] = {}
    for he in instance.hyperedges.values():
        pat = meta.patterns.get(he.pattern_type)
        fam = pat.family if pat else ""
        if not fam:
            continue
        for nid in he.node_ids:
            node_fams.setdefault(nid, set()).add(fam)
    # context per edge = union of families on its nodes, MINUS its own family
    own_fam = ""
    own_pat = meta.patterns.get(edges[0].pattern_type if edges else "")
    if own_pat:
        own_fam = own_pat.family
    contexts: list[frozenset] = []
    for he in edges:
        ctx: set[str] = set()
        for nid in he.node_ids:
            ctx |= node_fams.get(nid, set())
        ctx.discard(own_fam)  # exclude own family (every edge has it — not discriminating)
        contexts.append(frozenset(ctx))
    # if all edges share the same context, there's no split signal here
    if len(set(contexts)) <= 1:
        return None
    groups: dict[frozenset, list[int]] = {}
    for i, ctx in enumerate(contexts):
        groups.setdefault(ctx, []).append(i)
    return list(groups.values())


def _qualifier_is_discrete(edges: list[Hyperedge], max_ratio: float = 0.6) -> tuple[bool, str]:
    """A relation-kind qualifier is 'discrete' (treatable as a controlled enum)
    if its distinct values are few relative to edge count — i.e. values repeat.
    Free-text LLM qualifiers (almost all unique) are NOT discrete and must fall
    to embedding. Returns (is_discrete, the qualifier key used). max_ratio =
    distinct/total; <=0.6 means at least 40% repetition = discrete-ish."""
    REL_KEYS = ("dependency_type", "relation_type", "relation_kind", "kind")
    for k in REL_KEYS:
        vals = [he.qualifiers.get(k) for he in edges if he.qualifiers.get(k)]
        if not vals:
            continue
        distinct = len(set(vals))
        if distinct / len(vals) <= max_ratio:
            return True, k
    return False, ""


def cluster_pattern_instances(edges: list[Hyperedge], instance: InstanceHypergraph,
                              meta: MetaHypergraph, prefer: str = "auto") -> tuple[list[list[int]], str]:
    """Cluster a pattern's instance edges. Returns (clusters, method_used).
    method_used in {"discrete","embedding","token"} — always reported so the
    split decision's evidence base is auditable (A4: deterministic, but we
    must say WHICH deterministic signal).

    Tier order (CHANGED after real-data iteration — see HYPERGRAPH-EVOLUTION.md
    'Manual inspection'): a relation-kind qualifier that is DISCRETE (controlled
    enum, values repeat) is the strongest split signal — cluster on it directly.
    Only fall to embedding when the qualifier is free-text (values almost all
    unique, continuous-gradient). Embedding on a discrete enum was the bug: it
    collapsed 4 distinct enum values into 1 cluster because node-surface
    domain vocabulary dominated the embedding."""
    # tier 1: discrete qualifier clustering (controlled-enum relation kind)
    is_disc, qk = _qualifier_is_discrete(edges)
    if is_disc:
        # group by that qualifier's value (only edges that have it; others
        # form a residual group so no edge is lost)
        groups: dict[str, list[int]] = {}
        for i, he in enumerate(edges):
            v = he.qualifiers.get(qk, "(none)")
            groups.setdefault(v if v else "(none)", []).append(i)
        return list(groups.values()), "discrete"
    # tier 1.5: dependency-context clustering (REDESIGN v2 step 5 — the new
    # split dimension). Groups edges by the SET of other families co-occurring
    # on their nodes. Catches an over-wide pattern behaving differently in
    # different relational contexts (qualifier/embedding miss this when edge
    # text is similar but surrounding graph structure differs). Returns None
    # when all edges share the same context (no signal) -> falls through.
    dc = _dependency_context_clusters(edges, instance, meta)
    if dc is not None:
        return dc, "dependency_context"
    # tier 2: embedding (free-text qualifier — needs semantic clustering)
    texts = [_edge_cluster_text(he, instance) for he in edges]
    if prefer in ("auto", "embedding"):
        ec = _embedding_clusters(texts, SPLIT_COS_THRESHOLD)
        if ec is not None:
            return ec, "embedding"
    # tier 3: token fallback
    return _token_clusters(texts, SPLIT_TOKEN_THRESHOLD), "token"


SEMANTIC_CLUSTER_PROMPT = """You are grouping hyperedges of ONE pattern by the PHYSICAL QUANTITY they relate, to detect if the pattern is over-wide (conflating different physical laws). Each edge connects some nodes with a verbatim evidence_span.

Pattern: {pattern_id}
Edges:
{edges_block}

Group the edges by the physical quantity/relation they express (e.g. heat-flux laws vs stress laws vs transport-length laws). Output >=2 groups ONLY IF the edges genuinely fall into >=2 distinct physical categories each with >=3 edges. If most edges express the same category, output ONE group (no split).

Output ONLY a JSON array: [{{"label":"<category>","edge_ids":[0,3,5]}}, ...] where edge_ids are 0-indexed positions in the list above. Every edge id must appear in exactly one group."""


def _llm_semantic_clusters(edges: list[Hyperedge], instance: InstanceHypergraph,
                            pattern_id: str, llm: str = "deepseek") -> tuple[list[list[int]], str]:
    """LLM semantic grouping: ask deepseek to group a pattern's edges by the
    physical quantity/relation they express. Returns (clusters, "llm_semantic").

    Honesty on A4 (LLM-judge circularity): the SPLIT DECISION is still
    deterministic — detect fires when edge_count >= 2*MIN_CLUSTER, regardless
    of what the LLM says. The LLM only does the GROUPING (labeling task), and
    only produces >=2 groups if the edges genuinely fall into >=2 physical
    categories each >=3. If the LLM says one group, no split. So the LLM never
    decides WHETHER to split — it classifies, and split fires only if its
    classification has >=2 sizeable groups. This is half-deterministic:
    stronger than pure LLM-self-judgment, weaker than discrete clustering."""
    lines = []
    for i, he in enumerate(edges[:16]):  # cap prompt size
        ns = []
        for nid, r in zip(he.node_ids, he.node_roles):
            n = instance.nodes.get(nid)
            ns.append(f"{r}:{n.surface if n else nid}")
        ev = he.evidence_span[:100]
        lines.append(f"  {i}: [{','.join(ns)}] ev=\"{ev}\"")
    prompt = SEMANTIC_CLUSTER_PROMPT.format(pattern_id=pattern_id, edges_block="\n".join(lines))
    raw = _call(prompt, llm, max_tokens=800)
    parsed = parse_json_response(raw)
    if not isinstance(parsed, list):
        return [], "llm_semantic_fail"
    clusters = []
    for grp in parsed:
        if isinstance(grp, dict) and isinstance(grp.get("edge_ids"), list):
            ids = [int(x) for x in grp["edge_ids"] if isinstance(x, (int, str)) and str(x).lstrip("-").isdigit()]
            ids = [x for x in ids if 0 <= x < len(edges)]
            if ids:
                clusters.append(ids)
    # dedup overlapping / drop empties; ensure partition (each edge once)
    seen = set()
    clean = []
    for c in clusters:
        uniq = [i for i in c if i not in seen]
        if uniq:
            clean.append(uniq)
            seen.update(uniq)
    # leftover edges (not grouped) -> residual
    leftover = [i for i in range(min(len(edges), 16)) if i not in seen]
    if leftover:
        clean.append(leftover)
    return clean, "llm_semantic"


def detect_split_triggers(meta: MetaHypergraph, instance: InstanceHypergraph,
                          prefer: str = "auto", llm: str = "deepseek") -> list[dict]:
    """Scan all active (non-deprecated) patterns; return one trigger per
    pattern whose instances fall into >=2 clusters each of size >= MIN_CLUSTER.
    Each trigger: {pattern_id, clusters: [[edge_idx,...]], method, n_edges,
    cluster_sizes}. This is the DETERMINISTIC entry point — run_split then
    asks the LLM only to NAME the clusters, not to decide whether to split.

    Tier order: (1) discrete qualifier -> (2) embedding -> (3) LLM semantic
    grouping (only for patterns with STRUCTURAL SIGNAL — equations/formulas in
    evidence; pure-prose patterns like 'influences' are too broad to split
    meaningfully and the LLM groups by surface wording, producing misnamed
    catch-all sub-patterns. Such patterns evolve via add_pattern instead)."""
    # group instance hyperedges by their pattern_type
    by_pat: dict[str, list[Hyperedge]] = {}
    for he in instance.hyperedges.values():
        # only edges of active patterns are split candidates
        pat = meta.patterns.get(he.pattern_type)
        if pat is None or pat.deprecated:
            continue
        by_pat.setdefault(he.pattern_type, []).append(he)
    triggers = []
    for pid, edges in by_pat.items():
        if len(edges) < 2 * MIN_CLUSTER:
            continue  # not enough to form two real clusters
        clusters, method = cluster_pattern_instances(edges, instance, meta, prefer=prefer)
        big = [c for c in clusters if len(c) >= MIN_CLUSTER]
        # tier 3 fallback: LLM semantic grouping — only if the resulting
        # clusters pass a COHERENCE check (DIAL-KG style: a split is only
        # valid if each sub-cluster has high internal semantic coherence;
        # low-coherence clusters mean the pattern is too broad to split
        # meaningfully and the LLM grouped by surface wording, not relation
        # nature — those evolve via add_pattern instead).
        if len(big) < 2 and llm:
            lc, lmethod = _llm_semantic_clusters(edges, instance, pid, llm=llm)
            lbig = [c for c in lc if len(c) >= MIN_CLUSTER]
            # coherence gate: each cluster's edges must be internally coherent
            # (mean pairwise cosine >= SPLIT_COHERENCE_THRESH) for the split
            # to be meaningful. Drops catch-all clusters of unrelated edges.
            if len(lbig) >= 2 and _clusters_coherent(edges, lbig, instance):
                big, method = lbig, lmethod
        if len(big) >= 2:
            triggers.append({
                "pattern_id": pid,
                "n_edges": len(edges),
                "clusters": big,
                "cluster_sizes": [len(c) for c in big],
                "method": method,
                # representative evidence per cluster for the naming LLM:
                # show up to 3 edges per cluster (not just the first) so the
                # LLM names from the cluster's shared relation, not one edge's
                # wording — reduces misnamed catch-all sub-patterns.
                "representatives": [
                    " | ".join(edges[i].evidence_span[:100] for i in c[:3])
                    for c in big
                ],
            })
    return triggers


# Coherence threshold for split clusters (DIAL-KG style). A split is only
# valid if each resulting sub-cluster's edges are internally coherent —
# mean pairwise embedding cosine >= this. Lower = looser clustering (more
# splits accepted, but risk catch-all sub-patterns); higher = stricter
# (fewer splits, only when sub-groups are genuinely tight). 0.55 matches
# the embedding single-link threshold used elsewhere.
SPLIT_COHERENCE_THRESH = 0.55


def _clusters_coherent(edges, clusters, instance) -> bool:
    """DIAL-KG-style coherence gate: every cluster must have mean pairwise
    embedding cosine >= SPLIT_COHERENCE_THRESH. A cluster of unrelated edges
    (e.g. 21 mixed influences edges dumped together) fails this and the split
    is rejected — the pattern is too broad to split meaningfully and should
    evolve via add_pattern instead. Degrades to True (allow) if embedding
    unavailable (honest: don't block splits on infra failure)."""
    try:
        texts = [_edge_cluster_text(edges[i], instance) for c in clusters for i in c]
        if not texts:
            return True
        embs = _embed_texts_robust(texts)
        if embs is None or len(embs) != len(texts):
            return True  # degrade: don't block on embed failure
        import numpy as np
        A = np.asarray(embs, dtype=np.float32)
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        sim = A @ A.T
        idx = 0
        for c in clusters:
            n = len(c)
            if n < 2:
                idx += n
                continue
            sub = sim[idx:idx+n, idx:idx+n]
            # mean pairwise cosine (off-diagonal)
            mask = ~np.eye(n, dtype=bool)
            mean_cos = sub[mask].mean() if n > 1 else 1.0
            if mean_cos < SPLIT_COHERENCE_THRESH:
                return False  # this cluster is incoherent -> reject split
            idx += n
        return True
    except Exception:
        return True


SPLIT_NAMING_PROMPT = """You are naming the result of a SCHEMA SPLIT. A hyperedge pattern '{parent_id}' (description: {parent_desc}) was found to be over-wide: its instances fall into {k} clusters with distinct semantics. The split decision was made by deterministic clustering (method: {method}); your job is ONLY to NAME the sub-patterns, NOT to judge whether the split is correct.

Representative evidence for each cluster:
{cluster_evidence}

Existing sub-patterns of '{parent_id}' already in the schema (REUSE one of these if a cluster's semantics matches — do NOT mint a new name that means the same thing as an existing one; cross-paper name divergence is a known failure mode we must avoid):
{existing_subpatterns}

CRITICAL NAMING RULE — name from the RELATION'S NATURE, not from the entities:
The sub-pattern name must capture HOW the relation works (the verb/mechanism in the evidence), NOT WHAT entities are involved. Read the evidence verb phrase and name the sub-pattern after it.

GOOD names (suffix reflects the relation nature, derived from the evidence verb):
  {parent_id}_parameter_dependency   (evidence: "u depends on h and θ" — relation is parameter dependence)
  {parent_id}_force_balance          (evidence: "flow controlled by balance of gravity and friction" — force balance)
  {parent_id}_threshold_condition    (evidence: "flow occurs only above critical inclination" — threshold)
  {parent_id}_geometric_scaling      (evidence: "u scales with sqrt(g h)" — scaling law)
  {parent_id}_kinematic_relation     (evidence: "velocity field relates to shear rate" — kinematics)

BAD names (entity/loose-association names — DO NOT USE):
  {parent_id}_performance_enhancement  (WRONG: the edges are parameter dependencies, not "performance enhancement")
  {parent_id}_granular_chute          (WRONG: names the entity "granular chute", not the relation nature)
  {parent_id}_experimental            (WRONG: names the context, not the relation)
  {parent_id}_system_response         (WRONG: too vague, doesn't say how)

For each cluster, read its evidence, identify the RELATION VERB/MECHANISM (depends on / balances / scales with / thresholds at / induces / constrains / ...), and name the sub-pattern after that mechanism.

For each cluster, propose:
- pattern_id: a lowercase snake_case name, derived from {parent_id} + the relation mechanism (e.g. {parent_id}_parameter_dependency). Use the SAME case as {parent_id}. Each must be distinct. PREFER reusing an existing sub-pattern name from the list above when the cluster's semantics fit; only mint a new name when no existing one matches. The suffix should be the PRIMARY mechanism word from the evidence (<=30 chars); if a secondary mechanism matters, express it via qualifier, not a longer name.
- description: one sentence capturing what distinguishes this cluster's relation, citing the evidence verb.
- allowed_qualifiers (OPTIONAL): a list of qualifier keys this sub-pattern uses, drawn from the parent's set. Omit to inherit the parent's full set. A specialization may use a subset (e.g. a power-law sub-pattern needs function_form but not relation_type).

Output ONLY a JSON array of {k} objects:
[{{"pattern_id":"...","description":"..."}}]"""


def name_split_subpatterns(parent: MetaHyperedgePattern, trigger: dict,
                           meta: "MetaHypergraph" = None, llm: str = "deepseek") -> list[dict]:
    """LLM naming-only step. The split decision is already made (deterministic
    clustering); the LLM just labels the sub-patterns. This keeps the A4
    circularity broken: the LLM never decides WHETHER to split.

    If meta is provided, existing sub-patterns of the parent (same family,
    already evolved in earlier papers) are fed to the naming LLM so it REUSES
    names instead of minting divergent new ones per paper — the cross-paper
    name-divergence bug that shattered silhouette."""
    k = len(trigger["clusters"])
    ce = "\n".join(f"  Cluster {i+1}: \"{ev}\"" for i, ev in enumerate(trigger["representatives"]))
    # gather existing sub-patterns: descendants of parent, or same-family
    # evolved patterns whose id starts with parent_id (the split lineage)
    existing = []
    if meta is not None:
        for pid, pat in meta.active_patterns().items():
            if pid == parent.pattern_id or pat.deprecated:
                continue
            # lineage: name derived from parent (starts with parent_id + '_')
            # OR shares the parent's family
            if pid.startswith(parent.pattern_id + "_") or getattr(pat, "family", "") == getattr(parent, "family", "") and pid != parent.pattern_id:
                existing.append(f"- {pid}: {pat.description[:120]}")
    existing_str = "\n".join(existing) if existing else "(none yet — this is the first split of this pattern)"
    prompt = SPLIT_NAMING_PROMPT.format(
        parent_id=parent.pattern_id, parent_desc=parent.description,
        k=k, method=trigger["method"], cluster_evidence=ce,
        existing_subpatterns=existing_str)
    raw = _call(prompt, llm, max_tokens=600)
    parsed = parse_json_response(raw)
    if not isinstance(parsed, list) or len(parsed) != k:
        return []  # naming failed -> caller skips this split (honest, no auto-name)
    out = []
    for p in parsed:
        if not (isinstance(p, dict) and p.get("pattern_id")):
            continue
        # case-normalize to the seed's lowercase snake_case form. The naming
        # prompt says UPPER_SNAKE but the seed uses lowercase; deepseek emits
        # BOTH forms run-to-run, creating case-duplicate sub-patterns
        # (constitutive_law_power_law vs CONSTITUTIVE_LAW_POWER_LAW) that the
        # same-run merge gate catches unreliably. Normalizing at the source
        # prevents the dup rather than detecting it after.
        pid = str(p["pattern_id"]).strip().strip("`\"' ")
        pid = pid.lower()
        p["pattern_id"] = pid
        out.append(p)
    return out


# ===========================================================================
# RENAME trigger (the 5th bounded op, auto-triggered). Deterministic:
# flags evolved patterns whose id is an unwieldy concatenation (longer than
# RENAME_LEN_THRESH AND >=2 underscores — the suffix-stitched bloat signal,
# e.g. influences_causal_result_positive_mechanism_methodological_factor) OR
# an UPPER_SNAKE name the LLM emitted despite the lowercase instruction.
# The LLM only proposes a short replacement name (labeling, not judgment);
# rename_pattern re-keys + rewrites all references. A4 circularity preserved.
# ===========================================================================

RENAME_LEN_THRESH = 45   # chars — a pattern id longer than this is unwieldy


def detect_rename_triggers(meta: MetaHypergraph) -> list[dict]:
    """Find active evolved patterns whose id is unwieldy (long+stitched) or
    UPPER_SNAKE. Returns one trigger per such pattern. Skips seed patterns
    (short, lowercase) and abstract parents (keep their lineage name)."""
    seeded = {"measures", "constitutive_law", "influences", "defines",
              "composed_of", "claim_relation",
              "MATERIAL", "PROPERTY", "NUMERIC", "REGIME"}
    triggers = []
    for pid, pat in meta.active_patterns().items():
        if pid in seeded:
            continue
        if pat.is_abstract:
            continue  # keep abstract-parent lineage name
        is_upper = pid.isupper() and len(pid) > 4
        is_long = len(pid) >= RENAME_LEN_THRESH and pid.count("_") >= 2
        if is_upper or is_long:
            triggers.append({"pattern_id": pid, "reason": "upper" if is_upper else "long_stitched",
                              "len": len(pid)})
    return triggers


def _propose_rename(pat: MetaHyperedgePattern, llm: str = "deepseek") -> str | None:
    """LLM naming-only: propose a short lowercase snake_case replacement for
    an unwieldy pattern id. The rename decision is already made (deterministic
    detect); the LLM just picks a clean short name. Returns the name or None."""
    prompt = f"""A hyperedge pattern has an UNWIELDY id that should be shortened to a clean, memorable lowercase snake_case name (the rename decision is made; you ONLY pick the new name, do NOT judge whether to rename).

Current id: {pat.pattern_id}
Description: {pat.description}

Propose ONE short lowercase snake_case replacement (<=30 chars, derived from the description's core relation). Keep it general enough to match the description. Output ONLY the id, no explanation."""
    raw = _call(prompt, llm, max_tokens=40)
    if not raw:
        return None
    name = raw.strip().split()[0].strip("`\"',. ").lower()
    return name if name and name.replace("_", "").isalnum() and name != pat.pattern_id else None


def run_rename(meta: MetaHypergraph, llm: str = "deepseek") -> list[dict]:
    """Apply renames detected by detect_rename_triggers. Returns one record
    per rename. No-op if naming fails (honest, keeps the unwieldy id)."""
    triggers = detect_rename_triggers(meta)
    applied = []
    for t in triggers:
        pat = meta.patterns.get(t["pattern_id"])
        if not pat or pat.deprecated:
            continue
        new = _propose_rename(pat, llm=llm)
        if not new or new in meta.patterns:
            applied.append({"from": t["pattern_id"], "skipped": True,
                            "reason": "naming failed or target taken"})
            continue
        new_ver = meta.rename_pattern(t["pattern_id"], new,
                                      evidence=f"unwieldy id ({t['reason']}, len {t['len']})")
        if new_ver:
            applied.append({"from": t["pattern_id"], "into": new,
                            "reason": t["reason"], "version": new_ver})
    return applied


# ===========================================================================
# REDESIGN v2 step 2: pattern-level dependency inference (schema富拓扑).
# Infer pattern_dependencies FROM INSTANCES (deterministic graph reachability,
# no LLM — A4 preserved). If pattern A's edge references a node that pattern B
# (a definition-family pattern) also references => A depends_on B (A uses an
# entity B defines). This builds the schema-layer topology that DIAL-KG's
# flat schema lacks, and enables detect_constraint_violations (DIAL-KG-can't-do).
# ===========================================================================

def infer_pattern_dependencies(meta: MetaHypergraph, instance: InstanceHypergraph,
                               paper_id: str) -> list[dict]:
    """Infer pattern-level dependency edges from instance co-occurrence.
    For each pair (pattern_a, pattern_b) where pattern_b is a definition-family
    pattern: if a node in an a-edge also appears in a b-edge, then a depends_on
    b (a references an entity b defines). Deterministic."""
    # node_surface -> set of patterns referencing it
    node_pats: dict[str, set[str]] = {}
    for he in instance.hyperedges.values():
        for nid in he.node_ids:
            n = instance.nodes.get(nid)
            if n:
                node_pats.setdefault(n.surface, set()).add(he.pattern_type)
    applied = []
    for surface, pats in node_pats.items():
        definers = {p for p in pats if p in meta.patterns
                    and meta.patterns[p].family == "definition" and not meta.patterns[p].deprecated}
        if not definers:
            continue
        # any non-definition pattern referencing this node depends_on its definers
        for a in pats:
            if a in definers:
                continue
            pat_a = meta.patterns.get(a)
            if not pat_a or pat_a.deprecated:
                continue
            for b in definers:
                if a == b:
                    continue
                nv = meta.add_pattern_dependency(a, b, rel="depends_on",
                                                 evidence=f"node '{surface}' co-referenced",
                                                 paper_id=paper_id)
                if nv:
                    applied.append({"dependent": a, "depends_on": b,
                                    "via_node": surface, "version": nv})
    return applied


# ===========================================================================
# REDESIGN v2 step 3: pattern-level CONSTRAINT + COMPOSITION inference.
#
# Two more schema-layer topology edges, inferred FROM INSTANCES (deterministic,
# graph reachability, no LLM — A4 preserved). They differ from depends_on in
# DIRECTION + the family roles they apply to, so they are NOT redundant with it:
#
#   depends_on (A -> B):  A references an entity B DEFINES.        B = producer.
#                           A cannot exist without B. (definition is authority.)
#   constrains   (A -> B): A (a law/definition) GOVERNS B (a measure/claim).
#                           B must conform to A.                   A = authority.
#   composes      (A -> B): A (a composition) INCLUDES B's entity as a part.
#                           A structurally contains B.             A = whole.
#
# All three are co-occurrence-based but pick out different relations by which
# family plays which role. This is what makes the schema layer a directed
# constrained hypergraph (DIAL-KG's flat schema has none of these).
# ===========================================================================

# Authority families: patterns that FIX a functional relation among quantities.
# NARROWED after real-paper inspection (see DECISION-constraint-trigger-scope.md):
# only constitutive_law constrains — definitions just NAME an entity, and
# "definition A constrains dependency B" would be the reverse of depends_on
# (B depends_on A), i.e. two edges on one pair saying the same co-occurrence
# (redundant). A LAW, by contrast, fixes the functional FORM among quantities,
# which genuinely constrains the dependencies among them — distinct from
# depends_on (about definitions, not laws).
AUTHORITY_FAMILIES = ("constitutive_law",)
# Consumer families: relations whose form is bounded by a law (a dependency
# among quantities the law relates; a measure validated against the law; a
# claim scoped by the law).
CONSUMER_FAMILIES = ("dependency", "measure", "claim")


def infer_pattern_constraints(meta: MetaHypergraph, instance: InstanceHypergraph,
                               paper_id: str) -> list[dict]:
    """Infer pattern-level CONSTRAINT edges from instance co-occurrence.
    A constrains B when A is an AUTHORITY (definition/constitutive_law family)
    and B is a CONSUMER (measure/claim family) and they reference the same node
    surface — the law governs the quantity the measure/claim uses. Deterministic.

    Direction (authority -> constrains -> consumer) is the OPPOSITE of depends_on
    (consumer -> depends_on -> producer): constrains says who GOVERNS, depends_on
    says who NEEDS. A pair can carry both when the roles warrant it; they are
    not mutually exclusive, just different relations."""
    # surface -> set of patterns referencing it
    node_pats: dict[str, set[str]] = {}
    for he in instance.hyperedges.values():
        for nid in he.node_ids:
            n = instance.nodes.get(nid)
            if n:
                node_pats.setdefault(n.surface, set()).add(he.pattern_type)
    applied = []
    for surface, pats in node_pats.items():
        authorities = {p for p in pats if p in meta.patterns
                       and meta.patterns[p].family in AUTHORITY_FAMILIES
                       and not meta.patterns[p].deprecated}
        if not authorities:
            continue
        consumers = {p for p in pats if p in meta.patterns
                     and meta.patterns[p].family in CONSUMER_FAMILIES
                     and not meta.patterns[p].deprecated}
        if not consumers:
            continue
        # each authority constrains each consumer that shares this node
        for a in authorities:
            for b in consumers:
                if a == b:
                    continue
                nv = meta.add_pattern_dependency(a, b, rel="constrains",
                                                 evidence=f"node '{surface}' governed by authority + used by consumer",
                                                 paper_id=paper_id)
                if nv:
                    applied.append({"authority": a, "constrains": b,
                                    "via_node": surface, "version": nv})
    return applied


def infer_pattern_compositions(meta: MetaHypergraph, instance: InstanceHypergraph,
                                paper_id: str) -> list[dict]:
    """Infer pattern-level COMPOSITION edges from instance co-occurrence.
    A composes B when A is a COMPOSITION-family pattern (a whole made of parts)
    and B is any other pattern referencing a node A connects — A's whole
    includes an entity B also relates/defines. Deterministic.

    Direction (whole -> composes -> part): A is the composite relation, B is
    one of its components' patterns. This captures 'a constitutive relation is
    composed of a dependency + a definition' style structure at the pattern
    level, which depends_on (consumer->producer) and constrains (authority->
    consumer) do not."""
    # surface -> set of patterns referencing it
    node_pats: dict[str, set[str]] = {}
    for he in instance.hyperedges.values():
        for nid in he.node_ids:
            n = instance.nodes.get(nid)
            if n:
                node_pats.setdefault(n.surface, set()).add(he.pattern_type)
    applied = []
    for surface, pats in node_pats.items():
        wholes = {p for p in pats if p in meta.patterns
                  and meta.patterns[p].family == "composition"
                  and not meta.patterns[p].deprecated}
        if not wholes:
            continue
        for w in wholes:
            for b in pats:
                if b == w:
                    continue
                pat_b = meta.patterns.get(b)
                if not pat_b or pat_b.deprecated:
                    continue
                nv = meta.add_pattern_dependency(w, b, rel="composes",
                                                 evidence=f"node '{surface}' is a component of a composition + referenced by another pattern",
                                                 paper_id=paper_id)
                if nv:
                    applied.append({"whole": w, "composes": b,
                                    "via_node": surface, "version": nv})
    return applied


def run_split(meta: MetaHypergraph, instance: InstanceHypergraph, paper_id: str,
              llm: str = "deepseek", prefer: str = "auto") -> list[dict]:
    """Apply pattern-level splits detected by detect_split_triggers.
    Returns one record per split: {pattern_id, sub_patterns, method,
    reattributed, version}. Historical hyperedges of the split pattern are
    re-attributed to their cluster's sub-pattern (intra-paper retrace — the
    cross-paper retrace is a separate step)."""
    triggers = detect_split_triggers(meta, instance, prefer=prefer, llm=llm)
    applied = []
    for t in triggers:
        parent = meta.patterns.get(t["pattern_id"])
        if not parent or parent.deprecated:
            continue
        names = name_split_subpatterns(parent, t, meta=meta, llm=llm)
        if len(names) != len(t["clusters"]):
            applied.append({"pattern_id": t["pattern_id"], "skipped": True,
                            "reason": "naming failed", "method": t["method"]})
            continue
        # build sub-patterns: inherit parent role_slots (structure must match
        # for IS-A); qualifiers inherit parent UNLESS the LLM proposed a
        # tighter/relaxed set for this sub-pattern (a specialization may carry
        # different qualifiers than its generalization — e.g. a power-law
        # sub-pattern needs function_form but not relation_type).
        subs = []
        for nm in names:
            sub_quals = nm.get("allowed_qualifiers")
            if not isinstance(sub_quals, list) or not sub_quals:
                sub_quals = list(parent.allowed_qualifiers)
            else:
                # LLM-proposed qualifier set must still be a subset of the
                # registry (controlled extension — no ad-hoc keys on a split).
                sub_quals = [q for q in sub_quals if q in QUALIFIER_REGISTRY] or list(parent.allowed_qualifiers)
            subs.append(MetaHyperedgePattern(
                pattern_id=nm["pattern_id"], description=nm.get("description", ""),
                role_slots=[dict(s) for s in parent.role_slots],
                allowed_qualifiers=sub_quals))
        new_ver = meta.split_pattern(parent.pattern_id, subs,
                                      evidence="; ".join(t["representatives"][:2]),
                                      paper_id=paper_id)
        if not new_ver:
            continue
        # re-attribute historical hyperedges: each cluster's edges get their
        # sub-pattern as pattern_type. (Deprecated parent kept for provenance.)
        # IMPORTANT: use detect's clusters DIRECTLY (t["clusters"]) — re-running
        # cluster_pattern_instances(prefer="auto") would walk discrete/embedding
        # tiers and NOT reproduce the LLM-semantic grouping that triggered the
        # split, scattering edges wrong (a prior bug put all edges in 1 sub-pattern).
        edges = [he for he in instance.hyperedges.values()
                if he.pattern_type == parent.pattern_id]
        n_reattrib = 0
        big2 = t["clusters"]  # detect's already-sizeable clusters, in order
        for ci, sub in zip(range(len(big2)), subs):
            for idx in big2[ci]:
                if idx < len(edges):
                    edges[idx].pattern_type = sub.pattern_id  # re-attribute
                    n_reattrib += 1
        applied.append({"pattern_id": parent.pattern_id,
                        "sub_patterns": [s.pattern_id for s in subs],
                        "method": t["method"], "cluster_sizes": t["cluster_sizes"],
                        "reattributed": n_reattrib, "version": new_ver})
    return applied


# ===========================================================================
# MERGE + RETIRE triggers (DIAL-KG's operations, reimplemented on the
# hypergraph schema — cited as prior, NOT claimed as novel).
#
# Deterministic (embedding canonicalization, same as DIAL-KG's principle):
# two ACTIVE patterns with the SAME role-structure and embedding cosine
# >= MERGE_THRESHOLD are near-duplicate bloat -> merge. The LLM only names
# the merged pattern (labeling, not judgment) — A4 circularity preserved.
#
# RETIRE: a pattern whose instances were ALL re-attributed away (e.g. by a
# split that fully subsumed it, leaving 0 live instances) is soft-deprecated.
# This is the natural completion of split: once every edge of a pattern has
# moved to sub-patterns, the parent is retired. (DIAL-KG soft-deprecate.)
# ===========================================================================

MERGE_THRESHOLD = 0.85   # tried 0.80 -> 50 merge pairs, mostly false merges
                         # (claim_relation ~ closure_relation cos 0.81 — same
                         # structure but genuinely different). pattern_id name
                         # embedding is unreliable for merge: near-name !=
                         # same-relation. 0.85 (1 true pair) is conservative-but-
                         # honest; merge cross-paper is hard with id-name embed.
MERGE_THRESHOLD_DIFF_STRUCT = 0.90  # different role-structure pairs need higher


def detect_merge_triggers(meta: MetaHypergraph) -> list[dict]:
    """Find pairs of active patterns that are semantic near-duplicates. Same
    role-structure + cosine >= MERGE_THRESHOLD, OR different structure but
    cosine >= MERGE_THRESHOLD_DIFF_STRUCT (cross-paper reinvention with slightly
    different role naming). Returns one trigger per pair."""
    active = meta.active_patterns()
    pids = list(active.keys())
    if len(pids) < 2:
        return []
    cache = _ensure_pattern_embeds(meta)
    if not cache or len(cache) < 2:
        return []  # embedding unavailable -> honest skip (no silent merge)
    triggers = []
    seen_pairs = set()
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            pa, pb = active[pids[i]], active[pids[j]]
            # taxonomy guard: do not pair an abstract parent with its OWN
            # descendant (they are IS-A related, not duplicates — a parent
            # generalizes the child by design, embedding-similar by construction).
            if meta.is_subtype(pids[i], pids[j]) or meta.is_subtype(pids[j], pids[i]):
                continue
            # do not pair an abstract parent with a concrete leaf at all
            # (different ontological rank; merge_patterns will reject, but
            # skip here to avoid wasted embed + a spurious trigger record).
            if pa.is_abstract != pb.is_abstract:
                continue
            same_struct = (_role_sig(pa) == _role_sig(pb))
            ea, eb = cache.get(pids[i]), cache.get(pids[j])
            if not ea or not eb:
                continue
            s = cosine_sim(ea, eb)
            thr = MERGE_THRESHOLD if same_struct else MERGE_THRESHOLD_DIFF_STRUCT
            if s >= thr:
                key = tuple(sorted([pids[i], pids[j]]))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                triggers.append({"patterns": [pids[i], pids[j]],
                                 "cosine": round(s, 3),
                                 "role_sig": _role_sig(pa)})
    return triggers


def _name_merged_pattern(pa: MetaHyperedgePattern, pb: MetaHyperedgePattern,
                         llm: str = "deepseek") -> str | None:
    """LLM naming-only: propose ONE merged pattern_id from two near-dup
    patterns. The merge decision is already made (deterministic embedding);
    the LLM just picks a clean name. Returns the name or None."""
    prompt = f"""Two hyperedge patterns are near-duplicates (same role-structure, high semantic similarity) and should be merged. Your job is ONLY to name the merged pattern, NOT to judge whether they should merge (that decision is made).

Pattern A: {pa.pattern_id} — {pa.description}
Pattern B: {pb.pattern_id} — {pb.description}

Propose ONE short UPPER_SNAKE pattern_id for the merged pattern (derived from both, e.g. if A=measures and B=quantifies -> MEASURES). Output ONLY the id, no explanation."""
    raw = _call(prompt, llm, max_tokens=40)
    if not raw:
        return None
    name = raw.strip().split()[0].strip("`\"',. ")
    # basic sanity + case-fold to the seed's lowercase snake_case form (the
    # naming prompt says UPPER_SNAKE but the seed is lowercase; deepseek emits
    # both forms run-to-run, creating case-dups. add_pattern + split naming
    # already fold; merge naming must too or a merged survivor lands UPPERCASE).
    name = name.lower()
    return name if name and name.replace("_", "").isalnum() else None


def run_merge(meta: MetaHypergraph, instance: InstanceHypergraph, paper_id: str,
              llm: str = "deepseek") -> list[dict]:
    """Apply merges detected by detect_merge_triggers. Each merged-away pattern's
    historical hyperedges are re-attributed to the kept pattern (intra-paper
    retrace). Returns one record per merge."""
    triggers = detect_merge_triggers(meta)
    applied = []
    for t in triggers:
        pa_id, pb_id = t["patterns"]
        pa, pb = meta.patterns.get(pa_id), meta.patterns.get(pb_id)
        if not pa or not pb or pa.deprecated or pb.deprecated:
            continue
        # name the merged pattern (LLM labeling only)
        merged_name = _name_merged_pattern(pa, pb, llm=llm)
        if not merged_name:
            applied.append({"patterns": t["patterns"], "skipped": True,
                            "reason": "naming failed", "cosine": t["cosine"]})
            continue
        # keep merged_name as the survivor if it's new, else merge into pa
        into = merged_name if merged_name not in meta.patterns else pa_id
        new_ver = meta.merge_patterns([pa_id, pb_id], into=into,
                                       evidence=f"cosine {t['cosine']}", paper_id=paper_id)
        if not new_ver:
            continue
        # re-attribute historical edges: both old patterns' edges -> survivor
        n_reattrib = 0
        for he in instance.hyperedges.values():
            if he.pattern_type in (pa_id, pb_id):
                he.pattern_type = into
                n_reattrib += 1
        applied.append({"merged": [pa_id, pb_id], "into": into,
                        "cosine": t["cosine"], "reattributed": n_reattrib,
                        "version": new_ver})
    return applied


def run_retire(meta: MetaHypergraph, instance: InstanceHypergraph,
               paper_id: str) -> list[dict]:
    """Retire patterns that became empty AFTER a split/merge pass.

    Two retire cases (both deterministic):
    1. CONCRETE orphan: an evolved concrete pattern (split_from set, not
       seeded, not abstract) with zero live instances (all its edges were
       re-attributed to a sub-pattern by a recursive split).
    2. ABSTRACT empty-parent: an abstract parent whose descendant subtree is
       ENTIRELY gone (every child deprecated/retired/merged-away) — a
       "bare" generalization with nothing to generalize. This is the
       taxonomy's natural cleanup: once all children leave, the parent retires.
       (An abstract parent WITH active children is NEVER retired — 0 direct
       instances is the taxonomy working, not an orphan.)

    Never retired: seeded patterns (seed absent from one paper is normal);
    abstract parents with >=1 active descendant."""
    seeded = {"measures", "constitutive_law", "claim_relation",
              "MATERIAL", "PROPERTY", "NUMERIC", "REGIME"}
    live: dict[str, int] = {}
    for he in instance.hyperedges.values():
        live[he.pattern_type] = live.get(he.pattern_type, 0) + 1
    applied = []
    for pid, pat in list(meta.patterns.items()):
        if pat.deprecated:
            continue
        if pid in seeded:
            continue  # seed absent from one paper is normal, not retirement
        if not pat.is_abstract:
            # case 1: concrete evolved orphan with 0 live instances
            if live.get(pid, 0) == 0 and pat.split_from:
                new_ver = meta.retire_pattern(pid, evidence="zero live instances after repair",
                                              paper_id=paper_id)
                if new_ver:
                    applied.append({"pattern_id": pid, "retired": True,
                                    "reason": "zero instances (post-repair orphan)",
                                    "version": new_ver})
            continue
        # case 2: abstract parent — retire only if NO active descendant remains
        descendants = meta.pattern_subclasses(pid)
        active_desc = [d for d in descendants
                       if d in meta.patterns and not meta.patterns[d].deprecated]
        if not active_desc:
            new_ver = meta.retire_pattern(pid, evidence="empty-parent (all descendants gone)",
                                          paper_id=paper_id)
            if new_ver:
                applied.append({"pattern_id": pid, "retired": True,
                                "reason": "empty abstract parent (no active descendants)",
                                "version": new_ver})
    return applied


# ---- deterministic instance consolidation (post-extraction cleanup) -------
# Verified on 5 ARFM papers: METHOD labels were garbage-heavy (tools,
# geometries, equation refs, paper titles, generic phrases) and the same
# method had duplicate surfaces (NGF x5). These deterministic rules clean
# METHOD labels and merge duplicate method surfaces so rich-topology edges
# (method_parameter / method_phenomenon / method_regime) carry real method
# nodes. Idempotent. See DECISION_rich_topo_consolidation.md.
_TOOL_RE = re.compile(
    r'^(experiments?|discrete particle simulations?|numerical simulations?|'
    r'(finite|discrete) (element|difference) method|particle-image velocimetry|PIV|'
    r'MRI|X-ray (tomography|CT|imaging)|contact dynamics|DEM\b|CFD|direct simulation|'
    r'rheometer|laser sheet|image processing system|front tracking|high-speed camera|'
    r'particle tracking|digital image correlation|tomography|photogrammetry)$', re.I)
_GEO_RE = re.compile(
    r'^(plane shear|annular (shear|couette)|heap flow|rotating drum|silo flow?|'
    r'chute|inclined plane|couette flow|hopper flow?|annular shear flow|'
    r'(simple|pure) shear|3d annular|cylindrical couette|ring shear)$', re.I)
_GROUP_RE = re.compile(r'GDR\s*MiDi', re.I)
_GENERIC_RE = re.compile(
    r'^(unified framework|future model(\s+for.+)?|constitutive equations|'
    r'the equations|a model for.+|model for (dense|granular).+|'
    r'theoretical (approaches?|model|framework)|empirical fit(\s?\d)?|'
    r'empirical (law|relation|equation)s?|numerical (approach|method|model)|'
    r'constitutive model|continuum model|theoretical approaches?)$', re.I)
_EFFECT_RE = re.compile(
    r'(Bagnold[- ]like|Janssen effect|constitutive law for.+|effective friction law for.+|'
    r'profile$|velocity profile|dilatancy|Reynolds dilation)', re.I)
_EQREF_RE = re.compile(r'^eq(uation|\.)?\s*\.?\s*\(?[\dabivx]+\)?\.?$', re.I)
# Long surfaces are dropped as METHOD ONLY when they look like a sentence/paper-title,
# NOT a descriptive method phrase. A descriptive noun phrase like "gradient expansion
# of the yield parameter" (39 chars) is a real method concept — keep it. Only drop when
# the surface contains a verb/connective pattern typical of a full sentence or title.
_LONG_TITLE_RE = re.compile(
    r'^(.+\s(.of|from|into|towards|using|by|via|based on|in order to|allows|enables|shows|predicts|describes|proposes|introduces)\s.+'
    r'|.+(stopping|arrest|flow|flows|media|materials?)\s(down|in|on|across|through)\s.+)$',
    re.I)


def _relabel(n, new):
    if new == "drop":
        n.labels = [l for l in n.labels if l != "METHOD"]
    elif new in ("PROPERTY", "MATERIAL"):
        n.labels = [l for l in n.labels if l != "METHOD"]
        if new not in n.labels:
            n.labels.append(new)


def _norm_surface(s):
    s = s.lower()
    s = s.replace('μ', 'mu').replace('ξ', 'xi').replace('γ', 'g')
    s = re.sub(r'[^a-z0-9]', '', s)
    for suf in ('model', 'theory', 'approach', 'formulation', 'framework',
                'relation', 'equations'):
        s = s.replace(suf, '')
    return s


def consolidate_instance(instance: InstanceHypergraph) -> dict:
    """Deterministic post-extraction cleanup (idempotent):
    1. relabel garbage METHOD nodes (tools/geometries/generic/equation-refs/
       long titles) -> PROPERTY/MATERIAL or drop METHOD.
    2. merge duplicate METHOD surfaces (suffix variations); never merge
       'local' with 'nonlocal'.
    Mutates the instance in place. Returns a change summary."""
    n_relabel = 0
    for n in instance.nodes.values():
        if "METHOD" not in n.labels:
            continue
        s = n.surface.strip()
        if _GROUP_RE.search(s):
            _relabel(n, "MATERIAL"); n_relabel += 1
        elif _EQREF_RE.match(s):
            _relabel(n, "drop"); n_relabel += 1
        elif _LONG_TITLE_RE.match(s):
            _relabel(n, "drop"); n_relabel += 1
        elif _TOOL_RE.match(s):
            _relabel(n, "PROPERTY"); n_relabel += 1
        elif _GEO_RE.match(s):
            _relabel(n, "PROPERTY"); n_relabel += 1
        elif _EFFECT_RE.search(s):
            _relabel(n, "PROPERTY"); n_relabel += 1
        elif _GENERIC_RE.match(s):
            _relabel(n, "drop"); n_relabel += 1
    # dedup duplicate method surfaces
    method_nodes = [(nid, n) for nid, n in instance.nodes.items() if "METHOD" in n.labels]
    groups = {}
    for nid, n in method_nodes:
        groups.setdefault(_norm_surface(n.surface), []).append((nid, n.surface))
    merged = {}
    norms = list(groups.keys())
    for i, ni in enumerate(norms):
        for nj in norms:
            if ni == nj or len(ni) < 4 or len(nj) < 4:
                continue
            if ('nonlocal' in ni) != ('nonlocal' in nj):
                continue  # local vs nonlocal must NOT merge
            if ni in nj or nj in ni:
                canon_norm = ni if len(ni) <= len(nj) else nj
                canon = groups[canon_norm][0][0]
                for nidj, _ in groups[nj]:
                    if nidj != canon:
                        merged[nidj] = canon
    n_dedup = 0
    if merged:
        for he in instance.hyperedges.values():
            he.node_ids = [merged.get(x, x) for x in he.node_ids]
        for nid in list(merged.keys()):
            if nid in instance.nodes:
                del instance.nodes[nid]
        n_dedup = len(merged)
    return {"relabeled": n_relabel, "deduped": n_dedup}


def infer_rich_topology_direct(instance: InstanceHypergraph,
                                paper_id: str, include_other: bool = False) -> list[dict]:
    """Read rich-topology edges DIRECTLY from instance hyperedges (no co-occurrence).

    Each hyperedge already encodes a rich relation (its pattern_type + the
    labeled nodes it connects). Instead of scanning for shared nodes across
    patterns (co-occurrence, which produces noise), we read each hyperedge's
    OWN structure and classify it into a rich-topology edge by what node
    TYPES it connects:

    - law_parameter: a constitutive_law hyperedge -> (law, [PARAMETER nodes])
    - method_parameter: a hyperedge with METHOD + PARAMETER nodes -> (method, params)
    - method_phenomenon: a hyperedge with METHOD + PHENOMENON -> (method, phenomenon)
    - method_regime: a hyperedge with METHOD + REGIME -> (method, regime)
    - composition: a composed_of hyperedge -> (whole, [components])
    - definition: a defines hyperedge -> (subject, definition)
    - nary: any hyperedge with >=3 nodes of distinct types -> recorded as n-ary

    Returns list of {kind, nodes: [(surface, label)], pattern_type, evidence}.
    """
    edges = []
    for he in instance.hyperedges.values():
        nodes = []
        for nid in he.node_ids:
            n = instance.nodes.get(nid)
            if n:
                nodes.append({"surface": n.surface, "labels": list(n.labels)})
        if len(nodes) < 2:
            continue
        labels_present = set()
        for nd in nodes:
            labels_present.update(nd["labels"])
        # classify by what node-type combination this hyperedge connects
        has_method = bool(labels_present & {"METHOD"})
        has_param = bool(labels_present & {"PARAMETER"})
        has_phenom = bool(labels_present & {"PHENOMENON"})
        has_regime = bool(labels_present & {"REGIME"})
        has_numeric = bool(labels_present & {"NUMERIC"})
        n_distinct_types = sum([has_method, has_param, has_phenom, has_regime, has_numeric])

        # determine kind
        # evolution relations FIRST: extends/improves/compares/replaces/adapts/
        # background between METHODS (or PHENOMENA). These are cross-method
        # evolution relations the text states — distinct from intra-section
        # structure (law/definition/composition). Extracted by extract_hypergraph
        # when the text states the relation (NOT inferred — inference is downstream).
        _evo_types = {"extends", "improves", "compares", "replaces", "adapts", "background"}
        if he.pattern_type in _evo_types or any(t in he.pattern_type for t in _evo_types):
            # only count as evolution if it connects 2+ METHODS or 2+ PHENOMENA
            # (a compares-edge METHOD↔METHOD). Mixed-type evolution stays n-ary.
            method_count = sum(1 for nd in nodes if "METHOD" in nd["labels"])
            phenom_count = sum(1 for nd in nodes if "PHENOMENON" in nd["labels"])
            if method_count >= 2 or phenom_count >= 2:
                kind = "evolution"
            else:
                kind = "nary"  # evolution edge but mixed types → n-ary
        elif "constitutive_law" in he.pattern_type:
            kind = "law_parameter"
        elif "composed_of" in he.pattern_type:
            kind = "composition"
        elif "defines" in he.pattern_type:
            kind = "definition"
        elif n_distinct_types >= 3:
            kind = "nary"  # check n-ary FIRST so 3+type edges aren't swallowed by method_phenomenon
        elif has_method and has_param and not has_phenom:
            kind = "method_parameter"
        elif has_method and has_phenom:
            kind = "method_phenomenon"
        elif has_method and has_regime:
            kind = "method_regime"
        else:
            kind = "other"

        if kind == "other" and not include_other:
            continue  # skip low-value param→param dependencies
        edges.append({
            "kind": kind,
            "pattern_type": he.pattern_type,
            "nodes": [{"surface": nd["surface"][:40], "labels": nd["labels"]} for nd in nodes],
            "n_nodes": len(nodes),
            "n_distinct_types": n_distinct_types,
            "evidence": (he.evidence_span or "")[:80],
            "paper_id": paper_id,
            "section": (he.eid.split("_")[0] if he.eid and "_" in he.eid else (he.eid or "")),
        })
    return edges
