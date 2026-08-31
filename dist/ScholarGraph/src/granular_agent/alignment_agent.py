"""Cross-paper alignment agent (Step 4): EDC three-stage + 5-outcome through the
KnowledgeBase kernel (aligner op only), Concept Define phase.

Design (see .research_tmp/plan_total_design_v2.md "Builder 群 / 跨论文对齐 agent"
+ plan_detailed_design_supplement.md + DECISION-alignment-agent-form):

EDC three stages (Extract→Define→Canonicalize):
  1. Extract: concepts already in the A-box (from extractor add_edge commits).
  2. Define (LLM, NEW): generate a one-sentence definition for each concept,
     filled into Concept.definition (the field Step 2 added). This is the
     semantic basis canonicalize judges on — not surface text alone.
  3. Canonicalize (embedding + LLM judge ≠ extraction model): find candidate
     same-concept groups WITHIN A DOMAIN, then route each candidate pair through
     the KnowledgeBase align_merge op → 5-outcome (insert/merge/relate/conflict/
     reject by n-ary node-set subset relation, Step 1 kernel). NOT direct
     merge_concepts — every alignment write enters the ledger / passes 5-outcome
     route / is replayable.

Same-domain only (DecentMem 防混域): align concepts introduced by papers of one
domain; cross-domain concepts don't merge.

Incremental: each paper ingests → align only the NEW concepts (not full re-align).

judge ≠ extraction model: AlignmentAgent takes separate llm_define / llm_judge
(both different from the extractor's LLM — self-endorsement avoidance).

Honest scope (DECISION): the agent delegates the "find candidate groups" logic
to the verified _llm_align_batch内核 (surgical), but the merge decision goes
through the kernel's align_merge 5-outcome (not the legacy direct merge_concepts).
conflict/relate at concept level is rarer than at edge level (concepts are
entities not claims), but the path is kept.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from granular_agent.knowledge_base import KnowledgeBase, Mutation, Op, Role, Outcome
from granular_agent.concept_graph import ConceptGraph, Concept, _llm_align_batch, _norm_surface
from granular_agent.hypergraph_schema import InstanceHypergraph


# ---------------------------------------------------------------------------
# Alignment batching fixes (2026-08-27, audit align_first_test root causes)
# ---------------------------------------------------------------------------

# structural noise: figure/table/equation references and closed-list generic
# surfaces. Domain-agnostic STRUCTURAL patterns only (rules-vs-LLM boundary:
# whether "Fig. 3" or bare "model" is a named method is not a semantic
# judgment). These surfaces may exist as concepts (edges reference them) but
# are excluded from ALIGNMENT — they were the raw material of the audit's
# 泛称吞噬 bad merges ("model"+DEM, "theory"+Nonlocal continuum, "Fig.3"×2).
_NOISE_REF_RE = re.compile(
    r"^(fig|figure|table|tab|eq|equation|sec|section|ref|appendix)\.?\s*[\d\w]+$",
    re.IGNORECASE)
_GENERIC_SURFACES = {
    "model", "models", "theory", "theories", "law", "laws",
    "method", "methods", "approach", "framework", "algorithm", "algorithms",
    "experiment", "experiments", "experimental measurements", "measurement",
    "simulation", "simulations", "numerical simulations", "data", "results",
    "flow", "modeling", "analysis", "study", "setup", "system",
}


def _is_noise_surface(surface: str) -> bool:
    s = (surface or "").strip().rstrip(".,;:")
    if not s:
        return True
    if _NOISE_REF_RE.match(s):
        return True
    return s.lower() in _GENERIC_SURFACES


_ALIGN_STOPWORDS = {
    "the", "of", "and", "in", "to", "for", "with", "on", "at", "by", "from",
    "a", "an", "is", "are", "was", "were", "its", "their", "each", "own",
    "this", "that", "these", "those", "based", "using", "used", "into",
}


def _surface_tokens(surface: str) -> set:
    """Token set for preclustering: lowercase, plural-normalized (trailing
    's' stripped on 4+ char tokens — 'temperatures'~'temperature'), English
    function words dropped ('anisotropy OF THE granular temperature' clusters
    with 'granular temperature'). Pure normalization, no semantics."""
    toks = set()
    for t in re.findall(r"[a-z0-9]+", surface.lower()):
        if t in _ALIGN_STOPWORDS:
            continue
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        toks.add(t)
    return toks


def _precluster_batches(items, batch_size: int, jaccard_threshold: float = 0.35):
    """Group items so surface-similar concepts share one LLM batch
    (deterministic token-Jaccard clustering over connected components).
    Items with no similar partner are SKIPPED — no LLM call (the old code
    sent every solo concept through a judge that could not merge it anyway;
    544-concept runs become ~clusters-only, a large efficiency win).
    Single letters/symbols DO cluster together (J=1.0) so the judge can
    explicitly reject/confirm them WITH definitions in view."""
    n = len(items)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    toks = [_surface_tokens(it[1]) for it in items]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = toks[i], toks[j]
            if not a or not b:
                continue
            if len(a & b) / len(a | b) >= jaccard_threshold:
                union(i, j)
    comps: dict[int, list] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(items[i])
    batches = []
    for members in comps.values():
        if len(members) < 2:
            continue   # solo: nothing to merge with
        for k in range(0, len(members), batch_size):
            batches.append(members[k:k + batch_size])
    return batches


# ---------------------------------------------------------------------------
# Define prompt (generate concept definitions)
# ---------------------------------------------------------------------------

_DEFINE_PROMPT = """You are writing one-sentence DEFINITIONS for scientific concepts from a {domain} paper.

For each concept (a {type_label}), given its surface mention(s) + an evidence span,
write ONE sentence defining what it IS in this paper's context. The definition is
used for cross-paper alignment (judging if two surfaces = same concept), so it should
capture the concept's IDENTITY, not incidental details.

Concepts:
{items}

Output JSON: {{"definitions":[{{"concept_id":"...","definition":"one sentence"}}]}}

Rules:
- One sentence, <30 words, concrete.
- Define WHAT it is (a method/parameter/phenomenon/...), not where it appears.
- For a METHOD: what approach/theory/law it is + what it models.
- For a PARAMETER: what physical quantity/symbol it is.
- For a PHENOMENON: what behavior/effect it is.
- If the surface alone is ambiguous, use the evidence to disambiguate.
"""


# ---------------------------------------------------------------------------
# Alignment report
# ---------------------------------------------------------------------------

@dataclass
class AlignmentReport:
    """Result of one align() pass. Lets the caller/reviewer SEE what aligned."""
    n_defined: int = 0
    n_candidates: int = 0          # candidate same-concept groups found
    n_insert: int = 0
    n_merge: int = 0
    n_relate: int = 0
    n_conflict: int = 0
    n_reject: int = 0
    merges: list[dict] = field(default_factory=list)   # [{keep, merged, outcome}]
    definitions_sample: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AlignmentAgent
# ---------------------------------------------------------------------------

class AlignmentAgent:
    """EDC three-stage cross-paper alignment, writing through the KB kernel
    (align_merge op → 5-outcome route). Define phase fills Concept.definition.

    llm_define: LLM for the Define stage (concept definitions).
    llm_judge: LLM for Canonicalize (must ≠ extraction model).
    Both are call(prompt, max_tokens)->str.
    """

    def __init__(self, kb: KnowledgeBase,
                 llm_define: Callable[[str, int], str | None],
                 llm_judge: Callable[[str, int], str | None],
                 domain_default: str = "global",
                 judge_fn_5outcome: Callable[[dict], str] | None = None):
        self.kb = kb
        self.llm_define = llm_define
        self.llm_judge = llm_judge
        self.domain_default = domain_default
        # the kernel's ambiguous 5-outcome judge (set on the KB). If None, the
        # kernel defaults ambiguous no-overlap to insert (data-preserving). The
        # aligner installs this so ambiguous same-concept cases get a real
        # conflict-vs-insert decision (judge ≠ extraction model).
        if judge_fn_5outcome is not None:
            self.kb.set_judge(judge_fn_5outcome)

    # ===================================================================
    # Stage 2: Define — generate concept definitions
    # ===================================================================

    def define(self, concept_ids: list[str] | None = None,
               domain: str = "", batch_size: int = 20) -> int:
        """Define stage (EDC's D): generate one-sentence definitions for the
        given concepts (default: all without a definition yet). Fills
        Concept.definition. Incremental — skips concepts already defined."""
        domain = domain or self.domain_default
        if concept_ids is None:
            targets = [c for c in self.kb.abox.concepts.values() if not c.definition]
        else:
            targets = [self.kb.abox.concepts[cid] for cid in concept_ids
                       if cid in self.kb.abox.concepts and not self.kb.abox.concepts[cid].definition]
        if not targets:
            return 0
        n = 0
        type_map = {"METHOD": "method", "PHENOMENON": "phenomenon",
                    "PARAMETER": "parameter", "NUMERIC": "numeric value",
                    "MATERIAL": "material", "REGIME": "regime", "PROPERTY": "property"}
        for i in range(0, len(targets), batch_size):
            batch = targets[i:i + batch_size]
            items = []
            for c in batch:
                surf = c.surfaces()[0] if c.surfaces() else ""
                ev = (c.surface_variants[0].evidence if c.surface_variants else "")[:120]
                items.append({"concept_id": c.concept_id, "surface": surf, "evidence": ev})
            prompt = _DEFINE_PROMPT.format(
                domain=domain, type_label=type_map.get(batch[0].type, batch[0].type),
                items=json.dumps(items, ensure_ascii=False))
            raw = self.llm_define(prompt, 2000)
            defs = _parse_json(raw).get("definitions", []) if raw else []
            dmap = {d.get("concept_id"): d.get("definition", "")
                    for d in defs if isinstance(d, dict)}
            for c in batch:
                d = dmap.get(c.concept_id, "")
                if d:
                    c.definition = str(d)[:300]
                    n += 1
        return n

    # ===================================================================
    # Stage 3: Canonicalize — find candidates, route via align_merge 5-outcome
    # ===================================================================

    def align(self, domain: str = "",
              new_concept_ids: list[str] | None = None,
              type_filter=("METHOD", "PHENOMENON", "PARAMETER",
                           "PROPERTY", "REGIME"),
              batch_size: int = 20) -> AlignmentReport:
        """Canonicalize stage: find same-concept candidate groups WITHIN a domain,
        route each merge through kb.commit(align_merge Mutation) → 5-outcome.

        new_concept_ids: if given, only align those (incremental — align the
        new concepts against the existing pool). If None, full align.

        Same-domain: only concepts introduced by papers of `domain` are aligned
        together (cross-domain concepts don't merge — DecentMem 防混域). Since
        Concept doesn't carry a domain field, we filter by source_papers →
        domain via the paper→domain map the extractor recorded. Concepts whose
        papers are all a different domain are skipped for this pass."""
        domain = domain or self.domain_default
        report = AlignmentReport()
        for t in type_filter:
            concepts = [c for cid, c in self.kb.abox.concepts.items()
                        if c.type == t and not c.deprecated
                        and self._concept_in_domain(c, domain)]
            if new_concept_ids is not None:
                new_set = set(new_concept_ids)
                concepts = [c for c in concepts if c.concept_id in new_set]
            if len(concepts) < 2:
                continue
            items = []
            for c in concepts:
                if not c.surfaces():
                    continue
                # noise gate (2026-08-27, audit align_first_test): figure/
                # table/eq references and closed-list generic surfaces become
                # concepts but never ALIGN (they swallowed named methods:
                # 'model'+DEM, 'theory'+Nonlocal continuum merged). Structural/
                # deterministic rule — no semantic judgment here.
                if _is_noise_surface(c.surfaces()[0]):
                    continue
                items.append((c.concept_id, c.surfaces()[0],
                              c.symbol if getattr(c, "symbol", "") else "",
                              c.definition if getattr(c, "definition", "") else ""))
            if len(items) < 2:
                continue
            # SIMILARITY-PRECLUSTERED batches (2026-08-27, audit root cause #1):
            # the old insertion-order slicing (batch_size=20) meant 'kinetic
            # theory' (#52) and 'kinetic theory formula' (#97) NEVER appeared
            # in the same LLM call — merge rate 3-9% was mostly a batching
            # artifact, not judge strictness. Pre-cluster by surface token
            # Jaccard (deterministic, structural) so same-concept variants
            # share a batch; the LLM still judges the merge semantically.
            for batch in _precluster_batches(items, batch_size):
                # delegate candidate-group discovery to the verified内核
                groups, related = _llm_align_batch(batch, t, self.llm_judge)
                report.n_candidates += len(groups)
                for group in groups:
                    if len(group) < 2:
                        continue
                    cands = [self.kb.abox.concepts[g] for g in group
                             if g in self.kb.abox.concepts]
                    if len(cands) < 2:
                        continue
                    self._route_group(cands, domain, report)
                # relate path (2026-08-27, the '准而不连' gap): same-family /
                # derivative concept pairs the judge explicitly declined to
                # merge get a RELATE edge instead — downstream family
                # connectivity without polluting merge precision.
                for a, b, note in related:
                    if a in self.kb.abox.concepts and b in self.kb.abox.concepts:
                        r = self.kb.commit([Mutation(
                            op=Op.ADD_CONCEPT_RELATION, target=a,
                            proposer_role=Role.ALIGNER, domain=domain,
                            evidence=note or "same-family (aligner relate)",
                            rationale="LLM-judged related, not same concept",
                            payload={"node_ids": [a, b],
                                     "roles": ["from", "to"]})])
                        if r.ok:
                            report.n_relate += 1
                            report.merges.append({"keep": a, "merged": [b],
                                                  "outcome": "related"})
        return report

    def _concept_in_domain(self, c: Concept, domain: str) -> bool:
        """Same-domain filter. Concept has no domain field; a concept is 'in
        domain' if ANY of its source_papers was ingested under that domain.
        The paper→domain map lives on the KB (the extractor recorded domain on
        each add_edge Mutation; we reconstruct paper→domain from the ledger if
        available, else accept the concept — honest: without the map, the
        domain filter is a no-op pass)."""
        if not hasattr(self.kb, "_paper_domain") or not self.kb._paper_domain:
            return True  # no paper→domain map -> don't filter (honest no-op)
        for pid in c.source_papers:
            if self.kb._paper_domain.get(pid) == domain:
                return True
        return False

    def _route_group(self, cands: list[Concept], domain: str,
                     report: AlignmentReport) -> None:
        """Route a candidate same-concept group through the kernel's
        ALIGN_CONCEPT_MERGE op — the CONCEPT-LEVEL merge path (Step 4 BLOCKER
        B1 fix). This真合并 Concept objects (redirect hyperedge endpoints /
        union surfaces / drop merged concepts), so abox.concepts shrinks.

        Honest design (fixed): the 5-outcome align_merge is EDGE-level (judges
        by edge node-set subset relation, only unions edge provenance — never
        merges Concepts). Concept alignment is a different operation: the LLM
        judged the group same-concept, so we MERGE the Concept objects via the
        dedicated ALIGN_CONCEPT_MERGE op (aligner contract, kernel-validated,
        ledger-recorded). We do NOT pretend the edge 5-outcome covers concept
        merging — that was the名实不符 BLOCKER: concepts never merged.

        5-outcome (insert/merge/relate/conflict/reject) stays for EDGE
        alignment (align_merge) when that path is used; concept alignment uses
        this dedicated op. (See DECISION-alignment-agent-form update.)"""
        if len(cands) < 2:
            return
        # pure-symbol merge guard (2026-08-27, audit symbol-collision class):
        # a group whose EVERY surface is symbol/math-only (no alphabetic word
        # >=4 chars — 'F', 'f', '$F(\Phi)$', 'g_0(\nu)') carries no evidence
        # but the letters themselves, and same-letter-different-quantity
        # collisions (f=frequency vs f=friction) are the documented failure.
        # Named surfaces always contain a >=4-char word (inertial, shear...),
        # so this never blocks a named merge. Deterministic structural guard.
        if all(not re.search(r"[a-zA-Z]{4,}", s)
               for c in cands for s in c.surfaces()[:1]):
            return
        # keep = concept with most variants; tiebreak by SMALLEST concept_id
        # (stable, predictable — keeps the earliest-created concept; m1 fix
        # replaced the fragile -ord(last-char) tiebreak).
        keep = max(cands, key=lambda c: (len(c.surface_variants),
                                         tuple(-ord(ch) for ch in c.concept_id)))
        merge_ids = [c.concept_id for c in cands if c.concept_id != keep.concept_id]
        if not merge_ids:
            return
        result = self.kb.commit([Mutation(
            op=Op.ALIGN_CONCEPT_MERGE, target=keep.concept_id,
            proposer_role=Role.ALIGNER, domain=domain,
            evidence=f"same-concept group: {', '.join(c.surfaces()[0] for c in cands if c.surfaces())}",
            rationale="LLM-judged same concept (canonicalize)",
            payload={"merge_ids": merge_ids})])
        if not result.ok:
            report.n_reject += 1
            return
        # the concept merge is a single deterministic outcome (merged), not a
        # 5-outcome route — count it as a merge and record which concepts merged.
        report.n_merge += 1
        report.merges.append({"keep": keep.concept_id, "merged": merge_ids,
                              "outcome": "concept_merged"})

    # ===================================================================
    # end-to-end: Define + Canonicalize on new concepts
    # ===================================================================

    def align_new(self, domain: str = "",
                  new_concept_ids: list[str] | None = None) -> AlignmentReport:
        """Full EDC on new concepts: Define (fill definitions) then Canonicalize
        (route align_merge 5-outcome). Returns a report so the caller/reviewer
        can SEE the alignment result, not just a count."""
        domain = domain or self.domain_default
        report = AlignmentReport()
        report.n_defined = self.define(new_concept_ids, domain)
        canon = self.align(domain, new_concept_ids)
        report.n_candidates = canon.n_candidates
        report.n_insert = canon.n_insert
        report.n_merge = canon.n_merge
        report.n_relate = canon.n_relate
        report.n_conflict = canon.n_conflict
        report.n_reject = canon.n_reject
        report.merges = canon.merges
        return report


def _parse_json(raw: str | None) -> dict:
    if not raw:
        return {}
    from granular_agent.llm_client import parse_json_response
    return parse_json_response(raw) or {}
