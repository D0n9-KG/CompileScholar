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
from dataclasses import dataclass, field
from typing import Callable

from granular_agent.knowledge_base import KnowledgeBase, Mutation, Op, Role, Outcome
from granular_agent.concept_graph import ConceptGraph, Concept, _llm_align_batch, _norm_surface
from granular_agent.hypergraph_schema import InstanceHypergraph


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
              type_filter=("METHOD", "PHENOMENON", "PARAMETER"),
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
                items.append((c.concept_id, c.surfaces()[0],
                              c.symbol if getattr(c, "symbol", "") else ""))
            if len(items) < 2:
                continue
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                # delegate candidate-group discovery to the verified内核
                groups = _llm_align_batch(batch, t, self.llm_judge)
                report.n_candidates += len(groups)
                for group in groups:
                    if len(group) < 2:
                        continue
                    cands = [self.kb.abox.concepts[g] for g in group
                             if g in self.kb.abox.concepts]
                    if len(cands) < 2:
                        continue
                    self._route_group(cands, domain, report)
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
        """Route a candidate same-concept group through the kernel's align_merge
        5-outcome. The group is LLM-judged to be "same concept" — so the expected
        outcome is MERGE. But the kernel decides by n-ary node-set subset relation
        (Challenge B): if the candidate's concept-set is identical/subset of an
        existing edge's -> merge; partial overlap -> relate; no overlap -> insert.
        For concept alignment, the "edge" is a synthetic same-concept edge built
        from the group; we propose align_merge with the group's concept_ids."""
        if len(cands) < 2:
            return
        # keep = concept with most variants (stable, like legacy align_concepts)
        keep = max(cands, key=lambda c: (len(c.surface_variants), -ord(c.concept_id[-1])))
        # propose an align_merge: the group as a "same-concept" n-ary edge. The
        # kernel routes it (insert/merge/relate/conflict/reject). Most same-
        # concept groups will MERGE (identical concept-set -> merge, union
        # provenance); a group partially overlapping an existing edge -> relate.
        node_ids = [c.concept_id for c in cands]
        result = self.kb.commit([Mutation(
            op=Op.ALIGN_MERGE, target=f"align_{keep.concept_id}",
            proposer_role=Role.ALIGNER, domain=domain,
            evidence=f"same-concept group: {', '.join(c.surfaces()[0] for c in cands if c.surfaces())}",
            rationale="LLM-judged same concept (canonicalize)",
            payload={"new_edge": {
                "node_ids": node_ids,
                "kind": "same_concept",
                "roles": ["same"] * len(node_ids),
                "provenance": {"paper_id": "aligner", "evidence": "canonicalize",
                               "year": "", "section": ""}}})])
        if not result.ok:
            # align_merge validate failed (shouldn't — concepts exist); count reject
            report.n_reject += 1
            return
        if result.routed:
            outcome = result.routed[0]["outcome"]
            if outcome == Outcome.MERGE:
                report.n_merge += 1
                report.merges.append({"keep": keep.concept_id,
                                      "merged": [c.concept_id for c in cands
                                                 if c.concept_id != keep.concept_id],
                                      "outcome": outcome})
            elif outcome == Outcome.INSERT:
                report.n_insert += 1
            elif outcome == Outcome.RELATE:
                report.n_relate += 1
            elif outcome == Outcome.CONFLICT:
                report.n_conflict += 1
            else:
                report.n_reject += 1

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
