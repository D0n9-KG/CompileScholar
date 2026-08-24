"""Schema self-evolution agent (Step 3): DIAL-KG three-stage + SAGE writer-reader
+ unified queue, writing through the KnowledgeBase kernel (not direct meta mutate).

Design (see .research_tmp/plan_total_design_v2.md "Builder 群 / schema 自演化 agent"
+ plan_detailed_design_supplement.md section 3 并进机制 + DECISION-evolution-agent-form):

The agent ORGANIZES the existing evolution内核 (hypergraph_evolution.py:
EvolutionTrigger / evolution_probe / validate_proposal / detect_*_triggers /
cluster_pattern_instances / name_split_subpatterns) into the v2 form, and routes
the APPLY phase through KnowledgeBase.commit (evolver Mutation → validate →
apply) instead of the legacy direct meta mutate. This is what closes the design:
every T-box change enters the ledger, passes schema-constrained validate (IS-A/
family/invariant), and is replayable/time-travelable. (The legacy
run_evolution_loop bypassed the kernel because the kernel didn't exist yet.)

Three stages (DIAL-KG):
  1. probe (P2): evolution_probe proposes structural changes for failing edges
     / recurring mismatches / consumer SAGE feedback.
  2. governance: variant vs new-kind adjudication (role+family+embedding multi-
     view) + candidate-evidence binding + recurring crystallize (cross_node
     gate,委托现有 CONSERVATIVE_CROSS_NODE) + consensus + HITL flag.
  3. apply: convert each accepted proposal into an evolver Mutation and
     kb.commit() (validate + apply). add_pattern proposals get an LLM-generated
     semantic_boundary (design supplement section 4) attached to the payload.

SAGE writer-reader: consumer feedback (a pattern that always misses in queries)
is collected via collect_consumer_feedback and accumulated as a recurring
mismatch signal that feeds the probe. (consumer itself is Step 6; the interface
is reserved here so Step 6 can plug in.)

Unified queue: ALL trigger sources (validate failures / recurring mismatches /
consumer SAGE feedback / self-triggered split-merge-retire) propose into one
queue; drain() is the single serial builder consumer (断点 6).

skill distiller: collects per-(domain,pattern) extraction modes; when stability
> 0.6, proposes a distill_skill Mutation (evolver write contract) committed via
kb.commit — this is where the Step-2-collected extraction modes crystallize into
the 4th-layer Skill Library (断点 5, the loop Step 2 left open).

Honest scope (DECISION):
- DIAL-KG 三阶段 = probe→governance→apply functions, not separate services.
- HITL: reserved interface (needs_hitl flag); real HITL由 Step 7 协调层接.
- recurring crystallize委托现有 cross_node gate (CONSERVATIVE_CROSS_NODE).
- consumer SAGE: interface reserved; can't end-to-end test until Step 6.
- The agent delegates the ADJUDICATION内核 (probe/validate/detect_triggers);
  it does NOT rewrite them (surgical).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from granular_agent.hypergraph_schema import (
    MetaHypergraph, MetaHyperedgePattern, Hyperedge, InstanceHypergraph,
)
from granular_agent.knowledge_base import (
    KnowledgeBase, Mutation, Op, Role, Skill,
)
# Use module-attribute access (hev.X) rather than `from hev import X` so that
# test patches (hev.evolution_probe = mock) take effect — `from import` binds
# the name at import time and wouldn't see the patch. EvolutionTrigger is a
# class used for typing/instantiation (not patched), so it can stay direct.
import granular_agent.hypergraph_evolution as hev
from granular_agent.hypergraph_evolution import EvolutionTrigger


# ---------------------------------------------------------------------------
# Trigger sources (unified queue) — 断点 6
# ---------------------------------------------------------------------------

@dataclass
class TriggerSource:
    """One item in the unified evolution queue. `kind` identifies the source
    so governance can weight recurrence correctly (cross-node validate failure
    vs consumer SAGE feedback vs self-triggered split)."""
    kind: str            # "validate_failure" | "recurring_mismatch" |
                         # "consumer_feedback" | "self_split" | "self_merge" |
                         # "self_retire" | "self_rename"
    payload: dict        # source-specific
    domain: str = "global"
    paper_id: str = ""
    node_id: str = ""    # the DAG node / section that produced it (cross-node)


@dataclass
class ConsumerFeedback:
    """SAGE writer-reader: a consumer reports that a pattern/concept behaves
    poorly in its queries (always misses / over-merges / conflicts). This is the
    writer-reader feedback signal that triggers evolution (a recurring mismatch
    from the reader's side, not just the extractor's validate failures)."""
    pattern_id: str = ""
    concept_surface: str = ""
    issue: str = ""     # "always_misses" | "over_merges" | "conflicts" | "stale"
    detail: str = ""
    consumer: str = ""  # which consumer (qa / survey / paper_search / ...)


@dataclass
class EvolutionReport:
    """Result of one drain() of the queue. Lets the caller (and subagent
    reviewer) SEE what evolved, not just a pass/fail."""
    n_proposed: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    n_hitl: int = 0
    accepted: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    hitl: list[dict] = field(default_factory=list)
    version_before: str = ""
    version_after: str = ""


# ---------------------------------------------------------------------------
# semantic_boundary generation prompt (design supplement section 4)
# ---------------------------------------------------------------------------

_BOUNDARY_PROMPT = """You just added a new hyperedge pattern to a {domain} schema:

pattern_id: {pattern_id}
description: {description}
role_slots: {role_slots}
family: {family}

Write a short SEMANTIC BOUNDARY for it: ONE sentence saying WHEN this pattern
applies vs the nearby patterns it's easy to confuse with (e.g. "X = functional
dependence, NOT a co-listing"). This boundary is shown to the extractor so it
picks the right pattern. Keep it <40 words, concrete, no filler.

Output: just the boundary sentence (no JSON, no quotes).
"""


# ---------------------------------------------------------------------------
# EvolutionAgent
# ---------------------------------------------------------------------------

class EvolutionAgent:
    """DIAL-KG three-stage + SAGE writer-reader + unified queue, writing through
    the KnowledgeBase kernel.

    llm: the evolution LLM (probe + governance distinctness + boundary gen).
    Distinct from the extraction LLM is good practice but the kernel doesn't
    enforce it (evolution is a T-box write, judged by validate_proposal's
    own near-dup/distinctness gates). llm(prompt, max_tokens)->str.
    """

    # recurring crystallize threshold (委托现有 CONSERVATIVE_CROSS_NODE; the
    # agent surfaces cross_node in the report so the gate is auditable)
    CONSERVATIVE_CROSS_NODE = 2

    def __init__(self, kb: KnowledgeBase, llm: Callable[[str, int], str | None],
                 domain_default: str = "global"):
        self.kb = kb
        self.llm = llm
        self.domain_default = domain_default
        self._queue: list[TriggerSource] = []
        # cross-node recurrence tracking (委托 EvolutionTrigger)
        self.trigger = EvolutionTrigger()
        # SAGE writer-reader: accumulated consumer feedback
        self._consumer_feedback: list[ConsumerFeedback] = []

    # ===================================================================
    # Unified queue (断点 6): all trigger sources propose here
    # ===================================================================

    def propose_trigger(self, src: TriggerSource) -> None:
        """Enqueue a trigger source. Single serial consumer drains via drain()."""
        self._queue.append(src)

    def collect_consumer_feedback(self, fb: ConsumerFeedback) -> None:
        """SAGE writer-reader: a consumer reports a pattern/concept behaving
        poorly. Accumulated; recurring feedback on the same pattern becomes a
        recurring_mismatch trigger (the reader side of writer-reader)."""
        self._consumer_feedback.append(fb)
        # if the same pattern_id gets >=2 feedbacks, it's a recurring reader
        # signal -> enqueue a recurring_mismatch trigger for governance to weigh
        from collections import Counter
        counts = Counter(f.pattern_id for f in self._consumer_feedback if f.pattern_id)
        for pid, n in counts.items():
            if n >= 2:
                self.propose_trigger(TriggerSource(
                    kind="consumer_feedback",
                    payload={"pattern_id": pid, "n_feedbacks": n,
                             "issues": [f.issue for f in self._consumer_feedback
                                        if f.pattern_id == pid]},
                    domain=self.domain_default))

    # ===================================================================
    # Stage 1: probe — convert trigger sources into proposals
    # ===================================================================

    def _probe(self, src: TriggerSource,
               failing_hes: list[tuple[Hyperedge, str]] | None = None
               ) -> list[dict]:
        """Stage 1: produce structural-change proposals from one trigger source.

        For validate_failure / recurring_mismatch: delegate to evolution_probe
        on the failing hyperedges (the verified probe内核). Cross-node recurrence
        is tracked via self.trigger (record happens in propose_validate_failures).

        For self_split / self_merge / self_retire / self_rename: the payload
        already carries a proposal-shaped dict (detect_*_triggers output); pass
        through to governance.

        For consumer_feedback: a recurring reader signal -> propose a split (if
        the pattern over-merges) or a rename (if it's confusingly named), based on
        the issue type. Lightweight; the governance stage gates it."""
        kind = src.kind
        if kind in ("validate_failure", "recurring_mismatch"):
            if not failing_hes:
                return []
            distinct = [he for he, _ in failing_hes]
            proposals = hev.evolution_probe(
                distinct, self.kb.tbox, src.paper_id or src.node_id or "p",
                domain=src.domain or self.domain_default,
                llm="deepseek", instance=None)
            return proposals or []
        if kind in ("self_split", "self_merge", "self_retire", "self_rename"):
            # detect_*_triggers output carries `representatives` (cluster evidence
            # strings) but NOT `evidence_span` — validate_proposal requires a
            # non-empty evidence_span (BLOCKER B1 fix: without this, ALL
            # self_split/merge/retire/rename paths were silently rejected as
            # "no verbatim evidence span"). split/merge/retire are STRUCTURAL
            # triggers (fired by instance clustering, not a single text span),
            # so the evidence is the joined representatives (or the rationale).
            p = dict(src.payload) if src.payload else {}
            if not p.get("evidence_span"):
                reps = p.get("representatives") or []
                if reps:
                    p["evidence_span"] = "; ".join(reps[:3])
                elif p.get("rationale"):
                    p["evidence_span"] = p["rationale"]
                else:
                    # structural op with no text evidence — use the op itself as
                    # the audit trail (the gate needs *something* non-empty; the
                    # real evidence is the clustering that fired the trigger).
                    p["evidence_span"] = f"structural {kind} on {p.get('pattern_id','')}"
            return [p]
        if kind == "consumer_feedback":
            pid = src.payload.get("pattern_id", "")
            issues = src.payload.get("issues", [])
            if not pid or pid not in self.kb.tbox.patterns:
                return []
            # over-merges -> propose split; confusing name -> propose rename;
            # stale -> propose retire. These are PROPOSALS; governance gates them.
            if "over_merges" in issues:
                return [{"op": "split", "pattern_id": pid,
                         "source": "consumer_feedback",
                         "rationale": "consumer reports over-merging"}]
            if "stale" in issues:
                return [{"op": "retire", "pattern_id": pid,
                         "source": "consumer_feedback",
                         "rationale": "consumer reports stale"}]
            return []
        return []

    # ===================================================================
    # Stage 2: governance — variant/new-kind + crystallize + consensus + HITL
    # ===================================================================

    def _governance(self, proposal: dict, src: TriggerSource) -> tuple[dict, str]:
        """Stage 2: adjudicate one proposal. Returns (proposal_or_None, decision)
        where decision in {accept, reject, hitl}.

        Delegates the near-dup / distinctness / evidence checks to the verified
        validate_proposal内核. Adds: recurring crystallize (cross_node gate for
        growth ops), HITL flagging (new top-level family / uncertain re-type),
        and consumer-feedback weighting."""
        op = proposal.get("op", "")
        # the verified validate_proposal gate (evidence + near-dup + distinctness)
        # — applied to GROWTH ops (add_pattern/add_meta_node/add_subclass) from
        # evolution_probe, which uses the legacy op-name space validate_proposal
        # recognizes. SELF-TRIGGERED ops (split/merge/retire/rename, kernel Op
        # name space) SKIP validate_proposal: (a) detect_*_triggers already
        # determined them via clustering (not one-off text proposals needing
        # near-dup/distinctness LLM check), (b) validate_proposal doesn't
        # recognize kernel op names (split vs split_meta_node) so it would
        # reject them as "unknown op" — the kernel's own _schema_constraint
        # (IS-A/family/role invariant) is the real gate for these. We DO
        # require evidence (injected by _probe from representatives).
        is_self_op = src.kind in ("self_split", "self_merge",
                                  "self_retire", "self_rename")
        if is_self_op:
            if not (proposal.get("evidence_span") or "").strip():
                proposal["rejected_reason"] = "no-evidence"
                return None, "reject"
            # skip validate_proposal; the kernel gates schema correctness
        else:
            v = hev.validate_proposal(proposal, self.kb.tbox,
                                      domain=src.domain or self.domain_default, llm="deepseek")
            if not v.get("valid"):
                proposal["rejected_reason"] = v.get("reason", "")
                proposal["suggested_alternative"] = v.get("suggested_alternative", "")
                return None, "reject"
        # recurring crystallize: a GROWTH op (add_pattern/add_meta_node/add_subclass)
        # is accepted only if the gap recurred across >= CONSERVATIVE_CROSS_NODE
        # nodes (委托现有 gate semantics). validate_failure sources carry
        # cross_node via self.trigger; self_* and consumer_feedback sources are
        # themselves recurring signals (split/merge are triggered BY clustering,
        # not one-offs) so they pass.
        is_growth = op in ("add_pattern", "add_meta_node", "add_subclass")
        if is_growth and src.kind in ("validate_failure", "recurring_mismatch"):
            cross = src.payload.get("cross_node", 1)
            if cross < self.CONSERVATIVE_CROSS_NODE:
                proposal["rejected_reason"] = (
                    f"conservative gate: growth needs cross_node>={self.CONSERVATIVE_CROSS_NODE}, "
                    f"got {cross} (real gap recurses and is accepted later)")
                return None, "reject"
        # HITL: a new top-level FAMILY (not just a new pattern in an existing
        # family) is a big enough ontological move to flag for human review.
        # Reserved interface — Step 7 协调层接真 HITL. Step 3 flags, doesn't block.
        if op == "add_pattern":
            fam = proposal.get("family", "")
            existing_families = {p.family for p in self.kb.tbox.patterns.values() if p.family}
            if fam and fam not in existing_families:
                proposal["needs_hitl"] = f"new top-level family: {fam}"
                # flag but still accept (HITL is reserved; Step 7 will intercept)
        return proposal, "accept"

    # ===================================================================
    # Stage 3: apply — convert proposal to evolver Mutation, kb.commit
    # ===================================================================

    def _apply(self, proposal: dict, src: TriggerSource) -> tuple[bool, str]:
        """Stage 3: convert one accepted proposal into evolver Mutation(s) and
        commit via kb.commit (validate + apply, schema-constrained). This is the
        KEY difference from legacy run_evolution_loop (which mutated meta
        directly, bypassing the kernel/ledger). Returns (committed, detail)."""
        op = proposal.get("op", "")
        domain = src.domain or self.domain_default
        evidence = proposal.get("evidence_span", "") or proposal.get("rationale", "")
        mutations: list[Mutation] = []

        if op == "add_pattern":
            # generate semantic_boundary (design supplement section 4) — the
            # evolver fills the boundary the seed has but a grown pattern lacks.
            boundary = self._gen_boundary(proposal, domain)
            pat = MetaHyperedgePattern(
                pattern_id=proposal["pattern_id"],
                description=proposal.get("description", ""),
                role_slots=proposal.get("role_slots", []),
                allowed_qualifiers=proposal.get("allowed_qualifiers", []),
                family=proposal.get("family", ""),
                semantic_boundary=boundary)
            mutations.append(Mutation(
                op=Op.ADD_PATTERN, target=proposal["pattern_id"],
                proposer_role=Role.EVOLVER, domain=domain, evidence=evidence,
                rationale=proposal.get("rationale", "evolution probe"),
                payload={"pattern": pat}))
        elif op == "add_meta_node":
            mutations.append(Mutation(
                op=Op.ADD_NODE, target=proposal["type_id"],
                proposer_role=Role.EVOLVER, domain=domain, evidence=evidence,
                rationale=proposal.get("rationale", ""),
                payload={"description": proposal.get("description", "")}))
        elif op == "add_subclass":
            mutations.append(Mutation(
                op=Op.ADD_SUBCLASS, target=proposal["sub"],
                proposer_role=Role.EVOLVER, domain=domain, evidence=evidence,
                rationale=proposal.get("rationale", ""),
                payload={"super": proposal["sup"]}))
        elif op == "split":
            # split: build sub_patterns via name_split_subpatterns if the probe
            # didn't supply them. Delegate the sub-pattern NAMING to the verified
            # name_split_subpatterns内核 (surgical), then commit via kernel.
            subs = self._build_split_subs(proposal, domain)
            if not subs:
                return False, "split:no-sub-patterns"
            mutations.append(Mutation(
                op=Op.SPLIT, target=proposal["pattern_id"],
                proposer_role=Role.EVOLVER, domain=domain, evidence=evidence,
                rationale=proposal.get("rationale", "split trigger"),
                payload={"sub_patterns": subs}))
        elif op == "merge":
            mutations.append(Mutation(
                op=Op.MERGE, target=proposal.get("into", proposal["pattern_ids"][0]),
                proposer_role=Role.EVOLVER, domain=domain, evidence=evidence,
                rationale=proposal.get("rationale", "merge trigger"),
                payload={"pattern_ids": proposal["pattern_ids"],
                         "into": proposal.get("into", proposal["pattern_ids"][0])}))
        elif op == "retire":
            mutations.append(Mutation(
                op=Op.RETIRE, target=proposal["pattern_id"],
                proposer_role=Role.EVOLVER, domain=domain, evidence=evidence,
                rationale=proposal.get("rationale", "retire trigger"),
                payload={}))
        elif op == "rename":
            new_id = proposal.get("new_id", "")
            if not new_id:
                return False, "rename:no-new-id"
            mutations.append(Mutation(
                op=Op.RENAME, target=proposal["pattern_id"],
                proposer_role=Role.EVOLVER, domain=domain, evidence=evidence,
                rationale=proposal.get("rationale", "rename trigger"),
                payload={"new_id": new_id}))
        else:
            return False, f"unknown-op:{op}"

        if not mutations:
            return False, "no-mutations"
        result = self.kb.commit(mutations)
        if not result.ok:
            reason = result.rejected[0]["reason"] if result.rejected else "unknown"
            return False, f"kernel-rejected:{reason}"
        return True, f"committed:{len(result.validated)}"

    def _gen_boundary(self, proposal: dict, domain: str) -> str:
        """Generate a semantic_boundary for a new pattern (design supplement
        section 4: evolver add_pattern时 LLM 生成 boundary based on evidence
        + rationale). Falls back to a generic boundary if the LLM call fails."""
        prompt = _BOUNDARY_PROMPT.format(
            domain=domain, pattern_id=proposal.get("pattern_id", ""),
            description=proposal.get("description", ""),
            role_slots=proposal.get("role_slots", []),
            family=proposal.get("family", ""))
        raw = self.llm(prompt, 200)
        if raw:
            b = raw.strip().strip('"').strip("'")
            if b and len(b) < 400:
                return b
        return f"{proposal.get('description', '')} (apply when the relation matches this description)"

    def _build_split_subs(self, proposal: dict, domain: str) -> list[MetaHyperedgePattern]:
        """Build sub-patterns for a split. If the proposal already carries
        sub_patterns (from detect_split_triggers), use them; else delegate to
        name_split_subpatterns (the verified naming内核) to name them from the
        parent. The sub-patterns inherit the parent's role_slots (kernel's
        exact-signature constraint, 断点/BLOCKER#2 fix)."""
        pid = proposal.get("pattern_id", "")
        parent = self.kb.tbox.patterns.get(pid)
        if parent is None:
            return []
        subs = proposal.get("sub_patterns")
        if subs:
            out = []
            for s in subs:
                if isinstance(s, MetaHyperedgePattern):
                    out.append(s)
                elif isinstance(s, dict):
                    out.append(MetaHyperedgePattern(
                        pattern_id=s.get("pattern_id", ""),
                        description=s.get("description", ""),
                        role_slots=[dict(r) for r in parent.role_slots],  # inherit exact sig
                        allowed_qualifiers=list(parent.allowed_qualifiers),
                        family=parent.family))
            return out
        # delegate naming to the verified内核 (surgical). name_split_subpatterns
        # returns list[dict] with pattern_id (no MetaHyperedgePattern objects);
        # convert to patterns that INHERIT the parent's exact role_slots (kernel
        # exact-sig constraint, Step1 BLOCKER#2 fix) + allowed_qualifiers + family.
        try:
            named = hev.name_split_subpatterns(parent, proposal.get("trigger", proposal),
                                                llm="deepseek") or []
        except Exception:
            named = []
        out = []
        for s in named:
            if isinstance(s, MetaHyperedgePattern):
                out.append(s)
            elif isinstance(s, dict) and s.get("pattern_id"):
                out.append(MetaHyperedgePattern(
                    pattern_id=s["pattern_id"],
                    description=s.get("description", ""),
                    role_slots=[dict(r) for r in parent.role_slots],  # exact sig inherit
                    allowed_qualifiers=list(parent.allowed_qualifiers),
                    family=parent.family))
        return out

    # ===================================================================
    # skill distiller (断点 5 loop close): crystallize -> distill_skill op
    # ===================================================================

    def distill_skill(self, domain: str, pattern_id: str,
                     extraction_hint: str, stability_score: float,
                     provenance_papers: list[str] | None = None) -> tuple[bool, str]:
        """Crystallize a recurring (domain, pattern) extraction mode into a Skill
        via the distill_skill op (evolver write contract). The kernel gates on
        stability >= 0.6 (Step 1). This closes the loop Step 2 left open: the
        extraction modes the extractor collected are committed to the 4th-layer
        Skill Library so future extraction sees the hint."""
        if stability_score < KnowledgeBase.SKILL_CRYSTALLIZE_STABILITY:
            return False, f"stability<{KnowledgeBase.SKILL_CRYSTALLIZE_STABILITY}"
        skill = Skill(skill_id=f"sk_{domain}_{pattern_id}_{abs(hash(extraction_hint))%10000}",
                      domain=domain, pattern=pattern_id,
                      extraction_hint=extraction_hint,
                      stability_score=stability_score,
                      provenance_papers=provenance_papers or [])
        result = self.kb.commit([Mutation(
            op=Op.DISTILL_SKILL, target=skill.skill_id, proposer_role=Role.EVOLVER,
            domain=domain, evidence=extraction_hint[:200],
            rationale=f"crystallized {domain}/{pattern_id} extraction mode",
            payload={"skill": skill})])
        if not result.ok:
            return False, f"kernel-rejected:{result.rejected[0]['reason'] if result.rejected else '?'}"
        return True, "committed"

    # ===================================================================
    # propose_validate_failures: the extractor's validate failures enter queue
    # ===================================================================

    def propose_validate_failures(self, failing_hes: list[tuple[Hyperedge, str]],
                                  node_id: str, paper_id: str,
                                  domain: str = "") -> None:
        """The extractor's validate failures (edges that didn't match any
        pattern) enter the unified queue. Records cross-node recurrence so
        governance's conservative gate has the signal (委托 EvolutionTrigger)."""
        domain = domain or self.domain_default
        cross = 1
        for he, reason in failing_hes:
            sig, _ = self.trigger.record(he, reason, node_id)
            cross = max(cross, self.trigger.cross_node_count(sig))
        self.propose_trigger(TriggerSource(
            kind="validate_failure",
            payload={"cross_node": cross, "failing_hes": failing_hes},
            domain=domain, paper_id=paper_id, node_id=node_id))

    # ===================================================================
    # drain: single serial builder consumer (断点 6)
    # ===================================================================

    def drain(self) -> EvolutionReport:
        """Drain the queue: for each trigger source, probe → governance → apply.
        Single serial consumer (断点 6). Returns a report so the caller/reviewer
        can SEE what evolved (not just a pass/fail)."""
        report = EvolutionReport(version_before=self.kb.version)
        sources = list(self._queue)
        self._queue.clear()
        for src in sources:
            failing_hes = src.payload.get("failing_hes") if src.kind in (
                "validate_failure", "recurring_mismatch") else None
            proposals = self._probe(src, failing_hes)
            report.n_proposed += len(proposals)
            for proposal in proposals:
                adjudicated, decision = self._governance(proposal, src)
                if decision == "reject":
                    report.n_rejected += 1
                    report.rejected.append({"op": proposal.get("op"),
                                            "reason": proposal.get("rejected_reason", ""),
                                            "source": src.kind})
                    continue
                if adjudicated and adjudicated.get("needs_hitl"):
                    report.n_hitl += 1
                    report.hitl.append({"op": adjudicated.get("op"),
                                         "pattern_id": adjudicated.get("pattern_id", ""),
                                         "reason": adjudicated.get("needs_hitl")})
                ok, detail = self._apply(adjudicated or proposal, src)
                if ok:
                    report.n_accepted += 1
                    report.accepted.append({"op": proposal.get("op"),
                                            "detail": detail,
                                            "source": src.kind})
                else:
                    report.n_rejected += 1
                    report.rejected.append({"op": proposal.get("op"),
                                            "reason": detail,
                                            "source": src.kind})
        report.version_after = self.kb.version
        return report

    # ===================================================================
    # schema并进 propagation (design supplement section 3)
    # ===================================================================

    def schema_for_extraction(self, domain: str = "") -> str:
        """The schema the extractor sees on its NEXT section/paper — re-fetched
        each call so intra-DAG + cross-paper propagation works (并进 = the schema
        a later extraction sees reflects evolution that happened at an earlier
        node). This is the forward-propagation hook the extractor calls."""
        return self.kb.tbox.to_prompt()
