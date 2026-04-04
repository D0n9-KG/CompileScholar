# Phase 15: Cycle 3 Consolidation - Context

**Gathered:** 2026-04-04 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 15 continues the same bounded computational-mechanics slice and the same `package -> replay -> prior-review -> export` loop used in Phases 12 through 14. The phase stays focused on turning the completed Phase 14 export structure into the first newly generated cycle that direct manual review would actually call high-quality scientific-thinking training data, while also closing reviewed prior / anti-pattern knowledge into the final export fields or explicit exclusion records. It does not widen the packet, replace the runtime surface, or claim milestone-level consecutive stability early.
</domain>

<decisions>
## Implementation Decisions

### Iteration Surface And Comparison Boundary
- **D-01:** Phase 15 should keep using the bounded Phase 9 packet and the existing rerun surface (`backend/scripts/run_phase13_cycle_refinement.py` over `backend/app/research_logic/phase10_multi_paper_validation.py`) with fresh cycle-scoped outputs under a new `tmp/phase15_*` root. The phase should not introduce a new execution framework or broaden packet membership.
- **D-02:** The primary quality jump for Phase 15 must be judged directly against the completed Phase 14 `cycle2-best` review and artifacts, because that is the current best-cycle handoff and the official baseline for this phase. Phase 13 `cycle2-final` remains important provenance for stronger prior availability, but it is supporting evidence rather than the closeout baseline.

### Prior / Anti-Pattern Closure
- **D-03:** Export truth must remain route-backed and review-backed. Phase 15 should only land `selected_prior_ids` or selected anti-pattern ids in the audited export when the accepted reviewed cards truly support the exported primary route state; otherwise the cycle must preserve explicit machine-readable exclusion rationale instead of forcing the ids through.
- **D-04:** Because Phase 13's strongest prior surface came from the explicit `allow_scope_fallback_merge` path while Phase 14's default reruns reverted to `cluster_strategy = default` and produced zero accepted priors, Phase 15 should treat fallback-merge prior recovery as a deliberate, auditable comparison lever. Planning should compare that lever against upstream packet/route-state improvements instead of silently flipping defaults or ignoring the regression.

### Route Comparison And Packet Quality Focus
- **D-05:** `packet_construction` remains the lead recommendation for this phase. The first high-quality cycle should come from improving upstream packet / relation-assembly / route-state distinctness strongly enough that route comparison can emit a grounded advantage explanation and prior induction has real support to work with, not from more export-only packaging work.
- **D-06:** Route-comparison improvement is a first-class success surface for Phase 15. A candidate cycle should not be considered genuinely good unless the bounded artifact explains why the chosen route wins against the nearby alternative with reviewable evidence, instead of leaving `recommended_route_state_id` and `route_advantage_summary` empty while quality remains `yellow`.

### Review And Stabilization Evidence
- **D-07:** Phase 15 closes on the first newly generated bounded cycle that manual review judges genuinely useful as training data. That verdict must be captured through the repo's current review handoff pattern: cycle machine outputs under `tmp/`, operator-facing markdown under `docs/replay/reports/`, and best-cycle selection / recommendation evidence recorded in `best_cycle_selection.json` and the rendered report chain.
- **D-08:** Phase 15 should make the quality jump explicit by comparing the new good cycle against the completed Phase 14 best-cycle review and calling out whether prior closure, route-comparison clarity, and residual `yellow_route_state_present` / `weak_prior_support` defects actually moved. Consecutive-cycle stability proof is still deferred to Phase 16 and should not be claimed here.

### the agent's Discretion
- Exact cycle labels, output-root names, and report filenames under the new Phase 15 directory structure, as long as they stay cycle-scoped and auditable.
- Whether the winning Phase 15 cycle closes the prior gap through a recovered fallback-merge path, stronger packet-backed support, or a cleaner explicit exclusion rationale, as long as export truth remains route-backed and the reason is machine-readable.
- Exact manual-review note structure inside the best-cycle report, as long as it explicitly judges training usefulness from direct artifact reading and compares the result back to Phase 14.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project And Phase Framing
- `.planning/PROJECT.md` - Milestone-level goal, packet-first quality-recovery philosophy, and the requirement to end with genuinely trainable scientific-reasoning artifacts
- `.planning/REQUIREMENTS.md` - Active Phase 15 requirements `STAB-01` and `TRAIN-04`
- `.planning/ROADMAP.md` - Phase 15 goal, success criteria, and the explicit boundary between first-good-cycle closure and later stability verification
- `.planning/STATE.md` - Current handoff that points Phase 15 at the completed Phase 14 best-cycle review and prioritization bundle

### Prior Phase Handoff
- `.planning/phases/13-baseline-data-cycle-2-refine/13-CONTEXT.md` - Locks the bounded rerun surface and documents the fallback-cluster prior recovery path that produced the strongest prior availability so far
- `.planning/phases/14-cycle-2-optimization-and-review/14-CONTEXT.md` - Locks the additive training-view/export-structure work and the rule that export truth must stay review-backed
- `.planning/phases/14-cycle-2-optimization-and-review/14-VERIFICATION.md` - Verifies the final Phase 14 commands, chosen best-cycle root, and residual defects
- `docs/replay/reports/phase14-cycle2-optimization-best.md` - Human review of the current best-cycle content surface and why it is still only `needs_revision`
- `docs/replay/reports/phase14-next-cycle-prioritization.md` - Formal next-cycle recommendation that keeps `packet_construction` first

### Live Evidence Roots
- `tmp/phase13_cycle2_refine/cycle2-final/prior_review_bundle/candidate_review_summary.json` - Proof that the explicit fallback merge path produced one accepted prior in Phase 13
- `tmp/phase13_cycle2_refine/cycle2-final/prior_review_bundle/prior_candidates.json` - The reviewed fallback-cluster prior payload and its route-support coverage
- `tmp/phase13_cycle2_refine/cycle2-final/export_bundle/export_summary.json` - Phase 13 export posture showing accepted-but-not-selected prior tension
- `tmp/phase14_cycle2_optimization/cycle2-best/export_bundle/export_summary.json` - Current export posture with `weak_prior_support` and zero accepted / selected priors
- `tmp/phase14_cycle2_optimization/cycle2-best/export_bundle/outputs/training_view.json` - Current self-contained training-facing artifact surface and missing route-comparison / prior-closure fields
- `tmp/phase14_cycle2_optimization/cycle2-best/export_bundle/best_cycle_selection.json` - Canonical best-cycle review verdict and recommendation evidence refs
- `tmp/phase14_cycle2_optimization/cycle2-best/prior_review_bundle/candidate_review_summary.json` - Current default-path prior-review output showing `cluster_strategy = default` and no accepted priors

### Runtime And Schema Contracts
- `backend/scripts/run_phase13_cycle_refinement.py` - Existing bounded rerun entrypoint, reviewer wiring, baseline bundle wiring, and optional `--allow-scope-fallback-merge` lever
- `backend/app/research_logic/phase10_multi_paper_validation.py` - Canonical package / replay / prior-review / export orchestration, comparison-summary generation, and best-cycle bundle reporting
- `backend/app/research_logic/prior_induction.py` - Cluster strategy, fallback-scope merge behavior, and accepted prior generation rules
- `backend/app/research_logic/decision_episode_builder.py` - Route-backed prior selection logic and `weak_prior_support` quality-flag behavior
- `backend/app/research_logic/decision_episode_export.py` - Reviewed-id allowlisting plus accepted-but-unselected prior exclusion rationale
- `backend/app/research_logic/route_comparison_builder.py` - Comparison quality rules, grounded preference requirements, and `yellow` / `green` comparison thresholds
- `backend/app/research_logic/historical_replay_compiler.py` - Replay failure surfaces, reviewer gating, and route-comparison assembly path
- `backend/app/research_logic/iteration_prioritization.py` - Recommendation scoring that still keeps `packet_construction` first on the current evidence
- `backend/app/research_logic/models.py` - Review metadata, training acceptance, and prior-selection schema fields consumed across the export surface

### Specs And Guardrails
- `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md` - Layer boundaries that Phase 15 must preserve while improving content quality
- `docs/superpowers/specs/2026-04-01-logickg-decision-prior-and-episode-schema.md` - Canonical prior / anti-pattern / decision-episode contract
- `docs/superpowers/specs/2026-04-01-logickg-why-now-and-route-comparison-schema.md` - Canonical why-now and route-comparison contract
- `.planning/phases/06-decision-episode-audit-export/06-RESEARCH.md` - Prior export-truth guardrail: audit export must remain review-backed rather than replay-only
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `backend/scripts/run_phase13_cycle_refinement.py`: already provides the bounded rerun entrypoint, Phase 12 baseline wiring, reviewer ids, and the explicit `--allow-scope-fallback-merge` switch
- `backend/app/research_logic/phase10_multi_paper_validation.py`: already assembles package, replay, prior-review, export, comparison summary, training view, and best-cycle selection artifacts in one auditable flow
- `backend/app/research_logic/prior_induction.py`: already exposes both the default cluster strategy and the fallback-scope-merge strategy that recovered priors in Phase 13
- `backend/app/research_logic/decision_episode_export.py`: already preserves accepted-but-unselected prior rationale when reviewed priors do not support the exported route state
- `backend/app/research_logic/route_comparison_builder.py`: already encodes the threshold between `unclear` / `tie` / `prefer_*` comparison outcomes and marks weakly grounded comparisons `yellow`
- `backend/app/research_logic/iteration_prioritization.py`: already keeps the next-cycle recommendation auditable from machine-readable evidence refs

### Established Patterns
- Machine-readable JSON under `tmp/` remains the source of truth; markdown reports under `docs/replay/reports/` render that evidence for human review
- The bounded Phase 9 packet and the same rerun surface remain fixed across Phases 12 through 15 so quality movement is comparable
- Replay-time prior selection and audited export-time prior selection are intentionally different surfaces: export may drop replay-selected priors if reviewed support does not match the exported route state
- Best-cycle review truth currently lives in the rendered report plus `best_cycle_selection.json`; runtime-generated `comparison_summary.json` and `export_summary.json` can still reflect pre-review defaults

### Integration Points
- A Phase 15 candidate cycle will connect through the existing Phase 13 runner into Phase 10 validation, then feed manual review, best-cycle selection, and prioritization surfaces already used in Phase 14
- Prior-closure work will likely touch `prior_induction.py`, `decision_episode_builder.py`, and `decision_episode_export.py` together rather than in isolation
- Route-comparison improvement will likely depend on upstream packet / route-state distinctness plus `route_comparison_builder.py`, not just post-hoc export formatting
- Final Phase 15 review should compare the winning cycle against Phase 14 using the existing comparison-summary, report, and best-cycle-selection artifacts instead of inventing a separate audit chain
</code_context>

<specifics>
## Specific Ideas

- Phase 13 `cycle2-final` accepted one reviewed fallback-cluster prior (`prior:data_driven_constitutive_and_multiscale_computational_mechanics_scope_fallback_cluster`), while Phase 14 `cycle2-best` reverted to `cluster_strategy = default` and produced zero accepted priors.
- The fallback-cluster prior itself is already review-complete and route-aware enough to recommend scoped comparison-first actions; the remaining problem is deciding when that knowledge legitimately closes into the exported primary route state.
- Phase 14's training-facing artifact is structurally strong enough for direct review, but its `route_comparison` section still leaves `recommended_route_state_id` and `route_advantage_summary` empty and its priors section still shows no accepted or selected priors on the default path.
- The final manual verdict for the Phase 14 best cycle already exists in `best_cycle_selection.json` as `reviewed` / `needs_revision`, even though some runtime summary payloads still show pre-review default values.
- A successful Phase 15 closeout should therefore look like a content-quality jump on the same bounded slice, not another schema-only improvement.
</specifics>

<deferred>
## Deferred Ideas

- Claiming consecutive-cycle stability or milestone-level dataset readiness before Phase 16 verifies it
- Widening packet membership or rotating to a different topic slice just to get an easier-looking win
- New UI / operator-surface work for cycle management
- Full multi-view training-dataset publication and final handoff packaging, which remain part of Phase 16
</deferred>

---

*Phase: 15-cycle-3-consolidation*
*Context gathered: 2026-04-04*
