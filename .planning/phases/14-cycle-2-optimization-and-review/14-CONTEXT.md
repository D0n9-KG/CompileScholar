# Phase 14: Cycle 2 Optimization And Review - Context

**Gathered:** 2026-04-04 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 14 stays on the same bounded computational-mechanics slice and the same `package -> replay -> prior-review -> export` loop used in Phases 12 and 13. The phase may run multiple bounded optimize -> rerun cycles, but it only counts when one chosen cycle is manually judged genuinely useful as scientific-thinking training data and its machine-readable export becomes self-contained enough to review without cross-file reconstruction. It does not widen packet scope, replace the runtime surface, or bypass audit/review truth with replay-only shortcuts.
</domain>

<decisions>
## Implementation Decisions

### Execution Surface And Baseline Boundary
- **D-01:** Phase 14 should keep using the committed Phase 9 packet and assembly manifest plus the existing Phase 10/13 runner surface (`backend/scripts/run_phase13_cycle_refinement.py` over `backend/app/research_logic/phase10_multi_paper_validation.py`). New iterations should stay cycle-scoped under a fresh `tmp/phase14_*` root rather than inventing a new execution path.
- **D-02:** Phase 14 should compare candidate cycles against the fixed Phase 12 cycle-1 baseline while using the Phase 13 cycle-2 final review as the live blocker handoff. The bounded input boundary stays frozen; quality claims must come from improved outputs on the same slice.

### Export Truth And Prior Selection
- **D-03:** The audited export must remain a post-replay assembly driven by reviewed accepted ids. Phase 14 should not copy the green replay `decision_episode` directly into export as a shortcut.
- **D-04:** Phase 14 should close the current prior-selection gap by making reviewed priors actually applicable to the exported primary route state or by recording explicit structured exclusion rationale when accepted priors still cannot be selected. Export truth must remain consistent with review truth.

### Training-Facing Artifact Shape
- **D-05:** Phase 14 should extend the machine-readable export bundle so the chosen best cycle publishes a self-contained training-facing view alongside the current audit-grade episode, keeping evidence pack, why-now, route comparison, prior/anti-pattern posture, minimal attack path, final decision, and review verdicts together without markdown-only reconstruction.
- **D-06:** Phase 14 should add structured human review and training-acceptance fields inside JSON artifacts for the chosen best cycle. Markdown reports can still render the same judgment, but the machine-readable bundle becomes the authoritative carrier of training-acceptance evidence.

### Canonical Labels And Section Coverage
- **D-07:** Phase 14 should preserve canonical labels together with raw source phrases for route features, bottlenecks, conditions, capabilities, and similar training-relevant concept fields. This should be solved at the route/export schema boundary instead of relying on normalized labels alone.
- **D-08:** Phase 14 should keep every major training-data section in optimization scope and review explicitly across each candidate cycle. It should not optimize only blocker queues or a single stage while treating evidence pack, why-now, route comparison, priors, anti-patterns, attack path, final decision, or review labels as frozen background.

### the agent's Discretion
- Exact names and layouts for new training-facing JSON files, as long as they follow the repo's existing bundle/report pattern and clearly distinguish audit-only vs training-facing views.
- Whether the prior-selection fix lands by broadening reviewed prior applicability to the primary route, by carrying route-family-aware matching where justified, or by keeping non-selection but adding explicit structured exclusion records, as long as export truth stays review-backed and auditable.
- Exact field names and enum values for machine-readable review / training-acceptance metadata, as long as the reviewer verdict, rationale, and residual defects are explicit.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project And Phase Framing
- `.planning/PROJECT.md` - Milestone-level requirement that outputs become auditable, replayable, trainable scientific-reasoning artifacts
- `.planning/REQUIREMENTS.md` - Phase 14 requirements `REVIEW-02`, `REVIEW-03`, `TRAIN-01`, `TRAIN-02`, `TRAIN-03`, and `TRAIN-06`
- `.planning/ROADMAP.md` - Phase 14 goal, success criteria, and the rule to keep iterating on the same bounded slice until one cycle is manually judged good enough
- `.planning/STATE.md` - Current handoff from Phase 13 and the active focus on training-facing export quality
- `.planning/phases/12-baseline-data-cycle-1-direct-fix/12-CONTEXT.md` - Locks the bounded packet-first direct-fix boundary and the Phase 12 baseline contract
- `.planning/phases/13-baseline-data-cycle-2-refine/13-CONTEXT.md` - Locks the same bounded rerun surface, cycle-1 comparison anchor, and downstream fix focus

### Current Evidence Surface
- `docs/replay/reports/phase13-cycle2-refine.md` - Human review of the current best cycle and the explicit statement that export is still not training-ready
- `docs/replay/reports/phase13-next-cycle-prioritization.md` - Formal next-cycle recommendation and rationale that still ranks `packet_construction` first
- `tmp/phase13_cycle2_refine/cycle2-final/comparison_summary.json` - Current stage-by-stage delta and blocker queue for package, replay, prior review, and export
- `tmp/phase13_cycle2_refine/cycle2-final/replay_bundle/replay_summary.json` - Replay readiness and failure-count surface for the current best cycle
- `tmp/phase13_cycle2_refine/cycle2-final/replay_bundle/replay_inspection.json` - Detailed replay-stage quality and minimal-attack-path evidence
- `tmp/phase13_cycle2_refine/cycle2-final/replay_bundle/outputs/decision_episode.json` - Replay-time green `DecisionEpisode` that shows the current route-state-facing prior selection
- `tmp/phase13_cycle2_refine/cycle2-final/prior_review_bundle/candidate_review_summary.json` - Accepted prior/anti-pattern summary and fallback-cluster status
- `tmp/phase13_cycle2_refine/cycle2-final/prior_review_bundle/prior_candidates.json` - Exact accepted prior payload and support-route-state coverage
- `tmp/phase13_cycle2_refine/cycle2-final/export_bundle/export_summary.json` - Current export readiness summary showing `weak_prior_support` and empty `selected_prior_ids`
- `tmp/phase13_cycle2_refine/cycle2-final/export_bundle/export_inspection.json` - Current export visibility buckets, audit posture, and prior-selection notes

### Runtime And Schema Contracts
- `backend/scripts/run_phase13_cycle_refinement.py` - Existing Phase 13/14 rerun entrypoint that keeps the bounded iteration surface flat
- `backend/app/research_logic/phase10_multi_paper_validation.py` - Canonical package/replay/prior/export orchestration and comparison-summary builder
- `backend/app/research_logic/historical_replay_compiler.py` - Replay compiler and current route/why-now/comparison/prior/episode generation path
- `backend/app/research_logic/decision_episode_builder.py` - Route-state-facing prior selection and final `DecisionEpisode` construction rules
- `backend/app/research_logic/decision_episode_export.py` - Reviewed-id allowlisting, visibility buckets, and audit export assembly
- `backend/app/research_logic/replay_io.py` - Current `export_summary.json` and `export_inspection.json` bundle shape
- `backend/app/research_logic/prior_induction.py` - Support-cluster logic, fallback-scope merge behavior, and accepted prior generation
- `backend/app/research_logic/iteration_prioritization.py` - Current blocker-taxonomy and next-cycle ranking surface
- `backend/app/research_logic/models.py` - Schema definitions for route, why-now, comparison, prior, anti-pattern, and decision episode objects
- `backend/app/research_logic/route_state_synthesizer.py` - Current route-state label normalization and aggregation path that affects canonical/raw phrase preservation

### Specs And Guardrails
- `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md` - Layer boundaries that Phase 14 must preserve while optimizing export shape
- `docs/superpowers/specs/2026-04-01-logickg-decision-prior-and-episode-schema.md` - Canonical `DecisionPriorCard`, `AntiPatternCard`, and `DecisionEpisode` contract
- `docs/superpowers/specs/2026-04-01-logickg-why-now-and-route-comparison-schema.md` - Canonical `WhyNowCase` and `RouteComparisonCase` contract
- `.planning/phases/06-decision-episode-audit-export/06-RESEARCH.md` - Prior research guardrail for keeping audited export truth separate from replay-only truth

### Regression Coverage
- `backend/tests/test_phase10_multi_paper_validation.py` - Existing end-to-end bundle tests and selected-prior subset guardrails
- `backend/tests/test_decision_episode_export.py` - Export-specific tests for reviewed-id allowlisting, anti-pattern carryover, and hindsight visibility
- `backend/tests/test_prior_induction.py` - Fallback-cluster behavior that produced the current accepted prior
- `backend/tests/test_historical_replay_compiler.py` - Replay-time prior selection and `weak_prior_support` expectations
- `backend/tests/test_replay_io.py` - Bundle-writing tests for export summary/inspection structure and visibility buckets
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `backend/scripts/run_phase13_cycle_refinement.py`: already provides the bounded rerun entrypoint Phase 14 can continue using
- `backend/app/research_logic/phase10_multi_paper_validation.py`: already assembles package, replay, prior review, export, comparison summary, and report outputs in one auditable path
- `backend/app/research_logic/decision_episode_export.py`: already knows how to keep audit-only refs separate from visible input refs and label-only hindsight refs
- `backend/app/research_logic/replay_io.py`: already writes machine-readable export bundle files and summary/inspection JSON
- `backend/app/research_logic/prior_induction.py`: already produces reviewed accepted priors and exposes the fallback-cluster strategy used in Phase 13
- `backend/app/research_logic/iteration_prioritization.py`: already converts cycle artifacts into the next-cycle recommendation queue

### Established Patterns
- Machine-readable JSON under `tmp/` is the source of truth; markdown under `docs/replay/reports/` is rendered from those artifacts
- Export is rebuilt from replay and prior-review bundles, not copied directly from replay outputs
- Quality posture is carried through `quality_tier`, readiness booleans, and explicit `quality_flags`
- Bounded packet membership and baseline bundle references remain stable across iterative phases so quality movement is auditable

### Integration Points
- The chosen Phase 14 cycle root should feed `comparison_summary.json`, updated markdown review, and any refreshed prioritization handoff
- The export bundle is the natural home for new self-contained training-facing files and structured review / training-acceptance metadata
- Prior-selection fixes likely touch `prior_induction.py`, `decision_episode_builder.py`, `decision_episode_export.py`, and the Phase 10 validation/export assembly path
- Canonical-label-plus-raw-phrase preservation likely touches route-state schema/builders first, then the export surface that packages those fields for training use
</code_context>

<specifics>
## Specific Ideas

- The live Phase 13 mismatch is that `tmp/phase13_cycle2_refine/cycle2-final/replay_bundle/outputs/decision_episode.json` is green and includes `selected_prior_ids = ["prior:data_driven_constitutive_and_multiscale_computational_mechanics:2021"]`, while `tmp/phase13_cycle2_refine/cycle2-final/export_bundle/export_summary.json` is yellow because the reviewed accepted fallback-cluster prior does not support the exported primary route state.
- The current export bundle only writes `outputs/decision_episode.json`, `export_summary.json`, and `export_inspection.json`; there is no dedicated machine-readable training-facing artifact yet.
- A targeted search of the current runtime, docs, tests, and Phase 13 outputs did not surface an existing structured `training_acceptance` field family, so Phase 14 should treat this as new schema/bundle work rather than hidden plumbing.
- The current export already separates `visible_input_refs`, `audit_only_refs`, and `label_eval_only_refs`, which is a strong foundation for a self-contained training-facing view that still honors cutoff and leakage policy.
</specifics>

<deferred>
## Deferred Ideas

- Widening packet membership or changing the topic slice just to reach a better-looking result faster
- Full multiple-task training-view publication across the whole bounded dataset bundle; later phases still own the full multi-view closure target
- UI/productization of review or export operations inside this phase
- Loosening audit/review gates by treating replay-only green outputs as automatically export-ready
</deferred>

---

*Phase: 14-cycle-2-optimization-and-review*
*Context gathered: 2026-04-04*
