# Phase 13: Baseline Data Cycle 2 (Refine) - Context

**Gathered:** 2026-04-04 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 13 continues the same bounded computational-mechanics slice from Phase 12 and is allowed to run multiple bounded refinement iterations inside the phase so the team can verify that observed quality movement is repeatable rather than a one-pass artifact. It still stays on the same packet boundary and existing rerun pipeline, and it does not widen scope, introduce a new runtime foundation, or claim milestone-level stability early.

</domain>

<decisions>
## Implementation Decisions

### Baseline Anchor And Comparison Boundary
- **D-01:** `tmp/phase12_direct_fix_cycle/cycle1/` is the canonical baseline root for Phase 13. New reruns should compare against that cycle-1 replay/export surface rather than falling back to the older Phase 10 / Phase 11 evidence chain.
- **D-02:** Final Phase 13 judgment should compare the strongest Phase 13 iteration against the cycle-1 defect surface. Older Phase 10 / 11 artifacts remain historical provenance, not the primary success baseline for this phase.

### Iteration Cadence Within Phase 13
- **D-03:** Phase 13 is intentionally allowed to iterate multiple times within the phase, not just run one rerun/review pass, as long as every iteration stays on the same bounded packet and leaves auditable artifacts.
- **D-04:** The phase should close only after the team has enough repeated bounded evidence to judge whether quality movement is real, not just the result of one micro-fix. Multiple iterations are therefore part of the phase design, not accidental spillover.

### Execution Surface And Scope Lock
- **D-05:** Every Phase 13 iteration should reuse the existing Phase 10 execution surface (`package -> replay -> prior-review -> export`) through `backend/scripts/run_phase10_multi_paper_validation.py` or a thin wrapper around the same runtime path.
- **D-06:** The bounded packet boundary remains frozen on the committed Phase 9 packet and assembly manifest. Phase 13 should not widen packet membership, change the topic slice, or bypass the current bundle/report contract while iterating.

### Downstream Fix Focus
- **D-07:** The lead refinement surface is now downstream of packet construction: target `reviewer_missing`, the remaining `decision_prior_card` failure, empty prior candidates / accepted prior ids, and `weak_prior_support`.
- **D-08:** `yellow_route_state_present` should remain explicit as a residual package-quality signal, but planners should not reopen already-moved blockers such as `support_cluster_too_small` and `alternative_scope_not_distinct` unless a later iteration shows regression.
- **D-09:** Because cycle 1 produced three support clusters with support-count `1` each, Phase 13 should treat support clustering shape and reviewer metadata as first-class levers for unlocking usable priors instead of expecting export quality to improve automatically.

### Review And Handoff Contract
- **D-10:** Each iteration should write machine-readable artifacts under a fresh cycle-scoped `tmp/` root and keep operator-facing markdown reports under `docs/replay/reports/`, preserving the repo's summary-first audit pattern.
- **D-11:** Phase closeout must include an explicit manual defect-delta review across package, replay, prior-review, and export, using cycle-1 defects as the canonical reference point for whether the phase actually moved quality.
- **D-12:** The phase should end with a ranked next-cycle recommendation using the existing blocker / prioritization taxonomy, based on the final iteration's evidence instead of a freeform narrative-only handoff.

### the agent's Discretion
- Exact per-iteration labels and directory names inside Phase 13, as long as they remain cycle-scoped and auditable.
- Whether intermediate working notes compare only against the immediately previous micro-iteration or also against cycle 1, as long as the final phase verdict still anchors to the Phase 12 cycle-1 baseline.
- Whether prioritization is refreshed after every micro-iteration or only after the final Phase 13 iteration, as long as the final handoff is based on the strongest current evidence.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project And Phase Framing
- `.planning/PROJECT.md` - `v1.2` quality-stabilization loop, packet-first philosophy, and manual-review-first milestone framing
- `.planning/REQUIREMENTS.md` - `LOOPR-03` and `REVIEW-01`, including rapid fix -> rerun loops and explicit manual review on real outputs
- `.planning/ROADMAP.md` - Phase 13 goal, success criteria, and the same-baseline refinement boundary
- `.planning/STATE.md` - Current handoff, cycle-1 baseline anchor, and named downstream blockers
- `.planning/phases/12-baseline-data-cycle-1-direct-fix/12-CONTEXT.md` - Locked Phase 12 boundary and the decision that cycle 1 becomes the forward baseline
- `.planning/phases/12-baseline-data-cycle-1-direct-fix/12-VERIFICATION.md` - Exact Phase 12 rerun command and verified blocker movement

### Cycle-1 Source Of Truth
- `docs/replay/reports/phase12-cycle1-direct-fix.md` - Human-readable defect delta and verdict that quality shifted downstream after packet repair
- `tmp/phase12_direct_fix_cycle/cycle1/comparison_summary.json` - Canonical cycle-1 stage comparison and blocker queue
- `tmp/phase12_direct_fix_cycle/cycle1/replay_bundle/replay_summary.json` - Replay quality flags and stage/layer failure counts for cycle 1
- `tmp/phase12_direct_fix_cycle/cycle1/replay_bundle/replay_inspection.json` - Failure records, stage quality, and minimal-attack-path evidence for cycle 1
- `tmp/phase12_direct_fix_cycle/cycle1/prior_review_bundle/candidate_review_summary.json` - Evidence that prior-review stayed empty and support states clustered as three single-support groups
- `tmp/phase12_direct_fix_cycle/cycle1/export_bundle/export_summary.json` - Export quality and `weak_prior_support` surface for cycle 1

### Runtime And Comparison Contracts
- `backend/app/research_logic/phase10_multi_paper_validation.py` - Canonical rerun pipeline, baseline override support, comparison summary generation, and report rendering
- `backend/scripts/run_phase10_multi_paper_validation.py` - Operator-facing CLI for repeated bounded reruns
- `backend/app/research_logic/replay_io.py` - Summary-first bundle writing contract for replay, prior-review, and export artifacts
- `backend/tests/test_phase10_multi_paper_validation.py` - Contract coverage for bundle generation, baseline wiring, and comparison outputs

### Prior / Review / Prioritization Logic
- `backend/app/research_logic/historical_replay_compiler.py` - `reviewer_missing` blocking logic and replay readiness gating
- `backend/app/research_logic/prior_induction.py` - Cluster-driven prior candidate registry behavior used by prior review
- `backend/app/research_logic/decision_prior_builder.py` - Conditions that make a decision prior candidate usable or still weak
- `backend/app/research_logic/anti_pattern_builder.py` - Reviewer-linked anti-pattern quality behavior
- `backend/app/research_logic/iteration_prioritization.py` - Existing blocker taxonomy and next-cycle ranking contract
- `backend/tests/test_historical_replay_compiler.py` - Coverage showing reviewer ids and support-cluster shape affect replay readiness

### Stable Layer And Packet Contracts
- `docs/replay/pilot_packets/phase9-route-packet.json` - Frozen bounded packet Phase 13 must continue to use
- `docs/replay/pilot_packets/phase9-assembly-manifest.json` - Frozen `support / alternative / held_out` grouping for the bounded slice
- `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md` - Layer boundaries that Phase 13 must preserve while refining
- `docs/superpowers/specs/2026-04-01-logickg-route-packet-schema.md` - Route-packet contract that remains stable across iterations

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `backend/app/research_logic/phase10_multi_paper_validation.py`: already regenerates package, replay, prior-review, export, and comparison artifacts for the bounded packet while accepting baseline bundle overrides
- `backend/scripts/run_phase10_multi_paper_validation.py`: existing CLI surface for repeated bounded reruns
- `backend/app/research_logic/historical_replay_compiler.py`: replay compiler already supports `reviewer_ids` and explicitly records `reviewer_missing` as a blocking failure when metadata is absent
- `backend/app/research_logic/prior_induction.py`: prior registry groups support route states into clusters and only emits reusable prior candidates when cluster structure is strong enough
- `backend/app/research_logic/iteration_prioritization.py`: existing ranking layer for turning blocker surfaces into next-cycle recommendations

### Established Patterns
- The repo treats machine-readable summary / inspection JSON under `tmp/` as the source of truth and renders operator-facing markdown reports under `docs/replay/reports/`
- Bundle manifests and comparison summaries are preferred over prose-only reporting
- Quality flags such as `reviewer_missing`, `weak_prior_support`, and stage-specific blocker codes stay explicit in structured outputs instead of being buried in narrative text
- Existing rerun surfaces are reused across phases instead of introducing a new execution framework for each cycle

### Integration Points
- New Phase 13 iterations should pass the Phase 12 cycle-1 replay/export bundles into the baseline args of the existing rerun flow
- Each iteration should write to a fresh cycle-scoped output root so defect deltas stay auditable
- Manual review reports should continue to live under `docs/replay/reports/` and cite the machine-readable cycle outputs directly
- Final Phase 13 evidence can feed the existing iteration-prioritization layer for the next-cycle handoff

</code_context>

<specifics>
## Specific Ideas

- The overall workflow is correct, but Phase 13 should support multiple bounded iterations within the phase instead of defaulting to exactly one rerun.
- The packet boundary remains `phase9_comp_mech_2021_packet_01`; the phase should refine the same bounded slice rather than rotating to a new topic packet.
- The current empty prior surface is explained by both missing reviewer metadata and the fact that cycle 1 produced three single-support clusters instead of one reusable support cluster.
- Final closeout should still judge the best Phase 13 output against the Phase 12 cycle-1 baseline, even if the phase uses several intermediate micro-iterations.

</specifics>

<deferred>
## Deferred Ideas

- Widening the bounded packet with new papers just to mask downstream blockers
- Building a separate runtime foundation or operator UI for cycle management in this phase
- Declaring milestone-level stability inside Phase 13; repeated high-quality-cycle proof still belongs to the later stability phases
- Generalizing findings beyond the current bounded computational-mechanics slice

</deferred>

---

*Phase: 13-baseline-data-cycle-2-refine*
*Context gathered: 2026-04-04*
