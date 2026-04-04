# Phase 16: Stability Verification And Handoff - Context

**Gathered:** 2026-04-04 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 16 stays on the same bounded computational-mechanics slice and the same `package -> replay -> prior-review -> export` loop used in Phases 12 through 15. The phase must prove that the accepted Phase 15 `cycle3-best` quality bar reproduces on another bounded cycle, then publish the milestone closeout for stability, multi-view training data, final bounded dataset packaging, residual risks, and next-milestone focus. It does not widen packet membership, add new papers, switch topic slices, or replace the rerun surface.

</domain>

<decisions>
## Implementation Decisions

### Stability Reproduction Boundary
- **D-01:** Phase 16 should keep using the existing bounded rerun surface (`backend/scripts/run_phase13_cycle_refinement.py` over `backend/app/research_logic/phase10_multi_paper_validation.py`) with fresh outputs under a new `tmp/phase16_*` root. The phase should not introduce a new execution path.
- **D-02:** The accepted Phase 15 `cycle3-best` review is the primary comparison anchor for Phase 16. Stability must be judged by reproducing that accepted quality bar on the same bounded slice, not by widening the packet or rotating to new papers.
- **D-03:** Direct manual review of the generated reasoning artifacts remains the acceptance gate for `STAB-02` and `STAB-03`. Metric movement or structural green flags alone are not enough to declare stability.

### Stability Evidence And Review Truth
- **D-04:** Phase 16 should make the consecutive-cycle stability proof explicit by recording a second accepted bounded cycle and by explaining why the pair of accepted cycles is strong enough for milestone closeout.
- **D-05:** Residual defects may remain at closeout as long as the final verification and handoff explain them clearly. Phase 16 is about reproducible acceptance plus explicit residual-risk accounting, not about claiming zero-defect perfection.
- **D-06:** Runtime per-cycle artifacts and reviewed milestone-closeout truth should stay separate. Per-cycle export bundles remain immutable runtime evidence, while final reviewed stability conclusions should be published through dedicated handoff artifacts.

### Final Training Publication Surface
- **D-07:** Phase 16 should preserve the existing `decision_episode.json`, `training_view.json`, `export_summary.json`, `export_inspection.json`, and `best_cycle_selection.json` surfaces and add task-specific training views as additive outputs rather than replacements.
- **D-08:** The published task-specific views must at minimum cover route synthesis, why-now judgment, route comparison, prior / anti-pattern learning, and final decision episodes, and they should be assembled from validated later-cycle artifacts rather than from prose-only restatements.

### Final Dataset And Handoff Packaging
- **D-09:** Phase 16 should publish a separate final bounded dataset / handoff package under a fresh Phase 16 root that references the validated later-cycle artifact family, stable schema/readiness summary, residual risks, and next optimization focus. Milestone closeout should not be buried inside a single cycle export bundle.
- **D-10:** The next-milestone recommendation should carry forward the current evidence-backed lead (`packet_construction`) unless the repeated accepted cycle produces materially different reviewed evidence.

### the agent's Discretion
- Exact file names and directory layout for additive training views and the final handoff bundle, as long as they remain additive to the current export bundle contract and easy to audit.
- Whether the accepted-cycle streak is recorded in a final summary payload, a verification note, or both, as long as consecutive-cycle evidence is explicit in machine-readable and human-readable form.
- Exact schema names for the final dataset manifest and handoff payloads, as long as they clearly distinguish runtime evidence from milestone-closeout review truth.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project And Phase Framing
- `.planning/PROJECT.md` - Milestone-level goal to end with auditable, replayable, trainable scientific-reasoning artifacts
- `.planning/REQUIREMENTS.md` - Active Phase 16 requirements `STAB-02`, `STAB-03`, `TRAIN-05`, and `TRAIN-07`
- `.planning/ROADMAP.md` - Phase 16 goal, success criteria, and milestone-closeout expectations
- `.planning/STATE.md` - Current handoff pointing Phase 16 at the accepted Phase 15 `cycle3-best` baseline

### Prior Phase Handoff
- `.planning/phases/14-cycle-2-optimization-and-review/14-CONTEXT.md` - Locks the additive training-view/export structure and review-backed export truth guardrails
- `.planning/phases/15-cycle-3-consolidation/15-CONTEXT.md` - Locks the first accepted-cycle boundary and the rule that Phase 16 owns consecutive stability proof
- `.planning/phases/15-cycle-3-consolidation/15-VERIFICATION.md` - Verifies the accepted `cycle3-best` root, residual defects, and `accepted_cycle_streak = 1`
- `docs/replay/reports/phase15-cycle3-best.md` - Human review of the accepted Phase 15 best cycle and why it is accepted but not yet stable
- `docs/replay/reports/phase15-next-cycle-prioritization.md` - Current recommendation handoff that still keeps `packet_construction` first

### Live Accepted Baseline
- `tmp/phase15_cycle3_consolidation/cycle3-best/comparison_summary.json` - Machine-readable stage comparison and blocker queue for the accepted cycle
- `tmp/phase15_cycle3_consolidation/cycle3-best/export_bundle/best_cycle_selection.json` - Reviewed verdict, rationale, residual defects, and recommendation evidence refs for the accepted cycle
- `tmp/phase15_cycle3_consolidation/cycle3-best/export_bundle/export_summary.json` - Runtime export posture, prior-selection state, and quality flags for the accepted cycle
- `tmp/phase15_cycle3_consolidation/cycle3-best/export_bundle/export_inspection.json` - Runtime inspection of visibility buckets, prior closure, and review placeholders
- `tmp/phase15_cycle3_consolidation/cycle3-best/export_bundle/outputs/decision_episode.json` - Current audited decision-episode export payload
- `tmp/phase15_cycle3_consolidation/cycle3-best/export_bundle/outputs/training_view.json` - Current umbrella training-facing artifact that Phase 16 must preserve while extending
- `tmp/phase15_cycle3_consolidation/cycle3-best/prior_review_bundle/candidate_review_summary.json` - Reviewed prior/anti-pattern summary for the accepted cycle

### Runtime And Schema Contracts
- `backend/scripts/run_phase13_cycle_refinement.py` - Existing bounded rerun entrypoint with baseline bundle wiring and reviewer support
- `backend/app/research_logic/phase10_multi_paper_validation.py` - Canonical package / replay / prior-review / export orchestration, comparison summary generation, and markdown report rendering
- `backend/app/research_logic/replay_io.py` - Current export bundle writer for `training_view.json`, `best_cycle_selection.json`, summaries, inspections, and bundle manifests
- `backend/app/research_logic/decision_episode_export.py` - Review-backed export assembly, accepted-but-unselected prior records, and section-review contract
- `backend/app/research_logic/models.py` - Canonical schema definitions for training review, why-now, route comparison, priors, and decision episodes
- `backend/app/research_logic/iteration_prioritization.py` - Current recommendation ranking surface and evidence-ref propagation

### Specs And Guardrails
- `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md` - Layer boundaries that Phase 16 must preserve while publishing final handoff outputs
- `docs/superpowers/specs/2026-04-01-logickg-decision-prior-and-episode-schema.md` - Canonical `DecisionPriorCard`, `AntiPatternCard`, and `DecisionEpisode` contract
- `docs/superpowers/specs/2026-04-01-logickg-why-now-and-route-comparison-schema.md` - Canonical `WhyNowCase` and `RouteComparisonCase` contract
- `.planning/phases/06-decision-episode-audit-export/06-RESEARCH.md` - Guardrail that audited export truth must stay review-backed rather than replay-only

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `backend/scripts/run_phase13_cycle_refinement.py`: already provides the bounded rerun entrypoint, baseline bundle wiring, reviewer support, and the fallback-merge lever used in later cycles
- `backend/app/research_logic/phase10_multi_paper_validation.py`: already assembles route-state package, replay bundle, prior-review bundle, export bundle, comparison summary, and markdown report outputs in one auditable path
- `backend/app/research_logic/replay_io.py`: already writes `decision_episode.json`, `training_view.json`, `export_summary.json`, `export_inspection.json`, `best_cycle_selection.json`, and bundle manifests
- `backend/app/research_logic/decision_episode_export.py`: already preserves review-backed prior closure, accepted-but-unselected records, and section-level training review metadata
- `backend/app/research_logic/iteration_prioritization.py`: already carries recommendation ids and evidence refs into machine-readable prioritization outputs
- `backend/tests/test_replay_io.py` and `backend/tests/test_phase10_multi_paper_validation.py`: already guard the current export bundle surface and are the natural place to lock Phase 16 additive output contracts

### Established Patterns
- Machine-readable JSON under `tmp/` is the source of truth; markdown under `docs/replay/reports/` renders or summarizes those artifacts
- The bounded rerun surface has remained fixed across Phases 12 through 15 so quality movement stays auditable
- Runtime export payloads can remain `not_started` for review fields while reviewed closeout truth is carried by `best_cycle_selection.json` and rendered markdown review artifacts
- The current export contract publishes one umbrella `training_view.json`, so additive outputs are safer than replacing that file

### Integration Points
- Phase 16 will likely connect a new rerun root to the existing Phase 13 runner and Phase 10 orchestration, then add final stability/handoff artifacts on top
- Multi-view publication will likely extend `replay_io.py` and the export bundle writer plus associated backend tests without removing existing files
- Final stability proof will likely pull together `comparison_summary.json`, `best_cycle_selection.json`, final verification markdown, and prioritization evidence into one reviewed closeout chain
- Final dataset packaging should reference the validated later-cycle artifact family rather than recompute from unrelated inputs

</code_context>

<specifics>
## Specific Ideas

- The user explicitly confirmed that Phase 16 does not need new papers. Stability should be proven on the same bounded slice rather than by widening the packet.
- Current code and tests only emit and assert a single `outputs/training_view.json`, so task-specific training views are new additive work for this phase.
- `best_cycle_selection.json` already carries the reviewed verdict, reviewer ids, rationale, residual defects, and recommendation evidence refs, while `export_summary.json` and `training_view.json` still keep pre-review default review fields. Phase 16 should treat that split as real system behavior, not as something to hand-wave away.
- The current accepted cycle still carries explicit residual defects (`weak_prior_support`, `selected_prior_ids_empty`, `yellow_route_state_present`, `dominant_method_inventory_noisy`), so Phase 16 is about reproducible acceptance plus explicit residual-risk handoff rather than zero-defect perfection.

</specifics>

<deferred>
## Deferred Ideas

- Adding new papers or changing the topic slice to test broader generalization
- Replacing the current rerun surface with a new execution framework
- UI / productization work for dataset review or milestone handoff flows
- Open-ended scaling beyond the bounded dataset bundle promised for `v1.2`

</deferred>

---

*Phase: 16-stability-verification-and-handoff*
*Context gathered: 2026-04-04*
