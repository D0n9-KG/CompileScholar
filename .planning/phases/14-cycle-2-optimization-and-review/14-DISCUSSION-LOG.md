# Phase 14: Cycle 2 Optimization And Review - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md; this log preserves the assumption analysis.

**Date:** 2026-04-04
**Phase:** 14-cycle-2-optimization-and-review
**Mode:** assumptions
**Areas analyzed:** Execution Surface, Export Truth Source, Training-Facing Export And Review Fields, Canonical Labels And Optimization Scope

## Assumptions Presented

### Execution Surface

| Assumption | Confidence | Evidence |
|------------|------------|----------|
| Keep using the bounded Phase 9 packet and the existing Phase 10/13 `package -> replay -> prior-review -> export` runner, with new cycle-scoped reruns instead of scope widening or a new runtime path. | Confident | `.planning/ROADMAP.md`; `.planning/phases/12-baseline-data-cycle-1-direct-fix/12-CONTEXT.md`; `.planning/phases/13-baseline-data-cycle-2-refine/13-CONTEXT.md`; `.planning/STATE.md`; `backend/scripts/run_phase13_cycle_refinement.py`; `backend/app/research_logic/phase10_multi_paper_validation.py`; `docs/replay/reports/phase13-cycle2-refine.md` |

### Export Truth Source

| Assumption | Confidence | Evidence |
|------------|------------|----------|
| Keep treating export as a post-replay assembly driven by reviewed accepted ids, and fix the prior-selection gap by making reviewed priors applicable to the exported primary route or by preserving structured exclusion rationale rather than copying replay truth forward. | Confident | `backend/app/research_logic/phase10_multi_paper_validation.py`; `backend/app/research_logic/decision_episode_export.py`; `backend/app/research_logic/decision_episode_builder.py`; `backend/tests/test_decision_episode_export.py`; `tmp/phase13_cycle2_refine/cycle2-final/replay_bundle/outputs/decision_episode.json`; `tmp/phase13_cycle2_refine/cycle2-final/export_bundle/export_summary.json`; `tmp/phase13_cycle2_refine/cycle2-final/prior_review_bundle/prior_candidates.json`; `.planning/phases/06-decision-episode-audit-export/06-RESEARCH.md` |

### Training-Facing Export And Review Fields

| Assumption | Confidence | Evidence |
|------------|------------|----------|
| Extend the existing `export_bundle` with a self-contained training-facing view and structured machine-readable review / training-acceptance fields instead of leaving the judgment only in markdown reports. | Likely | `.planning/PROJECT.md`; `.planning/REQUIREMENTS.md`; `.planning/ROADMAP.md`; `tmp/phase13_cycle2_refine/cycle2-final/export_bundle/bundle_manifest.json`; `backend/app/research_logic/replay_io.py`; `docs/replay/reports/phase13-cycle2-refine.md` |

### Canonical Labels And Optimization Scope

| Assumption | Confidence | Evidence |
|------------|------------|----------|
| Add canonical-label-plus-raw-phrase preservation at the route/export schema layer and keep every major training-data section under explicit review pressure, rather than optimizing only blocker queues and stage flags. | Likely | `backend/app/research_logic/models.py`; `backend/app/research_logic/route_state_synthesizer.py`; `tmp/phase13_cycle2_refine/cycle2-final/replay_bundle/outputs/primary_route_state.json`; `backend/app/research_logic/iteration_prioritization.py`; `.planning/ROADMAP.md` |

## Corrections Made

No direct user corrections were captured in this assumptions-mode run. The final context reflects the codebase-grounded assumptions above without contradictory user input.
