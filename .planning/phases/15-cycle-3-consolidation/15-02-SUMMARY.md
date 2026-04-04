---
phase: 15-cycle-3-consolidation
plan: 02
subsystem: research-logic
tags: [phase15, route-comparison, why-now, export-closure]
requires:
  - phase: 15-cycle-3-consolidation
    provides: Plan 01 packet-validation detail and prior-recovery comparison metadata
provides:
  - Grounded route-comparison and why-now narrative fields in the training-facing export surface
  - Confidence caps that stay aligned to actual prior/comparison support
  - Structured route-backed anti-pattern exclusion records alongside prior exclusions
affects: [phase15-wave2-content-closure, route-comparison-grounding, export-selection-truth]
tech-stack:
  added: []
  patterns: [grounded comparison summaries, route-backed exclusion records, evidence-capped confidence]
key-files:
  created:
    - .planning/phases/15-cycle-3-consolidation/15-02-SUMMARY.md
  modified:
    - backend/app/research_logic/models.py
    - backend/app/research_logic/route_comparison_builder.py
    - backend/app/research_logic/why_now_builder.py
    - backend/app/research_logic/decision_episode_builder.py
    - backend/app/research_logic/decision_episode_export.py
    - backend/app/research_logic/replay_io.py
    - backend/app/research_logic/phase10_multi_paper_validation.py
    - backend/tests/test_route_comparison_builder.py
    - backend/tests/test_why_now_builder.py
    - backend/tests/test_decision_episode_builder.py
    - backend/tests/test_decision_episode_export.py
    - backend/tests/test_replay_io.py
    - backend/tests/test_phase10_multi_paper_validation.py
key-decisions:
  - "Only grounded preferred comparisons with decisive dimensions and evidence refs can name the winning route and summarize its advantage."
  - "Positive timing judgments must emit explicit `because_now` and `why_not_before` text instead of leaving the training view implicit."
  - "Accepted priors and anti-patterns stay route-backed: matched cards populate selected ids, unmatched accepted cards become explicit machine-readable exclusion records."
patterns-established:
  - "Export summary, export inspection, training view, and Phase 10 comparison summaries now preserve accepted and accepted-but-unselected anti-pattern closure together."
  - "Final decision confidence now stays capped at `0.85` unless selected priors or grounded route-comparison evidence justify more certainty."
requirements-completed: [STAB-01, TRAIN-04]
duration: 38min
completed: 2026-04-04
---

# Phase 15 Plan 02 Summary

**Phase 15 now explains why the chosen route wins, why the timing judgment holds, and how reviewed prior / anti-pattern knowledge closes honestly into the export surface.**

## Performance

- **Duration:** 38 min
- **Started:** 2026-04-04T18:41:00+08:00
- **Completed:** 2026-04-04T19:19:31+08:00
- **Tasks:** 2
- **Files modified:** 13 tracked source/test files

## Accomplishments

- Populated `recommended_route_state_id` and `route_advantage_summary` when route comparison has a genuinely grounded preferred side, making the winning route explicit in the machine-readable artifact.
- Extended why-now generation so positive timing judgments always carry concrete `because_now` and `why_not_before` text rather than leaving the reasoning implicit.
- Capped final decision confidence at `0.85` when neither selected priors nor grounded comparison evidence support stronger certainty.
- Preserved route-backed prior and anti-pattern closure together by carrying matched reviewed cards into selected ids and emitting explicit `accepted_but_unselected_*` exclusion records when accepted cards do not support the exported route.
- Threaded the new anti-pattern exclusion records through `export_summary.json`, `export_inspection.json`, `training_view.json`, and the Phase 10 comparison-summary surface.

## Verification

- `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_route_comparison_builder.py tests\test_why_now_builder.py tests\test_decision_episode_builder.py tests\test_historical_replay_compiler.py -q`
  - Result: `18 passed in 1.34s`
- `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_episode_builder.py tests\test_decision_episode_export.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py -q`
  - Result: `35 passed in 12.09s`

## Outcome

- Route comparison now names the winning route and explains its advantage with grounded evidence instead of leaving the preference implicit.
- Positive why-now paths now carry explicit causal and timing language that survives into training-facing outputs.
- Export truth for priors and anti-patterns remains route-backed while preserving explicit machine-readable mismatch records instead of silently dropping accepted reviewed cards.
- Phase 15 is ready for Wave 3 reruns and direct comparison against the reviewed Phase 14 best cycle.

## Files Created/Modified

- `.planning/phases/15-cycle-3-consolidation/15-02-SUMMARY.md` - records the plan outcome, verification, and readiness for Wave 3
- `backend/app/research_logic/models.py` - extends route-comparison and why-now case models with explicit grounded-output fields
- `backend/app/research_logic/route_comparison_builder.py` - emits the winning route id and advantage summary when comparison evidence is decisive
- `backend/app/research_logic/why_now_builder.py` - populates explicit `because_now` and `why_not_before` strings for positive timing paths
- `backend/app/research_logic/decision_episode_builder.py` - caps final decision confidence when support remains weak
- `backend/app/research_logic/decision_episode_export.py` - adds structured accepted-but-unselected anti-pattern exclusions and keeps route-backed selection explicit
- `backend/app/research_logic/replay_io.py` - serializes anti-pattern exclusions into export summary, inspection, and training-view payloads
- `backend/app/research_logic/phase10_multi_paper_validation.py` - preserves anti-pattern exclusion counts in comparison summaries and export blockers
- `backend/tests/test_route_comparison_builder.py` - verifies grounded preferred comparisons expose the winning route id and advantage summary
- `backend/tests/test_why_now_builder.py` - verifies positive timing labels keep explicit causal/timing text
- `backend/tests/test_decision_episode_builder.py` - verifies weak-support decision confidence stays capped
- `backend/tests/test_decision_episode_export.py` - verifies accepted-but-unselected anti-pattern records are explicit and structured
- `backend/tests/test_replay_io.py` - verifies anti-pattern exclusion fields survive summary, inspection, and training-view serialization
- `backend/tests/test_phase10_multi_paper_validation.py` - verifies workflow outputs preserve both prior and anti-pattern exclusion records

## Task Commits

1. `bc7ec468` (`feat`) - ground route comparison and timing narrative
2. `359647c1` (`feat`) - preserve route-backed prior and antipattern closure

## Decisions Made

- Treat grounded route-comparison evidence as a prerequisite for naming the winning route in the export surface.
- Keep the final decision conservative when neither priors nor comparison evidence fully supports a stronger confidence score.
- Mirror prior exclusions with anti-pattern exclusions instead of relying on a free-text anti-pattern selection note alone.

## Deviations from Plan

- No direct code changes were needed in `backend/tests/test_historical_replay_compiler.py`; the existing compiler coverage still held once the route-comparison and export surfaces were extended.

## Issues Encountered

- The first verification attempt for Task 2 failed because the synthetic weak-support comparison fixture still contained decisive comparison evidence. Tightening the fixture to remove decisive support resolved the failure and preserved the intended confidence-cap coverage.

## User Setup Required

None.

## Next Phase Readiness

- Wave 2 is complete and leaves Phase 15 ready for `15-03`, where the bounded rerun loop can test whether the improved comparison, timing, and selection-closure surfaces produce a genuinely good cycle against the reviewed Phase 14 baseline.

## Self-Check

PASSED

---
*Phase: 15-cycle-3-consolidation*
*Completed: 2026-04-04*
