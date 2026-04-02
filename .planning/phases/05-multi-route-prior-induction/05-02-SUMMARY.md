---
phase: 05-multi-route-prior-induction
plan: 02
subsystem: api
tags: [replay, prior-review, cli, reports, testing]
requires:
  - phase: 05-multi-route-prior-induction
    provides: prior candidate induction, anti-pattern extraction, package-to-induction bridge
provides:
  - accepted-card consumption path for replay and episode builders
  - JSON review-bundle writer for prior and anti-pattern candidates
  - CLI flag to emit the Phase 5 candidate-review bundle from a real route-state package
  - committed Phase 5 jamming pilot report
affects: [phase-05, replay-cli, reports, decision-layer]
tech-stack:
  added: []
  patterns: [accepted-card gating, dual-output replay workflow, review-bundle reporting]
key-files:
  created:
    - docs/replay/reports/phase5-prior-review-pilot.md
  modified:
    - backend/app/research_logic/historical_replay_compiler.py
    - backend/app/research_logic/replay_io.py
    - backend/scripts/run_replay_pilot.py
    - backend/tests/test_historical_replay_compiler.py
    - backend/tests/test_replay_io.py
    - backend/tests/test_route_state_package.py
requirements-completed: [L4-01]
duration: 20min
completed: 2026-04-02
---

# Phase 05 Plan 02 Summary

**Replay can now consume accepted prior cards, emit a separate Phase 5 review bundle, and publish a real jamming prior-review pilot without claiming Phase 6 export readiness**

## Performance

- **Duration:** 20 min
- **Started:** 2026-04-02T07:25:00Z
- **Completed:** 2026-04-02T07:45:42Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Added accepted-card consumption support to `HistoricalReplayCompiler` so injected green prior cards feed replay and episode selection while non-green candidates still leave `weak_prior_support` visible.
- Added `write_prior_candidate_review_bundle(...)` and `build_prior_candidate_review_summary(...)` so Phase 5 review artifacts stay JSON / CLI based and live alongside replay outputs.
- Extended `run_replay_pilot.py` with an optional `--prior-review-output-dir` path and used it on the real jamming package to produce:
  - `tmp/phase5_multi_route_prior_induction/review_bundle/`
  - `tmp/phase5_multi_route_prior_induction/replay_bundle/`

## Task Commits

Atomic task commits were intentionally skipped in this run because the workspace already contained unrelated in-progress changes in overlapping planning and research-logic files.

Verification and runtime proof used:

1. `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_prior_builder.py tests\test_decision_episode_builder.py tests\test_historical_replay_compiler.py tests\test_replay_io.py tests\test_route_state_package.py -q`
2. `cd backend; .\.venv\Scripts\python.exe scripts\run_replay_pilot.py --packet ..\tmp\phase3_route_state_package\replay_with_package\inputs\route_packet.json --trace-file ..\tmp\phase1_packet_traces\777\paper_logic_trace.json --trace-file ..\tmp\phase1_packet_traces\1742\paper_logic_trace.json --trace-file ..\tmp\phase1_packet_traces\1746\paper_logic_trace.json --trace-file ..\tmp\phase1_packet_traces\1591\paper_logic_trace.json --trace-file ..\tmp\phase1_packet_traces\814\paper_logic_trace.json --l1-snapshot ..\tmp\phase3_route_state_package\replay_with_package\inputs\historical_environment_snapshot.json --route-state-package ..\tmp\phase3_route_state_package\bundle --reviewer reviewer-1 --output-dir ..\tmp\phase5_multi_route_prior_induction\replay_bundle --prior-review-output-dir ..\tmp\phase5_multi_route_prior_induction\review_bundle`

## Files Created/Modified

- `backend/app/research_logic/historical_replay_compiler.py` - accept injected prior registries while preserving the old replay fallback path
- `backend/app/research_logic/replay_io.py` - write candidate review bundles and summary payloads
- `backend/scripts/run_replay_pilot.py` - add the optional Phase 5 review-bundle output path
- `backend/tests/test_historical_replay_compiler.py` - cover green vs non-green injected prior behavior
- `backend/tests/test_replay_io.py` - cover the new review-bundle writer
- `backend/tests/test_route_state_package.py` - cover CLI emission of replay plus review bundles from the same package
- `docs/replay/reports/phase5-prior-review-pilot.md` - record the real jamming pilot findings and review decisions

## Decisions Made

- Separated package-level candidate review from replay-level live prior consumption so sparse support clusters stay conservative even when the live replay path can still build a green prior.
- Kept accepted-card consumption strict: only green injected priors are used for replay / episode support.
- Made the review bundle optional so the existing replay CLI remains backward compatible.

## Deviations from Plan

### Auto-fixed Issues

**1. The review path was kept optional instead of replacing the main replay bundle**

- **Found during:** Task 2
- **Issue:** Overloading the main replay output would blur the difference between replay artifacts and candidate-review artifacts.
- **Fix:** Added a separate optional `--prior-review-output-dir` and distinct bundle writer.
- **Files modified:** `backend/app/research_logic/replay_io.py`, `backend/scripts/run_replay_pilot.py`
- **Verification:** Targeted pytest and the real jamming pilot run both produced replay and review bundles side by side

---

**Total deviations:** 1 auto-fixed
**Impact on plan:** Improves auditability and keeps backward compatibility. No scope creep.

## Issues Encountered

- The real package-only support cluster stayed yellow because it only contains two support route states. That is expected and is now explicitly surfaced as a review outcome instead of being hidden in prose.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 5 now has a real candidate-review workflow with explicit accepted ids.
- Phase 6 can build audited export policy on top of these accepted-card and replay-bundle boundaries rather than inventing a new review surface first.

---
*Phase: 05-multi-route-prior-induction*
*Completed: 2026-04-02*
