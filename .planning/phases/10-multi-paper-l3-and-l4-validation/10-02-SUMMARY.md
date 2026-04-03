---
phase: 10-multi-paper-l3-and-l4-validation
plan: 02
subsystem: api
tags: [fastapi, replay, prior-review, audited-export, validation]
requires:
  - phase: 10-multi-paper-l3-and-l4-validation
    provides: Phase 10 package/replay runtime bridge and CLI from 10-01
provides:
  - one Phase 10 workflow that emits prior-review and audited export bundles from the same bounded packet run
  - structured comparison_summary.json output against the jamming replay/export baseline
  - markdown report rendering driven directly from comparison_summary.json
affects: [10-03, phase-11-iteration-prioritization, replay, prior-review, audited-export]
tech-stack:
  added: []
  patterns: [phase10 full artifact-chain orchestration, comparison-summary-as-report-source-of-truth]
key-files:
  created:
    - backend/tests/test_phase10_multi_paper_validation_cli.py
  modified:
    - backend/app/research_logic/phase10_multi_paper_validation.py
    - backend/scripts/run_phase10_multi_paper_validation.py
    - backend/tests/test_phase10_multi_paper_validation.py
key-decisions:
  - "Keep the Phase 10 CLI machine-readable first: always write comparison_summary.json and render markdown from that file instead of recomputing report prose."
  - "Compare the computational-mechanics packet against the jamming baseline through replay/export bundle summaries and inspections, with blocker queues grouped by stage."
patterns-established:
  - "Phase 10 now carries one bounded packet run through route_state_package, replay_bundle, prior_review_bundle, export_bundle, and comparison_summary.json in one command."
  - "Baseline comparison defaults stay pinned to tmp/phase3_route_state_package/replay_with_package and tmp/phase6_decision_episode_audit_export while remaining overrideable via CLI flags."
requirements-completed: [PACK-02, AGGR-01]
duration: 19 min
completed: 2026-04-03
---

# Phase 10 Plan 02: Multi-Paper Runtime Comparison Summary

**Phase 10 now emits prior-review, audited export, and structured baseline-comparison artifacts for the bounded computational-mechanics packet in one backend workflow**

## Performance

- **Duration:** 19 min
- **Started:** 2026-04-03T18:26:32+08:00
- **Completed:** 2026-04-03T18:45:20+08:00
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Extended the Phase 10 backend workflow so one bounded packet run now writes `prior_review_bundle/` and `export_bundle/` after replay.
- Preserved the review-truth boundary by rebuilding export output from accepted review ids and keeping export visibility buckets explicit.
- Added `comparison_summary.json` plus markdown report rendering that compares the computational-mechanics run against the jamming replay/export baseline through machine-readable bundle surfaces.

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend the Phase 10 workflow to emit prior-review and audited export bundles from the same packet run** - `3fe1cf79` (feat)
2. **Task 2: Add structured comparison and blocker-report outputs against the jamming baseline** - `ad124f69` (feat)

**Plan metadata:** Recorded in the final closeout commit after summary and state updates.

## Files Created/Modified

- `backend/app/research_logic/phase10_multi_paper_validation.py` - Extended the Phase 10 workflow through prior review, audited export, comparison summary generation, and markdown report rendering.
- `backend/scripts/run_phase10_multi_paper_validation.py` - Added baseline bundle and report CLI options while keeping the workflow output JSON-focused.
- `backend/tests/test_phase10_multi_paper_validation.py` - Expanded the Phase 10 integration tests to assert the new bundle chain and comparison summary output.
- `backend/tests/test_phase10_multi_paper_validation_cli.py` - Added CLI regression coverage for baseline comparison flags and `--report-md`.

## Decisions Made

- Comparison and reporting stay bundle-first: the workflow writes `comparison_summary.json` every run and treats the markdown report as a pure render of that JSON contract.
- The jamming packaged replay and audited export bundles remain the default comparison baseline so Phase 10 can distinguish packet-specific weaknesses from generic compiler regressions without extra operator setup.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 10 is ready for `10-03` to run the real computational-mechanics packet and commit the human-readable validation report.
- The workflow now leaves stage-specific blockers machine-readable. The final smoke run surfaced `support_cluster_too_small`, `alternative_scope_not_distinct`, `reviewer_missing`, and empty accepted prior ids as explicit package/replay/prior/export outputs instead of flattening them into prose.

## Self-Check: PASSED

- Verified the summary and all claimed implementation files exist on disk.
- Verified task commits `3fe1cf79` and `ad124f69` exist in git history.

---
*Phase: 10-multi-paper-l3-and-l4-validation*
*Completed: 2026-04-03*
