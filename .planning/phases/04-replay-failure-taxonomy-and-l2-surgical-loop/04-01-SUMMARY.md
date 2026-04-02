---
phase: 04-replay-failure-taxonomy-and-l2-surgical-loop
plan: 01
subsystem: api
tags: [replay, taxonomy, diagnostics, backend, json]
requires:
  - phase: 03-grounded-route-replay-compilation
    provides: green packaged replay baseline and route-state package runtime artifacts
provides:
  - replay-native failure records with layer and stage ownership
  - aggregate taxonomy counts in replay summaries
  - baseline Phase 4 diagnosis report for the packaged jamming replay
affects: [04-02, 04-03, replay reports]
tech-stack:
  added: []
  patterns: [replay-native failure taxonomy, summary-vs-inspection split, nonblocking L2 repair queue]
key-files:
  created: [docs/replay/reports/phase4-failure-taxonomy-baseline.md]
  modified: [backend/app/research_logic/historical_replay_compiler.py, backend/app/research_logic/replay_io.py, backend/tests/test_historical_replay_compiler.py, backend/tests/test_replay_io.py]
key-decisions:
  - "Kept failure classification inside the existing replay compilation and output path instead of building a second diagnostics pipeline."
  - "Stored full failure records only in replay_inspection.json while replay_summary.json carries aggregate counts."
  - "Left nonblocking L2 weaknesses in the same failure_records queue so the repair loop stays auditable."
patterns-established:
  - "Replay outputs now separate surfaced stage from owning layer."
  - "Green replay bundles can still emit nonblocking L2 repair records."
requirements-completed: [L2-01, EVAL-02]
duration: 5m
completed: 2026-04-02
---

# Phase 4 Plan 1: Failure Taxonomy Summary

**Replay-native failure records now expose packet, L1, L2, and downstream ownership on the same packaged jamming replay artifact boundary.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-02T13:44:00+08:00
- **Completed:** 2026-04-02T13:49:00+08:00
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added normalized `failure_records` to replay compilation and inspection outputs.
- Added aggregate counts by `layer`, `stage`, and `blocking` split to `replay_summary.json`.
- Wrote the baseline diagnosis report for the Phase 3 packaged jamming replay.

## Task Commits

No atomic task commits were created. The repository already contained unrelated in-flight changes, so this plan was executed in place to avoid clobbering user work.

## Files Created/Modified

- `backend/app/research_logic/historical_replay_compiler.py` - maps replay symptoms into fixed failure codes with owner and repair target metadata
- `backend/app/research_logic/replay_io.py` - emits aggregate taxonomy counts in summary outputs and full records in inspection outputs
- `backend/tests/test_historical_replay_compiler.py` - covers underconstrained replay cases and L2/downstream ownership
- `backend/tests/test_replay_io.py` - verifies failure-record fields and aggregate summary counts
- `docs/replay/reports/phase4-failure-taxonomy-baseline.md` - records the baseline queue from the packaged jamming replay

## Decisions Made

- Kept the taxonomy inside the existing replay boundary instead of adding a sidecar report generator.
- Split full-record detail and aggregate counts across inspection and summary artifacts.
- Preserved nonblocking L2 issues as first-class failure records.

## Deviations from Plan

None - the plan goals were completed on the existing replay artifact boundary.

## Issues Encountered

- The worktree was already dirty, so plan execution had to avoid any cleanup or revert-style operations.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Comparator density, expected-slot completeness, and relation stitching now have explicit failure codes and evidence refs.
- Phase 4 plan 02 can target these L2 surfaces without guessing from coarse replay flags.

## Self-Check: PASSED

---
*Phase: 04-replay-failure-taxonomy-and-l2-surgical-loop*
*Completed: 2026-04-02*
