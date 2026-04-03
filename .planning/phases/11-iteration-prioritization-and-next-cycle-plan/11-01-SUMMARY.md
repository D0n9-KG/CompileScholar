---
phase: 11-iteration-prioritization-and-next-cycle-plan
plan: 01
subsystem: api
tags: [python, pydantic, pytest, prioritization, replay_io]
requires:
  - phase: 10-multi-paper-l3-and-l4-validation
    provides: Phase 10 comparison blockers and packet-first recommendation evidence
provides:
  - Phase 11 typed evidence loaders for Phase 8 and Phase 10 prioritization inputs
  - Packet-first ranking contract for the next optimization cycle
  - Auditable prioritization summary and inspection bundle writers
affects: [phase-11-reporting, next-cycle-selection, backend-reporting]
tech-stack:
  added: []
  patterns: [json-first evidence loading, explicit markdown fallback provenance, manifest-backed prioritization bundles]
key-files:
  created:
    - backend/app/research_logic/iteration_prioritization.py
    - backend/tests/test_iteration_prioritization.py
  modified:
    - backend/app/research_logic/__init__.py
    - backend/app/research_logic/replay_io.py
    - backend/tests/test_replay_io.py
key-decisions:
  - "Allow Phase 11 to consume explicit Phase 10 comparison JSON when present, but fall back to the committed verification/report pair with recorded provenance."
  - "Rank packet construction ahead of L4 and L2 when new package-validation blockers appear while replay L2 delta stays flat."
  - "Write Phase 11 bundle artifacts under outputs/ with a root manifest so later reporting can treat the summary JSON as the source of truth."
patterns-established:
  - "Phase 11 synthesis stays JSON-first and never reruns Phase 8 or Phase 10 workflows."
  - "Fallback evidence parsing must remain explicit in source refs instead of silently choosing default files."
requirements-completed: [LOOP-01]
duration: 6min
completed: 2026-04-03
---

# Phase 11 Plan 01: Iteration Prioritization Contract Summary

**Typed Phase 11 evidence loaders, packet-first recommendation ranking, and auditable prioritization bundle outputs over the committed Phase 8 and Phase 10 evidence chain**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-03T13:05:54Z
- **Completed:** 2026-04-03T13:11:37Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added a dedicated `iteration_prioritization` research-logic module with typed Phase 8 and Phase 10 evidence surfaces.
- Implemented explicit Phase 10 fallback parsing from `10-VERIFICATION.md` plus the committed Phase 10 report when comparison JSON is absent.
- Ranked `packet_construction`, `l4_aggregation`, and `l2_extraction` from structured signals and kept `relation_assembly` plus `slot_recovery` visible as supporting L2 evidence.
- Extended `replay_io` with Phase 11 summary/inspection payload builders and a manifest-backed prioritization bundle writer.
- Added focused pytest coverage for loading, fallback, missing-input preflight behavior, ranking order, and manifest file refs.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add typed Phase 11 evidence models and upstream loaders with explicit fallback handling** - `4ada2ffc` (`feat`)
2. **Task 2: Add prioritization logic and Phase 11 summary / inspection bundle writers** - `c350e34a` (`feat`)

## Files Created/Modified

- `backend/app/research_logic/iteration_prioritization.py` - Typed Phase 11 evidence models, Phase 8/10 loaders, fallback parsing, ranking, and summary/inspection builders.
- `backend/app/research_logic/replay_io.py` - Phase 11 summary/inspection payload helpers and bundle writer with manifest-relative output refs.
- `backend/app/research_logic/__init__.py` - Public exports for the new prioritization contracts and bundle helpers.
- `backend/tests/test_iteration_prioritization.py` - Loader, fallback, preflight, and current-evidence ranking coverage.
- `backend/tests/test_replay_io.py` - Phase 11 bundle manifest/output-path assertions.

## Decisions Made

- Used explicit source-ref tracking on both the JSON path and the markdown fallback pair so the prioritization output stays auditable on machines without Phase 10 runtime artifacts.
- Kept Phase 8 owner buckets and Phase 10 blocker stages separate in the summary instead of blending them into one opaque score.
- Treated Phase 10’s packet-first recommendation as a structured signal, but still recomputed the ordered queue from blocker deltas and owner-bucket evidence.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Large raw file reads timed out during context gathering, so execution switched to smaller targeted reads before implementation. No code changes were required.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 11 now has a reusable backend seam for CLI/report work and later real-run artifact generation.
- The current evidence chain remains packet-first, with `l4_aggregation` second and `l2_extraction` third, so downstream reporting can render a non-open-ended next-cycle recommendation.

## Self-Check

PASSED

---
*Phase: 11-iteration-prioritization-and-next-cycle-plan*
*Completed: 2026-04-03*
