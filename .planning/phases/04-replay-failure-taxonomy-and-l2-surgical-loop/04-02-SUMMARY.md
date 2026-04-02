---
phase: 04-replay-failure-taxonomy-and-l2-surgical-loop
plan: 02
subsystem: api
tags: [l2, derived-views, route-state, replay, pytest]
requires:
  - phase: 04-replay-failure-taxonomy-and-l2-surgical-loop
    provides: replay failure codes that identify comparator, completeness, and relation-stitching gaps
provides:
  - completeness signals in route compiler and route-state seed artifacts
  - comparator and alternative-route signal entries with route-level evidence ids
  - targeted fixture coverage for comparator, completeness, and relation-stitching behavior
affects: [04-03, replay taxonomy, route-state synthesis]
tech-stack:
  added: []
  patterns: [surgical L2 repair without schema change, evidence-id propagation through derived views]
key-files:
  created: []
  modified: [backend/app/paper_logic_trace/derived_views.py, backend/app/research_logic/route_state_synthesizer.py, backend/tests/test_paper_logic_trace_derived_views.py, backend/tests/test_route_state_synthesizer.py, backend/tests/test_historical_replay_compiler.py]
key-decisions:
  - "Added completeness signals as derived-view metadata instead of expanding ResearchMove or the L2 schema."
  - "Preserved comparator and future-work entries as replay-facing seed payloads with evidence_ids."
  - "Narrowed edits to shared L2 surfaces to avoid disturbing the larger in-flight L1 route synthesizer work."
patterns-established:
  - "Derived views can surface nonfatal slot gaps as explicit repair signals."
  - "Route-state synthesis can consume comparison and alternative-route seed entries without schema drift."
requirements-completed: [L2-01, L2-02]
duration: 8m
completed: 2026-04-02
---

# Phase 4 Plan 2: L2 Repair Summary

**Comparator, completeness, and alternative-route evidence now survive through derived views and route-state synthesis on targeted fixtures without changing the `ResearchMove` schema.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-02T13:45:00+08:00
- **Completed:** 2026-04-02T13:53:00+08:00
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added `build_completeness_signals(...)` and threaded completeness entries into `route_compiler_contract` and `route_state_seed`.
- Preserved comparison and alternative-route seed entries with route-level `evidence_ids`.
- Verified the surgical L2 repair path with targeted derived-view, route-state, replay-compiler, and replay-IO tests.

## Task Commits

No atomic task commits were created. The route synthesizer file already contained unrelated in-flight changes, so the plan was completed locally without trying to isolate or rewrite existing user work.

## Files Created/Modified

- `backend/app/paper_logic_trace/derived_views.py` - adds completeness signaling plus replay-facing comparison and alternative-route seed entries
- `backend/app/research_logic/route_state_synthesizer.py` - consumes repaired seed entries into route-level evidence and alternative-route aggregation
- `backend/tests/test_paper_logic_trace_derived_views.py` - covers comparator retention, thin-slot completeness signaling, and future-work signal preservation
- `backend/tests/test_route_state_synthesizer.py` - verifies route-level preservation of repaired evidence
- `backend/tests/test_historical_replay_compiler.py` - verifies repaired fixtures can clear or avoid targeted L2 degradation signals

## Decisions Made

- Kept the repair loop inside derived views and route-state synthesis instead of altering shared trace models.
- Used explicit `evidence_ids` on seed entries so replay-facing code can preserve provenance without re-deriving it.
- Worked around a shared in-flight `route_state_synthesizer.py` diff by limiting changes to the phase-4 repair surfaces.

## Deviations from Plan

The targeted fixture suite improved as planned, but the real same-slice jamming delta was deferred to Plan 03 and ultimately stayed flat. That was treated as a measured outcome rather than a reason to loosen the new tests.

## Issues Encountered

- `route_state_synthesizer.py` was already carrying a larger L1 integration patch, so edits had to stay carefully scoped.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The package and replay rerun can now exercise repaired L2 signals through the real phase baseline.
- The next step is a same-slice rerun and delta report to confirm whether the live jamming artifacts actually improve.

## Self-Check: PASSED

---
*Phase: 04-replay-failure-taxonomy-and-l2-surgical-loop*
*Completed: 2026-04-02*
