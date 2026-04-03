---
phase: 10-multi-paper-l3-and-l4-validation
plan: 01
subsystem: api
tags: [fastapi, replay, route-state-package, scientific-reasoning, validation]
requires:
  - phase: 09-bounded-packet-construction-from-corpus
    provides: committed phase9 route packet, assembly manifest, and role-scoped trace refs
provides:
  - real Phase 10 runtime bridge from committed Phase 9 inputs to route-state package inputs
  - dedicated CLI that compiles route_state_package and replay bundles for the bounded packet
  - replay inspection outputs that preserve package-validation blocker surfaces
affects: [10-02, 10-03, phase-11-iteration-prioritization, replay, route-state-package]
tech-stack:
  added: []
  patterns: [runtime-only packet adaptation for trace-native synthesis ids, shared replay/package bundle reuse]
key-files:
  created:
    - backend/app/research_logic/phase10_multi_paper_validation.py
    - backend/scripts/run_phase10_multi_paper_validation.py
    - backend/tests/test_phase10_multi_paper_validation.py
  modified:
    - backend/app/research_logic/__init__.py
    - backend/app/research_logic/route_state_synthesizer.py
key-decisions:
  - "Keep the committed Phase 9 packet and assembly manifest immutable, and generate disposable runtime packets under tmp/ for package and replay compilation."
  - "Adapt runtime packet paper_ids to trace-native ids only at synthesis boundaries so existing RouteStatePackage and replay contracts can run against real traces."
patterns-established:
  - "Phase 10 runtime bridge: load committed packet docs, emit generated manifests and snapshot under tmp/, and compile bundles from those disposable artifacts."
  - "Real-data verification: run the dedicated Phase 10 CLI against the committed packet fixtures so blocker surfaces stay visible in replay inspection outputs."
requirements-completed: [PACK-02]
duration: 16m
completed: 2026-04-03
---

# Phase 10 Plan 01: Multi-Paper Runtime Bridge Summary

**Committed Phase 9 packet bridging into a real L1 snapshot, route-state package bundle, and replay bundle for the bounded computational-mechanics packet**

## Performance

- **Duration:** 16 min
- **Started:** 2026-04-03T18:00:55+08:00
- **Completed:** 2026-04-03T18:16:56+08:00
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added a typed Phase 10 helper that reads the committed Phase 9 packet boundary, resolves repo-relative trace refs, writes a real L1 snapshot, and emits a generated route-state package manifest under `tmp/`.
- Added a dedicated Phase 10 CLI that compiles route-state package and replay bundles from the bridged inputs without mutating the committed packet docs.
- Extended regression coverage so the CLI help contract, bundle outputs, and replay inspection blocker surfaces are exercised against real committed packet inputs.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add a Phase 10 helper that converts the committed Phase 9 packet boundary into package-ready runtime inputs** - `68feb5ee` (feat)
2. **Task 2: Add the Phase 10 CLI path that compiles the route-state package and replay bundle from the bridged inputs** - `ead6f92c` (feat)

**Plan metadata:** Recorded in the final closeout commit after summary and state updates.

## Files Created/Modified

- `backend/app/research_logic/phase10_multi_paper_validation.py` - Phase 10 bridge helpers, runtime packet generation, and package/replay orchestration.
- `backend/scripts/run_phase10_multi_paper_validation.py` - Operator CLI for Phase 10 package and replay compilation.
- `backend/tests/test_phase10_multi_paper_validation.py` - Bridge and CLI regression coverage against committed Phase 9 inputs.
- `backend/app/research_logic/__init__.py` - Exported Phase 10 helper entrypoints.
- `backend/app/research_logic/route_state_synthesizer.py` - Normalized aggregated infrastructure types during real-data replay synthesis.

## Decisions Made

- Preserved the committed Phase 9 packet and assembly manifest as immutable source inputs and generated disposable runtime artifacts under `tmp/phase10_multi_paper_validation/`.
- Reused the existing route-state package and replay writers instead of inventing a Phase 10-specific bundle format, so package validation continues to flow through replay inspection.
- Kept canonical Phase 9 role membership on bridge artifacts while translating runtime packet paper ids to trace-native ids only where the synthesizer requires them.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Adapted runtime packet paper ids to real trace ids for synthesis**

- **Found during:** Task 2 (Add the Phase 10 CLI path that compiles the route-state package and replay bundle from the bridged inputs)
- **Issue:** The committed Phase 9 packet uses frozen corpus ids like `1000`, while the real Phase 8 traces carry hash-based `paper_metadata.paper_id` values. `RouteStateSynthesizer` rejects that mismatch during package and replay compilation.
- **Fix:** Generated runtime-only packet payloads that preserve the committed Phase 9 boundary in bridge artifacts and notes, but translate included `paper_id` values to trace-native ids before synthesis.
- **Files modified:** `backend/app/research_logic/phase10_multi_paper_validation.py`, `backend/tests/test_phase10_multi_paper_validation.py`
- **Verification:** `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase10_multi_paper_validation.py tests\test_route_state_package.py tests\test_replay_io.py -q`; manual CLI run completed with real packet inputs.
- **Committed in:** `ead6f92c` (part of Task 2 commit)

**2. [Rule 1 - Bug] Normalized aggregated infrastructure types before replay synthesis**

- **Found during:** Task 2 (Add the Phase 10 CLI path that compiles the route-state package and replay bundle from the bridged inputs)
- **Issue:** Real traces surfaced a `toolchain_candidates` resource type of `algorithm`, and `_aggregate_infrastructure_states(...)` passed raw resource types straight into `InfrastructureState`, violating the model literal contract and crashing the CLI.
- **Fix:** Reused `_resource_infra_type(...)` in the aggregation path so unsupported resource tokens collapse into the allowed infrastructure types instead of raising a validation error.
- **Files modified:** `backend/app/research_logic/route_state_synthesizer.py`
- **Verification:** `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase10_multi_paper_validation.py tests\test_route_state_package.py tests\test_replay_io.py -q`; manual CLI run completed with real packet inputs.
- **Committed in:** `ead6f92c` (part of Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes were required for real-data package and replay compilation. They preserved plan scope and kept blocker reporting explicit.

## Issues Encountered

- The committed Phase 9 packet boundary uses human-assigned corpus ids while the real Phase 8 traces expose hash-based `paper_metadata.paper_id` values, so runtime synthesis needed a compatibility seam.
- Real replay synthesis exposed a pre-existing infrastructure-type normalization gap that was not covered by earlier package-level tests.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 10 now has a reproducible package and replay CLI for the committed bounded packet, with real `route_state_package/` and `replay_bundle/` outputs available from one command.
- The generated artifacts intentionally keep blockers visible. The latest manual CLI run still reports `support_cluster_too_small`, `alternative_scope_not_distinct`, and `reviewer_missing`, which gives the next Phase 10 plans concrete multi-paper gaps to work from.

## Self-Check: PASSED

- Verified summary and implementation files exist on disk.
- Verified task commits `68feb5ee` and `ead6f92c` exist in git history.

---
*Phase: 10-multi-paper-l3-and-l4-validation*
*Completed: 2026-04-03*
