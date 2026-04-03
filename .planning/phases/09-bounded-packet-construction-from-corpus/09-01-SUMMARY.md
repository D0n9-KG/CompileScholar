---
phase: 09-bounded-packet-construction-from-corpus
plan: 01
subsystem: api
tags: [python, pydantic, route-packet, audit, cli, replay-io]
requires:
  - phase: 01-replay-pilot-packetization
    provides: canonical RoutePacket contract and packet-audit discipline
  - phase: 08-sampled-single-paper-l2-regression
    provides: Phase 8 trace reuse expectations and filesystem-first audit posture
provides:
  - typed bounded packet assembly manifest and audit contract layered on RoutePacket
  - replay_io bundle helpers for audit_summary.json and audit_inspection.json outputs
  - CLI entrypoint for packet and assembly-manifest audits with machine-readable summaries
affects: [09-02-PLAN.md, phase-10-multi-paper-l3-and-l4-validation, PACK-01]
tech-stack:
  added: []
  patterns: [companion-manifest packet auditing, replay_io audit bundle writer, strict-alignment-plus-explicit-flags CLI]
key-files:
  created: [backend/app/research_logic/bounded_packet_audit.py, backend/scripts/run_bounded_packet_audit.py, backend/tests/test_bounded_packet_audit.py]
  modified: [backend/app/research_logic/replay_io.py, backend/app/research_logic/__init__.py]
key-decisions:
  - "Keep RoutePacket canonical and layer support / alternative / held_out mapping in a companion manifest."
  - "Treat packet/manifest misalignment as CLI-failing validation errors while preserving packet quality blockers as explicit summary flags."
  - "Reuse replay_io bundle conventions so Phase 9 audit runtime artifacts match earlier summary/inspection patterns."
patterns-established:
  - "Companion manifest over canonical packet: downstream role mapping lives beside RoutePacket, not inside it."
  - "Gap-first bounded packet audit: missing traces, role gaps, held-out leakage, alternative distinctness, and exclusion coverage stay explicit."
requirements-completed: [PACK-01]
duration: 13min
completed: 2026-04-03
---

# Phase 9 Plan 01: Bounded Packet Audit Summary

**Typed bounded-packet assembly auditing with a reusable JSON bundle writer and CLI over the canonical RoutePacket contract**

## Performance

- **Duration:** 13 min
- **Started:** 2026-04-03T07:34:14Z
- **Completed:** 2026-04-03T07:47:14Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added a typed `BoundedPacketAssemblyManifest` plus bounded-packet audit model that validates packet/member alignment while keeping non-structural blockers as explicit flags.
- Added `replay_io` helpers that write reproducible Phase 9 runtime bundles with `audit_summary.json`, `audit_inspection.json`, copied inputs, and a bundle manifest.
- Added `backend/scripts/run_bounded_packet_audit.py` so one packet plus companion manifest can be audited from the command line without editing source code.
- Expanded regression coverage for overlap, missing packet members, exclusion-note validation, bundle filenames, CLI summaries, and non-zero exits on invalid alignment.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add a typed companion manifest and audit model for bounded packet assembly** - `cf9e7011` (`feat`)
2. **Task 2: Add a bounded packet audit bundle writer and CLI entrypoint** - `aa68f422` (`feat`)

**Plan metadata:** pending final docs commit created after summary/state updates

## Files Created/Modified

- `backend/app/research_logic/bounded_packet_audit.py` - companion manifest, audit result, strict validator, and markdown report renderer for bounded packet audits
- `backend/app/research_logic/replay_io.py` - bounded packet audit summary, inspection, and bundle writing helpers
- `backend/app/research_logic/__init__.py` - package exports for the new audit contract, helpers, and bundle writer
- `backend/scripts/run_bounded_packet_audit.py` - operator CLI for packet plus assembly-manifest audits
- `backend/tests/test_bounded_packet_audit.py` - focused regression coverage for contract validation, bundle outputs, and CLI behavior

## Decisions Made

- Kept `RoutePacket` as the only packet schema and modeled downstream `support / alternative / held_out` assignment in a separate manifest to avoid Phase 9 schema churn.
- Separated structural alignment failures from packet-quality flags so the CLI only returns non-zero for true packet/manifest mismatches while still surfacing red/yellow packet blockers in JSON.
- Wrote runtime audit bundles through `replay_io` to stay consistent with the repo’s existing `summary + inspection + bundle_manifest` artifact pattern.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - planned implementation and verification passed without code-level blockers.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None.

## Next Phase Readiness

- Phase `09-02` can now commit a real bounded packet plus assembly manifest against a stable typed audit boundary.
- Phase `10` has a CLI and runtime bundle format ready to inspect packet assembly blockers before replay/package claims are made.

## Self-Check: PASSED

- Verified summary and all five plan-scope files exist on disk.
- Verified task commits `cf9e7011` and `aa68f422` are present in git history.
