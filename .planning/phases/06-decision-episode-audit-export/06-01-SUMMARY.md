---
phase: 06-decision-episode-audit-export
plan: 01
subsystem: api
tags: [replay, decision-episode, audit-export, testing]
requires:
  - phase: 05-multi-route-prior-induction
    provides: reviewed accepted-card registry, replay bundle contract, prior review outputs
provides:
  - review-allowlisted decision episode export assembly
  - dedicated decision episode export manifest, summary, and inspection writers
  - explicit visible-input, audit-only, and label/eval-only visibility buckets
affects: [phase-06, export-cli, replay-io, reports]
tech-stack:
  added: []
  patterns: [review-allowlisted export assembly, dedicated audit bundle outputs, explicit visibility buckets]
key-files:
  created:
    - backend/app/research_logic/decision_episode_export.py
    - backend/tests/test_decision_episode_export.py
  modified:
    - backend/app/research_logic/__init__.py
    - backend/app/research_logic/replay_io.py
    - backend/tests/test_replay_io.py
key-decisions:
  - "Audited exports must rebuild DecisionEpisode objects from review-allowlisted cards instead of copying replay-time selected ids."
  - "Phase 6 export bundles stay separate from replay and review bundles so audit-grade packaging never mutates Phase 5 artifacts."
patterns-established:
  - "Review allowlists, not live replay priors, are the source of truth for export carryover."
  - "Every audit export exposes explicit visible-input, audit-only, and label/eval-only reference buckets."
requirements-completed: [L4-02]
duration: 27min
completed: 2026-04-02
---

# Phase 06 Plan 01 Summary

**Audited `DecisionEpisode` exports now rebuild from review-allowlisted cards and write separate manifest/inspection bundles with explicit leakage buckets**

## Performance

- **Duration:** 27 min
- **Started:** 2026-04-02T10:55:00Z
- **Completed:** 2026-04-02T11:21:37Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added `build_decision_episode_audit_export(...)` and typed export metadata so Phase 6 can rebuild an audited episode from replay context plus review-approved allowlists.
- Added dedicated export bundle writers in `replay_io.py` for `bundle_manifest.json`, `export_summary.json`, `export_inspection.json`, and `outputs/decision_episode.json`.
- Added regression coverage proving empty accepted-prior truthfulness, route-matching accepted anti-pattern carryover, and explicit visibility-bucket reporting.

## Task Commits

Each task was committed atomically:

1. **Task 1: Build a dedicated export assembler that rebuilds the episode from reviewed acceptance state** - `f1a7507b` (`feat`)
2. **Task 2: Add a dedicated export bundle writer and inspection surfaces without mutating replay outputs** - `ec2bd59a` (`feat`)

Plan metadata is recorded separately in the docs closeout commit for this summary/state update.

## Files Created/Modified

- `backend/app/research_logic/decision_episode_export.py` - typed audited-export assembly layer built on top of `DecisionEpisodeBuilder`
- `backend/app/research_logic/replay_io.py` - export summary, inspection, and bundle-manifest writers
- `backend/app/research_logic/__init__.py` - package exports for the new Phase 6 assembly and writer entry points
- `backend/tests/test_decision_episode_export.py` - contract tests for empty prior allowlists, anti-pattern carryover, and hindsight visibility
- `backend/tests/test_replay_io.py` - export bundle persistence and visibility-bucket regression coverage

## Decisions Made

- Used the review manifest allowlists as the authoritative export gate so audited samples never silently inherit replay-time prior selection.
- Kept the export bundle separate from the replay bundle to preserve the distinction between live replay artifacts and audit-grade packaged outputs.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 6 now has a reusable audited export assembly seam and bundle contract for real bundle consumption.
- Wave 2 can add a CLI and publish the bounded jamming pilot without rewriting the Phase 6 export core.

---
*Phase: 06-decision-episode-audit-export*
*Completed: 2026-04-02*
