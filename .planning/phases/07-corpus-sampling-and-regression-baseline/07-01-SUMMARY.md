---
phase: 07-corpus-sampling-and-regression-baseline
plan: 01
subsystem: backend
tags: [corpus-sampling, replay-io, testing]
requires:
  - phase: 06-decision-episode-audit-export
    provides: bundle manifest, summary, and inspection patterns for auditable local artifacts
provides:
  - typed filesystem-first corpus inventory and batch-selection helpers
  - dedicated Phase 7 sampling manifest, summary, and inspection bundle writers
  - regression tests for reproducible selection and corpus-health separation
affects: [phase-07, research-logic, replay-io, testing]
tech-stack:
  added: []
  patterns: [filesystem-first inventory, md-txt family collapse, manifest-summary-inspection bundle]
key-files:
  created:
    - backend/app/research_logic/corpus_sampling.py
    - backend/tests/test_corpus_sampling.py
  modified:
    - backend/app/research_logic/__init__.py
    - backend/app/research_logic/replay_io.py
    - backend/tests/test_replay_io.py
requirements-supported: [SAMPLE-01, SAMPLE-02, SAMPLE-03]
duration: local-session
completed: 2026-04-03
---

# Phase 07 Plan 01 Summary

**Phase 7 now has a typed sampling core and a dedicated local bundle contract built around filesystem inventory instead of graph-only eligibility**

## Accomplishments

- Added `backend/app/research_logic/corpus_sampling.py` with typed inventory, corpus-health, fixed-batch, random-batch, and bundle models plus `scan_corpus_inventory(...)`, `load_fixed_regression_manifest(...)`, `build_fixed_regression_batch(...)`, and `build_random_exploration_batch(...)`.
- Implemented recursive filesystem scanning with explicit `walk_error` and `unreadable_file` capture so broken corpus paths survive as auditable health artifacts instead of disappearing from counts.
- Extended `backend/app/research_logic/replay_io.py` with `build_corpus_sampling_summary(...)`, `build_corpus_sampling_inspection(...)`, and `write_corpus_sampling_bundle(...)` so Phase 7 artifacts follow the repo's existing manifest / summary / inspection style.
- Added focused regression coverage proving md-plus-txt family collapse, markdown-vs-txt preference rules, overlap-free random selection, and separation between corpus-health failures and downstream-unavailable selections.

## Verification

Verified during execution with:

1. `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_corpus_sampling.py tests\test_replay_io.py -q`

Result:

- `14 passed`

## Task Commit

1. **Plan 07-01 implementation checkpoint** - `af5188e9` (`feat`)

This plan's two tasks landed in one commit because the new public exports, bundle writers, and regression boundaries span the same backend files.

## Decisions Made

- Filesystem inventory is the Phase 7 eligibility gate; Neo4j metadata remains optional enrichment.
- Numeric filename prefixes collapse md and txt siblings into one stable `corpus_paper_id`, with a hash fallback only when no prefix exists.
- The new sampling artifact contract mirrors prior replay/review/export bundle conventions instead of inventing a one-off JSON dump format.

## Next Phase Readiness

- Wave 2 can add the operator CLI on top of the new sampling core without rethinking the data model.
- The bundle contract is now stable enough for a real shared-corpus baseline run and a committed fixed regression manifest.
