---
phase: 08-sampled-single-paper-l2-regression
plan: 02
subsystem: backend
tags: [sampled-regression, operator-cli, l2-quality, replay-reporting]
requires:
  - phase: 08-01
    provides: Filesystem-first sampled-paper evaluator and iteration bundle contract
  - phase: 07-corpus-sampling-and-regression-baseline
    provides: Fixed and random sampled-paper source bundle for the first real Phase 8 run
provides:
  - Comparison verdicts and owner-bucket summaries for sampled L2 iterations
  - Operator CLI that writes Phase 8 runtime bundles and markdown reports
  - First real baseline report over the 10 fixed and 5 random sampled papers
affects: [09-bounded-packet-construction-from-corpus, 10-sampled-packet-l3-l4-regression]
tech-stack:
  added: []
  patterns: [baseline-only comparison bundles, fixed-first owner queue prioritization, operator-cli plus report pair]
key-files:
  created: [backend/scripts/run_sampled_single_paper_l2_regression.py, backend/tests/test_sampled_single_paper_regression_cli.py, docs/replay/reports/phase8-sampled-single-paper-l2-regression-baseline.md]
  modified: [backend/app/ingest/rebuild.py, backend/app/research_logic/__init__.py, backend/app/research_logic/replay_io.py, backend/app/research_logic/sampled_single_paper.py, backend/tests/test_replay_io.py, backend/tests/test_sampled_single_paper_regression.py]
key-decisions:
  - "Treat a no-previous-bundle run as baseline-only while still emitting the same comparison summary and inspection contract."
  - "Prioritize owner buckets by fixed-set failures before random-only discoveries so the next L2 queue stays regression-led."
  - "Publish the first real Phase 8 report from the live runtime bundle while keeping the full per-paper artifacts local under tmp."
patterns-established:
  - "Sampled L2 comparisons key on cohort plus corpus_paper_id and keep availability-only rows out of regression counts."
  - "Phase 8 operator runs always emit machine-readable bundle files alongside a human-readable report with fixed headings."
requirements-completed: [L2Q-01, L2Q-02]
duration: 2h 2m
completed: 2026-04-03
---

# Phase 08 Plan 02: Sampled Single-Paper L2 Regression Summary

**Sampled-paper comparison workflow with owner-bucket queueing, operator CLI, and a real 15-paper baseline L2 report**

## Performance

- **Duration:** 2h 2m
- **Started:** 2026-04-03T04:07:22Z
- **Completed:** 2026-04-03T06:09:08Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- Added comparison verdicts for fixed and random sampled-paper cohorts plus owner-bucket prioritization derived from structured L2 quality signals.
- Added the Phase 8 operator CLI, comparison bundle writers, and CLI coverage for baseline-only and comparison-mode runs.
- Executed the first real baseline cycle from the Phase 7 bundle, producing `15/15` executed papers, `0` availability-only issues, `7` recurring fixed failures, `2` random edge cases, and a ranked owner queue led by `relation_assembly` then `slot_recovery`.

## Task Commits

Wave 2 landed in one implementation commit because the comparison logic, CLI wiring, tests, and first report all shared the same execution surface:

1. **Task 1 + Task 2: sampled L2 comparison layer, operator CLI, and first baseline report** - `f96b3dab` (feat)

## Files Created/Modified

- `backend/app/ingest/rebuild.py` - Preserved citation-semantic counts in sampled-paper evaluator results.
- `backend/app/research_logic/sampled_single_paper.py` - Added comparison verdicts, iteration diffing, owner-bucket derivation, and iteration bundle loading.
- `backend/app/research_logic/replay_io.py` - Added comparison summary/inspection builders and bundle writing helpers.
- `backend/app/research_logic/__init__.py` - Exported the new comparison and CLI-facing APIs.
- `backend/scripts/run_sampled_single_paper_l2_regression.py` - Added the operator CLI that runs Phase 8, writes bundle files, and renders the markdown report.
- `backend/tests/test_sampled_single_paper_regression.py` - Added comparison classification and owner-bucket prioritization coverage.
- `backend/tests/test_replay_io.py` - Added comparison bundle writer coverage.
- `backend/tests/test_sampled_single_paper_regression_cli.py` - Added CLI path, report, and comparison-mode tests.
- `docs/replay/reports/phase8-sampled-single-paper-l2-regression-baseline.md` - Published the first real sampled single-paper regression baseline report.

## Decisions Made

- Used baseline-only comparison output instead of a special-case report path so later iterations can diff against the exact same artifact contract.
- Ranked owner buckets with fixed-regression evidence first so the next L2 queue is driven by regression stability before exploratory findings.
- Kept availability-only issues as a separate surface even in the real run so the report cannot inflate regression counts with source-access noise.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The first shell invocation timed out after 30 minutes even though the real Phase 8 Python process kept running. I traced the surviving process, monitored it through completion, and verified the final bundle plus report once the live run finished.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 8 now provides a real baseline bundle and report that clearly identify `relation_assembly` and `slot_recovery` as the top fixed-set L2 owners.
- Phase 9 can consume the Phase 8 findings without replaying the sampled-paper baseline, because the bundle and report already capture the fixed/random execution surface and the next owner queue.

---
*Phase: 08-sampled-single-paper-l2-regression*
*Completed: 2026-04-03*
