---
phase: 07-corpus-sampling-and-regression-baseline
plan: 02
subsystem: backend
tags: [corpus-sampling, cli, docs, reports, testing]
requires:
  - phase: 07-corpus-sampling-and-regression-baseline
    provides: typed sampling core, sampling bundle contract, replay-io writers
provides:
  - operator-facing Phase 7 sampling baseline CLI
  - committed ten-paper fixed regression manifest and reviewer notes
  - real shared-corpus baseline bundle and committed audit report
affects: [phase-07, cli, neo4j, docs, reports, testing]
tech-stack:
  added: []
  patterns: [bundle-driven corpus CLI, best-effort graph enrichment, portable committed sampling docs]
key-files:
  created:
    - backend/scripts/run_corpus_sampling_baseline.py
    - backend/tests/test_corpus_sampling_cli.py
    - docs/replay/corpus_sampling/phase7-fixed-regression-set.json
    - docs/replay/corpus_sampling/phase7-fixed-regression-notes.md
    - docs/replay/reports/phase7-corpus-sampling-baseline.md
  modified:
    - backend/app/graph/neo4j_client.py
requirements-completed: [SAMPLE-01, SAMPLE-02, SAMPLE-03]
duration: local-session
completed: 2026-04-03
---

# Phase 07 Plan 02 Summary

**Phase 7 can now run a real shared-corpus sampling baseline end to end, freeze a stable ten-paper regression set, and publish a committed audit report without leaking the absolute corpus path into git**

## Accomplishments

- Added `backend/scripts/run_corpus_sampling_baseline.py` so operators can scan the shared corpus, load the committed fixed set, build the random exploration sample, and write the Phase 7 bundle in one command.
- Extended `backend/app/graph/neo4j_client.py` with markdown-path ingestion lookup so graph metadata can enrich sampled items when available without blocking the filesystem-first baseline.
- Added `backend/tests/test_corpus_sampling_cli.py` coverage for fixture-corpus runs, missing fixed ids, overlap prevention, and graceful continuation when Neo4j lookup fails.
- Curated and committed the first stable ten-paper fixed regression set plus reviewer notes under `docs/replay/corpus_sampling/`.
- Ran the real baseline against the shared corpus root and wrote the local runtime bundle under `tmp/phase7_corpus_sampling_baseline/`.
- Published `docs/replay/reports/phase7-corpus-sampling-baseline.md` with the real fixed batch, random batch, corpus-health counts, and Neo4j coverage outcome.

## Verification

Verified during execution with:

1. `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_corpus_sampling.py tests\test_corpus_sampling_cli.py tests\test_replay_io.py -q`
2. `cd backend; .\.venv\Scripts\python.exe scripts\run_corpus_sampling_baseline.py --corpus-root "\\\\192.168.199.138\\Share400T\\pub\\LLM_Data\\data\\hzy\\第一批文献\\文献\\文献中心\\HZY第一批论文全文\\output" --fixed-regression-manifest ..\\docs\\replay\\corpus_sampling\\phase7-fixed-regression-set.json --output-dir ..\\tmp\\phase7_corpus_sampling_baseline --fixed-count 10 --random-count 5 --seed 7`
3. Verified committed docs contain no `\\192.168.199.138` substring and that the runtime bundle contains `bundle_manifest.json`, `sampling_summary.json`, `sampling_inspection.json`, `outputs/fixed_regression_batch.json`, `outputs/random_exploration_batch.json`, and `outputs/corpus_health_failures.json`

Observed runtime result:

- inventory entries: `1756`
- eligible entries: `1755`
- fixed selected: `10`
- random selected: `5`
- corpus-health failures: `1505`
- Neo4j lookup status: `ready`
- selected items with Neo4j metadata: `0`

## Task Commit

1. **Plan 07-02 implementation checkpoint** - `13bc7174` (`feat`)

## Decisions Made

- Kept the fixed regression manifest committed and portable by storing only ids, titles, and corpus-relative refs.
- Chose a mixed ten-paper set that deliberately spans markdown-first pairs, txt fallbacks with broken markdown twins, and txt-only survivors.
- Treated a successful-but-empty Neo4j enrichment pass as downstream availability metadata, not as a sampling failure.

## Notes

- The first real random exploration batch used seed `7` and selected ids `17`, `1317`, `1844`, `580`, and `1107`.
- The real corpus already shows a substantial hygiene burden (`931` walk errors and `574` unreadable files), which is now visible as corpus-health signal instead of hidden selection loss.
