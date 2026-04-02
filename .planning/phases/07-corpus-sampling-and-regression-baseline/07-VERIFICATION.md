---
status: passed
phase: 07-corpus-sampling-and-regression-baseline
requirements: [SAMPLE-01, SAMPLE-02, SAMPLE-03]
updated: 2026-04-03T01:34:07+08:00
---

# Phase 07 Verification

## Goal

Verify that Phase 7 turns the shared corpus into a reproducible evaluation source with one fixed regression set, one random exploration batch, portable committed docs, and explicit corpus-health reporting that stays separate from downstream metadata gaps.

## Automated Checks

Passed:

- `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_corpus_sampling.py tests\test_corpus_sampling_cli.py tests\test_replay_io.py -q`

Result:

- `18 passed`

Coverage from the targeted suite:

- filesystem inventory grouping and md-vs-txt preference rules
- fixed and random batch reproducibility with overlap prevention
- sampling bundle manifest / summary / inspection persistence
- CLI behavior for fixture corpora, missing fixed ids, and best-effort Neo4j enrichment

## Runtime Validation

Real pilot command completed successfully against the shared corpus root:

- command: `cd backend; .\.venv\Scripts\python.exe scripts\run_corpus_sampling_baseline.py --corpus-root "\\\\192.168.199.138\\Share400T\\pub\\LLM_Data\\data\\hzy\\第一批文献\\文献\\文献中心\\HZY第一批论文全文\\output" --fixed-regression-manifest ..\\docs\\replay\\corpus_sampling\\phase7-fixed-regression-set.json --output-dir ..\\tmp\\phase7_corpus_sampling_baseline --fixed-count 10 --random-count 5 --seed 7`
- bundle manifest: `tmp/phase7_corpus_sampling_baseline/bundle_manifest.json`
- sampling summary: `tmp/phase7_corpus_sampling_baseline/sampling_summary.json`
- sampling inspection: `tmp/phase7_corpus_sampling_baseline/sampling_inspection.json`
- fixed batch: `tmp/phase7_corpus_sampling_baseline/outputs/fixed_regression_batch.json`
- random batch: `tmp/phase7_corpus_sampling_baseline/outputs/random_exploration_batch.json`
- corpus-health failures: `tmp/phase7_corpus_sampling_baseline/outputs/corpus_health_failures.json`
- committed report: `docs/replay/reports/phase7-corpus-sampling-baseline.md`

Observed results:

- inventory entry count: `1756`
- eligible entry count: `1755`
- fixed selected count: `10`
- random selected count: `5`
- corpus-health failure count: `1505`
- Neo4j lookup status: `ready`
- selected with Neo4j metadata: `0`
- selected without Neo4j metadata: `15`

## Requirement Check

### `SAMPLE-01`

Operator can define one fixed regression paper set and one random exploration paper set from the shared corpus for each iteration cycle.

Status: `passed`

Evidence:

- `backend/scripts/run_corpus_sampling_baseline.py` accepts explicit fixed/random counts, seed, corpus root, and fixed manifest path
- `docs/replay/corpus_sampling/phase7-fixed-regression-set.json` freezes exactly ten stable papers
- the real baseline bundle records a five-paper random exploration batch that excludes all fixed ids

### `SAMPLE-02`

Each sampling run records selected paper ids, source paths, and sampling mode so the exact batch can be reproduced.

Status: `passed`

Evidence:

- `backend/app/research_logic/replay_io.py` now writes `bundle_manifest.json`, `sampling_summary.json`, and `sampling_inspection.json`
- `outputs/fixed_regression_batch.json` and `outputs/random_exploration_batch.json` preserve selected ids, source refs, seed, and exclusion reasons
- committed fixed-set docs remain portable because they use corpus-relative refs rather than the absolute UNC root

### `SAMPLE-03`

Sampling reports missing or broken corpus paths separately from model-quality failures.

Status: `passed`

Evidence:

- `backend/app/research_logic/corpus_sampling.py` emits explicit `walk_error` and `unreadable_file` records during traversal
- the real runtime bundle reported `931` walk errors and `574` unreadable files
- `sampling_inspection.json` keeps `corpus_health_failures` distinct from `selected_but_downstream_unavailable`

## Notes

- The real Phase 7 run confirmed the filesystem is the correct eligibility gate: sampling succeeded even though graph enrichment returned no metadata for the selected items.
- The current corpus has substantial path hygiene debt, but that debt is now explicit and auditable instead of being silently absorbed into selection behavior.
