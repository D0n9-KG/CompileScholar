---
phase: 07
slug: corpus-sampling-and-regression-baseline
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-03
updated: 2026-04-03
---

# Phase 07 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest` |
| **Config file** | `backend/pytest.ini` |
| **Quick run command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_corpus_sampling.py tests\test_replay_io.py -q` |
| **Full suite command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_corpus_sampling.py tests\test_corpus_sampling_cli.py tests\test_replay_io.py -q` |
| **Estimated runtime** | ~15 seconds for local tests, excluding the real corpus pilot |

## Sampling Rate

- **After every task commit:** Run the quick command
- **After every plan wave:** Run the full suite command
- **Before Phase verify/closeout:** Full suite must be green and one real corpus pilot must be inspected
- **Max feedback latency:** 20 seconds for automated tests

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | SAMPLE-01, SAMPLE-03 | inventory / eligibility contract | quick | `backend/tests/test_corpus_sampling.py` | W0 CREATE |
| 07-01-02 | 01 | 1 | SAMPLE-02, SAMPLE-03 | artifact I/O | quick | `backend/tests/test_replay_io.py` | COVERED |
| 07-02-01 | 02 | 2 | SAMPLE-01, SAMPLE-02 | CLI smoke / metadata enrichment | full | `backend/tests/test_corpus_sampling_cli.py` | W0 CREATE |
| 07-02-02 | 02 | 2 | SAMPLE-01, SAMPLE-02, SAMPLE-03 | runtime baseline audit | full + report | `docs/replay/reports/phase7-corpus-sampling-baseline.md` | W0 CREATE |

*Status: `COVERED` = an adjacent regression surface already exists and must be extended. `W0 CREATE` = execution must add the missing verification surface in the first applicable wave.*

## Wave 0 Requirements

Execution must create the missing Phase 7 verification surfaces early:

- `backend/tests/test_corpus_sampling.py` - inventory grouping, preferred-source selection, fixed/random batch reproducibility, and corpus-health classification
- `backend/tests/test_corpus_sampling_cli.py` - fixture-corpus CLI coverage, fixed-set loading, missing-id handling, and best-effort Neo4j enrichment behavior
- `docs/replay/reports/phase7-corpus-sampling-baseline.md` - committed audit report for the first real sampling baseline run

## Runtime Artifact Audit

In addition to automated pytest coverage, Phase 7 closeout must inspect one real sampling bundle built from the shared corpus root.

Source artifacts to audit:

- shared corpus root `\\192.168.199.138\Share400T\pub\LLM_Data\data\hzy\第一批文献\文献\文献中心\HZY第一批论文全文\output`
- `docs/replay/corpus_sampling/phase7-fixed-regression-set.json`
- `docs/replay/corpus_sampling/phase7-fixed-regression-notes.md`
- `tmp/phase7_corpus_sampling_baseline/` (new local output directory)

At closeout, the audit must confirm:

- the fixed regression set contains exactly `10` documented papers;
- the random exploration batch contains exactly `5` papers and excludes all fixed-set ids;
- the runtime bundle writes `bundle_manifest.json`, `sampling_summary.json`, `sampling_inspection.json`, `outputs/fixed_regression_batch.json`, `outputs/random_exploration_batch.json`, and `outputs/corpus_health_failures.json`;
- corpus-health failures are listed separately from `selected_but_downstream_unavailable`;
- committed docs do not contain the absolute UNC corpus root;
- the bundle records whether Neo4j metadata lookup succeeded, failed, or was skipped.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Fixed-set docs remain portable and do not leak machine-local paths | SAMPLE-02 | Automated tests can grep for path substrings, but a reviewer should confirm the docs still read like committed audit assets rather than local runtime dumps | Inspect `docs/replay/corpus_sampling/phase7-fixed-regression-set.json` and `docs/replay/corpus_sampling/phase7-fixed-regression-notes.md` and confirm they use ids plus corpus-relative refs only |
| Corpus-health failures are semantically separated from downstream-unavailable selections | SAMPLE-03 | Tests can check key names, but a reviewer must judge whether the bucket meanings are actually distinct | Read `tmp/phase7_corpus_sampling_baseline/sampling_inspection.json` and confirm broken paths, unreadable files, and graph-unavailable selections are not merged into one count |
| The fixed set is intentionally stable rather than accidentally first-10 sorted rows | SAMPLE-01 | Tests can verify count, but not selection quality | Read `docs/replay/corpus_sampling/phase7-fixed-regression-notes.md` and confirm each of the ten papers has a documented selection reason |

## Validation Sign-Off

- [x] All tasks have automated verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency < 20s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-03
