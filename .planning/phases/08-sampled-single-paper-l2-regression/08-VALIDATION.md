---
phase: 08
slug: sampled-single-paper-l2-regression
status: ready
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-03
---

# Phase 08 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `backend/pytest.ini` equivalent not present; repo uses direct `python -m pytest` runs |
| **Quick run command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_sampled_single_paper_regression.py -q` |
| **Full suite command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_sampled_single_paper_regression.py tests\test_sampled_single_paper_regression_cli.py tests\test_replay_io.py tests\test_paper_logic_trace_gates.py tests\test_rebuild_gate.py -q` |
| **Estimated runtime** | ~20 seconds quick / ~45 seconds full |

---

## Sampling Rate

- **After every task commit:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_sampled_single_paper_regression.py -q`
- **After every plan wave:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_sampled_single_paper_regression.py tests\test_sampled_single_paper_regression_cli.py tests\test_replay_io.py tests\test_paper_logic_trace_gates.py tests\test_rebuild_gate.py -q`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 25 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 01 | 1 | L2Q-01 | unit | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_sampled_single_paper_regression.py -q` | `tests\test_sampled_single_paper_regression.py` | pending |
| 08-01-02 | 01 | 1 | L2Q-01 | unit | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_replay_io.py -q` | `tests\test_replay_io.py` | pending |
| 08-02-01 | 02 | 2 | L2Q-02 | unit | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_sampled_single_paper_regression.py tests\test_sampled_single_paper_regression_cli.py -q` | `tests\test_sampled_single_paper_regression_cli.py` | pending |
| 08-02-02 | 02 | 2 | L2Q-02 | integration | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_sampled_single_paper_regression_cli.py tests\test_paper_logic_trace_gates.py tests\test_rebuild_gate.py -q` | `tests\test_rebuild_gate.py` | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

Existing infrastructure covers the phase gate. No separate Wave 0 prerequisite is required before execution.

The phase plans themselves create:
- `backend/tests/test_sampled_single_paper_regression.py` for selection loading, filesystem resolution, per-paper result capture, and comparison classification
- `backend/tests/test_sampled_single_paper_regression_cli.py` for iteration bundle and report generation smoke coverage
- `backend/tests/fixtures/sampled_l2_regression/` only if current corpus-sampling fixtures are insufficient

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Prioritized owner queue is actually actionable | L2Q-02 | The report can be syntactically correct while still grouping failures too vaguely | Read the committed Phase 8 report and confirm the top owner buckets map to concrete signal families such as slot recovery, relation stitching, route seed richness, metadata repair, reference recovery, or citation semantics |
| First-run comparison messaging is clear when there is no prior Phase 8 iteration | L2Q-02 | Automated tests can assert payload shape but not whether the operator-facing narrative is understandable | Review the report sections covering fixed and random cohorts and confirm they explicitly state whether the run is baseline-only or compared against a previous iteration |

---

## Validation Sign-Off

- [x] All tasks have automated verify commands or explicit Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing test references
- [x] No watch-mode flags
- [x] Feedback latency < 25s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-03
