---
phase: 11
slug: iteration-prioritization-and-next-cycle-plan
status: ready
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-03
---

# Phase 11 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `backend/pytest.ini` equivalent not present; repo uses direct `python -m pytest` runs |
| **Quick run command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_iteration_prioritization.py tests\test_replay_io.py -q` |
| **Full suite command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_iteration_prioritization.py tests\test_iteration_prioritization_cli.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_sampled_single_paper_regression.py -q` |
| **Estimated runtime** | ~15 seconds quick / ~40 seconds full |

---

## Sampling Rate

- **After every task commit:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_iteration_prioritization.py tests\test_replay_io.py -q`
- **After every plan wave:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_iteration_prioritization.py tests\test_iteration_prioritization_cli.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_sampled_single_paper_regression.py -q`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 20 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 11-01-01 | 01 | 1 | LOOP-01 | unit | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_iteration_prioritization.py -q` | `tests\test_iteration_prioritization.py` | pending |
| 11-01-02 | 01 | 1 | LOOP-01, LOOP-02 | unit | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_iteration_prioritization.py tests\test_replay_io.py -q` | `tests\test_replay_io.py` | pending |
| 11-02-01 | 02 | 2 | LOOP-02 | unit | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_iteration_prioritization_cli.py -q` | `tests\test_iteration_prioritization_cli.py` | pending |
| 11-02-02 | 02 | 2 | LOOP-01, LOOP-02 | integration | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_iteration_prioritization.py tests\test_iteration_prioritization_cli.py tests\test_phase10_multi_paper_validation.py tests\test_sampled_single_paper_regression.py -q` | `tests\test_phase10_multi_paper_validation.py` | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

Existing infrastructure covers the phase gate. No separate Wave 0 prerequisite is required before execution.

The phase plans themselves should create:
- `backend/tests/test_iteration_prioritization.py` for evidence loading, normalization, ranking, and missing-input blocker behavior
- `backend/tests/test_iteration_prioritization_cli.py` for CLI/report generation and provenance smoke coverage
- focused fixtures only if current Phase 8 / Phase 10 bundle samples are insufficient for deterministic tests

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Primary recommendation is actually actionable | LOOP-02 | Automated tests can assert ranking shape but not whether the recommendation reads as a usable next-cycle decision | Read the committed Phase 11 report and confirm it clearly says why packet construction, `L4`, or `L2` was chosen first and what the next two follow-ups are |
| Missing-artifact fallback is honest | LOOP-01 | Tests can assert provenance fields exist, but not whether the operator-facing explanation is understandable | Review the report when Phase 10 JSON input is absent and confirm it states exactly which fallback files were used and what evidence remained unavailable |

---

## Validation Sign-Off

- [x] All tasks have automated verify commands or explicit Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing test references
- [x] No watch-mode flags
- [x] Feedback latency < 20s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-03
