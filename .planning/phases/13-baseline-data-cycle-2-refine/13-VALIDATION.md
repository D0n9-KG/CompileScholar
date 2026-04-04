---
phase: 13
slug: baseline-data-cycle-2-refine
status: ready
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-04
---

# Phase 13 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `backend/pytest.ini` equivalent not present; repo uses direct `python -m pytest` runs |
| **Quick run command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase13_cycle_refinement.py tests\test_phase10_multi_paper_validation.py tests\test_historical_replay_compiler.py tests\test_prior_induction.py tests\test_replay_io.py -q` |
| **Full suite command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase13_cycle_refinement.py tests\test_phase10_multi_paper_validation.py tests\test_phase10_multi_paper_validation_cli.py tests\test_historical_replay_compiler.py tests\test_prior_induction.py tests\test_replay_io.py tests\test_iteration_prioritization.py -q` |
| **Estimated runtime** | ~30 seconds quick / ~60 seconds full |

---

## Sampling Rate

- **After every task commit:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase13_cycle_refinement.py tests\test_phase10_multi_paper_validation.py tests\test_historical_replay_compiler.py tests\test_prior_induction.py tests\test_replay_io.py -q`
- **After every plan wave:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase13_cycle_refinement.py tests\test_phase10_multi_paper_validation.py tests\test_phase10_multi_paper_validation_cli.py tests\test_historical_replay_compiler.py tests\test_prior_induction.py tests\test_replay_io.py tests\test_iteration_prioritization.py -q`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 45 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 13-01-01 | 01 | 1 | LOOPR-03 | smoke | `cd backend; .\.venv\Scripts\python.exe scripts\run_phase13_cycle_refinement.py --help` | `backend/scripts/run_phase13_cycle_refinement.py` | pending |
| 13-01-02 | 01 | 1 | LOOPR-03 | unit | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase13_cycle_refinement.py -q` | `backend/tests/test_phase13_cycle_refinement.py` | pending |
| 13-02-01 | 02 | 2 | LOOPR-03 | integration | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase10_multi_paper_validation.py tests\test_phase10_multi_paper_validation_cli.py tests\test_historical_replay_compiler.py -q` | `backend/tests/test_phase10_multi_paper_validation.py` | pending |
| 13-02-02 | 02 | 2 | LOOPR-03 | integration | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_prior_induction.py tests\test_phase10_multi_paper_validation.py -q` | `backend/tests/test_prior_induction.py` | pending |
| 13-03-01 | 03 | 3 | LOOPR-03 | integration | `cd backend; .\.venv\Scripts\python.exe scripts\run_phase13_cycle_refinement.py --iteration-label iter-01 --reviewer reviewer-1 --reviewer reviewer-2 --output-root ..\tmp\phase13_cycle2_refine` | `.planning/phases/13-baseline-data-cycle-2-refine/13-VERIFICATION.md` | pending |
| 13-03-02 | 03 | 3 | LOOPR-03, REVIEW-01 | integration | `cd backend; .\.venv\Scripts\python.exe scripts\run_phase11_iteration_prioritization.py --phase8-summary ..\tmp\phase8_sampled_single_paper_l2\baseline-cycle-01\comparison_summary.json --phase8-inspection ..\tmp\phase8_sampled_single_paper_l2\baseline-cycle-01\comparison_inspection.json --phase10-summary ..\tmp\phase13_cycle2_refine\cycle2-final\comparison_summary.json --phase10-verification ..\.planning\phases\13-baseline-data-cycle-2-refine\13-VERIFICATION.md --phase10-report ..\docs\replay\reports\phase13-cycle2-refine.md --output-dir ..\tmp\phase13_cycle2_refine\final-prioritization --report-md ..\docs\replay\reports\phase13-next-cycle-prioritization.md` | `docs/replay/reports/phase13-next-cycle-prioritization.md` | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

Existing infrastructure already covers the phase gate. No separate Wave 0 framework work is required before execution.

The phase plans themselves should create or extend:
- focused Phase 13 regression tests around baseline wiring and repeated iteration output structure,
- replay/prior/export verification that proves reviewer metadata and prior-candidate movement are visible in structured outputs,
- report / verification assertions that the final closeout still cites the Phase 12 cycle-1 baseline.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Final iteration truly shows movement against cycle 1 | LOOPR-03 | Automated tests can prove artifact presence and field values, but not whether the final quality story is substantively improved | Read the final Phase 13 report and verification note; confirm they compare the chosen final iteration directly to `tmp/phase12_direct_fix_cycle/cycle1/` and explain what moved in package, replay, prior-review, and export |
| Multiple inner iterations stayed bounded and auditable | LOOPR-03 | Tests can assert output roots exist, but not whether the iteration trail is understandable to an operator | Review the Phase 13 output roots and report trail; confirm each meaningful iteration has distinct output paths and the final closeout summarizes how the iterations progressed |
| Manual review is based on real reasoning artifacts rather than metrics alone | REVIEW-01 | This is a human judgment requirement by definition | Confirm the final report contains explicit output-level observations on real reasoning artifacts and not only count deltas or quality-flag summaries |

---

## Validation Sign-Off

- [x] All tasks have automated verify commands or existing infrastructure coverage
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing test references
- [x] No watch-mode flags
- [x] Feedback latency < 45s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-04
