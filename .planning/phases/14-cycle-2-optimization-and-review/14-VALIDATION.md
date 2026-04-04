---
phase: 14
slug: cycle-2-optimization-and-review
status: ready
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-04
---

# Phase 14 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `backend/pytest.ini` equivalent not present; repo uses direct `python -m pytest` runs |
| **Quick run command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_episode_export.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_historical_replay_compiler.py tests\test_route_state_synthesizer.py tests\test_iteration_prioritization.py -q` |
| **Full suite command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_episode_export.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_phase10_multi_paper_validation_cli.py tests\test_historical_replay_compiler.py tests\test_route_state_synthesizer.py tests\test_iteration_prioritization.py tests\test_prior_induction.py -q` |
| **Estimated runtime** | ~35 seconds quick / ~70 seconds full |

---

## Sampling Rate

- **After every task commit:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_episode_export.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_historical_replay_compiler.py tests\test_route_state_synthesizer.py tests\test_iteration_prioritization.py -q`
- **After every plan wave:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_episode_export.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_phase10_multi_paper_validation_cli.py tests\test_historical_replay_compiler.py tests\test_route_state_synthesizer.py tests\test_iteration_prioritization.py tests\test_prior_induction.py -q`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 45 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 14-01-01 | 01 | 1 | TRAIN-02 | unit | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_route_state_synthesizer.py -q` | `backend/tests/test_route_state_synthesizer.py` | pending |
| 14-01-02 | 01 | 1 | TRAIN-03, TRAIN-06 | integration | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_episode_export.py tests\test_replay_io.py -q` | `backend/tests/test_decision_episode_export.py` | pending |
| 14-02-01 | 02 | 2 | TRAIN-01, TRAIN-03 | integration | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase10_multi_paper_validation.py tests\test_replay_io.py tests\test_decision_episode_export.py -q` | `backend/tests/test_phase10_multi_paper_validation.py` | pending |
| 14-02-02 | 02 | 2 | REVIEW-02, REVIEW-03, TRAIN-06 | integration | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_iteration_prioritization.py tests\test_phase10_multi_paper_validation.py tests\test_historical_replay_compiler.py -q` | `backend/tests/test_iteration_prioritization.py` | pending |
| 14-03-01 | 03 | 3 | REVIEW-02, TRAIN-01 | smoke | `cd backend; .\.venv\Scripts\python.exe scripts\run_phase13_cycle_refinement.py --iteration-label phase14-candidate-01 --output-root ..\tmp\phase14_cycle2_optimization --report-md ..\docs\replay\reports\phase14-cycle2-optimization-candidate-01.md --reviewer reviewer-1 --reviewer reviewer-2`, then repeat the same command with `--iteration-label cycle2-best` and `--report-md ..\docs\replay\reports\phase14-cycle2-optimization-best.md` | `backend/scripts/run_phase13_cycle_refinement.py` | pending |
| 14-03-02 | 03 | 3 | REVIEW-03, TRAIN-03, TRAIN-06 | integration | `cd backend; .\.venv\Scripts\python.exe scripts\run_phase11_iteration_prioritization.py --phase8-summary ..\tmp\phase8_sampled_single_paper_l2\baseline-cycle-01\comparison_summary.json --phase8-inspection ..\tmp\phase8_sampled_single_paper_l2\baseline-cycle-01\comparison_inspection.json --phase10-summary ..\tmp\phase14_cycle2_optimization\cycle2-best\comparison_summary.json --phase10-verification ..\.planning\phases\14-cycle-2-optimization-and-review\14-VERIFICATION.md --phase10-report ..\docs\replay\reports\phase14-cycle2-optimization-best.md --output-dir ..\tmp\phase14_cycle2_optimization\final-prioritization --report-md ..\docs\replay\reports\phase14-next-cycle-prioritization.md` | `docs/replay/reports/phase14-next-cycle-prioritization.md` | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

Existing infrastructure already covers the phase gate. No separate Wave 0 framework work is required before execution.

The phase plans themselves should create or extend:
- focused schema and export tests for canonical-label-plus-raw-phrase preservation,
- regression coverage for structured review / training-acceptance metadata and additive training-facing bundle files,
- end-to-end validation that the chosen best cycle emits review-backed training artifacts and recommendation evidence.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Chosen cycle is genuinely useful as scientific-thinking training data | TRAIN-01, TRAIN-03 | Only a human can judge whether the reasoning artifact is substantively useful for training rather than merely structurally valid | Read the chosen Phase 14 package / replay / prior / export outputs and the final training-facing JSON view; confirm the artifact would actually teach scientific reasoning rather than only pass flags |
| Every major training-data section stayed in optimization scope | TRAIN-06 | Automated tests can prove fields exist, but not that reviewers truly evaluated their content quality | Review the final markdown report and machine-readable section review records; confirm evidence pack, route synthesis, why-now, route comparison, priors / anti-patterns, minimal attack path, final decision, and review labels all receive explicit judgment |
| Next-cycle recommendation is justified by reviewed evidence | REVIEW-02, REVIEW-03 | Tests can verify the presence of recommendation fields, but a human must decide whether the rationale matches the observed defects | Compare `best_cycle_selection` style JSON, final report, and prioritization report; confirm the recommendation cites concrete reviewed defects and not metric-only summaries |

---

## Validation Sign-Off

- [x] All tasks have automated verify commands or existing infrastructure coverage
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing test references
- [x] No watch-mode flags
- [x] Feedback latency < 45s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-04
