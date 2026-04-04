---
phase: 16
slug: stability-verification-and-handoff
status: ready
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-04
---

# Phase 16 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `backend/pytest.ini` equivalent not present; repo uses direct `python -m pytest` runs |
| **Quick run command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_episode_export.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_phase13_cycle_refinement.py tests\test_iteration_prioritization.py -q` |
| **Full suite command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_episode_export.py tests\test_decision_episode_export_cli.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_phase10_multi_paper_validation_cli.py tests\test_phase13_cycle_refinement.py tests\test_iteration_prioritization.py tests\test_iteration_prioritization_cli.py -q` |
| **Estimated runtime** | ~45 seconds quick / ~90 seconds full |

---

## Sampling Rate

- **After every task commit:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_episode_export.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_phase13_cycle_refinement.py tests\test_iteration_prioritization.py -q`
- **After every plan wave:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_episode_export.py tests\test_decision_episode_export_cli.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_phase10_multi_paper_validation_cli.py tests\test_phase13_cycle_refinement.py tests\test_iteration_prioritization.py tests\test_iteration_prioritization_cli.py -q`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 16-01-01 | 01 | 1 | TRAIN-05 | integration | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_episode_export.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py -q` | `backend/tests/test_replay_io.py` | pending |
| 16-01-02 | 01 | 1 | TRAIN-07 | integration | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py -q` | `backend/tests/test_phase10_multi_paper_validation.py` | pending |
| 16-02-01 | 02 | 2 | STAB-03 | integration | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_phase13_cycle_refinement.py -q` | `backend/tests/test_phase13_cycle_refinement.py` | pending |
| 16-02-02 | 02 | 2 | STAB-03, TRAIN-07 | integration | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_iteration_prioritization.py -q` | `backend/tests/test_iteration_prioritization.py` | pending |
| 16-03-01 | 03 | 3 | STAB-02 | smoke | `cd backend; .\.venv\Scripts\python.exe scripts\run_phase13_cycle_refinement.py --iteration-label phase16-repeat-01 --output-root ..\tmp\phase16_stability_verification --baseline-replay-bundle ..\tmp\phase15_cycle3_consolidation\cycle3-best\replay_bundle --baseline-export-bundle ..\tmp\phase15_cycle3_consolidation\cycle3-best\export_bundle --report-md ..\docs\replay\reports\phase16-repeat-01.md --reviewer reviewer-1 --reviewer reviewer-2 --allow-scope-fallback-merge` | `backend/scripts/run_phase13_cycle_refinement.py` | pending |
| 16-03-02 | 03 | 3 | STAB-03, TRAIN-07 | integration | `cd backend; .\.venv\Scripts\python.exe scripts\run_phase11_iteration_prioritization.py --phase8-summary ..\tmp\phase8_sampled_single_paper_l2\baseline-cycle-01\comparison_summary.json --phase8-inspection ..\tmp\phase8_sampled_single_paper_l2\baseline-cycle-01\comparison_inspection.json --phase10-summary ..\tmp\phase16_stability_verification\cycle4-stable\comparison_summary.json --phase10-verification ..\.planning\phases\16-stability-verification-and-handoff\16-VERIFICATION.md --phase10-report ..\docs\replay\reports\phase16-cycle4-stable.md --output-dir ..\tmp\phase16_stability_verification\final-prioritization --report-md ..\docs\replay\reports\phase16-next-milestone-prioritization.md` | `docs/replay/reports/phase16-next-milestone-prioritization.md` | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

Existing infrastructure already covers the phase gate. No separate Wave 0 framework work is required before execution.

The phase plans themselves should create or extend:
- additive export coverage for task-specific training views that preserve the current umbrella `training_view.json`,
- bundle and manifest coverage for a final bounded dataset / handoff package that references validated later-cycle artifacts,
- runtime-vs-reviewed truth separation coverage so milestone closeout never overwrites runtime export defaults,
- CLI and comparison-summary coverage for new task-view refs and reviewed closeout refs,
- bounded rerun validation against the accepted Phase 15 `cycle3-best` replay and export bundles.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Repeated Phase 16 cycle is genuinely as convincing as the accepted Phase 15 cycle | STAB-02 | Only a human can judge whether the new bounded cycle still teaches usable scientific reasoning rather than merely reproducing structure | Read `comparison_summary.json`, `best_cycle_selection.json`, `training_view.json`, the five task-specific views, and the markdown report for the repeated Phase 16 cycle; compare directly against `phase15-cycle3-best.md` |
| Final verification explains stability honestly and does not overclaim beyond the evidence | STAB-03 | Tests can prove fields exist, but only a reviewer can judge whether the stability argument really matches the artifacts | Review `16-VERIFICATION.md`, `phase16-cycle4-stable.md`, `stability_handoff.json`, and both accepted-cycle roots; confirm the recorded streak, verdict, and residual-risk language match the artifacts |
| Final bounded dataset bundle is actually usable by a downstream consumer without guessing canonical files | TRAIN-07 | Automation can prove files exist, but not whether the bundle is understandable and operationally clear | Open `dataset_manifest.json`, `outputs/training_views_index.json`, and `bundle_manifest.json`; verify a reader can find the umbrella view, each task-specific view, schema refs, provenance roots, and residual-risk source without consulting repo history |

---

## Validation Sign-Off

- [x] All tasks have automated verify commands or existing infrastructure coverage
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing test references
- [x] No watch-mode flags
- [x] Feedback latency < 60s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-04
