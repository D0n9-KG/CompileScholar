---
phase: 15
slug: cycle-3-consolidation
status: ready
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-04
---

# Phase 15 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `backend/pytest.ini` equivalent not present; repo uses direct `python -m pytest` runs |
| **Quick run command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_bounded_packet_audit.py tests\test_route_state_package.py tests\test_prior_induction.py tests\test_route_comparison_builder.py tests\test_why_now_builder.py tests\test_decision_episode_builder.py tests\test_decision_episode_export.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_historical_replay_compiler.py tests\test_iteration_prioritization.py -q` |
| **Full suite command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_bounded_packet_audit.py tests\test_route_state_package.py tests\test_prior_induction.py tests\test_route_comparison_builder.py tests\test_why_now_builder.py tests\test_decision_episode_builder.py tests\test_decision_episode_export.py tests\test_decision_episode_export_cli.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_phase10_multi_paper_validation_cli.py tests\test_phase13_cycle_refinement.py tests\test_historical_replay_compiler.py tests\test_iteration_prioritization.py tests\test_iteration_prioritization_cli.py -q` |
| **Estimated runtime** | ~55 seconds quick / ~95 seconds full |

---

## Sampling Rate

- **After every task commit:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_bounded_packet_audit.py tests\test_route_state_package.py tests\test_prior_induction.py tests\test_route_comparison_builder.py tests\test_why_now_builder.py tests\test_decision_episode_builder.py tests\test_decision_episode_export.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_historical_replay_compiler.py tests\test_iteration_prioritization.py -q`
- **After every plan wave:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_bounded_packet_audit.py tests\test_route_state_package.py tests\test_prior_induction.py tests\test_route_comparison_builder.py tests\test_why_now_builder.py tests\test_decision_episode_builder.py tests\test_decision_episode_export.py tests\test_decision_episode_export_cli.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_phase10_multi_paper_validation_cli.py tests\test_phase13_cycle_refinement.py tests\test_historical_replay_compiler.py tests\test_iteration_prioritization.py tests\test_iteration_prioritization_cli.py -q`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 15-01-01 | 01 | 1 | STAB-01 | integration | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_bounded_packet_audit.py tests\test_route_state_package.py tests\test_phase10_multi_paper_validation.py -q` | `backend/tests/test_route_state_package.py` | pending |
| 15-01-02 | 01 | 1 | TRAIN-04 | integration | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_prior_induction.py tests\test_phase10_multi_paper_validation.py tests\test_phase13_cycle_refinement.py -q` | `backend/tests/test_prior_induction.py` | pending |
| 15-02-01 | 02 | 2 | STAB-01 | integration | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_route_comparison_builder.py tests\test_why_now_builder.py tests\test_decision_episode_builder.py tests\test_historical_replay_compiler.py -q` | `backend/tests/test_route_comparison_builder.py` | pending |
| 15-02-02 | 02 | 2 | TRAIN-04 | integration | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_episode_builder.py tests\test_decision_episode_export.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py -q` | `backend/tests/test_decision_episode_export.py` | pending |
| 15-03-01 | 03 | 3 | STAB-01, TRAIN-04 | smoke | `cd backend; .\.venv\Scripts\python.exe scripts\run_phase13_cycle_refinement.py --iteration-label phase15-default-01 --output-root ..\tmp\phase15_cycle3_consolidation --baseline-replay-bundle ..\tmp\phase14_cycle2_optimization\cycle2-best\replay_bundle --baseline-export-bundle ..\tmp\phase14_cycle2_optimization\cycle2-best\export_bundle --report-md ..\docs\replay\reports\phase15-cycle3-default-01.md --reviewer reviewer-1 --reviewer reviewer-2`, then `cd backend; .\.venv\Scripts\python.exe scripts\run_phase13_cycle_refinement.py --iteration-label phase15-fallback-01 --output-root ..\tmp\phase15_cycle3_consolidation --baseline-replay-bundle ..\tmp\phase14_cycle2_optimization\cycle2-best\replay_bundle --baseline-export-bundle ..\tmp\phase14_cycle2_optimization\cycle2-best\export_bundle --report-md ..\docs\replay\reports\phase15-cycle3-fallback-01.md --reviewer reviewer-1 --reviewer reviewer-2 --allow-scope-fallback-merge` | `backend/scripts/run_phase13_cycle_refinement.py` | pending |
| 15-03-02 | 03 | 3 | STAB-01 | integration | `cd backend; .\.venv\Scripts\python.exe scripts\run_phase11_iteration_prioritization.py --phase8-summary ..\tmp\phase8_sampled_single_paper_l2\baseline-cycle-01\comparison_summary.json --phase8-inspection ..\tmp\phase8_sampled_single_paper_l2\baseline-cycle-01\comparison_inspection.json --phase10-summary ..\tmp\phase15_cycle3_consolidation\cycle3-best\comparison_summary.json --phase10-verification ..\.planning\phases\15-cycle-3-consolidation\15-VERIFICATION.md --phase10-report ..\docs\replay\reports\phase15-cycle3-best.md --output-dir ..\tmp\phase15_cycle3_consolidation\final-prioritization --report-md ..\docs\replay\reports\phase15-next-cycle-prioritization.md` | `docs/replay/reports/phase15-next-cycle-prioritization.md` | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

Existing infrastructure already covers the phase gate. No separate Wave 0 framework work is required before execution.

The phase plans themselves should create or extend:
- packet-quality regression coverage that keeps alternative-route distinctness explicit and localizes `yellow_route_state_present`,
- prior-recovery lever coverage that distinguishes default and fallback-merge reruns,
- route-comparison and why-now coverage that proves grounded route advantage and timing explanation fields exist,
- export-closure coverage for route-backed selected ids plus explicit structured exclusion records,
- end-to-end rerun validation against the completed Phase 14 `cycle2-best` baseline bundles.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Chosen Phase 15 cycle is genuinely high quality as scientific-thinking training data | STAB-01 | Only a human can judge whether the artifact teaches usable scientific reasoning rather than merely satisfying structure or flags | Read the chosen `cycle3-best` training view, export summary, best-cycle selection, and markdown report; decide whether the reasoning is actually useful for training |
| Reviewed prior / anti-pattern knowledge closes honestly into export selection | TRAIN-04 | Tests can prove fields exist, but a reviewer must decide whether the selected or excluded heuristics truly match the exported route | Compare accepted cards, selected ids, exclusion records, and route-state support in the chosen Phase 15 cycle; confirm the export did not force mismatched ids through |
| Accepted-cycle streak is recorded honestly without overclaiming stability | STAB-01 | Automation can count candidate runs, but only a human can confirm whether multiple runs truly clear the bar versus only one | Review `15-VERIFICATION.md`, `best_cycle_selection.json`, and the final report; confirm the streak count and verdict language match what the artifacts actually show |

---

## Validation Sign-Off

- [x] All tasks have automated verify commands or existing infrastructure coverage
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing test references
- [x] No watch-mode flags
- [x] Feedback latency < 60s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-04
