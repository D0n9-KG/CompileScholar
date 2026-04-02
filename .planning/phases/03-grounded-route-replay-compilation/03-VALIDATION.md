---
phase: 03
slug: grounded-route-replay-compilation
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-02
updated: 2026-04-02
---

# Phase 03 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest 9.0.2` |
| **Config file** | [`backend/pytest.ini`](/C:/Users/D0n9/Desktop/LogicKG/backend/pytest.ini) |
| **Quick run command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_route_state_synthesizer.py tests\test_route_state_package.py tests\test_replay_io.py tests\test_historical_replay_compiler.py -q` |
| **Full suite command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_route_state_synthesizer.py tests\test_route_state_package.py tests\test_replay_io.py tests\test_historical_replay_compiler.py tests\test_route_state_pilot_cli.py -q` |
| **Estimated runtime** | ~4 seconds |

## Sampling Rate

- **After every task commit:** Run the quick command
- **After every plan wave:** Run the full suite command
- **Before Phase verify/closeout:** Full suite must be green
- **Max feedback latency:** 10 seconds

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | L3-01 | integration | quick | [`backend/tests/test_route_state_package.py`](/C:/Users/D0n9/Desktop/LogicKG/backend/tests/test_route_state_package.py) | COVERED |
| 03-01-02 | 01 | 1 | L3-03 | integration | full | [`backend/tests/test_replay_io.py`](/C:/Users/D0n9/Desktop/LogicKG/backend/tests/test_replay_io.py) | COVERED |
| 03-02-01 | 02 | 1 | L3-03 | integration | quick | [`backend/tests/test_route_state_package.py`](/C:/Users/D0n9/Desktop/LogicKG/backend/tests/test_route_state_package.py) | COVERED |
| 03-02-02 | 02 | 1 | L3-03 | integration | quick | [`backend/tests/test_replay_io.py`](/C:/Users/D0n9/Desktop/LogicKG/backend/tests/test_replay_io.py) | COVERED |
| 03-03-01 | 03 | 1 | L3-02 | unit | quick | [`backend/tests/test_route_state_synthesizer.py`](/C:/Users/D0n9/Desktop/LogicKG/backend/tests/test_route_state_synthesizer.py) | COVERED |
| 03-03-02 | 03 | 1 | L3-01, L3-03 | integration | quick | [`backend/tests/test_historical_replay_compiler.py`](/C:/Users/D0n9/Desktop/LogicKG/backend/tests/test_historical_replay_compiler.py) | COVERED |
| 03-03-03 | 03 | 1 | L3-02 | CLI smoke | full | [`backend/tests/test_route_state_pilot_cli.py`](/C:/Users/D0n9/Desktop/LogicKG/backend/tests/test_route_state_pilot_cli.py) | COVERED |

*Status: `COVERED` = automated command exists and passed green on 2026-04-02.*

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

## Runtime Artifact Audit

In addition to the automated pytest coverage above, the real jamming packaged-replay slice was audited from runtime artifacts:

- [`tmp/phase3_route_state_package/bundle/validation.json`](/C:/Users/D0n9/Desktop/LogicKG/tmp/phase3_route_state_package/bundle/validation.json)
  - `quality_tier = green`
  - `ready_for_replay = true`
  - `quality_flags = []`
- [`tmp/phase3_route_state_package/replay_with_package/replay_inspection.json`](/C:/Users/D0n9/Desktop/LogicKG/tmp/phase3_route_state_package/replay_with_package/replay_inspection.json)
  - `replay_quality_tier = green`
  - `route_comparison.selected_quality_tier = green`
  - `decision_prior_card.quality_tier = green`
  - `decision_episode.quality_tier = green`
  - `decision_episode.ready_for_training = true`

This bridges the gap between synthetic fixture tests and the real bounded corpus slice used to validate Phase 3.

## Manual-Only Verifications

All phase-critical behaviors have automated verification or runtime artifact audit coverage.

## Validation Sign-Off

- [x] All tasks have automated verify or artifact-audit coverage
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency < 10s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-02
