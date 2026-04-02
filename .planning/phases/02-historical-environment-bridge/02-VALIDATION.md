---
phase: 02
slug: historical-environment-bridge
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-02
updated: 2026-04-02
---

# Phase 02 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest 9.0.2` |
| **Config file** | `backend/pytest.ini` |
| **Quick run command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_historical_environment_builder.py tests\test_route_state_synthesizer.py -q` |
| **Full suite command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_historical_environment_builder.py tests\test_research_logic_models.py tests\test_route_state_pilot_cli.py tests\test_route_state_synthesizer.py tests\test_replay_io.py tests\test_historical_replay_compiler.py tests\test_phase1_route_packet_manifest.py -q` |
| **Estimated runtime** | ~8 seconds |

## Sampling Rate

- **After every task commit:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_historical_environment_builder.py tests\test_route_state_synthesizer.py -q`
- **After every plan wave:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_historical_environment_builder.py tests\test_research_logic_models.py tests\test_route_state_pilot_cli.py tests\test_route_state_synthesizer.py tests\test_replay_io.py tests\test_historical_replay_compiler.py tests\test_phase1_route_packet_manifest.py -q`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | L1-01 | unit / contract | quick | `backend/tests/test_historical_environment_builder.py` | COVERED |
| 02-02-01 | 02 | 2 | L1-01 | CLI smoke | `cd backend; .\.venv\Scripts\python.exe scripts\run_l1_snapshot_pilot.py --help` | `backend/scripts/run_l1_snapshot_pilot.py` | COVERED |
| 02-02-02 | 02 | 2 | L1-01 | runtime artifact audit | full + report audit | `docs/replay/reports/l1-lite-snapshot-pilot-2026-04-02.md` | COVERED |
| 02-03-01 | 03 | 3 | L1-02 | integration | quick | `backend/tests/test_route_state_synthesizer.py` | COVERED |
| 02-03-02 | 03 | 3 | L1-02 | replay I/O + compiler | full | `backend/tests/test_route_state_pilot_cli.py`, `backend/tests/test_replay_io.py`, `backend/tests/test_historical_replay_compiler.py` | COVERED |

*Status: `COVERED` = automated command exists and passed green or the real runtime artifact/report was audited successfully on 2026-04-02.*

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

## Runtime Artifact Audit

In addition to the automated pytest coverage above, the real Phase 2 `L1-lite` artifacts were audited from committed reports and runtime outputs:

- `backend/runs/l1/phase1-pilot/historical_environment_snapshot.json`
  - `snapshot_id = jamming_transition_in_frictionless_sphere_packings_near_point_j_2010_l1_snapshot_v1`
  - `quality_tier = yellow`
  - `quality_flags = [paper_only_snapshot, preferred_benchmark_missing, toolchain_timeline_empty]`
- `docs/replay/reports/l1-lite-snapshot-pilot-2026-04-02.md`
  - records the first real paper-grounded snapshot build and why `yellow` is the honest tier
- `docs/replay/reports/l1-benchmark-fallback-replay-2026-04-02.md`
  - records replay improvements after explicit `L1` snapshot consumption and conservative benchmark fallback
  - documents `active_benchmarks` moving from empty to concrete values and `readiness_scores.overall` improving from `0.70` to `0.80`

This is sufficient Nyquist coverage for Phase 2 because the phase's core claim is that `L1-lite` became a real, replay-consumable historical layer whose improvements can be observed in downstream route compilation.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Snapshot remains conservative rather than fabricating a full historical registry | L1-01 | Tests can confirm fields and flags, but a reviewer must judge whether the snapshot honestly preserves uncertainty | Inspect `docs/replay/reports/l1-lite-snapshot-pilot-2026-04-02.md` and `backend/runs/l1/phase1-pilot/historical_environment_snapshot.json` to confirm missing benchmark/toolchain evidence stays explicit and quality remains `yellow` for the right reasons |
| Replay delta is historically grounded rather than a cosmetic score bump | L1-02 | Automated tests can confirm outputs changed, but a reviewer must judge whether the added benchmark/resource signals are actually paper-grounded | Read `docs/replay/reports/l1-benchmark-fallback-replay-2026-04-02.md` and confirm the new benchmarks, toolchains, and required resources are traced to cited packet papers rather than inferred from post-cutoff hindsight |

## Validation Sign-Off

- [x] All tasks have automated verify or artifact-audit coverage
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency < 10s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-02
