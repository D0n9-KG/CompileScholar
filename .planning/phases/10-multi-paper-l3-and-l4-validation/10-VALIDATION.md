---
phase: 10
slug: multi-paper-l3-and-l4-validation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-03
---

# Phase 10 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `backend/pytest.ini` if present, otherwise repo-default pytest discovery |
| **Quick run command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase10_multi_paper_validation.py tests\test_replay_io.py -q` |
| **Full suite command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase10_multi_paper_validation.py tests\test_phase10_multi_paper_validation_cli.py tests\test_bounded_packet_audit.py tests\test_route_state_package.py tests\test_historical_replay_compiler.py tests\test_decision_prior_builder.py tests\test_decision_episode_builder.py tests\test_decision_episode_export.py tests\test_decision_episode_export_cli.py tests\test_replay_io.py -q` |
| **Estimated runtime** | ~120 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase10_multi_paper_validation.py tests\test_replay_io.py -q`
- **After every plan wave:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase10_multi_paper_validation.py tests\test_phase10_multi_paper_validation_cli.py tests\test_bounded_packet_audit.py tests\test_route_state_package.py tests\test_historical_replay_compiler.py tests\test_decision_prior_builder.py tests\test_decision_episode_builder.py tests\test_decision_episode_export.py tests\test_decision_episode_export_cli.py tests\test_replay_io.py -q`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 1 | PACK-02 | unit | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase10_multi_paper_validation.py -k package -q` | No, Wave 0 | pending |
| 10-01-02 | 01 | 1 | PACK-02 | integration | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase10_multi_paper_validation.py -k replay -q` | No, Wave 0 | pending |
| 10-02-01 | 02 | 2 | AGGR-01 | integration | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase10_multi_paper_validation.py -k review -q` | No, Wave 0 | pending |
| 10-02-02 | 02 | 2 | PACK-02, AGGR-01 | unit | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase10_multi_paper_validation_cli.py -q` | No, Wave 0 | pending |
| 10-03-01 | 03 | 3 | PACK-02, AGGR-01 | smoke | `cd backend; .\.venv\Scripts\python.exe scripts\run_phase10_multi_paper_validation.py --help` | No, Wave 0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `backend/tests/test_phase10_multi_paper_validation.py` - stubs for `PACK-02` and `AGGR-01`
- [ ] `backend/tests/test_phase10_multi_paper_validation_cli.py` - CLI coverage for the Phase 10 runner
- [ ] `backend/scripts/run_phase10_multi_paper_validation.py` - executable backend entrypoint for the end-to-end packet workflow

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Final report compares computational-mechanics output against the jamming baseline without overclaiming readiness | PACK-02, AGGR-01 | The blocker narrative and historical interpretation are qualitative even when the source metrics are structured | Read the committed Phase 10 report and confirm it includes package validation, replay blockers, export posture, baseline comparison, and a stage-level blocker queue grounded in bundle outputs |
| Phase 10 stays inside the frozen Phase 9 packet boundary | PACK-02 | A reviewer must confirm no silent topic or membership drift happened during runtime adaptation | Compare the runtime package/replay inputs with `docs/replay/pilot_packets/phase9-route-packet.json` and `docs/replay/pilot_packets/phase9-assembly-manifest.json`; verify no extra packet members were introduced without an explicit blocker note |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
