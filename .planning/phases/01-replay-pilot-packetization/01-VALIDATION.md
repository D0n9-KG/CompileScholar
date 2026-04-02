---
phase: 01
slug: replay-pilot-packetization
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-02
updated: 2026-04-02
---

# Phase 01 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest 9.0.2` |
| **Config file** | `backend/pytest.ini` |
| **Quick run command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_replay_io.py tests\test_phase1_route_packet_manifest.py -q` |
| **Full suite command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_replay_io.py tests\test_historical_replay_compiler.py tests\test_phase1_route_packet_manifest.py -q` |
| **Estimated runtime** | ~5 seconds |

## Sampling Rate

- **After every task commit:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_replay_io.py tests\test_phase1_route_packet_manifest.py -q`
- **After every plan wave:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_replay_io.py tests\test_historical_replay_compiler.py tests\test_phase1_route_packet_manifest.py -q`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | PKT-02 | contract | quick | `backend/tests/test_replay_io.py` | COVERED |
| 01-01-02 | 01 | 1 | PKT-02 | CLI smoke | `cd backend; .\.venv\Scripts\python.exe scripts\run_replay_pilot.py --help` | `backend/scripts/run_replay_pilot.py` | COVERED |
| 01-01-03 | 01 | 1 | PKT-02, EVAL-01 | integration | full | `backend/tests/test_replay_io.py`, `backend/tests/test_historical_replay_compiler.py` | COVERED |
| 01-02-01 | 02 | 2 | PKT-01 | contract | quick | `backend/tests/test_phase1_route_packet_manifest.py` | COVERED |
| 01-02-02 | 02 | 2 | PKT-01 | schema audit | quick | `docs/replay/pilot_packets/phase1-route-packet.json` | COVERED |
| 01-03-01 | 03 | 3 | EVAL-01 | runtime artifact audit | full + report audit | `docs/replay/reports/phase1-pilot-review.md` | COVERED |
| 01-03-02 | 03 | 3 | PKT-02, EVAL-01 | report contract | full + report audit | `docs/replay/reports/phase1-pilot-failure-inventory.md` | COVERED |

*Status: `COVERED` = automated command exists and passed green or the real runtime artifact/report was audited successfully on 2026-04-02.*

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

## Runtime Artifact Audit

In addition to the automated pytest coverage above, the real Phase 1 bounded replay pilot was audited from committed reports and runtime outputs:

- `backend/runs/replay/phase1-pilot/replay_summary.json`
  - `ready_for_pilot = true`
  - `quality_flags = [support_cluster_too_small, no_alternative_route_states, held_out_routes_missing]`
  - `selected_comparison_case_id = null`
- `docs/replay/reports/phase1-pilot-review.md`
  - records the actual pilot bundle path, packet boundary, runtime subset choice, and compile outcomes
- `docs/replay/reports/phase1-pilot-failure-inventory.md`
  - maps issues across packet / `L1` / `L2` / `L3` / process buckets with explicit next owners

This is sufficient Nyquist coverage for Phase 1 because the phase's core claim is not that replay was already green, but that the packetized replay workflow was real, inspectable, and honest about quality degradation.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Runtime subset packet still respects the committed packet boundary | PKT-01, EVAL-01 | Automated tests cannot judge whether omitted packet items were dropped only for local trace availability rather than scope drift | Compare `docs/replay/pilot_packets/phase1-selection-notes.md` against `docs/replay/reports/phase1-pilot-review.md` and confirm the runtime subset keeps the same topic scope, cutoff year, and rationale while explicitly naming the unresolved trace ids |
| Failure inventory remains actionably layer-aware | PKT-02, EVAL-01 | Tests can confirm files exist, but a reviewer must judge whether the taxonomy is useful for future planning | Read `docs/replay/reports/phase1-pilot-failure-inventory.md` and confirm each meaningful issue is tagged to a layer, marked blocking vs degrading, and assigned to a future owner phase |

## Validation Sign-Off

- [x] All tasks have automated verify or artifact-audit coverage
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency < 10s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-02
