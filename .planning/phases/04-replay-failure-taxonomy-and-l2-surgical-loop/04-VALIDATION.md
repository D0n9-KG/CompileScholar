---
phase: 04
slug: replay-failure-taxonomy-and-l2-surgical-loop
status: ready
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-03
updated: 2026-04-03
---

# Phase 04 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest 9.0.2` |
| **Config file** | `backend/pytest.ini` |
| **Quick run command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_paper_logic_trace_derived_views.py tests\test_route_state_synthesizer.py tests\test_historical_replay_compiler.py tests\test_replay_io.py -q` |
| **Full suite command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_paper_logic_trace_derived_views.py tests\test_paper_logic_trace_compiler.py tests\test_route_state_synthesizer.py tests\test_historical_replay_compiler.py tests\test_replay_io.py tests\test_route_state_package.py tests\test_route_state_pilot_cli.py -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_paper_logic_trace_derived_views.py tests\test_route_state_synthesizer.py tests\test_historical_replay_compiler.py tests\test_replay_io.py -q`
- **After every plan wave:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_paper_logic_trace_derived_views.py tests\test_paper_logic_trace_compiler.py tests\test_route_state_synthesizer.py tests\test_historical_replay_compiler.py tests\test_replay_io.py tests\test_route_state_package.py tests\test_route_state_pilot_cli.py -q`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 20 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | L2-01, EVAL-02 | integration | quick | `backend/tests/test_historical_replay_compiler.py`, `backend/tests/test_replay_io.py` | READY |
| 04-01-02 | 01 | 1 | L2-01, EVAL-02 | report contract | quick | `backend/tests/test_replay_io.py` | READY |
| 04-02-01 | 02 | 2 | L2-02 | unit/integration | quick | `backend/tests/test_paper_logic_trace_derived_views.py` | READY |
| 04-02-02 | 02 | 2 | L2-01, L2-02 | integration | quick | `backend/tests/test_route_state_synthesizer.py`, `backend/tests/test_historical_replay_compiler.py` | READY |
| 04-03-01 | 03 | 3 | L2-01, EVAL-02 | CLI/integration | full | `backend/tests/test_replay_io.py`, `backend/tests/test_route_state_pilot_cli.py` | READY |
| 04-03-02 | 03 | 3 | EVAL-02 | report audit | full | `backend/tests/test_replay_io.py` | READY |

*Status: `READY` = test file already exists and is the intended extension point during execution.*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

---

## Runtime Artifact Audit

Phase 4 execution must keep comparing against the same validated runtime baseline from Phase 3:

- `tmp/phase3_route_state_package/bundle/validation.json`
- `tmp/phase3_route_state_package/replay_with_package/replay_summary.json`
- `tmp/phase3_route_state_package/replay_with_package/replay_inspection.json`

The Phase 4 "after" rerun should write a parallel output under a new gitignored runtime directory such as:

- `tmp/phase4_l2_surgical_loop/...`

This gives the phase one stable before/after comparison boundary without requiring new committed corpus assets first.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Failure ownership is understandable from the committed baseline report | L2-01, EVAL-02 | Human reviewers must confirm that each important symptom is assigned to the correct layer and repair target | Read the committed Phase 4 baseline report and verify it includes `layer`, `stage`, `failure_code`, `blocking`, and `repair_target` columns or bullets for every highlighted issue |
| Same-slice replay delta is truly comparable | EVAL-02 | Automated tests cannot confirm the shared local runtime packet inputs stayed aligned | Compare the "before" Phase 3 report/artifacts against the new Phase 4 rerun report and confirm the packet id, cutoff year, role composition, and route-state package inputs match |

---

## Validation Sign-Off

- [x] All tasks have automated verify paths or explicit runtime artifact audit coverage
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency < 20s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** ready for execution planning on 2026-04-03
