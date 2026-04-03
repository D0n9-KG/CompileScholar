---
phase: 09
slug: bounded-packet-construction-from-corpus
status: ready
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-03
---

# Phase 09 - Validation Strategy

> Per-phase validation contract for bounded packet assembly before any Phase 10 replay work.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest` |
| **Config file** | repo uses direct `python -m pytest` runs; no dedicated backend config is required for this phase |
| **Quick run command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_bounded_packet_audit.py tests\test_phase9_route_packet_manifest.py -q` |
| **Full suite command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_bounded_packet_audit.py tests\test_phase9_route_packet_manifest.py tests\test_replay_io.py tests\test_route_state_package.py tests\test_phase1_route_packet_manifest.py -q` |
| **CLI smoke command** | `cd backend; .\.venv\Scripts\python.exe scripts\run_bounded_packet_audit.py --help` |
| **Estimated runtime** | ~10 seconds quick / ~25 seconds full |

---

## Sampling Rate

- **After every task commit:** Run `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_bounded_packet_audit.py tests\test_phase9_route_packet_manifest.py -q`
- **After every plan wave:** Run the full suite plus the CLI smoke command
- **Before `$gsd-verify-work`:** Full suite must be green and the real Phase 9 audit report must match the runtime audit summary
- **Max feedback latency:** 25 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | PACK-01 | contract | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_bounded_packet_audit.py -q` | `backend/tests/test_bounded_packet_audit.py` | pending |
| 09-01-02 | 01 | 1 | PACK-01 | CLI smoke | `cd backend; .\.venv\Scripts\python.exe scripts\run_bounded_packet_audit.py --help` | `backend/scripts/run_bounded_packet_audit.py` | pending |
| 09-02-01 | 02 | 2 | PACK-01 | manifest contract | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase9_route_packet_manifest.py -q` | `backend/tests/test_phase9_route_packet_manifest.py` | pending |
| 09-02-02 | 02 | 2 | PACK-01 | integration | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_bounded_packet_audit.py tests\test_phase9_route_packet_manifest.py tests\test_replay_io.py tests\test_route_state_package.py -q` | `docs/replay/reports/phase9-bounded-packet-audit.md` | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

Existing infrastructure is sufficient. Phase 9 reuses the established backend test runner and only adds:

- `backend/tests/test_bounded_packet_audit.py` for companion-manifest validation and audit flags
- `backend/tests/test_phase9_route_packet_manifest.py` for the committed Phase 9 packet artifact contract
- `backend/scripts/run_bounded_packet_audit.py` as a reproducible operator entrypoint for packet auditing

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Topic boundary is defensible rather than merely keyword-coherent | PACK-01 | Automated checks can validate shape and counts, but not whether the selected papers really belong in one bounded topic | Read `docs/replay/pilot_packets/phase9-selection-notes.md` and confirm the included and excluded papers tell one coherent data-driven / multiscale computational-mechanics story |
| Assembly gaps are actionable for Phase 10 | PACK-01 | Tests can confirm fields exist, but a reviewer must judge whether the audit helps the next phase | Compare the runtime audit summary with `docs/replay/reports/phase9-bounded-packet-audit.md` and confirm the remaining blockers are explicit, prioritized, and not softened into replay claims |

---

## Runtime Artifact Audit

Phase 9 execution should produce a runtime audit bundle under `tmp/phase9_bounded_packet_audit/` containing at minimum:

- `audit_summary.json`
- `audit_inspection.json`

The committed report should be derived from those runtime outputs and must repeat the packet boundary, role mapping counts, exclusion reasons, and unresolved assembly gaps.

---

## Validation Sign-Off

- [x] All tasks have automated verify commands or explicit manual audit coverage
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency < 25s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-03
