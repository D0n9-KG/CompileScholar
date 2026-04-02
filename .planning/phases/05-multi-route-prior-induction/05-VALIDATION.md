---
phase: 05
slug: multi-route-prior-induction
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-02
updated: 2026-04-02
---

# Phase 05 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest 9.0.2` |
| **Config file** | `backend/pytest.ini` |
| **Quick run command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_prior_builder.py tests\test_research_logic_models.py tests\test_route_state_package.py tests\test_historical_replay_compiler.py tests\test_replay_io.py -q` |
| **Full suite command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_prior_builder.py tests\test_research_logic_models.py tests\test_route_state_package.py tests\test_historical_replay_compiler.py tests\test_replay_io.py tests\test_decision_episode_builder.py tests\test_route_state_pilot_cli.py -q` |
| **Estimated runtime** | ~8 seconds |

## Sampling Rate

- **After every task commit:** Run the quick command
- **After every plan wave:** Run the full suite command
- **Before Phase verify/closeout:** Full suite must be green
- **Max feedback latency:** 15 seconds

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | L4-01 | unit | quick | `backend/tests/test_decision_prior_builder.py` | COVERED |
| 05-01-02 | 01 | 1 | L4-01 | contract | quick | `backend/tests/test_research_logic_models.py` | COVERED |
| 05-01-03 | 01 | 1 | L4-01 | integration | quick | `backend/tests/test_route_state_package.py` | COVERED |
| 05-02-01 | 02 | 2 | L4-01 | integration | quick | `backend/tests/test_historical_replay_compiler.py` | COVERED |
| 05-02-02 | 02 | 2 | L4-01 | downstream | full | `backend/tests/test_decision_episode_builder.py` | COVERED |
| 05-02-03 | 02 | 2 | L4-01 | artifact I/O + CLI smoke | full | `backend/tests/test_replay_io.py` and `backend/tests/test_route_state_pilot_cli.py` | COVERED |

*Status: `COVERED` = automated command exists and must remain green during Phase 5 execution.*

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

## Runtime Artifact Audit

In addition to the automated pytest coverage above, Phase 5 execution must inspect one real candidate-induction pilot output directory, reusing an existing route-state package slice where possible.

At closeout, the audit must confirm:

- at least one prior candidate and one anti-pattern candidate came from multiple `RouteState` objects,
- each reviewed candidate exposes `supporting_route_state_ids`, `counterexample_ids`, `held_out_consistency`, `review`, and `quality`,
- any accepted cards can still be traced back to `support`, `alternative`, and `held_out` route ids from the package boundary,
- the closeout report explicitly states that the result is an auditable pilot rather than a training-ready export dataset.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Candidate acceptance semantics stay specific rather than paper-specific | L4-01 | A reviewer must judge whether `applies_when` / `does_not_apply_when` are meaningful and not just restatements of one paper | Inspect the Phase 5 review report and candidate JSON, reject any accepted card that cannot explain both where it applies and where it does not |
| Anti-pattern wording is auditable and actionable | L4-01 | Automated tests can confirm fields exist, but a reviewer must judge whether the warning pattern and checklist are useful | Inspect one anti-pattern candidate and confirm `warning_signal_pattern`, `failure_examples`, and `corrective_checklist` tell a reviewer what to avoid and what to check next |

## Validation Sign-Off

- [x] All tasks have automated verify or artifact-audit coverage
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency < 15s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-02
