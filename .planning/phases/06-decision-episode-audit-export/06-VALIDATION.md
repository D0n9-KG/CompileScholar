---
phase: 06
slug: decision-episode-audit-export
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-02
updated: 2026-04-02
---

# Phase 06 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest 9.0.2` |
| **Config file** | `backend/pytest.ini` |
| **Quick run command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_episode_export.py tests\test_decision_episode_builder.py tests\test_historical_replay_compiler.py tests\test_replay_io.py -q` |
| **Full suite command** | `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_episode_export.py tests\test_decision_episode_export_cli.py tests\test_decision_episode_builder.py tests\test_historical_replay_compiler.py tests\test_replay_io.py -q` |
| **Estimated runtime** | ~10 seconds |

## Sampling Rate

- **After every task commit:** Run the quick command
- **After every plan wave:** Run the full suite command
- **Before Phase verify/closeout:** Full suite must be green
- **Max feedback latency:** 15 seconds

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | L4-02 | export assembly contract | quick | `backend/tests/test_decision_episode_export.py` | W0 CREATE |
| 06-01-02 | 01 | 1 | L4-02 | artifact I/O | quick | `backend/tests/test_replay_io.py` | COVERED |
| 06-01-03 | 01 | 1 | L4-02 | downstream regression | quick | `backend/tests/test_decision_episode_builder.py` and `backend/tests/test_historical_replay_compiler.py` | COVERED |
| 06-02-01 | 02 | 2 | L4-02 | CLI smoke | full | `backend/tests/test_decision_episode_export_cli.py` | W0 CREATE |
| 06-02-02 | 02 | 2 | L4-02 | runtime artifact audit | full + report | `docs/replay/reports/phase6-decision-episode-export-pilot.md` | W0 CREATE |

*Status: `COVERED` = automated command already exists in the repo. `W0 CREATE` = execution must add the missing file in the first applicable wave before Phase 6 can close out.*

## Wave 0 Requirements

Execution must create the missing export-specific verification surfaces early:

- `backend/tests/test_decision_episode_export.py` - contract tests for accepted-card carryover, empty-prior truthfulness, and leakage boundaries
- `backend/tests/test_decision_episode_export_cli.py` - CLI smoke coverage for bundle loading and export-bundle creation
- `docs/replay/reports/phase6-decision-episode-export-pilot.md` - committed pilot audit report for the first bounded export run

Existing framework, pytest config, and adjacent replay / episode suites already cover the rest of the phase.

## Runtime Artifact Audit

In addition to automated pytest coverage, Phase 6 closeout must inspect one real export bundle built from the current Phase 5 jamming artifacts.

Source artifacts to audit:

- `tmp/phase5_multi_route_prior_induction/review_bundle/bundle_manifest.json`
- `tmp/phase5_multi_route_prior_induction/review_bundle/candidate_review_summary.json`
- `tmp/phase5_multi_route_prior_induction/review_bundle/anti_pattern_candidates.json`
- `tmp/phase5_multi_route_prior_induction/replay_bundle/replay_summary.json`
- `tmp/phase5_multi_route_prior_induction/replay_bundle/replay_inspection.json`
- `tmp/phase5_multi_route_prior_induction/replay_bundle/outputs/decision_episode.json`
- `tmp/phase6_decision_episode_audit_export/` (new export output directory)

At closeout, the audit must confirm:

- exported `selected_prior_ids` matches the reviewed accepted-prior allowlist, even when that allowlist is empty;
- exported `selected_antipattern_ids` contains only reviewed accepted anti-pattern ids whose `failure_examples.route_state_ids` match the exported route;
- the export bundle exposes explicit `visible_input_refs`, `audit_only_refs`, and `label_eval_only_refs` or equivalent visibility buckets;
- exported `hindsight_outcome.input_visible` remains `false`;
- the report explicitly states that Phase 6 produced an audit-grade pilot rather than a generalized training dataset.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Visibility buckets are semantically correct rather than just present | L4-02 | Automated tests can confirm keys exist, but a reviewer must judge whether each ref truly belongs in visible, audit-only, or label/eval-only scope | Inspect `export_inspection.json` and confirm later-evidence refs and after-cutoff attachments never appear in visible-input buckets |
| Pilot report language stays conservative about maturity | L4-02 | Tests can check phrases, but a reviewer must judge whether the report overclaims generality or training readiness | Read `docs/replay/reports/phase6-decision-episode-export-pilot.md` and confirm it describes a bounded jamming audit pilot, not a production dataset pipeline |

## Validation Sign-Off

- [x] All tasks have automated verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency < 15s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-02
