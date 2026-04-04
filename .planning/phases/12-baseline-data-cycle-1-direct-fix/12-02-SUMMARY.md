---
phase: 12-baseline-data-cycle-1-direct-fix
plan: 02
subsystem: replay-reporting
tags: [markdown, python, pytest, phase12, verification, direct-fix]
requires:
  - phase: 12-baseline-data-cycle-1-direct-fix
    provides: Repaired Phase 10 runtime bridge and a dev-check rerun that clears the lead packet blockers
provides:
  - Fresh cycle-1 runtime outputs under tmp/phase12_direct_fix_cycle/cycle1
  - Committed Phase 12 direct-fix report tied to concrete blocker deltas on the bounded slice
  - Exact verification note that records commands, structured evidence, and the final bounded verdict
affects: [phase12-cycle-report, verification-evidence, next-cycle-baseline]
tech-stack:
  added: []
  patterns: [bundle-first reporting, verification quotes summary-json fields, bounded quality verdicts]
key-files:
  created:
    - docs/replay/reports/phase12-cycle1-direct-fix.md
    - .planning/phases/12-baseline-data-cycle-1-direct-fix/12-VERIFICATION.md
  modified: []
key-decisions:
  - "Treat tmp/phase12_direct_fix_cycle/cycle1/comparison_summary.json as the source of truth for the Phase 12 report and verification note."
  - "Judge this cycle on whole-slice quality movement, not only on whether the two lead packet blockers disappeared."
  - "Record the honest verdict as downstream blocker shift rather than overclaiming an end-to-end quality recovery."
patterns-established:
  - "Direct-fix cycle reports now link the inherited recommendation queue to fresh cycle output deltas and a bounded manual review verdict."
  - "Verification notes quote blocker queues and stage surfaces directly from the current cycle JSON bundle."
requirements-completed: [LOOPR-01, LOOPR-02]
completed: 2026-04-04
---

# Phase 12 Plan 02 Summary

**Ran the first real direct-fix cycle on the bounded computational-mechanics slice and committed a report plus verification note that explain exactly how quality moved**

## Accomplishments

- Ran the real Phase 10 CLI against the repaired bridge and wrote a fresh cycle root under `tmp/phase12_direct_fix_cycle/cycle1/`.
- Rewrote the generated markdown into a Phase 12 direct-fix report with the required sections: `Starting Evidence`, `Fixes Applied`, `Blocker Delta`, `Manual Output Review`, and `Verdict`.
- Added `.planning/phases/12-baseline-data-cycle-1-direct-fix/12-VERIFICATION.md` with the exact rerun command, the exact pytest command, the quoted cycle-1 summary fields, and the inherited Phase 11 recommendation evidence.
- Recorded the bounded final verdict honestly: packet construction improved, but the dominant blockers mostly shifted downstream into replay, prior-review, and export.

## Verification

- `cd backend; .\.venv\Scripts\python.exe scripts\run_phase10_multi_paper_validation.py --packet ..\docs\replay\pilot_packets\phase9-route-packet.json --assembly-manifest ..\docs\replay\pilot_packets\phase9-assembly-manifest.json --l1-snapshot-output ..\tmp\phase12_direct_fix_cycle\cycle1\shared\phase9-comp-mech-l1-snapshot.json --output-dir ..\tmp\phase12_direct_fix_cycle\cycle1 --baseline-replay-bundle ..\tmp\phase3_route_state_package\replay_with_package --baseline-export-bundle ..\tmp\phase6_decision_episode_audit_export --report-md ..\docs\replay\reports\phase12-cycle1-direct-fix.md`
  - Result: `comparison_summary.json` created with `package.current.quality_flags = ["yellow_route_state_present"]`
  - Result: `replay.current.failure_counts_by_stage = {"route_comparison": 1, "route_state": 2, "decision_prior_card": 1}`
  - Result: `prior_review.current.prior_candidate_count = 0`
  - Result: `export.current.ready_for_eval = true`
- `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase10_multi_paper_validation.py tests\test_phase10_multi_paper_validation_cli.py tests\test_route_state_package.py tests\test_bounded_packet_audit.py tests\test_historical_replay_compiler.py -q`
  - Result: `31 passed in 23.45s`

## Task Commits

Both plan tasks landed in one docs commit because the verification note depends on the exact report text and the same fresh cycle-1 outputs:

1. `49160d3f` (`docs`) - capture the Phase 12 cycle-1 report and verification evidence from the new output root

## Files Created/Modified

- `docs/replay/reports/phase12-cycle1-direct-fix.md` - direct-fix report with inherited evidence, applied fixes, blocker delta, manual output review, and bounded verdict
- `.planning/phases/12-baseline-data-cycle-1-direct-fix/12-VERIFICATION.md` - exact commands plus quoted structured evidence from the cycle-1 bundle
- `tmp/phase12_direct_fix_cycle/cycle1/comparison_summary.json` - fresh machine-readable comparison surface for the repaired cycle
- `tmp/phase12_direct_fix_cycle/cycle1/replay_bundle/replay_summary.json` - current replay stage quality summary
- `tmp/phase12_direct_fix_cycle/cycle1/export_bundle/export_summary.json` - current export posture for the bounded slice

## Issues Encountered

The cycle did not recover downstream prior-review or export quality. That is now an explicit result instead of a hidden ambiguity, and the report/verification pair call it out directly.

## Next Phase Readiness

- Future work can start from `tmp/phase12_direct_fix_cycle/cycle1/` instead of the older fallback-only Phase 10 / Phase 11 evidence chain.
- The repo now has a committed explanation of which blockers moved, which stayed, and why the net verdict is a downstream blocker shift rather than full quality recovery.

## Self-Check

PASSED

---
*Phase: 12-baseline-data-cycle-1-direct-fix*
*Completed: 2026-04-04*
