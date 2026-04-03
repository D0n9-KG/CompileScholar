# Phase 10 Verification

This note records the exact commands executed for Plan `10-03` and the runtime evidence that should drive the next optimization cycle.

## Commands Executed

### Phase 10 bounded packet run

Command required by the plan:

```powershell
cd backend; .\.venv\Scripts\python.exe scripts\run_phase10_multi_paper_validation.py --packet ..\docs\replay\pilot_packets\phase9-route-packet.json --assembly-manifest ..\docs\replay\pilot_packets\phase9-assembly-manifest.json --l1-snapshot-output ..\tmp\phase10_multi_paper_validation\shared\phase9-comp-mech-l1-snapshot.json --output-dir ..\tmp\phase10_multi_paper_validation\baseline --baseline-replay-bundle ..\tmp\phase3_route_state_package\replay_with_package --baseline-export-bundle ..\tmp\phase6_decision_episode_audit_export --report-md ..\docs\replay\reports\phase10-multi-paper-l3-l4-validation.md
```

Execution result:

- Exit code: `0`
- Report written: `docs/replay/reports/phase10-multi-paper-l3-l4-validation.md`
- Baseline bundle roots written under `tmp/phase10_multi_paper_validation/baseline/`
- Required outputs confirmed:
  - `route_state_package/bundle_manifest.json`
  - `replay_bundle/replay_summary.json`
  - `prior_review_bundle/bundle_manifest.json`
  - `export_bundle/export_summary.json`
  - `comparison_summary.json`

### Full backend verification

Command required by the plan:

```powershell
cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase10_multi_paper_validation.py tests\test_phase10_multi_paper_validation_cli.py tests\test_bounded_packet_audit.py tests\test_route_state_package.py tests\test_historical_replay_compiler.py tests\test_decision_prior_builder.py tests\test_decision_episode_builder.py tests\test_decision_episode_export.py tests\test_decision_episode_export_cli.py tests\test_replay_io.py -q
```

Execution result:

- Exit code: `0`
- Result: `59 passed in 29.10s`

## Observed Runtime Evidence

All values below come from `tmp/phase10_multi_paper_validation/baseline/comparison_summary.json` unless noted otherwise.

### Structural handoff is intact

- The frozen Phase 9 packet remains the bounded source input.
- `docs/replay/reports/phase9-bounded-packet-audit.md` already recorded the packet and assembly mapping as structurally aligned and ready for Phase 10 handoff.
- This Plan 10-03 run therefore validates the finished Phase 10 workflow on the real packet without changing the packet boundary.

### Package validation stays red

- `package.current.quality_tier = "red"`
- `package.current.ready_for_replay = false`
- `package.current.quality_flags = ["support_cluster_too_small", "alternative_scope_not_distinct", "yellow_route_state_present"]`
- `package.current.role_counts.support = 1`
- `package.baseline.role_counts.support = 2`
- `blocker_queue.package_validation` adds three new blockers relative to the jamming baseline:
  - `support_cluster_too_small`
  - `alternative_scope_not_distinct`
  - `yellow_route_state_present`

Interpretation: the first failing surface is package composition quality, not CLI stability.

### Replay inherits the package weakness and adds review gaps

- `replay.current.quality_tier = "red"`
- `replay.current.ready_for_pilot = false`
- `replay.current.quality_flags = ["support_cluster_too_small", "reviewer_missing"]`
- `replay.current.failure_counts_by_stage = {"route_comparison": 1, "route_state": 2, "decision_prior_card": 2}`
- `replay.current.failure_counts_by_layer = {"l2": 3, "l3_l4": 2}`
- `replay.delta.failure_counts_by_layer_delta = {"l2": 0, "l3_l4": 2, "packet": -1}`

Interpretation: the run is not getting worse at `L2` relative to the jamming baseline. The new regression is above `L2`, with two additional `decision_prior_card` failures and the same support-density weakness propagating into replay.

### Prior review and export stay conservative

- `prior_review.current.prior_candidate_count = 0`
- `prior_review.current.anti_pattern_candidate_count = 0`
- `prior_review.current.accepted_prior_ids = []`
- `prior_review.current.accepted_anti_pattern_ids = []`
- `blocker_queue.prior_induction` contains `no_prior_candidates` and `accepted_prior_ids_empty`
- `export.current.quality_tier = "yellow"`
- `export.current.ready_for_eval = true`
- `export.current.ready_for_training = false`
- `export.current.quality_flags = ["weak_prior_support"]`
- `export.current.selected_antipattern_count = 0`
- `export.baseline.selected_antipattern_count = 5`

Interpretation: export posture is still audit-grade and eval-safe, but the bounded computational-mechanics slice loses the reviewed anti-pattern carryover that the jamming baseline preserved.

## Verification Verdict

The strongest next-cycle blockers point to `packet construction`, not `L2`.

Why:

- The Phase 9 packet boundary is structurally valid, so this is not a "rebuild the whole packet audit" problem.
- `replay.delta.failure_counts_by_layer_delta.l2 = 0`, so the new regression is not primarily single-paper `L2`.
- The first new blockers appear in `blocker_queue.package_validation`, where support density and alternative-route distinctness fail before export logic has a chance to recover.
- `L4` aggregation remains a live secondary problem because `decision_prior_card` failures rise from `0` to `2`, but those failures sit downstream of the thin support cluster and empty prior/anti-pattern candidate surface.

Recommended next interpretation for Phase 11:

- prioritize bounded packet construction work that deepens support coverage and sharpens alternative-route distinctness inside the computational-mechanics slice
- only then re-run `L4` aggregation tuning against the improved packet to see whether prior candidates and anti-pattern carryover recover

## Boundaries

- This verification applies only to one bounded computational-mechanics slice rooted in the committed Phase 9 packet.
- It does not justify generalized multi-paper readiness.
- The committed report and this note are both grounded in the runtime bundle outputs under `tmp/phase10_multi_paper_validation/baseline/`.
