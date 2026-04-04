# Phase 12 Verification

This note records the exact commands executed for the Phase 12 cycle-1 rerun and cites the fresh cycle-1 outputs as the source of truth for the final quality verdict.

## Commands Executed

### Phase 12 cycle-1 rerun

Command required by the plan:

```powershell
cd backend; .\.venv\Scripts\python.exe scripts\run_phase10_multi_paper_validation.py --packet ..\docs\replay\pilot_packets\phase9-route-packet.json --assembly-manifest ..\docs\replay\pilot_packets\phase9-assembly-manifest.json --l1-snapshot-output ..\tmp\phase12_direct_fix_cycle\cycle1\shared\phase9-comp-mech-l1-snapshot.json --output-dir ..\tmp\phase12_direct_fix_cycle\cycle1 --baseline-replay-bundle ..\tmp\phase3_route_state_package\replay_with_package --baseline-export-bundle ..\tmp\phase6_decision_episode_audit_export --report-md ..\docs\replay\reports\phase12-cycle1-direct-fix.md
```

Execution result:

- Exit code: `0`
- Fresh cycle root written under `tmp/phase12_direct_fix_cycle/cycle1/`
- Report written: `docs/replay/reports/phase12-cycle1-direct-fix.md`
- Required outputs confirmed:
  - `route_state_package/bundle_manifest.json`
  - `replay_bundle/replay_summary.json`
  - `prior_review_bundle/bundle_manifest.json`
  - `export_bundle/export_summary.json`
  - `comparison_summary.json`

### Verification pytest command

Command required by the plan:

```powershell
cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_phase10_multi_paper_validation.py tests\test_phase10_multi_paper_validation_cli.py tests\test_route_state_package.py tests\test_bounded_packet_audit.py tests\test_historical_replay_compiler.py -q
```

Execution result:

- Exit code: `0`
- Result: `31 passed in 23.45s`

## Structured Evidence

All current-cycle values below come from `tmp/phase12_direct_fix_cycle/cycle1/comparison_summary.json`.

### Inherited recommendation that this cycle tested

The inherited Phase 11 summary at `tmp/phase11_iteration_prioritization/baseline/outputs/prioritization_summary.json` explicitly selected packet construction as the lead repair surface:

- `primary_recommendation_id = "packet_construction"`
- `recommendations[0].why_now = "New package-validation blockers are the first fresh regressions while replay L2 delta stays flat."`
- `recommendations[0].evidence = ["package blockers: support_cluster_too_small, alternative_scope_not_distinct, yellow_route_state_present", "replay l2 delta: 0", "phase10 recommendation: packet_construction"]`

This means Phase 12 should count as successful only if it moves the packet-construction blockers on the same bounded computational-mechanics slice without pretending the downstream replay/prior/export problems disappeared.

### Current package surface

- `package.current.quality_flags = ["yellow_route_state_present"]`
- `package.current.role_counts = {"primary": 0, "support": 3, "alternative": 1, "held_out": 1}`
- `package.current.ready_for_replay = true`
- `notes.route_state_package_validation.indistinct_alternative_entry_ids = []`
- `blocker_queue.package_validation[*].code = ["yellow_route_state_present"]`

Interpretation: the fresh cycle removes `support_cluster_too_small` and `alternative_scope_not_distinct`, but it does not hide the remaining yellow package signal.

### Current replay, prior-review, and export surfaces

- `replay.current.failure_counts_by_stage = {"route_comparison": 1, "route_state": 2, "decision_prior_card": 1}`
- `replay.current.quality_flags = ["reviewer_missing"]`
- `prior_review.current.prior_candidate_count = 0`
- `prior_review.current.accepted_prior_ids = []`
- `export.current.ready_for_eval = true`
- `export.current.ready_for_training = false`
- `export.current.quality_flags = ["weak_prior_support"]`

Blocker queue evidence:

- `blocker_queue.replay[*].code = ["reviewer_missing", "failure_stage:route_comparison", "failure_stage:route_state", "failure_stage:decision_prior_card"]`
- `blocker_queue.prior_induction[*].code = ["no_prior_candidates", "accepted_prior_ids_empty"]`
- `blocker_queue.export[*].code = ["weak_prior_support", "not_ready_for_training"]`

Interpretation: the bounded slice is now past the thin-support and indistinct-alternative package blockers, but replay is still red, prior-review still produces no reusable candidates, and export remains eval-safe rather than training-ready.

### Delta against the inherited Phase 10 / Phase 11 evidence chain

Relative to the inherited evidence that triggered Phase 12:

- `support_cluster_too_small`: resolved on the fresh cycle-1 package and replay surfaces
- `alternative_scope_not_distinct`: resolved on the fresh cycle-1 package surface
- `yellow_route_state_present`: still present
- `reviewer_missing`: still present
- `decision_prior_card`: improved from the inherited `2` replay failures to `1`, but not resolved

The current cycle therefore moved the lead packet blockers but did not produce a clean end-to-end bounded-slice rerun.

## Verification Verdict

Phase 12 cycle 1 improved the bounded packet/runtime bridge but mostly shifted the dominant blockers downstream rather than recovering overall quality on this slice.

Why:

- the exact blockers Phase 11 pointed at first, `support_cluster_too_small` and `alternative_scope_not_distinct`, are gone from the fresh package surface
- the package now reaches `ready_for_replay = true` with `support = 3`
- replay still ends `red` because `reviewer_missing` remains active and `decision_prior_card` still fails once
- prior-review still yields `prior_candidate_count = 0`
- export is still `ready_for_eval = true` but `ready_for_training = false` with `weak_prior_support`

Final bounded verdict: **quality shifted downstream after the packet repair**. That is real progress, and it gives later cycles a better starting surface than the inherited Phase 10 / Phase 11 evidence chain, but it does not justify generalized readiness beyond this computational-mechanics packet.
