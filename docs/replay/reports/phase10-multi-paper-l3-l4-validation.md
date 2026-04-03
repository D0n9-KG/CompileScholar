# Phase 10 Multi-Paper L3/L4 Validation

This report records the real bounded computational-mechanics run executed by `backend/scripts/run_phase10_multi_paper_validation.py` against the frozen Phase 9 packet boundary.

The machine-generated source of truth for this note is:

- `tmp/phase10_multi_paper_validation/baseline/comparison_summary.json`
- `tmp/phase10_multi_paper_validation/baseline/route_state_package/bundle_manifest.json`
- `tmp/phase10_multi_paper_validation/baseline/replay_bundle/replay_summary.json`
- `tmp/phase10_multi_paper_validation/baseline/prior_review_bundle/bundle_manifest.json`
- `tmp/phase10_multi_paper_validation/baseline/export_bundle/export_summary.json`

## Scope

- Packet id: `phase9_comp_mech_2021_packet_01`
- Topic scope: `data-driven constitutive and multiscale computational mechanics`
- Cutoff year: `2021`
- Baseline replay bundle: `tmp/phase3_route_state_package/replay_with_package`
- Baseline export bundle: `tmp/phase6_decision_episode_audit_export`

## Structural Handoff Versus Runtime Readiness

The Phase 9 packet handoff is still the stable part of the workflow. `docs/replay/reports/phase9-bounded-packet-audit.md`, derived from `tmp/phase9_bounded_packet_audit/baseline/`, recorded the packet and assembly mapping as structurally aligned and ready for Phase 10 handoff.

This Phase 10 run does not overturn that structural result. It shows that the bounded packet can flow through the finished Phase 10 tooling, but the downstream multi-paper package, replay, prior-review, and export surfaces remain below the jamming baseline.

## Runtime Summary

### Package Validation

- Current quality tier: `red`
- Baseline quality tier: `green`
- Ready for replay: `false` vs baseline `true`
- Current quality flags: `support_cluster_too_small`, `alternative_scope_not_distinct`, `yellow_route_state_present`
- Current role counts: `support=1`, `alternative=1`, `held_out=1`
- Baseline role counts: `support=2`, `alternative=1`, `held_out=1`

Interpretation: the structural packet handoff succeeds, but the compiled route-state package regresses relative to the jamming package because the support cluster compresses to one support route state and the alternative route remains insufficiently distinct.

### Replay

- Current quality tier: `red`
- Baseline quality tier: `green`
- Ready for pilot: `false` vs baseline `true`
- Current quality flags: `support_cluster_too_small`, `reviewer_missing`
- Failure record count: `5` vs baseline `4`
- Failure counts by stage: `route_state=2`, `route_comparison=1`, `decision_prior_card=2`
- Failure counts by layer: `l2=3`, `l3_l4=2`
- Failure counts by blocking: `blocking=2`, `nonblocking=3`

Interpretation: replay still reaches comparison-case generation, but it adds two `decision_prior_card` failures that were absent in the jamming baseline and stays blocked on support density plus missing reviewer coverage.

### Prior Review

- Cluster count: `1`
- Prior candidate count: `0`
- Anti-pattern candidate count: `0`
- Accepted prior ids: `[]`
- Accepted anti-pattern ids: `[]`
- Baseline accepted anti-pattern count: `5`

Interpretation: prior induction does not produce a reviewable multi-paper surface on this bounded packet. That keeps `accepted_prior_ids` empty and also drops the anti-pattern carryover that the jamming export preserved.

### Export

- Current quality tier: `yellow`
- Baseline quality tier: `yellow`
- Ready for training: `false` vs baseline `false`
- Ready for eval: `true` vs baseline `true`
- Current quality flags: `weak_prior_support`
- Selected prior count: `0` vs baseline `0`
- Selected anti-pattern count: `0` vs baseline `5`
- Visibility buckets: `visible_input_refs=28`, `audit_only_refs=26`, `label_eval_only_refs=0`

Interpretation: export posture stays conservative rather than collapsing. It remains audit-grade and eval-eligible, but it no longer carries reviewed anti-pattern evidence forward because the Phase 10 prior-review bundle accepted nothing.

## Baseline Delta Against The Jamming Slice

`comparison_summary.json` records the strongest regressions relative to the machine-generated jamming baseline:

- Package quality regressed from `green` to `red`.
- Replay quality regressed from `green` to `red`.
- Replay added `reviewer_missing` and preserved `support_cluster_too_small`.
- Replay failure stages shifted from `route_state=3` and `route_comparison=1` in the baseline to `route_state=2`, `route_comparison=1`, and `decision_prior_card=2` here.
- Prior review lost all anti-pattern carryover: current accepted anti-pattern count `0` vs baseline `5`.
- Export quality did not improve past `yellow`, and selected anti-pattern count fell from `5` to `0`.

The bounded computational-mechanics slice therefore clears the "real packet through real tooling" bar, but it does not clear the "baseline-quality multi-paper aggregation" bar.

## Blocker Queue

### Package Validation

- `support_cluster_too_small` is new relative to the baseline.
- `alternative_scope_not_distinct` is new relative to the baseline.
- `yellow_route_state_present` is new relative to the baseline.

### Replay

- `support_cluster_too_small` is new relative to the baseline.
- `reviewer_missing` is new relative to the baseline.
- `failure_stage:route_comparison` stays at `1`, unchanged from the baseline.
- `failure_stage:route_state` drops from `3` to `2`, but the stage is still active.
- `failure_stage:decision_prior_card` rises from `0` to `2`.

### Prior Induction

- `no_prior_candidates` remains true.
- `accepted_prior_ids_empty` remains true.

### Export

- `weak_prior_support` is carried forward from the baseline.
- `not_ready_for_training` remains unchanged.

## Verdict

Phase 10 now has one real bounded validation slice rooted in the committed Phase 9 packet and rendered from machine-generated bundle summaries. The structural handoff from Phase 9 is intact, but the current computational-mechanics run is not green-ready for multi-paper replay or `L4` carryover.

The strongest remaining blockers are:

- support-cluster depth at package and replay time
- alternative-route distinctness inside the bounded packet
- missing reviewer/prior surfaces that prevent anti-pattern carryover into export

That means this run should be used as optimization evidence, not as a readiness claim.
