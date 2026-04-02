# Route-State Package Pilot - 2026-04-02

## Purpose

This note records the first end-to-end validation of reusable route-state packaging on top of the strengthened `L1 + replay` chain.

The question was:

- can we package real `support / alternative / held_out` route states as explicit artifacts, get the package itself to `green`, and use them to remove the structural replay gaps seen in the jamming pilot?

## Runtime Artifacts

Runtime manifest:

- `tmp/phase3_route_state_package/route_state_package_manifest.json`

Compiled package bundle:

- `tmp/phase3_route_state_package/bundle/`

Replay bundle using the package:

- `tmp/phase3_route_state_package/replay_with_package/`

New inspection artifacts:

- `tmp/phase3_route_state_package/bundle/validation.json`
- `tmp/phase3_route_state_package/replay_with_package/replay_inspection.json`

## Package Composition

Runtime package id:

- `phase1-jamming-route-state-package-v1`

Route-state groups:

- support route states: `2`
- alternative route states: `1`
- held-out route states: `1`

Package validation summary:

- `quality_tier`: `green`
- `ready_for_replay`: `true`
- validation flags: `[]`

Interpretation:

- the package is structurally complete enough for replay
- and every grouped `support / alternative / held_out` route state is now `green`

The package used four hand-curated subset packets derived from the same bounded jamming pilot topic:

- `jamming_support_core_packet_v1`
- `jamming_support_context_packet_v1`
- `jamming_alternative_frictional_packet_v1`
- `jamming_heldout_random_packing_packet_v1`

These subsets are still runtime-only artifacts, not yet committed audited packet manifests.

## Replay Delta

Compared against `tmp/phase1_replay_run/with_l1_snapshot_v2/replay_summary.json`:

- `support_route_state_count`
  - before: `0`
  - after: `2`
- `alternative_route_state_count`
  - before: `0`
  - after: `1`
- `held_out_route_state_count`
  - before: `0`
  - after: `1`
- `selected_comparison_case_id`
  - before: `null`
  - after: populated with a frictional-route comparison case
- replay `quality_flags`
  - before:
    - `support_cluster_too_small`
    - `no_alternative_route_states`
    - `held_out_routes_missing`
  - after: `[]`

Package-to-replay alignment is now also clean:

- `route_state_package_validation_quality_tier = green`
- `route_state_package_validation_flags = []`

## Downstream Quality Change

The most important downstream change is not only that the structural flags disappeared.

It is that the prior and episode layers now cross the threshold from provisional structure to replay-ready decision artifacts:

- `DecisionPriorCard.quality.quality_tier = green`
- `DecisionPriorCard.held_out_consistency.pass_rate = 1.0`
- `DecisionEpisode.quality.quality_tier = green`
- `DecisionEpisode.ready_for_training = true`

This means the project now has one real example where:

- a bounded packet exists
- `L1-lite` is historically grounded enough to supply benchmarks and resources
- route comparison is no longer empty
- prior support and held-out checks are no longer missing
- the package itself is also green rather than merely replay-adjacent

## Interpretation

This is the first proof that the missing `L3/L4` structure was not fundamentally blocked by the schema itself.

It was blocked by the absence of a reusable packaging boundary for multi-route inputs.

Once explicit route-state packaging exists, the replay compiler can consume the kinds of inputs it was designed for.

The final step that pushed the package from yellow to green was not adding yet another packaging layer.

It was calibrating `RouteState` quality so that a route can count as method-grounded when the method evidence is genuinely multi-paper at the landscape level, even if one exact dominant-method label is not repeated in every paper. That matches the actual goal of `L3`: aggregate route evidence across papers, not demand a single literal method string.

## Remaining Limitations

This runtime success should not be over-read.

The current jamming package still has important limits:

- the subset packets live in `tmp/`, not as committed long-term audited assets
- the support / alternative / held_out slices are still manually curated
- the package is topic-specific and does not yet prove cross-topic scalability
- the current `L1` bridge is still `paper_grounded_l1_lite`, so external historical environment coverage remains intentionally conservative

## Recommended Next Step

The next step is no longer “keep hardening the same yellow package”.

1. do a formal Phase 3 verify / closeout pass
2. decide whether to convert the runtime jamming subset packets into more stable audited assets
3. start Phase 4 replay failure taxonomy so the next loop focuses on real `L2` weaknesses rather than missing route-state roles
4. use the current green packaged replay as the baseline for later `L2` surgical comparisons
