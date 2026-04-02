# Phase 4 Failure Taxonomy Baseline

## Taxonomy Contract

Phase 4 keeps the replay contract on the existing artifact boundary instead of adding a second diagnostics pipeline.

- `tmp/phase3_route_state_package/replay_with_package/replay_summary.json` now carries aggregate taxonomy counts:
  - `failure_record_count`
  - `failure_counts_by_layer`
  - `failure_counts_by_stage`
  - `failure_counts_by_blocking`
  - the existing top-level replay fields such as `replay_quality_tier`, `ready_for_pilot`, and `quality_flags`
- `tmp/phase3_route_state_package/replay_with_package/replay_inspection.json` now carries the full `failure_records` list.

Each `failure_records` item uses the same field contract:

- `failure_id`
- `layer`
- `stage`
- `severity`
- `blocking`
- `failure_code`
- `owner`
- `repair_target`
- `source_flags`
- `evidence_refs`

`stage` and `layer` are intentionally separate:

- `stage` records where the weakness surfaced in replay: `route_state`, `why_now_case`, `route_comparison`, `decision_prior_card`, or `decision_episode`
- `layer` records who should repair it: `packet`, `l1`, `l2`, or `l3_l4`

Non-blocking `L2` weaknesses remain inside the same `failure_records` queue with `blocking = false`. They are not moved to a separate “nice to have” list, because the goal of this baseline is to make the repair queue auditable on the same artifact boundary as the replay itself.

## Baseline Findings

Baseline sources:

- `tmp/phase3_route_state_package/bundle/validation.json`
- `tmp/phase3_route_state_package/replay_with_package/replay_summary.json`
- `tmp/phase3_route_state_package/replay_with_package/replay_inspection.json`

Current top-level replay status remains green:

- `replay_quality_tier = green`
- `ready_for_pilot = true`
- `quality_flags = []`

The new taxonomy still exposes four repair records on this green replay:

- total `failure_records`: `4`
- by `layer`:
  - `packet`: `1`
  - `l2`: `3`
- by `stage`:
  - `route_state`: `3`
  - `route_comparison`: `1`
- by blocking split:
  - `blocking`: `0`
  - `nonblocking`: `4`

Observed baseline records:

- `packet_role_coverage_missing`
  - owner: `route_packet`
  - surfaced stage: `route_state`
  - still nonblocking on this runtime subset because the replay already stayed green, but it records that the subset packet is missing the `limitation_or_critique` role
- `l2_comparator_sparse`
  - owner: `paper_logic_trace.derived_views`
  - surfaced stage: `route_comparison`
  - current baseline exposes `gap_moves:40`, which means comparator-bearing evidence is still thinner than the downstream replay would like even though the packaged comparison now succeeds
- `l2_expected_slot_missing`
  - owner: `paper_logic_trace.derived_views`
  - surfaced stage: `route_state`
  - current missing-slot signals include `metrics`, `comparators`, `conditions`, `limitation_types`, and `resource_mentions`
- `l2_relation_stitch_missing`
  - owner: `route_state_synthesizer`
  - surfaced stage: `route_state`
  - future-work and critique anchors exist in the baseline, but they are not yet stitched as tightly as they should be into replay-facing alternative-route evidence

The important baseline reading is that replay success is no longer blocked by missing `support / alternative / held_out` structure. The remaining queue is now dominated by nonblocking but repair-worthy `L2` density and stitching issues.

## L2 Repair Queue

The first bounded `L2` repair pass should stay focused on three targets.

### Comparator density

- Taxonomy record: `l2_comparator_sparse`
- Why it stays in queue even on a green replay:
  - the replay can choose an alternative route today, but the comparison evidence is still thinner than it should be for a reusable repair loop
- Downstream impact:
  - `RouteComparison` loses decisive evidence density
  - `DecisionPriorCard` inherits a weaker comparison rationale than the route package structure alone would suggest

### Expected-slot completeness

- Taxonomy record: `l2_expected_slot_missing`
- Why it stays in queue even on a green replay:
  - grounded moves are still arriving without enough `metrics`, `comparators`, `conditions`, `limitation_types`, or `resource_mentions` to make the replay contract maximally auditable
- Downstream impact:
  - `RouteState` coverage stays broader than it is precise
  - `WhyNow` and replay-facing summaries can only partially explain why one route looks stronger than another

### Relation stitching

- Taxonomy record: `l2_relation_stitch_missing`
- Why it stays in queue even on a green replay:
  - future-work / critique signals already exist in the slice, but their anchor-level evidence is not yet propagated cleanly enough into alternative-route evidence and comparison-facing output
- Downstream impact:
  - `RouteState` alternative-route candidates are less traceable than they should be
  - `RouteComparison` and later `DecisionPriorCard` reasoning are forced to lean on broader support evidence instead of tighter route-relationship anchors

These three queue items are the correct first-pass surgical targets because they are all `blocking = false`, they all remain `L2`-owned, and they directly influence the replay stages that still matter after Phase 3: `RouteState`, `WhyNow`, `RouteComparison`, and downstream prior selection.
