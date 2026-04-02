# Phase 4 L2 Surgical Delta

## Baseline Alignment

Before artifacts:

- `tmp/phase3_route_state_package/bundle/validation.json`
- `tmp/phase3_route_state_package/replay_with_package/replay_summary.json`
- `tmp/phase3_route_state_package/replay_with_package/replay_inspection.json`

After artifacts:

- `tmp/phase4_l2_surgical_loop/bundle/validation.json`
- `tmp/phase4_l2_surgical_loop/replay_with_package/replay_summary.json`
- `tmp/phase4_l2_surgical_loop/replay_with_package/replay_inspection.json`

The rerun stayed aligned with the Phase 3 baseline on the fields that matter for a fair delta:

- `packet_id`: `jamming_frictionless_packings_2010_packet_01_runtime_subset` before and after
- `cutoff_year`: `2010` before and after
- route-state package id: `phase1-jamming-route-state-package-v1` before and after
- grouped route-state counts before and after:
  - support: `2`
  - alternative: `1`
  - held_out: `1`
- package validation stayed `green` with `ready_for_replay = true` and `quality_flags = []`

This means the Phase 4 rerun changed code and runtime output directories, but not the packet boundary or route-package composition.

## Failure Delta

Top-level replay failure totals stayed flat on the same bounded jamming slice:

- `failure_record_count`: `4 -> 4`
- counts by `layer`:
  - `packet`: `1 -> 1`
  - `l2`: `3 -> 3`
- counts by `stage`:
  - `route_state`: `3 -> 3`
  - `route_comparison`: `1 -> 1`
- counts by `blocking` split:
  - `blocking`: `0 -> 0`
  - `nonblocking`: `4 -> 4`

The live failure-code set was unchanged:

- `packet_role_coverage_missing`
- `l2_comparator_sparse`
- `l2_expected_slot_missing`
- `l2_relation_stitch_missing`

The L2 details that remained flat on the live slice:

- `l2_comparator_sparse` still reported `gap_moves:40`
- `l2_expected_slot_missing` still surfaced the same missing field families:
  - `metrics`
  - `comparators`
  - `conditions`
  - `limitation_types`
  - `resource_mentions`
- `l2_relation_stitch_missing` still remained nonblocking but present at `route_state`

Phase 4 therefore produced a no-regression rerun, not a same-slice reduction in live failure records.

## Quality Delta

Replay stayed stable at the top level:

- `replay_quality_tier`: `green -> green`
- `ready_for_pilot`: `true -> true`
- replay `quality_flags`: `[] -> []`

The primary replay-facing route state also stayed numerically identical on the metrics most relevant to this phase:

- dominant methods: `14 -> 14`
- measurement protocols: `46 -> 46`
- toolchains: `2 -> 2`
- known bottlenecks: `14 -> 14`
- alternative routes: `3 -> 3`
- supporting evidence ids: `170 -> 170`
- challenging evidence ids: `11 -> 11`

The rebuilt package entries were also unchanged at the quality-summary level:

- every `support / alternative / held_out` entry remained `green`
- no entry gained or lost alternative routes
- no entry gained or lost challenging-evidence coverage

The Phase 4 code path is therefore safe on this slice, but the jamming artifacts did not yet show a measurable comparator-density, expected-slot, or relation-stitching win.

## Remaining Ownership

The remaining owners are clearer after the rerun:

- `packet_role_coverage_missing`
  - owner: `route_packet`
  - layer: `packet`
  - next action: repair the runtime subset packet so the missing `limitation_or_critique` role is no longer normalized away as an accepted gap
- `l2_comparator_sparse`
  - owner: `paper_logic_trace.derived_views`
  - layer: `l2`
  - next action: inspect the real `1591` comparator-bearing result moves, because the fixture-level preservation added in Phase 4 did not reduce the live `gap_moves:40` count
- `l2_expected_slot_missing`
  - owner: `paper_logic_trace.derived_views`
  - layer: `l2`
  - next action: tighten which live jamming moves should count as expected-slot gaps or improve extraction for the specific `1591` move families still surfacing missing fields
- `l2_relation_stitch_missing`
  - owner: `route_state_synthesizer`
  - layer: `l2`
  - next action: inspect the real `814` future-work and limitation chain, because the current alternative-route evidence survived into the seed but did not change the replay-facing failure classification

What did not remain as a blocker:

- no `l1_snapshot_missing` or `l1_support_missing` failures surfaced
- route-state package structure stayed healthy
- replay stayed green and pilot-ready

The honest closeout is that Phase 4 succeeded at making the live delta auditable, but it did not yet deliver a same-slice L2 failure-count reduction. The next repair loop should stay focused on the real `1591` and `814` traces instead of broadening package or replay infrastructure work.
