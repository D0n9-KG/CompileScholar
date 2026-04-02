# L1 Benchmark Fallback Replay Delta - 2026-04-02

## Purpose

This note records the first end-to-end improvement after wiring real `L1-lite` snapshots into replay consumption and adding a conservative benchmark fallback pass inside the `HistoricalEnvironmentBuilder`.

The key question was:

- can a paper-grounded `L1-lite` recover historically meaningful benchmark / reference anchors even when `l1_bridge_hints.benchmark_candidates` is empty?

## What Changed

Two connected improvements were added:

- `RouteStateSynthesizer` now consumes an explicit `HistoricalEnvironmentSnapshot` instead of only inheriting placeholder `L1` refs from the packet
- `HistoricalEnvironmentBuilder` now performs a conservative fallback search for packet-preferred benchmarks using:
  - explicit `benchmark_candidates`
  - protocol / resource / toolchain hint rows
  - move summaries and mention surfaces / normalized values

The fallback does **not** invent facts.

It only promotes a preferred benchmark when paper-grounded evidence inside the trace actually mentions an alias-consistent signal.

## Runtime Inputs

- packet: `tmp/phase1_runtime_subset_packet.json`
- traces:
  - `tmp/phase1_packet_traces/1591/paper_logic_trace.json`
  - `tmp/phase1_packet_traces/1742/paper_logic_trace.json`
  - `tmp/phase1_packet_traces/1746/paper_logic_trace.json`
  - `tmp/phase1_packet_traces/777/paper_logic_trace.json`
  - `tmp/phase1_packet_traces/814/paper_logic_trace.json`

Snapshot rebuild output:

- `tmp/phase1_replay_run/historical_environment_snapshot_v2.json`

Replay bundle using rebuilt snapshot:

- `tmp/phase1_replay_run/with_l1_snapshot_v2/`

## L1 Delta

Before fallback:

- `benchmark_timeline`
  - `maximally random jammed state`: `missing`
  - `packing fraction phi_c`: `missing`
- snapshot `quality_flags`
  - `paper_only_snapshot`
  - `preferred_benchmark_missing`
  - `toolchain_timeline_empty`

After fallback:

- `benchmark_timeline`
  - `maximally random jammed state`: `available`
  - `packing fraction phi c`: `available`
- snapshot `quality_flags`
  - `paper_only_snapshot`
  - `toolchain_timeline_empty`

Recovered paper support:

- `maximally random jammed state`
  - papers: `1742`, `1591`, `814`
- `packing fraction phi c`
  - papers: `1746`, `1591`, `814`

## Replay Delta

Compared against the original replay bundle in `backend/runs/replay/phase1-pilot/`:

- `active_benchmarks`
  - before: `[]`
  - after: `["maximally random jammed state", "packing fraction phi c"]`
- `toolchains_and_infrastructure`
  - before: `[]`
  - after: `["lubachevsky-stillinger compression protocol", "radial distribution function"]`
- `readiness_scores.data_resource`
  - before: `0.33`
  - after: `0.67`
- `readiness_scores.infrastructure`
  - before: `0.33`
  - after: `0.67`
- `readiness_scores.overall`
  - before: `0.70`
  - after: `0.80`
- `why_now_features.unlocking_factors`
  - before: no explicit benchmark-backed unlocking factor
  - after: includes `maximally random jammed state` and `packing fraction phi c`
- `not_now_features.missing_prerequisites`
  - before: benchmark coverage remained explicitly missing
  - after: benchmark-related missing prerequisites are cleared
- `DecisionEpisode.minimal_attack_path.required_resources`
  - before: `[]`
  - after:
    - `maximally random jammed state`
    - `packing fraction phi c`
    - `lubachevsky-stillinger compression protocol`
    - `radial distribution function`

What did **not** change yet:

- replay `quality_flags` remain:
  - `support_cluster_too_small`
  - `no_alternative_route_states`
  - `held_out_routes_missing`

## Interpretation

This is the first proof that improving `L1-lite` can materially change downstream route compilation without changing the `L2` schema.

The main win is not just that the snapshot looks better.

It is that the replay artifacts become more useful as historical reasoning carriers:

- the route now has explicit benchmark-level unlocking factors
- the minimal attack path has concrete historically grounded resources
- readiness is less distorted by empty benchmark / infrastructure fields

## Remaining Gap

The strongest remaining weakness is no longer benchmark recovery.

It is route-packaging structure:

- support cluster route states are still missing
- alternative route states are still missing in the replay runtime
- held-out route states are still missing

That means the next highest-value step is to package reusable support / alternative / held-out route states on top of the now-stronger `L1 + packet + replay` chain.
