# L1-Lite Snapshot Pilot - 2026-04-02

## Purpose

This note records the first real `paper_grounded_l1_lite` snapshot built from packetized `PaperLogicTrace` inputs.

The goal was not to prove that paper-only evidence is enough for a final `L1`, but to verify that the repo can now build a minimal historical environment snapshot when no external resource registry is available.

## Runtime Inputs

- packet: `tmp/phase1_runtime_subset_packet.json`
- traces:
  - `tmp/phase1_packet_traces/1591/paper_logic_trace.json`
  - `tmp/phase1_packet_traces/1742/paper_logic_trace.json`
  - `tmp/phase1_packet_traces/1746/paper_logic_trace.json`
  - `tmp/phase1_packet_traces/777/paper_logic_trace.json`
  - `tmp/phase1_packet_traces/814/paper_logic_trace.json`

Command shape:

```powershell
cd backend
.\.venv\Scripts\python.exe scripts/run_l1_snapshot_pilot.py `
  --packet ..\tmp\phase1_runtime_subset_packet.json `
  --output-path runs\l1\phase1-pilot\historical_environment_snapshot.json `
  --trace-file ..\tmp\phase1_packet_traces\1591\paper_logic_trace.json `
  --trace-file ..\tmp\phase1_packet_traces\1742\paper_logic_trace.json `
  --trace-file ..\tmp\phase1_packet_traces\1746\paper_logic_trace.json `
  --trace-file ..\tmp\phase1_packet_traces\777\paper_logic_trace.json `
  --trace-file ..\tmp\phase1_packet_traces\814\paper_logic_trace.json
```

## Output

Local snapshot output:

- `backend/runs/l1/phase1-pilot/historical_environment_snapshot.json`

Generated snapshot identity:

- `snapshot_id`: `jamming_transition_in_frictionless_sphere_packings_near_point_j_2010_l1_snapshot_v1`
- `topic_scope`: `jamming transition in frictionless sphere packings near point J`
- `cutoff_year`: `2010`

Generated logical refs:

- `resource_registry_ref`: `l1:jamming_transition_in_frictionless_sphere_packings_near_point_j_2010_l1_snapshot_v1:resource-registry`
- `resource_timeline_ref`: `l1:jamming_transition_in_frictionless_sphere_packings_near_point_j_2010_l1_snapshot_v1:resource-timeline`
- `benchmark_timeline_ref`: `l1:jamming_transition_in_frictionless_sphere_packings_near_point_j_2010_l1_snapshot_v1:benchmark-timeline`
- `toolchain_timeline_ref`: `l1:jamming_transition_in_frictionless_sphere_packings_near_point_j_2010_l1_snapshot_v1:toolchain-timeline`
- `protocol_registry_ref`: `l1:jamming_transition_in_frictionless_sphere_packings_near_point_j_2010_l1_snapshot_v1:protocol-registry`

## Quality Result

- `quality_tier`: `yellow`
- `quality_flags`:
  - `paper_only_snapshot`
  - `preferred_benchmark_missing`
  - `toolchain_timeline_empty`

## Interpretation

This result is useful precisely because it is incomplete in an honest way.

What worked:

- the repo can now generate a bounded, pre-cutoff `L1-lite` snapshot directly from multiple `PaperLogicTrace` objects
- the snapshot produces stable logical refs that can later replace placeholder `L1` refs in packets
- the builder preserves the distinction between available evidence and missing evidence instead of fabricating a full environment layer

What remained weak:

- benchmark recovery is still thin for this packet
- toolchain visibility in the current paper set is not strong enough to populate a useful timeline
- because the snapshot is paper-only, the best honest quality tier is still `yellow`

## Why This Matters

This pilot answers the practical project question:

- if no external historical registry exists yet, the project can still build a first `L1-lite` from paper evidence alone

It does **not** claim:

- that paper-only `L1` is sufficient forever
- that the resulting snapshot already matches a fully curated historical environment

The right next step is to use this builder as the Phase 2 foundation, then gradually strengthen it with better benchmark, protocol, and toolchain coverage.
