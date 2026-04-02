# Phase 1 Pilot Review

## Scope

This report records the first real historical replay pilot for Phase 1.

The committed packet definition lives in:

- `docs/replay/pilot_packets/phase1-route-packet.json`

The actual replay run used a local runtime subset packet because two committed packet items (`820`, `1647`) did not have resolved local trace exports during this pilot.

## Runtime Inputs

- committed packet basis: `docs/replay/pilot_packets/phase1-route-packet.json`
- runtime packet used for execution: `tmp/phase1_runtime_subset_packet.json`
- topic scope: `jamming transition in frictionless sphere packings near point J`
- cutoff year: `2010`
- included traces available at runtime: `5`
- reviewer ids: `codex-local`

Runtime papers included in the executed subset:

- `777` - `Jamming, Force Chains, and Fragile Matter`
- `1742` - `Is Random Close Packing of Spheres Well Defined?`
- `1746` - `Random Packings of Frictionless Particles`
- `1591` - `Jamming at zero temperature and zero applied stress: The epitome of disorder`
- `814` - `Jamming of frictional spheres and random loose packing`

## Command Shape

The pilot was replayed through the Phase 1 CLI entry point:

```powershell
cd backend
.\.venv\Scripts\python.exe scripts/run_replay_pilot.py `
  --packet ..\tmp\phase1_runtime_subset_packet.json `
  --output-dir runs\replay\phase1-pilot `
  --trace-file ..\tmp\phase1_packet_traces\1591\paper_logic_trace.json `
  --trace-file ..\tmp\phase1_packet_traces\1742\paper_logic_trace.json `
  --trace-file ..\tmp\phase1_packet_traces\1746\paper_logic_trace.json `
  --trace-file ..\tmp\phase1_packet_traces\777\paper_logic_trace.json `
  --trace-file ..\tmp\phase1_packet_traces\814\paper_logic_trace.json `
  --reviewer codex-local
```

## Bundle Location

Stable local replay bundle:

- `backend/runs/replay/phase1-pilot/`

Important bundle files:

- `backend/runs/replay/phase1-pilot/bundle_manifest.json`
- `backend/runs/replay/phase1-pilot/replay_summary.json`
- `backend/runs/replay/phase1-pilot/outputs/primary_route_state.json`
- `backend/runs/replay/phase1-pilot/outputs/decision_prior_card.json`
- `backend/runs/replay/phase1-pilot/outputs/decision_episode.json`

## Top-Level Outcome

Replay summary:

- `ready_for_pilot`: `true`
- `selected_comparison_case_id`: `null`
- replay-level `quality_flags`:
  - `support_cluster_too_small`
  - `no_alternative_route_states`
  - `held_out_routes_missing`

Interpretation:

- the current stack can compile a coherent replay bundle from real packetized traces
- the bundle is usable for pilot inspection
- the replay is still structurally incomplete for trustworthy prior induction and episode training

## Stage Quality Snapshot

| Stage | Quality | Notes |
|-------|---------|-------|
| `RouteState` | `green` | Main route compilation succeeded and is marked ready for `WhyNow` / route comparison |
| `DecisionPriorCard` | `yellow` | Flags: `weak_support_cluster`, `missing_counterexample_search` |
| `DecisionEpisode` | `yellow` | Flags: `weak_prior_support`, `weak_alternative_set`, `minimal_attack_path_missing` |

What compiled successfully:

- one grounded `RouteState`
- one `WhyNowCase`
- one `DecisionPriorCard`
- one `DecisionEpisode`

What did not compile into a useful object yet:

- no route-comparison case was selected
- no alternative route-state bundle was supplied
- no held-out route-state bundle was supplied

## Most Important Replay Signal

The generated candidate question was:

> What missing benchmark or resource would make jamming transition in frictionless sphere packings near point J feasible enough to test?

This is a good Phase 1 result for two reasons:

- the output is not a generic summary; it is already shaped like a research-decision prompt
- the prompt is grounded in a concrete missing-environment signal rather than an invented open-ended hypothesis

However, the same bundle also shows why the stack is not ready for training:

- `minimal_attack_path.required_resources` is empty
- comparison dimensions are empty because no alternative route-state bundle was supplied
- the prior support cluster contains only the primary route state

## Packet-Level Caveats Exposed By The Pilot

The pilot replayed a runtime subset, not the full committed packet.

Important packet caveats now confirmed by execution:

- the runtime subset is missing the intended `limitation_or_critique` paper
- two committed packet papers still lack resolved local trace export paths
- packet role coverage is good enough for a first route-state replay, but not yet good enough for a stronger comparison / critique bundle

## Supplemental L2 Spotcheck

To avoid overfitting conclusions to the jamming packet alone, a small random spotcheck was run on five additional papers sampled from the user-provided local corpus.

High-level result:

- sampled papers: `5`
- `gate_passed`: `5/5`
- `green`: `1`
- `yellow`: `4`

What this means for Phase 1:

- current `L2` is not narrowly working only for the packet topic
- single-paper traces can usually reach an auditable, structurally usable state
- but key downstream slots remain unstable, especially comparator coverage, expected-slot density, and relation stitching

See also:

- `docs/replay/reports/l2-random-spotcheck-2026-04-02.md`

## Conclusion

Phase 1 achieved its main pilot goal:

- a real packet can now be replayed through the new JSON artifact boundary
- the system emits inspectable quality flags instead of hiding failure behind free-form prose
- the first replay failure taxonomy is now concrete enough to drive Phase 2 and later `L2` surgical work

The replay does **not** justify claiming that `L3/L4` are solved.

What it does justify:

- Phase 2 should focus on real `L1` historical environment assets
- later `L2` fixes should be driven by replay failures, not by open-ended schema expansion
- support / alternative / held-out route-state packaging must become an explicit workflow rather than an implicit future assumption
