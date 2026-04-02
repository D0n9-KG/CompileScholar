---
phase: 01-replay-pilot-packetization
plan: 03
requirements-completed: [PKT-01, PKT-02, EVAL-01]
completed: 2026-04-02
---

# Plan 01-03 Summary

## Outcome

Phase 1 / Plan `01-03` is complete.

The repo now contains the first replay review artifacts for a real local pilot:

- `docs/replay/reports/phase1-pilot-review.md`
- `docs/replay/reports/phase1-pilot-failure-inventory.md`
- `docs/replay/reports/l2-random-spotcheck-2026-04-02.md`

## What Was Executed

The first replay pilot was executed through:

- `backend/scripts/run_replay_pilot.py`

Stable bundle location:

- `backend/runs/replay/phase1-pilot/`

Packet used at runtime:

- `tmp/phase1_runtime_subset_packet.json`

Why a runtime subset was used:

- the committed packet still includes `820` and `1647`
- those two items did not have resolved local trace exports during the pilot
- the runtime subset preserved the real topic boundary while allowing the replay bundle to execute end-to-end

## Replay Result

Top-level replay status:

- `ready_for_pilot`: `true`
- replay `quality_flags`:
  - `support_cluster_too_small`
  - `no_alternative_route_states`
  - `held_out_routes_missing`

Stage outcomes:

- `RouteState`: `green`
- `DecisionPriorCard`: `yellow`
- `DecisionEpisode`: `yellow`

The most important positive result:

- the system now compiles a real replay bundle from packetized traces instead of only toy fixtures

The most important negative result:

- the replay is structurally incomplete for trustworthy prior induction because support, alternative, and held-out route-state inputs are still missing

## New Review Artifacts

### 1. Pilot Review

`docs/replay/reports/phase1-pilot-review.md` records:

- the packet basis and runtime subset used
- the actual replay command shape
- the stable bundle directory
- replay readiness and quality flags
- which stage outputs compiled successfully
- why the episode is still not training-ready

### 2. Failure Inventory

`docs/replay/reports/phase1-pilot-failure-inventory.md` converts the pilot into a reusable taxonomy across:

- packet
- `L1`
- `L2`
- `L3`
- `L4`
- process/tooling

This is the main artifact that should steer the next phases.

### 3. Supplemental Random L2 Spotcheck

`docs/replay/reports/l2-random-spotcheck-2026-04-02.md` records five random extra corpus spotchecks.

Spotcheck summary:

- `5/5` passed the hot-path gate
- `1/5` was `green`
- `4/5` were `yellow`

Why this matters:

- the current `L2` layer is not only working on the packet topic
- but it is still too unstable on comparator density, expected slots, and relation stitching to assume reliable downstream aggregation

## Verification

Executed successfully:

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

Result:

- replay bundle written to `backend/runs/replay/phase1-pilot/`
- `ready_for_pilot = true`
- replay emitted explicit structural quality flags instead of failing silently

## Main Conclusion

Phase 1 has now done what it needed to do:

- the packet boundary is committed
- the replay runner works on real local traces
- the project now has a written diagnosis trail grounded in real failures

This pilot does **not** say that `L3/L4` are already sufficient.

It says something more useful:

- the current architecture is good enough to expose the real missing layers
- the next steps should follow the replay failure inventory rather than broad unsupervised `L2` expansion

## Recommended Next Step

Proceed to Phase 1 verification / completion, then start Phase 2:

- verify the Phase 1 artifacts as a coherent bundle
- build real `L1 Historical Environment` assets for the pilot route
- after that, harden support / alternative / held-out route-state packaging before starting the `L2` surgical loop
