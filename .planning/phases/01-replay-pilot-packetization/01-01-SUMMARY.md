# Plan 01-01 Summary

## Outcome

Phase 1 / Plan `01-01` is complete.

This plan established the first reusable replay workflow boundary for the `research_logic` layer:

- Added typed replay artifact I/O helpers in `backend/app/research_logic/replay_io.py`
- Added a local CLI runner in `backend/scripts/run_replay_pilot.py`
- Added regression coverage in `backend/tests/test_replay_io.py`
- Re-exported replay helpers from `app.research_logic`

## What Was Built

### 1. Replay Artifact I/O

`backend/app/research_logic/replay_io.py` now provides:

- `load_route_packet`
- `load_paper_logic_trace`
- `load_paper_logic_traces`
- `load_route_state`
- `load_route_states`
- `ensure_packet_trace_coverage`
- `build_replay_summary`
- `write_replay_bundle`

The new I/O layer keeps replay inputs explicit and local-path driven. No machine-specific corpus path or share path is hardcoded.

### 2. Local Replay Runner

`backend/scripts/run_replay_pilot.py` now supports:

- `--packet`
- `--trace-dir`
- `--trace-file`
- `--output-dir`
- `--support-route-state`
- `--alternative-route-state`
- `--held-out-route-state`
- `--reviewer`

The runner:

- loads packet / trace / route-state artifacts through the shared replay I/O helpers
- delegates compilation to `compile_historical_replay`
- writes a structured replay bundle directory
- prints a machine-readable JSON summary with `ready_for_pilot`, `selected_comparison_case_id`, and `quality_flags`
- exits non-zero on invalid artifacts or missing trace coverage

### 3. Regression Tests

`backend/tests/test_replay_io.py` now verifies:

- packet manifests load into `RoutePacket`
- trace directories and explicit trace files load into `PaperLogicTrace`
- missing `trace_id` coverage raises an explicit error
- serialized `RouteState` artifacts can be loaded back
- replay bundles write the expected JSON files
- CLI output exposes `ready_for_pilot` and `quality_flags`

## Verification

Executed successfully:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest tests\test_replay_io.py tests\test_historical_replay_compiler.py -q
.\.venv\Scripts\python.exe scripts\run_replay_pilot.py --help
```

Result:

- `8 passed`
- CLI help rendered correctly

## Artifacts Produced

Code artifacts:

- `backend/app/research_logic/replay_io.py`
- `backend/scripts/run_replay_pilot.py`
- `backend/tests/test_replay_io.py`
- `backend/app/research_logic/__init__.py`

Replay bundle contract written by the new helper:

- `bundle_manifest.json`
- `replay_summary.json`
- `inputs/route_packet.json`
- `inputs/paper_logic_traces.json`
- `inputs/support_route_states.json`
- `inputs/alternative_route_states.json`
- `inputs/held_out_route_states.json`
- `outputs/primary_route_state.json`
- `outputs/why_now_case.json`
- `outputs/route_comparison_cases.json`
- `outputs/decision_prior_card.json`
- `outputs/decision_episode.json`

## Remaining Gaps Before Real Pilot

- There is still no committed real `route packet` manifest for the first pilot scope
- `L1 Historical Environment` references are still placeholder-level for the actual pilot
- No real corpus replay review or failure inventory has been produced yet

## Recommended Next Step

Proceed to Phase 1 / Plan `01-02`:

- create the first committed pilot packet manifest
- add selection notes and packet README
- add a manifest regression test so `run_replay_pilot.py` can be pointed at a committed packet without per-run patching
