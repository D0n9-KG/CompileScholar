# Plan 01-02 Summary

## Outcome

Phase 1 / Plan `01-02` is complete.

The repo now contains the first committed pilot packet manifest for historical replay:

- `docs/replay/pilot_packets/README.md`
- `docs/replay/pilot_packets/phase1-route-packet.json`
- `docs/replay/pilot_packets/phase1-selection-notes.md`
- `backend/tests/test_phase1_route_packet_manifest.py`

## Packet Chosen

The first pilot packet is a bounded historical slice around:

- `topic_scope_candidate`: `jamming transition in frictionless sphere packings near point J`
- `cutoff_year`: `2010`

Why this topic was chosen:

- it has a historically clear cutoff
- the main route is easy to distinguish from nearby alternatives
- critique / instability papers exist before the cutoff
- it is narrow enough to make replay failures interpretable

## What Was Added

### 1. Committed Packet Workspace

`docs/replay/pilot_packets/README.md` now explains:

- what packet artifacts are committed
- what remains local-only
- why packet manifests and local replay bundles are separated

### 2. First Real Packet Manifest

`docs/replay/pilot_packets/phase1-route-packet.json` now provides a schema-valid `RoutePacket` with:

- explicit packet identity and cutoff
- role composition
- L1 snapshot references
- inclusion rules
- included items
- excluded items
- packet quality flags
- compiler hints

The packet is intentionally `yellow`, not `green`, because it is a bounded pilot with local-only trace exports and placeholder-level L1 backfill.

### 3. Audit Notes

`docs/replay/pilot_packets/phase1-selection-notes.md` now records:

- why the packet is bounded
- why the chosen topic is suitable for the first replay pilot
- included-paper role assignments
- excluded post-cutoff items
- leakage discipline
- known trace / L1 gaps

### 4. Manifest Regression Test

`backend/tests/test_phase1_route_packet_manifest.py` now protects the committed manifest by asserting:

- the JSON validates as `RoutePacket`
- packet role coverage remains explicit
- every included item has a role, reason, and `trace_id`
- leakage policy keeps post-cutoff and hindsight rules explicit
- no absolute Windows path or UNC share path slips into the committed manifest

## Verification

Executed successfully:

```powershell
cd backend
@'
import json
from pathlib import Path
from app.research_logic.models import RoutePacket
packet = json.loads(Path('..\\docs\\replay\\pilot_packets\\phase1-route-packet.json').read_text(encoding='utf-8'))
RoutePacket.model_validate(packet)
print('route-packet-ok')
'@ | .\.venv\Scripts\python.exe -

.\.venv\Scripts\python.exe -m pytest tests\test_phase1_route_packet_manifest.py -q
```

Result:

- `route-packet-ok`
- `1 passed`

## Remaining Gap Before Plan 01-03

The packet boundary is now committed, but the first real replay still needs:

- a local `PaperLogicTrace` export source for the included packet items
- the actual replay run against that local trace set
- the review report and failure inventory produced from the run

## Recommended Next Step

Proceed to Phase 1 / Plan `01-03`:

- resolve the local trace-export source for the seven included packet papers
- run `backend/scripts/run_replay_pilot.py` against the committed packet
- write the first `phase1-pilot-review.md`
- write the first `phase1-pilot-failure-inventory.md`
