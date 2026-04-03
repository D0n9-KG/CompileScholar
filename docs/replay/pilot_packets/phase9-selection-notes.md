# Phase 9 Packet Selection Notes

## Packet Identity

- `packet_id`: `phase9_comp_mech_2021_packet_01`
- `topic_scope_candidate`: `data-driven constitutive and multiscale computational mechanics`
- `cutoff_year`: `2021`
- `packet_status`: `reviewed`

## Why This Topic Boundary

This first corpus-backed packet stays inside the fixed-set, owner-queue-driven slice that Phase 8 already measured directly:

- `1000`, `1001`, `1005`, `1017`, and `1023` are the strongest recurring-failure anchors from the `relation_assembly` and `slot_recovery` buckets.
- The shared vocabulary is still one defensible packet boundary: data-driven constitutive response, multiscale mechanics, and clustering-based route construction.
- The packet is intentionally narrower than "computational mechanics plus any ML paper" because Phase 10 needs one auditable boundary, not a loosely related reading list.

## Why The Cutoff Freezes At 2021

`2021` is the tightest cutoff that still keeps the Phase 8 anchor set intact:

- it includes `1000`, `1001`, `1005`, `1017`, and `1023`
- it allows one same-slice data-conditioning expansion (`1002`) and one learned alternative (`1007`)
- it turns `1004` into a clear post-cutoff exclusion instead of silently stretching the packet boundary

The cutoff is therefore a deliberate historical discipline choice, not a convenience date.

## Included Papers And Packet Roles

| Paper ID | Year | Role | Why Included |
|----------|------|------|--------------|
| `1000` | 2017 | `core_method` | Main constitutive-route anchor and one of the top Phase 8 relation-assembly exemplars |
| `1001` | 2019 | `core_method` | Keeps the no-constitutive-equation stress-field route inside the support cluster |
| `1002` | 2020 | `resource_or_benchmark` | Makes noisy-database and data-conditioning constraints explicit instead of invisible background |
| `1005` | 2020 | `survey_or_review` | Frames the packet at the multiscale-mechanics level so the boundary stays coherent |
| `1007` | 2018 | `alternative_route` | Preserves one learned multiscale alternative without broadening to every ML-heavy mechanics paper |
| `1017` | 2019 | `core_method` | Keeps virtual-clustering analysis in the core route family under the same cutoff |
| `1023` | 2021 | `limitation_or_critique` | Preserves route sensitivity through reference-stiffness selection near the cutoff |

This packet does not pretend that every same-slice paper is included. It commits one bounded set that fills the packet roles cleanly and leaves the rest as explicit exclusions.

## Explicit Exclusions

| Paper ID | Exclusion | Reason |
|----------|-----------|--------|
| `1004` | `after_cutoff` | Post-cutoff 2023 neural-ODE viscoelasticity paper; excluded by cutoff discipline |
| `1010` | `off_topic` | Reinforcement-learning traction-separation paper would broaden the packet into a different subproblem |
| `1012` | `duplicate_signal` | Same-slice follow-on candidate, but it enlarges the first packet without filling a missing role |
| `1107` | `off_topic` | Random exploration explosive-materials paper is out-of-slice for this computational-mechanics packet |

`1004` is excluded because it is post-cutoff, not because it is irrelevant. `1107` is excluded because it is out-of-slice, not because Phase 8 never ran it.

## Fallback-Path Caveats

The packet intentionally mixes healthy markdown-first sources with fallback-heavy fixed-set papers:

- `1000` and `1001` came from healthy markdown-preferred paths in the fixed set
- `1002`, `1007`, `1017`, and `1023` rely on the fixed-set fallback discipline that Phase 7 and Phase 8 already recorded
- the committed packet stores only portable ids, trace ids, and relative artifact references; it does not commit machine-local UNC source paths

This means the packet boundary is portable even though some source-path recovery history remains messy in the underlying corpus.

## What This Packet Supports

This packet is meant to support the next handoff, not to claim replay success:

- Phase 10 can reuse the frozen topic boundary without inventing a new packet definition
- support, alternative, and held-out slicing can be layered on top of the committed members
- the remaining blockers stay visible: thin support density, fallback-heavy source mix, and placeholder `L1` refs

## What This Packet Should Not Be Used To Claim

This packet should not be treated as:

- a complete survey of data-driven computational mechanics
- proof that every same-slice paper belongs in one packet
- proof that Phase 10 replay is already green

Its job is to make the first bounded corpus packet auditable before multi-paper replay work begins.
