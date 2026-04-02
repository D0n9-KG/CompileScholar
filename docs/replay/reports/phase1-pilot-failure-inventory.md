# Phase 1 Pilot Failure Inventory

## Purpose

This document converts the first real replay pilot into a layer-aware failure taxonomy so later phases can act without rereading the full bundle.

Reference bundle:

- `backend/runs/replay/phase1-pilot/`

## Failure Table

| ID | Layer | Severity | Observed Symptom | Likely Root Cause | Target Phase |
|----|-------|----------|------------------|-------------------|--------------|
| `PKT-01` | Packet | degrading | The executed pilot had to use `tmp/phase1_runtime_subset_packet.json` instead of the full committed packet. | Two committed packet items (`820`, `1647`) did not have resolved local trace exports at replay time. | Phase 1 follow-up or Phase 3 replay data hygiene |
| `PKT-02` | Packet | degrading | The runtime packet has no `limitation_or_critique` item. | The intended critique paper was one of the missing-source items, so the packet lost a critique-bearing role in execution. | Phase 3 multi-paper replay compilation |
| `L1-01` | `L1` | blocking for stronger episodes | `minimal_attack_path.required_resources` is empty even though the episode asks a benchmark/resource-gap question. | `L1` snapshot refs are still logical placeholders; no real historical benchmark/resource registry has been connected to the replay. | Phase 2 Historical Environment Bridge |
| `L1-02` | `L1` | degrading | The route is framed as historically actionable, but environment readiness evidence is thin. | The replay currently depends almost entirely on paper-text evidence and lacks authoritative period environment assets. | Phase 2 Historical Environment Bridge |
| `L2-01` | `L2` | degrading | Thin conceptual / review papers such as `777` seed useful direction but weak route-state detail. | `L2` handles survey-like papers as direct move sources, but route-state seed density and role stitching remain thin for concept-heavy texts. | Phase 4 `L2` surgical loop |
| `L2-02` | `L2` | degrading | Random spotcheck papers frequently remain `yellow` because of `missing_expected_slots`, `sparse_expected_slots`, or `weak_relation_stitching`. | Comparator signals, resource framing, and move-to-move relation stitching are still unstable across paper styles and domains. | Phase 4 `L2` surgical loop |
| `L2-03` | `L2` | degrading | Some papers pass gate cleanly but still do not give strong downstream comparison cues. | `L2` is better at capturing local method/result structure than at surfacing reusable competition and critique structure for later aggregation. | Phase 4 `L2` surgical loop |
| `L3-01` | `L3` | blocking | Replay emitted `no_alternative_route_states` and no comparison case was selected. | The current workflow compiles only the primary route state unless alternative route-state artifacts are explicitly provided. | Phase 3 Grounded Route Replay Compilation |
| `L3-02` | `L3` | blocking | Replay emitted `support_cluster_too_small`. | There is no support-cluster packaging workflow yet; the prior layer only saw one route state. | Phase 3 Grounded Route Replay Compilation |
| `L3-03` | `L3` | blocking | Replay emitted `held_out_routes_missing`. | Held-out route-state generation and audit packaging have not been implemented yet. | Phase 3 and Phase 5 |
| `L4-01` | `L4` | degrading | `DecisionPriorCard` stayed `yellow` with `weak_support_cluster` and `missing_counterexample_search`. | Prior induction is functioning structurally, but not yet on a multi-route evidence basis. | Phase 5 Multi-Route Prior Induction |
| `L4-02` | `L4` | degrading | `DecisionEpisode` is `ready_for_eval` but not `ready_for_training`. | The episode is grounded enough for inspection, but lacks strong prior support, alternative routes, and explicit required resources. | Phase 5 and Phase 6 |
| `PROC-01` | Review process | degrading | The first replay run depended on local ad hoc trace preparation and manual runtime subseting. | The project has replay IO now, but not yet a fully standardized trace-resolution and packet-to-runtime preparation workflow. | Phase 3 tooling hardening |
| `PROC-02` | Review process | degrading | Random L2 spotchecks are informative but still semi-manual. | There is no committed repeatable spotcheck harness yet for cross-domain L2 regression tracking. | Phase 4 evaluation tooling |

## Supplemental Evidence From Random L2 Spotcheck

Five random extra papers were spotchecked from the user-provided local corpus.

Result summary:

- `5/5` passed the hot-path gate
- `1/5` was `green`
- `4/5` were `yellow`
- recurring flags:
  - `missing_expected_slots`
  - `sparse_expected_slots`
  - `weak_relation_stitching`

Interpretation:

- current `L2` is broadly viable as a single-paper evidence layer
- current `L2` is not yet stable enough to guarantee strong multi-paper route aggregation without targeted fixes

## Prioritization

Recommended priority order from this pilot:

1. Build real `L1` environment assets for the pilot route so resource/benchmark gaps stop collapsing into empty fields.
2. Build explicit support / alternative / held-out route-state packaging so `L3/L4` receive the inputs they were designed to consume.
3. Patch the highest-leverage `L2` gaps exposed by the pilot, especially comparator capture, expected-slot density, and relation stitching.

## What Should Not Happen Next

The pilot does **not** support the following next move:

- broad uncontrolled expansion of the `L2` schema without replay-grounded evidence

The pilot **does** support:

- targeted `L2` fixes after the missing `L1` and multi-route packaging boundaries are made explicit
