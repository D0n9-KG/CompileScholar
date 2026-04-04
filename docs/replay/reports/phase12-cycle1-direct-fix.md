# Phase 12 Cycle 1 Direct Fix Report

## Starting Evidence

Phase 11 named `packet_construction` as the first repair target for this bounded computational-mechanics slice. The inherited recommendation came from `tmp/phase11_iteration_prioritization/baseline/outputs/prioritization_summary.json`, which recorded `primary_recommendation_id = "packet_construction"` because the Phase 10 evidence surface was failing first at package construction rather than at `L2`.

At the start of this cycle, the inherited blocker queue from the committed Phase 10 / Phase 11 evidence chain was:

- package validation: `support_cluster_too_small`, `alternative_scope_not_distinct`, `yellow_route_state_present`
- replay: `support_cluster_too_small`, `reviewer_missing`, and `decision_prior_card` failures
- prior/export: `no_prior_candidates`, `accepted_prior_ids_empty`, and `weak_prior_support`

The frozen packet boundary did not change during this cycle:

- packet id: `phase9_comp_mech_2021_packet_01`
- cutoff year: `2021`
- packet scope: bounded data-driven constitutive and multiscale computational mechanics
- baseline comparison surfaces: `tmp/phase3_route_state_package/replay_with_package/` and `tmp/phase6_decision_episode_audit_export/`

## Fixes Applied

- Split the runtime Phase 10 support emission into three concrete support artifacts while preserving the committed Phase 9 membership:
  - `support-core`: papers `1000`, `1001`
  - `support-context`: papers `1002`, `1005`
  - `support-clustering`: paper `1017`
- Preserved `alternative.distinctness_rationale` from `docs/replay/pilot_packets/phase9-assembly-manifest.json` on the generated alternative route-state package entry so the validator can distinguish an explicitly documented alternative route from an undocumented scope overlap.
- Kept the existing Phase 10 CLI and the same frozen Phase 9 packet inputs. This cycle changed the runtime bridge and package validation behavior, not the packet boundary or the execution surface.

## Blocker Delta

- `support_cluster_too_small`: moved. The fresh cycle-1 package validation omits this flag, `package.current.role_counts.support` increased from the inherited single support entry to `3`, and replay no longer reports the packet-construction blocker.
- `alternative_scope_not_distinct`: moved. The fresh package validation omits this flag and records `indistinct_alternative_entry_ids = []`, so the Phase 9 alternative rationale is now surviving the runtime bridge.
- `yellow_route_state_present`: stayed. Package validation is now `yellow` rather than `red`, but it still reports `quality_flags = ["yellow_route_state_present"]`.
- `reviewer_missing`: stayed. Replay remains `red` and `ready_for_pilot = false` because `quality_flags = ["reviewer_missing"]`.
- `decision_prior_card`: improved but unresolved. The inherited Phase 10 / Phase 11 evidence chain showed `2` replay failures at `decision_prior_card`; the fresh cycle-1 run records `1`.

The fresh cycle-1 blocker queue therefore drops the two lead packet-construction blockers without hiding the remaining yellow package signal or the downstream replay/prior bottlenecks.

## Manual Output Review

Package quality improved at the exact surface this cycle targeted. The generated route-state package now has `support = 3`, `alternative = 1`, and `held_out = 1`, and it is `ready_for_replay = true`. On the bounded slice as a whole, that means the package is structurally usable instead of structurally blocked. It is not fully clean, though: one support entry and both non-support entries still land at yellow, so the package remains cautionary rather than green.

Replay quality improved only partially. The bounded slice no longer fails because the support cluster is too thin, and `route_state` failures stay lower than the old baseline bundle at `2` instead of `3`. But the replay bundle is still not pilot-ready because `reviewer_missing` remains the active blocking flag, and the run still records one `decision_prior_card` failure. That is better than the inherited Phase 10 surface, but it is not a full replay-quality recovery.

Prior-review quality did not recover. The cycle-1 bundle forms `3` clusters, but it still yields `prior_candidate_count = 0`, `anti_pattern_candidate_count = 0`, `accepted_prior_ids = []`, and `accepted_anti_pattern_ids = []`. On this bounded slice, the packet repair did not unlock reusable prior evidence or reviewed anti-pattern carryover.

Export quality stayed audit-safe but weak. The export bundle remains `quality_tier = "yellow"`, `ready_for_eval = true`, and `ready_for_training = false`, with `quality_flags = ["weak_prior_support"]`. The visibility mix improved slightly toward more visible inputs (`28` visible refs vs `24` in the baseline export, `26` audit-only refs vs `34`), but the bounded slice still exports with `selected_antipattern_count = 0` against a historical baseline of `5`.

## Verdict

Cycle 1 improved packet construction but did not produce an end-to-end bounded-slice quality recovery. The direct fix removed `support_cluster_too_small` and `alternative_scope_not_distinct`, kept `yellow_route_state_present` explicit, and cut `decision_prior_card` failures from `2` to `1`, but the run still ends with `reviewer_missing`, no prior candidates, and `weak_prior_support`.

The honest Phase 12 closeout for this cycle is that quality mostly shifted downstream after the packet repair rather than turning green overall. That is still useful progress: the repo now has a repaired packet/runtime bridge and a fresh cycle-1 output root under `tmp/phase12_direct_fix_cycle/cycle1/` that future work can treat as the new bounded baseline. This verdict applies only to the current computational-mechanics slice and does not claim generalized readiness beyond it.
