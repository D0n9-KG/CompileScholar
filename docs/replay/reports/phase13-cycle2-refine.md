# Phase 13 Cycle 2 Refine Report

## Baseline

Phase 13 uses `tmp/phase12_direct_fix_cycle/cycle1/` as the fixed comparison anchor. That cycle already removed the packet-construction blockers `support_cluster_too_small` and `alternative_scope_not_distinct`, but it still ended with:

- package `quality_flags = ["yellow_route_state_present"]`
- replay `quality_flags = ["reviewer_missing"]`
- replay `failure_counts_by_stage.decision_prior_card = 1`
- prior review `prior_candidate_count = 0`
- prior review `accepted_prior_ids = []`
- export `quality_flags = ["weak_prior_support"]`

The packet boundary did not change in Phase 13:

- packet id: `phase9_comp_mech_2021_packet_01`
- cutoff year: `2021`
- packet scope: bounded data-driven constitutive and multiscale computational mechanics

## Iteration Log

### `iter-01`

- Command: `cd backend; .\.venv\Scripts\python.exe scripts\run_phase13_cycle_refinement.py --iteration-label iter-01 --reviewer reviewer-1 --reviewer reviewer-2 --output-root ..\tmp\phase13_cycle2_refine`
- Output root: `tmp/phase13_cycle2_refine/iter-01/`
- Result: replay moved from `red` to `green`, `reviewer_missing` disappeared, and `ready_for_pilot` became `true`
- Remaining issue: prior review still recorded `prior_candidate_count = 0`, `accepted_prior_ids = []`, and export stayed at `weak_prior_support`

### `cycle2-final`

- Command: `cd backend; .\.venv\Scripts\python.exe scripts\run_phase13_cycle_refinement.py --iteration-label cycle2-final --reviewer reviewer-1 --reviewer reviewer-2 --allow-scope-fallback-merge --output-root ..\tmp\phase13_cycle2_refine`
- Output root: `tmp/phase13_cycle2_refine/cycle2-final/`
- Result: replay stayed `green`, prior review switched to `cluster_strategy = "fallback_scope_merge"`, and the final run accepted one prior id: `prior:data_driven_constitutive_and_multiscale_computational_mechanics_scope_fallback_cluster`
- Chosen final surface: `cycle2-final`

## Final Defect Delta

The final Phase 13 comparison is `tmp/phase13_cycle2_refine/cycle2-final/comparison_summary.json` versus `tmp/phase12_direct_fix_cycle/cycle1/comparison_summary.json`.

- `yellow_route_state_present`: stayed flat. Package validation is still `yellow` and still carries this residual signal.
- `reviewer_missing`: moved. The flag was present in Phase 12 cycle 1 and is gone in the final Phase 13 replay output.
- replay `decision_prior_card`: moved. Phase 12 cycle 1 recorded `failure_counts_by_stage.decision_prior_card = 1`; the final Phase 13 replay summary records no `decision_prior_card` failures.
- `prior_candidate_count`: moved. The final Phase 13 prior review increased from `0` to `1`.
- `accepted_prior_ids`: moved. The baseline had `[]`; the final Phase 13 prior review accepted `["prior:data_driven_constitutive_and_multiscale_computational_mechanics_scope_fallback_cluster"]`.
- `weak_prior_support`: stayed flat. Export still reports `quality_flags = ["weak_prior_support"]`.

Net effect by stage:

- Package: unchanged relative to Phase 12 cycle 1
- Replay: materially improved and now pilot-ready
- Prior review: materially improved only when the bounded fallback merge is enabled
- Export: improved evidence posture, but still not training-ready

## Manual Output Review

Package quality is stable but not cleaner. The final package still presents the same bounded role layout (`support = 3`, `alternative = 1`, `held_out = 1`) and remains `ready_for_replay = true`, but it does not move beyond the existing `yellow_route_state_present` warning. This is acceptable for the phase because the packet-construction blockers were already removed in Phase 12, but it means Phase 13 did not produce a stronger package surface by itself.

Replay quality is the clearest Phase 13 win. The final replay bundle is `green`, `ready_for_pilot = true`, and carries no replay-level quality flags. The final `replay_inspection.json` also shows the `decision_prior_card` stage at `green` with `supporting_route_state_count = 4` and `held_out_pass_rate = 1.0`, which is a real downstream improvement over the cycle-1 surface rather than a reporting-only change. The remaining replay defects are now nonblocking `l2` issues (`l2_comparator_sparse`, `l2_expected_slot_missing`, `l2_relation_stitch_missing`) rather than reviewer or prior-card gating failures.

Prior-review quality improved, but the improvement is conditional. `iter-01` proves that reviewer metadata alone is enough to clear replay `reviewer_missing`, but it still yields `prior_candidate_count = 0`. The final run only unlocks a reusable prior after the explicit fallback merge collapses the three same-scope singleton support clusters into one deterministic fallback cluster. That produces one reviewed green prior and two reviewed yellow anti-pattern candidates, but the anti-patterns still carry `missing_counterexamples`, so this is a bounded recovery path, not a globally strong prior library.

Export quality improved less than prior review. The final export summary records `accepted_prior_count = 1`, which is better than cycle 1, but `selected_prior_ids` still stay empty and `quality_flags` still include `weak_prior_support`. The export summary is explicit about why: the accepted prior does not support the primary route state, so the decision episode cannot actually select it. In practice, that means Phase 13 improved replay and prior availability more than final export usability.

## Next-Cycle Recommendation

The formal prioritization handoff at `docs/replay/reports/phase13-next-cycle-prioritization.md` ranks `packet_construction` first, because the bounded slice still carries `yellow_route_state_present` and the replay `l2` delta stays flat even after the replay/prior improvements in Phase 13. That recommendation is reasonable on the repo's current taxonomy, but it should be interpreted narrowly: the next cycle should deepen packet-construction and relation-assembly quality on the same bounded slice rather than reopening the already-fixed Phase 12 packet-membership problems.

Within that packet-first framing, the two most important follow-ups are:

- reduce the residual package / relation-assembly weakness behind `yellow_route_state_present` and the flat replay `l2` surface
- make the accepted prior(s) apply directly to the primary route state so `selected_prior_ids` can stop staying empty once the packet-facing evidence is stronger

In short: Phase 13 proved the bounded slice can move downstream when reviewer metadata and cluster strategy are addressed, but the structured next-cycle handoff still points back to packet-quality / relation-assembly work because that is the highest remaining first-class blocker on the current taxonomy.

## Verdict

Phase 13 achieved real bounded movement against the Phase 12 cycle-1 baseline, but that movement is uneven across stages. The phase clearly solved the reviewer-path blocker and removed the replay `decision_prior_card` failure. It also demonstrated that a bounded fallback cluster can unlock one accepted prior on the same frozen packet. However, package quality stayed flat, export still ends with `weak_prior_support`, and no prior is yet selected into the final decision episode.

The honest closeout is therefore: **Phase 13 improved replay readiness and prior availability, but it did not yet convert that progress into training-ready export quality.** That is still a successful repeatability phase, because the final iteration beats cycle 1 on the downstream blocker surface without widening scope, and it leaves the next cycle with a much sharper target than Phase 12 had.
