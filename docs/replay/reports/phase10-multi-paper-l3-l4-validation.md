# Phase 10 Multi-Paper Validation Report

- Packet id: `phase9_comp_mech_2021_packet_01`
- Cutoff year: `2021`
- Baseline replay bundle: `C:\Users\D0n9\Desktop\LogicKG\tmp\phase3_route_state_package\replay_with_package`
- Baseline export bundle: `C:\Users\D0n9\Desktop\LogicKG\tmp\phase6_decision_episode_audit_export`

## Stage Comparison

### Package
- Current quality: `red` vs baseline `green`
- Ready for replay: `False` vs baseline `True`
- Quality flags: `support_cluster_too_small, alternative_scope_not_distinct, yellow_route_state_present`
- Role counts: `{'primary': 0, 'support': 1, 'alternative': 1, 'held_out': 1}`

### Replay
- Current quality: `red` vs baseline `green`
- Ready for pilot: `False` vs baseline `True`
- Failure record count: `5` vs baseline `4`
- Failure counts by stage: `{'route_comparison': 1, 'route_state': 2, 'decision_prior_card': 2}`
- Quality flags: `support_cluster_too_small, reviewer_missing`

### Prior Review
- Prior candidates: `0`
- Anti-pattern candidates: `0`
- Accepted prior ids: `0` vs baseline accepted prior count `0`
- Accepted anti-pattern ids: `0` vs baseline accepted anti-pattern count `5`
- Quality flag counts: `{}`

### Export
- Current quality: `yellow` vs baseline `yellow`
- Ready for training: `False` vs baseline `False`
- Ready for eval: `True` vs baseline `True`
- Selected prior count: `0` vs baseline `0`
- Selected anti-pattern count: `0` vs baseline `5`
- Visibility buckets: `{'visible_input_refs': 28, 'audit_only_refs': 26, 'label_eval_only_refs': 0}`
- Quality flags: `weak_prior_support`

## Blocker Queue

### Package Validation
- `support_cluster_too_small`: Package validation reports `support_cluster_too_small`. (current=`True`, baseline=`False`, vs_baseline=`new`)
- `alternative_scope_not_distinct`: Package validation reports `alternative_scope_not_distinct`. (current=`True`, baseline=`False`, vs_baseline=`new`)
- `yellow_route_state_present`: Package validation reports `yellow_route_state_present`. (current=`True`, baseline=`False`, vs_baseline=`new`)

### Replay
- `support_cluster_too_small`: Replay summary reports `support_cluster_too_small`. (current=`True`, baseline=`False`, vs_baseline=`new`)
- `reviewer_missing`: Replay summary reports `reviewer_missing`. (current=`True`, baseline=`False`, vs_baseline=`new`)
- `failure_stage:route_comparison`: Replay recorded 1 failure record(s) at the `route_comparison` stage. (current=`1`, baseline=`1`, vs_baseline=`unchanged`)
- `failure_stage:route_state`: Replay recorded 2 failure record(s) at the `route_state` stage. (current=`2`, baseline=`3`, vs_baseline=`lower`)
- `failure_stage:decision_prior_card`: Replay recorded 2 failure record(s) at the `decision_prior_card` stage. (current=`2`, baseline=`0`, vs_baseline=`higher`)

### Prior Induction
- `no_prior_candidates`: Prior induction produced no DecisionPriorCard candidates for review. (current=`0`, baseline=`0`, vs_baseline=`unchanged`)
- `accepted_prior_ids_empty`: Prior review accepted no prior ids, so export must keep selected_prior_ids empty. (current=`0`, baseline=`0`, vs_baseline=`unchanged`)

### Export
- `weak_prior_support`: Export summary reports `weak_prior_support`. (current=`True`, baseline=`True`, vs_baseline=`carried_forward`)
- `not_ready_for_training`: Export remains unavailable for training. (current=`False`, baseline=`False`, vs_baseline=`unchanged`)

## Source Of Truth

- This report is rendered from `comparison_summary.json`, not from bundle prose.
