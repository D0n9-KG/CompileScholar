# Phase 10 Multi-Paper Validation Report

- Packet id: `phase9_comp_mech_2021_packet_01`
- Cutoff year: `2021`
- Baseline replay bundle: `C:\Users\D0n9\Desktop\LogicKG\tmp\phase12_direct_fix_cycle\cycle1\replay_bundle`
- Baseline export bundle: `C:\Users\D0n9\Desktop\LogicKG\tmp\phase12_direct_fix_cycle\cycle1\export_bundle`

## Stage Comparison

### Package
- Current quality: `yellow` vs baseline `yellow`
- Ready for replay: `True` vs baseline `True`
- Quality flags: `yellow_route_state_present`
- Role counts: `{'primary': 0, 'support': 3, 'alternative': 1, 'held_out': 1}`

### Replay
- Current quality: `green` vs baseline `red`
- Ready for pilot: `True` vs baseline `False`
- Failure record count: `3` vs baseline `4`
- Failure counts by stage: `{'route_comparison': 1, 'route_state': 2}`
- Quality flags: `none`

### Prior Review
- Prior candidates: `0`
- Anti-pattern candidates: `0`
- Accepted prior ids: `0` vs baseline accepted prior count `0`
- Accepted anti-pattern ids: `0` vs baseline accepted anti-pattern count `0`
- Quality flag counts: `{}`

### Export
- Current quality: `yellow` vs baseline `yellow`
- Ready for training: `False` vs baseline `False`
- Ready for eval: `True` vs baseline `True`
- Selected prior count: `0` vs baseline `0`
- Selected anti-pattern count: `0` vs baseline `0`
- Accepted but unselected prior count: `0` vs baseline `0`
- Visibility buckets: `{'visible_input_refs': 28, 'audit_only_refs': 26, 'label_eval_only_refs': 0}`
- Training view sections: `['evidence_pack', 'final_decision', 'minimal_attack_path', 'priors_antipatterns', 'review_labels', 'route_comparison', 'route_synthesis', 'why_now']`
- Training view file: `C:\Users\D0n9\Desktop\LogicKG\tmp\phase14_cycle2_optimization\phase14-candidate-01\export_bundle\outputs\training_view.json`
- Best-cycle selection file: `C:\Users\D0n9\Desktop\LogicKG\tmp\phase14_cycle2_optimization\phase14-candidate-01\export_bundle\best_cycle_selection.json`
- Current recommendation: `not stated`
- Recommendation evidence refs: `['training_view:C:\\Users\\D0n9\\Desktop\\LogicKG\\tmp\\phase14_cycle2_optimization\\phase14-candidate-01\\export_bundle\\outputs\\training_view.json', 'export_summary:C:\\Users\\D0n9\\Desktop\\LogicKG\\tmp\\phase14_cycle2_optimization\\phase14-candidate-01\\export_bundle\\export_summary.json', 'export_inspection:C:\\Users\\D0n9\\Desktop\\LogicKG\\tmp\\phase14_cycle2_optimization\\phase14-candidate-01\\export_bundle\\export_inspection.json', 'review_bundle:candidate_review_summary:C:\\Users\\D0n9\\Desktop\\LogicKG\\tmp\\phase14_cycle2_optimization\\phase14-candidate-01\\prior_review_bundle\\candidate_review_summary.json']`
- Quality flags: `weak_prior_support`

## Blocker Queue

### Package Validation
- `yellow_route_state_present`: Package validation reports `yellow_route_state_present`. (current=`True`, baseline=`True`, vs_baseline=`carried_forward`)

### Replay
- `failure_stage:route_comparison`: Replay recorded 1 failure record(s) at the `route_comparison` stage. (current=`1`, baseline=`1`, vs_baseline=`unchanged`)
- `failure_stage:route_state`: Replay recorded 2 failure record(s) at the `route_state` stage. (current=`2`, baseline=`2`, vs_baseline=`unchanged`)

### Prior Induction
- `no_prior_candidates`: Prior induction produced no DecisionPriorCard candidates for review. (current=`0`, baseline=`0`, vs_baseline=`unchanged`)
- `accepted_prior_ids_empty`: Prior review accepted no prior ids, so export must keep selected_prior_ids empty. (current=`0`, baseline=`0`, vs_baseline=`unchanged`)

### Export
- `weak_prior_support`: Export summary reports `weak_prior_support`. (current=`True`, baseline=`True`, vs_baseline=`carried_forward`)
- `not_ready_for_training`: Export remains unavailable for training. (current=`False`, baseline=`False`, vs_baseline=`unchanged`)

## Source Of Truth

- This report is rendered from `comparison_summary.json`, not from bundle prose.
