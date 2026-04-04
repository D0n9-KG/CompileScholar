# Phase 11 Iteration Prioritization Report

## Input Evidence

- Phase 8 summary JSON: `C:\Users\D0n9\Desktop\LogicKG\tmp\phase8_sampled_single_paper_l2\baseline-cycle-01\comparison_summary.json`
- Phase 8 inspection JSON: `C:\Users\D0n9\Desktop\LogicKG\tmp\phase8_sampled_single_paper_l2\baseline-cycle-01\comparison_inspection.json`
- Phase 10 summary JSON: `C:\Users\D0n9\Desktop\LogicKG\tmp\phase13_cycle2_refine\cycle2-final\comparison_summary.json`
- Phase 10 verification note: `C:\Users\D0n9\Desktop\LogicKG\.planning\phases\13-baseline-data-cycle-2-refine\13-VERIFICATION.md`
- Phase 10 report markdown: `C:\Users\D0n9\Desktop\LogicKG\docs\replay\reports\phase13-cycle2-refine.md`
- Phase 8 iteration label: `baseline-cycle-01`
- Packet id: `phase9_comp_mech_2021_packet_01`

## Recommendation Queue

1. `packet_construction` - Deepen bounded packet construction (score=123)
   Why now: New package-validation blockers are the first fresh regressions while replay L2 delta stays flat.
   Supporting blocker stages: package_validation, replay
   Supporting owner buckets: relation_assembly, slot_recovery
   Evidence: package blockers: yellow_route_state_present; replay l2 delta: 0; phase10 recommendation: not stated
2. `l4_aggregation` - Tune downstream L4 aggregation (score=70)
   Why now: L4 follow-up stays next because replay-level decision_prior_card failures and L3/L4 delta still rise downstream of packet issues.
   Supporting blocker stages: replay, prior_induction
   Supporting owner buckets: relation_assembly, slot_recovery
   Evidence: replay l3_l4 delta: -1; decision_prior_card regressions: 0; prior blockers: quality_flag:missing_counterexamples
3. `l2_extraction` - Run a targeted L2 extraction pass (score=65)
   Why now: Phase 8 still shows recurring L2 work, but it is supporting evidence behind the newer packet-first regression surface.
   Supporting blocker stages: phase8_owner_queue
   Supporting owner buckets: relation_assembly, slot_recovery
   Evidence: phase8 lead owners: relation_assembly, slot_recovery; fixed recurring failures: 7; replay l2 delta: 0

## Supporting Evidence

- `relation_assembly`: fixed=7, random=2, total=9
- `slot_recovery`: fixed=6, random=0, total=6
- Ranking signals: replay_l2_delta=0, replay_l3_l4_delta=-1
- Phase 10 current recommendation carried forward: `not stated`

## Missing Or Fallback Evidence

- Phase 10 JSON unavailable: `no`
- Fallback file paths: not used.
- Supplemental Phase 10 verification provenance: `C:\Users\D0n9\Desktop\LogicKG\.planning\phases\13-baseline-data-cycle-2-refine\13-VERIFICATION.md`
- Supplemental Phase 10 report provenance: `C:\Users\D0n9\Desktop\LogicKG\docs\replay\reports\phase13-cycle2-refine.md`

## Source Of Truth

- This markdown was rendered from the Phase 11 summary and inspection payloads, not from ad-hoc ranking logic.
- Primary recommendation in summary payload: `packet_construction`
- Phase 8 summary source: `C:\Users\D0n9\Desktop\LogicKG\tmp\phase8_sampled_single_paper_l2\baseline-cycle-01\comparison_summary.json`
- Phase 8 inspection source: `C:\Users\D0n9\Desktop\LogicKG\tmp\phase8_sampled_single_paper_l2\baseline-cycle-01\comparison_inspection.json`
- Phase 10 summary JSON source: `C:\Users\D0n9\Desktop\LogicKG\tmp\phase13_cycle2_refine\cycle2-final\comparison_summary.json`
- Phase 10 evidence mode: `json`
- Inspection replay L2 delta: `0`
- Inspection replay L3/L4 delta: `-1`
