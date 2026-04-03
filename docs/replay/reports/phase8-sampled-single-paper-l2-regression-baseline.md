# Phase 8 Sampled Single-Paper L2 Regression

Built from runtime bundle `C:\Users\D0n9\Desktop\LogicKG\tmp\phase8_sampled_single_paper_l2\baseline-cycle-01` and written to `C:\Users\D0n9\Desktop\LogicKG\docs\replay\reports\phase8-sampled-single-paper-l2-regression-baseline.md`.

## Iteration Inputs

- Iteration label: `baseline-cycle-01`
- Sampling bundle: `C:\Users\D0n9\Desktop\LogicKG\tmp\phase7_corpus_sampling_baseline`
- Sampling manifest: `C:\Users\D0n9\Desktop\LogicKG\tmp\phase7_corpus_sampling_baseline\bundle_manifest.json`
- Output directory: `C:\Users\D0n9\Desktop\LogicKG\tmp\phase8_sampled_single_paper_l2\baseline-cycle-01`
- Comparison mode: baseline-only
- Graph writes enabled: `false`

## Fixed Regression Outcomes

This run is baseline-only, so fixed-set outcomes are classified against the current baseline surface.
- Fixed executed count: `10`
- `stable_pass`: 3
- `recurring_failure`: 7
- `new_regression`: 0
- `improved`: 0
- `availability_only`: 0

## Random Exploration Outcomes

This run is baseline-only, so random failures are treated as new edge cases unless they are availability-only.
- Random executed count: `5`
- `new_edge_case`: 2
- `repeated_random_failure`: 0
- `random_improved`: 0
- `stable_random_pass`: 3
- `availability_only`: 0

## Owner Buckets

- `relation_assembly`: fixed=7, random=2, exemplars=1000, 1001, 1005, 1017, 1023
- `slot_recovery`: fixed=6, random=0, exemplars=1001, 1005, 1017, 1023, 1004
- `metadata_repair`: fixed=0, random=1, exemplars=1107
- `reference_recovery`: fixed=0, random=1, exemplars=1107

## Recommended L2 Queue

1. `relation_assembly` - fixed=7, random=2, exemplars=1000, 1001, 1005, 1017, 1023
2. `slot_recovery` - fixed=6, random=0, exemplars=1001, 1005, 1017, 1023, 1004
3. `metadata_repair` - fixed=0, random=1, exemplars=1107
4. `reference_recovery` - fixed=0, random=1, exemplars=1107

## Execution Limits

- Availability-only issues remained separate from regression counts: `0` issue(s).
- This report only covers sampled single-paper L2 behavior; it does not make packet-level L3/L4 claims.
- Source-path availability still depends on the Phase 7 bundle and the current machine being able to reach each preferred source path.
- This run did not compare against an earlier Phase 8 bundle.
