# Phase 15 Verification

## Commands Run

- `cd backend; .\.venv\Scripts\python.exe scripts\run_phase13_cycle_refinement.py --iteration-label phase15-default-01 --output-root ..\tmp\phase15_cycle3_consolidation --baseline-replay-bundle ..\tmp\phase14_cycle2_optimization\cycle2-best\replay_bundle --baseline-export-bundle ..\tmp\phase14_cycle2_optimization\cycle2-best\export_bundle --report-md ..\docs\replay\reports\phase15-cycle3-default-01.md --reviewer reviewer-1 --reviewer reviewer-2`
- `cd backend; .\.venv\Scripts\python.exe scripts\run_phase13_cycle_refinement.py --iteration-label phase15-fallback-01 --output-root ..\tmp\phase15_cycle3_consolidation --baseline-replay-bundle ..\tmp\phase14_cycle2_optimization\cycle2-best\replay_bundle --baseline-export-bundle ..\tmp\phase14_cycle2_optimization\cycle2-best\export_bundle --report-md ..\docs\replay\reports\phase15-cycle3-fallback-01.md --reviewer reviewer-1 --reviewer reviewer-2 --allow-scope-fallback-merge`
- `cd backend; .\.venv\Scripts\python.exe scripts\run_phase13_cycle_refinement.py --iteration-label cycle3-best --output-root ..\tmp\phase15_cycle3_consolidation --baseline-replay-bundle ..\tmp\phase14_cycle2_optimization\cycle2-best\replay_bundle --baseline-export-bundle ..\tmp\phase14_cycle2_optimization\cycle2-best\export_bundle --report-md ..\docs\replay\reports\phase15-cycle3-best.md --reviewer reviewer-1 --reviewer reviewer-2 --allow-scope-fallback-merge`

## Candidate Roots

- `tmp/phase15_cycle3_consolidation/phase15-default-01/`
- `tmp/phase15_cycle3_consolidation/phase15-fallback-01/`
- `tmp/phase15_cycle3_consolidation/cycle3-best/`

Verified canonical artifacts under `cycle3-best/export_bundle/`:
`outputs/decision_episode.json`, `outputs/training_view.json`, `export_summary.json`, `export_inspection.json`, and `best_cycle_selection.json`

## Lever Comparison

- `phase15-default-01` beats the Phase 14 baseline on timing and route-comparison clarity, but it recovers no reviewed priors.
- `phase15-fallback-01` preserves the same timing and route-comparison gains while recovering one reviewed accepted prior and recording an explicit route-backed exclusion because the prior does not support the exported primary route.
- Chosen lever: `allow_scope_fallback_merge`
- Selection reason: it is the only compared candidate that both fixes the major Phase 14 reasoning-surface defects and demonstrates real reviewed prior closure in the final export family.

## Best Cycle

- Chosen best cycle: `cycle3-best`
- Baseline replay bundle: `C:\Users\D0n9\Desktop\LogicKG\tmp\phase14_cycle2_optimization\cycle2-best\replay_bundle`
- Baseline export bundle: `C:\Users\D0n9\Desktop\LogicKG\tmp\phase14_cycle2_optimization\cycle2-best\export_bundle`
- Best-cycle report: `docs/replay/reports/phase15-cycle3-best.md`
- accepted_cycle_streak = 1

## Section Review Observations

- Evidence pack: still self-contained and reviewable with the same bounded artifact coverage as Phase 14.
- Route synthesis: primary route remains `green`, but the dominant-method list is still noisy enough to keep this section short of ideal.
- Why-now: Phase 15 closes the missing-prose defect by filling both `because_now` and `why_not_before`.
- Route comparison: Phase 15 closes the missing-recommendation defect by filling both `recommended_route_state_id` and `route_advantage_summary`.
- Priors / anti-patterns: fallback recovery yields one reviewed accepted prior, and the export now preserves the route mismatch explicitly through `accepted_but_unselected_priors`.
- Final decision: still lands on `primary` with `confidence = 0.95`; the grounded route-comparison surface makes that easier to justify than in Phase 14, though the absence of selected priors still leaves some residual caution.

## Manual Review Verdict

- Overall review status: `reviewed`
- training acceptance verdict: `accepted`
- Rationale: `cycle3-best` is the first bounded cycle whose evidence pack, why-now reasoning, route comparison, and explicit prior-closure behavior together make it genuinely useful as scientific-thinking training data, even though it still carries residual prior-support defects.
- Residual defects:
  - `weak_prior_support`
  - `selected_prior_ids_empty`
  - `yellow_route_state_present`
  - `dominant_method_inventory_noisy`

## Recommendation

- primary recommendation: `packet_construction`
- Prioritization summary: `tmp/phase15_cycle3_consolidation/final-prioritization/outputs/prioritization_summary.json`
- Prioritization report: `docs/replay/reports/phase15-next-cycle-prioritization.md`
- Supporting evidence: `packet_construction` remains first because `yellow_route_state_present` still carries forward from the Phase 14 baseline while replay L2 delta stays flat; the best-cycle evidence now shows stronger timing and route-comparison reasoning, but the remaining blockers still point upstream to bounded packet quality rather than to export-only packaging.
