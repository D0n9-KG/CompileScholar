# Phase 14 Verification

## Commands Run

- `cd backend; .\.venv\Scripts\python.exe scripts\run_phase13_cycle_refinement.py --iteration-label phase14-candidate-01 --output-root ..\tmp\phase14_cycle2_optimization --report-md ..\docs\replay\reports\phase14-cycle2-optimization-candidate-01.md --reviewer reviewer-1 --reviewer reviewer-2`
- `cd backend; .\.venv\Scripts\python.exe scripts\run_phase13_cycle_refinement.py --iteration-label cycle2-best --output-root ..\tmp\phase14_cycle2_optimization --report-md ..\docs\replay\reports\phase14-cycle2-optimization-best.md --reviewer reviewer-1 --reviewer reviewer-2`
- `cd backend; .\.venv\Scripts\python.exe scripts\run_phase11_iteration_prioritization.py --phase8-summary ..\tmp\phase8_sampled_single_paper_l2\baseline-cycle-01\comparison_summary.json --phase8-inspection ..\tmp\phase8_sampled_single_paper_l2\baseline-cycle-01\comparison_inspection.json --phase10-summary ..\tmp\phase14_cycle2_optimization\cycle2-best\comparison_summary.json --phase10-verification ..\.planning\phases\14-cycle-2-optimization-and-review\14-VERIFICATION.md --phase10-report ..\docs\replay\reports\phase14-cycle2-optimization-best.md --output-dir ..\tmp\phase14_cycle2_optimization\final-prioritization --report-md ..\docs\replay\reports\phase14-next-cycle-prioritization.md`

## Candidate Roots

- `tmp/phase14_cycle2_optimization/phase14-candidate-01/`
- `tmp/phase14_cycle2_optimization/cycle2-best/`
- Verified additive Phase 14 artifacts under `cycle2-best/export_bundle/`:
  `outputs/decision_episode.json`, `outputs/training_view.json`, `export_summary.json`, `export_inspection.json`, and `best_cycle_selection.json`

## Best Cycle

- Chosen best cycle: `cycle2-best`
- Selection reason: it reproduces the new Phase 14 training-facing bundle contract cleanly under the default bounded rerun path and is the canonical final label for the phase
- Important caveat: `cycle2-best` is the best Phase 14 carrier of the new export contract, but it does **not** beat the completed Phase 13 `cycle2-final` content surface on prior availability; Phase 13 `cycle2-final` still accepted one fallback-cluster prior, while both Phase 14 runs without fallback merge accept zero priors

## Section Review Observations

- Evidence pack: pass for structure, not for final acceptance. `training_view.json` exposes `7` visible papers, `7` visible traces, `12` evidence refs, and one explicit cutoff exclusion, so a reviewer can inspect the bounded slice without cross-file reconstruction.
- Route synthesis: mixed. The primary route-state payload is `green` and the candidate versus alternative questions are explicit, but the dominant-method and bottleneck label lists are still noisy and overly long, which makes the scientific story harder to teach directly from the artifact.
- Why-now: mixed. The `why_now_label` is `almost_now` and the case is structurally present, but the exported `because_now` and `why_not_before` prose remain `null`, so the section still needs synthesis-level clarification.
- Route comparison: failing the training-usefulness bar. The case remains `yellow`, and both `recommended_route_state_id` and `route_advantage_summary` are missing, so the artifact does not yet explain why the chosen route wins against the nearby alternative strongly enough for training.
- Priors / anti-patterns: failing the training-usefulness bar. `prior_candidate_count = 0`, `accepted_prior_ids = []`, `selected_prior_ids = []`, and export still carries `weak_prior_support`, so the bundle cannot yet teach reusable decision priors from this slice.
- Minimal attack path: useful but still dependent on upstream fixes. The section lists concrete prerequisite steps, resources, measurements, and checkpoints, which makes it reviewable, but those checkpoints are not reinforced by a selected prior or strong route-comparison justification.
- Final decision: mixed. The export still lands on `final_choice = primary` with `confidence = 0.95`, but that confidence currently overstates the strength of the missing prior and route-comparison support.
- Review labels: structurally successful. The bundle now exposes `quality_tier`, `quality_flags`, `why_now_label`, `final_choice`, `training_acceptance_verdict`, and `review_status` in one place, which satisfies the Phase 14 review-surface goal even though the content verdict is still negative.

## Manual Review Verdict

- Overall review status: `reviewed`
- Overall training acceptance verdict: `needs_revision`
- Rationale: Phase 14 successfully added the self-contained training-view and best-cycle-selection artifacts, but the reviewed bounded content is still not genuinely useful as scientific-thinking training data because it lacks reusable prior support and a persuasive route-comparison explanation.
- Residual defects:
  - `weak_prior_support` remains on the export surface
  - prior review produces zero candidates and zero accepted priors on the default bounded rerun path
  - route comparison remains `yellow` without a recommended route id or advantage summary
  - package validation still carries `yellow_route_state_present`
  - replay failure counts at `route_state` and `route_comparison` stay flat relative to the completed Phase 13 final run

## Recommendation

- Confirmed primary recommendation: `packet_construction`
- Source: `tmp/phase14_cycle2_optimization/final-prioritization/outputs/prioritization_summary.json`
- Supporting evidence: `yellow_route_state_present` stayed flat, replay `l2` delta stayed flat, prior review remained empty on the default bounded rerun path, and the reviewed best-cycle evidence refs now point directly to `training_view.json`, `export_summary.json`, `export_inspection.json`, and `candidate_review_summary.json`
