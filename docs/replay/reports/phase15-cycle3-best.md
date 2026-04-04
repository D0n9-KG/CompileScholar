# Phase 15 Cycle 3 Best-Cycle Review

## Phase 14 Baseline

Phase 15 compares directly against the completed reviewed Phase 14 best cycle at `tmp/phase14_cycle2_optimization/cycle2-best/`.

Relative to that baseline:

- Phase 14 `cycle2-best` still had `because_now = null` and `why_not_before = null`
- Phase 14 `cycle2-best` still left `recommended_route_state_id` and `route_advantage_summary` empty
- Phase 14 `cycle2-best` accepted zero priors, selected zero priors, and did not close any reviewed prior knowledge into explicit exclusion records
- Phase 14 therefore remained `reviewed` / `needs_revision` and was the best packaging surface rather than a genuinely useful scientific-thinking training example

Phase 15 keeps the same bounded packet and the same canonical rerun surface, so the quality movement below is a direct comparison on the same slice rather than a scope change.

## Candidate Cycle Log

### `phase15-default-01`

- Command: `cd backend; .\.venv\Scripts\python.exe scripts\run_phase13_cycle_refinement.py --iteration-label phase15-default-01 --output-root ..\tmp\phase15_cycle3_consolidation --baseline-replay-bundle ..\tmp\phase14_cycle2_optimization\cycle2-best\replay_bundle --baseline-export-bundle ..\tmp\phase14_cycle2_optimization\cycle2-best\export_bundle --report-md ..\docs\replay\reports\phase15-cycle3-default-01.md --reviewer reviewer-1 --reviewer reviewer-2`
- Result: replay stayed `green`, export stayed `yellow`, and the new training-facing artifact now populated explicit why-now prose plus a grounded route-comparison conclusion.
- Review outcome: materially better than Phase 14 on timing and route-comparison clarity, but it still carried zero accepted priors and therefore did not exercise the new prior-closure surface in real output.

### `phase15-fallback-01`

- Command: `cd backend; .\.venv\Scripts\python.exe scripts\run_phase13_cycle_refinement.py --iteration-label phase15-fallback-01 --output-root ..\tmp\phase15_cycle3_consolidation --baseline-replay-bundle ..\tmp\phase14_cycle2_optimization\cycle2-best\replay_bundle --baseline-export-bundle ..\tmp\phase14_cycle2_optimization\cycle2-best\export_bundle --report-md ..\docs\replay\reports\phase15-cycle3-fallback-01.md --reviewer reviewer-1 --reviewer reviewer-2 --allow-scope-fallback-merge`
- Result: preserved the same timing and route-comparison gains as the default run while recovering one reviewed accepted fallback-cluster prior.
- Review outcome: stronger than the default run because the export now demonstrates the intended Plan 02 closure behavior, carrying the accepted prior into `accepted_prior_ids` and preserving a route-mismatch explanation in `accepted_but_unselected_priors`.

### `cycle3-best`

- Command: `cd backend; .\.venv\Scripts\python.exe scripts\run_phase13_cycle_refinement.py --iteration-label cycle3-best --output-root ..\tmp\phase15_cycle3_consolidation --baseline-replay-bundle ..\tmp\phase14_cycle2_optimization\cycle2-best\replay_bundle --baseline-export-bundle ..\tmp\phase14_cycle2_optimization\cycle2-best\export_bundle --report-md ..\docs\replay\reports\phase15-cycle3-best.md --reviewer reviewer-1 --reviewer reviewer-2 --allow-scope-fallback-merge`
- Result: reproduced the fallback candidate under the canonical final Phase 15 label and wrote the reviewed artifact family to `tmp/phase15_cycle3_consolidation/cycle3-best/`.
- Review outcome: chosen as the canonical best cycle because it preserves the newly grounded why-now and route-comparison sections and is the only Phase 15 lever that closes reviewed prior knowledge honestly into the export surface.

## Lever Comparison

- Default lever: cleaner on paper because there are no accepted priors to reconcile, but it leaves Phase 15 without any real prior-closure evidence and therefore undershoots the plan's export-truth goal.
- Fallback lever: recovers one reviewed accepted prior and shows that the export can now keep route truth honest by recording an explicit `route_state_not_supported` exclusion instead of silently forcing the prior into `selected_prior_ids`.
- Shared gain across both levers: the training-facing artifact now includes explicit `because_now`, `why_not_before`, `recommended_route_state_id`, and `route_advantage_summary`, which were all missing in the reviewed Phase 14 best cycle.
- Chosen lever: `allow_scope_fallback_merge`, because it is the only candidate that both beats the Phase 14 narrative defects and demonstrates reviewed prior closure on the live bounded slice.

## Best-Cycle Selection

Chosen best cycle: `cycle3-best`

Why this cycle:

- it matches the stronger `phase15-fallback-01` content surface under the canonical final label for the phase
- it preserves the new explicit why-now prose and grounded route-comparison summary that Phase 14 lacked
- it carries the reviewed accepted prior into the audited export surface and leaves the route mismatch machine-readable instead of implicit

Why this is better than the reviewed Phase 14 best cycle:

- the Phase 14 best cycle was still missing timing prose and route-advantage explanation
- `cycle3-best` now names the winning route, explains why it wins, and states why the route is actionable now rather than earlier
- `cycle3-best` also introduces the first explicit prior-closure record on the reviewed bounded slice, even though that prior still remains unselected for the exported primary route

## Section Review

- Evidence pack: accepted. The training view still exposes `7` visible papers, `7` visible traces, `12` evidence refs, and one cutoff exclusion, so the bounded evidence basis remains explicit and self-contained.
- Route synthesis: candidate. The primary route state remains `green`, but the dominant-method inventory is still long and somewhat noisy, which keeps the scientific story less crisp than it should be.
- Why-now: accepted. `because_now` and `why_not_before` are now explicit, specific, and reviewable, which directly closes one of the biggest Phase 14 defects.
- Route comparison: accepted. The artifact now carries `recommended_route_state_id` and a concrete `route_advantage_summary`, making the route choice materially more teachable than the Phase 14 best cycle.
- Priors / anti-patterns: candidate. The accepted fallback prior is real progress, and the route mismatch is now explicit and machine-readable, but the final export still selects no priors and no anti-patterns.
- Minimal attack path: accepted. The section remains concrete and bounded with `2` prerequisite steps, `4` required resources, `4` required measurements, and `3` expected checkpoints.
- Final decision: candidate. The route-comparison grounding now supports the primary decision better than Phase 14, but the final answer still lacks selected prior support and therefore remains slightly more confident than ideal.
- Review labels: accepted. The label surface keeps the full verdict package together and now corresponds to a materially stronger reasoning artifact than the Phase 14 baseline.

## Prior / Anti-Pattern Closure

- Accepted prior ids: `["prior:data_driven_constitutive_and_multiscale_computational_mechanics_scope_fallback_cluster"]`
- Selected prior ids: `[]`
- Accepted but unselected prior ids: `["prior:data_driven_constitutive_and_multiscale_computational_mechanics_scope_fallback_cluster"]`
- Prior closure verdict: the reviewed accepted fallback prior does not support the exported primary route state directly, so `selected_prior_ids` correctly stays empty and the export records the mismatch explicitly through `accepted_but_unselected_priors`.
- Anti-pattern closure verdict: no anti-pattern ids were accepted in this cycle, so the anti-pattern surface remains empty but internally consistent.

This is the first bounded cycle where reviewed prior knowledge closes into the export surface honestly enough to be inspected and taught from, even though the final selected-prior surface is still empty.

## Residual Defects

- `weak_prior_support` still remains on the export surface.
- `selected_prior_ids` is still empty, so the prior closure is explicit but not yet fully resolved into a reusable selected prior.
- Package validation still carries `yellow_route_state_present`.
- Route synthesis still includes a noisy dominant-method inventory.

## Accepted-Cycle Streak

- `accepted_cycle_streak = 1`
- This is the first Phase 15 cycle judged genuinely useful as scientific-thinking training data.
- The streak is only `1`, so this phase does not claim full stability. Phase 16 must verify that a second accepted cycle can reproduce the same quality bar.

## Verdict

`cycle3-best` is genuinely useful as scientific-thinking training data.

Final human review verdict:

- review status: `reviewed`
- training acceptance verdict: `accepted`

Why this verdict is now positive:

- the artifact no longer leaves the timing judgment implicit
- the route comparison now explains why the chosen route wins
- reviewed prior knowledge now closes honestly into the export surface instead of disappearing silently
- the bundle remains bounded, auditable, and self-contained enough for direct training review

Why this still is not a stability claim:

- the export still carries `weak_prior_support`
- the accepted prior remains unselected for the exported route
- only one accepted cycle has been demonstrated so far
