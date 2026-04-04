# Phase 16: Stability Verification And Handoff - Research

**Researched:** 2026-04-04
**Domain:** reproduce the accepted Phase 15 bounded cycle on the same slice, then publish stable multi-view training artifacts plus a final bounded dataset and residual-risk handoff
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Phase 16 must stay on the same bounded computational-mechanics slice and keep using the current `package -> replay -> prior-review -> export` execution surface.
- The accepted Phase 15 `cycle3-best` review is the comparison anchor; stability must be judged by reproducing that accepted bar on the same bounded slice.
- No new papers should be added for this phase. Stability is about repeatability on the same slice, not wider generalization.
- Direct manual review of generated reasoning artifacts remains the real acceptance gate for `STAB-02` and `STAB-03`.
- Phase 16 should preserve the current export family (`decision_episode.json`, `training_view.json`, `export_summary.json`, `export_inspection.json`, `best_cycle_selection.json`) and add task-specific training views as additive outputs.
- Runtime cycle artifacts and reviewed milestone-closeout truth should remain separate; final stability conclusions should be published through dedicated handoff artifacts rather than by mutating runtime bundle truth.
- The final handoff must explain residual risk honestly; zero-defect perfection is not required.

### The Agent's Discretion
- Exact file names and directory layout for the additive task-specific training views and final handoff bundle.
- Whether accepted-cycle streak is recorded in a final payload, verification note, or both, as long as the evidence is explicit and machine-readable.
- Exact schema names for final dataset manifests and handoff payloads, as long as they clearly separate runtime evidence from reviewed milestone closeout.

### Deferred / Out Of Scope
- Adding new papers or changing the topic slice to test broader generalization
- Replacing the current rerun surface with a new execution framework
- UI or operator-surface work for dataset review
- Open-ended scaling beyond the bounded dataset bundle promised for `v1.2`
</user_constraints>

<research_summary>
## Summary

Phase 16 is not another open-ended optimization phase. It is a bounded closeout phase with four concrete obligations:

1. reproduce the accepted Phase 15 quality bar on the same slice and record a consecutive accepted-cycle streak of `2`,
2. preserve the existing runtime export bundle contract while adding task-specific training views for the major training tasks,
3. publish a final bounded dataset / handoff package that points at the validated later-cycle artifacts and the stable schema/readiness story,
4. close the milestone with a reviewed verification note and residual-risk handoff instead of relying on runtime summary defaults.

The codebase already contains most of the ingredients needed for this:

- the rerun surface is already stable through `backend/scripts/run_phase13_cycle_refinement.py`,
- the bundle writer in `backend/app/research_logic/replay_io.py` already produces `training_view.json`, `export_summary.json`, `export_inspection.json`, `best_cycle_selection.json`, and a manifest,
- `decision_episode_export.py` and `models.py` already define section-level review metadata,
- `best_cycle_selection.json` already carries the reviewed verdict and residual defects for Phase 15 `cycle3-best`.

The main gap is that the current system has one umbrella `training_view.json`, but no dedicated additive outputs for:

- route synthesis training view
- why-now judgment training view
- route comparison training view
- prior / anti-pattern learning training view
- final decision episode training view

There is also no existing final dataset / handoff scaffold in the runtime code. That means Phase 16 should be planned as additive output work plus one final execution/reporting wave, not as a minor docs-only pass.

No external ecosystem research is necessary for this phase. The work is repo-specific and already well-scoped by Phase 15 artifacts, existing schema specs, and current bundle-writer behavior.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Asset | Status | Purpose | Why Standard Here |
|-------|--------|---------|-------------------|
| `backend/scripts/run_phase13_cycle_refinement.py` | Existing | Canonical bounded rerun entrypoint with baseline bundle wiring and reviewer ids | Reuse it for the second accepted-cycle reproduction instead of inventing a Phase 16 runner |
| `backend/app/research_logic/phase10_multi_paper_validation.py` | Existing | Canonical package / replay / prior-review / export orchestration and comparison-summary assembly | Best place to thread new additive export outputs and stable handoff references into the bounded rerun flow |
| `backend/app/research_logic/replay_io.py` | Existing | Writes `training_view.json`, `best_cycle_selection.json`, summaries, inspections, and bundle manifests | Natural home for task-specific views, final dataset manifests, and milestone-closeout payload builders |
| `backend/app/research_logic/decision_episode_export.py` | Existing | Builds review-backed export payloads and section-level training review metadata | Must remain the audit-grade truth boundary while Phase 16 adds handoff surfaces above it |
| `backend/app/research_logic/models.py` | Existing | Defines `TrainingSectionReview`, `WhyNowCase`, `RouteComparisonCase`, `DecisionEpisode`, and related review contracts | Best place for any additive final dataset / handoff schema models if Phase 16 needs typed payloads |
| `backend/app/research_logic/iteration_prioritization.py` | Existing | Preserves next-focus recommendation ids and evidence refs | Should continue to feed the final handoff's “next optimization focus” section |

### Supporting
| Asset | Status | Purpose | When to Use |
|-------|--------|---------|-------------|
| `.planning/phases/15-cycle-3-consolidation/15-VERIFICATION.md` | Existing | Records accepted `cycle3-best`, `accepted_cycle_streak = 1`, and residual defects | Baseline truth for the Phase 16 streak target |
| `docs/replay/reports/phase15-cycle3-best.md` | Existing | Human review of the accepted bounded cycle | Direct content-quality bar that Phase 16 must reproduce |
| `docs/replay/reports/phase15-next-cycle-prioritization.md` | Existing | Current next-focus handoff (`packet_construction`) | Baseline recommendation unless the second accepted cycle changes the evidence materially |
| `tmp/phase15_cycle3_consolidation/cycle3-best/comparison_summary.json` | Existing | Machine-readable stage comparison for the accepted cycle | Baseline machine evidence for Phase 16 reruns and final verification |
| `tmp/phase15_cycle3_consolidation/cycle3-best/export_bundle/best_cycle_selection.json` | Existing | Reviewed verdict, rationale, section reviews, residual defects, recommendation evidence refs | Canonical reviewed truth for the accepted cycle |
| `tmp/phase15_cycle3_consolidation/cycle3-best/export_bundle/export_summary.json` | Existing | Runtime export posture and prior-selection state | Important because it still carries pre-review defaults; Phase 16 should not overwrite that truth |
| `tmp/phase15_cycle3_consolidation/cycle3-best/export_bundle/outputs/training_view.json` | Existing | Current umbrella training-facing artifact | Must be preserved while task-specific views are added |
| `backend/tests/test_replay_io.py` | Existing | Bundle writer coverage for `training_view.json`, `best_cycle_selection.json`, summary, inspection, manifest | Primary regression surface for additive view and final-bundle outputs |
| `backend/tests/test_phase10_multi_paper_validation.py` | Existing | End-to-end bundle-writing and comparison-summary coverage | Primary regression surface for additive export outputs in the bounded rerun flow |
| `backend/tests/test_decision_episode_export.py` | Existing | Review-backed export and section-review metadata coverage | Needed if additive task views or final bundle payloads reuse export review fields |
| `backend/tests/test_iteration_prioritization.py` | Existing | Recommendation evidence propagation and fallback handling | Needed if final handoff continues to render next-focus recommendation from machine-readable evidence |
| `backend/tests/test_phase13_cycle_refinement.py` | Existing | CLI rerun contract coverage | Needed when Phase 16 reruns must preserve baseline bundle wiring and new output refs |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| additive task-specific views | replace `training_view.json` with many files | Breaks current consumers and weakens continuity from Phase 14/15 |
| dedicated handoff payloads above runtime truth | writing reviewed verdicts back into `export_summary.json` or `training_view.json` | Blurs runtime evidence with reviewed milestone truth |
| same bounded rerun surface | new Phase 16 execution script | Adds unnecessary execution variance to a stability-proof phase |
| final bounded dataset manifest | report-only milestone closeout | Leaves `TRAIN-07` under-specified and harder to consume programmatically |
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Pattern 1: Additive Export Extension
**What:** Keep the current export bundle intact and add new files rather than replacing existing ones.  
**When to use:** For task-specific views and final handoff packaging.  
**Why recommended:** Existing tests and downstream logic already assume `training_view.json` and `best_cycle_selection.json` exist.

### Pattern 2: Runtime Truth vs Reviewed Closeout Truth
**What:** Preserve runtime bundle outputs as run-generated evidence, and publish milestone-closeout conclusions in separate reviewed payloads or reports.  
**When to use:** For accepted-cycle streak, final stability verdict, residual-risk summary, and dataset-readiness handoff.  
**Why recommended:** Phase 15 already shows that `best_cycle_selection.json` carries reviewed truth while `export_summary.json` and `training_view.json` keep runtime defaults.

### Pattern 3: Stability Proof Through Repeated Accepted Reruns
**What:** Use the same bounded slice, same canonical runner, and same reviewer framing to generate a second accepted cycle.  
**When to use:** For `STAB-02` and `STAB-03`.  
**Why recommended:** Any widened packet, new papers, or changed execution path would weaken the meaning of “stable.”

### Pattern 4: Final Dataset As A Referenced Bundle, Not A Recomputed Corpus
**What:** Assemble a final bounded dataset package by referencing validated later-cycle artifacts and their additive task views rather than by inventing a separate transformation pipeline.  
**When to use:** For `TRAIN-07`.  
**Why recommended:** The milestone promises a bounded usable dataset bundle, not a fresh ingest/export system.

### Pattern 5: Recommendation Carry-Forward With Evidence Override
**What:** Carry forward `packet_construction` by default, but allow Phase 16 review evidence to override it if the repeated accepted cycle moves the blocker profile materially.  
**When to use:** In the final handoff and next-milestone focus section.  
**Why recommended:** Keeps the final recommendation evidence-backed instead of ceremonial.

### Anti-Patterns To Avoid
- Replacing `training_view.json` instead of extending the bundle
- Treating one more rerun as “stable” without direct reviewed acceptance
- Writing milestone closeout language only in markdown with no machine-readable counterpart
- Mutating runtime bundle review fields post hoc to reflect final milestone decisions
- Expanding packet membership to make the second run look more impressive
</architecture_patterns>

<recommended_plan_slices>
## Recommended Plan Slices

### Slice 1: Additive multi-view and final-bundle scaffolding
- Extend the export / bundle-writing surface so the existing export bundle keeps `training_view.json` while also writing dedicated task-specific views.
- Add a final bounded dataset / handoff payload or manifest family that references the validated export artifacts instead of replacing them.
- Lock tests around the new additive files and manifest refs.

### Slice 2: Stability-proof and reviewed closeout plumbing
- Extend the comparison / summary / best-cycle / handoff chain so a repeated accepted cycle can be recorded as a consecutive streak.
- Keep runtime bundle truth and reviewed milestone-closeout truth separate.
- Ensure final verification and handoff payloads can explain residual risk and schema/readiness state explicitly.

### Slice 3: Phase 16 rerun, final verification, and milestone handoff
- Run at least one new bounded Phase 16 candidate cycle on the same slice against the accepted Phase 15 baseline.
- Select a canonical accepted Phase 16 cycle only if direct manual review still says the outputs are genuinely useful as scientific-thinking training data.
- Publish final verification, final dataset/handoff package, and next-milestone focus with explicit residual defects and streak length.

The natural planning split is therefore 3 plans across 3 waves:
- Wave 1: additive multi-view plus final-bundle scaffolding
- Wave 2: stability-proof / reviewed-closeout plumbing
- Wave 3: bounded rerun, final verification, and milestone handoff
</recommended_plan_slices>

<validation_architecture>
## Validation Architecture

Phase 16 should validate at five levels:

1. **additive export contract validation**
   - prove the existing `training_view.json`, `best_cycle_selection.json`, `export_summary.json`, and `export_inspection.json` still exist,
   - prove task-specific view files are added without removing current files,
   - prove new final dataset / handoff manifests include stable refs to the validated later-cycle artifacts.

2. **review-truth separation validation**
   - prove runtime export payloads remain runtime-generated,
   - prove reviewed milestone closeout information is emitted in separate reviewed payloads or reports,
   - prove accepted-cycle streak and residual-risk data are explicit and machine-readable.

3. **bounded rerun reproducibility validation**
   - prove the Phase 16 rerun uses the same bounded packet and the same rerun entrypoint as Phase 15,
   - prove baseline replay/export bundle refs point back to the accepted Phase 15 `cycle3-best` artifacts,
   - prove the new Phase 16 cycle writes a complete comparable bundle family under a dedicated `tmp/phase16_*` root.

4. **manual stability validation**
   - a reviewer must read the generated package / replay / prior / export artifacts directly and decide whether the second accepted cycle still clears the training-data bar,
   - the final verification must compare the repeated accepted cycle back to the Phase 15 accepted cycle rather than rely on metric-only deltas,
   - the final closeout must explain why the milestone is “stable enough” and where residual risk remains.

5. **handoff recommendation validation**
   - prove the final handoff includes one explicit next optimization focus,
   - prove the recommendation is grounded in current evidence refs rather than inherited blindly,
   - prove the final dataset package lists the task-specific views and canonical schema/readiness sources it depends on.

**Expected commands:**
- Quick path: `cd backend; .\\.venv\\Scripts\\python.exe -m pytest tests\\test_decision_episode_export.py tests\\test_replay_io.py tests\\test_phase10_multi_paper_validation.py tests\\test_phase13_cycle_refinement.py tests\\test_iteration_prioritization.py -q`
- Full path: `cd backend; .\\.venv\\Scripts\\python.exe -m pytest tests\\test_decision_episode_export.py tests\\test_decision_episode_export_cli.py tests\\test_replay_io.py tests\\test_phase10_multi_paper_validation.py tests\\test_phase10_multi_paper_validation_cli.py tests\\test_phase13_cycle_refinement.py tests\\test_iteration_prioritization.py tests\\test_iteration_prioritization_cli.py -q`

Manual validation remains mandatory for the final rerun, final verification note, residual-risk handoff, and bounded dataset-readiness judgment.
</validation_architecture>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Phase 16 rerun execution | a new custom Phase 16 runner | `backend/scripts/run_phase13_cycle_refinement.py` with Phase 16 output roots and explicit Phase 15 baseline overrides | Keeps the stability proof comparable |
| task-specific views | a separate parallel export pipeline | additive writers in `replay_io.py` plus existing export bundle manifests | Preserves continuity and testability |
| final dataset closeout | prose-only milestone recap | machine-readable handoff / manifest payload plus markdown verification/report | Satisfies `TRAIN-07` more cleanly |
| residual-risk communication | editing runtime summaries after review | separate reviewed final verification and handoff artifacts | Protects runtime truth |
| next-focus recommendation | hand-authored guess | existing `iteration_prioritization.py` evidence chain plus current reviewed outputs | Keeps the handoff grounded |
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: “Second run exists” gets mistaken for “stable”
**What goes wrong:** A second rerun completes, but nobody verifies that it actually clears the same direct-review bar as Phase 15.  
**Why it happens:** The milestone is near the end, so there is pressure to equate repetition with stability.  
**How to avoid:** Require direct section-level review of the repeated cycle and make the accepted-cycle streak explicit.  
**Warning signs:** Verification language says “reproduced” without an accepted verdict.

### Pitfall 2: Task-specific views fragment the export contract
**What goes wrong:** New files appear, but the existing `training_view.json` contract breaks or stops reflecting the whole artifact.  
**Why it happens:** It is tempting to split the current umbrella view into many files all at once.  
**How to avoid:** Keep `training_view.json` as the umbrella surface and add targeted views beside it.  
**Warning signs:** Tests stop asserting `training_view.json` exists, or new bundle manifests no longer point to it.

### Pitfall 3: Final handoff overwrites runtime truth
**What goes wrong:** Final reviewed milestone decisions get written back into `export_summary.json` or `training_view.json`, making it unclear what came from the run vs the review.  
**Why it happens:** The current bundle already holds review-shaped fields, so it feels natural to update them.  
**How to avoid:** Publish reviewed closeout truth in separate dedicated artifacts.  
**Warning signs:** Final docs can no longer explain why runtime payloads and reviewed verdicts differ.

### Pitfall 4: Dataset bundle becomes a vague folder copy
**What goes wrong:** The final dataset package is just a directory dump with no manifest, no schema pointers, and no explanation of which files are canonical.  
**Why it happens:** The required artifacts already exist, so the final assembly work looks deceptively trivial.  
**How to avoid:** Make the final bundle explicitly list the task views, canonical schema refs, source cycle roots, and residual-risk note.  
**Warning signs:** A downstream consumer would have to guess which files to use.
</common_pitfalls>

<open_questions>
## Open Questions

1. **Should the final bounded dataset bundle include only the repeated accepted Phase 16 cycle, or both accepted cycles (Phase 15 and Phase 16) as the validated later-cycle set?**  
   Recommendation: include both as provenance, but make the Phase 16 repeated accepted cycle the primary canonical example and explicitly label Phase 15 as the first accepted predecessor.

2. **Should the final handoff payload reuse `best_cycle_selection.json` fields or define a separate closeout schema?**  
   Recommendation: define a separate closeout payload or manifest that references `best_cycle_selection.json` rather than overloading it, because Phase 16 must express milestone-level stability and dataset closure, not just best-cycle selection.

3. **Should `packet_construction` remain the next optimization focus if the second accepted cycle still carries `weak_prior_support` and `yellow_route_state_present`?**  
   Recommendation: yes, unless the repeated accepted cycle materially changes the reviewed blocker profile.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- `.planning/PROJECT.md`
- `.planning/REQUIREMENTS.md`
- `.planning/ROADMAP.md`
- `.planning/STATE.md`
- `.planning/phases/16-stability-verification-and-handoff/16-CONTEXT.md`
- `.planning/phases/15-cycle-3-consolidation/15-CONTEXT.md`
- `.planning/phases/15-cycle-3-consolidation/15-VERIFICATION.md`
- `docs/replay/reports/phase15-cycle3-best.md`
- `docs/replay/reports/phase15-next-cycle-prioritization.md`
- `tmp/phase15_cycle3_consolidation/cycle3-best/comparison_summary.json`
- `tmp/phase15_cycle3_consolidation/cycle3-best/export_bundle/best_cycle_selection.json`
- `tmp/phase15_cycle3_consolidation/cycle3-best/export_bundle/export_summary.json`
- `tmp/phase15_cycle3_consolidation/cycle3-best/export_bundle/export_inspection.json`
- `tmp/phase15_cycle3_consolidation/cycle3-best/export_bundle/outputs/training_view.json`
- `backend/scripts/run_phase13_cycle_refinement.py`
- `backend/app/research_logic/phase10_multi_paper_validation.py`
- `backend/app/research_logic/replay_io.py`
- `backend/app/research_logic/decision_episode_export.py`
- `backend/app/research_logic/models.py`
- `backend/app/research_logic/iteration_prioritization.py`
- `backend/tests/test_decision_episode_export.py`
- `backend/tests/test_replay_io.py`
- `backend/tests/test_phase10_multi_paper_validation.py`
- `backend/tests/test_phase13_cycle_refinement.py`
- `backend/tests/test_iteration_prioritization.py`

### Secondary (MEDIUM confidence)
- `.planning/phases/15-cycle-3-consolidation/15-RESEARCH.md`
- `.planning/phases/15-cycle-3-consolidation/15-VALIDATION.md`
- `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md`
- `docs/superpowers/specs/2026-04-01-logickg-decision-prior-and-episode-schema.md`
- `docs/superpowers/specs/2026-04-01-logickg-why-now-and-route-comparison-schema.md`

### External Research
- None required. Phase 16 planning is fully grounded in local code, prior phase artifacts, and repo-specific contracts.
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: bounded rerun reproducibility, additive export surfaces, final bounded dataset packaging, reviewed closeout truth, and residual-risk handoff
- Ecosystem: `research_logic`, rerun CLI scripts, export bundle writers, comparison-summary generation, and pytest validation
- Patterns: additive bundle extension, runtime vs reviewed truth separation, repeated accepted-cycle stability proof, final manifest-based dataset handoff
- Pitfalls: pseudo-stability claims, broken umbrella view contract, review truth bleeding into runtime payloads, vague dataset assembly

**Confidence breakdown:**
- Reusing the existing rerun surface and Phase 15 accepted baseline: HIGH
- Need for additive task-specific views rather than replacement: HIGH
- Need for a separate final dataset / handoff payload: HIGH
- Exact naming and layout of final handoff files: MEDIUM
</metadata>

---

*Phase: 16-stability-verification-and-handoff*
*Research completed: 2026-04-04*
