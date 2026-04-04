# Phase 14: Cycle 2 Optimization And Review - Research

**Researched:** 2026-04-04
**Domain:** bounded training-facing export optimization and review-backed rerun selection on the existing multi-paper replay pipeline
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Phase 14 must stay on the existing bounded computational-mechanics slice and keep using the current `package -> replay -> prior-review -> export` execution surface.
- Candidate cycles must continue to compare against the fixed Phase 12 cycle-1 baseline while taking the completed Phase 13 review as the live blocker handoff.
- Export truth must remain review-backed. The phase must not shortcut by copying the green replay `decision_episode` directly into the export bundle.
- The chosen best cycle must publish a self-contained training-facing view inside the machine-readable export bundle rather than relying on markdown-only reconstruction.
- Structured human review and training-acceptance verdicts must live in JSON artifacts, not only in rendered reports.
- Canonical labels must be preserved together with raw source phrases for route-state and training-relevant concept fields.
- Every major content section of the final training artifact stays in optimization scope: evidence pack, route synthesis, why-now reasoning, route comparison, priors / anti-patterns, minimal attack path, final decision, and review labels.

### The Agent's Discretion
- Exact file names and enum values for the new training-facing bundle outputs, as long as audit-only and training-facing views are clearly separated.
- Whether the current prior-selection gap is improved by widening reviewed-prior applicability to the exported primary route or by carrying explicit structured exclusion rationale for accepted-but-unselected priors.
- Exact schema layout for section-level review records, best-cycle selection metadata, and recommendation evidence references.

### Deferred / Out Of Scope
- Widening the bounded packet or changing the topic slice
- Replacing the existing Phase 13 runner with a new runtime surface
- Product or UI work for review workflows
- Claiming repeatable stability across multiple good cycles; that belongs to Phases 15 and 16
</user_constraints>

<research_summary>
## Summary

Phase 14 should be planned as a three-wave optimization phase that keeps the current audited rerun loop intact while making the export bundle review-complete and training-facing. The phase should not behave like a generic "improve quality" pass. It should target the specific mismatch now visible in Phase 13:

- replay already emits a green `decision_episode` with a selected prior,
- export remains yellow because the review-backed export path still cannot justify the selected prior on the exported primary route,
- and the machine-readable bundle is still too fragmented to review as a training artifact without cross-file reconstruction.

The safest planning move is additive, not replacement-oriented:

1. extend route/export schema contracts to preserve raw phrases and carry structured review metadata,
2. add a dedicated training-facing export view plus best-cycle selection metadata alongside the existing audit bundle,
3. run bounded candidate cycles and choose one reviewed best cycle whose JSON artifacts and markdown review agree.

This keeps Phase 6's audit-truth boundary intact, satisfies the new training-data requirements, and avoids coupling the phase to one brittle "selected prior must always flow through" assumption. If accepted reviewed priors still cannot be selected for the exported primary route, Phase 14 should at minimum publish explicit machine-readable exclusion rationale so the artifact remains reviewable and the next cycle recommendation stays evidence-backed.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Asset | Status | Purpose | Why Standard Here |
|-------|--------|---------|-------------------|
| `backend/scripts/run_phase13_cycle_refinement.py` | Existing | Canonical bounded rerun entrypoint for Phase 13/14 cycles | Keeps one execution surface and already supports cycle labels, reviewers, and baseline override |
| `backend/app/research_logic/phase10_multi_paper_validation.py` | Existing | Package / replay / prior-review / export orchestration and bundle assembly | Phase 14 should extend this surface, not bypass it |
| `backend/app/research_logic/decision_episode_export.py` | Existing | Review-backed export assembly and selected-prior allowlisting | Natural place to preserve audit truth while adding training-facing structure |
| `backend/app/research_logic/replay_io.py` | Existing | Bundle manifest and machine-readable export summary / inspection writers | Natural home for new export bundle outputs and best-cycle metadata |
| `backend/app/research_logic/models.py` | Existing | Shared schema definitions for route, why-now, comparison, priors, anti-patterns, and decision episodes | Best place to add canonical label plus raw phrase fields in a durable way |
| `backend/app/research_logic/route_state_synthesizer.py` | Existing | Current route-state normalization and aggregation path | Most likely place to preserve raw source phrases alongside canonical labels |
| `backend/app/research_logic/iteration_prioritization.py` | Existing | Structured cycle recommendation and blocker ranking surface | Best place to keep recommendation evidence machine-readable and review-backed |

### Supporting
| Asset | Status | Purpose | When to Use |
|-------|--------|---------|-------------|
| `tmp/phase13_cycle2_refine/cycle2-final/export_bundle/export_summary.json` | Existing | Current export readiness and `weak_prior_support` evidence | Use as the starting blocker record |
| `tmp/phase13_cycle2_refine/cycle2-final/export_bundle/export_inspection.json` | Existing | Current audit posture and visibility buckets | Use when designing additive training-facing files |
| `tmp/phase13_cycle2_refine/cycle2-final/export_bundle/outputs/decision_episode.json` | Existing | Current audit-grade export output | Keep intact as the audited episode, do not replace it |
| `tmp/phase13_cycle2_refine/cycle2-final/replay_bundle/outputs/decision_episode.json` | Existing | Replay-time green episode with current selected prior | Use as evidence of the replay/export mismatch, not as an export shortcut |
| `docs/replay/reports/phase13-cycle2-refine.md` | Existing | Human review of the current best cycle | Use as the live blocker handoff |
| `docs/replay/reports/phase13-next-cycle-prioritization.md` | Existing | Structured next-cycle recommendation report | Reuse its evidence-backed recommendation pattern |
| `backend/tests/test_decision_episode_export.py` | Existing | Export truth and selection-allowlist coverage | Best regression anchor for new training-facing export fields |
| `backend/tests/test_replay_io.py` | Existing | Bundle-writing coverage for summary / inspection JSON | Best regression anchor for new bundle outputs |
| `backend/tests/test_historical_replay_compiler.py` | Existing | Replay-time episode and quality-flag coverage | Use to guard replay/export separation while evolving downstream fields |
| `backend/tests/test_route_state_synthesizer.py` | Existing | Route-state normalization coverage | Best regression anchor for canonical-plus-raw phrase preservation |
| `backend/tests/test_phase10_multi_paper_validation.py` | Existing | End-to-end rerun / bundle contract coverage | Best place to prove the new artifacts are emitted in bounded reruns |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| additive training-facing bundle output | replace `export_bundle/outputs/decision_episode.json` | Simpler on paper, but breaks the audit/export contract and muddies Phase 6 truth boundaries |
| structured exclusion records for accepted-but-unselected priors | silently keep empty selected-prior export fields | Hides the real review/export mismatch and weakens later prioritization |
| route-state schema preservation of raw phrases | derive provenance from normalized labels only | Loses source wording needed for training-quality review |
| section-level review records inside JSON | rely on markdown review prose | Fails `TRAIN-03` and makes best-cycle selection hard to consume programmatically |
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Pattern 1: Dual-Layer Export Output
**What:** Keep the current audit-grade export episode intact while publishing a separate self-contained training-facing JSON view under the export bundle.  
**When to use:** When the project must preserve the review-backed audited export and also provide a reviewable training artifact.  
**Why recommended:** It meets `TRAIN-01` without weakening the audit contract.

### Pattern 2: Additive Review Metadata, Not Report-Only Verdicts
**What:** Add structured fields such as `review_status`, `training_acceptance_verdict`, `reviewer_ids`, `reviewed_at`, `rationale`, `residual_defects`, and `recommendation_evidence_refs` to machine-readable artifacts.  
**When to use:** When manual review determines whether a cycle is "good enough" for scientific-thinking training data.  
**Why recommended:** It satisfies `TRAIN-03` and makes `REVIEW-02` / `REVIEW-03` evidence portable.

### Pattern 3: Canonical Label Plus Raw Phrase Preservation At The Schema Boundary
**What:** Extend route-state and training-facing concept fields to retain canonical labels together with raw source phrases.  
**When to use:** For route features, bottlenecks, conditions, capabilities, and related training-relevant concept payloads.  
**Why recommended:** It satisfies `TRAIN-02` while keeping normalization stable.

### Pattern 4: Section-Level Review Coverage
**What:** Carry review records for every major training-data section, for example `evidence_pack`, `route_synthesis`, `why_now`, `route_comparison`, `priors_antipatterns`, `minimal_attack_path`, `final_decision`, and `review_labels`.  
**When to use:** When choosing the best cycle and recording why a cycle is or is not training-usable.  
**Why recommended:** It keeps `TRAIN-06` explicit and prevents optimization from collapsing to one blocker queue.

### Pattern 5: Best-Cycle Selection As A First-Class Artifact
**What:** Publish a machine-readable best-cycle selection record that names the chosen iteration, states the verdict, captures the rationale, and cites recommendation evidence references.  
**When to use:** At phase closeout after running one or more bounded candidate cycles.  
**Why recommended:** It ties `REVIEW-03` to concrete reviewed outputs instead of markdown-only narration.

### Anti-Patterns To Avoid
- Copying the replay `decision_episode` into export because it already looks green
- Treating empty export prior selection as a formatting issue instead of a review-backed reasoning mismatch
- Preserving only normalized labels and expecting provenance to be reconstructed later
- Publishing training-acceptance judgment only in markdown review notes
- Optimizing only export summary flags while leaving evidence pack, why-now, route comparison, priors, anti-patterns, attack path, and final decision outside explicit review scope
</architecture_patterns>

<recommended_plan_slices>
## Recommended Plan Slices

### Slice 1: Schema and bundle contract groundwork
- Extend route-state and export-facing schema models to preserve canonical labels together with raw source phrases.
- Define the structured review / training-acceptance field family and section-level review shape.
- Add regression tests that lock the new schema and bundle contract before rerun work begins.

### Slice 2: Training-facing export assembly and best-cycle metadata
- Implement additive training-facing bundle outputs, for example `training_view.json` and `best_cycle_selection.json`.
- Preserve the current audited `decision_episode.json` and visibility policy while assembling the new training-facing view.
- Add structured exclusion records for accepted-but-unselected priors when export truth still cannot select them.
- Ensure recommendation evidence references and section-level review data flow into the final machine-readable bundle.

### Slice 3: Bounded reruns, review, and closeout
- Run one or more bounded candidate cycles under a fresh Phase 14 output root.
- Manually review every major training-data section for each meaningful candidate cycle.
- Select one best cycle, publish the final markdown report and machine-readable verdicts, and generate the next-cycle recommendation from reviewed evidence.

The natural planning split is therefore 3 plans across 3 waves:
- Wave 1: schema / metadata / regression groundwork
- Wave 2: export assembly and best-cycle selection artifacts
- Wave 3: bounded reruns, manual review, and final recommendation
</recommended_plan_slices>

<validation_architecture>
## Validation Architecture

Phase 14 should validate at five levels:

1. **schema preservation tests**
   - prove canonical labels and raw source phrases coexist in route-state and training-facing concept fields,
   - prove the new review metadata fields have stable machine-readable shape,
   - prove section-level review records exist for all required training-data sections.

2. **export truth and bundle contract tests**
   - prove the audit-grade export episode remains review-backed and is not replaced by replay output,
   - prove additive training-facing outputs are written into the export bundle,
   - prove accepted-but-unselected priors produce explicit structured exclusion rationale where needed.

3. **end-to-end bounded rerun tests**
   - prove the existing Phase 13 runner still emits package, replay, prior-review, export, and comparison artifacts under a Phase 14 output root,
   - prove the new machine-readable training-facing files are present in rerun outputs,
   - prove best-cycle selection and recommendation evidence files are written in a deterministic structure.

4. **manual training-quality review**
   - a reviewer must read the chosen cycle's generated reasoning artifacts and confirm whether the artifact is genuinely useful for scientific-thinking training data,
   - manual review must judge the real evidence pack, route synthesis, why-now reasoning, route comparison, priors / anti-patterns, attack path, final decision, and review labels,
   - manual review must identify residual defects and explain the next optimization focus.

5. **recommendation and handoff validation**
   - prove the next-cycle recommendation is justified by reviewed evidence from the chosen best cycle,
   - prove the best-cycle selection artifact and markdown review agree on verdict and residual risk,
   - prove the phase handoff cites structured machine-readable evidence rather than summary prose alone.

**Expected commands:**
- Quick path: `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_episode_export.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_historical_replay_compiler.py tests\test_route_state_synthesizer.py tests\test_iteration_prioritization.py -q`
- Full path: `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_decision_episode_export.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_phase10_multi_paper_validation_cli.py tests\test_historical_replay_compiler.py tests\test_route_state_synthesizer.py tests\test_iteration_prioritization.py tests\test_prior_induction.py -q`

Phase 14 does not need to prove multi-cycle stability yet. It needs to prove that at least one bounded candidate cycle is review-backed, training-facing, and self-contained enough to judge directly as training data.
</validation_architecture>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| training-facing export | a brand-new replay/export pipeline | additive files under the existing export bundle | Keeps the audited execution path intact |
| review verdict storage | markdown-only review labels | structured JSON review / acceptance metadata | Needed for `TRAIN-03` and machine-readable best-cycle selection |
| raw phrase provenance | ad hoc markdown notes or comments | schema-level `raw_source_phrases` style fields | Keeps provenance testable and reusable |
| next-cycle recommendation | one-off prose summary | `iteration_prioritization.py` style structured evidence and ranking | Keeps `REVIEW-03` explicit and comparable |
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Replay green is mistaken for export truth
**What goes wrong:** The plan treats the replay `decision_episode` as if it can be copied into export because it already carries a selected prior.  
**Why it happens:** The replay artifact looks more complete than the current export output.  
**How to avoid:** Keep the export path review-backed and additive. Compare replay and export, but never collapse them.  
**Warning signs:** The planned implementation removes or bypasses `decision_episode_export.py`.

### Pitfall 2: Training-view work ignores section coverage
**What goes wrong:** The phase optimizes one blocker surface, usually prior selection or summary flags, while evidence pack and reasoning sections remain effectively frozen.  
**Why it happens:** Those sections are already present and can look "good enough" structurally.  
**How to avoid:** Make section-level review records part of the core schema and closeout checklist.  
**Warning signs:** The plan has no explicit task that names evidence pack, why-now, route comparison, attack path, or review labels.

### Pitfall 3: Raw phrases are reconstructed too late
**What goes wrong:** The system tries to infer source wording from normalized labels after export assembly.  
**Why it happens:** Normalized labels already exist, so provenance seems derivable later.  
**How to avoid:** Preserve raw phrases at the route-state synthesis boundary and thread them forward.  
**Warning signs:** The plan only touches export rendering files and not route-state schema/builders.

### Pitfall 4: Best-cycle selection becomes a markdown-only judgment
**What goes wrong:** Human reviewers can explain the best cycle in prose, but no deterministic machine-readable artifact identifies the chosen cycle or its rationale.  
**Why it happens:** Markdown reports are already part of the workflow.  
**How to avoid:** Publish `best_cycle_selection` style JSON with verdict, rationale, and evidence references.  
**Warning signs:** The final closeout depends on reading the report narrative to know which cycle won.
</common_pitfalls>

<open_questions>
## Open Questions

1. **Should Phase 14 standardize one final selected-cycle file name in addition to per-iteration roots?**  
   Recommendation: yes. Keep per-iteration roots for auditability, but publish one machine-readable best-cycle selector so downstream phases do not guess.

2. **Should accepted reviewed priors flow into export selection in this phase, or should Phase 14 stop at explicit exclusion rationale when the route still does not justify selection?**  
   Recommendation: plan for both outcomes. Prefer real reviewed selection where justified, but require explicit structured exclusions so the training artifact stays reviewable even if full closure waits for Phase 15.

3. **Should review metadata live only at the final cycle level, or also on individual sections?**  
   Recommendation: both. Final-cycle verdict alone is too coarse for `TRAIN-06`.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- `.planning/PROJECT.md`
- `.planning/REQUIREMENTS.md`
- `.planning/ROADMAP.md`
- `.planning/STATE.md`
- `.planning/phases/14-cycle-2-optimization-and-review/14-CONTEXT.md`
- `.planning/phases/13-baseline-data-cycle-2-refine/13-CONTEXT.md`
- `.planning/phases/13-baseline-data-cycle-2-refine/13-RESEARCH.md`
- `.planning/phases/13-baseline-data-cycle-2-refine/13-VERIFICATION.md`
- `docs/replay/reports/phase13-cycle2-refine.md`
- `docs/replay/reports/phase13-next-cycle-prioritization.md`
- `tmp/phase13_cycle2_refine/cycle2-final/export_bundle/export_summary.json`
- `tmp/phase13_cycle2_refine/cycle2-final/export_bundle/export_inspection.json`
- `tmp/phase13_cycle2_refine/cycle2-final/export_bundle/outputs/decision_episode.json`
- `tmp/phase13_cycle2_refine/cycle2-final/replay_bundle/outputs/decision_episode.json`
- `backend/scripts/run_phase13_cycle_refinement.py`
- `backend/app/research_logic/phase10_multi_paper_validation.py`
- `backend/app/research_logic/decision_episode_export.py`
- `backend/app/research_logic/replay_io.py`
- `backend/app/research_logic/models.py`
- `backend/app/research_logic/route_state_synthesizer.py`
- `backend/app/research_logic/iteration_prioritization.py`
- `backend/tests/test_decision_episode_export.py`
- `backend/tests/test_replay_io.py`
- `backend/tests/test_phase10_multi_paper_validation.py`
- `backend/tests/test_historical_replay_compiler.py`
- `backend/tests/test_route_state_synthesizer.py`
- `backend/tests/test_iteration_prioritization.py`

### Secondary (MEDIUM confidence)
- `.planning/phases/06-decision-episode-audit-export/06-RESEARCH.md`
- `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md`
- `docs/superpowers/specs/2026-04-01-logickg-decision-prior-and-episode-schema.md`
- `docs/superpowers/specs/2026-04-01-logickg-why-now-and-route-comparison-schema.md`
- `backend/tests/test_prior_induction.py`
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: bounded multi-paper rerun pipeline, review-backed export assembly, training-facing JSON publication, and recommendation handoff
- Ecosystem: `research_logic`, rerun CLI scripts, bundle writers, route-state schema/builders, and pytest validation
- Patterns: additive export views, schema-level provenance retention, section-level review coverage, machine-readable best-cycle selection
- Pitfalls: replay/export truth collapse, markdown-only verdicts, missing provenance, section-scope drift

**Confidence breakdown:**
- Reuse of the existing Phase 13 runner and Phase 10 orchestration surface: HIGH
- Need for additive training-facing export outputs rather than replacement: HIGH
- Need for structured review metadata and section-level review coverage: HIGH
- Exact field names for best-cycle and exclusion metadata: MEDIUM
</metadata>
