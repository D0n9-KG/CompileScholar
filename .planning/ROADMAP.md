# Milestone v1.2: Fast Iteration Research Logic Quality

**Status:** COMPLETE 2026-04-04
**Phases:** 12-16
**Requirements:** 16 mapped
**Numbering Mode:** Continue from `v1.1`

## Overview

`v1.2` is a continuous optimization milestone.

Inside one milestone, the team repeatedly runs: generate real outputs -> manually review output quality -> optimize pipeline -> rerun. Milestone completion depends on stable high-quality outputs across consecutive cycles, judged by directly reading the generated reasoning artifacts and deciding whether they are actually suitable as scientific-thinking training data, not by metrics alone or by a single good run. The later phases also optimize the exported data shape itself so the final artifacts become self-contained, evidence-grounded, and training-friendly rather than only audit-friendly. Every major section of the final training artifact stays in optimization scope, and the milestone aims to end with a bounded but genuinely usable scientific-thinking dataset rather than one-off exports.

## Phase Summary

| Phase | Name | Goal | Requirements | Success Criteria |
|-------|------|------|--------------|------------------|
| 12 | Baseline Data Cycle 1 (Direct Fix) | Use Phase 10/11 existing artifacts directly, apply focused fixes, rerun, and review output quality. | `LOOPR-01`, `LOOPR-02` | Complete (`2026-04-04`) |
| 13 | Baseline Data Cycle 2 (Refine) | Run a second bounded rerun on the fixed cycle-1 baseline, recover replay quality, and leave an explicit defect-delta handoff. | `LOOPR-03`, `REVIEW-01` | Complete (`2026-04-04`) |
| 14 | Cycle 2 Optimization And Review | Add self-contained training-facing exports, review two bounded candidate cycles, and confirm the next-cycle recommendation from reviewed evidence. | `REVIEW-02`, `REVIEW-03`, `TRAIN-01`, `TRAIN-02`, `TRAIN-03`, `TRAIN-06` | Complete (`2026-04-04`) |
| 15 | Cycle 3 Consolidation | Continue bounded optimize/rerun cycles after the Phase 14 export-structure improvements until one cycle is manually judged genuinely useful as training data and reviewed prior/anti-pattern knowledge begins closing into the final export surface. | `STAB-01`, `TRAIN-04` | Complete (`2026-04-04`) |
| 16 | Stability Verification And Handoff | Verify that consecutive high-quality cycles stay manually convincing as training data and publish the final stability, schema, dataset, and residual-risk handoff. | `STAB-02`, `STAB-03`, `TRAIN-05`, `TRAIN-07` | Complete (`2026-04-04`) |

## Phase Details

### Phase 12: Baseline Data Cycle 1 (Direct Fix)

**Goal:** Start immediately from Phase 10/11 artifacts, apply targeted fixes, rerun, and review real outputs without building a new runtime foundation first.
**Depends on:** Phase `11`
**Requirements:** `LOOPR-01`, `LOOPR-02`

**Success criteria:**
1. Existing commands and artifacts from Phase 10/11 are reused to run the first fix-rerun cycle.
2. Real output review notes identify concrete defects in generated reasoning artifacts.
3. Fixes are linked to observed defects and rerun results in the same cycle.
4. Review explicitly checks that fixes improve overall quality, not only one-paper behavior.

### Phase 13: Baseline Data Cycle 2 (Refine)

**Goal:** Run another cycle on the same baseline data to verify improvements are repeatable and not overfit to single-paper artifacts.
**Depends on:** Phase `12`
**Requirements:** `LOOPR-03`, `REVIEW-01`

**Success criteria:**
1. Second cycle reruns after targeted fixes from phase 12 review findings.
2. Manual review compares cycle-2 outputs against cycle-1 defects and confirms real quality movement.
3. Fix impact is evaluated at packet/replay/prior/export levels, not only single-document patterns.
4. Remaining defects are prioritized for the next optimization cycle.

### Phase 14: Cycle 2 Optimization And Review

**Goal:** Keep optimizing the bounded slice from the Phase 13 defect review until at least one rerun produces reasoning artifacts that manual review judges genuinely usable as scientific-thinking training data.
**Depends on:** Phase `13`
**Requirements:** `REVIEW-02`, `REVIEW-03`, `TRAIN-01`, `TRAIN-02`, `TRAIN-03`, `TRAIN-06`

**Success criteria:**
1. Pipeline fixes are implemented from explicit Phase 13 defect findings rather than from metric-only guesses.
2. Phase 14 may run multiple bounded optimize -> rerun loops inside the phase until one cycle is manually judged "good enough" on the generated reasoning content itself.
3. Manual review explicitly reads real package / replay / prior / export outputs and judges whether they would help train a model to perform scientific reasoning, not merely satisfy stage flags.
4. The chosen best cycle publishes a self-contained training-facing export view that keeps key evidence snippets, comparison reasoning, and final decision content together without requiring cross-file reconstruction.
5. Final artifacts preserve canonical labels alongside raw source phrases for route features, bottlenecks, and conditions, and they record structured human review / training-acceptance fields inside the machine-readable export.
6. Every major content section of the final training artifact, including evidence pack, route synthesis, why-now, route comparison, priors / anti-patterns, minimal attack path, final decision, and review labels, is explicitly reviewed and left inside the optimization scope rather than treated as frozen background.
7. Recommendation priority and remaining defects are justified by reviewed output evidence from the chosen best cycle.

### Phase 15: Cycle 3 Consolidation

**Goal:** Continue bounded optimization after the Phase 14 export-structure improvements until the first cycle is actually judged high quality by direct manual review, while also tightening how reviewed prior / anti-pattern knowledge closes into the final export.
**Depends on:** Phase `14`
**Requirements:** `STAB-01`, `TRAIN-04`

**Success criteria:**
1. Another full bounded cycle runs with targeted fixes linked to the remaining reviewed defects from Phase 14's best-cycle review.
2. Manual review judges the generated reasoning artifacts high quality based on direct reading, not just metrics or readiness flags.
3. Reviewed accepted prior / anti-pattern knowledge either flows into the final selected export fields when it truly matches the route or carries explicit structured exclusion rationale when it does not.
4. The new good cycle is compared directly against the completed Phase 14 best-cycle review so the quality jump is explicit.
5. The resulting outputs look genuinely useful as scientific-thinking training data rather than remaining structurally valid but substantively weak.

### Phase 16: Stability Verification And Handoff

**Goal:** Confirm that consecutive good cycles are truly stable under direct manual review and close the milestone with a clear dataset-readiness and residual-risk handoff.
**Depends on:** Phase `15`
**Requirements:** `STAB-02`, `STAB-03`, `TRAIN-05`, `TRAIN-07`

**Success criteria:**
1. Consecutive-cycle evidence demonstrates reproducible high-quality outputs under direct manual review of the generated reasoning artifacts.
2. Final verification explains why the outputs are considered stably useful for scientific-thinking training data and where residual risk remains.
3. The final handoff publishes the stable training-data schema and multiple task-specific training views for route synthesis, why-now judgment, route comparison, prior / anti-pattern learning, and final decision episodes.
4. The milestone closes with a bounded but genuinely training-usable dataset bundle assembled from the validated later-cycle artifacts rather than only isolated one-off exports.
5. A clear next optimization focus is documented for the following milestone.

## Coverage

| Requirement | Phase |
|-------------|-------|
| `LOOPR-01` | Phase `12` |
| `LOOPR-02` | Phase `12` |
| `LOOPR-03` | Phase `13` |
| `REVIEW-01` | Phase `13` |
| `REVIEW-02` | Phase `14` |
| `REVIEW-03` | Phase `14` |
| `TRAIN-01` | Phase `14` |
| `TRAIN-02` | Phase `14` |
| `TRAIN-03` | Phase `14` |
| `TRAIN-06` | Phase `14` |
| `STAB-01` | Phase `15` |
| `TRAIN-04` | Phase `15` |
| `STAB-02` | Phase `16` |
| `STAB-03` | Phase `16` |
| `TRAIN-05` | Phase `16` |
| `TRAIN-07` | Phase `16` |

**Coverage status:** `16/16` requirements mapped

## Notes

- This roadmap keeps phase numbering continuity from v1.1.
- v1.2 starts directly from existing Phase 10/11 data and commands instead of building a separate foundation layer first.
- v1.2 progress is judged by real generated output quality and cycle-to-cycle stability, not metric-only improvements.
- Manual review findings are first-class evidence and must directly steer optimization work.
- A cycle counts as "good" only when direct human review says the generated content itself would be useful for training scientific reasoning, not when the pipeline merely turns more flags green.
- Later phases must optimize the exported training-data shape itself, including self-contained evidence views, canonicalized concept labels, structured human review fields, and usable multi-view training artifacts.
- Later phases must also optimize every major content section of the final training artifact and aim to ship a bounded but genuinely usable scientific-thinking dataset bundle by milestone close.
- Overfitting guardrail: do not generalize single-paper tricks to pipeline-wide logic unless multi-stage quality actually improves.

## Next Up

**Milestone `v1.2` complete** - the bounded dataset bundle and stability handoff are now published, and the next milestone should start from `packet_construction` with `relation_assembly` and `slot_recovery` as the leading owner buckets.

Also available: `$gsd-complete-milestone`

---
*Roadmap created: 2026-04-03*
*Last updated: 2026-04-04 after completing Phase 16*
