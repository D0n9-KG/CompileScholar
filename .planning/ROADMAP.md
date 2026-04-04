# Milestone v1.2: Fast Iteration Research Logic Quality

**Status:** ACTIVE 2026-04-03
**Phases:** 12-16
**Requirements:** 9 mapped
**Numbering Mode:** Continue from `v1.1`

## Overview

`v1.2` is a continuous optimization milestone.

Inside one milestone, the team repeatedly runs: generate real outputs -> manually review output quality -> optimize pipeline -> rerun. Milestone completion depends on stable high-quality outputs across consecutive cycles, not a single good run.

## Phase Summary

| Phase | Name | Goal | Requirements | Success Criteria |
|-------|------|------|--------------|------------------|
| 12 | Baseline Data Cycle 1 (Direct Fix) | Use Phase 10/11 existing artifacts directly, apply focused fixes, rerun, and review output quality. | `LOOPR-01`, `LOOPR-02` | Complete (`2026-04-04`) |
| 13 | Baseline Data Cycle 2 (Refine) | 2/3 | In Progress|  |
| 14 | Cycle 2 Optimization And Review | Apply fixes from cycle 1, rerun, and re-review outputs with explicit defect deltas. | `REVIEW-02`, `REVIEW-03` | 4 |
| 15 | Cycle 3 Consolidation | Continue optimize/rerun to reach repeated high-quality output under manual review. | `STAB-01` | 3 |
| 16 | Stability Verification And Handoff | Verify consecutive-cycle stability and publish final verification with residual risk notes. | `STAB-02`, `STAB-03` | 3 |

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

**Goal:** Optimize based on cycle 1 review findings, rerun, and compare output-level quality changes.
**Depends on:** Phase `13`
**Requirements:** `REVIEW-02`, `REVIEW-03`

**Success criteria:**
1. Pipeline fixes are implemented from explicit cycle 1 defects.
2. Cycle 2 outputs are manually reviewed with defect delta comparisons.
3. Recommendation priority is justified by reviewed output evidence.
4. Remaining defects are prioritized for next iteration.

### Phase 15: Cycle 3 Consolidation

**Goal:** Continue optimize/rerun until high-quality outputs become repeatable under manual review.
**Depends on:** Phase `14`
**Requirements:** `STAB-01`

**Success criteria:**
1. Another full cycle runs with additional targeted fixes.
2. Manual review judges output quality as high for this cycle.
3. Consecutive high-quality-cycle evidence is recorded.

### Phase 16: Stability Verification And Handoff

**Goal:** Confirm stable quality across consecutive cycles and close milestone with a clear next-cycle handoff.
**Depends on:** Phase `15`
**Requirements:** `STAB-02`, `STAB-03`

**Success criteria:**
1. Consecutive-cycle evidence demonstrates reproducible high-quality outputs.
2. Final verification explains why quality is stable and where residual risk remains.
3. A clear next optimization focus is documented for the following milestone.

## Coverage

| Requirement | Phase |
|-------------|-------|
| `LOOPR-01` | Phase `12` |
| `LOOPR-02` | Phase `12` |
| `LOOPR-03` | Phase `13` |
| `REVIEW-01` | Phase `13` |
| `REVIEW-02` | Phase `14` |
| `REVIEW-03` | Phase `14` |
| `STAB-01` | Phase `15` |
| `STAB-02` | Phase `16` |
| `STAB-03` | Phase `16` |

**Coverage status:** `9/9` requirements mapped

## Notes

- This roadmap keeps phase numbering continuity from v1.1.
- v1.2 starts directly from existing Phase 10/11 data and commands instead of building a separate foundation layer first.
- v1.2 progress is judged by real generated output quality and cycle-to-cycle stability, not metric-only improvements.
- Manual review findings are first-class evidence and must directly steer optimization work.
- Overfitting guardrail: do not generalize single-paper tricks to pipeline-wide logic unless multi-stage quality actually improves.

## Next Up

**Phase 13: Baseline Data Cycle 2 (Refine)** - use the new Phase 12 cycle-1 artifacts to target the downstream blockers that remained after packet repair.

`$gsd-discuss-phase 13`

Also available: `$gsd-plan-phase 13`

---
*Roadmap created: 2026-04-03*
*Last updated: 2026-04-04 after completing Phase 12*
