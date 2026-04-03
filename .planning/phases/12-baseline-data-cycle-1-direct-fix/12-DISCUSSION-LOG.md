# Phase 12: Baseline Data Cycle 1 (Direct Fix) - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md; this log preserves the analysis that produced them.

**Date:** 2026-04-04
**Phase:** 12-baseline-data-cycle-1-direct-fix
**Mode:** assumptions
**Areas analyzed:** Starting Evidence And Input Boundary, Fix Scope For Cycle 1, Rerun And Artifact Contract, Review And Defect-Linkage Contract, Provenance And Comparison Guardrails

## Assumptions Presented

### Starting Evidence And Input Boundary
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 12 should start from the existing Phase 11 prioritization bundle instead of recomputing the lead target before any fixes. | Confident | `tmp/phase11_iteration_prioritization/baseline/outputs/prioritization_summary.json`; `tmp/phase11_iteration_prioritization/baseline/outputs/prioritization_inspection.json`; `backend/app/research_logic/iteration_prioritization.py`; `.planning/phases/11-iteration-prioritization-and-next-cycle-plan/11-VERIFICATION.md` |
| The bounded computational-mechanics packet from Phase 9 remains the canonical slice for cycle 1. | Confident | `docs/replay/pilot_packets/phase9-route-packet.json`; `docs/replay/pilot_packets/phase9-assembly-manifest.json`; `.planning/phases/10-multi-paper-l3-and-l4-validation/10-CONTEXT.md` |

### Fix Scope For Cycle 1
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 12 should keep packet construction as the lead intervention and target the Phase 10 blocker codes `support_cluster_too_small`, `alternative_scope_not_distinct`, and `yellow_route_state_present` first. | Confident | `docs/replay/reports/phase10-multi-paper-l3-l4-validation.md`; `.planning/phases/10-multi-paper-l3-and-l4-validation/10-VERIFICATION.md`; `tmp/phase11_iteration_prioritization/baseline/outputs/prioritization_summary.json` |
| Phase 8 `relation_assembly` and `slot_recovery` evidence stays visible, but should remain secondary until the packet-first rerun is measured. | Likely | `tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/comparison_summary.json`; `tmp/phase11_iteration_prioritization/baseline/outputs/prioritization_summary.json`; `backend/app/research_logic/iteration_prioritization.py` |

### Rerun And Artifact Contract
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| The safest Phase 12 rerun path is to reuse the existing Phase 10 validation runner or a thin wrapper around it, because that already produces the required package/replay/prior/export/comparison outputs. | Confident | `backend/app/research_logic/phase10_multi_paper_validation.py`; `backend/scripts/run_phase10_multi_paper_validation.py`; `backend/tests/test_phase10_multi_paper_validation.py` |
| The cycle should regenerate a fresh machine-readable comparison summary under a new cycle-scoped `tmp/` root instead of relying on the missing original Phase 10 JSON. | Confident | `.planning/phases/11-iteration-prioritization-and-next-cycle-plan/11-VERIFICATION.md`; `backend/app/research_logic/iteration_prioritization.py`; workspace `tmp/` layout showing no `tmp/phase10_multi_paper_validation/baseline/comparison_summary.json` root preserved today |

### Review And Defect-Linkage Contract
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 12 needs an explicit same-cycle review artifact that links implemented fixes to rerun defect movement on real outputs. | Confident | `.planning/ROADMAP.md`; `.planning/REQUIREMENTS.md`; `.planning/PROJECT.md` |
| The most repo-consistent review shape is a same-slice before/after delta report, similar to Phase 4, rather than a brand-new review framework. | Likely | `docs/replay/reports/phase4-l2-surgical-delta.md`; `.planning/phases/04-replay-failure-taxonomy-and-l2-surgical-loop/04-VERIFICATION.md`; existing `tmp/` plus `docs/replay/reports/` reporting pattern |

### Provenance And Comparison Guardrails
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 12 should preserve fallback provenance to the committed Phase 10 verification/report and the Phase 11 prioritization bundle until the new cycle artifacts are generated. | Confident | `tmp/phase11_iteration_prioritization/baseline/outputs/prioritization_summary.json`; `.planning/phases/11-iteration-prioritization-and-next-cycle-plan/11-VERIFICATION.md`; `backend/app/research_logic/iteration_prioritization.py` |
| Rerunning Phase 11 prioritization after the cycle is optional discretion, not the defining deliverable of Phase 12. | Likely | `.planning/ROADMAP.md`; `.planning/phases/11-iteration-prioritization-and-next-cycle-plan/11-CONTEXT.md`; `backend/scripts/run_phase11_iteration_prioritization.py` |

## Corrections Made

No corrections were recorded in this assumptions-mode run. The context was captured directly from prior phase evidence, current runtime artifacts, and existing codebase patterns.
