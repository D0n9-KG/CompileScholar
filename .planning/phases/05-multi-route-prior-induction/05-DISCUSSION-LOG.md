# Phase 5: Multi-Route Prior Induction - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md - this log preserves the analysis.

**Date:** 2026-04-02
**Phase:** 05-Multi-Route Prior Induction
**Mode:** assumptions
**Areas analyzed:** phase positioning, induction input boundary, L4 extraction boundary, prior and anti-pattern architecture, review workflow

## Assumptions Presented

### Phase positioning
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 5 should be treated as a bounded `L4` induction pilot rather than a claim that the `L4` layer is already mature. | Confident | `.planning/ROADMAP.md`, `.planning/REQUIREMENTS.md`, `.planning/STATE.md`, `docs/replay/reports/phase4-l2-surgical-delta.md` |
| Phase 6 should be framed as an audit-grade `DecisionEpisode` export pilot rather than immediate large-scale training-data production. | Likely | `.planning/ROADMAP.md`, `backend/app/research_logic/decision_episode_builder.py`, `backend/tests/test_decision_episode_builder.py` |

### Induction input boundary
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 5 should reuse the existing `support / alternative / held_out` route-state package boundary instead of inventing a new multi-route input format. | Confident | `backend/app/research_logic/route_state_package.py`, `backend/app/research_logic/historical_replay_compiler.py`, `docs/replay/reports/route-state-package-pilot-2026-04-02.md` |
| The current green route-state package baseline is sufficient as the first induction input surface even though the assets are still runtime-only. | Likely | `tmp/phase3_route_state_package/bundle/validation.json`, `tmp/phase4_l2_surgical_loop/bundle/validation.json`, `.planning/phases/03-grounded-route-replay-compilation/03-CONTEXT.md` |

### L4 extraction boundary
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| `L4` should be distilled primarily from multiple `RouteState` objects, not directly from raw traces or single papers. | Confident | `docs/superpowers/specs/2026-04-01-logickg-decision-prior-and-episode-schema.md`, `backend/app/research_logic/decision_prior_builder.py`, `backend/app/research_logic/models.py` |
| `L1/L2` should remain grounding and provenance constraints for `L4`, not the direct prior-generation layer. | Likely | `.planning/PROJECT.md`, `backend/app/research_logic/models.py`, `backend/app/research_logic/decision_episode_builder.py` |

### Prior and anti-pattern architecture
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 5 likely needs a batch candidate-induction layer that can emit multiple prior candidates rather than continuing with a one-card replay path. | Likely | `docs/superpowers/specs/2026-04-01-logickg-decision-prior-and-episode-schema.md`, `backend/app/research_logic/decision_prior_builder.py`, `backend/app/research_logic/historical_replay_compiler.py` |
| `AntiPatternCard` should be implemented in this phase instead of remaining schema-only. | Likely | `.planning/ROADMAP.md`, `backend/app/research_logic/models.py`, `backend/tests/test_research_logic_models.py` |

### Review workflow
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Lightweight review should stay file-based and report-driven in this phase rather than introducing a new UI-first workflow. | Likely | `.planning/phases/04-replay-failure-taxonomy-and-l2-surgical-loop/04-CONTEXT.md`, `backend/app/research_logic/replay_io.py`, `backend/scripts/run_replay_pilot.py` |
| Green prior acceptance should continue to require support cluster quality, held-out results, counterexample search, and reviewer metadata. | Confident | `backend/app/research_logic/models.py`, `backend/app/research_logic/decision_prior_builder.py`, `backend/tests/test_decision_prior_builder.py` |

## Corrections Made

### Phase positioning
- **Original assumption:** Phase 5 and Phase 6 could be discussed as the normal next `L4` and export phases without additional framing.
- **User correction:** Treat both phases as “run the cross-layer loop first, then use failures to optimize each layer,” not as proof that upstream `L1/L2/L3` is already mature.
- **Reason:** The user explicitly called out that current `L2/L3` quality is still insufficient for any mature-data interpretation.

### Training maturity framing
- **Original assumption:** Phase 6 might reasonably be described as moving directly into training-data output once `DecisionEpisode` export exists.
- **User correction:** Phase 6 should only prove an auditable, leakage-safe export path; it must not be framed as direct large-scale training-data production.
- **Reason:** The user wants the project to avoid overstating maturity while upstream layers still need significant improvement.

### L4 extraction boundary
- **Original assumption:** `L4` could be described broadly as a cross-layer object built from the whole stack without clarifying the extraction anchor.
- **User correction:** `L4` is mainly distilled from `L3 RouteState`; `L1/L2` remain constraints and provenance layers rather than the direct induction layer.
- **Reason:** The user wanted the derivation logic to stay grounded in multi-route state aggregation instead of collapsing back to raw paper-level heuristics.

## External Research

No external research was required. The phase assumptions were grounded by the local roadmap, replay reports, runtime package artifacts, and the current `research_logic` contracts and tests.
