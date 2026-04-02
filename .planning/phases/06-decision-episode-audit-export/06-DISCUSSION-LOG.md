# Phase 6: Decision Episode Audit Export - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md - this log preserves the analysis.

**Date:** 2026-04-02
**Phase:** 06-decision-episode-audit-export
**Mode:** assumptions
**Areas analyzed:** Export boundary and artifact shape, L4 card selection policy, leakage and visibility discipline, implementation path and reuse

## Assumptions Presented

### Export Boundary And Artifact Shape
| Assumption | Confidence | Evidence |
|------------|------------|----------|
| Phase 6 should add a dedicated audited export bundle, separate from the replay bundle and prior-review bundle, while reusing the repo's existing manifest/summary/inspection pattern. | Likely | `.planning/ROADMAP.md`; `backend/app/research_logic/replay_io.py`; `tmp/phase5_multi_route_prior_induction/replay_bundle/replay_summary.json`; `tmp/phase5_multi_route_prior_induction/review_bundle/bundle_manifest.json` |

### L4 Card Selection Policy
| Assumption | Confidence | Evidence |
|------------|------------|----------|
| Audited exports should be driven by reviewed accepted card ids from the registry/manifest, not raw candidate lists or live replay-time candidates. | Confident | `backend/app/research_logic/prior_induction.py`; `tmp/phase5_multi_route_prior_induction/review_bundle/bundle_manifest.json`; `docs/replay/reports/phase5-prior-review-pilot.md`; `backend/app/research_logic/historical_replay_compiler.py` |
| Empty accepted prior ids should remain empty in the audited sample, while accepted anti-pattern ids should be eligible for export selection when they match the route. | Likely | `tmp/phase5_multi_route_prior_induction/review_bundle/candidate_review_summary.json`; `tmp/phase5_multi_route_prior_induction/review_bundle/anti_pattern_candidates.json`; `backend/app/research_logic/models.py`; `tmp/phase5_multi_route_prior_induction/replay_bundle/outputs/decision_episode.json` |

### Leakage And Visibility Discipline
| Assumption | Confidence | Evidence |
|------------|------------|----------|
| Hindsight must remain label/eval-only, and the export should make visibility boundaries explicit at the artifact level instead of relying only on nested fields. | Confident | `docs/superpowers/specs/2026-04-01-logickg-decision-prior-and-episode-schema.md`; `.planning/ROADMAP.md`; `backend/app/research_logic/models.py`; `backend/app/research_logic/decision_episode_builder.py`; `backend/tests/test_decision_episode_builder.py` |

### Implementation Path And Reuse
| Assumption | Confidence | Evidence |
|------------|------------|----------|
| Phase 6 should extend the existing `backend/app/research_logic/` file-based artifact flow and associated regression tests rather than adding a new UI or database-first export path. | Confident | `.planning/phases/04-replay-failure-taxonomy-and-l2-surgical-loop/04-CONTEXT.md`; `.planning/phases/05-multi-route-prior-induction/05-CONTEXT.md`; `backend/app/research_logic/replay_io.py`; `backend/tests/test_replay_io.py`; `backend/tests/test_historical_replay_compiler.py` |

## Corrections Made

No corrections were captured in this non-interactive assumptions refresh.

## External Research

None - the current repo state, specs, tests, and pilot artifacts were sufficient to ground the assumptions.
