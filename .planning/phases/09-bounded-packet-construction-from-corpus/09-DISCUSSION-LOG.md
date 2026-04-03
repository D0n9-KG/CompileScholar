# Phase 9: Bounded Packet Construction From Corpus - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in `09-CONTEXT.md` preserve the locked assumptions for downstream work.

**Date:** 2026-04-03
**Phase:** 09-bounded-packet-construction-from-corpus
**Mode:** assumptions
**Areas analyzed:** Topic Selection And Scope, Packet Artifact Shape, Role Mapping Boundary, Validation And Gap Reporting

## Assumptions Presented

### Topic Selection And Scope
| Assumption | Confidence | Evidence |
|------------|------------|----------|
| Start the first bounded packet from the Phase 8 owner-queue exemplars and fixed-regression slice, then expand conservatively through the filesystem-backed corpus inventory only as needed to justify one bounded topic and cutoff. | Likely | `.planning/ROADMAP.md`; `.planning/STATE.md`; `.planning/phases/07-corpus-sampling-and-regression-baseline/07-CONTEXT.md`; `.planning/phases/08-sampled-single-paper-l2-regression/08-CONTEXT.md`; `docs/replay/reports/phase8-sampled-single-paper-l2-regression-baseline.md`; `docs/replay/corpus_sampling/phase7-fixed-regression-set.json`; `backend/app/research_logic/corpus_sampling.py`; `backend/app/research_logic/sampled_single_paper.py` |

### Packet Artifact Shape
| Assumption | Confidence | Evidence |
|------------|------------|----------|
| Keep the canonical Phase 9 artifact as a committed file-based `RoutePacket` plus packet notes rather than inventing a UI, database object, or new packet schema. | Confident | `backend/app/research_logic/models.py`; `docs/superpowers/specs/2026-04-01-logickg-route-packet-schema.md`; `docs/replay/pilot_packets/README.md`; `docs/replay/pilot_packets/phase1-route-packet.json`; `docs/replay/pilot_packets/phase1-selection-notes.md`; `.planning/phases/01-replay-pilot-packetization/01-CONTEXT.md` |

### Role Mapping Boundary
| Assumption | Confidence | Evidence |
|------------|------------|----------|
| Preserve the current paper-level `RoutePacket` roles and express downstream `support / alternative / held_out` as a companion mapping or subset-packet layer instead of replacing the packet schema. | Likely | `backend/app/research_logic/models.py`; `backend/app/research_logic/route_state_package.py`; `backend/scripts/run_route_state_package.py`; `backend/tests/test_route_state_package.py`; `docs/replay/reports/route-state-package-pilot-2026-04-02.md`; `tmp/phase3_route_state_package/route_state_package_manifest.json` |

### Validation And Gap Reporting
| Assumption | Confidence | Evidence |
|------------|------------|----------|
| Log pre-`L3/L4` packet-assembly gaps explicitly using the existing readiness vocabulary: missing traces, role imbalance, weak support density, scope drift, and availability-only blockers kept separate from semantic packet quality. | Confident | `.planning/REQUIREMENTS.md`; `.planning/ROADMAP.md`; `.planning/phases/07-corpus-sampling-and-regression-baseline/07-CONTEXT.md`; `.planning/phases/08-sampled-single-paper-l2-regression/08-CONTEXT.md`; `backend/app/research_logic/replay_io.py`; `backend/app/research_logic/route_state_package.py`; `backend/tests/test_route_state_package.py`; `docs/replay/reports/route-state-package-pilot-2026-04-02.md` |

## Corrections Made

No user corrections were made in this pass. The assumptions above were captured as the default Phase 9 context from codebase and artifact evidence and can be revised in a later discuss refresh if needed.
