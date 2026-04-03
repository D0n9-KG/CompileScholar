# Phase 10: Multi-Paper L3 And L4 Validation - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md - this log preserves the analysis that produced them.

**Date:** 2026-04-03
**Phase:** 10-multi-paper-l3-and-l4-validation
**Mode:** assumptions
**Areas analyzed:** workflow orchestration, packet-to-package inputs, blocker reporting, review/export truth, baseline comparison

## Assumptions Presented

### Workflow Orchestration
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 10 should add one dedicated workflow that composes the existing package, replay, and export CLIs instead of introducing new schemas or a UI surface. | Likely | `backend/scripts/run_route_state_package.py`, `backend/scripts/run_replay_pilot.py`, `backend/scripts/export_decision_episode_pilot.py`, and no existing Phase 10 runner under `backend/scripts/` |

### Packet-To-Package Inputs
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| The committed Phase 9 packet and assembly manifest are the frozen input boundary, and Phase 10 should derive runtime package inputs from them without changing the canonical packet contract. | Confident | `docs/replay/pilot_packets/phase9-route-packet.json`, `docs/replay/pilot_packets/phase9-assembly-manifest.json`, `.planning/phases/09-bounded-packet-construction-from-corpus/09-CONTEXT.md`, `backend/app/research_logic/route_state_package.py` |
| Phase 10 needs a real `L1` snapshot before replay quality can be judged meaningfully. | Confident | `docs/replay/reports/phase9-bounded-packet-audit.md`, `backend/app/research_logic/route_state_package.py` |

### Blocker Reporting
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 10 is allowed to end with explicit blockers rather than only a green replay/export story, as long as the blocking stage and evidence are preserved in the artifacts. | Confident | `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `backend/app/research_logic/replay_io.py`, `docs/replay/reports/phase9-bounded-packet-audit.md` |
| Thin support density, one-paper alternative depth, one-paper held-out depth, and fallback-heavy source mix should remain visible blocker hypotheses rather than be hidden by scope expansion. | Likely | `docs/replay/pilot_packets/phase9-assembly-manifest.json`, `docs/replay/reports/phase9-bounded-packet-audit.md`, `.planning/STATE.md` |

### Review And Export Truth
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Replay-time priors and anti-patterns are provisional; the export must rebuild from the review bundle's accepted ids and preserve the existing visibility-bucket audit boundary. | Confident | `backend/app/research_logic/decision_episode_export.py`, `backend/scripts/export_decision_episode_pilot.py`, `backend/tests/test_decision_episode_export.py`, `docs/replay/reports/phase6-decision-episode-export-pilot.md` |

### Baseline Comparison
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 10 should compare the computational-mechanics run against the earlier jamming packaged baseline so Phase 11 can tell compiler regressions apart from packet-specific weaknesses. | Likely | `.planning/ROADMAP.md`, `docs/replay/reports/route-state-package-pilot-2026-04-02.md`, `docs/replay/reports/phase6-decision-episode-export-pilot.md` |

## Corrections Made

No corrections were provided in this run. The assumptions above were recorded as the default assumptions-mode decisions for Phase 10.
