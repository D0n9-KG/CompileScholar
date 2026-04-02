# Phase 4: Replay Failure Taxonomy And L2 Surgical Loop - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md - this log preserves the analysis.

**Date:** 2026-04-02
**Phase:** 04-Replay Failure Taxonomy And L2 Surgical Loop
**Mode:** assumptions
**Areas analyzed:** Failure taxonomy boundary, output contract, L2 repair scope, evaluation loop

## Assumptions Presented

### Failure taxonomy boundary
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Failure taxonomy should extend the existing replay artifact boundary rather than create a parallel diagnostics path. | Confident | `backend/app/research_logic/replay_io.py`, `backend/app/research_logic/historical_replay_compiler.py`, `.planning/phases/03-grounded-route-replay-compilation/03-VALIDATION.md` |
| Failure attribution should stay layer-aware across packet, `L1`, `L2`, and downstream replay layers. | Confident | `.planning/ROADMAP.md`, `backend/app/research_logic/historical_replay_compiler.py`, `docs/replay/reports/route-state-package-pilot-2026-04-02.md` |

### L2 repair scope
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 4 should optimize `L2` surgically rather than redesign the schema. | Confident | `.planning/PROJECT.md`, `.planning/ROADMAP.md`, `.planning/STATE.md` |
| Comparator density, expected-slot completeness, and relation stitching are the first repair targets. | Likely | `.planning/STATE.md`, `.planning/codebase/CONCERNS.md`, `backend/app/paper_logic_trace/models.py` |

### Evaluation loop
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Every `L2` repair should be judged against the same bounded jamming replay baseline. | Confident | `.planning/phases/03-grounded-route-replay-compilation/03-VALIDATION.md`, `docs/replay/reports/route-state-package-pilot-2026-04-02.md` |
| New diagnostics should remain file-based and reusable from the current `research_logic` scripts and bundle outputs. | Confident | `backend/app/research_logic/replay_io.py`, `backend/scripts/run_replay_pilot.py`, `backend/scripts/run_route_state_package.py` |

## Corrections Made

### Failure taxonomy boundary
- **Original assumption:** Failure attribution mainly needed a layer-aware contract at the replay artifact boundary.
- **User correction:** Keep one unified `failure_records` list for both hard failures and non-blocking `L2` weaknesses, but require non-blocking items to stay explicitly marked `blocking=false`.
- **Reason:** Phase 4 should generate a bounded repair queue instead of acting as a hard-failure-only reporter.

### Output contract
- **Original assumption:** Exact field placement between `replay_summary` and `replay_inspection` could remain agent discretion.
- **User correction:** Keep `stage` and `layer` as separate fields, and fix the artifact split so `replay_summary` carries aggregate taxonomy counts while `replay_inspection` carries full `failure_records` detail.
- **Reason:** The user wants clear separation between where a weakness surfaced, which layer owns the repair, and how much detail each artifact should expose.

### Evaluation loop
- **Original assumption:** The replay taxonomy might expand into a broader optimization evaluation surface immediately.
- **User correction:** Treat the Phase 4 taxonomy as sufficient for the current replay-grounded repair loop only; deeper layer-optimization evaluation can be added later.
- **Reason:** Phase 4 should stay focused on the bounded `L2` surgical loop rather than trying to solve the long-term evaluation stack in one phase.

## Auto-Resolved

- Comparator density / expected-slot completeness / relation stitching were selected as the recommended first repair targets because they match the current post-Phase-3 blocker list in `.planning/STATE.md`

## External Research

No external research was required. The phase assumptions were grounded by the local roadmap, validation artifacts, and current replay code.
