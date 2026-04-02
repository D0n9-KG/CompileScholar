# Phase 7: Corpus Sampling And Regression Baseline - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `07-CONTEXT.md`; this log preserves the assumptions and confirmations.

**Date:** 2026-04-03
**Phase:** 07-corpus-sampling-and-regression-baseline
**Mode:** assumptions
**Areas analyzed:** Sampling Surface, Batch Structure, Corpus Health Separation, Phase Boundary, Sampling Source Of Truth

## Assumptions Presented

### Sampling Surface
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 7 should use file-based manifests and reports rather than a frontend workflow or database-backed state machine. | Likely | `backend/scripts/run_replay_pilot.py`, `backend/scripts/run_route_state_package.py`, `backend/scripts/export_decision_episode_pilot.py`, `backend/app/research_logic/replay_io.py`, `docs/replay/pilot_packets/README.md` |

### Batch Structure
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Every iteration should include both a fixed regression set and a random exploration set. | Likely | `.planning/PROJECT.md`, `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `backend/app/graph/neo4j_client.py`, `eval_quality.py` |
| The first baseline should use `10` fixed papers and `5` random papers per cycle. | Unclear | User intent needed to lock the operating point; project docs supported the dual-track idea but not exact counts. |

### Corpus Health Separation
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Broken or missing shared-corpus paths must be reported separately from model-quality failures. | Confident | Shared-corpus recursive scan surfaced many `DirectoryNotFound` issues; `backend/app/research_logic/replay_io.py`; `backend/tests/test_replay_io.py` |

### Phase Boundary
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 7 should stop at baseline sampling/reporting and hand off `L2` fixes to Phase 8 and multi-paper packet work to Phase 9. | Confident | `.planning/ROADMAP.md`, `.planning/phases/04-replay-failure-taxonomy-and-l2-surgical-loop/04-CONTEXT.md`, `.planning/phases/05-multi-route-prior-induction/05-CONTEXT.md`, `.planning/phases/06-decision-episode-audit-export/06-CONTEXT.md` |

### Sampling Source Of Truth
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| The shared corpus filesystem inventory should be the source of truth for sampling eligibility, with Neo4j ingestion tracked as metadata only. | Unclear | User-provided shared corpus root; existing replay/pilot scripts are file-path driven; `docs/replay/pilot_packets/README.md`; current planning direction in `.planning/PROJECT.md` |

## Corrections Made

No assumption reversals were needed. The user confirmed the recommended defaults after clarifying terminology:

- **Batch size confirmation:** fixed regression set `10`, random exploration set `5`
- **Sampling source confirmation:** shared corpus filesystem inventory is the source of truth; Neo4j ingestion state is tracked as metadata, not the primary eligibility gate

## External Research

None. Codebase and local project context were sufficient for Phase 7 discussion.
