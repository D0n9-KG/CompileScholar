# Phase 8: Sampled Single-Paper L2 Regression - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md - this log preserves the assumptions that were surfaced and confirmed.

**Date:** 2026-04-03
**Phase:** 08-sampled-single-paper-l2-regression
**Mode:** assumptions
**Areas analyzed:** Execution path, Sample source and availability policy, Artifact and reporting shape, Regression semantics, Owner-oriented failure framing

## Assumptions Presented

### Execution Path

| Assumption | Confidence | Evidence |
|------------|------------|----------|
| Phase 8 should add a dedicated sampled-paper regression runner that reuses the existing ingest/rebuild plus `PaperLogicTrace` path instead of building a parallel evaluator. | Likely | `backend/app/ingest/pipeline.py`, `backend/app/ingest/rebuild.py`, `backend/app/extraction/orchestrator.py`, `backend/app/paper_logic_trace/compiler.py` |

### Sample Source And Availability Policy

| Assumption | Confidence | Evidence |
|------------|------------|----------|
| Phase 8 should execute from the Phase 7 fixed/random sample artifacts and support filesystem-backed sample refs even when Neo4j metadata is absent. | Confident | `tmp/phase7_corpus_sampling_baseline/sampling_summary.json`, `tmp/phase7_corpus_sampling_baseline/sampling_inspection.json`, `docs/replay/corpus_sampling/phase7-fixed-regression-set.json` |

### Artifact And Reporting Shape

| Assumption | Confidence | Evidence |
|------------|------------|----------|
| Phase 8 outputs should stay file-based and auditable, using manifest / summary / inspection JSON plus a committed human-readable report. | Confident | `backend/app/research_logic/replay_io.py`, `.planning/phases/07-corpus-sampling-and-regression-baseline/07-CONTEXT.md`, `.planning/phases/06-decision-episode-audit-export/06-CONTEXT.md` |

### Regression Semantics

| Assumption | Confidence | Evidence |
|------------|------------|----------|
| The fixed set should be the strict cross-iteration regression surface, while the random set should surface new edge cases and be reported separately. | Confident | `.planning/ROADMAP.md`, `.planning/REQUIREMENTS.md`, `docs/replay/corpus_sampling/phase7-fixed-regression-notes.md` |

### Owner-Oriented Failure Framing

| Assumption | Confidence | Evidence |
|------------|------------|----------|
| Phase 8 should summarize failures in concrete `L2` owner buckets derived from existing quality signals rather than only reporting aggregate scores. | Likely | `backend/app/paper_logic_trace/gates.py`, `backend/tests/test_paper_logic_trace_gates.py`, `.planning/phases/04-replay-failure-taxonomy-and-l2-surgical-loop/04-CONTEXT.md` |

## Corrections Made

No corrections - all assumptions were confirmed by the user.

## External Research

No external research was needed. The assumptions were grounded in the local roadmap, prior phase artifacts, and the current codebase.
