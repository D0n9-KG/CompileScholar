# Phase 8: Sampled Single-Paper L2 Regression - Context

**Gathered:** 2026-04-03 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 8 repeatedly reruns sampled single-paper extraction and evaluation on the Phase 7 fixed regression set plus random exploration samples, so `L2` quality can be measured on both stable and novel papers. It does not redesign the `PaperLogicTrace` schema, run the full corpus end to end, fix every discovered owner inside the same phase, or move into bounded multi-paper packet construction.

</domain>

<decisions>
## Implementation Decisions

### Execution Path
- **D-01:** Phase 8 should add a dedicated sampled-paper regression runner, but it must reuse the existing ingest/rebuild and `PaperLogicTrace` compilation path (`run_phase1_paper_logic_trace`, `compile_paper_logic_trace`, rebuild helpers) rather than creating a parallel extraction-only evaluator.
- **D-02:** The runner should measure the same quality surfaces the mainline already produces: gate result, `quality_tier`, `quality_tier_score`, `quality_flags`, `l2_completeness_audit`, and any reference/citation recovery side effects that materially affect the canonical trace.
- **D-03:** Sample execution problems caused by corpus drift, missing preferred sources, or graph-unavailable records must remain separate from true `L2` quality failures, continuing the Phase 7 discipline that availability problems are not silently counted as model regressions.

### Sample Source And Availability Policy
- **D-04:** The Phase 7 fixed regression manifest and random batch artifacts are the source of truth for the first Phase 8 evaluation cycle. The fixed ten-paper set stays stable across iterations unless deliberately reviewed, while the random sample can refresh per cycle.
- **D-05:** Phase 8 must be able to execute from filesystem-backed sample references (`corpus_relative_ref`, preferred source paths, committed fixed-set docs) even when Neo4j enrichment is absent, because the real Phase 7 baseline selected `15/15` papers without graph metadata.
- **D-06:** Phase 8 should start from the real local Phase 7 baseline under `tmp/phase7_corpus_sampling_baseline/` instead of inventing a new seed or a different first-cycle sample contract.

### Artifact And Reporting Shape
- **D-07:** Phase 8 outputs should remain file-based and auditable: per-iteration bundles under `tmp/` with manifest / summary / inspection JSON plus per-paper artifacts, matching the repository's existing replay and sampling bundle conventions.
- **D-08:** The committed human-readable artifact for each iteration should live under `docs/replay/reports/` and summarize the evaluated papers, cohort-level outcomes, owner buckets, and the next recommended `L2` work queue rather than only dumping raw JSON.
- **D-09:** Per-paper results must preserve enough detail for replayable diagnosis, including the canonical `paper_logic_trace`, quality report, and any recovery artifacts that explain why a paper failed or downgraded.

### Regression Semantics
- **D-10:** The fixed regression set is the strict cross-iteration comparison surface for recurring failures and regressions; the random exploration set is an edge-case discovery surface and must be reported separately rather than merged into the same regression verdict.
- **D-11:** Iteration comparisons should distinguish at least three result classes: recurring baseline failures, newly introduced regressions on the fixed set, and new edge cases surfaced only by the random exploration batch.
- **D-12:** Phase 8 ends with a concrete prioritized short list of `L2` owners derived from sampled-paper evidence. It does not auto-expand into multi-paper packet work or full repair execution inside the same phase.

### Owner-Oriented Failure Framing
- **D-13:** The summary should map failures into concrete `L2` owner buckets using the quality and extraction signals the codebase already emits, such as missing expected roles/slots, weak relation stitching, `route_state_seed_thin`, metadata-summary mismatch, residual noise moves, reference recovery gaps, and citation-purpose / citation-event issues.
- **D-14:** Owner buckets should be chosen to guide the next optimization cycle, not to maximize metric verbosity. The output should make it obvious whether the highest-leverage next work is in slot recovery, relation assembly, route-seed richness, metadata repair, or citation semantics.
- **D-15:** Phase 8 should preserve the earlier replay-layer lesson that `stage` and `layer` are different concerns: paper-level extraction symptoms are measured here, while later multi-paper compilation work remains separate in Phases 9-10.

### the agent's Discretion
- Exact CLI flags, output directory naming, and whether iteration ids are timestamp-based or user-provided
- The exact owner-bucket rubric and threshold cutoffs, as long as it stays grounded in existing quality signals
- Whether the first implementation runs selected papers through direct filesystem rebuild, graph-backed rebuild, or a hybrid path, as long as the measured path remains faithful to production `L2` generation

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project And Milestone Framing
- `.planning/PROJECT.md` - Current `v1.1` goal, corpus-driven iteration strategy, and active Phase 8 scope
- `.planning/REQUIREMENTS.md` - `L2Q-01` and `L2Q-02` plus the requirement that Phase 8 compare sampled single-paper results across iterations
- `.planning/ROADMAP.md` - Phase 8 goal, success criteria, and handoff boundaries to Phases 9-11
- `.planning/STATE.md` - Current milestone handoff and the explicit Phase 7 -> Phase 8 transition state

### Prior Phase Constraints And Handoff
- `.planning/phases/04-replay-failure-taxonomy-and-l2-surgical-loop/04-CONTEXT.md` - Keeps owner-oriented failure framing and the distinction between surfaced stage symptoms and repair layers
- `.planning/phases/06-decision-episode-audit-export/06-CONTEXT.md` - Reinforces the project's file-based, auditable artifact philosophy
- `.planning/phases/07-corpus-sampling-and-regression-baseline/07-CONTEXT.md` - Locks the filesystem-first sampling source, stable fixed set, and corpus-health separation
- `.planning/phases/07-corpus-sampling-and-regression-baseline/07-VERIFICATION.md` - Confirms the real Phase 7 baseline counts and the `0/15` Neo4j metadata outcome

### Phase 7 Sample Inputs And Reports
- `docs/replay/corpus_sampling/phase7-fixed-regression-set.json` - The committed ten-paper fixed regression source of truth
- `docs/replay/corpus_sampling/phase7-fixed-regression-notes.md` - Why the fixed set stays stable and how fixed-vs-random roles differ
- `docs/replay/reports/phase7-corpus-sampling-baseline.md` - Human-readable summary of the real baseline sample and corpus-health findings
- `tmp/phase7_corpus_sampling_baseline/bundle_manifest.json` - Local runtime manifest for the first real baseline bundle
- `tmp/phase7_corpus_sampling_baseline/sampling_summary.json` - Real Phase 7 counts, including `selected_with_neo4j_metadata_count = 0`
- `tmp/phase7_corpus_sampling_baseline/sampling_inspection.json` - The exact fixed/random selected ids and selected-paper availability details

### L2 Contracts And Quality Signals
- `docs/superpowers/specs/2026-03-22-logickg-paperlogictrace-l2-redesign-design.md` - Canonical `PaperLogicTrace` design and hot-path vs audit expectations
- `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md` - Why Phase 8 should optimize `L2` based on downstream compilation usefulness rather than paper-summary aesthetics
- `backend/app/paper_logic_trace/compiler.py` - Canonical `PaperLogicTrace` compilation entrypoint
- `backend/app/paper_logic_trace/direct_extraction.py` - Slot extraction, move/relation construction, and trace quality reporting
- `backend/app/paper_logic_trace/gates.py` - Existing quality tiers, completeness audit, and route-seed thinness logic

### Implementation Reuse Points
- `backend/app/extraction/orchestrator.py` - Shared phase-1 trace runner used by ingest and rebuild flows
- `backend/app/ingest/pipeline.py` - Mainline ingestion path that already produces and persists sampled-paper traces
- `backend/app/ingest/rebuild.py` - Paper-by-paper rerun path and gate-aware artifact persistence
- `backend/app/research_logic/corpus_sampling.py` - Typed corpus inventory and batch-selection helpers from Phase 7
- `backend/app/research_logic/replay_io.py` - Manifest / summary / inspection writer pattern to reuse for Phase 8 outputs
- `backend/scripts/run_corpus_sampling_baseline.py` - Existing operator-facing baseline CLI pattern that Phase 8 should extend rather than bypass
- `backend/tests/test_corpus_sampling.py` - Regression boundary for batch reproducibility and corpus-health separation
- `backend/tests/test_corpus_sampling_cli.py` - CLI expectations and real operator usage shape
- `backend/tests/test_replay_io.py` - Bundle-writing contract tests
- `backend/tests/test_paper_logic_trace_gates.py` - Quality-signal and owner-bucket source behavior

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `backend/app/research_logic/corpus_sampling.py`: typed inventory, fixed/random batch models, and stable sample selection helpers from Phase 7
- `backend/scripts/run_corpus_sampling_baseline.py`: existing operator CLI pattern for turning sample definitions into local bundles
- `backend/app/research_logic/replay_io.py`: reusable manifest / summary / inspection writer contract for auditable bundles
- `backend/app/ingest/rebuild.py`: paper-by-paper rerun path that already reparses markdown, rebuilds citations, recompiles `PaperLogicTrace`, and writes artifacts
- `backend/app/ingest/pipeline.py` and `backend/app/extraction/orchestrator.py`: mainline ingestion path and shared phase-1 trace compiler entrypoint
- `backend/app/paper_logic_trace/gates.py` and `backend/app/paper_logic_trace/direct_extraction.py`: existing source of the owner-oriented quality signals Phase 8 should summarize
- `backend/app/api/routers/papers.py` and `backend/app/api/routers/graph.py`: existing read/export surfaces for canonical per-paper traces if inspection tooling needs them

### Established Patterns
- The repo prefers file-based, manifest-driven, auditable workflow artifacts over hidden database state or UI-only orchestration
- Gate failures are explicit and preserved as downgraded or skipped canonical writes rather than silently treated as success
- `PaperLogicTrace` quality is already expressed in structured fields (`quality_tier`, `quality_flags`, `l2_completeness_audit`, `route_state_seed_audit`) that can be reused for owner bucketing
- Focused pytest suites under `backend/tests/` are the normal regression boundary for backend workflow additions

### Integration Points
- The first Phase 8 iteration should consume `tmp/phase7_corpus_sampling_baseline/` plus the committed fixed-set docs as the sample source of truth
- Phase 8 reruns should write per-paper artifacts in a way that can be compared across iterations without reparsing ad hoc logs
- The resulting summary must be directly usable by Phase 9 and Phase 11 as evidence about whether the next work belongs in `L2` extraction or elsewhere
- New tests should live under `backend/tests/` and verify sample execution, bundle output, and comparison classification rather than frontend behavior

</code_context>

<specifics>
## Specific Ideas

- The first Phase 8 cycle should begin from the real fixed ids `1000`, `1001`, `1005`, `1017`, `1023`, `1002`, `1004`, `1007`, `1010`, `1012` and the real seed-`7` random ids `17`, `1317`, `1844`, `580`, `1107`
- The real Phase 7 baseline selected `15/15` papers without Neo4j metadata, so early Phase 8 execution should assume filesystem-backed papers can still be valid evaluation inputs
- The point of Phase 8 is not broad corpus coverage; it is a repeatable sampled-paper loop that produces a short, evidence-backed `L2` optimization queue
- Corpus drift or missing-path rerun failures should stay visible, but they should not be misreported as `PaperLogicTrace` regressions

</specifics>

<deferred>
## Deferred Ideas

- Running the full shared corpus end to end as part of Phase 8
- Folding bounded multi-paper packet assembly into the same phase
- Building a dedicated UI for sampled-paper regression execution or review
- Auto-fixing discovered `L2` owners inside the same phase instead of first producing a stable measured queue

</deferred>

---

*Phase: 08-sampled-single-paper-l2-regression*
*Context gathered: 2026-04-03*
