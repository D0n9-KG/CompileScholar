# Phase 7: Corpus Sampling And Regression Baseline - Context

**Gathered:** 2026-04-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 7 establishes the corpus-driven sampling baseline for `v1.1`. It defines how papers are selected from the shared corpus, how each batch is recorded for replayability, and how corpus-health failures are separated from model-quality failures. It does not perform `L2` quality fixes, build bounded multi-paper packets, or change `L3/L4` contracts.

</domain>

<decisions>
## Implementation Decisions

### Sampling Source And Eligibility
- **D-01:** The shared corpus filesystem inventory is the source of truth for sampling eligibility in Phase 7. If a paper exists under the shared corpus root and can be indexed into the sampling inventory, it is eligible for selection even if it has not yet been ingested into Neo4j.
- **D-02:** Neo4j ingestion state should be tracked as metadata on each sampled paper, not used as the primary gate for whether the paper can enter a batch.
- **D-03:** Sampling inventory records should preserve paper identifiers, source paths, and availability signals so downstream phases can distinguish "not in graph yet" from "not present in the corpus" and from "present but unreadable."

### Batch Structure
- **D-04:** Every iteration cycle should include two batch types: one fixed regression set and one random exploration set.
- **D-05:** The fixed regression set is locked to `10` papers per cycle for the first Phase 7 baseline.
- **D-06:** The random exploration set is locked to `5` papers per cycle for the first Phase 7 baseline.
- **D-07:** The fixed regression set should remain stable across iterations until an explicit review updates it, while the random exploration set should be regenerated per cycle.

### Artifact And Reporting Shape
- **D-08:** Phase 7 should use file-based manifests and reports rather than a frontend workflow or database-backed batch state machine.
- **D-09:** Every sampling batch must record enough metadata to reproduce the exact selection: batch id, sampling mode, selected paper ids, source paths, timestamp, and any eligibility filters or exclusions applied.
- **D-10:** The output shape should align with the repository's existing bundle/report style so later phases can consume Phase 7 artifacts without inventing a new storage convention.

### Corpus Health Separation
- **D-11:** Missing directories, broken subpaths, unreadable files, and similar shared-corpus issues must be reported as corpus-health failures rather than mixed into model-quality metrics.
- **D-12:** Phase 7 reports should separate at least three categories: eligible and selected papers, corpus-health failures, and selected papers that are unavailable for downstream processing because of environment or ingestion gaps.

### Phase Handoff Boundary
- **D-13:** Phase 7 ends once the sampling baseline, reproducible batch metadata, and corpus-health reporting loop are in place.
- **D-14:** `L2` extraction quality comparisons belong to Phase 8, and bounded multi-paper packet construction belongs to Phase 9.

### the agent's Discretion
- Exact manifest/report filenames and directory layout for Phase 7 outputs
- The heuristic used to curate the first fixed regression set, as long as the chosen ten papers are intentionally stable and documented
- Whether the random batch is drawn purely uniformly or with lightweight guardrails (for example excluding unreadable entries or duplicates already in the fixed set)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project And Phase Framing
- `.planning/PROJECT.md` - Current `v1.1` milestone framing and the corpus-driven optimization goal
- `.planning/REQUIREMENTS.md` - `SAMPLE-01` through `SAMPLE-03` requirements that Phase 7 must satisfy
- `.planning/ROADMAP.md` - Phase 7 goal, success criteria, and handoff to Phases 8-11
- `.planning/STATE.md` - Current milestone state and active next phase

### Prior Phase Constraints
- `.planning/phases/04-replay-failure-taxonomy-and-l2-surgical-loop/04-CONTEXT.md` - Keeps `L2` repair work separate from baseline reporting setup
- `.planning/phases/05-multi-route-prior-induction/05-CONTEXT.md` - Preserves the rule that `L3/L4` depend on structured multi-route inputs, not arbitrary random paper mixes
- `.planning/phases/06-decision-episode-audit-export/06-CONTEXT.md` - Preserves the file-based, auditable artifact philosophy for downstream workflow surfaces

### Existing Sampling And Artifact Patterns
- `backend/app/graph/neo4j_client.py` - Existing inspiration-paper sampling helper and graph-side sampling patterns
- `backend/app/research_logic/replay_io.py` - Existing file-based summary, inspection, and manifest writing conventions
- `backend/scripts/run_replay_pilot.py` - Existing CLI pattern for file-based replay bundle generation
- `backend/scripts/run_route_state_package.py` - Existing CLI pattern for bounded package compilation
- `backend/scripts/export_decision_episode_pilot.py` - Existing CLI pattern for audited export bundle generation
- `backend/tests/test_replay_io.py` - Regression boundary for bundle/summary/inspection output structure
- `eval_quality.py` - Existing quality-evaluation script that can inform Phase 7 reporting shape, even if it is broader than the new sampling loop

### Codebase Guidance
- `.planning/codebase/STRUCTURE.md` - Where to place new backend scripts, planning artifacts, and tests
- `.planning/codebase/TESTING.md` - Existing testing expectations and the current gap around replay on real corpus inputs
- `.planning/codebase/INTEGRATIONS.md` - External-service and local-corpus constraints relevant to shared-corpus iteration

### External Corpus Root
- `\\192.168.199.138\Share400T\pub\LLM_Data\data\hzy\第一批文献\文献\文献中心\HZY第一批论文全文\output` - Shared corpus root that Phase 7 treats as the primary sampling inventory source

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `backend/scripts/run_replay_pilot.py`: existing file-driven runner pattern for later sampled-paper and packet workflows
- `backend/scripts/run_route_state_package.py`: existing packet compilation entrypoint pattern for Phase 9 handoff
- `backend/scripts/export_decision_episode_pilot.py`: existing file-driven export pattern that reinforces Phase 7's artifact style
- `backend/app/research_logic/replay_io.py`: central bundle writer/loader pattern that should inform Phase 7 manifest and report design
- `backend/app/graph/neo4j_client.py`: existing graph-side inspiration-paper sampling helper that can inform metadata fields or future enrichment, even though Neo4j is not the source-of-truth gate in Phase 7
- `eval_quality.py`: existing batch-style quality reporting script that shows how the project already aggregates per-paper outputs

### Established Patterns
- The project prefers typed/file-based artifacts and CLI scripts over productized UI or hidden database state for pilot workflows
- Replay, review, and export outputs are already summarized via manifest + summary + inspection surfaces
- The repo consistently keeps data availability problems distinct from semantic quality problems whenever contracts are strong enough to make that distinction explicit
- Multi-paper work is delayed until explicit packet boundaries exist; random single-paper sampling should not blur that rule

### Integration Points
- Phase 7 outputs should be easy for Phase 8 to consume for sampled single-paper evaluation
- Phase 7 inventory metadata should expose enough information for Phase 9 to construct a bounded packet candidate set from the same corpus
- New tests should likely live under `backend/tests/` and verify manifest/report generation rather than frontend interactions
- If a helper module is added, it should likely live near existing backend script and research-logic file IO patterns, not in frontend or planning-only code

</code_context>

<specifics>
## Specific Ideas

- The fixed regression set should be intentionally stable, not re-randomized every cycle
- The random exploration set exists to surface new edge cases rather than to replace the fixed set
- The shared corpus is large enough to support repeated random sampling, but messy enough that broken-path reporting is part of the baseline design
- A paper can be "present in corpus but not yet ready for downstream use," and that distinction should survive into reports

</specifics>

<deferred>
## Deferred Ideas

- Running the full `1000+` paper corpus end to end in Phase 7
- Using Neo4j ingestion state as the only eligibility gate for sampling
- Letting `L3/L4` consume arbitrary random paper batches directly
- Building a dedicated UI for corpus-batch management during this milestone
- Performing `L2` repair work inside the same phase as baseline sampling setup

</deferred>

---

*Phase: 07-corpus-sampling-and-regression-baseline*
*Context gathered: 2026-04-03*
