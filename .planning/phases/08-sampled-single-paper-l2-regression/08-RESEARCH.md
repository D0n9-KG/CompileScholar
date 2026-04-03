# Phase 8: Sampled Single-Paper L2 Regression - Research

**Researched:** 2026-04-03
**Domain:** sampled single-paper replay loops, per-paper `L2` quality measurement, and cross-iteration regression classification
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Phase 8 must reuse the existing ingest / rebuild / `PaperLogicTrace` compilation path instead of inventing a parallel extraction-only evaluator.
- The runner must surface the same quality outputs the production path already emits, including gate result, `quality_tier`, `quality_tier_score`, `quality_flags`, `l2_completeness_audit`, and recovery side effects that explain canonical trace quality.
- Corpus drift, missing preferred sources, and graph-unavailable records must remain separate from true `L2` quality failures.
- The Phase 7 fixed regression manifest and random batch artifacts are the source of truth for the first Phase 8 cycle.
- Phase 8 must execute from filesystem-backed sample references even when Neo4j metadata is absent, because the real Phase 7 baseline found `0/15` selected papers with graph metadata.
- Outputs must stay file-based and auditable under `tmp/` plus committed human-readable reports under `docs/replay/reports/`.
- Comparisons must separate recurring fixed-set failures, newly introduced fixed-set regressions, and new edge cases surfaced by the random sample.
- Phase 8 ends with a concrete prioritized short list of `L2` owners. It does not auto-expand into packet construction or in-phase repair work.
- Owner framing should be grounded in existing extraction signals such as slot recovery, relation stitching, `route_state_seed_thin`, metadata-summary mismatch, residual noise moves, reference recovery, and citation semantics.
- Phase 8 preserves the distinction between paper-level `L2` extraction symptoms and later multi-paper `L3/L4` compilation behavior.

### the agent's Discretion
- Exact CLI names and flag shapes
- Exact output bundle filenames as long as manifest / summary / inspection style remains auditable
- Exact iteration id strategy and report layout
- Exact owner-bucket rubric and thresholds as long as they map to existing quality signals
- Whether the rerun path is filesystem-only, graph-assisted, or hybrid as long as it remains faithful to production `L2` generation

### Deferred Ideas (OUT OF SCOPE)
- Full-corpus end-to-end evaluation in this phase
- Packet construction or multi-paper role assignment
- A new frontend for sampled-paper regression review
- Automatically fixing discovered `L2` issues in the same phase
</user_constraints>

<research_summary>
## Summary

Phase 8 should be built as a dedicated sampled-paper regression workflow layered on top of the existing Phase 7 sampling bundle and the current per-paper rebuild pipeline. The most important findings are:

1. The sample source and batch semantics already exist in Phase 7, so Phase 8 does not need a new sampler.
2. The rebuild path in `backend/app/ingest/rebuild.py` already performs the exact `L2` work this phase wants to measure: markdown parse, reference recovery, citation-event recovery, `run_phase1_paper_logic_trace(...)`, gate evaluation, artifact persistence, and graph updates.
3. `backend/app/paper_logic_trace/gates.py` already exposes the structured quality signals needed to form owner buckets and detect regressions without inventing a second scoring system.
4. The repo already prefers manifest / summary / inspection bundle contracts through `backend/app/research_logic/replay_io.py`, so Phase 8 should follow that pattern instead of writing ad hoc comparison JSON.
5. Because the real Phase 7 baseline selected `15/15` papers without Neo4j metadata, the Phase 8 runner must resolve selected papers from committed or local filesystem references first and treat graph availability as diagnostic metadata only.

That leads to the recommended Phase 8 architecture:

1. Add a dedicated sampled-paper runner that loads the Phase 7 fixed manifest plus one random batch reference and resolves each selected paper to a runnable filesystem source.
2. Reuse the existing rebuild / `PaperLogicTrace` path to generate per-paper outputs, but capture Phase 8 iteration metadata, per-paper result rows, and comparison-ready summaries in a dedicated bundle.
3. Normalize per-paper outcomes into explicit categories: `availability_issue`, `gate_failed`, `yellow_quality`, `green_quality`, and owner-signal buckets derived from `quality_flags` plus `l2_completeness_audit`.
4. Add an iteration comparison step that treats the fixed set as the strict regression surface and the random set as an edge-case discovery surface.
5. Publish one committed report per iteration under `docs/replay/reports/` that names the highest-leverage `L2` owners and keeps availability drift separate from extraction regressions.

**Primary recommendation:** implement one backend workflow slice for sampled execution and artifact capture, then a second slice for comparison classification and owner-priority reporting. Keep the plan centered on backend modules, CLI execution, and tests under `backend/tests/`.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Asset | Status | Purpose | Why Standard Here |
|-------|--------|---------|-------------------|
| `backend/scripts/run_corpus_sampling_baseline.py` | Existing | Operator entrypoint for the Phase 7 sample baseline | Best source for loading fixed/random inputs and preserving current batch semantics |
| `backend/app/research_logic/corpus_sampling.py` | Existing | Typed inventory and fixed/random batch models | Reuse the Phase 7 batch contract instead of redefining sample schemas |
| `backend/app/research_logic/replay_io.py` | Existing | Manifest / summary / inspection writer pattern | Best place to add a dedicated Phase 8 sampled-regression bundle contract |
| `backend/app/ingest/rebuild.py` | Existing | Canonical per-paper rerun path with artifact persistence and gate handling | Already performs the `L2` work Phase 8 wants to measure |
| `backend/app/extraction/orchestrator.py` | Existing | Shared phase-1 trace compilation entrypoint | Confirms Phase 8 can stay aligned with production extraction behavior |
| `backend/app/paper_logic_trace/gates.py` | Existing | Canonical `L2` quality scoring and audit signals | Source of truth for owner buckets and regression comparisons |
| `backend/app/paper_logic_trace/direct_extraction.py` | Existing | Slot, move, and relation extraction behavior | Helps map recurring failures to real extraction surfaces |

### Supporting
| Asset | Status | Purpose | When to Use |
|-------|--------|---------|-------------|
| `backend/tests/test_corpus_sampling.py` | Existing | Sample-batch reproducibility and file-backed selection behavior | Extend or reference when loading Phase 7 manifests in Phase 8 |
| `backend/tests/test_corpus_sampling_cli.py` | Existing | Operator CLI shape for sample bundles | Use as the baseline for a Phase 8 CLI smoke test style |
| `backend/tests/test_rebuild_gate.py` | Existing | Gate-related rebuild expectations | Reuse for confidence about gate-failure semantics |
| `backend/tests/test_paper_logic_trace_gates.py` | Existing | Structured quality and flag behavior | Best test boundary for owner-signal mapping assumptions |
| `backend/tests/test_replay_io.py` | Existing | Bundle writer contract coverage | Extend with a Phase 8 bundle format rather than creating an untested writer |
| `docs/replay/reports/phase7-corpus-sampling-baseline.md` | Existing | Human-readable baseline report shape | Reference when creating the first committed Phase 8 report |
| `tmp/phase7_corpus_sampling_baseline/` | Existing | First-cycle local sample source of truth | Use as the direct input for the first iteration rather than inventing a new sample contract |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| dedicated sampled-paper runner | manual one-off rebuild commands per paper | Too brittle and not auditable enough for repeated comparisons |
| reuse rebuild path | direct calls into extraction-only helpers | Faster to script, but breaks the user's requirement to stay faithful to production `L2` generation |
| separate fixed-vs-random comparison surfaces | a single merged failure table | Simpler, but destroys the distinction between regressions and new edge cases |
| owner buckets from existing quality signals | hand-labeled spreadsheet summaries | Easier short-term, but not reproducible or testable |
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Pattern 1: Phase 7 Bundle In, Phase 8 Bundle Out
**What:** Treat `tmp/phase7_corpus_sampling_baseline/` plus the committed fixed-set manifest as Phase 8 input artifacts, then emit a new Phase 8 iteration bundle under `tmp/`.
**When to use:** When a phase should extend a prior artifact contract instead of rediscovering its own inputs.
**Why recommended:** It keeps the first iteration grounded in the real baseline and makes later iteration comparisons explicit.

### Pattern 2: Filesystem-First Selected Paper Resolution
**What:** Resolve each sampled item from `preferred_source_path`, committed `corpus_relative_ref`, or other file-backed references before consulting graph metadata.
**When to use:** When selected items are known to be graph-unavailable but still valid evaluation inputs.
**Why recommended:** The real Phase 7 baseline had `0/15` selected items with Neo4j metadata, so graph-first execution would collapse the measured set incorrectly.

### Pattern 3: Rebuild-Backed `L2` Measurement
**What:** Run per-paper evaluation through the rebuild / phase-1 trace path so Phase 8 measures the same canonical trace and quality report used elsewhere in the system.
**When to use:** When the goal is to measure real `L2` behavior rather than build a synthetic benchmark path.
**Why recommended:** It respects the locked decision to reuse `run_phase1_paper_logic_trace(...)` and downstream recovery steps.

### Pattern 4: Comparison As Classification, Not Just Counts
**What:** Store both raw per-paper outputs and derived comparison classes such as `recurring_failure`, `new_regression`, `random_edge_case`, `availability_only`, and `quality_improved`.
**When to use:** When teams need to prioritize the next cycle from evidence rather than just inspect large raw tables.
**Why recommended:** Phase 8 must end with a short owner queue, not only aggregate counts.

### Pattern 5: Owner Buckets Derived From Existing Quality Signals
**What:** Map `quality_flags`, `l2_completeness_audit`, and trace-side artifacts into stable owner buckets like slot recovery, relation stitching, route-seed richness, metadata repair, reference recovery, and citation semantics.
**When to use:** When failures need to be actionable for the next optimization cycle.
**Why recommended:** The repo already emits the necessary structured signals, so Phase 8 should classify them rather than invent a second interpretation layer.

### Pattern 6: Separate Availability From Quality
**What:** Preserve corpus path drift, unreadable sources, and graph-unavailable rows as explicit execution-availability outcomes rather than merging them into `L2` regression counts.
**When to use:** Any time the execution environment is noisier than the extraction logic itself.
**Why recommended:** This carries forward the core Phase 7 discipline and prevents false regression signals.

### Anti-Patterns To Avoid
- Re-sampling the random batch inside the evaluator instead of consuming the Phase 7 selection source of truth.
- Measuring extraction by calling internal helpers that bypass rebuild-time recovery, gating, or artifact persistence.
- Treating missing filesystem sources as `L2` failures.
- Publishing a single blended summary that mixes fixed-set regressions and random-sample discoveries.
- Hand-writing owner conclusions in docs without storing the per-paper evidence that justified them.
</architecture_patterns>

<recommended_plan_slices>
## Recommended Plan Slices

### Slice 1: Sampled iteration runner and bundle contract
- Load Phase 7 fixed/random selections from the existing artifacts.
- Resolve each sampled paper to a filesystem-backed runnable source.
- Execute each paper through the rebuild-backed `L2` path while capturing availability outcomes separately.
- Write a dedicated Phase 8 iteration bundle with manifest, per-paper results, summary, and inspection surfaces.

### Slice 2: Comparison, owner bucketing, and operator reporting
- Compare the current iteration against a previous iteration or baseline-ready reference data.
- Classify fixed-set outcomes into recurring failures, regressions, improvements, and unchanged passes.
- Classify random-sample outcomes as edge-case discoveries, availability-only issues, or quality wins.
- Aggregate owner buckets from existing quality signals and publish a committed report under `docs/replay/reports/`.

These two slices should likely become separate PLAN files in separate waves, with bundle generation first and comparison/reporting second.
</recommended_plan_slices>

<validation_architecture>
## Validation Architecture

Phase 8 should validate at five levels:

1. **selection-loading and source-resolution tests**
   - prove the runner can load the Phase 7 fixed manifest and random selection inputs,
   - prove filesystem-backed selected papers remain runnable even when Neo4j metadata is absent,
   - prove missing preferred sources become availability issues instead of `L2` regressions.

2. **per-paper execution contract tests**
   - prove sampled execution captures per-paper trace, gate, schema, and evidence-slot quality outputs,
   - prove gate failures still persist enough artifacts for diagnosis,
   - prove recovery side effects such as reference and citation-event recovery are retained in result payloads.

3. **bundle writer contract tests**
   - prove the Phase 8 bundle writes manifest / summary / inspection outputs plus per-paper result files,
   - prove summary counts separate fixed-set, random-set, availability, and owner-bucket surfaces,
   - prove the inspection payload preserves enough data to replay paper-level diagnoses later.

4. **comparison classification tests**
   - prove fixed-set comparisons distinguish recurring failures, new regressions, improvements, and stable passes,
   - prove random-set comparisons surface new edge cases separately,
   - prove availability-only issues do not appear in regression counts.

5. **reporting smoke tests**
   - prove the committed report includes the evaluated cohorts, comparison results, prioritized owner queue, and explicit limits,
   - prove the owner queue is derived from structured signals rather than missing-data guesses.

**Expected commands:**
- Quick path: `cd backend; .\.venv\Scripts\python.exe -m pytest tests\\test_sampled_l2_regression.py tests\\test_replay_io.py -q`
- Full path: `cd backend; .\.venv\Scripts\python.exe -m pytest tests\\test_sampled_l2_regression.py tests\\test_sampled_l2_regression_cli.py tests\\test_replay_io.py tests\\test_paper_logic_trace_gates.py tests\\test_rebuild_gate.py -q`

The phase does not need to prove packet-level `L3/L4` behavior. It only needs to prove sampled single-paper `L2` execution, comparison, and owner prioritization are reproducible and auditable.
</validation_architecture>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Phase 8 result schema | a custom spreadsheet-like dict assembled inside the CLI | typed result models plus `replay_io.py` bundle writers | Keeps the output auditable and testable |
| paper execution | an extraction-only loop that never hits rebuild semantics | the existing rebuild / phase-1 trace pipeline | Preserves production-faithful `L2` behavior |
| owner diagnosis | manual prose tags disconnected from signals | a stable mapping from `quality_flags` and `l2_completeness_audit` | Makes prioritization reproducible |
| comparison logic | counts inferred from report text | explicit comparison categories stored in JSON | Prevents regression math from drifting across iterations |
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Missing file-backed sources get counted as regressions
**What goes wrong:** A paper that cannot be reopened from disk is reported as an `L2` quality failure.
**Why it happens:** Availability and extraction are measured in the same loop.
**How to avoid:** Record availability outcomes first and exclude them from quality-regression counts.
**Warning signs:** Fixed-set regression totals spike whenever the shared corpus path moves.

### Pitfall 2: The runner bypasses rebuild-time behavior
**What goes wrong:** Phase 8 measures a simplified extraction path and misses reference recovery, citation-event recovery, or canonical artifact persistence behavior.
**Why it happens:** Direct compiler calls look easier than the rebuild flow.
**How to avoid:** Keep the sampled evaluator on top of the existing rebuild / phase-1 trace path.
**Warning signs:** Per-paper outputs lack fields that the rebuild path usually persists.

### Pitfall 3: Fixed and random cohorts get merged
**What goes wrong:** The team cannot tell whether failures are regressions on known papers or genuinely new edge cases.
**Why it happens:** The summary only records one blended failure bucket.
**How to avoid:** Preserve separate fixed-set and random-set sections in both summary and inspection outputs.
**Warning signs:** A report talks about "new failures" without saying which cohort they came from.

### Pitfall 4: Owner buckets are too vague to drive the next cycle
**What goes wrong:** The report restates that quality is "mixed" without indicating whether the next work belongs in slots, relations, route seeds, metadata, or citation semantics.
**Why it happens:** Raw `quality_flags` are dumped without interpretation.
**How to avoid:** Create one stable signal-to-owner mapping and store the resulting bucket counts and exemplar papers.
**Warning signs:** The report ends with a broad "improve L2 quality" recommendation.

### Pitfall 5: Comparison logic depends on prose instead of stored artifacts
**What goes wrong:** Later iterations cannot be compared reliably because the previous run stored only a markdown report.
**Why it happens:** JSON outputs are treated as optional and the markdown report becomes the only record.
**How to avoid:** Make bundle summary and inspection files the authoritative comparison surface, with the markdown report as the human-readable layer.
**Warning signs:** Operators need to read long reports to determine whether something regressed.
</common_pitfalls>

<open_questions>
## Open Questions

1. **Should Phase 8 call the existing `rebuild_paper(...)` entrypoint directly or extract a filesystem-capable helper beneath it?**
   - What we know: the current rebuild flow starts from a Neo4j paper lookup and therefore may need a file-backed seam for graph-unavailable selected items.
   - Recommendation: plan for a small backend seam that preserves rebuild behavior while accepting filesystem-resolved sample entries.

2. **How should iteration identity be assigned?**
   - What we know: comparisons need a stable current-vs-previous reference, but the context leaves naming flexible.
   - Recommendation: support a user-provided iteration label with a timestamp fallback so reports and bundle directories stay traceable.

3. **Should the first Phase 8 report compare against a prior Phase 8 iteration or only classify the current baseline?**
   - What we know: there may be no older Phase 8 bundle on the first run.
   - Recommendation: allow first-run reports to emit `baseline_only` comparison state while still generating owner priorities and cohort summaries.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- `.planning/PROJECT.md`
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/STATE.md`
- `.planning/phases/08-sampled-single-paper-l2-regression/08-CONTEXT.md`
- `.planning/phases/07-corpus-sampling-and-regression-baseline/07-CONTEXT.md`
- `.planning/phases/07-corpus-sampling-and-regression-baseline/07-RESEARCH.md`
- `.planning/phases/07-corpus-sampling-and-regression-baseline/07-VERIFICATION.md`
- `backend/scripts/run_corpus_sampling_baseline.py`
- `backend/app/research_logic/corpus_sampling.py`
- `backend/app/research_logic/replay_io.py`
- `backend/app/ingest/rebuild.py`
- `backend/app/extraction/orchestrator.py`
- `backend/app/paper_logic_trace/gates.py`
- `backend/app/paper_logic_trace/direct_extraction.py`
- `backend/tests/test_corpus_sampling.py`
- `backend/tests/test_corpus_sampling_cli.py`
- `backend/tests/test_paper_logic_trace_gates.py`
- `backend/tests/test_rebuild_gate.py`
- `backend/tests/test_replay_io.py`

### Observed Runtime Evidence (HIGH confidence)
- `tmp/phase7_corpus_sampling_baseline/sampling_summary.json` reports `selected_with_neo4j_metadata_count = 0`
- Phase 7 fixed set contains ten committed ids and the first real random batch contains five selected ids
- The repo already writes auditable local bundles under `tmp/` and committed reports under `docs/replay/reports/`

### Secondary (MEDIUM confidence)
- `docs/replay/reports/phase7-corpus-sampling-baseline.md`
- `docs/superpowers/specs/2026-03-22-logickg-paperlogictrace-l2-redesign-design.md`
- `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md`
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: sampled-paper iteration loops, rebuild-backed `L2` execution, and structured comparison artifacts
- Ecosystem: `research_logic`, rebuild orchestration, `PaperLogicTrace` quality gates, backend CLI workflows, and file-based reports
- Patterns: Phase 7 bundle reuse, manifest / summary / inspection outputs, fixed-vs-random cohort separation, owner-bucket prioritization
- Pitfalls: graph-first execution, availability-vs-quality conflation, prose-only comparisons, vague owner queues

**Confidence breakdown:**
- Phase 7 artifact reuse: HIGH
- rebuild-path reuse for faithful `L2` measurement: HIGH
- owner-bucket derivation from existing quality signals: HIGH
- need for a file-backed rebuild seam: MEDIUM-HIGH
- exact iteration naming convention: MEDIUM
</metadata>
