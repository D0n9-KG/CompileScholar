# Phase 11: Iteration Prioritization And Next Cycle Plan - Research

**Researched:** 2026-04-03
**Domain:** evidence synthesis, next-cycle prioritization, and audit-grade summary/report generation
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Phase 11 must be a synthesis step over existing Phase 8 and Phase 10 evidence rather than a rerun of those workflows.
- Machine-readable summaries are the canonical inputs; markdown reports are operator-facing mirrors, not the source of truth.
- The phase must preserve two evidence taxonomies in one output: Phase 8 `L2` owner buckets and Phase 10 multi-paper blocker queues grouped by stage/layer.
- The output must convert those taxonomies into an explicit next-cycle queue with a primary focus, secondary follow-ups, and rationale.
- The current recommendation should rank packet construction first, `L4` follow-up second, and keep the Phase 8 `L2` queue visible as supporting evidence.
- Missing or stale upstream evidence must surface as explicit preflight blockers rather than being silently inferred from report prose.
- Phase 11 should remain a reporting/orchestration phase and must not change `PaperLogicTrace`, `RoutePacket`, route-state package validation, or export truth-boundary contracts.

### the agent's Discretion
- Exact scoring or ranking rules that combine Phase 8 owner buckets with Phase 10 blockers
- Exact output filenames, bundle layout, and operator-facing report structure
- Whether the summary is cycle-numbered, packet-numbered, or baseline-numbered, as long as provenance stays explicit

### Deferred / Out Of Scope
- Re-running sampled-paper extraction or multi-paper validation as part of prioritization
- Broadening the bounded packet or selecting a brand-new packet topic
- Fixing the selected owners inside Phase 11
- Building a dedicated UI or dashboard for the prioritization loop
</user_constraints>

<research_summary>
## Summary

Phase 11 should be planned as a thin orchestration-and-synthesis layer that reads already-produced Phase 8 and Phase 10 artifacts, normalizes their key signals into one shared prioritization model, and emits two outputs:

1. a machine-readable Phase 11 summary bundle under `tmp/`
2. a committed markdown report under `docs/replay/reports/`

The important technical observation is that the repo already has the two evidence contracts Phase 11 needs:

- Phase 8 exposes stable sampled-paper comparison summaries with fixed/random verdict counts and ranked `owner_buckets` via `backend/app/research_logic/sampled_single_paper.py` and `backend/app/research_logic/replay_io.py`.
- Phase 10 exposes stage-aware comparison summaries with `package`, `replay`, `prior_review`, `export`, and a structured `blocker_queue` via `backend/app/research_logic/phase10_multi_paper_validation.py`.

What does not exist yet is the Phase 11 seam that joins those contracts together and turns them into an explicit next-cycle decision. That means the phase should not redesign earlier logic; it should:

1. load Phase 8 and Phase 10 evidence from explicit paths,
2. validate that the required upstream artifacts are present and mutually intelligible,
3. derive one prioritized recommendation queue,
4. render a summary that makes the decision auditable and replayable.

The main planning consequence is that Phase 11 is mostly backend reporting/orchestration work:

- one new research-logic module for loading and synthesizing upstream summaries,
- one CLI script for operators,
- one report renderer,
- and focused tests around missing-input handling, ranking, bundle/report output, and provenance.

**Primary recommendation:** split planning into one implementation plan for the synthesis/ranking contract and one implementation plan for operator-facing reporting plus real-run artifact generation. Keep the final execution phase grounded in the already committed Phase 8/10 evidence rather than recomputing either source.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Asset | Status | Purpose | Why Standard Here |
|-------|--------|---------|-------------------|
| `backend/app/research_logic/sampled_single_paper.py` | Existing | Canonical owner-bucket and sampled comparison logic | Phase 11 should consume these outputs, not recreate owner heuristics |
| `backend/app/research_logic/replay_io.py` | Existing | Summary / inspection builders and bundle-writing patterns | Natural place to mirror the Phase 8 style for a new Phase 11 bundle |
| `backend/app/research_logic/phase10_multi_paper_validation.py` | Existing | Structured stage-level comparison summary and `blocker_queue` | Phase 11 should read this contract directly when available |
| `backend/scripts/run_sampled_single_paper_l2_regression.py` | Existing | CLI shape for summary-first bundle/report workflows | Best operator pattern to mirror for Phase 11 |
| `backend/scripts/run_phase10_multi_paper_validation.py` | Existing | CLI shape for structured comparison plus report rendering | Shows the exact summary-first report pattern Phase 11 should reuse |
| `docs/replay/reports/phase8-sampled-single-paper-l2-regression-baseline.md` | Existing | Operator-facing owner queue and fixed/random cohort framing | Confirms what needs to stay visible in the final synthesis |
| `.planning/phases/10-multi-paper-l3-and-l4-validation/10-VERIFICATION.md` | Existing | Explicit recommendation logic grounded in runtime evidence | Strongest source for the initial packet-first recommendation |

### Supporting
| Asset | Status | Purpose | When to Use |
|-------|--------|---------|-------------|
| `tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/comparison_summary.json` | Existing | Canonical Phase 8 machine-readable summary | Primary source for owner buckets and verdict counts |
| `tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/comparison_inspection.json` | Existing | Detailed per-paper comparison evidence | Use for provenance and deeper report sections |
| `docs/replay/reports/phase10-multi-paper-l3-l4-validation.md` | Existing | Committed human-readable Phase 10 report | Secondary operator-facing input when JSON is not available locally |
| `backend/tests/test_replay_io.py` | Existing | Summary/bundle contract tests | Best place to mirror Phase 11 bundle writer expectations |
| `backend/tests/test_phase10_multi_paper_validation.py` | Existing | Comparison-summary integration coverage | Reference for stage/blocker summary tests |
| `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md` | Existing | Stable layer semantics | Ensures packet/L4 decisions do not drift into schema changes |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| one new Phase 11 synthesis module | ad hoc shell scripts that grep reports | Faster short-term, but not durable or testable |
| JSON-first provenance | markdown-only parsing of Phase 8/10 reports | Easier to read, but fragile and not suitable for repeatable prioritization |
| explicit input-path parameters | hard-coded `tmp/phase10_multi_paper_validation/baseline/...` | Simpler, but breaks as soon as machine-local tmp artifacts are missing |
| explicit queue output | a prose-only recommendation paragraph | Less work, but fails the roadmap requirement to make the iteration non-open-ended |
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Pattern 1: Upstream Summary In, Phase 11 Summary Out
**What:** Treat Phase 8 and Phase 10 machine-readable artifacts as the only authoritative inputs, then emit a new Phase 11 summary/report pair.  
**When to use:** When a phase exists to make a decision from prior evidence rather than produce new lower-level runtime artifacts.  
**Why recommended:** It preserves auditability and avoids recomputation drift.

### Pattern 2: Explicit Provenance And Preflight Validation
**What:** Require explicit source refs for each upstream evidence set and record which files were used in the Phase 11 bundle.  
**When to use:** When upstream `tmp/` bundles may be missing or machine-local.  
**Why recommended:** The current workspace still has Phase 8 bundle files but not the expected Phase 10 baseline comparison bundle path named in verification.

### Pattern 3: Keep Taxonomies Separate, Then Rank
**What:** Preserve Phase 8 owner-bucket evidence and Phase 10 stage/layer blocker evidence as separate surfaces before combining them into one next-cycle queue.  
**When to use:** When the recommendation depends on different layers of the system failing for different reasons.  
**Why recommended:** It prevents packet issues from masking `L2` issues and vice versa.

### Pattern 4: Recommendation Built From Structured Signals
**What:** Derive the primary focus from stable structured fields such as owner-bucket counts, package/replay deltas, blocker queue contents, and readiness flags.  
**When to use:** When the team needs to explain why a given path was prioritized.  
**Why recommended:** The current recommendation already rests on “replay `l2` stayed flat while new blockers appeared first in package validation,” which is a structured claim.

### Pattern 5: Report Rendered From Summary, Not Hand-Written
**What:** Follow the Phase 10 pattern where the markdown report is generated from a machine-readable summary instead of hand-authored first.  
**When to use:** Any time a human-readable conclusion needs to stay replayable.  
**Why recommended:** It keeps the decision computable for later cycles and easier to compare across iterations.

### Anti-Patterns To Avoid
- Re-running Phase 8 or Phase 10 during prioritization instead of reading their existing outputs
- Ranking solely from markdown report prose instead of structured JSON surfaces
- Hiding missing upstream evidence by silently falling back to partial data
- Flattening packet, replay, prior, export, and `L2` signals into one blended severity number with no provenance
</architecture_patterns>

<recommended_plan_slices>
## Recommended Plan Slices

### Slice 1: Phase 11 synthesis contract
- Add a dedicated backend research-logic module, likely `backend/app/research_logic/iteration_prioritization.py`, that loads Phase 8 and Phase 10 evidence.
- Define typed models for input provenance, normalized evidence surfaces, and the next-cycle recommendation queue.
- Implement preflight validation for missing or inconsistent inputs, especially around absent Phase 10 JSON artifacts.
- Produce a structured Phase 11 summary/inspection bundle and focused tests.

### Slice 2: CLI and report rendering
- Add an operator CLI, likely `backend/scripts/run_phase11_iteration_prioritization.py`, that accepts explicit Phase 8 / Phase 10 input paths and output locations.
- Render a committed markdown report under `docs/replay/reports/` from the structured Phase 11 summary.
- Include recommendation sections for primary focus, secondary follow-up, deferred paths, and source provenance.
- Add CLI smoke tests and report assertions.

### Slice 3: Real evidence run and verification
- Run the new Phase 11 workflow against the committed/current Phase 8 and Phase 10 evidence chain.
- Commit the generated report plus a verification note that states exactly which artifacts were used.
- Confirm the recommendation is still packet-first for the current evidence chain, or explicitly record why it changed.

The first two slices are the natural planning boundary; the third is the real-run closeout that should only happen after synthesis and reporting contracts are stable.
</recommended_plan_slices>

<validation_architecture>
## Validation Architecture

Phase 11 should validate at five levels:

1. **input-loading and provenance tests**
   - prove Phase 11 can load Phase 8 comparison summaries and inspections,
   - prove it can load Phase 10 comparison data when present,
   - prove missing Phase 10 JSON artifacts become explicit preflight blockers rather than silent fallbacks.

2. **normalization and ranking tests**
   - prove owner buckets stay visible in the normalized Phase 11 evidence model,
   - prove stage/layer blocker queues remain attributable after normalization,
   - prove current packet-first logic wins when `L2` deltas stay flat and package validation blockers are new.

3. **bundle writer contract tests**
   - prove Phase 11 writes a manifest, summary, and inspection bundle under `tmp/`,
   - prove source artifact refs are included in the output,
   - prove the queue includes primary focus, secondary follow-up, and supporting evidence.

4. **report rendering tests**
   - prove the markdown report is rendered from the structured summary,
   - prove the report includes the selected primary focus, visible supporting owner queues, and explicit rationale,
   - prove missing upstream JSON inputs are disclosed in the report if fallback sources were used.

5. **real-run verification**
   - run the Phase 11 CLI against the current evidence chain,
   - confirm the output chooses between `L2`, packet construction, and `L4` on explicit evidence,
   - confirm the report and verification note identify the exact upstream files used.

**Expected commands:**
- Quick path: `cd backend; .\.venv\Scripts\python.exe -m pytest tests\\test_iteration_prioritization.py tests\\test_replay_io.py -q`
- Full path: `cd backend; .\.venv\Scripts\python.exe -m pytest tests\\test_iteration_prioritization.py tests\\test_iteration_prioritization_cli.py tests\\test_replay_io.py tests\\test_phase10_multi_paper_validation.py tests\\test_sampled_single_paper_regression.py -q`

The phase does not need to prove that the recommended path is already fixed. It only needs to prove that the decision process is reproducible, explicit, and grounded in current evidence.
</validation_architecture>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Phase 11 queue logic | markdown-only heuristics in the CLI | typed synthesis module + structured summary | Keeps the prioritization testable |
| upstream evidence parsing | regex over committed reports | existing JSON summary contracts where available | Avoids brittle prose coupling |
| recommendation provenance | implicit assumptions in report text | explicit source refs in summary and inspection outputs | Makes later cycles replayable |
| missing-input handling | silent defaults to whatever file exists | preflight blockers and explicit fallback notes | Prevents false confidence |
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Phase 11 quietly re-runs upstream logic
**What goes wrong:** The prioritization output reflects fresh runtime drift rather than the evidence chain that closed Phases 8-10.  
**Why it happens:** It seems convenient to call existing CLIs again instead of reading their outputs.  
**How to avoid:** Treat Phase 8 and Phase 10 artifacts as immutable inputs to Phase 11.  
**Warning signs:** The Phase 11 report changes when the environment changes, even with the same input paths.

### Pitfall 2: Phase 10 JSON assumptions are too rigid
**What goes wrong:** The CLI hard-codes the missing `tmp/phase10_multi_paper_validation/baseline/comparison_summary.json` path and fails on otherwise usable evidence.  
**Why it happens:** Verification mentions that path as the source of truth, but the current workspace does not retain it.  
**How to avoid:** Support explicit Phase 10 input paths and report what was actually used.  
**Warning signs:** The only failure is “file not found” even though the committed report and verification note exist.

### Pitfall 3: Recommendation loses layer boundaries
**What goes wrong:** The final queue says “improve quality” without indicating whether the issue is `relation_assembly`, support density, alternative distinctness, or prior induction.  
**Why it happens:** Owner buckets and blocker queues are collapsed too early.  
**How to avoid:** Keep upstream taxonomies intact until the final ranking step.  
**Warning signs:** The summary cannot explain why packet construction beat `L2` or `L4`.

### Pitfall 4: Report is stronger than evidence
**What goes wrong:** The markdown report makes a confident recommendation but the structured summary does not preserve enough rationale or provenance to justify it.  
**Why it happens:** Narrative writing gets ahead of the data model.  
**How to avoid:** Render the report from the summary and assert key reasoning fields in tests.  
**Warning signs:** The report mentions priorities that do not appear in the bundle JSON.
</common_pitfalls>

<open_questions>
## Open Questions

1. **Should Phase 11 require Phase 10 JSON summaries, or allow a documented fallback to committed verification/report files?**  
   Recommendation: require explicit Phase 8 JSON, allow Phase 10 fallback only when JSON is absent, and record the fallback in provenance.

2. **How should packet-first vs `L4`-first ties be broken if later cycles reduce package blockers but keep prior-induction failures high?**  
   Recommendation: make stage-of-first-new-failure the primary rule, then use downstream blocker volume as a secondary tiebreaker.

3. **Should the queue produce only one recommended next cycle or also a ranked top three?**  
   Recommendation: emit one primary recommendation and at least two ranked follow-ups so the iteration does not become open-ended if the team wants alternatives.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- `.planning/PROJECT.md`
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/STATE.md`
- `.planning/phases/11-iteration-prioritization-and-next-cycle-plan/11-CONTEXT.md`
- `.planning/phases/08-sampled-single-paper-l2-regression/08-CONTEXT.md`
- `.planning/phases/09-bounded-packet-construction-from-corpus/09-CONTEXT.md`
- `.planning/phases/10-multi-paper-l3-and-l4-validation/10-CONTEXT.md`
- `.planning/phases/10-multi-paper-l3-and-l4-validation/10-VERIFICATION.md`
- `docs/replay/reports/phase8-sampled-single-paper-l2-regression-baseline.md`
- `docs/replay/reports/phase10-multi-paper-l3-l4-validation.md`
- `tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/comparison_summary.json`
- `tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/comparison_inspection.json`
- `backend/app/research_logic/sampled_single_paper.py`
- `backend/app/research_logic/replay_io.py`
- `backend/app/research_logic/phase10_multi_paper_validation.py`
- `backend/scripts/run_sampled_single_paper_l2_regression.py`
- `backend/scripts/run_phase10_multi_paper_validation.py`
- `backend/tests/test_replay_io.py`
- `backend/tests/test_phase10_multi_paper_validation.py`

### Observed Runtime Evidence (HIGH confidence)
- The current workspace still contains the full Phase 8 baseline cycle bundle under `tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/`.
- The current workspace does not retain the Phase 10 `baseline/comparison_summary.json` path named in verification; only generated bridge artifacts remain under `tmp/phase10_multi_paper_validation/`.
- The committed Phase 10 verification note already states that `packet construction`, not `L2`, is the current highest-leverage next-cycle target because `replay.delta.failure_counts_by_layer_delta.l2 = 0` while new blockers appear first in package validation.

### Secondary (MEDIUM confidence)
- `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md`
- `docs/superpowers/specs/2026-04-01-logickg-route-packet-schema.md`
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: backend evidence synthesis, summary/report rendering, and reproducible prioritization
- Ecosystem: `research_logic`, existing Phase 8/10 CLIs, bundle writers, verification notes, and pytest-based backend verification
- Patterns: JSON-first source of truth, explicit provenance, preserved taxonomies, summary-derived markdown reports
- Pitfalls: missing upstream artifacts, implicit fallbacks, taxonomy collapse, prose outrunning evidence

**Confidence breakdown:**
- Need for a dedicated Phase 11 synthesis/reporting seam: HIGH
- Reuse of Phase 8 owner-bucket and Phase 10 blocker-summary contracts: HIGH
- Need for explicit input-path and provenance handling: HIGH
- Exact ranking heuristic details: MEDIUM
</metadata>
