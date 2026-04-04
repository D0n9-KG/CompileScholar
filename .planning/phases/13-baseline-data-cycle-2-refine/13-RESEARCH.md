# Phase 13: Baseline Data Cycle 2 (Refine) - Research

**Researched:** 2026-04-04
**Domain:** repeated bounded rerun / review refinement on the existing multi-paper replay pipeline
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Phase 13 must keep `tmp/phase12_direct_fix_cycle/cycle1/` as the canonical comparison baseline instead of falling back to the older Phase 10 / 11 evidence chain.
- Phase 13 must reuse the existing Phase 10 execution surface (`package -> replay -> prior-review -> export`) on the frozen Phase 9 packet boundary.
- The phase is explicitly allowed to perform multiple bounded iterations inside the same phase rather than closing after one rerun.
- The lead refinement surface is downstream, not packet-construction-first anymore: `reviewer_missing`, the remaining `decision_prior_card` failure, empty prior candidates, and `weak_prior_support` should shape the work.
- `yellow_route_state_present` should remain an explicit residual signal if it persists; the implementation must not hide it.
- Phase closeout must include manual defect-delta review across package, replay, prior-review, and export, plus a next-cycle recommendation.

### the agent's Discretion
- Exact cycle / iteration directory naming inside the new Phase 13 output root
- Whether to refresh prioritization after every inner iteration or only once at the final handoff
- Exact shape of any thin orchestration wrapper around the existing Phase 10 runner

### Deferred / Out Of Scope
- Widening the bounded packet with new papers
- Building a separate UI or a new runtime foundation for cycle management
- Claiming milestone-level stability inside this phase
- Generalizing quality claims beyond the current computational-mechanics slice
</user_constraints>

<research_summary>
## Summary

Phase 13 should be planned as a bounded iterative refinement loop layered on top of the already-working Phase 10 runner and the fresh Phase 12 cycle-1 outputs. The key planning move is to separate:

1. iteration infrastructure and baseline wiring,
2. downstream blocker repair,
3. repeated real reruns on the same bounded slice,
4. explicit manual defect-delta review and reprioritization.

The codebase already contains almost everything needed for the loop:

- `backend/scripts/run_phase10_multi_paper_validation.py` already supports explicit `--baseline-replay-bundle` and `--baseline-export-bundle`, so Phase 13 can compare directly against Phase 12 cycle 1 without inventing a new comparison layer.
- `backend/app/research_logic/historical_replay_compiler.py` already models `reviewer_missing` as a blocking replay failure, so reviewer metadata is a real lever, not just a reporting afterthought.
- `backend/app/research_logic/prior_induction.py` already explains why prior review remained empty in cycle 1: it only emits reusable prior candidates when at least two support route states fall into the same cluster.
- `backend/app/research_logic/iteration_prioritization.py` already provides the ranking vocabulary Phase 13 should use for the final handoff.

The main missing seam is not a new compiler. It is a repeatable Phase 13 refinement loop that can:

- preserve cycle-1 as the baseline,
- write multiple inner iterations under a fresh Phase 13 output root,
- expose exactly which code or metadata changes were tested in each iteration,
- and determine whether quality movement is real across the full package/replay/prior/export surface.

**Primary planning recommendation:** split Phase 13 into three plans:

1. build or refine the iteration harness and baseline-anchored comparison/report contract,
2. implement the targeted downstream fixes and supporting regression tests,
3. run repeated bounded reruns and produce the final review / reprioritization artifacts.

That split matches the user correction: multiple iterations should happen inside Phase 13, but they should remain auditable and bounded rather than becoming an open-ended manual loop.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Asset | Status | Purpose | Why Standard Here |
|-------|--------|---------|-------------------|
| `backend/app/research_logic/phase10_multi_paper_validation.py` | Existing | Canonical bounded multi-stage rerun pipeline and comparison summary generator | Reuse is required by context and avoids a second execution surface |
| `backend/scripts/run_phase10_multi_paper_validation.py` | Existing | Operator-facing CLI with baseline override support | Natural entrypoint for repeated bounded reruns |
| `backend/app/research_logic/historical_replay_compiler.py` | Existing | Replay-quality gating and failure-record generation | Contains the real `reviewer_missing` blocking logic |
| `backend/app/research_logic/prior_induction.py` | Existing | Cluster-driven prior / anti-pattern candidate generation | Explains and controls the empty prior surface |
| `backend/app/research_logic/replay_io.py` | Existing | Summary / inspection / bundle writers | Best place to preserve audit-grade multi-iteration outputs |
| `backend/app/research_logic/iteration_prioritization.py` | Existing | Ranked next-cycle recommendation synthesis | Gives Phase 13 a ready-made reprioritization contract |

### Supporting
| Asset | Status | Purpose | When to Use |
|-------|--------|---------|-------------|
| `tmp/phase12_direct_fix_cycle/cycle1/comparison_summary.json` | Existing | Canonical cycle-1 defect baseline | Always use as the top-level success comparison anchor |
| `tmp/phase12_direct_fix_cycle/cycle1/replay_bundle/replay_inspection.json` | Existing | Failure records and stage-quality detail | Use to drive downstream-fix task decomposition |
| `tmp/phase12_direct_fix_cycle/cycle1/prior_review_bundle/candidate_review_summary.json` | Existing | Explains empty prior / anti-pattern output and cluster shape | Use when planning support-cluster and reviewer fixes |
| `tmp/phase12_direct_fix_cycle/cycle1/export_bundle/export_summary.json` | Existing | Export readiness and `weak_prior_support` surface | Use for final defect-delta review |
| `docs/replay/reports/phase12-cycle1-direct-fix.md` | Existing | Human defect-delta baseline | Reuse its reporting pattern for cycle-2 and later inner iterations |
| `backend/tests/test_phase10_multi_paper_validation.py` | Existing | Contract coverage for bundle generation and comparison surfaces | Best template for new phase-level regression tests |
| `backend/tests/test_historical_replay_compiler.py` | Existing | Verifies reviewer ids and prior-card quality gating | Best template for targeted replay-failure fixes |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| baseline-anchored repeated reruns | compare only the latest rerun against Phase 10 / 11 | Easier operationally, but breaks the Phase 13 repeatability goal |
| a thin Phase 13 orchestration wrapper | manually rerun the Phase 10 CLI with ad hoc commands each time | Faster once, but loses auditability across multiple inner iterations |
| targeted downstream repair | reopening packet-construction work | Misaligned with current blocker surface and wastes the Phase 12 packet fix |
| existing prioritization taxonomy | prose-only “what next” notes | Harder to test, reuse, and compare across cycles |
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Pattern 1: One Fixed Baseline, Many Inner Iterations
**What:** Keep Phase 12 cycle 1 as the canonical baseline while allowing several Phase 13 inner iterations (for example `iter-01`, `iter-02`, `cycle2-final`) under one new output root.  
**When to use:** When the phase needs repeated bounded refinement without moving the success anchor.  
**Why recommended:** It satisfies the user’s request for multiple iterations while preserving one stable measurement point.

### Pattern 2: Thin Wrapper Around The Existing Phase 10 Runner
**What:** Introduce a small Phase 13 orchestration seam, if needed, that calls the existing Phase 10 runner with explicit baseline inputs and cycle-specific output paths rather than forking the core runtime logic.  
**When to use:** When repeated reruns need naming, provenance, and report conveniences, but not a new execution engine.  
**Why recommended:** It preserves the current runner contract and keeps fixes localized.

### Pattern 3: Separate “Enable Audited Priors” From “Improve Route Content”
**What:** Treat reviewer metadata wiring and support-cluster shape as distinct but related levers for clearing `reviewer_missing`, prior-candidate emptiness, and export weakness.  
**When to use:** When prior review is blocked by both process metadata and cluster topology.  
**Why recommended:** Cycle 1 shows both issues at once, and either one alone may fail to move the export surface.

### Pattern 4: Final Phase Verdict Uses The Best / Final Inner Iteration, Not The First Attempt
**What:** Allow several inner iterations, but require the final report and verification note to explicitly compare the chosen final iteration against cycle 1 and to summarize what changed across inner iterations.  
**When to use:** When the phase contains multiple attempts but still needs one stable handoff artifact.  
**Why recommended:** It keeps execution flexible without making closeout ambiguous.

### Pattern 5: Reprioritization Uses Existing Structured Signals
**What:** Feed the final iteration’s package/replay/prior/export surfaces into the current blocker / recommendation taxonomy rather than inventing a one-off Phase 13 conclusion format.  
**When to use:** At final closeout after the last bounded iteration.  
**Why recommended:** It keeps Phase 14 input aligned with the existing `packet_construction` / `l4_aggregation` / `l2_extraction` vocabulary.

### Anti-Patterns To Avoid
- Treating Phase 13 as a single rerun despite the explicit multi-iteration requirement
- Comparing later inner iterations only to the immediately previous inner iteration and forgetting the cycle-1 anchor
- Solving `reviewer_missing` by hiding the flag instead of supplying real reviewer metadata
- Hand-authoring final review prose without citing the structured cycle outputs
- Reopening packet membership or role mapping to manufacture easier downstream priors
</architecture_patterns>

<recommended_plan_slices>
## Recommended Plan Slices

### Slice 1: Phase 13 iteration harness and baseline contract
- Add or refine a thin orchestration/reporting seam for Phase 13 repeated runs.
- Standardize Phase 13 output roots and iteration labels.
- Ensure every iteration passes Phase 12 cycle-1 replay/export bundles as the baseline.
- Add contract tests proving the baseline wiring and iteration output structure.

### Slice 2: Downstream blocker repair
- Expose or thread reviewer metadata through the Phase 13 rerun surface where appropriate.
- Improve the conditions for prior candidate generation or support-cluster reuse on the bounded slice without broadening packet scope.
- Add focused tests around `reviewer_missing`, prior candidate generation, and resulting export posture.

### Slice 3: Repeated bounded reruns plus closeout
- Run multiple bounded Phase 13 iterations on the same packet and cycle-1 baseline.
- Keep a machine-readable and markdown trail for each meaningful iteration.
- Produce one final Phase 13 report / verification note that compares the chosen final iteration against cycle 1 and records the next-cycle recommendation.

The natural planning split is therefore 3 plans across 3 waves:
- Wave 1: iteration harness / baseline contract
- Wave 2: downstream blocker fixes
- Wave 3: repeated real reruns, manual review, and reprioritization
</recommended_plan_slices>

<validation_architecture>
## Validation Architecture

Phase 13 should validate at five levels:

1. **baseline and iteration wiring tests**
   - prove the Phase 13 runner or helper always points to `tmp/phase12_direct_fix_cycle/cycle1/` for baseline replay/export inputs,
   - prove each inner iteration writes to a distinct cycle-scoped output root,
   - prove final comparison still references cycle 1 rather than the immediately previous inner run alone.

2. **downstream blocker regression tests**
   - prove reviewer metadata, when provided, reaches replay summaries and affects `reviewer_missing`,
   - prove prior-induction clustering behavior is visible and tested rather than inferred from report prose,
   - prove export readiness / `weak_prior_support` changes are reflected in structured outputs.

3. **bundle / report contract tests**
   - prove each Phase 13 iteration writes the expected package, replay, prior-review, export, and comparison artifacts,
   - prove report sections cover package, replay, prior-review, export, and blocker delta,
   - prove intermediate iteration metadata or naming stays auditable.

4. **manual defect-delta validation**
   - a reviewer should compare the final iteration against cycle 1 and confirm whether quality movement is real across the full bounded slice,
   - manual review must state whether blocker movement was broad or only shifted within one stage,
   - the report must identify remaining defects for the next cycle.

5. **reprioritization validation**
   - prove the final handoff uses the existing prioritization taxonomy or an equivalent structured queue,
   - prove the recommendation is based on final iteration artifacts, not stale Phase 11 guidance.

**Expected commands:**
- Quick path: `cd backend; .\.venv\Scripts\python.exe -m pytest tests\\test_phase10_multi_paper_validation.py tests\\test_historical_replay_compiler.py tests\\test_replay_io.py -q`
- Full path: `cd backend; .\.venv\Scripts\python.exe -m pytest tests\\test_phase10_multi_paper_validation.py tests\\test_phase10_multi_paper_validation_cli.py tests\\test_historical_replay_compiler.py tests\\test_replay_io.py tests\\test_iteration_prioritization.py -q`

Phase 13 does not need to prove final milestone stability. It only needs to prove that repeated bounded refinement inside the phase is auditable and that the final iteration shows real movement against cycle 1.
</validation_architecture>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| repeated rerun orchestration | a brand-new replay/export engine | thin wrapper around `run_phase10_multi_paper_validation.py` | Keeps one execution surface |
| reviewer handling | markdown-only notes claiming who reviewed | actual `reviewer_ids` wiring through replay / prior paths | Needed for real blocker movement |
| final reprioritization | prose-only “next steps” | `iteration_prioritization.py` style structured queue | Easier to compare across cycles |
| phase closeout | one huge report with no iteration provenance | per-iteration artifacts plus one final synthesized review | Preserves auditability across multiple inner iterations |
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Multi-iteration intent collapses back into one rerun
**What goes wrong:** The phase technically “passes” after a single rerun, but the user’s repeatability goal is not met.  
**Why it happens:** Execution work gets framed around the roadmap’s “cycle 2” name instead of the clarified “multiple bounded iterations within phase 13.”  
**How to avoid:** Make iteration cadence an explicit plan artifact, not just a narrative note.  
**Warning signs:** Only one new output root appears under the Phase 13 `tmp/` path.

### Pitfall 2: Reviewer metadata is discussed but never wired into the actual runner
**What goes wrong:** Reports keep saying `reviewer_missing` even after manual review happened outside the pipeline.  
**Why it happens:** `run_replay_pilot.py` supports reviewers, but the Phase 10 multi-paper rerun surface does not currently expose the same path.  
**How to avoid:** Add an explicit task to decide how reviewer ids enter the repeated rerun contract and test it.  
**Warning signs:** Structured outputs still show empty reviewer lists even after review-related code changes.

### Pitfall 3: Prior-candidate emptiness is treated as a pure review problem
**What goes wrong:** Reviewer metadata is added, but export still stays weak because support routes never cluster into reusable priors.  
**Why it happens:** The empty prior surface has two causes in cycle 1: missing reviewer metadata and all support clusters being singleton clusters.  
**How to avoid:** Plan and verify both levers separately.  
**Warning signs:** `reviewer_missing` disappears but `prior_candidate_count` stays `0` with the same cluster topology.

### Pitfall 4: Final review compares only against the previous inner iteration
**What goes wrong:** Local progress looks good, but the final report cannot prove movement against cycle 1.  
**Why it happens:** Inner-loop convenience displaces the canonical baseline.  
**How to avoid:** Require final closeout to cite cycle 1 explicitly.  
**Warning signs:** The final report references `iter-02 -> iter-03` deltas but never mentions cycle 1.
</common_pitfalls>

<open_questions>
## Open Questions

1. **Should Phase 13 produce one final named `cycle2` artifact root plus intermediate dev iterations, or should every committed rerun inside the phase be named as its own iteration?**  
   Recommendation: allow multiple inner iteration roots, but reserve one final committed “cycle2-final” or equivalent closeout root for the phase verdict.

2. **Is reviewer metadata enough to move the replay/prior/export surface, or must support clustering be improved in the same phase?**  
   Recommendation: assume both need explicit tasks, because cycle 1 evidence shows both issues simultaneously.

3. **How often should reprioritization run inside the phase?**  
   Recommendation: lightweight notes after meaningful inner iterations, but one formal reprioritization only at the final closeout iteration.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- `.planning/PROJECT.md`
- `.planning/REQUIREMENTS.md`
- `.planning/ROADMAP.md`
- `.planning/STATE.md`
- `.planning/phases/13-baseline-data-cycle-2-refine/13-CONTEXT.md`
- `.planning/phases/12-baseline-data-cycle-1-direct-fix/12-CONTEXT.md`
- `.planning/phases/12-baseline-data-cycle-1-direct-fix/12-VERIFICATION.md`
- `docs/replay/reports/phase12-cycle1-direct-fix.md`
- `tmp/phase12_direct_fix_cycle/cycle1/comparison_summary.json`
- `tmp/phase12_direct_fix_cycle/cycle1/replay_bundle/replay_summary.json`
- `tmp/phase12_direct_fix_cycle/cycle1/replay_bundle/replay_inspection.json`
- `tmp/phase12_direct_fix_cycle/cycle1/prior_review_bundle/candidate_review_summary.json`
- `tmp/phase12_direct_fix_cycle/cycle1/export_bundle/export_summary.json`
- `backend/app/research_logic/phase10_multi_paper_validation.py`
- `backend/scripts/run_phase10_multi_paper_validation.py`
- `backend/app/research_logic/historical_replay_compiler.py`
- `backend/app/research_logic/prior_induction.py`
- `backend/app/research_logic/decision_prior_builder.py`
- `backend/app/research_logic/anti_pattern_builder.py`
- `backend/app/research_logic/replay_io.py`
- `backend/app/research_logic/iteration_prioritization.py`
- `backend/tests/test_phase10_multi_paper_validation.py`
- `backend/tests/test_historical_replay_compiler.py`

### Observed Runtime Evidence (HIGH confidence)
- Phase 12 cycle 1 removed `support_cluster_too_small` and `alternative_scope_not_distinct`, but left `yellow_route_state_present`, `reviewer_missing`, one replay `decision_prior_card` failure, empty prior candidates, and `weak_prior_support`.
- `candidate_review_summary.json` shows `cluster_count = 3` and `cluster_support_counts` of `1` for each cluster, which explains the lack of reusable prior candidates.
- The current codebase supports `reviewer_ids` in replay / prior builders, but the current multi-paper rerun surface does not expose an obvious reviewer parameter at the CLI layer.

### Secondary (MEDIUM confidence)
- `.planning/phases/11-iteration-prioritization-and-next-cycle-plan/11-RESEARCH.md`
- `.planning/phases/10-multi-paper-l3-and-l4-validation/10-RESEARCH.md`
- `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md`
- `docs/superpowers/specs/2026-04-01-logickg-route-packet-schema.md`
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: repeated bounded replay/package/prior/export reruns on existing backend logic
- Ecosystem: `research_logic`, CLI runners, bundle writers, iteration-priority outputs, and pytest validation
- Patterns: fixed baseline anchor, multi-iteration audit trail, summary-first reporting, explicit downstream blocker tracking
- Pitfalls: single-iteration collapse, hidden reviewer gaps, singleton prior clusters, drifting closeout baseline

**Confidence breakdown:**
- Need for a multi-iteration Phase 13 loop with one fixed cycle-1 baseline: HIGH
- Reuse of the Phase 10 runner and existing comparison/bundle writers: HIGH
- Importance of reviewer wiring plus support clustering as separate levers: HIGH
- Exact final iteration naming convention: MEDIUM
</metadata>
