# Phase 15: Cycle 3 Consolidation - Research

**Researched:** 2026-04-04
**Domain:** bounded cycle-3 quality consolidation on the existing multi-paper replay pipeline, with explicit comparison between default reruns and the fallback-merge prior-recovery lever
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Phase 15 must stay on the existing bounded computational-mechanics slice and keep using the current `package -> replay -> prior-review -> export` execution surface.
- Success must be judged directly against the completed Phase 14 `cycle2-best` review and artifact chain, not against abstract metrics or a widened packet.
- Export truth must remain route-backed and review-backed. Accepted reviewed priors / anti-patterns may enter final selected export fields only when they truly support the exported primary route.
- The explicit fallback-merge path from Phase 13 is allowed only as an auditable comparison lever. Phase 15 must not silently flip the default cluster strategy.
- `packet_construction` remains the lead recommendation. Route-comparison quality and prior-closure quality are the primary content surfaces to improve.
- Phase 15 closes on the first newly generated cycle that manual review judges genuinely useful as scientific-thinking training data. Consecutive-cycle stability proof belongs to Phase 16.

### The Agent's Discretion
- Exact Phase 15 iteration labels, output-root names, and report file names, as long as all candidate cycles remain auditable under a dedicated `tmp/phase15_*` root.
- Whether the winning cycle closes the prior gap through upstream packet improvements on the default path, the explicit fallback-merge lever, or honest structured exclusion rationale.
- Exact field names for any new machine-readable anti-pattern exclusion records, as long as the outcome is route-backed and structured.

### Deferred / Out Of Scope
- Widening packet membership or changing the topic slice
- Replacing the existing Phase 13 rerun entrypoint with a new execution framework
- Declaring stable consecutive-cycle quality before Phase 16
- New UI or operator-surface work for review management
</user_constraints>

<research_summary>
## Summary

Phase 15 should be planned as a three-wave consolidation phase whose goal is not "more export polish," but the first bounded cycle whose generated reasoning artifacts are actually judged high quality by direct reading.

The evidence now points to a narrow set of levers:

1. upstream packet and route-state quality still need to improve enough that the route comparison becomes genuinely grounded instead of staying `yellow`,
2. prior induction must be compared across two explicit modes on the same packet: the normal default path and the auditable `allow_scope_fallback_merge` recovery path,
3. export closure must stay honest: if a reviewed prior or anti-pattern supports the exported primary route, it should flow into `selected_prior_ids` or `selected_antipattern_ids`; if not, Phase 15 should preserve deterministic structured exclusion records instead of forcing the ids through.

The strongest prior evidence seen so far still comes from Phase 13's fallback-merge run, where one accepted prior survived review. Phase 14 improved the training-facing bundle shape, but the best cycle still failed manual review because the content remained weak: route comparison never named a winning route explicitly, why-now reasoning stayed underspecified, and prior / anti-pattern closure remained empty on the default path.

The safest Phase 15 planning move is therefore:

1. strengthen packet and route-state distinctness enough that route comparison and prior support have better raw material,
2. make the default-vs-fallback prior-recovery comparison explicit and machine-readable on every rerun,
3. tighten decision, export, and training-view assembly so route comparison, why-now, priors, and anti-patterns either close cleanly or explain why they do not,
4. run at least two auditable bounded candidate cycles against the completed Phase 14 best-cycle artifacts, select one best Phase 15 cycle, and record whether the accepted-cycle streak is now `1` or higher without overclaiming stability.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Asset | Status | Purpose | Why Standard Here |
|-------|--------|---------|-------------------|
| `backend/scripts/run_phase13_cycle_refinement.py` | Existing | Canonical bounded rerun entrypoint with reviewer ids, baseline overrides, and `--allow-scope-fallback-merge` | Keeps Phase 15 on the same audited execution surface while enabling explicit lever comparison |
| `backend/app/research_logic/phase10_multi_paper_validation.py` | Existing | Canonical package / replay / prior-review / export orchestration plus `comparison_summary.json` assembly | Best place to compare default and fallback runs and thread reviewed best-cycle evidence |
| `backend/app/research_logic/route_state_package.py` | Existing | Validates route-state package role balance, distinctness, and quality flags such as `yellow_route_state_present` | Primary upstream packet-quality surface still called out by prioritization |
| `backend/app/research_logic/bounded_packet_audit.py` | Existing | Audits bounded packet role grouping and alternative-route distinctness rationale | Supports packet-construction-first repairs without widening the packet |
| `backend/app/research_logic/prior_induction.py` | Existing | Controls cluster strategy, fallback merge behavior, and prior candidate generation | Key comparison point between Phase 14 default runs and Phase 13 fallback recovery |
| `backend/app/research_logic/route_comparison_builder.py` | Existing | Computes route comparison quality, preference grounding, and decisive evidence | Main blocker behind missing `recommended_route_state_id` and `route_advantage_summary` |
| `backend/app/research_logic/why_now_builder.py` | Existing | Produces why-now reasoning fields that still miss `because_now` and `why_not_before` in reviewed artifacts | Needed to make the training-facing reasoning more complete |
| `backend/app/research_logic/decision_episode_builder.py` | Existing | Produces final route choice, confidence, and selected prior / anti-pattern ids | Best place to prevent overconfident final decisions and route-mismatched selected ids |
| `backend/app/research_logic/decision_episode_export.py` | Existing | Preserves route-backed audited export truth and explicit accepted-but-unselected prior rationale | Must stay the audit boundary for TRAIN-04 |
| `backend/app/research_logic/replay_io.py` | Existing | Writes export summaries, training views, best-cycle selection artifacts, and bundle manifests | Natural home for any new structured anti-pattern exclusion surface |
| `backend/app/research_logic/iteration_prioritization.py` | Existing | Keeps recommendation evidence machine-readable and comparable across phases | Should continue to summarize reviewed evidence for the Phase 15 handoff |

### Supporting
| Asset | Status | Purpose | When to Use |
|-------|--------|---------|-------------|
| `tmp/phase14_cycle2_optimization/cycle2-best/comparison_summary.json` | Existing | Runtime comparison snapshot of the current best Phase 14 cycle | Use to identify unchanged blocker queues and baseline paths |
| `tmp/phase14_cycle2_optimization/cycle2-best/export_bundle/best_cycle_selection.json` | Existing | Canonical reviewed verdict for Phase 14 `cycle2-best` | Treat as the real baseline truth over any pre-review summary defaults |
| `tmp/phase14_cycle2_optimization/cycle2-best/export_bundle/outputs/training_view.json` | Existing | Current self-contained training-facing artifact | Use to judge which section-level fields still fail manual review |
| `tmp/phase14_cycle2_optimization/cycle2-best/prior_review_bundle/candidate_review_summary.json` | Existing | Default-path prior review summary with zero accepted priors | Baseline for the Phase 15 default path |
| `tmp/phase13_cycle2_refine/cycle2-final/prior_review_bundle/candidate_review_summary.json` | Existing | Fallback-merge prior review summary with one accepted prior | Evidence that the lever can recover reusable prior support |
| `docs/replay/reports/phase14-cycle2-optimization-best.md` | Existing | Human review of the current best-cycle content surface | Direct handoff of residual defects and section-level weakness |
| `docs/replay/reports/phase14-next-cycle-prioritization.md` | Existing | Evidence-backed next-cycle recommendation | Confirms that `packet_construction` is still the lead focus |
| `backend/tests/test_bounded_packet_audit.py` | Existing | Packet audit regression surface | Use when changing alternative distinctness or packet audit detail |
| `backend/tests/test_route_state_package.py` | Existing | Route-state package validation coverage | Use when targeting `yellow_route_state_present` and distinctness detail |
| `backend/tests/test_prior_induction.py` | Existing | Prior induction strategy and fallback merge coverage | Use to lock default vs fallback behavior |
| `backend/tests/test_route_comparison_builder.py` | Existing | Route comparison preference and quality flag coverage | Use to prove grounded route-advantage output |
| `backend/tests/test_why_now_builder.py` | Existing | Why-now reasoning coverage | Use to lock `because_now` and `why_not_before` behavior |
| `backend/tests/test_decision_episode_builder.py` | Existing | Selected prior / anti-pattern and confidence behavior | Use to keep export selection route-backed and confidence honest |
| `backend/tests/test_decision_episode_export.py` | Existing | Audited export truth and accepted-but-unselected prior coverage | Best anchor for explicit exclusion rationale |
| `backend/tests/test_replay_io.py` | Existing | Export summary, training view, and best-cycle selection output coverage | Best place to add new structured anti-pattern exclusion fields if needed |
| `backend/tests/test_phase10_multi_paper_validation.py` | Existing | End-to-end bounded rerun output coverage | Must prove Phase 15 artifacts stay deterministic |
| `backend/tests/test_phase13_cycle_refinement.py` | Existing | CLI-level rerun contract coverage | Use when the plan adds explicit default vs fallback comparison runs |
| `backend/tests/test_historical_replay_compiler.py` | Existing | Replay failure-stage and route-comparison assembly coverage | Guards upstream replay quality movement |
| `backend/tests/test_iteration_prioritization.py` | Existing | Recommendation taxonomy and evidence-link coverage | Use to keep the Phase 15 handoff comparable to earlier phases |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| explicit default-vs-fallback rerun comparison | silently enable fallback merge everywhere | Hides the real lever, weakens auditability, and makes later stability claims untrustworthy |
| upstream packet and comparison repair | export-only cleanup | Keeps the bundle prettier, but does not create better scientific-reasoning content |
| Phase 14 best-cycle reviewed verdict as the baseline truth | runtime `comparison_summary.json` review fields | Summary payloads still show pre-review defaults, so they are insufficient for closeout truth |
| route-backed selected ids or explicit structured exclusions | forcing reviewed ids into export fields | Violates the audited export contract and obscures whether the route actually supports the heuristic |
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Pattern 1: Auditable Lever-Labeled Reruns
**What:** Run the same bounded cycle under at least two explicit modes, default and fallback-merge, and preserve `allow_scope_fallback_merge`, `cluster_strategy`, and `fallback_reason` in machine-readable outputs.  
**When to use:** When Phase 15 needs to compare a recovered-prior path against the default path without hiding which lever produced the improvement.  
**Why recommended:** It keeps manual review honest and prevents a silent change in default semantics.

### Pattern 2: Packet-First Content Recovery
**What:** Fix packet and route-state distinctness issues before expecting downstream route comparison or prior selection to improve.  
**When to use:** When `packet_construction` remains the leading recommendation and package validation still carries `yellow_route_state_present`.  
**Why recommended:** Better packet material is the cleanest path to grounded route comparison and reusable priors.

### Pattern 3: Route Comparison As A Hard Quality Gate
**What:** Treat `recommended_route_state_id`, `route_advantage_summary`, and grounded decisive evidence as required for a genuinely good Phase 15 cycle.  
**When to use:** For any candidate cycle that manual review might accept as training-usable.  
**Why recommended:** The current best artifact still fails because the route win is implied, not actually explained.

### Pattern 4: Honest Closure Of Reviewed Priors And Anti-Patterns
**What:** Flow reviewed ids into final selected fields only when they support the exported primary route; otherwise preserve deterministic structured exclusion rationale.  
**When to use:** For both priors and anti-patterns in export summary and training-facing views.  
**Why recommended:** It satisfies TRAIN-04 without collapsing audit truth.

### Pattern 5: First Good Cycle, Not Premature Stability
**What:** Record the accepted-cycle streak and compare candidate cycles, but allow Phase 15 to close once the first new cycle is manually judged high quality.  
**When to use:** At Phase 15 closeout and handoff to Phase 16.  
**Why recommended:** It aligns with the roadmap boundary between first-good-cycle closure and later consecutive stability proof.

### Anti-Patterns To Avoid
- Silently turning fallback merge into the default prior strategy
- Treating missing route-comparison fields as a markdown-reporting issue instead of an upstream reasoning issue
- Preserving high final-decision confidence when prior support and route comparison still do not justify it
- Using pre-review summary defaults as the phase-closing truth instead of `best_cycle_selection.json`
- Claiming Phase 15 proves stability when it only establishes the first genuinely good cycle
</architecture_patterns>

<recommended_plan_slices>
## Recommended Plan Slices

### Slice 1: Packet quality and explicit prior-recovery lever groundwork
- Tighten bounded packet and route-state package quality so overlapping alternatives remain explicitly distinct and remaining yellow surfaces are inspectable.
- Preserve default-path and fallback-merge-path prior-induction outputs as clearly labeled machine-readable comparison surfaces.
- Add regression coverage that proves the default strategy remains default while fallback merge stays explicit and auditable.

### Slice 2: Route-comparison closure and route-backed export selection
- Improve route comparison so grounded runs emit `recommended_route_state_id` and `route_advantage_summary`.
- Fill why-now reasoning gaps that still leave `because_now` and `why_not_before` empty.
- Tighten final-decision confidence, selected-prior / anti-pattern flow, and explicit structured exclusion records so reviewed knowledge either closes honestly or explains why it does not.

### Slice 3: Bounded candidate reruns, manual best-cycle review, and streak recording
- Run at least one default-path and one fallback-merge-path Phase 15 candidate against the completed Phase 14 `cycle2-best` replay/export baseline.
- Select a canonical `cycle3-best` root, write a section-level manual review, and record whether the accepted-cycle streak is `1` or higher.
- Render the next-cycle prioritization handoff from reviewed evidence without overclaiming stability.

The natural planning split is therefore 3 plans across 3 waves:
- Wave 1: packet and prior-recovery groundwork
- Wave 2: route comparison plus export closure
- Wave 3: bounded reruns, best-cycle review, and handoff
</recommended_plan_slices>

<validation_architecture>
## Validation Architecture

Phase 15 should validate at five levels:

1. **packet and route-state validation**
   - prove alternative-route distinctness remains explicit under overlapping scope,
   - prove route-state package summaries expose enough detail to localize `yellow_route_state_present`,
   - prove packet-validation changes do not reintroduce `support_cluster_too_small` or `alternative_scope_not_distinct`.

2. **prior-recovery lever validation**
   - prove default runs still serialize `cluster_strategy = default`,
   - prove fallback runs serialize `cluster_strategy = fallback_scope_merge` and `fallback_reason = singleton_support_clusters`,
   - prove both the rerun CLI and Phase 10 comparison summaries preserve which lever produced the candidate-cycle result.

3. **route-comparison and export-closure validation**
   - prove grounded comparison paths emit `recommended_route_state_id` and `route_advantage_summary`,
   - prove why-now reasoning includes `because_now` and `why_not_before` on positive paths,
   - prove reviewed priors and anti-patterns either enter final selected fields or generate explicit machine-readable exclusion records.

4. **end-to-end bounded rerun validation**
   - prove Phase 15 candidate cycles write package, replay, prior-review, export, training-view, best-cycle-selection, and comparison-summary artifacts under a dedicated Phase 15 root,
   - prove candidate-cycle summaries preserve baseline override refs pointing at the Phase 14 `cycle2-best` bundles,
   - prove the chosen `cycle3-best` root records the accepted-cycle streak instead of claiming full stability.

5. **manual best-cycle review**
   - a reviewer must read the chosen Phase 15 artifacts directly and decide whether they are genuinely useful for scientific-thinking training,
   - review must explicitly judge evidence pack, route synthesis, why-now, route comparison, priors / anti-patterns, minimal attack path, final decision, and review labels,
   - the final verdict must compare the winning Phase 15 cycle against the completed Phase 14 best-cycle review, not just against baseline metrics.

**Expected commands:**
- Quick path: `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_bounded_packet_audit.py tests\test_route_state_package.py tests\test_prior_induction.py tests\test_route_comparison_builder.py tests\test_why_now_builder.py tests\test_decision_episode_builder.py tests\test_decision_episode_export.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_historical_replay_compiler.py tests\test_iteration_prioritization.py -q`
- Full path: `cd backend; .\.venv\Scripts\python.exe -m pytest tests\test_bounded_packet_audit.py tests\test_route_state_package.py tests\test_prior_induction.py tests\test_route_comparison_builder.py tests\test_why_now_builder.py tests\test_decision_episode_builder.py tests\test_decision_episode_export.py tests\test_decision_episode_export_cli.py tests\test_replay_io.py tests\test_phase10_multi_paper_validation.py tests\test_phase10_multi_paper_validation_cli.py tests\test_phase13_cycle_refinement.py tests\test_historical_replay_compiler.py tests\test_iteration_prioritization.py tests\test_iteration_prioritization_cli.py -q`

Phase 15 does not need to prove milestone-level stability yet. It needs to prove that at least one new bounded cycle clears the direct manual-review bar while preserving honest route-backed prior / anti-pattern closure.
</validation_architecture>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Phase 15 rerun execution | a new one-off runner | `backend/scripts/run_phase13_cycle_refinement.py` with explicit Phase 15 output roots and baseline overrides | Keeps the cycle auditable and comparable |
| prior recovery | a silent global config flip | explicit `--allow-scope-fallback-merge` comparison runs | Makes the lever visible in artifacts and review notes |
| export closure | optimistic selected-id promotion | route-backed selection plus explicit structured exclusions | Preserves audit truth |
| route-comparison handoff | markdown-only explanation | `route_comparison_builder.py` fields plus machine-readable bundle outputs | Keeps reasoning testable and reusable |
| best-cycle truth | report prose alone | `best_cycle_selection.json` plus report plus verification note | Prevents confusion between pre-review defaults and reviewed verdicts |
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: The fallback lever "wins" without being named
**What goes wrong:** A Phase 15 cycle looks better, but the artifact chain does not preserve whether fallback merge was used.  
**Why it happens:** The runner already supports the flag, so it is easy to treat it as a hidden tweak.  
**How to avoid:** Require lever-specific fields in run summaries, comparison summaries, and review notes.  
**Warning signs:** The best-cycle report cannot answer whether `cluster_strategy` was `default` or `fallback_scope_merge`.

### Pitfall 2: Packet blockers stay abstract
**What goes wrong:** Phase 15 keeps citing `yellow_route_state_present`, but no task localizes which route-state artifacts remain yellow or why.  
**Why it happens:** The top-level flag is easy to discuss without tracing the underlying route-state package surface.  
**How to avoid:** Surface exact entry ids and distinctness rationale through packet and route-state validation artifacts.  
**Warning signs:** Plans mention packet quality but do not name `route_state_package.py` or packet-audit outputs.

### Pitfall 3: Route comparison improves cosmetically, not semantically
**What goes wrong:** The report writes a nicer explanation, but machine-readable comparison fields remain empty or weakly grounded.  
**Why it happens:** The current defect shows up as missing prose, so the fix can drift into rendering work.  
**How to avoid:** Make `recommended_route_state_id`, `route_advantage_summary`, and decisive evidence refs part of the tested output contract.  
**Warning signs:** The task only edits markdown or training-view assembly files.

### Pitfall 4: Phase 15 overclaims stability
**What goes wrong:** One good cycle is treated as proof that quality is now stable.  
**Why it happens:** The milestone is already close to the finish line, so the temptation is to collapse Phase 15 and Phase 16.  
**How to avoid:** Record the accepted-cycle streak explicitly and defer reproducibility claims to Phase 16.  
**Warning signs:** The closeout language says "stable" without naming a streak length or repeated good runs.
</common_pitfalls>

<open_questions>
## Open Questions

1. **If the first genuinely good Phase 15 cycle is the fallback-merge candidate, should `cycle3-best` canonically use that lever or should the phase keep iterating until the default path also clears the bar?**  
   Recommendation: accept the fallback-merge cycle if it is genuinely good, but record the streak honestly as `1` and push default-path reproducibility to Phase 16.

2. **Should anti-pattern exclusions mirror the prior-exclusion schema exactly?**  
   Recommendation: yes, if implementation cost is reasonable. A parallel `accepted_but_unselected_antipatterns` shape would satisfy TRAIN-04 more cleanly than a prose-only note.

3. **Should Phase 15 reuse Phase 12 as the machine baseline inside comparison summaries or override the baseline bundles to Phase 14 `cycle2-best`?**  
   Recommendation: override to the completed Phase 14 best-cycle replay/export bundles for Phase 15 candidate runs, because that is the current official baseline for this phase.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- `.planning/PROJECT.md`
- `.planning/REQUIREMENTS.md`
- `.planning/ROADMAP.md`
- `.planning/STATE.md`
- `.planning/phases/15-cycle-3-consolidation/15-CONTEXT.md`
- `.planning/phases/14-cycle-2-optimization-and-review/14-CONTEXT.md`
- `.planning/phases/14-cycle-2-optimization-and-review/14-VERIFICATION.md`
- `.planning/phases/13-baseline-data-cycle-2-refine/13-CONTEXT.md`
- `docs/replay/reports/phase14-cycle2-optimization-best.md`
- `docs/replay/reports/phase14-next-cycle-prioritization.md`
- `tmp/phase14_cycle2_optimization/cycle2-best/comparison_summary.json`
- `tmp/phase14_cycle2_optimization/cycle2-best/export_bundle/best_cycle_selection.json`
- `tmp/phase14_cycle2_optimization/cycle2-best/export_bundle/outputs/training_view.json`
- `tmp/phase14_cycle2_optimization/cycle2-best/prior_review_bundle/candidate_review_summary.json`
- `tmp/phase13_cycle2_refine/cycle2-final/prior_review_bundle/candidate_review_summary.json`
- `backend/scripts/run_phase13_cycle_refinement.py`
- `backend/app/research_logic/bounded_packet_audit.py`
- `backend/app/research_logic/route_state_package.py`
- `backend/app/research_logic/prior_induction.py`
- `backend/app/research_logic/route_comparison_builder.py`
- `backend/app/research_logic/why_now_builder.py`
- `backend/app/research_logic/decision_episode_builder.py`
- `backend/app/research_logic/decision_episode_export.py`
- `backend/app/research_logic/replay_io.py`
- `backend/app/research_logic/phase10_multi_paper_validation.py`
- `backend/app/research_logic/historical_replay_compiler.py`
- `backend/app/research_logic/iteration_prioritization.py`
- `backend/tests/test_bounded_packet_audit.py`
- `backend/tests/test_route_state_package.py`
- `backend/tests/test_prior_induction.py`
- `backend/tests/test_route_comparison_builder.py`
- `backend/tests/test_why_now_builder.py`
- `backend/tests/test_decision_episode_builder.py`
- `backend/tests/test_decision_episode_export.py`
- `backend/tests/test_replay_io.py`
- `backend/tests/test_phase10_multi_paper_validation.py`
- `backend/tests/test_phase13_cycle_refinement.py`
- `backend/tests/test_historical_replay_compiler.py`
- `backend/tests/test_iteration_prioritization.py`

### Secondary (MEDIUM confidence)
- `.planning/phases/14-cycle-2-optimization-and-review/14-RESEARCH.md`
- `.planning/phases/06-decision-episode-audit-export/06-RESEARCH.md`
- `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md`
- `docs/superpowers/specs/2026-04-01-logickg-decision-prior-and-episode-schema.md`
- `docs/superpowers/specs/2026-04-01-logickg-why-now-and-route-comparison-schema.md`
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: bounded multi-paper reruns, route-state package quality, auditable prior recovery, route comparison grounding, and route-backed export closure
- Ecosystem: `research_logic`, rerun CLI scripts, export bundle writers, comparison summary generation, and pytest validation
- Patterns: explicit lever comparison, packet-first reasoning repair, route-backed selected-id closure, reviewed best-cycle handoff
- Pitfalls: hidden fallback defaults, packet blocker abstraction, cosmetic route-comparison fixes, premature stability claims

**Confidence breakdown:**
- Reuse of the existing bounded runner and Phase 10 orchestration surface: HIGH
- Need to compare default and fallback prior-recovery paths explicitly: HIGH
- Need to improve route comparison and why-now content rather than only bundle shape: HIGH
- Exact schema name for anti-pattern exclusion records: MEDIUM
</metadata>

---

*Phase: 15-cycle-3-consolidation*
*Research completed: 2026-04-04*
