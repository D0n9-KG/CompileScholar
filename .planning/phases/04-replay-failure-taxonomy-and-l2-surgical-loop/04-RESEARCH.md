# Phase 4: Replay Failure Taxonomy And L2 Surgical Loop - Research

**Researched:** 2026-04-03
**Domain:** replay-grounded failure attribution and bounded `L2` repair for scientific reasoning compilation
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Extend the existing replay artifact boundary instead of building a second diagnostics pipeline.
- Keep every actionable failure layer-aware: `packet`, `L1`, `L2`, or downstream `L3/L4`.
- Treat Phase 4 as a surgical `L2` loop, not an `L2` schema redesign.
- Preserve the current `paper_grounded_l1_lite` and packaged multi-route replay baseline while improving `L2`.
- Judge every repair against the same bounded jamming packaged-replay slice validated in Phase 3.
- Keep outputs auditable and file-based so later phases can reuse them directly.

### the agent's Discretion
- Exact failure-record field names and whether the first version is fully typed or lightweight JSON.
- How much of the taxonomy should live in `replay_summary` vs `replay_inspection`.
- Whether the first `L2` repair pass is split by signal family or implemented as one narrow batch.

### Deferred Ideas (OUT OF SCOPE)
- Redesigning the `L2` schema or broadening `ResearchMove` fields.
- Building a new replay/diagnostics subsystem outside `research_logic`.
- Cross-topic validation before the same-slice loop is working.
- Jumping ahead to `L4` hypothesis generation.
</user_constraints>

<research_summary>
## Summary

Phase 4 should convert replay quality from "green/yellow/red plus a few flags" into a repair-ready contract that answers three questions at once:

1. what failed or remained weak,
2. which layer owns it,
3. what the next bounded repair should target.

The important finding from the Phase 3 baseline is that the project is no longer blocked by missing multi-route structure. The green jamming package proves the replay chain can run end to end. That changes the Phase 4 job: we now need a taxonomy that can expose *latent* `L2` weaknesses even when the top-level replay still compiles. In practice that means combining two signal families:

- explicit replay-stage failures already visible in `route_state`, `why_now`, `route_comparison`, `decision_prior_card`, and `decision_episode`;
- paper-level and route-level `L2` quality signals such as comparator sparsity, expected-slot omissions, and weak relation stitching that may not yet flip the whole replay bundle to red.

The safest design is to add a normalized `failure_records` surface to replay outputs and generate it from existing contracts rather than inventing a separate review format. Each record should carry at least:

- `failure_id`
- `layer`
- `stage`
- `severity`
- `blocking`
- `failure_code`
- `owner`
- `repair_target`
- `evidence_refs`
- `source_flags`

For the first pass, the taxonomy should map existing signals into a small fixed code set instead of trying to infer arbitrary diagnoses. A concrete starter set is:

- `packet_trace_coverage_missing`
- `packet_role_coverage_missing`
- `l1_snapshot_missing`
- `l1_support_missing`
- `l2_comparator_sparse`
- `l2_expected_slot_missing`
- `l2_relation_stitch_missing`
- `l2_challenging_evidence_missing`
- `route_comparison_missing`
- `prior_support_cluster_too_small`
- `held_out_routes_missing`
- `reviewer_missing`

That keeps the system auditable while still giving Phase 4 enough structure to drive the first repair queue. The most valuable `L2` targets remain the same ones already surfaced in context gathering:

- comparator density
- expected-slot completeness
- relation stitching

These three are high leverage because they directly affect whether a single paper contributes usable evidence to multi-paper `RouteState`, `WhyNow`, and `RouteComparison` compilation. They also answer the user's core concern: if `L2` cannot model a paper with enough structure and provenance, `L3/L4` quality will plateau no matter how good the downstream aggregation logic is.

**Primary recommendation:** add replay-native failure records first, then repair only the `L2` surfaces those records keep pointing to, and measure all changes with before/after deltas on the same jamming packaged-replay baseline.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Asset | Status | Purpose | Why Standard Here |
|-------|--------|---------|-------------------|
| `backend/app/research_logic/historical_replay_compiler.py` | Existing | Central replay-stage quality decisions | Best place to classify stage and downstream failure ownership |
| `backend/app/research_logic/replay_io.py` | Existing | Stable replay artifact writer | Best place to expose normalized failure records in JSON outputs |
| `backend/app/research_logic/route_state_synthesizer.py` | Existing | Primary `RouteState` quality and evidence aggregation | Main route-level source of `L2`-visible degradation |
| `backend/app/paper_logic_trace/derived_views.py` | Existing | Single-paper derived `L2` surfaces | Best place for comparator, slot-completeness, and relation-stitching fixes |

### Supporting
| Asset | Status | Purpose | When to Use |
|-------|--------|---------|-------------|
| `backend/tests/test_paper_logic_trace_derived_views.py` | Existing | Covers single-paper extraction-to-derived-view behavior | Use for comparator/slot/relation fixes |
| `backend/tests/test_route_state_synthesizer.py` | Existing | Covers `RouteState` quality and evidence bundle behavior | Use for route-level regression checks |
| `backend/tests/test_historical_replay_compiler.py` | Existing | Covers downstream replay decisions | Use when failure ownership changes affect prior/episode gating |
| `backend/tests/test_replay_io.py` | Existing | Covers replay artifact output shape | Use when failure records are added to JSON artifacts |
| `docs/replay/reports/route-state-package-pilot-2026-04-02.md` | Existing | Real bounded replay baseline | Use as the canonical "before" reference |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| replay-native failure records | a separate diagnostics script or standalone spreadsheet | Faster to prototype, but breaks reuse and drifts away from the real artifact boundary |
| fixed failure codes | free-form LLM-written diagnoses | More flexible wording, but much worse for regression and auditability |
| bounded same-slice delta loop | cross-topic reruns immediately | Broader coverage, but makes it unclear whether failures come from the fix or the dataset change |
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Pattern 1: Failure Records At The Same Boundary As Replay Outputs
**What:** Put normalized failure records directly into `replay_summary` and/or `replay_inspection`, derived from existing stage outputs and quality flags.
**When to use:** Any diagnosis meant to drive code changes or later evaluation.
**Why recommended:** The replay artifact is already the stable contract between `RoutePacket`, `RouteState`, `WhyNow`, `RouteComparison`, `DecisionPriorCard`, and `DecisionEpisode`.

### Pattern 2: Layer Ownership Before Repair
**What:** Every failure record must name the owning layer before any repair task is created.
**When to use:** Whenever the same symptom could be caused upstream or downstream.
**Why recommended:** This prevents Phase 4 from treating all replay weakness as an `L2` problem.

### Pattern 3: Latent L2 Risk, Not Just Hard Failure
**What:** Track both blocking failures and degrading-but-nonblocking `L2` gaps.
**When to use:** When the packaged replay remains green but paper-level evidence is still thin.
**Why recommended:** The current baseline already proves that some `L2` weaknesses can hide under an end-to-end green replay.

### Anti-Patterns To Avoid
- Adding a parallel failure-reporting path that bypasses `replay_io.py`.
- Treating every yellow signal as an `L2` bug before checking packet and `L1`.
- Redesigning `PaperLogicTrace` or `ResearchMove` just to patch a bounded derived-view gap.
- Comparing repairs across different packets or different corpus slices.
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Failure ownership | a new ad hoc review notebook | normalized `failure_records` in replay outputs | Keeps diagnosis aligned with the actual runtime contract |
| `L2` repair scope | a new `L2 v2` schema | narrow fixes in `derived_views.py` and `route_state_synthesizer.py` | The project already decided against schema redesign here |
| Delta comparison | manual JSON eyeballing only | a committed Markdown delta report plus runtime bundle paths | Human-readable and auditable without hiding the raw artifacts |

**Key insight:** Phase 4 should make replay outputs *repairable*, not merely more descriptive.
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: A green replay hides weak single-paper evidence
**What goes wrong:** The end-to-end bundle is green, so the team assumes `L2` is good enough.
**Why it happens:** Current replay quality is mostly stage-level and can miss degrading `L2` sparsity.
**How to avoid:** Emit degrading `L2` records even when `ready_for_pilot` remains true.
**Warning signs:** Comparator, protocol, or limitation signals are thin in derived views, but top-level replay still looks healthy.

### Pitfall 2: Repair tasks are created from symptoms, not ownership
**What goes wrong:** The team patches `L2` when the real issue is missing packet coverage or held-out inputs.
**Why it happens:** Current flags describe the symptom but not the owning layer.
**How to avoid:** Require `layer` and `repair_target` on every failure record.
**Warning signs:** The same replay symptom keeps returning after unrelated `L2` edits.

### Pitfall 3: Delta comparison changes both code and baseline
**What goes wrong:** A repair seems helpful, but the packet inputs or review setup also changed.
**Why it happens:** Runtime reruns are not anchored to one bounded slice.
**How to avoid:** Reuse the Phase 3 jamming package and compare against the exact earlier artifact paths.
**Warning signs:** The "after" report references a different packet or a different role composition.
</common_pitfalls>

<open_questions>
## Open Questions

1. **Should the first taxonomy be fully typed or lightweight JSON-first?**
   - What we know: the repo prefers typed models for public contracts.
   - What's unclear: whether Phase 4 needs a new model file immediately or can start inside `replay_io.py`.
   - Recommendation: start with a lightweight but fixed JSON record shape; promote to a dedicated model only if multiple files need to share it.

2. **How should latent `L2` gaps be surfaced when replay stays green?**
   - What we know: the baseline may still compile successfully after Phase 3.
   - What's unclear: whether degrading `L2` records should be top-level `failure_records` or a sibling `repair_queue`.
   - Recommendation: keep one `failure_records` list and differentiate with `blocking: false` plus `severity: minor|major`.

3. **Should the jamming subset packets be committed before or after Phase 4?**
   - What we know: they are still runtime-only assets under `tmp/`.
   - What's unclear: whether formal asset promotion is needed before the first delta rerun.
   - Recommendation: do not block Phase 4 on this; reuse the current runtime assets first.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- `.planning/phases/04-replay-failure-taxonomy-and-l2-surgical-loop/04-CONTEXT.md`
- `.planning/STATE.md`
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/phases/03-grounded-route-replay-compilation/03-VALIDATION.md`
- `.planning/phases/03-grounded-route-replay-compilation/03-03-SUMMARY.md`
- `docs/replay/reports/route-state-package-pilot-2026-04-02.md`
- `backend/app/research_logic/historical_replay_compiler.py`
- `backend/app/research_logic/replay_io.py`
- `backend/app/research_logic/route_state_synthesizer.py`
- `backend/app/paper_logic_trace/derived_views.py`

### Secondary (MEDIUM confidence)
- `backend/tests/test_paper_logic_trace_derived_views.py`
- `backend/tests/test_route_state_synthesizer.py`
- `backend/tests/test_historical_replay_compiler.py`
- `backend/tests/test_replay_io.py`
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: replay-native failure taxonomy and bounded `L2` repair
- Ecosystem: `research_logic`, `paper_logic_trace`, replay reports, pytest regression tests
- Patterns: normalized failure records, layer ownership, same-slice delta loop
- Pitfalls: latent green-path weakness, wrong-layer repair, moving baseline

**Confidence breakdown:**
- Standard stack: HIGH
- Architecture: HIGH
- Pitfalls: HIGH
- Initial failure-code set: MEDIUM

**Research date:** 2026-04-03
**Valid until:** 2026-05-03
</metadata>

---

*Phase: 04-replay-failure-taxonomy-and-l2-surgical-loop*
*Research completed: 2026-04-03*
*Ready for planning: yes*
