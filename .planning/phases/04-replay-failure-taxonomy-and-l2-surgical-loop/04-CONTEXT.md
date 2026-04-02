# Phase 4: Replay Failure Taxonomy And L2 Surgical Loop - Context

**Gathered:** 2026-04-02 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase turns replay failure signals into a structured taxonomy that can point to the exact layer causing failure, then uses that taxonomy to drive narrow `L2` repairs on the current bounded replay baseline. It does not redesign the `L2` schema, replace the replay pipeline, or expand the scope into new packet-discovery or prior-induction capabilities.

</domain>

<decisions>
## Implementation Decisions

### Failure taxonomy boundary
- **D-01:** Build the failure taxonomy on top of the existing replay artifact boundary instead of introducing a parallel diagnostics pipeline. The taxonomy should extend or enrich `replay_summary` / `replay_inspection`, because these are already the stable handoff artifacts between `RoutePacket`, `RouteState`, `WhyNow`, `RouteComparison`, `DecisionPriorCard`, and `DecisionEpisode`.
- **D-02:** Keep failure attribution layer-aware. Every actionable failure should be attributable to one of: packet selection/coverage, `L1` environment coverage, `L2` extraction quality, or downstream `L3/L4` compilation behavior.
- **D-03:** Keep `stage` and `layer` separate in the failure contract. `stage` records where the weakness surfaced (`route_state`, `why_now_case`, `route_comparison`, `decision_prior_card`, `decision_episode`), while `layer` records the repair owner (`packet`, `L1`, `L2`, or downstream `L3/L4`).
- **D-04:** Use one unified `failure_records` surface for both hard failures and non-blocking but repair-worthy `L2` weaknesses. Non-blocking items must remain explicitly marked `blocking=false` so they act as repair leads rather than overclaiming sole-root-cause certainty.

### L2 repair scope
- **D-05:** Treat Phase 4 as a surgical `L2` improvement loop, not a schema rewrite. The first-pass repair targets are comparator density, expected-slot completeness, and relation stitching because Phase 3 has already removed the structural `support / alternative / held_out` blockers.
- **D-06:** Preserve the current `paper_grounded_l1_lite` and packaged multi-route replay baseline while improving `L2`. The point of the phase is to compare replay quality before and after bounded `L2` changes, not to move multiple layers at once.

### Evaluation loop
- **D-07:** Split replay outputs by purpose instead of duplicating the same payload everywhere. `replay_summary` should carry aggregate totals such as counts by `layer`, counts by `blocking` vs non-blocking, and top-level replay quality; `replay_inspection` should carry the full `failure_records` detail with evidence references.
- **D-08:** Judge every `L2` repair against a before/after replay comparison on the same bounded topic slice. The jamming packaged-replay artifacts validated in Phase 3 are the default regression baseline.
- **D-09:** Keep phase outputs auditable and file-based. New diagnostics should remain consumable from `backend/app/research_logic/` builders, CLI scripts, and JSON bundle outputs so later phases can reuse them directly.
- **D-10:** Treat the Phase 4 taxonomy as sufficient for the current replay-grounded repair loop, but not as the project's final long-horizon evaluation stack. Broader fidelity and cross-topic generalization evaluation remain later work.

### the agent's Discretion
- Exact failure-schema field names, severity ladder, and owner granularity within the locked `summary` / `inspection` split
- Whether the first `L2` surgical pass is best expressed as one plan or multiple narrowly scoped plans
- How much of the failure taxonomy should be surfaced as reusable typed models versus lightweight reporting structures in the first iteration

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project and milestone direction
- `.planning/PROJECT.md` - Project-level goal, layer boundaries, and anti-hindsight constraints
- `.planning/REQUIREMENTS.md` - Phase-to-requirement mapping, especially `L2-01`, `L2-02`, and `EVAL-02`
- `.planning/ROADMAP.md` - Phase 4 goal, success criteria, and plan slots
- `.planning/STATE.md` - Latest handoff after Phase 3 completion

### Phase 3 baseline
- `.planning/phases/03-grounded-route-replay-compilation/03-03-SUMMARY.md` - Why the last Phase 3 gap was a route-state quality heuristic rather than packaging structure
- `.planning/phases/03-grounded-route-replay-compilation/03-VALIDATION.md` - Validated Phase 3 quick/full commands and runtime artifact audit
- `docs/replay/reports/route-state-package-pilot-2026-04-02.md` - Real jamming packaged-replay baseline and the green runtime chain

### Product and research framing
- `docs/科学家思维AI项目技术文档.md` - High-level `L1 -> L2 -> L3 -> L4` scientific-reasoning architecture and intended end goal

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `backend/app/research_logic/replay_io.py` - already centralizes replay summaries, inspections, and bundle writing; best insertion point for structured failure reporting
- `backend/app/research_logic/historical_replay_compiler.py` - already emits replay-level `quality_flags` and knows where current failure signals originate
- `backend/app/research_logic/route_state_synthesizer.py` - already computes route-level quality, uncertainty, and evidence bundles; likely source for finer-grained `L2`-related failure hooks
- `backend/app/paper_logic_trace/models.py` - canonical `L2` slot surface for methods, metrics, comparators, conditions, limitations, resources, and move relations
- `backend/tests/test_route_state_synthesizer.py`, `backend/tests/test_route_state_package.py`, `backend/tests/test_replay_io.py`, `backend/tests/test_historical_replay_compiler.py` - current green regression boundary for replay logic

### Established Patterns
- `research_logic` uses typed Pydantic models plus JSON artifact boundaries rather than ad hoc dicts as the main public contract
- Replay quality is currently expressed as small, explicit string flags and structured inspection payloads, not opaque free-text evaluation
- Backend verification is phase-local and fixture-heavy, with focused pytest files under `backend/tests/`

### Integration Points
- New failure taxonomy should plug into the replay compilation/reporting chain rather than bypass it
- `L2` repairs should land in `backend/app/paper_logic_trace/` or adjacent extraction/derived-view logic, with replay-facing assertions added under `backend/tests/`
- Comparison against the current jamming baseline should continue to read existing runtime artifacts under `tmp/phase3_route_state_package/` rather than introducing a new evaluation slice too early
- `backend/app/research_logic/replay_io.py` now has a fixed reporting split to honor: aggregate taxonomy counts in `replay_summary`, full `failure_records` detail in `replay_inspection`

</code_context>

<specifics>
## Specific Ideas

- Phase 4 should explain *why* a replay failed in a way that is actionable for `L2`, not merely mark the output as yellow or red
- The first repaired signals should be ones that materially affect route comparison and downstream prior selection, rather than cosmetic completeness
- The jamming slice remains the reference packet until the failure taxonomy can prove it is pointing at the right missing `L2` evidence
- The taxonomy should behave like a conservative repair-queue generator: hard failures stay visible, and non-blocking `L2` weaknesses are still recorded when they sharpen repair priority
- `stage` answers where a weakness surfaced, `layer` answers who should repair it, and the two should never be collapsed into one overloaded field

</specifics>

<deferred>
## Deferred Ideas

- Promoting the runtime jamming subset packets into committed canonical assets - valuable, but orthogonal to first-pass failure taxonomy work
- Cross-topic replay validation - still important, but better after the taxonomy can explain failures on the current baseline
- Multi-route prior induction and `L4` expansion - belongs to Phase 5 onward, not this phase
- A fuller long-horizon evaluation stack for deep optimization across `L1/L2/L3/L4` - important, but outside the bounded Phase 4 replay-repair loop

</deferred>

---

*Phase: 04-replay-failure-taxonomy-and-l2-surgical-loop*
*Context gathered: 2026-04-02*
