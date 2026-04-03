# Phase 11: Iteration Prioritization And Next Cycle Plan - Context

**Gathered:** 2026-04-03 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 11 synthesizes the evidence already produced by sampled single-paper `L2` runs and bounded multi-paper `L3/L4` validation into one explicit next-cycle recommendation. It does not rerun those workflows, repair the chosen owners inside the same phase, widen the packet boundary, or change lower-level compiler contracts. Its job is to turn existing evidence into a prioritized optimization queue the team can act on.
</domain>

<decisions>
## Implementation Decisions

### Summary Artifact
- **D-01:** Phase 11 should be implemented as a synthesis step that reads existing Phase 8 and Phase 10 evidence and writes one new auditable summary bundle plus one committed markdown report, rather than rerunning either workflow.
- **D-02:** The canonical inputs should be machine-readable summary artifacts first, with markdown reports treated as operator-facing mirrors of those summaries instead of the source of truth.
- **D-03:** Phase 11 should accept explicit input paths for upstream evidence and record source provenance in its output, because `tmp/` bundles are machine-local and Phase 10 comparison artifacts may not persist at a default path on every workspace.

### Priority Taxonomy
- **D-04:** Phase 11 should preserve the two existing evidence taxonomies in one synthesis: Phase 8 `L2` owner buckets and Phase 10 multi-paper blocker queues grouped by stage/layer.
- **D-05:** The synthesis must keep fixed-regression `L2` failures, random-only discoveries, package-validation blockers, replay blockers, prior-induction blockers, and export posture distinct rather than flattening them into one generic severity score.
- **D-06:** The output must convert those taxonomies into an explicit next-cycle queue with a primary focus, secondary follow-ups, and a short explanation for why the other branches were not chosen first.

### Current Recommendation
- **D-07:** For the current baseline evidence chain, Phase 11 should rank packet-construction work first, keep `L4` aggregation as the next downstream follow-up, and carry forward the Phase 8 `L2` queue as supporting evidence rather than the immediate top target.
- **D-08:** That recommendation should be justified by where new blockers first appear in the current evidence: package validation and replay-stage blocker deltas, not just raw failure totals.
- **D-09:** The summary should keep the Phase 8 `relation_assembly` and `slot_recovery` queues visible so they remain part of the next-cycle evidence even when packet construction is chosen as the lead focus.

### Scope Boundary
- **D-10:** Phase 11 should stay a reporting and orchestration phase. It should not change `PaperLogicTrace`, `RoutePacket`, route-state package validation, or export truth-boundary contracts while making the recommendation.
- **D-11:** Missing or stale upstream evidence should surface as explicit preflight blockers in the Phase 11 output rather than being silently inferred from partial markdown prose.
- **D-12:** The new implementation should be a dedicated synthesis layer that reuses existing summary builders and bundle-writing patterns instead of scraping report text or embedding prioritization logic in an ad hoc shell-only flow.

### the agent's Discretion
- Exact scoring weights or tie-break rules between Phase 8 owner buckets and Phase 10 blocker queues, as long as stage/layer provenance remains visible.
- Exact output filenames and directory labels, as long as the repo's `tmp/...` plus `docs/replay/reports/...` audit pattern is preserved.
- Whether the operator-facing report is cycle-numbered, baseline-numbered, or packet-id-based, as long as source artifact refs stay explicit and replayable.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project And Milestone Framing
- `.planning/PROJECT.md` - Current `v1.1` goal, artifact philosophy, and the rule that evidence should drive the next optimization cycle
- `.planning/REQUIREMENTS.md` - `LOOP-01` and `LOOP-02`, including the requirement to connect sampled-paper and bounded multi-paper evidence in one summary
- `.planning/ROADMAP.md` - Phase 11 goal, success criteria, and handoff boundary from Phase 10 into prioritization
- `.planning/STATE.md` - Current milestone handoff, carry-forward decisions, and the explicit note that current evidence points to packet construction before `L2`

### Phase 8 Single-Paper Evidence
- `.planning/phases/08-sampled-single-paper-l2-regression/08-CONTEXT.md` - Locks the sampled-paper comparison surface, owner-bucket framing, and file-based artifact expectations
- `docs/replay/reports/phase8-sampled-single-paper-l2-regression-baseline.md` - Human-readable owner queue and cohort verdicts for the first measured Phase 8 cycle
- `tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/comparison_summary.json` - Canonical machine-readable Phase 8 summary, including verdict counts and owner buckets
- `tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/comparison_inspection.json` - Detailed per-paper Phase 8 evidence behind the owner queue

### Phase 9 And Phase 10 Multi-Paper Evidence
- `.planning/phases/09-bounded-packet-construction-from-corpus/09-CONTEXT.md` - Packet-boundary decisions and the known support / alternative / held_out gaps that Phase 10 carried forward
- `.planning/phases/10-multi-paper-l3-and-l4-validation/10-CONTEXT.md` - Locks the Phase 10 blocker framing and comparison boundary to the jamming baseline
- `.planning/phases/10-multi-paper-l3-and-l4-validation/10-VERIFICATION.md` - Exact runtime evidence and the strongest existing recommendation for the next cycle
- `docs/replay/reports/phase10-multi-paper-l3-l4-validation.md` - Operator-facing Phase 10 report rendered from comparison data
- `docs/replay/pilot_packets/phase9-route-packet.json` - Canonical bounded packet whose weaknesses feed the current recommendation
- `docs/replay/pilot_packets/phase9-assembly-manifest.json` - Canonical role grouping and packet-composition handoff used by Phase 10

### Reuse And Reporting Contracts
- `backend/app/research_logic/sampled_single_paper.py` - Owner-bucket computation and sampled-paper comparison logic to reuse rather than reimplement
- `backend/app/research_logic/replay_io.py` - Existing summary / inspection builders and auditable bundle-writing pattern
- `backend/app/research_logic/phase10_multi_paper_validation.py` - Structured `blocker_queue`, source-artifact refs, and report-rendering contract for multi-paper evidence
- `backend/scripts/run_sampled_single_paper_l2_regression.py` - Existing Phase 8 operator CLI pattern for summary-first reporting
- `backend/scripts/run_phase10_multi_paper_validation.py` - Existing Phase 10 operator CLI pattern for comparison-summary-backed reporting
- `backend/tests/test_replay_io.py` - Summary and bundle contract coverage for sampled-paper evidence outputs
- `backend/tests/test_phase10_multi_paper_validation.py` - Integration coverage for Phase 10 bundle generation and comparison-summary expectations

### Layer Contracts
- `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md` - Layer intent that Phase 11 must respect when choosing between `L2`, packet construction, and `L4`
- `docs/superpowers/specs/2026-04-01-logickg-route-packet-schema.md` - Packet contract that Phase 11 should treat as stable while summarizing evidence
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `backend/app/research_logic/sampled_single_paper.py`: exposes `compare_sampled_l2_iterations()` and owner-bucket summaries that already rank Phase 8 `L2` issues
- `backend/app/research_logic/replay_io.py`: already writes machine-readable summary and inspection files for sampled-paper evidence bundles
- `backend/app/research_logic/phase10_multi_paper_validation.py`: exposes `build_phase10_comparison_summary()` and `render_phase10_validation_report()` with a structured `blocker_queue`
- `backend/scripts/run_sampled_single_paper_l2_regression.py`: shows the established CLI pattern for reading structured evidence and emitting both `tmp/` artifacts and a committed markdown report
- `backend/scripts/run_phase10_multi_paper_validation.py`: shows the same summary-first pattern on the bounded multi-paper side

### Established Patterns
- The repo prefers auditable machine-readable bundles under `tmp/` plus a committed markdown report under `docs/replay/reports/`
- Structured summaries are the source of truth; markdown is rendered from those summaries rather than hand-authored independently
- Stage/layer boundaries stay explicit instead of collapsing package, replay, prior, export, and `L2` symptoms into one pass/fail result
- Phase-level workflows compose existing builders and bundle writers instead of redesigning core contracts each time

### Integration Points
- Phase 11 should read Phase 8 `comparison_summary.json` and `comparison_inspection.json` from `tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/`
- Phase 11 should read Phase 10 comparison data from a machine-readable bundle path when available, while also citing the committed report and verification note for operator review
- Phase 11 should write its own auditable bundle under `tmp/` and a committed markdown report under `docs/replay/reports/`
- New tests should live under `backend/tests/` and cover prioritization classification, missing-input preflight behavior, and report rendering
</code_context>

<specifics>
## Specific Ideas

- The current evidence should recommend packet construction first, specifically improving support density and alternative-route distinctness inside the bounded computational-mechanics slice.
- The existing Phase 8 `relation_assembly` and `slot_recovery` queues should remain visible as carry-forward work, even if they are not selected as the immediate next-cycle lead.
- The current workspace still contains `tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/`, but it does not retain the Phase 10 `baseline/comparison_summary.json` bundle mentioned in verification; only generated bridge artifacts and committed reports remain. Phase 11 should therefore support explicit Phase 10 input paths and record the provenance it used.
- The first implementation should prefer reading structured JSON artifacts over recomputing Phase 8 or Phase 10 logic during prioritization.
</specifics>

<deferred>
## Deferred Ideas

- Auto-fixing `L2` owners, rebuilding packets, or rerunning Phase 10 inside the same prioritization phase
- Broadening the bounded packet scope or selecting a brand-new packet topic during Phase 11
- Changing `PaperLogicTrace`, `RoutePacket`, route-state package, or export truth-boundary contracts while summarizing the current cycle
- Building a dedicated UI or dashboard for next-cycle prioritization in the same phase
</deferred>

---

*Phase: 11-iteration-prioritization-and-next-cycle-plan*
*Context gathered: 2026-04-03*
