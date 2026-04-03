# Phase 12: Baseline Data Cycle 1 (Direct Fix) - Context

**Gathered:** 2026-04-04 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 12 starts from the existing Phase 10 and Phase 11 baseline artifacts already present in this workspace, applies one bounded packet-first direct-fix cycle on that same computational-mechanics slice, reruns real package -> replay -> prior/review -> export outputs, and records manual quality review on the regenerated reasoning artifacts. It does not reopen packet/topic scope, introduce a new runtime foundation, or turn this cycle into a broad `L2` or `L4` redesign.

</domain>

<decisions>
## Implementation Decisions

### Starting Evidence And Input Boundary
- **D-01:** Phase 12 should treat `tmp/phase11_iteration_prioritization/baseline/outputs/prioritization_summary.json` and `tmp/phase11_iteration_prioritization/baseline/outputs/prioritization_inspection.json` as the canonical starting defect queue for this cycle, rather than recomputing a new lead focus before any fixes are made.
- **D-02:** The bounded packet boundary stays frozen on the committed Phase 9 computational-mechanics assets (`phase9-route-packet.json` plus `phase9-assembly-manifest.json`) and the same `phase9_comp_mech_2021_packet_01` topic slice used by Phase 10 and Phase 11.

### Fix Scope For Cycle 1
- **D-03:** The first direct-fix cycle should stay packet-first. Planning should target the Phase 10 blocker codes that Phase 11 already ranked highest: `support_cluster_too_small`, `alternative_scope_not_distinct`, and `yellow_route_state_present`.
- **D-04:** `reviewer_missing`, `decision_prior_card` regressions, and the carried-forward Phase 8 `relation_assembly` / `slot_recovery` owner queues should remain visible as downstream evidence, but they should not replace packet construction as the lead intervention unless the packet-first pass cannot move the Phase 10 comparison surface at all.

### Rerun And Artifact Contract
- **D-05:** The rerun should reuse the real Phase 10 execution surface (`backend/scripts/run_phase10_multi_paper_validation.py`) or a thin orchestration wrapper around it, so the cycle regenerates the same package, replay, prior-review, export, and comparison-summary artifacts instead of inventing a separate validation path.
- **D-06:** Phase 12 should preserve the repo's summary-first audit pattern: machine-readable artifacts under a fresh cycle-scoped `tmp/` directory are the source of truth, and committed markdown reports under `docs/replay/reports/` are rendered from those artifacts.

### Review And Defect-Linkage Contract
- **D-07:** Phase 12 must add an explicit same-cycle review artifact that records observed output defects on real reasoning artifacts and links each implemented fix to the resulting rerun evidence. The report should follow the repo's existing before/after delta style rather than a freeform narrative-only note.
- **D-08:** Manual review for this cycle must judge overall quality movement across package validation, replay behavior, prior-review output, and export posture, not only whether one paper or one isolated defect improved.

### Provenance And Comparison Guardrails
- **D-09:** Because the original Phase 10 `comparison_summary.json` is not present in the current workspace, Phase 12 should keep fallback provenance to the committed Phase 10 verification/report and the Phase 11 prioritization bundle while generating new cycle-1 machine-readable outputs that become the forward baseline for later cycles.
- **D-10:** Phase 12 should keep the milestone's anti-overfitting guardrail explicit: the closeout review must explain whether the packet-first fixes improved the overall bounded multi-paper slice, not merely one-paper behavior or one handcrafted artifact.

### the agent's Discretion
- Whether the cycle closes with only the rerun plus manual review bundle, or also re-executes the Phase 11 prioritization runner against the new cycle artifacts, as long as the main deliverables remain direct-fix evidence on the same bounded slice
- Exact cycle-scoped output and report names, as long as they clearly distinguish the new run from the inherited Phase 10 / Phase 11 baseline
- Exact structure of the review-note payload, as long as each fix is tied to concrete defect codes, rerun evidence, and an overall quality verdict

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project And Milestone Framing
- `.planning/PROJECT.md` - Active `v1.2` goal, packet-first quality-recovery philosophy, and manual-review-first milestone framing
- `.planning/REQUIREMENTS.md` - `LOOPR-01` and `LOOPR-02`, including the direct-fix cycle and auditable cycle-evidence requirements
- `.planning/ROADMAP.md` - Phase 12 goal, success criteria, and the rule to start from Phase 10 / 11 artifacts instead of a new foundation
- `.planning/STATE.md` - Current milestone handoff and carried-forward note that packet construction is the current lead

### Canonical Baseline Inputs
- `.planning/phases/10-multi-paper-l3-and-l4-validation/10-CONTEXT.md` - Locks the bounded Phase 10 execution boundary and audit contract that Phase 12 is reusing
- `.planning/phases/10-multi-paper-l3-and-l4-validation/10-VERIFICATION.md` - Exact runtime command, blocker codes, and the original bounded multi-paper verdict that Phase 12 must improve against
- `.planning/phases/11-iteration-prioritization-and-next-cycle-plan/11-CONTEXT.md` - Locks the recommendation logic boundary and keeps packet construction first
- `.planning/phases/11-iteration-prioritization-and-next-cycle-plan/11-VERIFICATION.md` - Exact Phase 11 command and proof that `packet_construction` ranked first on the inherited evidence chain
- `docs/replay/reports/phase10-multi-paper-l3-l4-validation.md` - Operator-facing rendering of the current Phase 10 blocker surface
- `docs/replay/reports/phase11-iteration-prioritization-next-cycle.md` - Operator-facing rendering of the current next-cycle queue
- `tmp/phase11_iteration_prioritization/baseline/outputs/prioritization_summary.json` - Canonical starting recommendation queue and supporting evidence for Phase 12
- `tmp/phase11_iteration_prioritization/baseline/outputs/prioritization_inspection.json` - Detailed supporting evidence and ranking signals behind the Phase 11 recommendation

### Bounded Packet And Replay Contracts
- `docs/replay/pilot_packets/phase9-route-packet.json` - Frozen bounded packet that Phase 12 continues to use
- `docs/replay/pilot_packets/phase9-assembly-manifest.json` - Frozen `support / alternative / held_out` role mapping for the same bounded slice
- `docs/replay/pilot_packets/phase9-selection-notes.md` - Topic boundary rationale and packet-composition caveats that still apply to this cycle
- `tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/comparison_summary.json` - Supporting `L2` owner-bucket evidence Phase 11 carried forward as secondary pressure
- `tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/comparison_inspection.json` - Detailed sampled-paper evidence behind `relation_assembly` and `slot_recovery`

### Runtime Runners And Contracts
- `backend/app/research_logic/phase10_multi_paper_validation.py` - Canonical multi-paper rerun pipeline Phase 12 should reuse to regenerate package/replay/prior/export/comparison artifacts
- `backend/scripts/run_phase10_multi_paper_validation.py` - Operator-facing CLI for the Phase 10 rerun surface
- `backend/tests/test_phase10_multi_paper_validation.py` - Existing contract coverage for the rerun surface and comparison-summary outputs
- `backend/app/research_logic/iteration_prioritization.py` - Current recommendation-ranking and fallback-provenance logic that feeds the inherited Phase 12 fix target
- `backend/scripts/run_phase11_iteration_prioritization.py` - Optional closeout runner if the cycle wants a refreshed priority queue after rerun
- `backend/tests/test_iteration_prioritization.py` - Contract coverage for recommendation ranking and fallback evidence handling

### Review And Audit Patterns
- `docs/replay/reports/phase4-l2-surgical-delta.md` - Existing same-slice before/after delta reporting pattern to reuse for Phase 12 defect linkage
- `.planning/phases/04-replay-failure-taxonomy-and-l2-surgical-loop/04-VERIFICATION.md` - Verifies the repo's precedent for honest rerun reporting even when quality movement is flat
- `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md` - Layer intent and boundaries that Phase 12 must preserve while optimizing
- `docs/superpowers/specs/2026-04-01-logickg-route-packet-schema.md` - Route-packet contract and cutoff-discipline rules that remain stable during the direct-fix cycle

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `backend/app/research_logic/phase10_multi_paper_validation.py`: already regenerates route-state package, replay, prior-review, export, and comparison-summary artifacts for the bounded multi-paper slice
- `backend/scripts/run_phase10_multi_paper_validation.py`: existing CLI entrypoint for the real rerun Phase 12 should reuse instead of inventing a new execution path
- `backend/app/research_logic/iteration_prioritization.py`: already converts Phase 8 and Phase 10 evidence into a ranked next-cycle queue and preserves fallback provenance when Phase 10 JSON is missing
- `backend/scripts/run_phase11_iteration_prioritization.py`: existing summarizer that can optionally be rerun after the new cycle to refresh the recommendation queue
- `backend/app/research_logic/replay_io.py`: existing summary / inspection bundle-writing contract that keeps machine-readable artifacts and operator-facing reports aligned
- `tmp/phase11_iteration_prioritization/baseline/outputs/prioritization_summary.json`: concrete inherited defect queue Phase 12 can read directly

### Established Patterns
- The repo uses auditable machine-readable bundle outputs under `tmp/` plus committed operator-facing markdown reports under `docs/replay/reports/`
- Summary and inspection payloads are the source of truth; markdown reports are rendered from those machine-readable artifacts rather than authored independently
- The project prefers same-slice before/after delta reporting when evaluating targeted fixes, even when the honest result is "flat" rather than "improved"
- Fallback provenance is acceptable when a prior machine-readable artifact is missing, but a fresh cycle should regenerate the missing structured outputs whenever possible

### Integration Points
- Phase 12 should read Phase 11 baseline prioritization outputs first to lock the lead fix target before code changes
- The direct-fix cycle should rerun the bounded computational-mechanics packet through the existing Phase 10 validation flow, producing a new cycle-scoped output root and comparison summary
- Manual review outputs should connect the inherited defect queue, the implemented fixes, and the regenerated package / replay / prior / export evidence in one auditable chain
- If planners choose to refresh prioritization at cycle close, the new cycle's regenerated machine-readable outputs should replace the current fallback-only Phase 10 provenance

</code_context>

<specifics>
## Specific Ideas

- The inherited lead target for this cycle is still `packet_construction`, not a fresh debate over `L2` vs `L4`
- The same bounded packet `phase9_comp_mech_2021_packet_01` should remain the comparison anchor for cycle 1
- The cycle review should explicitly call out whether `support_cluster_too_small`, `alternative_scope_not_distinct`, `yellow_route_state_present`, `reviewer_missing`, and `decision_prior_card` moved, stayed flat, or revealed new downstream defects
- The regenerated cycle artifacts should become the machine-readable handoff for Phase 13 so the project no longer depends on Phase 10 fallback parsing alone

</specifics>

<deferred>
## Deferred Ideas

- Broadening the packet boundary with new papers just to mask current blocker codes
- Switching the lead optimization target from packet construction to broad `L2` extraction or `L4` tuning before the packet-first rerun is measured
- Building a new UI/dashboard or a separate runtime foundation for cycle management inside this phase
- Generalizing the cycle outcome beyond the current bounded computational-mechanics slice

</deferred>

---

*Phase: 12-baseline-data-cycle-1-direct-fix*
*Context gathered: 2026-04-04*
