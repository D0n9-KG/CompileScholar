# Phase 9: Bounded Packet Construction From Corpus - Context

**Gathered:** 2026-04-03 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 9 assembles one bounded topic packet from the larger shared corpus so Phase 10 can test `L3/L4` on structured multi-paper evidence instead of arbitrary random mixes. This phase chooses the topic and cutoff, records explicit inclusion and exclusion notes, defines how the bounded packet maps into downstream `support / alternative / held_out` replay inputs, and audits packet-assembly gaps before any `L3/L4` compilation claim is made. It does not claim a green multi-route replay, redesign the packet or package contracts, or expand into full replay / prior / export execution inside the same phase.

</domain>

<decisions>
## Implementation Decisions

### Topic Selection And Scope
- **D-01:** The first Phase 9 packet should start from the Phase 8 owner-queue exemplars and fixed-regression slice, which already form a coherent data-driven / multiscale computational mechanics cluster (`1000`, `1001`, `1005`, `1017`, `1023` as the strongest seed papers).
- **D-02:** Topic expansion beyond those seed papers should stay filesystem-first and conservative: add only nearby corpus papers needed to make one topic boundary, one cutoff year, and one audited inclusion / exclusion story defensible.
- **D-03:** Topic and cutoff selection are manual, auditable packet-builder choices in this phase, not a new automatic discovery or clustering system.

### Packet Artifact And Role Structure
- **D-04:** The canonical Phase 9 artifact should remain a file-based audited `RoutePacket` plus human-readable packet selection notes, following the existing committed packet pattern instead of introducing a UI surface, database object, or ad hoc schema.
- **D-05:** The packet should keep the existing paper-level item roles already encoded in `RoutePacket.included_items`: `core_method`, `resource_or_benchmark`, `limitation_or_critique`, `survey_or_review`, and `alternative_route`.
- **D-06:** The downstream `support / alternative / held_out` grouping needed by replay and review should be recorded as a layer on top of the bounded packet, not by replacing the current `RoutePacket` schema. The Phase 9 handoff may be subset packet definitions or an explicit companion assembly manifest, but the mapping must be written down clearly enough for Phase 10 to consume without inventing new slicing rules.

### Trace Reuse And Assembly Readiness
- **D-07:** Phase 9 should reuse existing Phase 8 per-paper traces and Phase 7 inventory metadata wherever possible. Additional corpus papers may still enter the packet-selection audit, but missing trace coverage must remain an explicit blocker or yellow/red packet flag rather than trigger a hidden new extraction path inside this phase.
- **D-08:** Packet readiness reporting must keep availability and corpus-path issues separate from semantic packet quality, continuing the Phase 7 and Phase 8 rule that missing sources and missing traces are not silently mixed into model-quality conclusions.
- **D-09:** The assembly audit for Phase 9 must explicitly record topic-boundary confidence, cutoff discipline, inclusion reasons, exclusion reasons, trace coverage, role coverage, weak support density, and whether `alternative` or `held_out` downstream slices are still underspecified.

### Handoff Boundary To Phase 10
- **D-10:** Phase 9 ends when one bounded packet and its downstream role mapping are defined clearly enough that Phase 10 can attempt route-state-package compilation and replay or report explicit blockers. Phase 9 does not claim a green packaged replay, `L4` prior induction, or audited export on its own.

### the agent's Discretion
- Exact committed manifest and note filenames, as long as they follow the repo's existing `docs/replay/...` + `tmp/...` audited-artifact pattern
- Whether the Phase 10 handoff is expressed as subset packet manifests, a skeletal route-state-package manifest, or both, as long as `support / alternative / held_out` mapping is explicit
- The exact cutoff year inside the chosen computational-mechanics slice once the inclusion / exclusion audit is complete

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project And Phase Framing
- `.planning/PROJECT.md` - Current `v1.1` strategy and the rule that `L3/L4` must use bounded topic packets rather than arbitrary random paper mixes
- `.planning/REQUIREMENTS.md` - `PACK-01` plus the requirement that assembly gaps be made explicit before `L3/L4` runs
- `.planning/ROADMAP.md` - Phase 9 goal, success criteria, and handoff to Phase 10
- `.planning/STATE.md` - Current milestone handoff and the explicit note to use the Phase 8 owner queue to guide topic choice and role assignment

### Prior Phase Constraints And Inputs
- `.planning/phases/01-replay-pilot-packetization/01-CONTEXT.md` - Auditable packet discipline, leakage policy, and the earlier rule that bounded packet quality matters more than raw paper volume
- `.planning/phases/07-corpus-sampling-and-regression-baseline/07-CONTEXT.md` - Filesystem-first corpus inventory and corpus-health separation
- `.planning/phases/08-sampled-single-paper-l2-regression/08-CONTEXT.md` - Phase 7 sample baseline as source of truth and the measured sampled-paper handoff into packet work
- `docs/replay/corpus_sampling/phase7-fixed-regression-set.json` - The fixed sampled-paper slice that already anchors the first coherent topic cluster
- `docs/replay/reports/phase8-sampled-single-paper-l2-regression-baseline.md` - Owner queue and exemplar ids that should seed bounded packet selection
- `tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/comparison_inspection.json` - Detailed per-paper evidence behind the owner buckets and seed-paper trace quality

### Packet Contracts And Committed Examples
- `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md` - Why packet quality and packet-first progression matter for `L3/L4`
- `docs/superpowers/specs/2026-04-01-logickg-route-packet-schema.md` - Canonical `RoutePacket` contract, role coverage, cutoff discipline, and packet quality flags
- `docs/replay/pilot_packets/README.md` - Repo rule that committed artifacts capture bounded packet definitions while machine-local trace paths stay local
- `docs/replay/pilot_packets/phase1-route-packet.json` - Existing committed packet example
- `docs/replay/pilot_packets/phase1-selection-notes.md` - Existing human-readable inclusion / exclusion and leakage-note pattern

### Downstream Package And Replay Boundary
- `docs/replay/reports/route-state-package-pilot-2026-04-02.md` - Existing `support / alternative / held_out` package baseline and the meaning of those downstream roles
- `backend/app/research_logic/models.py` - Current `RoutePacket` item roles and packet quality gates
- `backend/app/research_logic/route_state_package.py` - Existing package-level role model and validation rules
- `backend/scripts/run_route_state_package.py` - Current package compilation CLI boundary
- `backend/scripts/run_replay_pilot.py` - Current replay entrypoint that consumes route packets plus optional package bundles
- `backend/app/research_logic/replay_io.py` - Trace-coverage checks and bundle-writing conventions
- `backend/tests/test_route_state_package.py` - Regression boundary for package completeness and blocker flags

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `backend/app/research_logic/corpus_sampling.py`: filesystem-backed inventory, fixed/random sample metadata, and eligibility handling from Phase 7
- `backend/app/research_logic/sampled_single_paper.py`: owner-bucket logic and the existing Phase 8 handoff surface from sampled-paper evidence into next-step prioritization
- `backend/app/research_logic/models.py`: typed `RoutePacket` contract, packet quality model, and the current item-role vocabulary
- `backend/app/research_logic/replay_io.py`: `ensure_packet_trace_coverage()` plus the existing manifest / summary / inspection bundle pattern
- `backend/app/research_logic/route_state_package.py`: downstream package-role model for `primary / support / alternative / held_out`
- `docs/replay/pilot_packets/phase1-route-packet.json`: concrete packet example the new corpus packet can mirror structurally
- `tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/paper_artifacts/`: the natural first trace-reuse surface for seed papers already rerun in Phase 8

### Established Patterns
- The repo prefers file-based, typed, auditable workflow artifacts over UI-driven or database-only orchestration
- Corpus availability problems stay separate from semantic quality conclusions
- Packet item roles and package roles are distinct layers in the current architecture and should stay distinct
- Multi-paper work remains bounded and manually auditable; arbitrary random mixes are explicitly out of scope

### Integration Points
- Phase 9 should read the Phase 7 fixed manifest and the Phase 8 owner-bucket outputs to choose the first bounded computational-mechanics slice
- The resulting packet artifact must hand off cleanly to the existing Phase 10 route-state-package and replay tooling without needing a schema rewrite
- If newly selected papers lack traces, the packet audit should surface that blocker so Phase 10 can treat it as an explicit preflight failure instead of a mysterious replay regression
- New tests for Phase 9 should live under `backend/tests/` and focus on packet-selection reproducibility, packet-note completeness, and assembly-gap reporting

</code_context>

<specifics>
## Specific Ideas

- The first corpus-backed packet should likely stay within the data-driven / multiscale computational mechanics slice already represented by fixed papers `1000`, `1001`, `1005`, `1017`, and `1023`, because that is the only slice Phase 8 has already measured deeply enough to provide an evidence-backed starting point
- The Phase 8 owner queue (`relation_assembly`, then `slot_recovery`) should inform not only which seed papers to start from, but also what evidence density the packet needs before it can be considered compile-ready
- The packet should be bounded tightly enough that a later `support / alternative / held_out` split can be derived without redefining the topic boundary halfway through Phase 10
- Runtime traces already produced in `tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/` are the natural first reuse point for seed-paper trace refs

</specifics>

<deferred>
## Deferred Ideas

- Automatic full-corpus topic discovery or route-family clustering
- A dedicated UI or database-backed packet-management surface
- Full route-state-package compilation, replay validation, prior induction, or decision-episode export inside Phase 9 itself
- Cross-topic or multi-packet scale-out beyond the first bounded corpus-backed packet

</deferred>

---

*Phase: 09-bounded-packet-construction-from-corpus*
*Context gathered: 2026-04-03*
