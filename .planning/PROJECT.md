# LogicKG

## What This Is

LogicKG is a scientific-reasoning compiler workbench built on top of the existing paper and textbook platform.

`v1.0` proved a bounded end-to-end `RoutePacket -> replay -> review -> DecisionEpisode` export flow on a real jamming slice. `v1.1` established corpus-driven iteration and explicit next-cycle prioritization. `v1.2` now focuses on packet-first quality recovery for bounded multi-paper compilation and downstream `L4` signal restoration.

## Core Value

Produce auditable, replayable, trainable scientific-reasoning artifacts rather than paper-like summaries or one-off knowledge graphs.

## Current State

- Shipped milestone: `v1.0` on `2026-04-02`
- Active planning milestone: `v1.2 Fast Iteration Research Logic Quality`
- Phase `7` completed on `2026-04-03` with a committed ten-paper fixed regression set and a real seed-`7` random exploration batch of five papers
- Phase `9` completed on `2026-04-03` with a committed `2017-2021` computational-mechanics packet, explicit `support / alternative / held_out` mapping, and a runtime-backed audit report
- Phase `10` completed on `2026-04-03` with a real bounded computational-mechanics validation run, committed report/verification artifacts, and a packet-construction-first next-cycle recommendation
- Phase `11` completed on `2026-04-03` with a runtime-backed iteration prioritization bundle, committed report/verification artifacts, and a packet-construction-first next-cycle queue
- The first real Phase `7` corpus baseline recorded `1756` inventory entries, `1755` eligible entries, and `1505` corpus-health failures
- Stable bounded pilot: `jamming transition in frictionless sphere packings near point J` with `cutoff_year = 2010`
- End-to-end Phase 03 -> 06 artifact flow is verified on real local bundles
- Reviewed anti-pattern carryover now works across route variants through stable `route_family_id`
- A user-provided shared corpus root now provides roughly `1756` `.txt` papers and `945` `.md` derivatives for iterative sampling
- Recursive corpus scans already surfaced broken or missing subpaths, so corpus-health tolerance is part of the next optimization loop rather than a separate cleanup task

## Current Milestone: v1.2 Fast Iteration Research Logic Quality

**Goal:** Build a fast, evidence-driven optimization loop that improves the full research reasoning production line (`packet -> replay -> prior/review -> export`) until quality is genuinely production-worthy in real outputs, not only better on rule metrics.

**Target features:**
- a packet-quality repair loop that explicitly resolves current `package_validation` blockers and records provenance for each fix
- a fast rerun workflow that compares repaired packet outcomes against the existing baseline with minimal manual steps
- an `L4` recovery check that tracks prior-candidate and anti-pattern carryover readiness after packet fixes
- a rolling prioritization summary that keeps each cycle focused and avoids slow one-by-one unstructured patching
- a result-level quality gate that inspects real reasoning artifacts and rejects metric-only gains that do not improve actual scientific reasoning outcomes
## Requirements

### Validated

- [x] Existing platform baseline for ingest, graph operations, Ask, similarity/community, and ops workflows is solid enough to host the reasoning layer.
- [x] A bounded `RoutePacket` plus replay contract now exists and has been exercised on a real topic slice in `v1.0`.
- [x] `L1 HistoricalEnvironmentSnapshot` assets can be built and consumed without violating cutoff discipline in `v1.0`.
- [x] `RouteState / WhyNowCase / RouteComparisonCase` now compile from real multi-paper replay packages in `v1.0`.
- [x] Replay failure taxonomy and bounded `L2` repair loops now exist in `v1.0`.
- [x] `DecisionPriorCard / AntiPatternCard` review workflow and audited `DecisionEpisode` export both shipped in `v1.0`.
- [x] Phase `7` now provides a filesystem-first corpus sampling workflow with one committed fixed regression set, one reproducible random exploration batch, and explicit corpus-health reporting.
- [x] Phase `9` now provides one bounded corpus packet with explicit inclusion / exclusion notes plus `support / alternative / held_out` role mapping for downstream multi-paper work.
- [x] Phase `10` now compiles replay/package/review/export artifacts for the committed bounded packet, compares them against the jamming baseline, and keeps remaining multi-paper blockers explicit for the next cycle.
- [x] Repeated sampled single-paper evidence plus bounded packet validation now feed one auditable prioritization summary for next-cycle decisions. (Validated in Phase `11`)
- [x] The next optimization cycle can now be selected from explicit evidence (`packet_construction`, `l4_aggregation`, `l2_extraction`) instead of open-ended debate. (Validated in Phase `11`)

### Active

- [ ] Packet construction blockers from the current bounded slice can be resolved with explicit role-balance and support-density evidence.
- [ ] Repaired packet runs can demonstrate whether `L4` prior/review surfaces recover relative to the committed baseline.
- [ ] The next optimization cycle can be re-prioritized from a new summary grounded in the repaired packet evidence chain.
- [ ] Quality promotion requires both rule-threshold pass and explicit review of real reasoning outputs; metric-only improvement is not sufficient.

### Out of Scope

- Running all `1000+` candidate papers end to end in one `v1.2` sweep.
- Letting `L3/L4` consume arbitrary random paper mixes without topic boundaries or explicit role assignment.
- Full UI productization of packet/replay/review/export operations during the same milestone.
- Open-ended question discovery or hypothesis generation before iterative compiler quality is stable on more than one bounded slice.

## Context

- Tech stack remains `FastAPI + React + Vite + Neo4j + FAISS`, with the reasoning flow layered into the existing platform instead of split into a second prototype.
- `v1.0` proved the architecture can run a real bounded route-packet -> review -> export loop without toy fixtures.
- The new shared corpus is large enough to support repeated sampling without overfitting to one tiny hand-picked benchmark.
- The corpus is also messy enough that missing-path and broken-directory handling must be treated as part of evaluation hygiene.
- The jamming slice remains the current baseline reference while `v1.1` expands into corpus-driven iteration.
- Committed packet docs already exist under `docs/replay/pilot_packets/`, and they can serve as the baseline reference while the corpus-driven loop matures.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep `L2` as the single-paper evidence layer | `L3/L4` should emerge from multi-paper historical aggregation, not from rewriting a single-paper trace | `Good` |
| Use replay outputs and failures to steer the roadmap | Real artifact reruns expose the highest-value gaps faster than speculative redesign | `Good` |
| Keep review bundles separate from replay bundles | Review state must stay explicit and auditable instead of being hidden inside replay output prose | `Good` |
| Keep export bundles separate from replay and review bundles | Audit-grade packaging should not mutate runtime artifacts | `Good` |
| Prefer conservative audit truth over optimistic promotion stories | Honest bounded outputs are more useful than overclaiming generalized readiness | `Good` |
| Use stable `route_family_id` for reviewed anti-pattern carryover | The real mismatch was between route variants in the same family, not unrelated route ids | `Good` |
| Use a fixed regression set plus random exploration samples in `v1.1` | Fixed papers catch regressions while random samples keep the system from overfitting to a tiny benchmark | `Good` |
| Use filesystem inventory as the Phase 7 sampling gate and Neo4j only as enrichment | The real corpus is broader and messier than current graph coverage, so eligibility must reflect what is actually present on disk | `Good` |
| Use topic-bounded multi-paper packets rather than unconstrained random mixes for `L3/L4` | `L3/L4` need structured `support / alternative / held_out` evidence, not arbitrary co-occurrence | `Good` |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `$gsd-transition`):
1. Requirements invalidated? Move to Out of Scope with reason
2. Requirements validated? Move to Validated with phase reference
3. New requirements emerged? Add to Active
4. Decisions to log? Add to Key Decisions
5. "What This Is" still accurate? Update if drifted

**After each milestone** (via `$gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check: still the right priority?
3. Audit Out of Scope: reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-03 after starting milestone v1.2*


