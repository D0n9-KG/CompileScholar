# LogicKG

## What This Is

LogicKG is a scientific-reasoning compiler workbench built on top of the existing paper and textbook platform.

`v1.0` proved a bounded end-to-end `RoutePacket -> replay -> review -> DecisionEpisode` export flow on a real jamming slice. `v1.1` now focuses on using a large shared paper corpus to iteratively harden single-paper extraction quality and then validate bounded multi-paper packet compilation for `L3/L4`.

## Core Value

Produce auditable, replayable, trainable scientific-reasoning artifacts rather than paper-like summaries or one-off knowledge graphs.

## Current State

- Shipped milestone: `v1.0` on `2026-04-02`
- Active planning milestone: `v1.1 Corpus-Driven Iterative Quality Hardening`
- Phase `7` completed on `2026-04-03` with a committed ten-paper fixed regression set and a real seed-`7` random exploration batch of five papers
- The first real Phase `7` corpus baseline recorded `1756` inventory entries, `1755` eligible entries, and `1505` corpus-health failures
- Stable bounded pilot: `jamming transition in frictionless sphere packings near point J` with `cutoff_year = 2010`
- End-to-end Phase 03 -> 06 artifact flow is verified on real local bundles
- Reviewed anti-pattern carryover now works across route variants through stable `route_family_id`
- A user-provided shared corpus root now provides roughly `1756` `.txt` papers and `945` `.md` derivatives for iterative sampling
- Recursive corpus scans already surfaced broken or missing subpaths, so corpus-health tolerance is part of the next optimization loop rather than a separate cleanup task

## Current Milestone: v1.1 Corpus-Driven Iterative Quality Hardening

**Goal:** Use the large shared corpus to create a repeatable optimization loop that combines fixed regression papers, random exploration papers, and bounded multi-paper packets so `L2`, `L3`, and `L4` can be improved with evidence instead of guesswork.

**Target features:**
- a corpus sampling workflow with one fixed regression set and one random exploration set per iteration cycle
- repeatable sampled single-paper evaluation that exposes `L2` trace, schema, and evidence-slot weaknesses
- bounded multi-paper packet construction from the larger corpus with explicit `support / alternative / held_out` roles for `L3/L4`
- an iteration summary that turns sampled-paper failures and multi-paper packet outcomes into a prioritized optimization queue

## Requirements

### Validated

- [x] Existing platform baseline for ingest, graph operations, Ask, similarity/community, and ops workflows is solid enough to host the reasoning layer.
- [x] A bounded `RoutePacket` plus replay contract now exists and has been exercised on a real topic slice in `v1.0`.
- [x] `L1 HistoricalEnvironmentSnapshot` assets can be built and consumed without violating cutoff discipline in `v1.0`.
- [x] `RouteState / WhyNowCase / RouteComparisonCase` now compile from real multi-paper replay packages in `v1.0`.
- [x] Replay failure taxonomy and bounded `L2` repair loops now exist in `v1.0`.
- [x] `DecisionPriorCard / AntiPatternCard` review workflow and audited `DecisionEpisode` export both shipped in `v1.0`.
- [x] Phase `7` now provides a filesystem-first corpus sampling workflow with one committed fixed regression set, one reproducible random exploration batch, and explicit corpus-health reporting.

### Active

- [ ] Repeatedly run sampled single-paper extraction and evaluation so `L2` quality issues can be measured instead of guessed.
- [ ] Assemble bounded multi-paper packets from the larger corpus so `L3/L4` can be validated on structured topic slices rather than arbitrary random mixes.
- [ ] Use evidence from sampled-paper runs and bounded packet runs to choose the next optimization cycle.

### Out of Scope

- Running all `1000+` candidate papers end to end in one `v1.1` sweep.
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
| Use topic-bounded multi-paper packets rather than unconstrained random mixes for `L3/L4` | `L3/L4` need structured `support / alternative / held_out` evidence, not arbitrary co-occurrence | `Pending` |

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
*Last updated: 2026-04-03 after Phase 7 execution and verification*
