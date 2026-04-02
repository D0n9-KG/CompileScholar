# LogicKG

## What This Is

LogicKG is a scientific-reasoning compiler workbench built on top of the existing paper and textbook platform.

`v1.0` proved a bounded end-to-end `RoutePacket -> replay -> review -> DecisionEpisode` export flow on a real jamming slice. `v1.1` now focuses on turning that bounded success into a reproducible canonical baseline and validating the same flow on one additional topic before the project chooses its next major direction.

## Core Value

Produce auditable, replayable, trainable scientific-reasoning artifacts rather than paper-like summaries or one-off knowledge graphs.

## Current State

- Shipped milestone: `v1.0` on `2026-04-02`
- Active planning milestone: `v1.1 Canonical Asset Promotion And Cross-Topic Validation`
- Stable bounded pilot: `jamming transition in frictionless sphere packings near point J` with `cutoff_year = 2010`
- End-to-end Phase 03 -> 06 artifact flow is verified on real local bundles
- Reviewed anti-pattern carryover now works across route variants through stable `route_family_id`
- The committed repo captures the primary pilot packet and reports, but the full package/review/export truth is still split between committed docs and `tmp/` runtime bundles

## Current Milestone: v1.1 Canonical Asset Promotion And Cross-Topic Validation

**Goal:** Convert the bounded jamming pilot into a reproducible canonical baseline and validate the same package -> review -> export flow on a second topic so the next major milestone direction is evidence-backed.

**Target features:**
- committed canonical manifests and audit notes for the bounded jamming packet, subset packets, replay package, review bundle, and export bundle boundaries
- reproducible local rerun instructions and scripts that rebuild the canonical jamming flow without hardcoded machine-specific paths
- one second bounded topic that runs through package, review, and audited export with comparable quality summaries
- a milestone closeout decision package that recommends either ops/productization or question-discovery for the following milestone

## Requirements

### Validated

- [x] Existing platform baseline for ingest, graph operations, Ask, similarity/community, and ops workflows is solid enough to host the reasoning layer.
- [x] A bounded `RoutePacket` plus replay contract now exists and has been exercised on a real topic slice in `v1.0`.
- [x] `L1 HistoricalEnvironmentSnapshot` assets can be built and consumed without violating cutoff discipline in `v1.0`.
- [x] `RouteState / WhyNowCase / RouteComparisonCase` now compile from real multi-paper replay packages in `v1.0`.
- [x] Replay failure taxonomy and bounded `L2` repair loops now exist in `v1.0`.
- [x] `DecisionPriorCard / AntiPatternCard` review workflow and audited `DecisionEpisode` export both shipped in `v1.0`.

### Active

- [ ] Canonicalize the jamming pilot asset surfaces beyond packet-level docs so the repo reflects the full bounded compiler flow instead of pointing back to `tmp/` bundles.
- [ ] Rebuild the bounded jamming compiler flow from committed manifests plus local trace inputs and keep provenance and audit boundaries explicit.
- [ ] Run one second-topic Phase 03 -> 06 validation slice and compare its package, review, and export quality against the jamming baseline.
- [ ] Finish `v1.1` with an evidence-backed recommendation for whether the next milestone should prioritize operational productization or question-discovery work.

### Out of Scope

- Open-ended question discovery or hypothesis generation before a second topic proves the reviewed `DecisionEpisode` flow generalizes.
- Full UI productization of packet/replay/review/export operations during the same milestone; `v1.1` is about proving and packaging the flow, not polishing every operator surface.
- Full-corpus automatic packet discovery before canonical bounded assets and rerun provenance are stable.
- A wholesale `L2` rewrite unless cross-topic validation isolates a specific failure owner that justifies it.

## Context

- Tech stack remains `FastAPI + React + Vite + Neo4j + FAISS`, with the reasoning flow layered into the existing platform instead of split into a second prototype.
- `v1.0` proved the architecture can run a real bounded route-packet -> review -> export loop without toy fixtures.
- Committed packet docs already exist under `docs/replay/pilot_packets/`, but the Phase 03 -> 06 package/review/export truth is still partially runtime-only.
- `tmp/phase3_route_state_package/`, `tmp/phase5_multi_route_prior_induction/`, and `tmp/phase6_decision_episode_audit_export/` remain the key evidence surfaces that `v1.1` needs to canonicalize or rehydrate cleanly.
- The second-topic validation must stay bounded and auditable rather than broadening into generalized benchmark claims.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep `L2` as the single-paper evidence layer | `L3/L4` should emerge from multi-paper historical aggregation, not from rewriting a single-paper trace | `Good` |
| Use replay outputs and failures to steer the roadmap | Real artifact reruns expose the highest-value gaps faster than speculative redesign | `Good` |
| Keep review bundles separate from replay bundles | Review state must stay explicit and auditable instead of being hidden inside replay output prose | `Good` |
| Keep export bundles separate from replay and review bundles | Audit-grade packaging should not mutate runtime artifacts | `Good` |
| Prefer conservative audit truth over optimistic promotion stories | Honest bounded outputs are more useful than overclaiming generalized readiness | `Good` |
| Use stable `route_family_id` for reviewed anti-pattern carryover | The real mismatch was between route variants in the same family, not unrelated route ids | `Good` |
| Use `v1.1` to settle the next direction with canonicalization plus one cross-topic run | The current next-step uncertainty is about maturity and reuse, not about inventing a brand-new reasoning layer | `Pending` |

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
*Last updated: 2026-04-02 after starting v1.1 milestone planning*
