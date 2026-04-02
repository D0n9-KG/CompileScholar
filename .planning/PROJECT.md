# LogicKG

## What This Is

LogicKG is now a working scientific-reasoning compiler pilot built on top of the existing paper and textbook workbench.

v1.0 ships a bounded but real workflow that turns:

- packetized historical slices
- paper-grounded `L1` snapshots
- multi-paper `RouteState` packages
- replay inspection and failure reports
- reviewed priors / anti-patterns
- audited `DecisionEpisode` exports

into auditable artifacts that can be rerun and inspected end to end.

The project goal is no longer "make the paper QA system stronger." The shipped direction is to compile historically bounded, reviewable reasoning artifacts that can later support question discovery and hypothesis work.

## Core Value

Produce auditable, replayable, trainable scientific-reasoning artifacts rather than paper-like summaries or one-off knowledge graphs.

## Current State

- Shipped milestone: `v1.0` on `2026-04-02`
- Audit closeout status: `tech_debt`
- Stable bounded pilot: `jamming transition in frictionless sphere packings near point J` with `cutoff_year = 2010`
- End-to-end Phase 03 -> 06 artifact flow is verified on real local bundles
- Reviewed anti-pattern carryover now works across route variants through stable `route_family_id`

What is now real rather than planned:

- committed bounded packet selection and replay entry points
- typed `HistoricalEnvironmentSnapshot` assets that feed replay
- grouped `support / alternative / held_out` route-state packaging
- replay failure taxonomy and rerun comparison artifacts
- auditable prior / anti-pattern review bundles
- dedicated audited `DecisionEpisode` export bundles and CLI

## Requirements

### Validated

- [x] Existing platform baseline for ingest, graph operations, Ask, similarity/community, and ops workflows is solid enough to host the reasoning layer.
- [x] A bounded `RoutePacket` plus replay contract now exists and has been exercised on a real topic slice in `v1.0`.
- [x] `L1 HistoricalEnvironmentSnapshot` assets can be built and consumed without violating cutoff discipline in `v1.0`.
- [x] `RouteState / WhyNowCase / RouteComparisonCase` now compile from real multi-paper replay packages in `v1.0`.
- [x] Replay failure taxonomy and bounded `L2` repair loops now exist in `v1.0`.
- [x] `DecisionPriorCard / AntiPatternCard` review workflow and audited `DecisionEpisode` export both shipped in `v1.0`.

### Active

- [ ] Promote the current bounded jamming packet, replay, review, and export artifacts from `tmp/` into committed canonical assets.
- [ ] Run at least one cross-topic Phase 03 -> 06 validation slice to confirm that the current packaging, review, and export rules generalize beyond the jamming pilot.
- [ ] Decide whether the next milestone should focus on productizing packet/review/export operations or on building question-discovery work on top of stable `DecisionEpisode` samples.
- [ ] Improve thin support density and resolve the remaining Phase 01 pilot limitations where runtime subset execution still covered for missing local trace exports.

### Out of Scope

- Open-ended hypothesis generation or unconstrained `L4` ideation before reviewed priors and negative patterns are stable.
- Full-corpus automatic packet discovery before packet, review, and export contracts are canonicalized.
- A wholesale `L2` rewrite before replay-driven evidence justifies it.
- Treating `L3/L4` as single-paper abstractions instead of multi-paper historical compilation layers.

## Context

- Tech stack remains `FastAPI + React + Vite + Neo4j + FAISS`, with the new reasoning flow intentionally layered into the existing platform instead of spun out into a second prototype.
- v1.0 proved that the current architecture can run a real bounded route-packet -> review -> export loop without relying on toy fixtures.
- The milestone is complete, but the current success is still bounded. Canonical asset promotion and cross-topic validation are the main next-step questions, not new schema invention.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep `L2` as the single-paper evidence layer | `L3/L4` should emerge from multi-paper historical aggregation, not from rewriting a single-paper trace | `Good` |
| Use replay outputs and failures to steer the roadmap | Real artifact reruns expose the highest-value gaps faster than speculative redesign | `Good` |
| Keep review bundles separate from replay bundles | Review state must stay explicit and auditable instead of being hidden inside replay output prose | `Good` |
| Keep export bundles separate from replay and review bundles | Audit-grade packaging should not mutate runtime artifacts | `Good` |
| Prefer conservative audit truth over optimistic promotion stories | Honest bounded outputs are more useful than overclaiming generalized readiness | `Good` |
| Use stable `route_family_id` for reviewed anti-pattern carryover | The real mismatch was between route variants in the same family, not unrelated route ids | `Good` |
| Delay broad productization until pilot assets are canonical | `tmp/`-only success is enough to close v1.0, but not enough to claim operational maturity | `Revisit next milestone` |

## Next Milestone Goals

1. Canonicalize the current bounded pilot assets so the shipped path no longer depends on `tmp/` roots for the main demonstration slice.
2. Run at least one cross-topic validation slice through Phase 03 -> 06 to stress the same packaging, review, and export seams in another domain.
3. Choose between an ops/productization milestone and a question-discovery milestone based on what the cross-topic run reveals.

---

*Last updated: 2026-04-02 after v1.0 milestone completion*
