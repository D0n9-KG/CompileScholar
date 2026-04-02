# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 - Scientific Reasoning Compiler Pilot

**Shipped:** 2026-04-02
**Phases:** 6 | **Plans:** 16 | **Tasks:** 36

### What Was Built

- A bounded `RoutePacket` -> replay workflow for a real topic/cutoff slice.
- A typed `L1 HistoricalEnvironmentSnapshot` layer that feeds replay and route-state compilation.
- Grouped multi-route `RouteState` packaging, validation, replay inspection, prior review, and audited `DecisionEpisode` export.

### What Worked

- Keeping the work bounded to one real topic made failures interpretable instead of abstract.
- Treating replay bundles, review bundles, and export bundles as separate artifact families kept audit boundaries clear.
- Re-running the real jamming slice was the fastest way to settle route-family carryover and closeout questions.

### What Was Inefficient

- Several important pilot assets stayed under `tmp/` too long, which made closeout feel later and messier than it needed to.
- Early strict route-id matching hid a family-level carryover assumption that only became obvious during the real export rerun.
- Support density for package-level priors stayed thin, so some late-stage review debates were really data-shape issues.

### Patterns Established

- Use bounded packet/replay artifacts as the source of truth for roadmap decisions.
- Keep review allowlists separate from live replay outputs.
- Prefer stable family-level identifiers when route variants are different views of the same historical route.

### Key Lessons

1. A truthful rerun on real artifacts is usually cheaper and more reliable than arguing over abstract policy.
2. Audit surfaces should preserve debt explicitly; "flat but honest" is better than a false improvement story.
3. The next milestone should canonicalize working pilot assets earlier so closeout is about scope, not file location.

### Cost Observations

- Model mix: not normalized across the milestone; execution combined direct implementation, backfilled planning, and closeout cleanup passes.
- Sessions: not normalized; work landed through multiple phase execution and audit-remediation waves.
- Notable: the route-family fix was proven fastest by replaying the real bounded slice instead of inventing a separate mapping subsystem.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| `v1.0` | not normalized | 6 | Switched from "single-paper extraction quality" to a bounded route-packet -> review -> export compiler workflow. |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|-------------------|
| `v1.0` | targeted regression suites across Phases 1-6 | not normalized | multiple new artifact contracts and CLIs without adding external runtime services |

### Top Lessons (Verified Across Milestones)

1. Not enough history yet; cross-milestone lessons start accumulating after the next shipped milestone.
