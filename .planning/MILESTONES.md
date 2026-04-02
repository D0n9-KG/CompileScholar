# Milestones

## v1.0 - Scientific Reasoning Compiler Pilot

**Shipped:** 2026-04-02
**Status:** shipped with bounded `tech_debt`
**Phases:** 6
**Plans:** 16
**Tasks:** 36
**Tag:** `v1.0`
**Archive Files:**

- `.planning/milestones/v1.0-ROADMAP.md`
- `.planning/milestones/v1.0-REQUIREMENTS.md`
- `.planning/milestones/v1.0-MILESTONE-AUDIT.md`

### Delivered

- Shipped the first committed bounded `RoutePacket` and replay workflow for a real topic/cutoff slice.
- Added a typed `L1 HistoricalEnvironmentSnapshot` layer and connected it to route-state and replay compilation.
- Standardized grouped `support / alternative / held_out` route-state packaging plus package/replay inspection outputs.
- Added structured failure taxonomy reporting and used it to drive bounded `L2` repair work.
- Added auditable prior / anti-pattern review bundles with explicit accepted ids.
- Added a real audited `DecisionEpisode` export CLI and fixed reviewed anti-pattern carryover through stable `route_family_id` matching.

### Known Debt

- The bounded jamming replay, review, and export bundles still live under `tmp/` rather than committed canonical asset roots.
- Cross-topic validation has not yet exercised the full Phase 03 -> 06 package -> review -> export workflow.
- The package-level prior support cluster is still intentionally sparse, so accepted priors remain empty.
- The earliest live pilot still depended on a runtime subset packet because two packet entries lacked resolved local trace exports.

### Notes

- Audit outcome at closeout: `requirements 13/13`, `phases 6/6`, `integration 2/2`, `flows 2/2`.
- The milestone was closed on branch `codex/l2-content-quality`; no branch merge or remote push is implied by this document.
