---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: corpus-driven-iterative-quality-hardening
current_phase: 7
current_phase_name: corpus-sampling-and-regression-baseline
current_plan: null
status: roadmap_revised
stopped_at: Milestone v1.1 was refocused around corpus-driven iterative quality hardening; Phase 7 is ready for discussion or planning
last_updated: "2026-04-02T16:07:16.1584826Z"
last_activity: 2026-04-03
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-04-03)

**Core value:** Produce auditable, replayable, trainable scientific-reasoning artifacts rather than paper-like summaries or one-off knowledge graphs.
**Current focus:** Milestone `v1.1` roadmap is revised; Phase `7` is next

## Current Position

Milestone: `v1.1` - ROADMAP REVISED

- Active phase: none yet
- Next phase: `07 corpus-sampling-and-regression-baseline`
- Status: ready for discussion or direct phase planning
- Last activity: `2026-04-03` - refined milestone `v1.1` around corpus-driven iteration

## Milestone Snapshot

- `v1.1` continues numbering after `v1.0`, so the active roadmap starts at Phase `7`.
- The milestone now uses a large shared corpus as the candidate pool for fixed regression papers and random exploration papers.
- `L2` is iterated on sampled single papers, while `L3/L4` are validated on bounded topic packets rather than arbitrary random mixes.
- `10/10` milestone requirements are mapped across five planned phases.
- Corpus scans already surfaced broken or missing paths, so corpus-health reporting is part of the evaluation loop.

## Decisions Carried Forward

- Keep `L2` as the single-paper evidence layer and use multi-paper compilation for `L3/L4`.
- Keep replay, review, and export artifact families separate.
- Preserve bounded pilot truth instead of overclaiming generalized dataset readiness.
- Use a fixed regression set plus random exploration samples to avoid overfitting to one tiny benchmark.
- Use topic-bounded packets rather than unconstrained random mixes for `L3/L4`.

## Accumulated Context

- The user-provided shared corpus root currently exposes roughly `1756` `.txt` papers and `945` `.md` derivatives.
- Recursive scans over that corpus already surfaced missing or broken subpaths that should be tracked as data-health issues.
- The jamming slice remains the current baseline reference for replay, review, and export behavior.
- The Phase 6 `route_family_id` carryover fix is part of the baseline that `v1.1` must preserve while it broadens beyond the original pilot.

## Active Requirements

- Phase `7`: `SAMPLE-01`, `SAMPLE-02`, `SAMPLE-03`
- Phase `8`: `L2Q-01`, `L2Q-02`
- Phase `9`: `PACK-01`
- Phase `10`: `PACK-02`, `AGGR-01`
- Phase `11`: `LOOP-01`, `LOOP-02`

## Pending Follow-Ups

- Define the fixed regression paper set and random exploration sampling loop over the shared corpus.
- Build sampled single-paper evaluation summaries that can reveal recurring `L2` failure owners.
- Assemble one bounded topic packet from the larger corpus for `L3/L4` validation.
- End the first corpus-driven cycle with an explicit next-iteration priority list.

## Blockers

- Shared-corpus path instability may create access failures that need to be separated from model-quality failures.
- `L3/L4` still require role-balanced bounded packets, so arbitrary random paper batches are not sufficient for multi-paper validation.
- Existing `.planning/phases/01-*` through `06-*` directories are still present, so the roadmap continues numbering from Phase `7`.

## Session

**Last Date:** 2026-04-03
**Stopped At:** Refined milestone `v1.1` and prepared the revised roadmap
**Resume File:** Start with `.planning/ROADMAP.md`, `.planning/REQUIREMENTS.md`, and `$gsd-discuss-phase 7`
