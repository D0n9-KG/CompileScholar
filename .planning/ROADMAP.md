# Milestone v1.2: Fast Iteration Research Logic Quality

**Status:** ACTIVE 2026-04-03
**Phases:** 12-16
**Requirements:** 9 mapped
**Numbering Mode:** Continue from `v1.1`

## Overview

`v1.2` turns the packet-first recommendation from Phase 11 into a high-speed optimization loop.

The milestone focuses on reducing cycle latency while improving actual reasoning quality on real artifacts. Success is measured by both rule-level quality gates and explicit review of generated reasoning outputs.

## Phase Summary

| Phase | Name | Goal | Requirements | Success Criteria |
|-------|------|------|--------------|------------------|
| 12 | Unified Iteration Runner And Telemetry | Build a reproducible single-entry optimization run with stage-level timing and artifact lineage. | `ITER-01`, `ITER-02` | 3 |
| 13 | Packet Blocker Recovery Loop | Resolve current packet-quality blockers with auditable role-balance evidence and reduced manual handoff steps. | `ITER-03`, `PACK-03`, `PACK-04` | 4 |
| 14 | L4 Recovery Revalidation | Re-run replay/prior surfaces on repaired packets and compare against committed baseline artifacts. | `AGGR-02`, `AGGR-03` | 3 |
| 15 | Real-Outcome Quality Gate | Enforce pass/fail gates that combine rule thresholds with explicit review of real reasoning outputs. | `QUAL-01` | 3 |
| 16 | Next-Cycle Prioritization Refresh | Produce the next ranked recommendation from fresh evidence and close the iteration loop. | `QUAL-02` | 3 |

## Phase Details

### Phase 12: Unified Iteration Runner And Telemetry

**Goal:** Provide a single reproducible execution path for one full optimization cycle and capture timing/bottleneck telemetry.
**Depends on:** Phase `11`
**Requirements:** `ITER-01`, `ITER-02`

**Success criteria:**
1. One entrypoint can run packet -> replay -> report generation with explicit input/output roots.
2. Stage timings and bottleneck stage are recorded in machine-readable outputs.
3. Artifact lineage lists exactly which sources and generated files were used for the run.

### Phase 13: Packet Blocker Recovery Loop

**Goal:** Implement a focused fix cycle for current packet blockers while preserving auditable role balance and minimizing manual glue work.
**Depends on:** Phase `12`
**Requirements:** `ITER-03`, `PACK-03`, `PACK-04`

**Success criteria:**
1. Current package blockers are either resolved or emitted as explicit unresolved blockers with evidence.
2. Role-balance evidence (`support`/`alternative`/`held_out`) is exported in machine-readable form per cycle.
3. Operators can rerun packet fix workflows without manual artifact stitching.
4. Updated packet outputs remain comparable to the committed baseline.

### Phase 14: L4 Recovery Revalidation

**Goal:** Verify whether packet fixes recover downstream `L4` and prior-review quality relative to baseline.
**Depends on:** Phase `13`
**Requirements:** `AGGR-02`, `AGGR-03`

**Success criteria:**
1. Repaired packet outputs are rerun through replay/prior surfaces with baseline comparison.
2. Reports explicitly show prior candidate counts, accepted prior ids, and anti-pattern carryover deltas.
3. The run identifies whether remaining issues are still packet-first or have shifted to `L4`/`L2`.

### Phase 15: Real-Outcome Quality Gate

**Goal:** Make promotion contingent on real artifact quality, not only aggregate metric movement.
**Depends on:** Phase `14`
**Requirements:** `QUAL-01`

**Success criteria:**
1. Quality gate evaluates both rule thresholds and real reasoning artifact review checks.
2. Any metric-only improvement without real-outcome quality gain is rejected.
3. Verification outputs clearly state pass/fail reasons and blocking evidence.

### Phase 16: Next-Cycle Prioritization Refresh

**Goal:** Close the loop with a refreshed recommendation queue grounded in the latest repaired-cycle evidence.
**Depends on:** Phase `15`
**Requirements:** `QUAL-02`

**Success criteria:**
1. A new ranked recommendation queue is generated from current cycle artifacts.
2. Recommendation includes explicit rationale and provenance for top choice.
3. The milestone ends with a clear next-cycle focus and audit-ready verification note.

## Coverage

| Requirement | Phase |
|-------------|-------|
| `ITER-01` | Phase `12` |
| `ITER-02` | Phase `12` |
| `ITER-03` | Phase `13` |
| `PACK-03` | Phase `13` |
| `PACK-04` | Phase `13` |
| `AGGR-02` | Phase `14` |
| `AGGR-03` | Phase `14` |
| `QUAL-01` | Phase `15` |
| `QUAL-02` | Phase `16` |

**Coverage status:** `9/9` requirements mapped

## Notes

- This roadmap continues numbering from `v1.1`; no phase renumber reset is used.
- v1.2 quality decisions must be grounded in real reasoning outputs, not only numeric deltas.
- The committed Phase 10/11 artifacts remain the baseline comparison anchor for this milestone.

## Next Up

**Phase 12: Unified Iteration Runner And Telemetry** - build the single-entry cycle runner and timing/lineage outputs.

`$gsd-discuss-phase 12`

Also available: `$gsd-plan-phase 12`

---
*Roadmap created: 2026-04-03*
*Last updated: 2026-04-03 for milestone v1.2 initialization*
