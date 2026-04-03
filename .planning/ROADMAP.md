# Milestone v1.1: Corpus-Driven Iterative Quality Hardening

**Status:** ACTIVE 2026-04-03
**Phases:** 7-11
**Requirements:** 10 mapped
**Numbering Mode:** Continue from `v1.0`

## Overview

`v1.1` uses the large shared paper corpus to turn compiler improvement into an explicit loop instead of an ad hoc patch sequence.

The milestone first establishes a fixed regression set plus random exploration sampling, then hardens sampled single-paper extraction for `L2`, then assembles one bounded multi-paper packet for `L3/L4`, and finally uses the combined evidence to decide the next optimization cycle.

## Phase Summary

| Phase | Name | Goal | Requirements | Success Criteria |
|-------|------|------|--------------|------------------|
| 7 | Corpus Sampling And Regression Baseline | Turn the shared corpus into a usable source for fixed regression papers, random exploration papers, and corpus-health reporting. | `SAMPLE-01`, `SAMPLE-02`, `SAMPLE-03` | 3 |
| 8 | Sampled Single-Paper L2 Regression | Repeatedly run sampled single-paper extraction and compare results across iterations. | `L2Q-01`, `L2Q-02` | 3 |
| 9 | Bounded Packet Construction From Corpus | Build one bounded topic packet from the larger corpus with explicit multi-paper role assignment. | `PACK-01` | 3 |
| 10 | Multi-Paper L3 And L4 Validation | 3/3 | Complete    | 2026-04-03 |
| 11 | Iteration Prioritization And Next Cycle Plan | 3/3 | Complete    | 2026-04-03 |

## Phase Details

### Phase 7: Corpus Sampling And Regression Baseline

**Goal:** Turn the shared corpus into a usable evaluation source with one fixed regression set, one random exploration strategy, reproducible sampling metadata, and explicit corpus-health reporting.
**Depends on:** `v1.0` archived baseline
**Requirements:** `SAMPLE-01`, `SAMPLE-02`, `SAMPLE-03`
**Status:** Complete on `2026-04-03`

**Success criteria:**
1. The project defines a fixed regression paper set and a random exploration sampling method over the shared corpus.
2. Each sampling run records selected paper ids, source references, and sampling mode so failures can be replayed.
3. Missing or broken corpus paths are reported as corpus-health issues instead of being mixed into model-quality failure counts.

### Phase 8: Sampled Single-Paper L2 Regression

**Goal:** Repeatedly run sampled single-paper extraction and evaluation so `L2` quality can be measured on both stable and novel papers.
**Depends on:** Phase `7`
**Requirements:** `L2Q-01`, `L2Q-02`
**Status:** Complete on `2026-04-03`

**Success criteria:**
1. Sampled single-paper runs emit per-paper trace, schema, and evidence-slot quality outputs.
2. Regression summaries distinguish recurring failures, newly introduced regressions, and new random-sample edge cases.
3. The project can identify a concrete short list of `L2` owners from sampled-paper evidence instead of vague quality impressions.

### Phase 9: Bounded Packet Construction From Corpus

**Goal:** Build one bounded topic packet from the larger corpus so `L3/L4` can be tested on structured multi-paper evidence instead of arbitrary random mixes.
**Depends on:** Phase `8`
**Requirements:** `PACK-01`
**Status:** Complete on `2026-04-03`

**Success criteria:**
1. One bounded topic and cutoff are selected from the larger corpus with explicit packet inclusion and exclusion notes.
2. The packet defines `support`, `alternative`, and `held_out` roles clearly enough for downstream replay and review.
3. Packet assembly gaps such as missing traces, weak support density, or role imbalance are recorded explicitly before `L3/L4` runs.

### Phase 10: Multi-Paper L3 And L4 Validation

**Goal:** Compile replay/package/review/export artifacts for the bounded packet and inspect the true multi-paper failure surface.
**Depends on:** Phase `9`
**Requirements:** `PACK-02`, `AGGR-01`
**Status:** Complete on `2026-04-03`

**Success criteria:**
1. The bounded packet produces replay and package-validation artifacts that make `L3` quality gaps inspectable.
2. The same packet produces `L4` prior/review/export artifacts, or explicit blockers that explain why multi-paper aggregation failed.
3. The result can be compared against the existing jamming baseline to separate reusable compiler behavior from topic-specific limitations.

### Phase 11: Iteration Prioritization And Next Cycle Plan

**Goal:** Convert the evidence from sampled single-paper runs and bounded multi-paper runs into the next optimization cycle.
**Depends on:** Phase `10`
**Requirements:** `LOOP-01`, `LOOP-02`

**Success criteria:**
1. A single summary links single-paper `L2` failures with bounded multi-paper `L3/L4` outcomes.
2. The summary prioritizes which owners should be tackled next instead of leaving the iteration open-ended.
3. The team can choose whether the next cycle should emphasize `L2` extraction, packet construction, or `L4` aggregation based on explicit evidence.

## Coverage

| Requirement | Phase |
|-------------|-------|
| `SAMPLE-01` | Phase `7` |
| `SAMPLE-02` | Phase `7` |
| `SAMPLE-03` | Phase `7` |
| `L2Q-01` | Phase `8` |
| `L2Q-02` | Phase `8` |
| `PACK-01` | Phase `9` |
| `PACK-02` | Phase `10` |
| `AGGR-01` | Phase `10` |
| `LOOP-01` | Phase `11` |
| `LOOP-02` | Phase `11` |

**Coverage status:** `10/10` requirements mapped

## Notes

- This roadmap deliberately continues numbering from `v1.0` because `.planning/phases/01-*` through `06-*` are still present.
- `v1.1` does not try to process the full corpus in one pass; it uses the corpus as a large candidate pool for repeated sampling and bounded packet construction.
- The jamming slice remains the reference baseline, but it is no longer the only intended source of optimization evidence.
- Shared-corpus path instability is part of the quality loop because bad corpus hygiene can otherwise masquerade as extraction failure.
- Phase `7` completed on `2026-04-03` with a committed ten-paper fixed regression set, a real seed-`7` random exploration batch of five papers, and `1505` recorded corpus-health failures in `tmp/phase7_corpus_sampling_baseline/`.
- Phase `8` completed on `2026-04-03` with a real `15/15` sampled-paper baseline run, `7` recurring fixed failures, `2` random edge cases, and an owner queue led by `relation_assembly` and `slot_recovery`.
- Phase `9` completed on `2026-04-03` with a committed `2017-2021` computational-mechanics packet, explicit `support / alternative / held_out` mapping, and a runtime-backed audit report that keeps Phase `10` blockers visible.
- Plan `11-01` completed on `2026-04-03` with typed prioritization loaders, a packet-first ranking contract, and manifest-backed Phase `11` bundle outputs over the committed Phase `8` and Phase `10` evidence chain.
- Plan `11-02` completed on `2026-04-03` with an operator-facing Phase `11` CLI, report rendering from summary/inspection payloads, and explicit Phase `10` fallback disclosure in the markdown output.

## Next Up

**Phase 11 Plan 03: Iteration Prioritization And Next Cycle Plan** - Run the real Phase `11` prioritization workflow, commit the generated report, and record the final verification note for downstream planning.

`$gsd-discuss-phase 11`

Also available: `$gsd-plan-phase 11`

---
*Roadmap created: 2026-04-03*
*Last updated: 2026-04-03 after Phase 11 Plan 02 execution and verification*
