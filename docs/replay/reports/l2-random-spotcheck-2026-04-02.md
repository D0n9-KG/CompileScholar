# L2 Random Spotcheck - 2026-04-02

## Purpose

This note records a small random spotcheck against the user-provided local corpus so Phase 1 conclusions do not rely only on the jamming replay packet.

The committed repo does **not** store the raw corpus path. Local runtime artifacts remain under `tmp/`.

## Spotcheck Summary

- sampled papers with completed traces: `5`
- `gate_passed`: `5/5`
- quality tiers:
  - `green`: `1`
  - `yellow`: `4`

## Sample Results

| Paper ID | Title | Quality | Move Count | Comparators | Resource Mentions | Flags |
|----------|-------|---------|------------|-------------|-------------------|-------|
| `135` | `Temporal correlation of force and position in granular materials` | `yellow` | `22` | `0` | `2` | `missing_expected_slots` |
| `436` | `A Simple Representation of Three-Dimensional Molecular Structure` | `green` | `44` | `19` | `22` | - |
| `1221` | `Volume of fluid methods for immiscible-fluid and free-surface flows` | `yellow` | `33` | `6` | `10` | - |
| `1166` | `Recent Advance in Chaotic Mixing in a Mixing Equipment` | `yellow` | `26` | `0` | `15` | `weak_relation_stitching`, `missing_expected_slots` |
| `1055` | `Method of Mixture Ratio Control of Bipropellant Blowdown Propulsion System` | `yellow` | `19` | `1` | `11` | `sparse_expected_slots` |

## What The Spotcheck Says

Positive signal:

- the current `L2` layer is not limited to the pilot packet topic
- across physics, chemistry, fluid methods, and a Chinese engineering paper, the pipeline can usually build an auditable single-paper trace
- critical roles (`problem`, `method`, `result`) were recovered consistently enough to pass the hot-path gate in every completed sample

Remaining weakness:

- comparator coverage is still fragile
- expected downstream slots are still uneven across paper styles
- relation stitching is not yet robust enough to assume stable `L3` aggregation

The strongest sample (`436`) suggests the current design can reach `green` when a paper is method-heavy, benchmark-rich, and explicit about resources and comparisons.

The weaker samples show the real boundary:

- `L2` is already useful as a single-paper evidence layer
- `L2` is not yet reliable enough to be the only thing protecting `L3/L4` from thin comparison structure or missing readiness signals

## Local Artifact Pointers

Spotcheck artifacts produced during this session:

- `tmp/l2_random_sample_eval_20260402_small/progress.json`
- `tmp/l2_single_paper_eval_1166/summary.json`
- `tmp/l2_single_paper_eval_1055/summary.json`
- `tmp/l2_spotcheck_aggregate.json`

These local artifacts should be treated as pilot evidence, not committed dataset assets.
