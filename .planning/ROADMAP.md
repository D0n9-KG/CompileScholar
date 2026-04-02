# Milestone v1.1: Canonical Asset Promotion And Cross-Topic Validation

**Status:** ACTIVE 2026-04-02
**Phases:** 7-11
**Requirements:** 9 mapped
**Numbering Mode:** Continue from `v1.0`

## Overview

`v1.1` turns the bounded jamming pilot from a one-off success story into a reusable baseline.

The milestone first promotes the jamming packet/package/review/export flow into canonical surfaces with clear provenance rules, then proves the same Phase 03 -> 06 flow on one second bounded topic, and finally closes with an evidence-backed recommendation for the next major roadmap direction.

## Phase Summary

| Phase | Name | Goal | Requirements | Success Criteria |
|-------|------|------|--------------|------------------|
| 7 | Canonical Pilot Asset Promotion | Promote the bounded jamming Phase 03 -> 06 artifact surfaces into canonical committed or reproducible locations with explicit provenance boundaries. | `CANON-01`, `CANON-03` | 3 |
| 8 | Reproducible Replay Operations | Make the canonical jamming flow rebuildable from committed manifests plus machine-local trace inputs. | `CANON-02` | 3 |
| 9 | Second-Topic Packet And Package Validation | Choose one second bounded topic and reach a replay-ready package baseline outside the jamming slice. | `XVAL-01`, `XVAL-02` | 3 |
| 10 | Cross-Topic Review And Export Comparison | Run the second topic through review/export and compare it directly with the jamming baseline. | `XVAL-03`, `XVAL-04` | 3 |
| 11 | Next-Milestone Decision Package | Turn `v1.1` evidence into a decision-ready recommendation for the following milestone. | `DECIDE-01`, `DECIDE-02` | 3 |

## Phase Details

### Phase 7: Canonical Pilot Asset Promotion

**Goal:** Promote the bounded jamming packet, subset packet, package, review, and export surfaces into canonical repo-readable manifests and docs while keeping machine-local runtime data out of the repository.
**Depends on:** `v1.0` archived baseline
**Requirements:** `CANON-01`, `CANON-03`

**Success criteria:**
1. Canonical locations exist for the bounded jamming pilot surfaces beyond the already committed top-level packet docs, and each artifact family states whether it is committed or local-only.
2. Every canonical manifest or README points to the correct runtime bundle provenance without hardcoding machine-specific trace paths.
3. Project docs no longer rely on `tmp/` directory memory alone to explain where core Phase 03 -> 06 pilot artifacts come from.

### Phase 8: Reproducible Replay Operations

**Goal:** Make the canonical jamming flow rebuildable from committed manifests plus local trace inputs so another maintainer can rerun the same bounded flow.
**Depends on:** Phase `7`
**Requirements:** `CANON-02`

**Success criteria:**
1. A maintainer can follow a documented command path to rebuild the bounded jamming packet/package/review/export outputs locally from canonical inputs and local trace roots.
2. The rerun path emits stable manifest or summary outputs for each stage without embedding machine-local paths in committed artifacts.
3. Local prerequisites and likely failure modes are documented well enough to distinguish environment gaps from compiler regressions.

### Phase 9: Second-Topic Packet And Package Validation

**Goal:** Choose one second bounded topic and prove the compiler can reach a package/replay-ready baseline outside the original jamming slice.
**Depends on:** Phase `8`
**Requirements:** `XVAL-01`, `XVAL-02`

**Success criteria:**
1. One second topic and cutoff are chosen with packet selection notes, local trace expectations, and explicit inclusion/exclusion rationale.
2. The second-topic route-state package and replay bundle compile with validation and inspection artifacts that can be reviewed alongside the jamming baseline.
3. Missing trace coverage, support-density issues, or package blockers are recorded explicitly instead of hidden behind fallback-only success claims.

### Phase 10: Cross-Topic Review And Export Comparison

**Goal:** Run the second topic through prior review and audited export, then compare the result directly with the jamming baseline.
**Depends on:** Phase `9`
**Requirements:** `XVAL-03`, `XVAL-04`

**Success criteria:**
1. The second-topic flow yields prior-review and audited `DecisionEpisode` export artifacts with explicit accepted ids and leakage-safe references.
2. A comparison report summarizes what carried over cleanly from the jamming baseline and what remained topic-specific.
3. The comparison distinguishes compiler-contract stability from upstream corpus or trace limitations.

### Phase 11: Next-Milestone Decision Package

**Goal:** Turn `v1.1` findings into a decision-ready recommendation for the following milestone.
**Depends on:** Phase `10`
**Requirements:** `DECIDE-01`, `DECIDE-02`

**Success criteria:**
1. A single milestone summary captures canonical asset maturity, cross-topic results, unresolved debt, and readiness limits.
2. The summary includes an explicit recommendation for either ops/productization or question-discovery as the next milestone focus.
3. The recommendation is backed by concrete evidence links to `v1.1` artifacts rather than abstract preference.

## Coverage

| Requirement | Phase |
|-------------|-------|
| `CANON-01` | Phase `7` |
| `CANON-02` | Phase `8` |
| `CANON-03` | Phase `7` |
| `XVAL-01` | Phase `9` |
| `XVAL-02` | Phase `9` |
| `XVAL-03` | Phase `10` |
| `XVAL-04` | Phase `10` |
| `DECIDE-01` | Phase `11` |
| `DECIDE-02` | Phase `11` |

**Coverage status:** `9/9` requirements mapped

## Notes

- This roadmap deliberately continues numbering from `v1.0` because `.planning/phases/01-*` through `06-*` are still present.
- Optional external research was skipped for milestone setup because the next work is project-specific follow-through on already-known artifact and validation gaps.
- `v1.1` is successful only if the second-topic evidence actually clarifies the next direction. Finishing the asset promotion work without that decision package is not enough.

## Next Up

**Phase 7: Canonical Pilot Asset Promotion** - Promote the bounded jamming asset surfaces into canonical, provenance-safe planning targets.

`$gsd-discuss-phase 7`

Also available: `$gsd-plan-phase 7`

---
*Roadmap created: 2026-04-02*
*Last updated: 2026-04-02 after milestone initialization*
