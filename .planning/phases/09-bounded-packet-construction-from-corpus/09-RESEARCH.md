# Phase 9: Bounded Packet Construction From Corpus - Research

**Researched:** 2026-04-03
**Domain:** bounded multi-paper packet construction, corpus-grounded packet auditing, and downstream route-role mapping
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Phase 9 must start from the Phase 8 owner queue and the fixed regression slice rather than re-sampling a brand-new topic from the whole corpus.
- Topic expansion should stay filesystem-first and conservative, using nearby corpus papers only when they strengthen one defensible topic boundary and one cutoff year.
- The canonical artifact remains a committed file-based `RoutePacket` plus human-readable selection notes.
- `RoutePacket.included_items` keeps the existing paper-level roles: `core_method`, `resource_or_benchmark`, `limitation_or_critique`, `survey_or_review`, and `alternative_route`.
- Downstream `support / alternative / held_out` grouping must be recorded as a layer on top of the packet, not by rewriting the `RoutePacket` schema.
- Phase 9 must explicitly record assembly gaps such as trace coverage, role imbalance, topic-boundary uncertainty, and weak support density before any Phase 10 replay claim.
- Phase 9 ends at one bounded packet plus explicit downstream mapping and blockers. It does not need a green `L3/L4` replay inside the same phase.

### Agent Discretion
- Exact companion-manifest filename and field names for downstream packet-role mapping
- Exact helper / CLI shape for packet audit and bundle writing
- Exact report layout for the Phase 9 packet audit
- Exact packet size inside the bounded topic as long as the topic boundary and exclusion ledger stay auditable

### Deferred / Out Of Scope
- Rewriting `RoutePacket` or `RouteStatePackage` schemas
- General corpus-wide packet generation automation
- UI or database surfaces for packet curation
- Claiming Phase 10 replay, prior, or export success inside Phase 9
</user_constraints>

<research_summary>
## Summary

Phase 9 should ship one committed bounded packet centered on the data-driven / multiscale computational-mechanics slice already frozen by the Phase 7 fixed regression set and prioritized by the Phase 8 owner queue. The strongest first boundary is a **2017-2021 topic slice** anchored by the recurring Phase 8 failure exemplars:

- `1000` (`2017`) - *Data-based derivation of material response*
- `1001` (`2019`) - *Measuring stress field without constitutive equation*
- `1005` (`2020`) - *Data-Driven multiscale modeling in mechanics*
- `1017` (`2019`) - *Self consistent clustering analysis for multiscale modeling at finite strains*
- `1023` (`2021`) - *Adaptive selection of reference stiffness in virtual clustering analysis*

That boundary is better than a looser "computational mechanics + any ML paper" packet for three reasons:

1. It directly follows the highest-priority Phase 8 buckets (`relation_assembly`, then `slot_recovery`) instead of mixing in unrelated random edge cases.
2. It stays inside the same family of constitutive / multiscale / clustering phrasing that the fixed regression notes already describe as a coherent slice.
3. It gives Phase 9 a clean cutoff discipline: later same-neighborhood papers like `1004` (`2023`) can become explicit exclusion examples rather than silently stretching the packet boundary.

The recommended execution shape is:

1. Build a **typed packet-audit helper** that validates one committed `RoutePacket` plus a companion role-mapping manifest.
2. Commit the first **Phase 9 packet artifact set**:
   - `phase9-route-packet.json`
   - `phase9-selection-notes.md`
   - `phase9-assembly-manifest.json`
   - `phase9-bounded-packet-audit.md`
3. Record downstream `support / alternative / held_out` groups as a layer over packet members and trace refs, with explicit unresolved gaps.

**Primary recommendation:** implement Phase 9 in two execution slices. First add the bounded-packet audit contract and CLI. Then commit the real Phase 9 packet and companion mapping using that contract, with tests and a human-readable audit report.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Asset | Status | Purpose | Why Standard Here |
|-------|--------|---------|-------------------|
| `docs/replay/corpus_sampling/phase7-fixed-regression-set.json` | Existing | Frozen ten-paper regression slice | Canonical starting pool for the first bounded packet |
| `docs/replay/corpus_sampling/phase7-fixed-regression-notes.md` | Existing | Topic-shape and path-shape rationale for the fixed set | Best evidence that the first packet should stay inside one related slice |
| `docs/replay/reports/phase8-sampled-single-paper-l2-regression-baseline.md` | Existing | Owner queue and exemplar paper ids | Tells Phase 9 which papers most clearly expose the current multi-paper boundary problem |
| `backend/app/research_logic/models.py` | Existing | Canonical `RoutePacket` schema and quality rules | Phase 9 must keep this as the packet contract |
| `backend/app/research_logic/route_state_package.py` | Existing | Current `primary / support / alternative / held_out` semantics and validation | Best source for downstream role expectations without rewriting packet schema |
| `backend/tests/test_phase1_route_packet_manifest.py` | Existing | Regression pattern for committed packet-manifest contract tests | Reusable test style for a real committed Phase 9 packet |

### Supporting
| Asset | Status | Purpose | When To Use |
|-------|--------|---------|-------------|
| `tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/paper_artifacts/fixed_regression/` | Existing | Real trace and paper-level artifacts for the Phase 7 fixed set | Use as Phase 9 trace refs and gap-audit evidence |
| `docs/replay/reports/route-state-package-pilot-2026-04-02.md` | Existing | Proven package-level meaning of `support / alternative / held_out` | Reuse its grouping semantics for the Phase 9 handoff |
| `docs/superpowers/specs/2026-04-01-logickg-route-packet-schema.md` | Existing | Packet scope, cutoff, and role-coverage expectations | Use for selection notes and audit wording |
| `backend/app/research_logic/replay_io.py` | Existing | Bundle / summary writer pattern | Good home for a Phase 9 audit bundle writer if one is needed |
| `backend/tests/test_route_state_package.py` | Existing | Validation patterns for grouped role semantics | Reference when designing packet-role mapping checks |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| fixed-set seeded bounded packet | re-sample a fresh corpus neighborhood | Would discard the Phase 8 owner evidence and make the first packet harder to interpret |
| companion role-mapping manifest | rewrite `RoutePacket` to add `support / alternative / held_out` directly | Higher schema churn for little gain and conflicts with the locked decision |
| explicit audit helper and report | hand-curated docs only | Faster once, but weaker for repeatability and harder to regression-test |
| `2021` cutoff with later-paper exclusions | `2023` cutoff including every nearby paper | Easier to stuff more papers into the packet, but weakens historical discipline and increases boundary drift |
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Pattern 1: Owner-Queue-Seeded Topic Boundary
**What:** Start from the Phase 8 owner-bucket exemplars and expand only within the same nearby slice.  
**Why recommended:** It keeps the first packet tied to the exact papers that already exposed the current `L3/L4` preparation problem.

### Pattern 2: Cutoff-First Packet With Exclusion Ledger
**What:** Freeze one cutoff year first, then explicitly document which nearby papers are excluded because they are post-cutoff, out-of-slice, or too weakly connected.  
**Why recommended:** Phase 9 needs an auditable boundary, not an ever-expanding "interesting related papers" list.

### Pattern 3: `RoutePacket` Plus Companion Assembly Manifest
**What:** Keep the packet itself in the canonical schema, and layer `support / alternative / held_out` mapping in a second manifest that references packet members and trace artifacts.  
**Why recommended:** It satisfies both the Phase 9 constraint to preserve packet roles and the Phase 10 need for downstream replay grouping.

### Pattern 4: Filesystem-First Trace Grounding
**What:** Prefer corpus-relative refs and existing Phase 8 trace artifact paths over graph metadata or machine-local UNC paths.  
**Why recommended:** Phase 7 and Phase 8 already proved the fixed slice is graph-light but filesystem-real.

### Pattern 5: Gap-First Audit Before Replay
**What:** Validate trace coverage, role coverage, support density, alternative distinctness, held-out separation, and exclusion completeness before any Phase 10 run.  
**Why recommended:** The phase goal is honest bounded-packet assembly, not premature replay optimism.

### Pattern 6: Conservative Same-Slice Expansion
**What:** Expand from the five owner-queue anchors into nearby fixed-set candidates such as `1002`, `1007`, `1010`, and `1012` only if they strengthen the same story rather than broadening the topic.  
**Why recommended:** The fixed set already contains enough same-slice variation to reach role coverage without jumping to random cross-domain papers like `1107`.
</architecture_patterns>

<recommended_boundary>
## Recommended Boundary

### Topic Scope Candidate

`data-driven constitutive and multiscale computational mechanics`

This label is narrow enough to explain why the packet is about constitutive response, multiscale modeling, clustering analysis, and nearby route alternatives, while still broad enough to include the fixed-set papers already selected for this slice.

### Recommended Cutoff

`2021`

Why this cutoff is the best first choice:

- It includes the five strongest owner-queue exemplars (`1000`, `1001`, `1005`, `1017`, `1023`).
- It supports same-slice expansion candidates from the fixed set such as `1002`, `1007`, `1010`, and `1012`.
- It creates a clean, explicit post-cutoff exclusion example for `1004` (`2023`) instead of blurring the packet boundary.

### Seed And Expansion Guidance

| Paper ID | Year | Role In Boundary Design |
|----------|------|-------------------------|
| `1000` | `2017` | anchor seed for data-driven constitutive response |
| `1001` | `2019` | anchor seed for no-constitutive-equation framing |
| `1005` | `2020` | anchor seed for multiscale mechanics framing |
| `1017` | `2019` | anchor seed for clustering-based route family |
| `1023` | `2021` | late-cutoff anchor and likely held-out candidate |
| `1002` | `2020` | same-slice expansion candidate with noisy-database / constrained framing |
| `1007` | `2018` | same-slice expansion candidate for alternative route coverage |
| `1010` | `2018` | same-slice expansion candidate for alternative-route breadth |
| `1012` | `2020` | same-slice expansion candidate for computational-mechanics / deep-learning bridge |
| `1004` | `2023` | explicit post-cutoff exclusion candidate |
| `1107` | unknown / random | explicit out-of-slice exclusion candidate from Phase 8 random edge cases |

### Downstream Role-Mapping Direction

The Phase 9 handoff should not guess final `RouteState` outcomes, but it should pre-declare one plausible mapping surface:

- `support`: at least two packet members representing the main data-driven / constitutive route family
- `alternative`: at least one distinct route-family paper from the same packet boundary
- `held_out`: at least one packet member reserved for later consistency checks and not reused in `support`

Those roles should be stored in a companion manifest with explicit reasons and trace refs, not inferred later from memory.
</recommended_boundary>

## Validation Architecture

Phase 9 should create its own validation boundary rather than waiting for Phase 10 replay.

The execution plans should cover four validation surfaces:

1. **Packet contract validation**  
   The committed `phase9-route-packet.json` must load through the canonical `RoutePacket` model and preserve portable corpus-relative references only.

2. **Assembly-manifest validation**  
   The companion manifest must reference only packet members, require non-empty reasons, keep `support` and `held_out` disjoint, and flag missing role groups.

3. **Packet audit validation**  
   The audit bundle must expose concrete counts and flags for:
   - trace coverage gaps
   - role coverage gaps
   - weak support density
   - indistinct alternatives
   - held-out leakage
   - exclusion-note completeness

4. **Manual boundary review**  
   A reviewer must still read the committed notes and confirm that the topic boundary is defensible, the excluded papers are understandable, and the phase does not overclaim replay readiness.

This is sufficient Nyquist coverage for Phase 9 because the phase claim is about bounded-packet assembly honesty, not about already-green multi-paper replay.

## Implementation Notes

- The first packet should prefer committed docs under `docs/replay/pilot_packets/` plus runtime audit JSON under `tmp/phase9_bounded_packet_audit/`.
- The packet and companion manifest should use only repo-portable identifiers and relative refs. No UNC share paths should be committed.
- Any trace refs should point to existing Phase 8 paper-artifact outputs first; missing refs must become explicit audit flags rather than silent omissions.
- Random-only edge-case paper `1107` should stay outside the first packet and appear only as an out-of-slice exclusion note if needed.
- Phase 10 should be able to consume the Phase 9 output without inventing a new packet boundary or new role semantics.

## Recommended Plan Split

### Plan 01
Create the typed bounded-packet audit contract, bundle writer, and CLI around the existing `RoutePacket` schema.

### Plan 02
Commit the real Phase 9 packet, selection notes, companion mapping manifest, audit report, and regression tests.
