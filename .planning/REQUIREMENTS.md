# Requirements: LogicKG

**Defined:** 2026-04-02
**Core Value:** Produce auditable, replayable, trainable scientific-reasoning artifacts rather than paper-like summaries or one-off knowledge graphs.

## v1.1 Requirements

Requirements for milestone `v1.1`. These define what must be true before the next major direction is chosen.

### Canonical Assets

- [ ] **CANON-01**: Maintainer can locate committed canonical manifests or READMEs for the bounded jamming packet, subset packets, replay package, review bundle, and export bundle boundaries without depending on `tmp/` directory names alone.
- [ ] **CANON-02**: Maintainer can rebuild the bounded jamming replay, package, review, and export flow from committed manifests plus machine-local trace inputs without hardcoding local paths in the repository.
- [ ] **CANON-03**: Audit documentation clearly distinguishes committed canonical artifacts from local-only runtime outputs and records provenance between them.

### Cross-Topic Validation

- [ ] **XVAL-01**: Maintainer can define one second bounded topic and cutoff with packet selection notes, local trace input expectations, and explicit inclusion or exclusion rationale.
- [ ] **XVAL-02**: The second-topic flow can compile a route-state package and replay bundle with inspectable validation and quality-summary artifacts.
- [ ] **XVAL-03**: The second-topic flow can produce prior-review and audited `DecisionEpisode` export artifacts with explicit accepted ids and leakage-safe references.
- [ ] **XVAL-04**: The project can compare jamming and second-topic results in one report that separates reusable compiler behavior from topic-specific blockers.

### Next-Milestone Decision

- [ ] **DECIDE-01**: Team can inspect a milestone summary of canonical asset maturity, cross-topic validation results, and remaining technical debt in one place.
- [ ] **DECIDE-02**: Team can choose the following milestone direction between ops/productization work and question-discovery work using explicit evidence from `v1.1`.

## v2 Requirements

Deferred to a later milestone after `v1.1` settles the direction choice.

### Productization

- **OPS-01**: Operator can manage route-packet, replay, review, and export runs from dedicated workflow surfaces instead of manual file-system orchestration.
- **OPS-02**: Team can run and compare bounded replay, review, and export workflows across multiple topics and cutoffs without hand-assembling every bundle.

### Question Discovery

- **GEN-01**: System can generate constrained scientific question candidates from stable reviewed `DecisionEpisode` artifacts with audit-visible evidence.
- **GEN-02**: System can explore bounded hypothesis generation only after priors and anti-patterns are stable enough to keep the search auditable.

### Quality Hardening

- **QUAL-01**: Support density and trace completeness are high enough that reviewed prior acceptance is no longer blocked by intentionally sparse evidence slices.

## Out of Scope

Explicitly excluded from `v1.1`.

| Feature | Reason |
|---------|--------|
| Open-ended question discovery or hypothesis generation | Wait until one second topic proves the reviewed `DecisionEpisode` flow generalizes. |
| Full UI productization of packet, replay, review, and export operations | `v1.1` is focused on canonicalization and validation rather than complete operator polish. |
| Full-corpus automatic packet discovery | Scaling before canonical bounded assets are stable would amplify noise and provenance ambiguity. |
| Wholesale `L2` redesign | Cross-topic evidence should identify precise owners before any large rewrite is justified. |

## Traceability

Roadmap mapping is still pending.

| Requirement | Phase | Status |
|-------------|-------|--------|
| `CANON-01` | Pending roadmap | Pending |
| `CANON-02` | Pending roadmap | Pending |
| `CANON-03` | Pending roadmap | Pending |
| `XVAL-01` | Pending roadmap | Pending |
| `XVAL-02` | Pending roadmap | Pending |
| `XVAL-03` | Pending roadmap | Pending |
| `XVAL-04` | Pending roadmap | Pending |
| `DECIDE-01` | Pending roadmap | Pending |
| `DECIDE-02` | Pending roadmap | Pending |

**Coverage:**
- v1.1 requirements: `9` total
- Mapped to phases: `0`
- Unmapped: `9`

---
*Requirements defined: 2026-04-02*
*Last updated: 2026-04-02 after milestone requirement definition*
