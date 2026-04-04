# Requirements: LogicKG

**Defined:** 2026-04-03
**Core Value:** Produce auditable, replayable, trainable scientific-reasoning artifacts rather than paper-like summaries or one-off knowledge graphs.

## v1.2 Requirements

Requirements for milestone `v1.2`. This milestone uses repeated in-milestone cycles: generate outputs, manually review output quality, optimize the pipeline, and rerun until quality is stable.

### Iteration Loop Capability (Direct From Existing Data)

- [ ] **LOOPR-01**: Operators can run a full generation cycle (`packet -> replay -> prior/review -> export`) directly from existing Phase 10/11 artifacts and commands.
- [ ] **LOOPR-02**: Each cycle records comparable evidence outputs and review notes so cycle-to-cycle quality movement is auditable.
- [ ] **LOOPR-03**: The workflow supports rapid fix -> rerun loops on the same baseline data without introducing a separate foundation refactor first.

### Manual Output Quality Review

- [ ] **REVIEW-01**: Each cycle includes explicit manual review notes on real reasoning outputs (not only metric deltas).
- [ ] **REVIEW-02**: Review notes must identify concrete output-level defects and map each defect to a pipeline stage to optimize next.
- [ ] **REVIEW-03**: Cycle recommendations (`packet_construction`, `l4_aggregation`, `l2_extraction`) must be justified by reviewed output evidence.

### Stability Target

- [ ] **STAB-01**: The milestone records consecutive successful cycles where manual review judges output quality as high.
- [ ] **STAB-02**: Stability is declared only when high-quality output is reproducible across consecutive cycles, not one-off.
- [ ] **STAB-03**: The final verification artifact summarizes why quality is considered genuinely stable and what residual risks remain.

## v2 Requirements

Deferred until v1.2 completes stable iterative quality.

### Scale

- **SCALE-01**: Team can generate and compare multiple bounded topic packets from the large corpus in one batch without hand-assembling every packet.

### Productization

- **OPS-01**: Operator can manage packet, replay, review, and export workflows from dedicated product surfaces instead of manual file-system orchestration.

### Question Discovery

- **GEN-01**: System can generate constrained scientific question candidates from stable reviewed `DecisionEpisode` artifacts with audit-visible evidence.

## Out of Scope

Explicitly excluded from `v1.2`.

| Feature | Reason |
|---------|--------|
| Full-corpus end-to-end processing in a single cycle | v1.2 focuses on bounded slices and fast iterative learning loops. |
| UI productization of every operator workflow | v1.2 prioritizes output quality stabilization and iteration speed. |
| Treating metric-only improvement as success | v1.2 requires manual review of real generated reasoning outputs. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| `LOOPR-01` | Phase `12` | Complete |
| `LOOPR-02` | Phase `12` | Complete |
| `LOOPR-03` | Phase `13` | Pending |
| `REVIEW-01` | Phase `13` | Pending |
| `REVIEW-02` | Phase `14` | Pending |
| `REVIEW-03` | Phase `14` | Pending |
| `STAB-01` | Phase `15` | Pending |
| `STAB-02` | Phase `16` | Pending |
| `STAB-03` | Phase `16` | Pending |

**Coverage:**
- v1.2 requirements: `9` total
- Mapped to phases: `9`
- Unmapped: `0`

---
*Requirements defined: 2026-04-03*
*Last updated: 2026-04-04 after completing Phase 12*
