# Requirements: LogicKG

**Defined:** 2026-04-03
**Core Value:** Produce auditable, replayable, trainable scientific-reasoning artifacts rather than paper-like summaries or one-off knowledge graphs.

## v1.1 Requirements

Requirements for milestone `v1.1`. These define what must be true before the next optimization cycle can be chosen confidently.

### Corpus Sampling

- [x] **SAMPLE-01**: Operator can define one fixed regression paper set and one random exploration paper set from the shared corpus for each iteration cycle.
- [x] **SAMPLE-02**: Each sampling run records selected paper ids, source paths, and sampling mode so the exact batch can be reproduced.
- [x] **SAMPLE-03**: Sampling reports missing or broken corpus paths separately from model-quality failures.

### Single-Paper Extraction Iteration

- [ ] **L2Q-01**: Operator can run sampled single-paper extraction and evaluation and collect per-paper trace, schema, and evidence-slot quality results.
- [ ] **L2Q-02**: Team can compare sampled single-paper results across iterations to identify recurring failures, regressions, and newly surfaced edge cases.

### Bounded Multi-Paper Packet Validation

- [ ] **PACK-01**: Operator can assemble one bounded topic packet from the larger corpus with explicit `support`, `alternative`, and `held_out` role assignments plus exclusion notes.
- [ ] **PACK-02**: The bounded packet can compile replay and package-validation artifacts that make `L3` quality gaps inspectable.
- [ ] **AGGR-01**: The same bounded packet can produce `L4` prior/review/export artifacts or explicit blockers explaining why multi-paper aggregation failed.

### Iteration Prioritization

- [ ] **LOOP-01**: Team can inspect one summary that connects sampled single-paper failures with bounded multi-paper packet outcomes.
- [ ] **LOOP-02**: Team can choose the next optimization cycle based on explicit evidence about whether the highest-leverage work is in `L2` extraction, packet construction, or `L4` aggregation.

## v2 Requirements

Deferred until the corpus-driven iteration loop is stable.

### Scale

- **SCALE-01**: Team can generate and compare multiple bounded topic packets from the large corpus in one batch without hand-assembling every packet.

### Productization

- **OPS-01**: Operator can manage packet, replay, review, and export workflows from dedicated product surfaces instead of manual file-system orchestration.

### Question Discovery

- **GEN-01**: System can generate constrained scientific question candidates from stable reviewed `DecisionEpisode` artifacts with audit-visible evidence.

## Out of Scope

Explicitly excluded from `v1.1`.

| Feature | Reason |
|---------|--------|
| Run every candidate paper end to end in one milestone sweep | `v1.1` is about iterative sampling and feedback loops, not full-corpus throughput. |
| Let `L3/L4` consume arbitrary random paper mixes | Multi-paper aggregation needs bounded topics and explicit roles to stay meaningful. |
| Full UI productization of packet, replay, review, and export operations | The current milestone is focused on compiler quality, not operator polish. |
| Open-ended question discovery or hypothesis generation | The compiler loop should stabilize first. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| `SAMPLE-01` | Phase `7` | Complete |
| `SAMPLE-02` | Phase `7` | Complete |
| `SAMPLE-03` | Phase `7` | Complete |
| `L2Q-01` | Phase `8` | Pending |
| `L2Q-02` | Phase `8` | Pending |
| `PACK-01` | Phase `9` | Pending |
| `PACK-02` | Phase `10` | Pending |
| `AGGR-01` | Phase `10` | Pending |
| `LOOP-01` | Phase `11` | Pending |
| `LOOP-02` | Phase `11` | Pending |

**Coverage:**
- v1.1 requirements: `10` total
- Mapped to phases: `10`
- Unmapped: `0`

---
*Requirements defined: 2026-04-03*
*Last updated: 2026-04-03 after Phase 7 execution and verification*
