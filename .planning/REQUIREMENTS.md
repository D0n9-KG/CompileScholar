# Requirements: LogicKG

**Defined:** 2026-04-03
**Core Value:** Produce auditable, replayable, trainable scientific-reasoning artifacts rather than paper-like summaries or one-off knowledge graphs.

## v1.2 Requirements

Requirements for milestone `v1.2`. This milestone optimizes for both iteration speed and real outcome quality on the full reasoning production line.

### Iteration Throughput

- [ ] **ITER-01**: Operators can run one full packet-to-report optimization cycle with a single command entrypoint and reproducible inputs.
- [ ] **ITER-02**: Each cycle records timing, bottleneck stage, and changed artifacts so cycle-to-cycle speed can be compared.
- [ ] **ITER-03**: The team can run at least one automated rerun path that avoids manual file stitching between packet, replay, and verification stages.

### Packet Quality Recovery

- [ ] **PACK-03**: Packet-construction fixes can explicitly resolve current package blockers (`support_cluster_too_small`, `alternative_scope_not_distinct`, `yellow_route_state_present`) or produce explicit unresolved blockers.
- [ ] **PACK-04**: Packet role balance (`support` / `alternative` / `held_out`) remains auditable after each fix cycle with machine-readable evidence.

### L4 And Prior Surface Recovery

- [ ] **AGGR-02**: After packet fixes, replay/prior surfaces are rerun and compared against the committed Phase 10/11 baseline to determine whether `L4` quality recovered.
- [ ] **AGGR-03**: Prior-review output must explicitly report candidate availability, accepted prior ids, and anti-pattern carryover deltas.

### Real-Outcome Quality Gate

- [ ] **QUAL-01**: Promotion decisions require both rule-threshold checks and explicit review of real reasoning artifacts; metric-only gains cannot pass.
- [ ] **QUAL-02**: Each cycle ends with a ranked next-step recommendation (`packet_construction`, `l4_aggregation`, `l2_extraction`) grounded in generated evidence artifacts.

## v2 Requirements

Deferred until v1.2 reaches stable cycle quality and throughput.

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
| Full-corpus end-to-end processing in a single cycle | v1.2 focuses on bounded slices with fast feedback loops and quality control. |
| UI productization of every operator workflow | v1.2 prioritizes pipeline correctness, throughput, and evidence quality. |
| Open-ended hypothesis generation as primary goal | v1.2 must first stabilize reasoning quality on concrete bounded runs. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| `ITER-01` | Phase `12` | Pending |
| `ITER-02` | Phase `12` | Pending |
| `ITER-03` | Phase `13` | Pending |
| `PACK-03` | Phase `13` | Pending |
| `PACK-04` | Phase `13` | Pending |
| `AGGR-02` | Phase `14` | Pending |
| `AGGR-03` | Phase `14` | Pending |
| `QUAL-01` | Phase `15` | Pending |
| `QUAL-02` | Phase `16` | Pending |

**Coverage:**
- v1.2 requirements: `9` total
- Mapped to phases: `9`
- Unmapped: `0`

---
*Requirements defined: 2026-04-03*
*Last updated: 2026-04-03 for milestone v1.2 initialization*
