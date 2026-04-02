# Requirements: LogicKG

**Defined:** 2026-04-01
**Core Value:** 持续产出可审计、可回放、可训练的科研判断样本，而不只是产出“看起来像懂论文”的摘要或图谱。

## Validated Baseline

### Existing Platform

- [x] **BASE-01**: 团队可以把论文与教材 Markdown 导入为可追溯的图谱资产，并通过 Neo4j 持久化。
- [x] **BASE-02**: 系统可以生成带 evidence anchors、citation acts、quality gates 的 `PaperLogicTrace` / `ResearchMove` 风格单篇论文结构化结果。
- [x] **BASE-03**: 使用者可以通过 frontend workbench 完成图谱浏览、论文详情查看、Ask 检索问答、导入与运维配置操作。
- [x] **BASE-04**: 系统具备 similarity / community / task queue / rebuild 等平台能力，可作为科研推理层的基础设施。

## v1 Requirements

当前 v1 不是“完整科研自动发现系统”，而是“训练可用的 scientific reasoning compiler pilot”。

### Route Packet And Replay Input

- [x] **PKT-01**: 团队可以为一个 `topic_scope + cutoff_year` 构建并冻结一个 `RoutePacket`，其中包含 included / excluded items、角色覆盖、trace refs、泄漏策略与 `L1` snapshot refs。
- [x] **PKT-02**: `RoutePacket` 在进入 replay compilation 前会暴露 role coverage、trace completeness、topic boundary 与 leakage 风险等质量信号。

### Historical Environment Bridge

- [x] **L1-01**: 系统可以为至少一个试点领域生成最小可用的历史科研环境快照，覆盖 resource / benchmark / toolchain / protocol 的时间切片。
- [x] **L1-02**: `RoutePacket` 与 `RouteState` 编译流程可以消费上述 `L1` 快照，且不会混入 cutoff 之后的信息。

### L3 Replay Compilation

- [x] **L3-01**: 系统可以基于多篇论文和受控 packet 编译 `RouteState`，而不是从单篇论文 seed 直接改写。
- [x] **L3-02**: `RouteState` 至少要稳定表达 dominant methods、active benchmarks、measurement protocols、known bottlenecks、enabling conditions、alternative routes、evidence bundle 和 uncertainty points。
- [x] **L3-03**: 系统可以从 `RouteState` 稳定生成 `WhyNowCase` 与 `RouteComparisonCase`，并保留显式 evidence chain 与 uncertainty。

### L2 Surgical Quality Loop

- [x] **L2-01**: replay evaluation 能定位是哪些 `L2` 缺口导致 `RouteState` / `WhyNow` / `Prior` 编译失败，并形成可执行的修补任务。
- [x] **L2-02**: 每个进入 replay 的 `PaperLogicTrace` 都必须保留 methods、metrics、resources、conditions、limitations、future-work 等关键槽位及其 provenance。

### L4 Decision Layer

- [x] **L4-01**: 系统可以从多个 `RouteState` 归纳出 `DecisionPriorCard` / `AntiPatternCard`，并显式记录 support、counterexample、held-out consistency 与 review status。
- [ ] **L4-02**: 系统可以组装 `DecisionEpisode`，把 `L1/L2/L3/L4` 对象放进同一个可训练、可评估、可审计的样本中，同时把 hindsight 仅保留在 label / eval 侧。

### Evaluation

- [x] **EVAL-01**: 团队可以在至少一个真实的 route packet 上完成历史 replay，并得到结构化输出而不只是自然语言总结。
- [x] **EVAL-02**: 评估结果必须能按阶段给出质量 tier、阻塞原因、失败样本与下一轮改进建议。

## v2 Requirements

### Question Discovery And Hypothesis Work

- **GEN-01**: 在 `DecisionEpisode` 稳定后，系统可以生成受约束的科研问题候选，并保留 novelty / feasibility critic 所需证据。
- **GEN-02**: 在 priors 与 anti-patterns 足够稳定后，系统可以探索受约束的假说候选生成，而不是开放式脑暴。

### Productization

- **OPS-01**: 平台提供 route packet 管理、replay 运行、失败诊断与人工审核的可视化运维入口。
- **CORP-01**: 系统可以在多个 topic / cutoff packet 上进行批量 replay，对不同领域的编译质量进行横向比较。

## Out of Scope

| Feature | Reason |
|---------|--------|
| 现在就做开放式全自动科研代理 | 当前基础还不足以支撑可审计的高层判断 |
| 现在就重构全部 `L2` 主结构 | 当前更需要先用 replay 暴露真实缺口 |
| 先扩到全语料再定义 packet / replay 合同 | 规模化会在错误合同上累积噪声和成本 |
| 把 `L3/L4` 当作单篇论文抽象层 | 这违背多论文聚合与时间切片原则 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PKT-01 | Phase 1 | Completed |
| PKT-02 | Phase 1 | Completed |
| EVAL-01 | Phase 1 | Completed |
| L1-01 | Phase 2 | Completed |
| L1-02 | Phase 2 | Completed |
| L3-01 | Phase 3 | Validated |
| L3-02 | Phase 3 | Validated |
| L3-03 | Phase 3 | Validated |
| L2-01 | Phase 4 | Complete |
| L2-02 | Phase 4 | Complete |
| EVAL-02 | Phase 4 | Complete |
| L4-01 | Phase 5 | Complete |
| L4-02 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 13 total
- Mapped to phases: 13
- Unmapped: 0 ✅

---
*Requirements defined: 2026-04-01*
*Last updated: 2026-04-02 after Phase 5 verified multi-route prior induction and candidate review outputs*
