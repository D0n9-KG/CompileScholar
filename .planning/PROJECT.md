# LogicKG

## What This Is

LogicKG 当前是一个面向论文与教材 Markdown 的科研知识工作台：它能导入文献、抽取 `PaperLogicTrace` / 图谱对象、写入 Neo4j，并通过 FastAPI + React 提供图谱浏览、Ask 检索问答、导入中心和运维界面。

这个项目的最终目标不只是知识图谱抽取，而是把论文内容、历史科研环境和跨论文路线状态编译成 `L1-L4` 多层科研知识对象，用来提升 AI 发现高价值、可行科研问题与假说的能力。

## Core Value

持续产出可审计、可回放、可训练的科研判断样本，而不只是产出“看起来像懂论文”的摘要或图谱。

## Requirements

### Validated

- [x] 论文与教材可以导入为结构化图谱资产，并通过 Neo4j / FastAPI / React 工作台消费。
- [x] 单篇论文已经有 `PaperLogicTrace` / `ResearchMove` 风格的 `L2` 结构化抽取基础，并带有 evidence anchors、citation acts 与 quality gates。
- [x] 系统已经具备 Ask、paper detail、community/similarity、ops/config/import 等平台能力，可作为后续科研推理层的底座。

### Active

- [ ] 用 `topic_scope + cutoff_year + route packet` 驱动真实的历史回放编译，而不是继续按“多抽几篇论文”推进。
- [ ] 建立最小可用的 `L1 Historical Environment` 资产，让 `RouteState` 能消费真实的资源 / benchmark / toolchain 时间切片。
- [ ] 把已有 `RouteState / WhyNowCase / RouteComparisonCase / DecisionPriorCard / DecisionEpisode` 原型变成可在真实语料上反复运行的 replay pipeline。
- [ ] 用 replay failure taxonomy 反向驱动 `L2` 外科式改进，而不是先重做 `L2` 结构。

### Out of Scope

- 现在就做开放式、自由生成的科研假说系统 - `L4` 还需要先收敛成 decision priors 与反模式层。
- 现在就重做整个 `L2` 主结构 - 当前策略是先用 replay failure 暴露缺口，再做定向修补。
- 现在就做大规模全语料自动抽取和自动 packet 发现 - 在 packet 合同、`L1` 资产和 replay 评测稳定前，规模化会放大噪声。
- 把 `L3/L4` 当成单篇论文的“高级摘要” - 这会破坏多论文聚合与历史切片的目标。

## Context

- 当前代码库已经是一个可运行的全栈平台：`backend/app/` 负责导入、抽取、图谱、检索、community、tasks；`frontend/src/` 提供工作台界面；Neo4j / FAISS / LLM provider 构成运行底座。
- 你的总设计文档 `docs/科学家思维AI项目技术文档.md` 已明确项目方向：`L1 HistoricalEnvironment -> L2 PaperLogicTrace -> L3 RouteState -> L4 DecisionPrior -> DecisionEpisode`，并强调项目目标是训练科研判断，而不是做更强的论文问答。
- 2026-04-01 已经补齐了一批 research-logic 规格与原型实现：`RouteState`、`WhyNowCase`、`RouteComparisonCase`、`DecisionPriorCard`、`DecisionEpisode` 以及 `HistoricalReplayCompiler` 均有 schema/spec 或 builder 原型与单元测试。
- 当前真正的工程缺口不再是“有没有单篇论文抽取”，而是“这些单篇信号能否在一个受限、可审计的 route packet 内被稳定编译为 `L3/L4` 对象”。

## Constraints

- **Tech stack**: 现有系统基于 FastAPI + React + Vite + Neo4j + FAISS，后续方案要优先复用现有平台，而不是另起一套离线研究原型。
- **Historical discipline**: 所有高层对象都必须遵守 `cutoff_year` 和 anti-hindsight 约束，否则训练样本不可信。
- **L2 boundary**: `L2` 仍然是单篇论文证据层；`L3/L4` 必须由多论文聚合与重建产生。
- **Evidence auditability**: 高层字段必须能回溯到 `PaperLogicTrace` 与必要的 `L1` 资产，避免 fluent-but-ungrounded 的输出。
- **Brownfield reality**: 这是已有代码库，不是 greenfield。`.planning/` 必须回填真实现状、已做研究和当前风险，而不能把项目写成从零开始。
- **Execution strategy**: 短期优先做 bounded route packet replay，而不是一开始就跑上千篇论文的全量 pipeline。

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 保持 `L2` 为单篇论文证据层 | `L3/L4` 的价值来自多论文聚合与历史切片，单篇层不应被误当成路线层 | ✅ Good |
| 先做 replay / compiler 主链，再按失败样本回补 `L2` | 否则会在不知道下游真正需要什么的情况下过度优化抽取 | ✅ Good |
| `L4` 先收敛为 decision priors / anti-patterns | 这比直接做开放式假说生成更可审计、更可训练 | ✅ Good |
| 当前 milestone 以 `route packet + historical replay` 为主线 | 这是把现有 research-logic 原型变成真实工程闭环的最短路径 | ✅ Good |
| GSD 采用 `interactive + assumptions` 风格 | 这个项目的关键风险常常来自隐藏假设，先显式化再规划更稳妥 | ⚠️ Revisit |

## Current State

- Phase 5 is complete.
- The codebase can now induce typed prior and anti-pattern candidates from grouped route-state packages.
- The jamming pilot produced a real Phase 5 review bundle under `tmp/phase5_multi_route_prior_induction/review_bundle/`.
- Package-level prior review remains intentionally conservative: accepted prior ids stayed empty while accepted anti-pattern ids were recorded explicitly.
- Phase 6 now owns audited export policy and training-ready packaging for these reviewed decision-layer artifacts.

---
*Last updated: 2026-04-02 after Phase 5 closeout*
