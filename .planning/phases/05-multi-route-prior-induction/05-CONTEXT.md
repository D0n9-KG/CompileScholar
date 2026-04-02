# Phase 5: Multi-Route Prior Induction - Context

**Gathered:** 2026-04-02 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 5 的目标不是宣称 `L4` 已经成熟，也不是直接产出可规模训练的数据集，而是先做一个受限、可审计的 `L4` induction pilot：从多个 `RouteState` 中归纳 `DecisionPriorCard / AntiPatternCard` 候选，并补齐 held-out、counterexample、review 等验收约束。

本阶段关注的是“先把跨层闭环跑通，再用真实 prior / episode 失败反向优化 `L1/L2/L3/L4`”。它不重做 `L1/L2/L3` 主合同，不扩展成新的 UI 审核系统，也不把 `DecisionEpisode` 最终导出和训练集成熟度问题提前到本阶段解决。

</domain>

<decisions>
## Implementation Decisions

### Phase positioning
- **D-01:** Phase 5 被锁定为受限、可审计的 `L4` induction pilot，而不是“成熟 L4 层”交付。目标是先验证跨层决策闭环能否成立，再根据失败反向优化上游各层。
- **D-02:** Phase 5 的输出只能被视为 prior / anti-pattern 候选与审计产物，不能被表述为已经具备大规模训练使用条件的稳定知识层。
- **D-03:** Phase 6 的角色被前置约束为“审计级 `DecisionEpisode` export pilot”，不是“直接进入训练数据生产”。

### Induction inputs and extraction boundary
- **D-04:** `L4` 的主输入层是多 `RouteState` 聚类结果，而不是原始 trace、单篇论文或单个 packet。prior/anti-pattern 应从 `L3` 提炼，而不是绕过 `L3` 直接从 `L1/L2` 生成。
- **D-05:** `L1/L2` 不直接生成 prior，但必须继续作为 `L4` 的历史边界、证据溯源和适用条件约束来源。任何 prior 都必须能回溯到 `L2` evidence anchors，并受 `L1` cutoff / environment 约束。
- **D-06:** Phase 5 复用现有 `support / alternative / held_out` route-state package 边界作为 induction 输入合同，不重新设计一套新的多路由输入格式。

### Prior and anti-pattern induction path
- **D-07:** Phase 5 需要把当前“单张 prior card builder 原型”提升为“批量候选归纳路径”：按多 route cluster 生成多个 prior / anti-pattern 候选，而不是继续把一轮 replay 强行压成单张 prior。
- **D-08:** `AntiPatternCard` 与 `DecisionPriorCard` 同 phase 落地，不再继续停留在 schema-only 状态。anti-pattern 应作为第一类对象显式建模，而不是藏在自由文本备注里。
- **D-09:** anti-pattern 的第一版归纳来源优先复用现有 replay failure、route comparison 和 prior 边界信号，先证明对象合同成立，再讨论更复杂的归纳策略。

### Acceptance and review gates
- **D-10:** `green` prior 的硬门槛保持不变：必须有多 route support cluster、显式 counterexample 搜索、held-out 结果和 reviewer metadata；缺失任一项都不得升级为 `green`。
- **D-11:** Phase 5 的 review workflow 先做成文件化 / CLI / report 驱动的 lightweight 流程，不在这一阶段扩展为新的前端审核界面或数据库状态机。
- **D-12:** Phase 5 必须保持与现有 replay / episode 合同兼容，让后续 `DecisionEpisode` 组装可以直接消费已接受的 prior / anti-pattern cards，但不在本阶段宣称 `DecisionEpisode` 已达到训练数据成熟度。

### the agent's Discretion
- cluster 具体切分特征、候选 ranking 公式、distinctiveness 评分方式
- anti-pattern 初版更偏 replay failure 驱动还是 comparison / not-now 信号驱动
- review 报告的具体 JSON / Markdown 文件布局，以及是否单独产出 candidate registry

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project and milestone framing
- `.planning/PROJECT.md` - 项目总体目标、`L1/L2/L3/L4` 分层边界、anti-hindsight 约束，以及“先跑闭环再反向优化各层”的核心策略
- `.planning/REQUIREMENTS.md` - `L4-01` 与 `L4-02` 的 requirement 映射，说明 Phase 5 与 Phase 6 的验收边界
- `.planning/ROADMAP.md` - Phase 5 与 Phase 6 的目标、成功标准和 plan slots
- `.planning/STATE.md` - 当前 handoff、剩余 `L2/L3` 风险和“上游尚未成熟”的现实状态
- `docs/科学家思维AI项目技术文档.md` - 全局科学推理架构与 `DecisionPrior / DecisionEpisode` 在全链路中的定位

### Phase 3 and Phase 4 baselines
- `.planning/phases/03-grounded-route-replay-compilation/03-CONTEXT.md` - Phase 3 已锁定多 route packaging contract，不允许 Phase 5 回退到单 route / 单论文抽象
- `.planning/phases/04-replay-failure-taxonomy-and-l2-surgical-loop/04-CONTEXT.md` - Phase 4 已锁定 file-based、auditable replay 产物边界，Phase 5 需在此之上扩展而非绕开
- `docs/replay/reports/route-state-package-pilot-2026-04-02.md` - 现有绿色 route-state package baseline，说明 multi-route packaging 已经跑通
- `docs/replay/reports/phase4-l2-surgical-delta.md` - Phase 4 同 slice delta 仍未压低 live `L2` failures，证明 Phase 5/6 只能按 pilot 理解，不能假设上游已成熟

### L4 contracts and current implementation
- `docs/superpowers/specs/2026-04-01-logickg-decision-prior-and-episode-schema.md` - `DecisionPriorCard` / `AntiPatternCard` / `DecisionEpisode` 的 canonical contract，以及推荐的 clustering / held-out / review 工作流
- `backend/app/research_logic/models.py` - 现有 typed contracts，特别是 prior/anti-pattern 的 green gate、review metadata 和 quality flags
- `backend/app/research_logic/decision_prior_builder.py` - 当前单 prior builder 原型，说明 Phase 5 的核心缺口是“从单卡生成升级为多候选归纳”
- `backend/app/research_logic/decision_episode_builder.py` - 下游 `DecisionEpisode` 已能消费 prior / anti-pattern lists，Phase 5 必须保持兼容
- `backend/app/research_logic/historical_replay_compiler.py` - 当前 replay 仍只串单张 prior card，展示 Phase 5 需要扩展但不能破坏的主链
- `backend/app/research_logic/route_state_package.py` - 现有 multi-route package 合同与 grouped route-state 资产边界
- `backend/app/research_logic/replay_io.py` - file-based bundle / summary / inspection 输出合同，适合作为 lightweight review 的复用面

### Regression boundaries and current fixtures
- `backend/tests/test_decision_prior_builder.py` - prior builder 的 support / held-out / review regression 边界
- `backend/tests/test_decision_episode_builder.py` - `DecisionEpisode` 对 prior / anti-pattern 消费方式的下游边界
- `backend/tests/test_historical_replay_compiler.py` - 当前 replay pipeline 如何消费 support / alternative / held_out route states
- `backend/tests/test_route_state_package.py` - route-state package 的结构性质量边界
- `backend/tests/test_research_logic_models.py` - `AntiPatternCard` / `DecisionPriorCard` / `DecisionEpisode` typed contract 的最小回归样例

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `backend/app/research_logic/route_state_package.py`: 已经提供 package manifest、grouped route states、validation、bundle 写出与加载能力，适合直接作为 Phase 5 induction 输入面
- `backend/app/research_logic/decision_prior_builder.py`: 已有单 prior 归纳原型，可复用其中 readiness / bottleneck / held-out / review gate 逻辑
- `backend/app/research_logic/decision_episode_builder.py`: 已能接收 `prior_cards` 和 `anti_pattern_cards` 列表，说明 Phase 5 可以先扩展候选层，而不必立即改写 episode contract
- `backend/app/research_logic/historical_replay_compiler.py`: 已串起 `RouteState -> WhyNow -> RouteComparison -> DecisionPrior -> DecisionEpisode` 闭环，是观察 prior/episode 失败如何反向暴露上游问题的现成入口
- `tmp/phase3_route_state_package/` 与 `tmp/phase4_l2_surgical_loop/`: 已有真实 grouped route-state 资产和 prior / replay runtime baseline，可直接作为 Phase 5 早期样本来源

### Established Patterns
- backend research-logic 采用 typed Pydantic contracts + JSON artifact boundary，而不是临时 dict 协议
- 绿色可用性通过 `quality_tier`、`quality_flags`、`review`、`held_out_consistency` 等显式字段控制，而不是靠隐式约定
- 新增能力优先接入现有 replay/package/report 链路，维持 file-based、可审计、可回放的工作流
- 当前全项目策略是“先跑通闭环，再用真实 failure 反向修正上游层”，不是先把单层优化到自认为完美

### Integration Points
- Phase 5 的新候选归纳层应落在 `backend/app/research_logic/`，紧邻 `decision_prior_builder.py`，并直接消费 grouped route states 或 replay bundle
- anti-pattern 生成最自然的接入点是 replay failure / route comparison / not-now 信号，而不是新增独立数据源
- lightweight review 应优先复用 `replay_io.py`、CLI scripts 和 Markdown/JSON 报告面，而不是引入新的前端审核面
- Phase 6 应直接消费 Phase 5 被接受的 prior / anti-pattern assets，但不改变其 pilot / audit-first 定位

</code_context>

<specifics>
## Specific Ideas

- “先把跨层决策闭环跑通，再用闭环失败反向优化每一层” 是 Phase 5/6 的总策略
- `L4` 主要从多 `RouteState` 中提炼，但必须受 `L1/L2` 约束并可回溯到证据
- Phase 5/6 现在都是 pilot，不代表上游 `L1/L2/L3` 已经足够成熟，更不代表训练数据已经 ready
- `AntiPatternCard` 需要与 prior 一起落地，避免继续停留在 schema-only 状态
- 如果后续要主张“训练级样本成熟”，还需要单独补 stronger `L2/L3`、canonical packet 资产化和 cross-topic validation

</specifics>

<deferred>
## Deferred Ideas

- 新的前端 review UI 或数据库化审核流程 - 重要，但不属于 Phase 5 的 lightweight review 范围
- 大规模训练数据生产或“训练集 ready”声明 - 必须等 `L2/L3` 和跨 topic 验证更稳后再谈
- 跨 topic prior 泛化评估与长期质量栈 - 应在闭环跑通后单独扩展，而不是塞进本 phase
- 重新设计 `L1/L2/L3` 主合同 - 当前策略是先通过 `L4 / DecisionEpisode` 闭环暴露真实缺口，再做定向修补

</deferred>

---

*Phase: 05-multi-route-prior-induction*
*Context gathered: 2026-04-02*
