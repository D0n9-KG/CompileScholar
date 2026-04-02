# Phase 1: Replay Pilot Packetization - Context

**Gathered:** 2026-04-01
**Status:** Ready for planning

<domain>
## Phase Boundary

本 phase 的目标是在现有 `research_logic` 原型之上，做出第一个真实 `topic_scope + cutoff_year` 的 route packet replay pilot。它关注的是“如何把已有 builder 串成一个可回放、可审计、可暴露缺口的闭环”，而不是在这一阶段就完成自动 packet discovery、完整 `L1` 资产体系或开放式科研问题生成。

</domain>

<decisions>
## Implementation Decisions

### Scope discipline
- **D-01:** 本 phase 只做一个 bounded 的手工 packet，不追求自动 packet builder。
- **D-02:** packet 必须显式记录 included / excluded items、role coverage、trace refs 与 leakage policy，不能继续依赖会话记忆。
- **D-03:** packet 的核心成功标准是“可审计、可复现、可暴露缺口”，不是“摘要写得好”。

### L2 / L3 / L4 boundary
- **D-04:** `L2` 结构本 phase 不重做；只记录 replay 暴露出的高价值缺口，留到后续 `L2` surgical loop 处理。
- **D-05:** `L3/L4` 依旧被视为多论文编译对象，禁止从单篇论文直接生成最终路线结论。
- **D-06:** 现有 `RouteState / WhyNow / RouteComparison / DecisionPrior / DecisionEpisode / HistoricalReplayCompiler` 原型应被整合成 pilot workflow，而不是继续停留在分散测试里。

### Quality and evaluation
- **D-07:** replay 输出必须保留 stage-level `quality_flags`、`ready_for_*` gate 与人工审查建议。
- **D-08:** 首次 pilot 可以接受人工 packet 选择，但不能接受人为 hindsight 泄漏。
- **D-09:** 本 phase 的产物应直接服务 Phase 2/3，而不是另起一套一次性脚本。

### the agent's Discretion
packet artifact 文件布局、pilot 命令入口、review checklist 呈现形式、最小可用 replay report 格式。

</decisions>

<specifics>
## Specific Ideas

- 优先选择一个历史节点清晰、论文量足够、而且本地语料可访问的 topic 作为 pilot。
- packet 早期目标规模保持在 20-30 篇左右；如果 trace coverage 不足，可以先更小，但必须写清楚缺口。
- 评测时不要只盯自然语言输出，重点看 packet completeness、route-state grounding、prior support / counterexample 边界。

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Intent
- `docs/科学家思维AI项目技术文档.md` - 项目的最终目标、四层知识结构、为什么先做 training-usable scientific judgment objects。
- `README.md` - 当前已存在的平台能力与运行边界。
- `TECHNICAL_OVERVIEW.zh-CN.md` - 当前主线架构与 legacy boundary。

### Research Logic Contracts
- `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md` - `L2 -> L3/L4` 字段依赖、当前缺口与实施顺序。
- `docs/superpowers/specs/2026-04-01-logickg-route-packet-schema.md` - `RoutePacket` canonical schema。
- `docs/superpowers/specs/2026-04-01-logickg-routestate-schema.md` - `RouteState` canonical schema。
- `docs/superpowers/specs/2026-04-01-logickg-why-now-and-route-comparison-schema.md` - `WhyNowCase` 与 `RouteComparisonCase` contract。
- `docs/superpowers/specs/2026-04-01-logickg-decision-prior-and-episode-schema.md` - `DecisionPriorCard` / `AntiPatternCard` / `DecisionEpisode` contract。

### Existing Code Paths
- `backend/app/paper_logic_trace/models.py` - `L2` canonical typed models。
- `backend/app/paper_logic_trace/derived_views.py` - `L2.5` / route-facing derived views。
- `backend/app/research_logic/models.py` - 当前 typed schemas。
- `backend/app/research_logic/route_state_synthesizer.py` - `RouteState` builder 原型。
- `backend/app/research_logic/why_now_builder.py` - `WhyNowCase` builder 原型。
- `backend/app/research_logic/route_comparison_builder.py` - `RouteComparisonCase` builder 原型。
- `backend/app/research_logic/decision_prior_builder.py` - prior builder 原型。
- `backend/app/research_logic/decision_episode_builder.py` - episode builder 原型。
- `backend/app/research_logic/historical_replay_compiler.py` - 端到端 replay compiler 原型。
- `backend/tests/test_route_state_synthesizer.py` - `RouteState` 行为与质量门测试。
- `backend/tests/test_decision_prior_builder.py` - prior builder 行为测试。
- `backend/tests/test_historical_replay_compiler.py` - replay compiler 端到端 fixture 测试。

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `PaperLogicTrace` typed models and derived views: 可以直接作为 packet replay 的上游输入。
- `backend/app/research_logic/` builders: 已经具备 schema-validating 输出与基础质量分级。
- `backend/tests/test_route_state_synthesizer.py` / `test_decision_prior_builder.py` / `test_historical_replay_compiler.py`: 已有最小 fixture / regression 思路，可扩展到 pilot artifacts。

### Established Patterns
- 以 Pydantic model 固化 contract，再用 builder 生成对象。
- 质量判断通过 `quality_tier`, `quality_flags`, `ready_for_*` 等显式字段表达。
- 文档驱动开发已经开始成型：spec 先于大规模实现。

### Integration Points
- packet replay 很可能最终落在 `backend/app/research_logic/` 与独立 artifact 输出之间。
- 如果需要接入平台工作流，后续可考虑 `tasks/`、`ops` 或独立 API route，但本 phase 不强求 UI 暴露。

</code_context>

<deferred>
## Deferred Ideas

- 自动 `topic_scope` 发现与 route family clustering - 后续 phase
- 完整 `L1 Historical Environment` 资产体系 - Phase 2
- 多 packet / 多 route 的 prior induction - Phase 5
- 基于 `DecisionEpisode` 的科研问题 / 假说候选生成 - v2

</deferred>

---

*Phase: 01-replay-pilot-packetization*
*Context gathered: 2026-04-01*
