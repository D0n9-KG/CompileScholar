# Codebase Concerns

**Analysis Date:** 2026-04-01

## Tech Debt

**Scientific-reasoning path still sits beside the main product path:**
- Issue: `research_logic` 已经有一批 typed models 和 builders，但主要停留在库层 / 单元测试层
- Why: 这条主线是最近才明确冻结出来的
- Impact: 容易出现“spec 很清楚，但真实语料 workflow 还没接上”的落差
- Fix approach: 先做一个 bounded route packet replay pilot，再逐步接入 tasks / artifacts / ops

**Brownfield planning gap:**
- Issue: 在本次回填前，仓库没有 `.planning/`，GSD 无法读取真实项目状态
- Why: 前期主要是探索式推进，而不是用 GSD 管理里程碑
- Impact: `gsd-progress` / `gsd-next` / `gsd-plan-phase` 缺少长期记忆与上下文
- Fix approach: 维持这次建立的 `.planning/`，后续所有 phase / plan / summary 都落到 GSD 流程里

## Known Bugs / Behavior Risks

**Replay quality still may be fixture-only stable:**
- Symptoms: builder 在 toy fixture 上全部通过，但真实 packet replay 可能暴露大量 boundary failure
- Trigger: 首次接入真实语料 packet
- Workaround: 先做一个小而可审计的 pilot packet，不直接全量跑语料
- Root cause: 目前还没有真实 replay 闭环

**Legacy discovery footprint can still confuse future work:**
- Symptoms: 代码树里还有 `discovery/` 等历史模块或语义残留
- Trigger: 新 phase 没有读当前 README / technical overview 就直接搜索旧概念
- Workaround: 以当前 README、`TECHNICAL_OVERVIEW.zh-CN.md` 与 `.planning/PROJECT.md` 为准
- Root cause: 仓库经历过主线迁移

## Security Considerations

**Secrets live in env files:**
- Risk: API key / 数据库密码容易因误操作进入版本控制或日志
- Current mitigation: `.env.example` 与 `.gitignore`，AGENTS 也明确禁止提交 secrets
- Recommendations: 保持 `.env` 本地化；任何新增服务都先补 `.env.example`

**Local corpus paths are operationally sensitive:**
- Risk: 大语料目录、共享盘路径与本地挂载信息不适合写入可提交文档
- Current mitigation: 目前主要保留在对话上下文中
- Recommendations: 后续用 packet manifests / relative refs 替代硬编码路径知识

## Performance Bottlenecks

**Large-scale corpus processing before contract freeze:**
- Problem: 在 packet / replay 合同没稳定前就全量处理上千篇文献，成本高且回报低
- Cause: 上游抽取很贵，下游目标仍在演化
- Improvement path: 继续坚持 bounded packet、failure-driven iteration

**Graph-wide rebuild workflows:**
- Problem: similarity / community / rebuild 可能在大语料上变慢
- Cause: 图谱、向量、聚类链路都偏重
- Improvement path: 只在需要时重建，并把 replay 所需资产单独组织成 packet 级工作流

## Fragile Areas

**`paper_logic_trace` <-> `research_logic` boundary:**
- Why fragile: 高层编译强依赖 `derived_views` 与 slot signals，一旦 `L2` 字段变化，`L3/L4` 很容易静默退化
- Common failures: 单篇结果看起来更完整，但 replay success rate 下降
- Safe modification: 任何 `L2` 结构改动都要配 replay-oriented regression checks
- Test coverage: 单元测试有基础，但真实 replay 回归还缺失

**Task / artifact / filesystem assumptions:**
- Why fragile: 平台很多流程依赖本地路径和 artifact 目录
- Common failures: 路径变动、共享盘访问方式变化、artifact 命名漂移
- Safe modification: 让 packet / replay 尽量使用 manifest 和 typed refs，而不是临时路径
- Test coverage: 这部分目前较弱

## Missing Critical Features

**Route packet builder and audit workflow:**
- Problem: 没有稳定的 packet manifest / review / replay 入口
- Current workaround: 手工在测试或脚本中拼对象
- Blocks: 真实历史 replay、后续 GSD phase execution
- Implementation complexity: Medium

**Historical environment layer (`L1`):**
- Problem: `RouteState` 缺少真实 resource / benchmark / toolchain 时间切片
- Current workaround: 先用 placeholder refs 或纯 `L2` 信号
- Blocks: grounded why-now / not-now reasoning
- Implementation complexity: Medium to High

## Test Coverage Gaps

**Real corpus replay:**
- What's not tested: 用真实文献 packet 编译 `RouteState -> DecisionEpisode`
- Risk: 现在的质量判断还不够可信
- Priority: High

**Cross-layer evaluation metrics:**
- What's not tested: `L2` 修补是否真的提升 `L3/L4` 成功率
- Risk: 可能持续优化错误目标
- Priority: High

**Ops / UI for replay management:**
- What's not tested: 未来 replay 入口、packet 管理与人工审核界面
- Risk: 后期产品化时 UI 质量与流程约束都可能缺位
- Priority: Medium

---
*Concerns audit: 2026-04-01*
*Update as replay workflow becomes real and risks move*
