# Phase 1: Replay Pilot Packetization - Research

**Researched:** 2026-04-01
**Domain:** route-packet historical replay for multi-paper scientific reasoning compilation
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- 本 phase 只做一个 bounded 的手工 packet，不追求自动 packet builder。
- packet 必须显式记录 included / excluded items、role coverage、trace refs 与 leakage policy，不能继续依赖会话记忆。
- packet 的核心成功标准是“可审计、可复现、可暴露缺口”，不是“摘要写得好”。
- `L2` 结构本 phase 不重做；只记录 replay 暴露出的高价值缺口，留到后续 `L2` surgical loop 处理。
- `L3/L4` 依旧被视为多论文编译对象，禁止从单篇论文直接生成最终路线结论。
- 现有 `RouteState / WhyNow / RouteComparison / DecisionPrior / DecisionEpisode / HistoricalReplayCompiler` 原型应被整合成 pilot workflow，而不是继续停留在分散测试里。
- replay 输出必须保留 stage-level `quality_flags`、`ready_for_*` gate 与人工审查建议。
- 首次 pilot 可以接受人工 packet 选择，但不能接受人为 hindsight 泄漏。
- 本 phase 的产物应直接服务 Phase 2/3，而不是另起一套一次性脚本。

### the agent's Discretion
- packet artifact 文件布局
- pilot 命令入口
- review checklist 呈现形式
- 最小可用 replay report 格式

### Deferred Ideas (OUT OF SCOPE)
- 自动 `topic_scope` 发现与 route family clustering
- 完整 `L1 Historical Environment` 资产体系
- 多 packet / 多 route 的 prior induction
- 基于 `DecisionEpisode` 的科研问题 / 假说候选生成

</user_constraints>

<research_summary>
## Summary

Phase 1 的标准做法不应该是“继续往 builder 上堆逻辑”，而应该是先补一层稳定的 replay artifact workflow，让现有 `RoutePacket` schema、`PaperLogicTrace` 导出、`HistoricalReplayCompiler` 和质量 gate 真的能围绕一个真实 packet 跑起来。当前代码里最有价值的现成资产已经都有了：`RoutePacket` / `RouteState` / `DecisionEpisode` 等 contract 已冻结，`HistoricalReplayCompiler` 也已把 `L3/L4` builder 串起来，但项目还缺少三个关键环节：一是 committed packet manifest，二是 runtime replay runner，三是 pilot review report。

对这个 phase 来说，最稳妥的架构是把“被版本控制的事实”和“本地运行时产物”分开。被提交到仓库的应该只有 packet manifest、selection notes、review report 和 failure inventory；真实 trace 输入目录、共享盘路径、runtime bundle 则由本地 CLI 参数提供并落到未提交的 `backend/runs/` 或等价目录。这样既符合长期 GSD 使用，也不会把本地路径和大语料细节写死到项目文档里。

**Primary recommendation:** 先把 replay workflow 产品化为“packet manifest + runner + report”三件套，再用首个真实 packet 的失败样本决定 Phase 2/3 的 `L1` 和 `L3` 优先级。
</research_summary>

<standard_stack>
## Standard Stack

当前 phase 最适合复用 repo 内已有 stack，而不是引入新框架。

### Core
| Asset | Status | Purpose | Why Standard Here |
|-------|--------|---------|-------------------|
| `backend/app/research_logic/models.py` | Existing | `RoutePacket` / `RouteState` / `DecisionEpisode` typed contracts | 已经是项目内部 canonical schema |
| `backend/app/research_logic/historical_replay_compiler.py` | Existing | 端到端 replay 编译主链 | 已经串联了 `RouteState -> WhyNow -> Prior -> Episode` |
| `backend/app/paper_logic_trace/exporter.py` | Existing | 从图数据库导出 `PaperLogicTrace` | 是从现有平台拿 `L2` 资产的最直接入口 |

### Supporting
| Asset | Status | Purpose | When to Use |
|-------|--------|---------|-------------|
| `backend/scripts/` | Existing | 放本地可执行脚本入口 | Phase 1 适合在这里放 replay runner |
| `backend/tests/test_historical_replay_compiler.py` | Existing | replay fixture regression | 新增 artifact / runner 测试时直接沿用 |
| `docs/superpowers/specs/*.md` | Existing | phase 的 canonical design source | 所有 packet / replay 设计都应先对齐这些文档 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| committed packet manifest + local runner | 直接在测试文件里手工拼 packet | 快但不可审计，无法长期复用 |
| local CLI runner | 直接加 API/前端入口 | 产品化过早，会把 Phase 1 复杂度放大 |
| committed review report | 只保留 runtime JSON bundle | 机器可读但不利于长期人工复盘和 GSD 阶段总结 |

</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Recommended Project Structure
```text
backend/
|- app/research_logic/              # replay artifact I/O and compilation helpers
|- scripts/run_replay_pilot.py      # local replay entry point
|- tests/                           # runner + manifest regression tests
docs/
|- replay/pilot_packets/            # committed packet manifests and selection notes
\- replay/reports/                  # committed pilot review and failure inventory
```

### Pattern 1: Manifest In Git, Bundle Out Of Git
**What:** 将 packet manifest、selection rationale、review report 放在仓库里；将真实运行产物、bundle JSON、共享盘路径输入放在本地运行时参数和 gitignored 输出目录。
**When to use:** Phase 1 replay、后续任何需要本地大语料参与但又要留下可复盘记录的流程。
**Why recommended:** 同时满足审计性、可复现性和对本地敏感路径的隔离。

### Pattern 2: Compile From Typed Inputs, Report In Markdown
**What:** 运行时严格消费 `RoutePacket` + `PaperLogicTrace[]` + optional route-state supports，输出 typed JSON bundle；再把关键结论压缩成 committed Markdown report。
**When to use:** 当前所有 replay pilot 与 failure review。
**Why recommended:** JSON 适合后续机器消费，Markdown 适合 GSD 和人工决策。

### Anti-Patterns to Avoid
- **把 packet 选择逻辑埋进测试 helper:** 会让真实 replay 与 fixture replay 混在一起，后续很难追踪 packet 边界。
- **把共享盘或本地绝对路径写进 committed manifest:** 后续换机器或分享仓库时会立刻失效，还会污染提交历史。
- **在 Phase 1 直接做 UI / API 暴露:** 当前更需要先收敛 artifact contract，而不是先做入口。
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 高层 schema | 再造一套 phase1 专用 packet / replay JSON | 直接复用 `research_logic.models` | 否则 contract 会再次分叉 |
| replay 主链 | 新写一条平行 compiler | 直接包裹 `HistoricalReplayCompiler` | 当前主问题不是 builder 不够多，而是缺少 artifact workflow |
| trace 导出 | 临时解析原始文件生成伪 trace | 尽量复用 `PaperLogicTrace` exporter 或现有 trace artifacts | 避免绕开现有 `L2` canonical 层 |

**Key insight:** Phase 1 的价值是把现有 contract 串起来，而不是再制造一套新 contract。
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Packet 看似完整，实则没有真实 trace coverage
**What goes wrong:** manifest 里列了论文，但缺 trace_id 或 trace 文件无法加载，结果 replay 只是对空壳 packet 运行。
**Why it happens:** 只盯 `paper_id`，忽略 compile-ready 约束。
**How to avoid:** 在 runner 和测试里把 trace coverage 作为硬 gate；manifest regression test 必须校验 trace refs。
**Warning signs:** `included_items` 有 paper 但 `trace_id` 缺失；runner 只输出 warning 不降级。

### Pitfall 2: Hindsight 泄漏通过 selection notes 混进 packet
**What goes wrong:** 选择 packet 时引用 cutoff 之后的 survey / hindsight judgement，导致 replay 结果失真。
**Why it happens:** 手工 packet 很容易偷看后验知识。
**How to avoid:** 在 manifest 和 selection notes 中单独写 `leakage_policy` 与 `excluded_after_cutoff`。
**Warning signs:** selection notes 中出现“后续证明”“后来成为主流”等语言。

### Pitfall 3: 把 replay report 写成 fluent summary
**What goes wrong:** 输出看起来通顺，但没有 quality flags、missing setup、failure taxonomy，无法服务下一 phase。
**Why it happens:** 过度追求 narrative，而不是可执行诊断。
**How to avoid:** review report 固定包含 packet quality、compile outcome、blocking failures、next fixes。
**Warning signs:** 报告没有列出具体 failure ids / flags / missing inputs。
</common_pitfalls>

<open_questions>
## Open Questions

1. **首个 pilot topic_scope 具体选什么**
   - What we know: 用户已有大量本地论文语料，且此前一直在关注 L2->L3/L4 的科研路线问题。
   - What's unclear: 哪个 topic 的历史节点最适合 Phase 1 的 bounded pilot。
   - Recommendation: 在 Plan 01-02 中先做 selection notes，优先选 cutoff 清晰、替代路线明显、trace 可获得的 topic。

2. **真实 `PaperLogicTrace` 从哪里读取**
   - What we know: repo 里已有 `export_paper_logic_trace(client, paper_id)`，但真实本地语料也可能已有离线 trace / JSON 产物。
   - What's unclear: Phase 1 将从 Neo4j 导出、从 artifact 目录读取，还是两者都支持。
   - Recommendation: Runner 同时支持 packet manifest + local trace directory，优先用最少改动能跑通的输入路径。
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- `docs/科学家思维AI项目技术文档.md` - 总目标与四层设计
- `docs/superpowers/specs/2026-04-01-logickg-route-packet-schema.md` - packet contract
- `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md` - 下游依赖与缺口判断
- `backend/app/research_logic/models.py` - typed schemas
- `backend/app/research_logic/historical_replay_compiler.py` - replay 主链原型
- `backend/app/paper_logic_trace/exporter.py` - `L2` 导出入口

### Secondary (MEDIUM confidence)
- `backend/tests/test_historical_replay_compiler.py` - 当前 replay fixture 的真实能力边界
- `.planning/PROJECT.md`, `.planning/ROADMAP.md`, `.planning/phases/01-replay-pilot-packetization/01-CONTEXT.md` - 当前 GSD 约束与 phase 范围

</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: route packet replay workflow
- Ecosystem: existing `research_logic`, `paper_logic_trace`, backend scripts, docs artifacts
- Patterns: manifest + runner + report
- Pitfalls: trace coverage, hindsight leakage, narrative-only reporting

**Confidence breakdown:**
- Standard stack: HIGH - 主要基于当前 repo 已存在能力
- Architecture: HIGH - 直接由 phase goal 和现有 code paths 推出
- Pitfalls: HIGH - 都是当前 Phase 1 最可能踩到的坑
- Code examples: MEDIUM - 本 phase 更偏流程编排，代码细节将在实施时具体化

**Research date:** 2026-04-01
**Valid until:** 2026-05-01

</metadata>

---

*Phase: 01-replay-pilot-packetization*
*Research completed: 2026-04-01*
*Ready for planning: yes*
