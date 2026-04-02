# Roadmap: LogicKG

## Overview

当前 roadmap 不再把 LogicKG 继续当作“更强的论文问答系统”来推进，而是把现有知识图谱平台推进成一条可回放的科研判断编译链。短期目标是围绕一个真实 `topic_scope + cutoff_year` 做出高质量 route packet replay，用真实失败样本来决定 `L1/L2/L3/L4` 下一轮该补什么，而不是继续盲目放大抽取规模。

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): 当前 milestone 的主线工作
- Decimal phases (2.1, 2.2): 后续如果出现紧急插入项，再用 inserted phase

- [ ] **Phase 1: Replay Pilot Packetization** - 在真实 topic + year 上冻结第一个 route packet，并把现有 research-logic 原型串成可回放 pilot。
- [ ] **Phase 2: Historical Environment Bridge** - 为试点领域补齐最小 `L1` 历史环境资产，并接入 packet 编译。
- [ ] **Phase 3: Grounded Route Replay Compilation** - 把 `RouteState / WhyNow / RouteComparison` 从“原型 builder”推进到真实多论文 replay 产物。
- [ ] **Phase 4: Replay Failure Taxonomy And L2 Surgical Loop** - 用 replay 失败样本反向驱动 `L2` 外科式修补与评测闭环。
- [x] **Phase 5: Multi-Route Prior Induction** - 从多个 `RouteState` 归纳 `DecisionPriorCard / AntiPatternCard` 并加上 held-out / review 约束。
- [ ] **Phase 6: Decision Episode Audit Export** - 把 `DecisionEpisode` 做成可训练、可评估、可审计的导出资产。

## Phase Details

### Phase 1: Replay Pilot Packetization
**Goal**: 在一个真实 `topic_scope + cutoff_year` 上冻结第一份 `RoutePacket`，并把当前 research-logic builder 串成一次可复现、可人工审查的 replay pilot。
**Depends on**: Nothing (first phase)
**Requirements**: PKT-01, PKT-02, EVAL-01
**Success Criteria** (what must be TRUE):
  1. 团队可以产出一个冻结的 packet manifest，清楚记录 included / excluded papers、角色覆盖、trace refs 与 leakage policy。
  2. 同一 packet 可以反复编译出结构化 replay bundle，而不是依赖手工临时拼装。
  3. replay 输出会显式暴露各阶段 quality flags 与人工审查点，而不是只给自然语言摘要。
**Plans**: 3 plans

Plans:
- [ ] 01-01: Wire route-packet artifacts and a replay entry point around the existing `research_logic` builders
- [ ] 01-02: Build one manually curated real-paper packet from the local corpus with clear inclusion / exclusion rationale
- [ ] 01-03: Run the first replay bundle and write a pilot review checklist plus failure inventory

### Phase 2: Historical Environment Bridge
**Goal**: 为试点路线建立最小 `L1 Historical Environment` 快照，让 packet replay 不再完全依赖论文文本内部信号。
**Depends on**: Phase 1
**Requirements**: L1-01, L1-02
**Success Criteria** (what must be TRUE):
  1. 至少一个试点领域拥有 resource / benchmark / toolchain / protocol 的最小历史快照资产。
  2. replay pipeline 能把 `L1` snapshot refs 接入 `RoutePacket` / `RouteState` 编译，而不会引入 cutoff 之后的信息。
  3. `L1` 字段与 `L2/L3` 消费关系有明确的 contract 和审计边界。
**Plans**: 3 plans

Plans:
- [ ] 02-01: Define the minimum viable `L1` asset schemas and storage layout for the pilot domain
- [ ] 02-02: Build one historical snapshot from authoritative sources plus corpus evidence
- [ ] 02-03: Connect `L1` snapshot consumption to packet / route-state compilation with leakage checks

### Phase 3: Grounded Route Replay Compilation
**Goal**: 把 `RouteState / WhyNowCase / RouteComparisonCase` 从“schema + unit-test 原型”推进到真实多论文 replay 结果。
**Depends on**: Phase 2
**Requirements**: L3-01, L3-02, L3-03
**Success Criteria** (what must be TRUE):
  1. `RouteState` 明确来源于多篇论文与 `L1` 信号聚合，而不是对单篇 seed 的改写。
  2. `RouteState` 稳定输出 dominant methods、benchmarks、bottlenecks、enabling conditions、alternative routes、evidence bundle 与 uncertainty。
  3. `WhyNowCase` 与 `RouteComparisonCase` 的判断链条能回溯到 route-state fields 与 evidence ids。
**Plans**: 3 plans

Plans:
- [x] 03-01: Build reusable support / alternative / held-out route-state packaging
- [x] 03-02: Add package validation and replay inspection artifacts
- [x] 03-03: Harden the remaining yellow route-state slices and comparison quality on real packaged routes

### Phase 4: Replay Failure Taxonomy And L2 Surgical Loop
**Goal**: 把 replay 失败从“感觉不对”变成结构化 failure taxonomy，并据此做 `L2` 定向修补。
**Depends on**: Phase 3
**Requirements**: L2-01, L2-02, EVAL-02
**Success Criteria** (what must be TRUE):
  1. replay 结果能分阶段指出是 packet、`L1`、`L2` 还是 `L3` 的缺口导致失败。
  2. `L2` 改动被 failure taxonomy 驱动，而不是继续无边界补字段。
  3. 至少一轮 `L2` 修补前后 replay 结果可比较，能看到成功率或质量信号变化。
**Plans**: 3 plans

Plans:
- [x] 04-01: Design a replay failure schema and reporting format
- [x] 04-02: Patch the highest-leverage `L2` gaps exposed by the pilot
- [x] 04-03: Re-run replay and compare quality deltas against the previous baseline

### Phase 5: Multi-Route Prior Induction
**Goal**: 从多个 `RouteState` 而不是单个 packet 中归纳 decision priors 与 anti-patterns。
**Depends on**: Phase 4
**Requirements**: L4-01
**Success Criteria** (what must be TRUE):
  1. prior 候选来自多个 route states 的聚合，而不是来自单一论文或单一 packet。
  2. 每个 prior 都有 applies / does-not-apply 边界、support cluster、counterexample search 和 held-out consistency。
  3. anti-patterns 与 prior 一样是显式对象，而不是隐藏在自由文本备注里。
**Plans**: 2 plans

Plans:
- [x] 05-01: Build a clustering and candidate-induction path for priors / anti-patterns
- [x] 05-02: Add held-out checks and lightweight review workflow for prior acceptance

### Phase 6: Decision Episode Audit Export
**Goal**: 把 `DecisionEpisode` 做成最终可训练、可评估、可回放的跨层样本导出。
**Depends on**: Phase 5
**Requirements**: L4-02
**Success Criteria** (what must be TRUE):
  1. `DecisionEpisode` 能组合 `L1/L2/L3/L4`，并显式记录 candidate question context、alternatives、why-this-not-that 与 hindsight labels。
  2. hindsight 仅位于 label / eval 侧，不进入训练输入。
  3. 样本导出格式能够被后续 question-discovery / hypothesis-generation 研究直接消费。
**Plans**: 2 plans

Plans:
- [ ] 06-01: Define the audited export format and leakage-safe assembly path for `DecisionEpisode`
- [ ] 06-02: Produce and inspect a first batch of training / eval-ready decision samples

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Replay Pilot Packetization | 3/3 | Completed | 2026-04-02 |
| 2. Historical Environment Bridge | 3/3 | Completed | 2026-04-02 |
| 3. Grounded Route Replay Compilation | 3/3 | Completed | 2026-04-02 |
| 4. Replay Failure Taxonomy And L2 Surgical Loop | 3/3 | Complete | 2026-04-02 |
| 5. Multi-Route Prior Induction | 2/2 | Complete | 2026-04-02 |
| 6. Decision Episode Audit Export | 0/2 | Not started | - |

---
*Roadmap defined: 2026-04-01*
*Last updated: 2026-04-02 after Phase 5 added multi-route prior review bundles and accepted anti-pattern outputs on the jamming package*
