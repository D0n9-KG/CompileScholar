# LogicKG 从 L2 到 L3/L4 的编译契约与缺口诊断

## 1. 背景

当前项目已经基本完成了一个重要转向：

1. 第二层不再以 `Claim / LogicStep` 作为最终输出，而是以 `PaperLogicTrace` 作为单篇论文的 canonical export。
2. 系统已经开始显式为后续层准备派生结构，例如 `route_compiler_contract`、`route_state_seed`、`ready_for_l3` 与 `ready_for_l4`。
3. 但系统仍然停留在“单篇论文是否足够支撑后续编译”的阶段，尚未进入真正的第三层 `RouteState` 重建与第四层 `DecisionPrior` 归纳阶段。

这意味着，下一步最重要的工作不应是继续泛化地“补更多抽取”，而应是冻结一份明确的编译契约：

1. `L3/L4` 需要哪些高层字段。
2. 这些高层字段分别依赖哪些 `L2.5` 信号与 `L1` 信号。
3. 当前代码已经提供了哪些可复用接口。
4. 当前真实样本缺口主要集中在哪里。

本文件就是这份契约与诊断。

## 2. 目标

1. 给出一份面向后续实现的 `L2 -> L3/L4` 依赖表。
2. 明确什么叫“L2 足够支撑 L3/L4”，避免把第二层误做成“论文全文镜像”。
3. 结合当前代码与随机样本审计，区分：
   1. 已有能力
   2. 局部薄弱点
   3. 结构性缺失
4. 为下一阶段实现 `RouteState`、`DecisionPriorCard`、`DecisionEpisode` 提供工程顺序。

## 3. 核心结论

### 3.1 L2 必须足够强，但优化目标必须改变

后续 `L3/L4` 的质量一定受 `L2` 上限约束。

但这里的“足够强”不是：

1. 尽可能覆盖论文中的所有句子。
2. 尽可能抽取更多 claim。
3. 尽可能让单篇论文摘要更像人工综述。

而是：

1. 对后续判断真正依赖的信号做到稳定保真。
2. 对每类高价值信号保留可审计证据锚点。
3. 允许把与下游无关的局部细节留在原文或中间层，而不是都塞入 canonical schema。

### 3.2 当前项目已经进入“L2.5 规范化”，但还未进入真正的 L3/L4

当前代码已经有以下后续接口：

1. `PaperLogicTrace` canonical schema
2. `route_compiler_contract`
3. `route_state_seed`
4. `ready_for_l3 / ready_for_l4` 质量 gate

但这些都还是：

1. 单篇论文视角
2. 候选信号视角
3. 编译前置视角

它们不是：

1. 跨论文、跨年份聚合后的 `RouteState`
2. 从多个 `RouteState` 归纳后的 `DecisionPriorCard`
3. 跨四层组装后的 `DecisionEpisode`

### 3.3 下一阶段应从“抽取优化”切换到“编译器实现”

下一阶段最有价值的主线应是：

1. 冻结 `L3/L4 -> L2.5 -> L1` 映射表
2. 定义 `route packet` 试点数据包
3. 实现 `RouteStateSynthesizer`
4. 实现 `DecisionPriorCard` 归纳与审核流程
5. 用 `L3/L4` 的失败样本反向驱动 `L2` 修补

## 4. 当前代码状态

### 4.1 已有的第二层 canonical 结构

当前 `PaperLogicTrace` 已包含：

1. `paper_metadata`
2. `canonical_core`
3. `derived_views`
4. `quality`

其中 `canonical_core` 已覆盖：

1. `evidence_anchors`
2. `moves`
3. `move_relations`
4. `citation_acts`
5. `figure_refs`
6. `table_refs`

`ResearchMove` 已覆盖主要 `L2.5` 槽位：

1. `research_objects`
2. `methods`
3. `observed_variables`
4. `metrics`
5. `comparators`
6. `conditions`
7. `effects`
8. `limitation_types`
9. `resource_mentions`

这说明项目已经具备“面向后续层的单篇结构化资产”基础。

### 4.2 已有的第三层前置接口

当前 `derived_views` 已提供：

1. `route_compiler_contract`
2. `route_state_seed`
3. `l1_bridge_hints`
4. `route_feature_candidates`
5. `future_work_signals`

这说明工程已经默认接受一个事实：

1. 单篇 L2 的最终价值不在于展示，而在于编译。
2. 后续层的主要消费方式不是“直接读摘要”，而是“消费结构化候选信号”。

### 4.3 当前仍未实现的对象

仓库里目前还没有真正实现：

1. `RouteState`
2. `WhyNowCase`
3. `RouteComparisonCase`
4. `DecisionPriorCard`
5. `AntiPatternCard`
6. `DecisionEpisode`

也没有真正实现：

1. route packet 组织器
2. cutoff year 历史切片编译器
3. novelty critic
4. feasibility critic

因此当前系统本质上仍是：

1. 强化后的 `L2`
2. 再加上少量 `L3/L4` 前置抽象

而不是完整的四层科研知识系统。

## 5. 什么叫“L2 足够支撑 L3/L4”

### 5.1 不是论文级完备，而是下游级充分

本项目后续层真正需要的不是“论文内容越多越好”，而是以下五类充分统计量：

1. 研究对象与问题边界
2. 方法与执行成熟度
3. 结果、指标、比较与效果方向
4. 条件、资源、工具、benchmark 与环境约束
5. limitation、瓶颈、未来方向与失败模式

如果这五类信号缺失，即使摘要写得再顺，也无法稳定生成高质量 `RouteState`。

如果这五类信号保真，即使 `L2` 没有覆盖全文所有局部细节，后续层仍然可能做得很好。

### 5.2 判断标准必须改成“编译成功率”

后续对 `L2` 的评估应从“单篇是否更完整”改成：

1. 这次 `L2` 变更是否提高了 `RouteState` 编译成功率。
2. 这次 `L2` 变更是否减少了 `WhyNow / not-now` 判断中的证据缺口。
3. 这次 `L2` 变更是否减少了 `DecisionPrior` 归纳中的噪声聚类。

也就是说，`L2` 应由下游失败样本来驱动，而不是由“还能再抽什么”来驱动。

## 6. L3 字段映射

### 6.1 `RouteState` 的建议核心字段

建议冻结一个最小可用 `RouteState`：

1. `topic_scope`
2. `cutoff_year`
3. `known_capabilities`
4. `dominant_methods`
5. `active_benchmarks`
6. `measurement_protocols`
7. `toolchains_and_infrastructure`
8. `known_bottlenecks`
9. `enabling_conditions`
10. `alternative_routes`
11. `supporting_evidence`
12. `challenging_evidence`
13. `why_now_features`
14. `not_now_features`
15. `source_packet_ids`

### 6.2 `RouteState` 依赖表

| L3 字段 | 主要依赖 L2.5 | 主要依赖 L1 | 当前状态 | 主要缺口 |
|---|---|---|---|---|
| `topic_scope` | `research_objects`, `problem`, `result`, `future_work` | topic 时间切片边界 | 部分已有 | 缺跨论文聚合与 topic 去歧义 |
| `known_capabilities` | `methods`, `effects`, `metrics`, `conditions`, `results` | 同期可用资源与实验环境 | 偏弱 | 缺跨论文能力归并与时间切片 |
| `dominant_methods` | `methods`, `move_relations`, `comparison_target` | tool timeline | 候选已有 | 缺方法族归并、成熟度建模 |
| `active_benchmarks` | `resource_mentions`, `metrics`, `comparators` | benchmark registry | 明显不足 | benchmark 识别与标准化偏弱 |
| `measurement_protocols` | `methods`, `metrics`, `conditions` | protocol registry | 明显不足 | 协议级信息抽取弱，跨论文合并缺失 |
| `toolchains_and_infrastructure` | `resource_mentions`, `methods`, `conditions` | software/hardware timeline | 明显不足 | 工具链、平台、算力环境表达不足 |
| `known_bottlenecks` | `limitation_types`, `future_work`, `challenging_evidence` | 资源短缺时间线 | 中等 | limitation 分类仍不稳定，跨论文归并未做 |
| `enabling_conditions` | `conditions`, `resource_mentions`, `future_work` | readiness features | 中等 | 条件与环境约束还未结构化到领域层 |
| `alternative_routes` | `methods`, `comparators`, `citation_acts`, `future_work` | community / route clusters | 很弱 | 缺 route family 发现与竞争路线建模 |
| `supporting_evidence` | `evidence_anchors`, `outcome signals` | 无 | 已有基础 | 缺跨论文证据汇总策略 |
| `challenging_evidence` | `limitation_types`, `limitation moves` | 无 | 中等 | 反例、失败、边界条件仍偏稀疏 |
| `why_now_features` | `effect_direction`, `comparison_target`, `resource_mentions`, `citation_acts` | resource timeline, benchmark timeline | 很弱 | 缺历史时点环境信息与聚合规则 |
| `not_now_features` | `limitation_types`, `missing resources`, `conditions` | readiness gaps | 很弱 | 缺显式“不可做”证据聚合 |

## 7. L4 字段映射

### 7.1 `DecisionPriorCard` 的建议核心字段

建议冻结一个最小可用 `DecisionPriorCard`：

1. `prior_id`
2. `label`
3. `applies_when`
4. `do_not_apply_when`
5. `recommended_actions`
6. `expected_failure_modes`
7. `supporting_route_states`
8. `counterexamples`
9. `confidence`

### 7.2 `DecisionPriorCard` 依赖表

| L4 字段 | 主要依赖 L3 | 主要依赖 L2/L1 | 当前状态 | 主要缺口 |
|---|---|---|---|---|
| `label` | 多个 `RouteState` 聚类结果 | 无 | 未实现 | 缺 route-state clustering |
| `applies_when` | `known_capabilities`, `dominant_methods`, `enabling_conditions`, `why_now_features` | readiness features | 未实现 | 缺归纳层与审核层 |
| `do_not_apply_when` | `known_bottlenecks`, `not_now_features` | limitation evidence | 未实现 | 缺反例组织与 hard blocker 表达 |
| `recommended_actions` | `alternative_routes`, `future directions`, route comparison | future_work, citation acts | 未实现 | 缺候选路线比较器 |
| `expected_failure_modes` | `challenging_evidence`, `counter-routes` | limitation types | 未实现 | 缺失败模式抽象器 |
| `supporting_route_states` | route-state support set | source packet traces | 未实现 | 缺 packet 编号与来源管理 |
| `counterexamples` | held-out `RouteState` | challenging evidence | 未实现 | 缺 held-out consistency check |

### 7.3 `DecisionEpisode` 的依赖

`DecisionEpisode` 不应直接从单篇 L2 生成。

它应依赖：

1. `L1` 的历史环境
2. `L2` 的单篇逻辑资产
3. `L3` 的路线状态快照
4. `L4` 的决策先验

因此在当前阶段，任何直接从 `PaperLogicTrace` 生成高层科研问题或假说的尝试，都应被视为：

1. 可做实验
2. 不可作为主路线

## 8. 当前样本与代码暴露的真实缺口

### 8.1 从随机审计看，当前主问题不是“读不懂单篇”，而是“无法稳定支撑 route compilation”

现有审计显示：

1. `20260328_sample10_run` 中 10 篇样本全部为 `yellow`。
2. 这 10 篇全部带有 `route_state_seed_thin`。
3. 后续若干定向修复后，`20260329` 三组宏观样本中已有 8/9 达到 `ready_for_l3`，但只有 4/9 达到 `ready_for_l4`，且其中 1 个样本仍有解析错误。

这说明当前瓶颈主要不是：

1. 系统完全不会做单篇逻辑建模

而是：

1. 单篇信号还不够稳，无法稳定转成后续层对象
2. 某些样本虽有足够内容，但分类、聚合与时点约束仍不够

### 8.2 当前已解决或部分解决的问题

基于近期代码与审计，以下问题已有明显进展：

1. 标题修复与元数据清洗
2. 中文方法句恢复
3. bilingual summary coherence
4. theory/modeling 论文的 gate 校准
5. explicit drawback / limitation 的恢复
6. `route_state_seed` 对瓶颈与挑战信号的更合理容忍

这些工作说明：

1. 当前 `L2` 已不再是纯展示层
2. 团队已经在隐式地用 `L3/L4` 需求反向驱动 `L2` 了

### 8.3 当前仍然结构性缺失的问题

以下问题不是再补几个 extractor 就能解决的：

1. 没有 `route packet` 层
2. 没有按 `topic_scope + cutoff_year` 的历史切片编译
3. 没有真正的跨论文 route aggregation
4. 没有 `why-now / not-now` 的显式特征层
5. 没有独立 novelty critic
6. 没有独立 feasibility critic
7. 没有 `DecisionPriorCard` 的聚类、候选、审核与 held-out 检查

这些都属于下一阶段编译器与评测系统的工作，不属于继续局部修 `L2` 的范畴。

## 9. 下一阶段建议实现顺序

### 9.1 Step 1: 冻结 `L3/L4 -> L2.5 -> L1` 映射表

输出物：

1. 本文档的定稿版
2. `RouteState` schema
3. `DecisionPriorCard` schema
4. `DecisionEpisode` schema

验收标准：

1. 每个高层字段都能追溯到 `L2.5` 与 `L1`
2. 每个字段都标明生成方式是规则、聚合、LLM 解释还是人工审核

### 9.2 Step 2: 定义 `route packet`

建议新增一个最小 `route packet` 合同：

1. `topic_scope`
2. `cutoff_year`
3. `paper_ids`
4. `included_traces`
5. `excluded_after_cutoff`
6. `l1_environment_snapshot`

验收标准：

1. 一个 packet 可以稳定重放
2. 可以切断 hindsight leakage

### 9.3 Step 3: 实现 `RouteStateSynthesizer`

输入：

1. 一个 `route packet`
2. packet 内所有 `PaperLogicTrace`
3. `L1` 环境快照

输出：

1. `RouteState`
2. `WhyNowCase`
3. `RouteComparisonCase`

实现原则：

1. 先规则聚合，再让 LLM 做受约束解释
2. 不允许 LLM 直接自由写路线

### 9.4 Step 4: 实现两个 critic

必须拆开：

1. `NoveltyCritic`
2. `FeasibilityCritic`

原因：

1. novelty 与 feasibility 在证据来源上不同
2. 这两类判断都不应混在生成器 prompt 里
3. 它们应成为 `DecisionPrior` 调用前的独立审核模块

### 9.5 Step 5: 反向驱动 L2 修补

只有当以下情况发生时，才回头继续改 `L2`：

1. `RouteState` 编译器明确缺某类输入字段
2. 某类 route packet 在多个样本上反复失败
3. `DecisionPrior` 聚类结果被无关噪声干扰

此时再精确补：

1. benchmark 抽取
2. protocol 抽取
3. toolchain / infrastructure 抽取
4. limitation taxonomy
5. citation purpose summary

## 10. 推荐的近期交付物

在不大改现有主线代码的前提下，推荐下一个里程碑只交付四个东西：

1. `RouteState` schema 文档
2. `DecisionPriorCard / AntiPatternCard / DecisionEpisode` schema 文档
3. 一个试点领域的 `route packet` 构建规范
4. 一个 packet 级的人工审核模板

## 11. 决策建议

当前项目不应选择：

1. 继续把主要资源放在单篇 L2 全面增强
2. 直接做一个开放式“科研问题生成器”
3. 在没有 packet 和历史切片的前提下训练高层科研判断模型

当前项目应选择：

1. 把 `L2` 视为服务于编译器的稳定中间层
2. 尽快进入 `route packet -> RouteState -> DecisionPrior -> DecisionEpisode` 主线
3. 用后续层失败样本反向决定 `L2` 还要补什么

一句话总结：

下一阶段最重要的不是把 `PaperLogicTrace` 继续做成更强的单篇论文表示，而是把它真正接入一个可重放、可时间切片、可审核的 `L3/L4` 编译系统。
