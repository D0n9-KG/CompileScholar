# 调研：分层证据获取 + Agent 预算平衡（2026-09-29）

来源：调研代理一手验证（arXiv ID 全部核实；WebSearch 污染弃用）。
与 RESEARCH-KB-ASSISTED-RETRIEVAL-0929.md 互补。

## 方向 A 判决

**A1 级联触发器四族**：前置分类（Adaptive-RAG 2403.14403）/ 内部不确定性（Self-RAG 2310.11511, FLARE 2305.06983, DRAGIN 2403.10081）/ 证据质量评估（CRAG 2401.15884, Self-Route 2407.16833, BalanceRAG 2605.20084）/ 学习策略（FrugalRAG 2507.07634, CoRAG 2501.14342）。
**与我们三层证据最同构 = 证据质量评估族（CRAG）**。
关键负面教训：When to Retrieve (2404.19705) 实证 **prompt 自评系统性不可靠**（模型心里知道但嘴上不说）——自评阈值必须用判分结果回标定。

**A2 片段 vs 全文**：返回相关片段是文献明确主流——Lost in the Middle (2307.03172) U 型利用；4K+检索≈16K 全文 (2310.03025)；**~2.5k token 出现 context cliff (2601.14123)**。deep_read 2-5 定向 chunk 设计正确。强化三点：chunk 自包含（Dense X 2312.06648 + LongRAG 2410.18050——带 section 语境头）；回传 ≤2.5k；Self-Route 式兜底（不足才升全文，全文是 fallback 不是默认）。

**A3 按需触发**：Sufficient Context (2411.06037) "上下文是否够"分类器与三层证据最对齐；S2G-RAG (2604.23783) 充分性输出**结构化缺口项**→缺口直接变下一轮查询（停止信号=检索规划）；SeaKR (2406.19215) 预期不确定性消减=deep_read 候选排序信号。

## 方向 B 判决

**B1 预算感知**：**BATS (2511.17006) 几乎逐字回答我们的问题**——Budget Tracker 每步播报剩余预算→agent 动态调整→更优 Pareto（Google Research 开源）。s1 budget forcing (2501.19393) = "预算将尽强制转写作"的最直接工程解。TALE (2412.18547) 预算写 prompt 即 -67% token 掉点<3%；BAGEN (2606.00198) 前沿模型系统性过度乐观——rollout-replay 可诊断我们第几步失控。AgentOccam (2410.13825) 剃刀式简化反超——审计仪式性动作。EVAR (2608.29835) claim 槽位填满即转写作。

**B2 早停四族**：证据覆盖自评 / **答案稳定性（实证性价比最高——语义早停 2606.27009 零 LLM 开销省 38%）** / 边际收益递减（EVPI 理论底座 2606.07071） / 置信门控。Multi-Round RAG Stop (2608.13237) 纪律：阈值独立验证集选定冻结+诚实报"省多少掉多少"。

**B3 IR 理论骨架**：级联结构（Cascade Ranking SIGIR'11/'17, RankFlow'22——30 步=多级漏斗，广度×深度两旋钮**联合**调）+ 信息觅食理论（Pirolli & Card 1999——单源边际收益<环境平均即切换，"读透这篇 vs 换下一篇"的规范化判据）。

## 三条可落地建议（调研代理原文精炼）

### 建议 1：升级触发 = 逐 claim 证据充分性判断（不是问模型自信不自信）
问"当前 claim 是否被编译记录的 quote 直接支持"——充分→锁死；不充分→输出结构化缺口项→缺口映射为 deep_read 目标篇目+段落（FLARE claim 改写查询 + DRAGIN 何时/读什么解耦）。**永远不做整篇下潜**。阈值按最终答案错误率联合调（BalanceRAG），用判分管线回收错配样本回标定。两层都不够→typed unknown（弃权档有先例）。

### 建议 2：预算治理三层栈
(1) 第 1 步复杂度分诊（Adaptive-RAG 式，标签用各层实际答对率自动造）——简单题 3-5 步，30 步是上限不是定额；
(2) 每步注入 Budget Tracker："剩余步数 X/30、已 deep_read N 篇、剩余可承受 M 篇"（BATS 验证）——充裕期鼓励并行广度，信息觅食判据切换深度；
(3) 26/30 注入"停止探索立即作答"，28/30 强制只留写作工具（s1 budget forcing，harness 层零模型依赖）。

### 建议 3：deep_read 维持片段式，补三点工程强化
(1) chunk 自包含（section 语境头）；(2) 回传 ≤2.5k token；(3) Self-Route 兜底。
进阶：FrugalRAG（deep_read 次数进 RL 目标）/ 语义早停（每 k 步草稿比 embedding 稳定性，零开销省 38%，保留全部草稿终选）。
