# 华为赛题三 × LogicKG 超图项目适配方案（调研综合）

日期：2026-08-27。依据：赛题文档 + 两份子代理调研（recall_architecture.md /
benchmark_analysis.md，源码级）+ 项目实测数据（FIX1/对齐修复/EVO3）。

## 一、核心问题直答：海量召回怎么办

**不需要自建库，也不该建**。三系统收敛的召回三板斧（查询分解改写+引文滚雪球
+LLM 相关性过滤）全部建立在开放学术 API 上（S2/OpenAlex/arXiv/Google）：
- PaSa：Google `site:arxiv.org` + ar5iv 全文
- SPAR：五源按域路由（S2/OpenAlex/PubMed/ArXiv/Google）
- APF：S2 snippet 语义检索（Vespa，~500 词片段级，覆盖 2 亿+篇）为主干

"海量"由 API 层背。我们的工程选择：**S2 snippet search 为主干（语义检索现成
端点，APF 验证 $0.063/query/30s）+ OpenAlex 引文遍历（免费、覆盖好）**。

## 二、口径判决（比架构更先决）

评测集分两个世界：
- **小 gold**（AutoScholar 2.4 篇/query，集合 F1，gold 外一律 FP）→ 精找宁缺毋滥。
  PaSa recall 0.79 但 F1 0.24 惨死；SPAR 精找 F1 0.38 胜。
- **大 gold**（RealScholarQuery 15.8 篇/query）→ 广撒网+引文扩展。
- PaperFindingBench semantic 类唯一容许 gold 外合理论文（LLM judge+nDCG）。

**华为赛题文本强调"区分高度相关与部分相关""综合排序"→ 大概率大 gold 世界
（或 APF 式 adjusted 口径）**。设计按"广召回+严格精排+输出数量自适应查询类型"。
风险对冲：系统带 query-type 路由（navigational 返 1 篇/semantic 返一批），
两种口径都能应对。

## 三、系统架构（三层，超图卡在两个得分位）

```
L1 召回层（覆盖广度，抄 APF）
  查询分解（LLM 拆子意图+方法论约束+时间/venue 约束）
  → S2 snippet search（k 个改写并行）+ OpenAlex 引文滚雪球（前向+后向，1-2 层）
  → 候选池（~200-500 篇/query）

L2 精排层（超图第一得分位：结构化关联推理重排）
  候选池内论文：
  a) 已在超图子集 → evolution 边（extends/improves/compares/ablates）
     + 方法族（relate 路径勾连）做"为什么相关"的结构化评分，
     与 embedding 相似度互补——分档（高度/部分相关）天然由此出
  b) 图外论文 → 按需增量建图（单篇 ~8min，DeepSeek-V4-Flash ~$0.01/篇）
     只对进 top-50 的候选做，成本可控
  → LLM 相关性判断（APF 式逐子准则 4 档）+ 结构分融合

L3 归纳层（超图第二得分位：结构化输出+可溯源）
  结果按查询意图组织：方法演化链（A extends B，证据边 verbatim）
  / 方法族分组 / 对比矩阵——每篇带 evidence（赛题"结构化 10%"）
  + 超图 evidence 可溯源（我方独有：每边 evidence+cited_from+paper meta）
```

## 四、超图资产 → 得分位对照（含实测底牌）

| 赛题得分位 | 超图资产 | 实测状态 |
|---|---|---|
| F1 精排（高度/部分相关分档） | evolution 边质量=全管线最高（FIX1 75.4%，method_relation 覆盖 80%） | ✅ 够格 |
| 跨文献关联推理（赛题原话） | evolution 边+富拓扑+relate 路径 | relate 未做（1-2 天）|
| 结构化展示 10% | 超图天然关系图+evidence | ✅ |
| 创新性 15%（专家分） | "结构化关联推理重排" vs 三系统全部只用相关分+权威性+时间 | 差异点成立 |
| 效率 20% | 增量建图 ~8min/篇只做 top-50；判断层抄 APF 的 MAB+shortcircuit | 需实测压 |

## 五、与三系统的差异化声明（创新性答辩用）

1. **选择性引文导航**：PaSa 用 RL 学"Expand 哪篇哪节"（382 actions/query）；
   我们用方法演化边直接知道该沿哪条引文链走（结构先验替代 RL）
2. **跨文献关联推理**：三系统 rerank 全部=相关分+权威性+时间的启发式；
   我们加第四维：图结构关联（同方法族/演化链/参数约束匹配）
3. **可溯源输出**：每条相关性判断挂超图边 evidence（PFB 的 markdown_evidence
   思路，但我们的 evidence 在建图时已验证 verbatim）

## 六、诚实的风险清单

| 风险 | 概率 | 对冲 |
|---|---|---|
| 赛题隐藏集是小 gold 口径 | 中 | query-type 路由+输出数量自适应（精找模式） |
| 查询域与已建图（granular/ML）不重叠 | 高 | 增量建图管线现成（这就是为什么地基先行） |
| 增量建图成本吃掉效率分 | 中 | 只建 top-50；单篇 $0.01 级；报成本对比表 |
| 超图重排实测不优于朴素重排 | 中 | **参赛前必补对照实验**（超图重排 vs BM25/embedding 重排，LitSearch 64K 封闭语料现成）|
| S2/OpenAlex API 限流 | 低 | 多 key 轮换+缓存+OpenAlex 免费

## 七、执行序（若参赛）

1. relate 路径（1-2 天）——对齐最后一块，本来就排了
2. 薄 consumer（2-4 天）：S2/OpenAlex 客户端+查询分解+引文扩展+LLM 精排
3. 超图重排对照实验（1 天，LitSearch）——立住差异点才继续投入
4. ML 域 A2 跑通（语料就绪）→ 两域图当测试床
5. AutoScholarQuery test 当练习赛（1000 条+gold 现成，口径最严）

与论文线（WWW2027）的关系：不动摇——参赛系统=论文的 downstream demo 实例化，
评测数字反哺论文的"下游可用性"主张。
