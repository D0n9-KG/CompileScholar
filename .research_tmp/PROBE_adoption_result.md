# 采纳探针结果（PROBE: pattern adoption）— 判读与判决

日期：2026-08-25。设计+预注册判读标准：PROBE_adoption_design.md（commit d9cfc970）。
运行：Arm B（注入 gold pattern）/ Arm C（对照）各 1 run，真实 MinerU 块（PPR_24493BE6E8C2，DQN Nature 2015，143 块），零 monkey-patch。
数据：.research_tmp/runs/kernel_v2/PROBE_B_PPR_24493BE6E8C2/ + PROBE_C_PPR_24493BE6E8C2/（bundle 内 probe_stats.json / all_kept_edges_full.json）。

---

## 一、数字

| 指标 | Arm B（注入） | Arm C（对照） |
|---|---|---|
| kept 边 | 142 | 54 |
| dropped 边 | 14 | 22 |
| outperforms_on 使用 | **5** | 0（schema 无此 pattern，符合预期） |
| ablates 使用 | **0** | 0 |
| 复合斜杠 pattern_type | **0**（修复前前科：81+ 条） | 0 |
| pattern_type 分布 | 13 种干净单一类型 | 9 种 |
| 运行时长 | ~6-7min（进程崩溃前未打印） | 380s（6.3min） |
| KB ledger | 15 entries（144 add_edge + 9 align_concept_merge） | 同构 |

语义审计（子代理逐条对照原文，从严）：
- **5 条 outperforms_on 边：严格 0/5 PASS，宽松 1/5（20%）**——远低于 70% 门柱
  - 2 条绑定错位（loser 绑到领域类别"reinforcement learning"；跨句偷换 loser）
  - 1 条极性夸大（"comparable to"+">75% of human score" ≠ outperforms）
  - 1 条 evidence 不支撑（evidence 是 baseline setup 描述，无比较结果——**verifier 漏放**）
  - 1 条过度泛化（"in all these games"=6 款被泛化为 49 款——唯一方向/极性/verbatim 全对的边）
- evidence 子串验证 5/5 通过（无拼接、无捏造——verbatim 纪律仍然成立）
- influences 抽查 6 条：~1-2 条可辩护（绑定/方向缺陷与审计 43% 基线一致）

## 二、预注册判读：**分支 2 触发**（usage>0 但语义合格率 <70%）

按 PROBE_adoption_design.md 第 5 节预注册表：

> B 臂 usage>0 但语义合格率 <70% → **LLM 能力不够**：看得见也用不好（绑定/方向错）→ in-loop 卖点砍或重定位为 governance；抽取栈重写时 pattern-following 是验收项

**如实触发分支 2**。但审计证据支持比"LLM 能力不够"更精确的失效模式诊断（非门柱移动，是失效模式归因）：

### 判读一（重大正面）：采纳机制活了
- 修复前 0/215（接线断裂 W1：边定型 LLM 调用从未见过 schema）→ 修复后 gold pattern 5 次使用。**"并进"管道在机制层打通——这是项目第一次拿到 pattern 注入→被使用的直接证据。**
- 复合斜杠 pattern_type 从 81+ → 0。schema 可见性+一句消歧 cured 类型污染。
- **W1 是 0/215 的充分根因**（代码级证明 + 修复后行为翻转），审计的"planner 惯性"解释作废。

### 判读二（负面，如期）：槽位绑定系统性弱
- 失效模式高度一致：**"抓对句子、选对 pattern、绑错槽位"**——loser 绑到领域、跨句偷换节点、把限定范围泛化。这是三步分解的设计层缺陷（step3 拿脱离句子的实体清单绑边，绑定无局部语境），非 LLM 不会跟随 schema。
- **boundary 护栏失效**：ablates=0 的根因是消融句被 composed_of 抢走——boundary 里白纸黑字写 "NOT composed_of" 仍挡不住。pattern 竞争无确定性检查，纯 prompt 层护栏不够。
- **verifier 漏放**：边 4 的 evidence 不支撑关系，verifier 判 keep——语义校验（方向/极性/evidence-支撑）缺位，如审计所判。

### 判读三（对照）：注入带来结构增益非重标注
- C 里同类事实：1 条弱捕获为 compares、评测句错打为 influences（评测过程≠功能依赖，C 自身语义错）、部分缺失。
- B 的 outperforms_on 是 4 元结构（winner/loser/task/metric）——n-ary 富结构 vs C 的 2 元错型。
- ⚠️ 方差警示：142 vs 54 部分来自运行方差（C 的 n6/c1 chunk step1 解析 0 节点 vs B 的 74 节点；审计已知同状态 Jaccard 0.245）。evidence 重叠 29/54。总量对比不作论文证据，方向性结论（结构增益）可信。

## 三、对卖点的影响（诚实版）

1. **in-loop 卖点未死也未复活——被重写接住**。机制层管道已打通（判读一），语义质量门在抽取栈重写（预注册分支 2 的行动项原文："抽取栈重写时 pattern-following 是验收项"）。若重写后 judge 交叉语义 ≥70%（含 pattern-following + 槽位绑定），A2 跨篇判决性实验按 DECISION 执行序进行；若达不到，按分支 2 预案砍或转 governance。
2. **重写验收标准加两条**（从探针证据直接导出）：① 槽位绑定正确率（探针失效模式）② pattern 竞争的确定性护栏（boundary 声明需有规则层检查，如 type 不符即拒——"NOT composed_of"不能只活在 prompt 里）。
3. **verifier 语义校验缺口确认**（边 4 漏放）——重写的语义 verifier 设计里方向/极性/evidence-支撑三项都是必检。
4. 探针数据进 bundle 留档，作重写前基线：宽松口径 20% → 重写后同口径重测（门柱 70% 不变）。

## 四、方法学教训（记 memory）

- 六路审计漏掉 W1（"接线完好"结论错误）——因为它只验了 execute() 构建 schema_prompt，没追进 multistep 核心函数体。**接线审计必须追到参数消费点。**
- 预注册判读这次真正起效：没有预注册，5 次 usage 很容易被当成"卖点复活"的好消息报出去。

## 五、遗留给确定性修复/重写阶段的具体病灶清单

| 病灶 | 证据 | 修法方向 |
|---|---|---|
| step3 绑定无局部语境 | 边 1/2/5 绑错槽位 | 联合抽取（边与实体同 prompt 同句绑定）——DECISION 既定 |
| boundary 纯 prompt 护栏 | ablates 被 composed_of 抢 | 规则层 pattern 竞争检查（type/结构不符即拒） |
| verifier 不查 evidence-支撑 | 边 4 漏放 | 语义 verifier 三必检：方向/极性/evidence-支撑 |
| 运行间方差 | n6/c1 解析 0 vs 74 节点 | 确定性切块+重试（既定）；多 seed 报告 |
| 单篇 6.3min > 5min 目标 | Arm C 380s | 重写效率项（联合抽取省调用+并发加深） |
