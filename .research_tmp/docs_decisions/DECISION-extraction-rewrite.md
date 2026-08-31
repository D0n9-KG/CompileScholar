# DECISION: 抽取栈重写（B+ 执行序第 5 步）— 联合抽取 + 语义 verifier

日期：2026-08-25。依据：DECISION-2026-08-25-b-plus-rebuild 第 2 节"推倒重写" + 探针判决（PROBE_adoption_result.md 病灶清单）+ 方差判别实验（DETFIX/DETFIX2）。铁律：先 commit 本文档再改码。

## 0. 重写依据（证据链回顾）

| 证据 | 结论 |
|---|---|
| 探针 5 条 outperforms_on 边严格 0/5 | 失效模式 = **绑对 pattern、抓对句子、绑错槽位**（跨句偷换 loser / 绑类别不绑方法 / 限定范围泛化） |
| ablates=0（消融句被 composed_of 抢走） | boundary 纯 prompt 护栏失效 → **需要确定性检查** |
| 边 4 evidence 纯 setup 描述仍 keep | verifier 缺 evidence-支撑检查 |
| 边 3 "comparable to"→outperforms_on | 极性无人校验 |
| 六审 43% 语义基线 + influences 抽查 ~30% 可辩护 | 三步分解架构层缺陷（绑定无局部语境） |

## 1. 设计：联合抽取（joint extraction）

**替换**：`_run_hg_node_multistep` 三步（step1 实体→step2 类型→step3 绑边）→ `_run_hg_node_joint` 单调用。

**单 prompt 输出完整边**（chunk 级，~4k 字符，chunking 不变）：
```
输入: chunk 文本 + schema（_retrieved_schema_prompt，含 boundary）+ planner relation_outline（若有）
输出 JSON:
  nodes: [{nid, surface, type∈{METHOD,PARAMETER,PHENOMENON,REGIME,MATERIAL,NUMERIC,PROPERTY}, evidence_span}]
  hyperedges: [{eid, pattern_type, node_ids, node_roles, evidence_span, qualifiers}]
约束（prompt 显式 + 规则后检双保险）:
  1. 每条边的 evidence_span = 支撑该边的**最小连续原文句**
  2. 每个参与节点的 surface 必须**逐字出现在该边的 evidence_span 内**（绑定局部性——直接杀跨句偷换）
  3. n-ary：一句内的并列参与者进同一条边
  4. pattern 优先从 schema 选；boundary 是判据不是装饰
```

**为什么治槽位绑定**：三步分解里 step3 拿到的是脱离句子的实体清单，绑定靠模型"回忆"哪个实体配哪个槽；联合抽取在**读句子的同时**完成抽取+绑定+类型三件事，且规则 2 把"节点必须在 evidence 句内"变成可机检的硬约束（edge 1 的 loser="reinforcement learning" 不在 evidence 句内 → 规则层直接拒，不浪费 verifier）。

**效率账**：3 调用/chunk → 1 调用/chunk（省 2/3 抽取调用）；chunk 并发 4→8；verifier 仍按 chunk 批量。目标保持 <5min/篇。

## 2. 设计：确定性规则层（新增，规则只管结构/确定性——符合设计铁律）

抽取后、LLM verifier 前，逐边机检（拒掉的不进 verifier，省调用）：
1. **verbatim**（既有）：evidence 是 chunk 的子串（LaTeX/空白归一化后）
2. **binding-locality（新）**：每个节点 surface（归一化后）∈ 该边 evidence_span——治跨句偷换
3. **role-legality（既有）**：roles ⊆ pattern 声明的 role_slots
4. **slot-type guard（新）**：pattern role_slots 声明 type=METHOD 的槽，节点 type 必须 METHOD（或其子类）——治"绑定类别不绑方法"（edge 1 的 reinforcement-learning-是领域的情况若被抽成 PROPERTY 也会被拒）
5. **pattern 竞争护栏（新，有限）**：边界注释 "NOT X" 的 pattern 与 X 同边出现时打标交 verifier 强审（不硬拒——竞争最终靠 LLM 语义判断，但必须被强制看一眼）

拒绝原因记 dropped_edges（沿用 fixer 的记录格式，evolver 继续吃）。

## 3. 设计：语义 verifier 重写

**三必检**（每边，prompt 显式分项问）：
1. **evidence-支撑**：evidence 句是否**陈述**了这个关系（不是暗示/不是 setup 描述/不是相邻句的事实）
2. **槽位正确**：每个节点是否真是它那个角色（winner 真赢了？）
3. **极性/方向**：句子说的是 outperforms 还是 comparable-to？"X 优于 Y"有没有绑反？

- verifier 模型 ≠ 抽取模型（维持 deepseek-chat vs V4-Flash 分工）
- 输入：边 + evidence 句 + ±1 句上下文 + pattern 的 description+boundary+role_slots（逐边给，不给全 schema——聚焦）
- verdict 结构沿用 Verdict（fix: keep/retype/rolefix/reextract/drop）+ 三检各自 bool

## 4. 不动的东西（surgical）

- ExtractionAgent 的 plan-execute-verify-fix-commit 流程结构、KB commit、dropped→evolver 馈送、HGBlackboard predecessor context、_retrieved_schema_prompt、chunking（CHUNK_THRESH=4000）
- planner 不动（其 relation_outline 现在真正进入联合抽取 prompt——比三步时代更近）
- 旧的 `_run_hg_node_multistep` 与 `_run_hg_node` 删除（单管线不留死代码）

## 5. 验收（预注册）

固定样本 = DQN（PPR_24493BE6E8C2）真实块，跑 1 次联合抽取（后续多 seed 是评测阶段的事）：

| 指标 | 门柱 | 基线 |
|---|---|---|
| 语义正确率（judge 交叉 GLM-5-Turbo+qwen3.5，两 judge 均非抽取/verify 模型；分歧我裁） | **≥70%** | 探针宽松 20%（outperforms_on 边） |
| verbatim / binding-locality / role-legality | 全部 100%（规则层保证，报告确认） | — |
| 槽位绑定正确率（重点盯 winner/loser/task/metric 类角色） | 显著高于探针（逐条列样本边+原文对照） | 探针 1/5 |
| 效率 | <5min/篇 + 每边调用数报告 | 4.9min/64 calls |
| 机制不回退 | pattern 采纳仍工作（schema 仍进 prompt） | DETFIX run1 采纳 4 边 |

**逐条对照原文**：验收报告必须列固定样本的每条注入 pattern 边+随机 20 条其他边，给 evidence 原文对照——不许只报数字（用户纪律）。

## 6. 风险与诚实预判

- 联合抽取单 prompt 负担重（实体+类型+边+绑定），可能召回下降 → 验收看总边数与探针/DETFIX 同量级，若大幅掉召回需调 chunk 大小或两遍式（先句级后聚合）——**但两遍式是退路不是首选**
- binding-locality 可能过严（代词回指/省略主语的句子）→ 拒绝时记原因，验收时统计该规则误杀率；>10% 误杀则放宽为"evidence 句±1"
- LLM 方差（DETFIX/DETFIX2 实测同输入差异大）→ 验收单次通过是开发口径，论文数字必须多 seed
