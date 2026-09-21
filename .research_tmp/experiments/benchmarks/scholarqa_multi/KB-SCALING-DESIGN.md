# Multi-431 建库的 registry/vocab 规模化设计（v0.1 讨论稿，2026-09-21）

> 状态：**待用户裁**。skeleton（按篇线性）不受影响先行；registry/vocab/round2 三处单调用设计在 431 篇规模确认不可行，需改造后才进建库。

## 1. 问题（实测外推，非猜测）

建库管线三处依赖"一次 LLM 调用合并全量表面名"的设计：

| 环节 | 设计 | PS 时代规模 | Multi-431 外推 | 结论 |
|---|---|---|---|---|
| registry.merge_entities | 单调用，MAX_SURFACES_ONE_CALL=600 | AirQA 25篇→1095实体/1178表面名；PSr2 16篇→668/870 | ~8k 表面名输入（卡片 related+identity ~20/篇），输出需 30k+ tokens 索引分组 | **超限 13×，物理不可行**（输入 ~200k+ tokens） |
| registry.build_vocab | 每维度单调用，max_tokens=12000 | 16-53 篇，候选数百 | subject/setup/variant 各 1.5k-4k 候选（QLoRA 单篇 54 个候选） | 输出截断高风险 → coverage fallback 大量 singleton → 家族碎片化 |
| registry_round2 | queue→registry 单调用映射 | AirQA 增量 3005（已有记录） | queue 估 5k-15k 表面名 + canonical 清单 15k-19k 实体 | 同上，双侧都超 |

**为什么 PS-53 没暴露**：build_ps53.py 复用了 16 篇时代的 registry_v32_pilot.json，从未在放大规模上跑过 merge 路径。这正是"不要把现有实现当稳定最佳版本"点名的那类问题。

**截断的后果（已被防呆设计确认）**：_covered_groups 会把丢失索引兜底成 singleton——不崩，但实体碎片化=round2 教训（OFF 时链接率 6.3%）的放大版，typed tools 可用性头号短板复发。

## 2. 方案 A（推荐）：嵌入分块 + 块内 LLM 合并 + 跨块 canonical 复并

基础设施现成：本地 qwen3-embedding-8b 与 27B 同机（.env EMBEDDING_MODEL 已配）。

```
Step 1  collect_mentions（不动，确定性 norm 合并）
Step 2  嵌入分块（确定性）：全部表面名 → qwen3-embedding-8b 向量
        → cosine ≥ τ_high 连通分量成块（保守高阈值：宁漏同块，不错并块）
        → 块 >400 表面名的按子聚类再切
Step 3  块内 LLM 合并：现有 MERGE_PROMPT + _covered_groups + F34 '+'守卫
        原样复用，每块一次调用（本地零成本，块间可并行）
Step 4  跨块复并：全部块 canonical（~3-6k）再嵌入聚类
        → canonical 块上 LLM 复并（输入含 canonical+aliases 摘要）
        → 迭代至跨块合并数收敛（预期 1-2 轮）
Step 5  确定性收尾（不动）：own-paper 类型覆盖 / origin_year 投票 / surface_index
```

**对"批次切片隔离"教训（08-26 对齐修复根因）的防御**：
- 切片=嵌入语义块，非随机批切——同实体的缩写/全称/变体嵌入距离近，大概率同块；
- 跨块复并 pass 专门捞跨块别名（单调用设计的核心收益在 canonical 层保留）；
- QC 仪表：块大小分布 / 块内合并率 / **跨块合并数**（=复并 pass 有效性证据）/ singleton 比例；
- 人工闸门：top-100 高频实体合并结果抽读。

**round2 同型改造**：queue 表面名嵌入 vs registry canonical 嵌入 → 每个 queue 表面名只与 top-k 近邻 canonical 组成小候选块 → LLM match/new 判定在块内做（输出空间小，coverage check 保留）。

## 3. vocab 方案（D-VOCAB）：按 subject 域分批 + 跨域 family 复并

- manifest 自带 subject（bio 74 / cs_nlp 136 / biophysics 50 / photonics 137 / physics 21）——按域切批是**语义连贯切分**（family 天然域内；跨域共享基准极少）；
- 单域候选仍超输出容量时，按频次二分（n≥2 头部批 / n=1 长尾批；长尾批 family 判定输出 ~5 tokens/项，1500 项 ≈ 7.5k tokens 可行）；
- 跨域 family 名复并 pass（family 名数百级，单调用足够）；
- **BENCHMARK_BLOCKLIST 扩域**（gold-blind 纪律不动，题面永不进 prompt）：从 vocab subject 维度 + registry entities 按域提取 benchmark/dataset 名候选 → LLM 提案 → **人工审** → 扩入 gate 侧名单。

## 4. 备选方案（不推荐的理由）

- **B 纯增量 grow**：按提及数排序分 chunk，每 chunk 带全部已有 canonical 清单调用——后期 canonical 清单 15k+ 实体本身就超 context，且调用次数 × 大 prompt 更慢；
- **C 硬提单调用上限**：8k 表面名 ≈ 200k+ tokens 输入，超过任何可用模型窗口，物理排除。

## 5. 成本与工期

- 嵌入 8k 表面名：分钟级（本地）；
- LLM 调用：round1 ~25-40 块 + 复并 ~10-15 次；round2 ~15-25 次；vocab ~10-15 次——全本地零 API 成本，墙钟估 <1h；
- 实现工期：merge_entities 分块改造 + round2 候选块改造 + vocab 域分批，估 1 天内（复用现有 prompt/守卫/coverage 机制，只改调用组织层）。

## 6. 验收门（实现后、建库前）

1. 单元：合成表面名集（含已知别名对跨块分布）→ 合并召回率 ≥ 单调用基线（小规模可对照真单调用）；
2. 活体：431 篇真实数据跑 round1 → QC 仪表全记录 → top-100 人工抽读（错并/漏并各计）；
3. 对照 PS 遗产：16 篇 PaperScope 子集上分块结果 vs registry_v32_pilot.json 实体对齐率（回归门）。
