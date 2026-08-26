# 对齐层修复报告（2026-08-27，接 align_first_test 审计）

代码：commit（本次）。考卷：`.research_tmp/align_regression/`（run_exam.py，在 A2G_EVO2
10 篇累积图 544 概念上全量重放 align，7 个审计家族作判据）。

## 根因（4 个，前 3 个是审计预判，第 4 个考卷调试中偶然发现）

1. **批次切片结构性隔离（主因）**：`align()` 按 dict 插入序切 20/批——"kinetic
   theory"(#52) 与 "kinetic theory formula"(#97) **永不在同一 LLM 调用里**。merge
   通过率 3-9% 主要是 batching 伪影，不是判得严。干净基线实证：全量重放 **0 合并**。
2. **definition 不进对齐 prompt**：LLM 只见 surface+symbol，无法分辨跨篇单字母碰撞
   （f=频率 vs f=摩擦系数）。
3. **噪声 surface 无闸**：Fig.3/"model"/"theory" 等照常参与对齐（泛称吞噬原料）。
4. **PROPERTY/REGIME 不在 type_filter**：113 个 PROPERTY + 19 个 REGIME 概念生产上
   从不参与对齐（audit 的 steady_state/granular_temperature MISS 的根因）。

## 修复（4 处，规则/LLM 边界合规）

| 修复 | 层 | 性质 |
|---|---|---|
| `_precluster_batches`：token-Jaccard≥0.35 连通分量预聚类，同概念变体保证同批；solo 概念跳过（候选 324→113，效率 2.9 倍） | alignment_agent | 确定性结构 |
| 对齐 prompt 传 definition + 判据 5 条（变体必并/单字母需定义确证/泛指不吞并/同头词不同机制不并/存疑不并） | concept_graph | LLM 语义 |
| `_is_noise_surface` 闸：图表/公式引用+封闭泛称表跳过对齐（概念保留，边不断链） | alignment_agent | 确定性结构 |
| 纯符号组合并闸：组内全部 surface 无 ≥4 字符英文词=纯数学面→拒绝（字母相同不是证据；命名面必含长词，零误伤）；`_surface_tokens` 复数归一+停用词滤除 | alignment_agent | 确定性结构 |
| type_filter 加 PROPERTY/REGIME | alignment_agent | 范围修正 |

## 考卷判决（基线 vs 修后）

| 家族 | 基线 | 修后 |
|---|---|---|
| kinetic theory (14 成员) | 14 distinct | **5 distinct**（余 5 个语义可辩：Chapman-Enskog 具体 KT/binary mixtures 变体等） |
| Chapman-Enskog 篇内 (6) | 6 | **1** |
| restitution (3) | 3 | **1** |
| molecular dynamics (2) | 2 | **1** |
| inertial number (2) | 2 | **1** |
| steady state (2) | 2 | **1**（REGIME 入 filter 后） |
| granular temperature (4) | 4 | 4（判读：3 个是语义真不同概念拒合正确，1 个单复数边缘） |
| **总合并数** | **0** | **34**（符号碰撞 0 例，泛称吞噬 0 例，抽检干净） |

## 生产影响与诚实边界

- 修复对未来 run 生效；**盘上 A2 图仍是碎片态**——需要重跑 A2 或做 retro-align
  迁移（考卷已证明 retro-pass 可行）才能让下游用上
- 考卷重放里 definitions 多为空（define 阶段未跑）；生产 align_new 先 define 后
  align，单字母判据会更稳
- 判据 4（同头词不同机制不并）在考卷里未直接考（density/size segregation 已在
  数据里错误合并，无法在重放中拆开）——A2 重跑后用 analyze.py 复测
- 预聚类阈值 0.35/停用词表是结构归一化，但家族键粗细影响考卷读数（granular
  temperature 家族键含 3 个真不同概念）

## 下一步

多 seed × SOFT_ROUTING 两态（已含本修复）时 A2 重跑即自带对齐修复验证：跑完用
`align_first_test/analyze.py` 复测合并率（目标：跨篇合并率 5.5%→显著上升且坏合并
不升）。

---

## EVO3 生产复测（2026-08-27 追加，10 篇 A2 重跑对齐修复版）

| 指标 | EVO2（修前） | EVO3（修后） | 判 |
|---|---|---|---|
| 篇内 METHOD 碎片对（J≥0.6） | 12 | **3**（均语义可辩边缘） | 修复生效 ✓ |
| 跨篇 METHOD 嫌疑对 | 3 | **1** | ✓ |
| 泛称吞噬（model+DEM 型） | 有 | 0 | 噪声闸生效 ✓ |
| 裸符号跨篇互撞 | 有 | 0（残余 T/T- 为下标变体） | 纯符号闸生效 ✓ |
| 跨篇合并率 | 5.5% | 6.2% | 微升（见判读） |

**判读**：修复兑现的是"该合的准合+不该合的不乱合"（碎片 12→3/坏合并清零）。
跨篇合并率未起飞**不是回归**：生产环境 definitions 在场，LLM 把 "kinetic theory
formula"（KT 衍生的具体表达式）判为独立概念语义可辩护——考卷 30% 是无定义环境的
宽松判读。**下游"查 kinetic theory 勾连全家族"的需求应由 relate 路径承担**
（ADD_CONCEPT_RELATION：相关不合并），当前考卷/生产都未用此 op——下一层对齐
设计工作，未做。

EVO3 全程 79 次真合并（merge→del concept 落地验证），Chapman-Enskog 6 碎片在第 1
篇内即合并（CM0057）。
