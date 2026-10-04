# Split 操作语义纯度验证 — 诚实负面结论 (2026-08-14)

## 核心发现
**split 操作集（核心创新卖点）在 ARFM2024 5篇试水上，没有通过语义纯度验证。**
无论用什么评测角度，full（split）的 silhouette 都没超过 frozen/add_only。

## 完整数据（aligned+公平，排除小簇统计偏差）

| arm | n_pat_real | role+qualifier sil | entity-text sil |
|-----|-----------|-------------------|-----------------|
| full | 10 | 0.1057 | -0.0328 |
| add_only | 6 | 0.1630 | 0.0124 |
| frozen | 6 | 0.1585 | 0.0026 |
| no_intra_dag | 22 | 0.0342 | -0.0266 |

两个特征空间都不利 split。frozen/add_only 在 role+qualifier 空间天然高（大 pattern 装"X影响Y"边，role 一致），split 拆散后纯度降。

## 已排除的可能解释（都验证过）
1. ✗ 命名跨篇不收敛 — 已修(commit 3ce0f502)，Aguilar系列复用，但修复后full更低(-0.0242→-0.1650)
2. ✗ 小簇统计偏差 — 已排除(只算≥5边pattern)，趋势不变
3. ✗ 评测特征错配 — role+qualifier 和 entity-text(split自己用的)两个空间都试，都不利full
4. ✗ 科普文噪声 — 排除科普文后趋势不变
5. ✗ 指标实现bug — 重写clean silhouette，结果一致

## 根因分析
split 的 _llm_semantic_clusters 让 LLM 按"物理量/关系语义"分组(看 role:surface+evidence)，
把语义相近的边按物理子领域拆开(intrusion影响 vs locomotion影响 vs flow影响)。
这些边在结构上相同(source→target 无qualifier)，LLM按实体语义拆，
但拆出的子pattern在统计上不比原来的大pattern更纯——因为:
- 同物理领域内边仍多样(intrusion内各种force/velocity/material)
- 子pattern小，统计不稳

frozen的大pattern天然聚: 所有"X影响Y"边role一致+qualifier空→role+qualifier空间天然最近。

## ★ 更深的认识(2026-08-14, 修正"split有害"结论)
查看full实际influences分布: 299条influences未被split(占89%), 仅37条拆成子类小簇。
即split触发不足+拆出小簇碎片化, 不是"拆错维度"。

**silhouette指标与自演化价值方向相反**:
- silhouette测"同pattern边相似/异pattern边不同"→系统惩罚小簇(新发现)
- 自演化schema价值在"发现新结构"(add新pattern)+细化过宽(split)
- 新add的pattern(如derives_from)可能就2条边→silhouette必然低,但代表新发现
- silhouette系统性奖励"保守不拆"(frozen大簇)→与演化目标对立

**结论修正**: silhouette不适合评测自演化schema质量。full低silhouette部分是split触发不足+小簇碎片,
部分是指标本身惩罚演化方向。需换评测角度(如下游任务/方法角色消歧/综述重建),
而非继续调silhouette。

split是否真有效, 取决于: 拆出的子pattern是否对下游有用(角色消歧/检索), 不是统计纯度。

## 诚实结论（不粉饰）
1. pilot流程已过(抽取+gold+匹配+指标全跑通)，但**核心创新卖点split未证实价值**
2. split在5篇颗粒流试水上损害而非改善语义纯度
3. 这不是工程bug(命名已修)，是split聚类判据+评测的深层问题
4. 当前不能宣称"形式化操作集(五操作)提升schema质量"

## 可能的下一步（需和你商量）
A. 修split聚类判据: 不让LLM按物理领域拆(那拆太碎), 改按限定符/角色结构拆(更结构化)
B. 换验证场景: 5篇太小+科普文噪声, 扩到全147篇看split是否在足够样本上收敛有效
C. 重新审视创新定位: split可能不是有效卖点, 主卖点转向intra-DAG传播或nary超图
   (但intra-DAG acc信号也复杂: full acc=109但no_intra_dag acc=30, 传播效果待查)
D. 换评测: silhouette对"按语义拆分"不敏感, 用下游任务(方法角色消歧/综述重建)验证

我的判断: 先B(扩样本)排除"5篇太少"假说, 同时A(修split判据)。
如果扩到全147篇split仍不改善纯度, 则C(重新定位)是必须的诚实选择。
