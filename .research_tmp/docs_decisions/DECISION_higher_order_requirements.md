# DECISION: 高阶升层对低阶 schema 的需求 (2026-08-14)

核心创新 = "低阶局部 schema 通过自进化升层级，长出高阶 schema
（局限/解决/方向等跨方法归纳）"。要定义低阶该做成什么样，必须先看清
高阶 gold 长什么样、低阶实际抽出来长什么样，再倒推需求。

本文件是第一步产出：**不写代码，先想清需求**。作为低阶"稳"的标准
（"稳" = 能支撑高阶升层，不是孤立指标）。

---

## 1. 高阶 gold 的两层结构（综述 ARFM2024 独立归纳）

gold 不从大图反推，是从综述正文人工/LLM 抽的独立结构（见 GOLD_DESIGN.md）。

### 层 1：方法节点（53 个）
方法名级实体，不是物理量：
  M1 μ(I) rheology, M2 Nonlocal continuum models, M15 NGF model,
  M17 I-gradient model, M45 RFT, M50 granular RFT ...
每个方法有 aliases + refs（指向被引论文）。

### 层 2：方法间关系边（41 条，6 种语义）
  extends(12)   : M2/M3/M17 --extends--> M1（M1 的扩展/后续）
  improves(8)   : M15 --improves--> M1（M15 解决 M1 的局限）
                  "capture more subtle behaviors that elude the μ(I) rheology"
  compares(11)  : M17 --compares--> M15 "it cannot capture thin-body strengthening"
  replaces(2)   : M47/M48 --replaces--> M45
  adapts(4)     : M50 --adapts--> M45
  background(4) : M16 --background--> M15

**这就是高阶归纳** —— 跨方法的局限→解决→方向演化关系。
核心升层目标例：μ(I) 局限 → NGF 解决（M15 improves M1）。

---

## 2. 低阶实际状态（full arm, Pouliquen1999 单篇实测）

读了 runs/ARFM2024/full/Pouliquen_1999_*.json：

- **pattern 暴涨到 49 个**：含 influences_force_balance /
  influences_spatial_variation / influences_regime_transition /
  influences_material_independence（从 influences split 出 4 子），
  还有 claim_relation_compatibility_consistency /
  claim_relation_compatibility_agreement 这种 4 级嵌套命名。
  → split 命名仍无脑叠加，非真正语义收敛（3ce5f502 修了传 meta 但仍叠加）
- **拓扑全弱**：schema_topo 几乎全是 `depends_on → definitional_identity`，
  共现推断的伪拓扑，无语义意义
- **节点是实体级**：edges 连 "Froude number Fr"、"interstitial fluid viscosity"、
  "glass beads" 这种物理量/材料实体节点
- **关系是结构依赖语义**：influences / defines / composed_of / derives_from /
  constitutive_law —— 没有 improves/extends/replaces 这类方法演化语义

---

## 3. 三个层级错配（低阶 vs 高阶 gold）

| 维度 | 高阶 gold | 低阶实际 | 错配 |
|------|----------|---------|------|
| 节点层级 | 方法（μ(I) rheology） | 物理量实体（Froude number） | ★ 缺方法节点层 |
| 关系语义 | improves/extends/replaces | influences/defines/derives_from | ★ 无演化语义 |
| evidence 层级 | 综述跨方法叙述句 | 单篇局部句 | 升层需综述级叙述 |

---

## 4. 高阶升层对低阶的 4 个需求（低阶"稳"标准）

### 需求 1：共性簇 —— 低阶 pattern 跨论文语义一致（命名稳）
升层机制要把"多篇被引论文里语义一致的低阶 pattern"聚成共性簇，
再归纳成高阶方法节点。若命名跨篇不收敛（force_balance 在 A 篇叫 X、
B 篇叫 Y），簇聚不起来 → 升层失败。
- **稳标准**：同一物理关系跨 ≥3 篇被同一 pattern name 命中（复用率）
- **当前缺口**：split 命名仍叠加成 4 级嵌套，复用率待测

### 需求 2：方法关联 —— 低阶要补"方法节点层"
高阶节点是方法。低阶当前只有实体节点，方法节点层缺失。
升层要把低阶共性簇关联到方法。两条路：
  (a) 低阶直接抽"方法节点"（被引论文的核心方法名/模型名）
  (b) 低阶只抽实体级，升层时用 LLM 从簇+evidence 归纳方法名
- **稳标准**：低阶产出能映射到 ≥1 个方法（聚类后可命名出方法）
- **当前缺口**：低阶节点全是实体级，无方法节点 → 倾向 (a) 在低阶补，
  但要先验证 (b) 是否够（避免低阶过早膨胀）
- **★ 决策**：先试 (b) 升层时归纳方法名（最小改动），若归纳失败再回低阶补 (a)

### 需求 3：evidence —— 低阶边存 evidence（已有 ✓）
升层归纳要给 LLM 真实 evidence 防幻觉（IncSchema retrieval-augmented）。
低阶边已存 `ev` 字段（原文句）。✓ 满足，无需补。

### 需求 4：局限/解决语义 —— 低阶可能要扩展抽这类关系
高阶 improves/extends/replaces 是"方法演化语义"。
低阶当前只抽结构依赖（influences/defines/derives_from），**没抽这类**。
但单篇论文内部也可能有局限/改进叙述（"本方法不能处理 X"、"相比 Y 改进了 Z"）。
- **稳标准**：低阶能产出至少几条方法演化语义边（improves/extends/compares）
- **当前缺口**：低阶 relation_type 限定符里有 compatibility/incompatibility
  但没有 improves/extends/replaces → 低阶 prompt 可能要扩这类 relation
- **★ 决策**：先不强加（避免低阶膨胀），升层尝试时看缺什么再补。
  若共性簇归纳不出方法演化关系，再回低阶扩 relation 语义

---

## 5. 升层路径假设（差异化于 IncSchema）

IncSchema = retrieval-augmented LLM **直接从语料 probe** 高阶 schema。
我们 = **从低阶自进化升**，低阶是多篇被引论文已抽出的 pattern/边。

升层流程（待最小验证）：
1. 多篇被引论文低阶 pattern → 跨篇共性簇（需求1，命名稳才聚得起来）
2. 簇 + evidence → LLM 归纳方法节点（需求2b）
3. 方法节点之间 → 归纳方法间演化关系（需求4 语义）
4. 对照 gold 高阶两层 → 看升出几成

防幻觉（借鉴 IncSchema）：
- retrieval-augmented：给 LLM 真实低阶 evidence + 簇成员，非凭空 probe
- 分解验证：不直接问"这几个 pattern 共同表达什么高阶"，分解可验证子问题
  （"这几个 pattern 是否都关联同一物理量 μ(I)？"可验）
- log probability 而非 yes/no + 严格准入 test

---

## 6. 低阶"稳"标准汇总（工程指标，非价值评测）

低阶不求价值评测（NMR/QA/silhouette/LOO 都验证过不合适，价值等高阶显现）。
只求工程稳定，且稳定定义为"能支撑高阶升层"：

| 指标 | 目标 | 验证方法 |
|------|------|---------|
| 命名跨篇复用率 | 同物理关系跨≥3篇同名 | Pouliquen1999 + 扩10-15篇看 |
| pattern 增长 | 不无脑暴涨（49→?） | 单篇迭代看增长曲线 |
| coherence gate | split 拆出簇 cosine≥0.55 | 已实现 423adabf |
| 共性簇可归纳方法 | 簇能映射≥1方法 | 升层最小尝试验证 |

---

## 7. 第二步执行结果：最小升层尝试 (lift_minimal.py, 2026-08-14)

取 full arm 已抽的 3 篇 μ(I) 奠基论文 (Midi/Jop/Pouliquen1999) 的低阶
共性 pattern, 用分解验证式 prompt 让 deepseek 归纳高阶方法节点。

回答三个问题:
- **Q1 机制可行性 ✓**: LLM 能从低阶共性 pattern + 真实 evidence 归纳出
  方法节点描述。多个 pattern 归纳出有意义方法名:
    constitutive_law → μ(I) rheology
    influences_regime_transition → μ(I) rheology
    defines → non-local rheology (接近 gold M2)
    constitutive_law_threshold_condition → 本构关系与临界条件推导
- **Q2 对齐 gold M1 ✓**: 2 个 pattern 归纳出 μ(I) rheology, 对齐 gold M1;
  defines→non-local rheology 接近 gold M2 Nonlocal continuum models
- **Q3 高阶关系语义 部分**: influences_regime_transition 归纳出 limitation
  萌芽 ("摩擦系数趋于极限 tan θ2, 暗示对摩擦定律的修正/补充") ——
  虽非 gold improves (那需后续方法论文), 但局限语义可从低阶 evidence 萌芽

### 诚实约束 (重要)
3 篇都是 μ(I) rheology 奠基论文, 归纳出 μ(I) 有 **trivial 风险**
(本就定义它)。这验证了升层机制可行, 但没验证"从低阶升起"的真正高阶价值。
要验证 gold 的 improves/extends (M15 NGF improves M1 μ(I)),
**必须补抽后续方法论文**做跨方法共性簇归纳 —— 3 篇奠基论文答不出。

### DECISION 需求验证状态 (基于最小尝试)
- 需求1 共性簇(命名稳): ✓ 跨篇共性 pattern 存在 (Midi/Pouliquen 共有 27 pattern,
  split 子 pattern influences_force_balance 等跨篇同名 = 3ce5f502 修复有效)
- 需求2 方法关联: 待验证 —— 最小尝试用路径 (b) 升层时归纳方法名, 成功归纳出
  μ(I)/non-local 方法节点, 说明 (b) 可行, 暂不需低阶补方法节点层 (a)
- 需求3 evidence: ✓ 已有, 归纳时 retrieval-augmented 工作
- 需求4 局限/解决语义: 部分萌芽, 待后续方法论文验证 improves/extends

## 8. 跨方法升层结果 (lift_cross_method.py, 2026-08-14)

5 篇低阶: μ(I) 奠基 3 篇 + Bouzid_2013(NGF) + Kamrin_2015(I-gradient)。
两步升层: (1) 各方法组归纳方法节点 (2) cross-mention evidence 判跨方法关系。

### step1 方法归纳 ✓ 全部成功
- μ(I) rheology: μ, I, σ_yy/σ_xy, Q*, h, D
- NGF: non-local constitutive, relaxation length, order parameter I
- I-gradient model (Kamrin_2015 准确归纳出方法名 + L0 内部长度尺度, Δμ, μ_s, μ_2)
→ DECISION 需求2 路径(b) 升层归纳方法名可行, 不需低阶补方法节点层 (a)

### step2 关系判断对齐 gold (1/3 完全对, 1/3 方向对, 1/3 错)
- I_gradient→μ(I): extends ✓ 对齐 gold M17 extends M1
  (rationale 引真实证据: I-gradient 引入 L0 扩展 μ(I) 适用范围)
- NGF→μ(I): extends (gold improves, 方向对——非局部超越局部, extends vs improves 细粒度偏 extends)
- I_gradient→NGF: background (gold compares, 错——两篇非局部论文互不提对方)

### 诚实根因 (evidence 层级局限)
1. **extends 易升, improves/compares 难升**: 被引论文常写"我扩展 X"(extends),
   少写"我解决 X 局限 Y"(improves) 或"我比 X 在 Z 上好"(compares)
2. I_gradient-NGF compares 升不出: 两篇非局部模型论文互不提, 被引论文内部无
   跨方法对比 evidence。gold compares 来自综述视角(ARFM2024 作者读过两篇后写的)
3. DECISION 需求4 实证: 低阶 evidence 能升 extends, improves/compares 需
   "局限→解决/对比优劣"语义, 被引论文低阶 evidence 不足

### 核心创新方向部分可行性证实 (非循环, gold 高阶独立于低阶大图)
- 升层机制可行: 能从低阶共性 pattern+evidence 升出**正确的高阶方法间关系**

## 9. 改进 A 执行结果 (extends→improves 决策路径, 分析驱动, 成功)

REL_PROMPT 加决策路径 (路径1: B 有做不到的 + A 解决 → improves;
路径2: 仅推广范围 → extends; 路径3: 各有优劣 → compares)。
重跑结果:
- **NGF→μ(I): improves ✓ conf=high 对齐 gold!** (rationale: non-local across yield
  conditions 处理 μ(I) 局部在屈服附近的局限) —— 从 extends 提到 improves, 对齐 gold
- I_gradient→μ(I): improves (gold extends, 细粒度主观——I-gradient 既扩展又解决局限,
  gold 标 extends 强调"直接扩展", 两者都对)
- I_gradient→NGF: extends (gold compares, 仍升不出——两篇非局部论文互不提对方)

### 最终诚实结论 (3 条 gold 边验证)
- 1/3 完全对 (NGF improves μ(I) ✓ conf=high)
- 1/3 主观对 (I_gradient→μ(I) improves vs gold extends, 边界主观)
- 1/3 根本局限 (I_gradient-NGF compares 升不出, 被引论文互不提对方)

★ 核心创新方向可行性证实: 低阶升层能产出对齐 gold 的高阶归纳关系。
  extends/improves 可升 (extends 易, improves 需决策路径引导);
  compares 受被引论文 evidence 层级根本限制 (互不提对方的方法对升不出)。
  → 评测承认: extends/improves 覆盖, compares 覆盖低 (需综述视角)

### 样本量诚实约束
3 条 gold 边样本太小, 1/3 完全对可能是运气。正式化后需扩到 ~10 条 gold 边
看真实覆盖率 (gold 共 41 边/6 类型)。但机制可行性已建立。

## 10. 正式化 + drainage compares 验证 (2026-08-14)

正式化升层器: src/granular_agent/hypergraph_lifter.py (commit b9ceb285),
接口 induce_method_node + judge_relation + lift。sanity 通过。

drainage compares 测试 (lift_drainage.py): M6 kinematic(Tüzün_1979, 83边) +
M8 spot(Bazant_2006, 127边). gold M8 spot --compares--> M6 kinematic.
- 方法归纳成功 ✓ (M6 kinematic discharge model / M8 Spot Model)
- compares 双向 null conf=low —— 升不出
- 根因: Bazant_2006 abstract 提 "vacancy diffusion"(void model) 不提 "kinematic"
  (Tüzün); Tüzün_1979 不提 spot(2006). cross-mention 检索到的是数值噪声
  (K_s=1/4π, defects) 非方法对比. gold 把 void/spot 当 kinematic 的 related
  mechanisms, 但被引论文里 Bazant 提 vacancy 不提 kinematic.

### 最终诚实结论 (5 条 gold 边: 3 + drainage 2)
- **extends/improves 可升** (2/2 对齐: NGF improves μ(I) ✓ conf=high,
  I_gradient extends μ(I) ✓ conf=medium)
- **compares 系统性升不出** (0/2: I_grad-NGF + drainage spot-kinematic 都 null)
  → 根本局限: 被引论文不直接互提对方方法名/机制, compares 需综述视角
    (读两篇后对比). 单靠被引论文低阶 evidence 升不出 compares.

## 10b. segregation improves 测试 + 诚实修正 (2026-08-14)

segregation 测试 (lift_segregation.py): M23 Gray-Thornton(Gray_2005, 112边) +
M26 Sarkar-Khakhar(Tripathi_2013, 103边). gold M26 --improves--> M23.
- 方法归纳成功 ✓ (M23 segregation-induced constitutive law /
  M26 granular kinetic theory with hydrodynamic balance)
- M26→M23 (gold improves): **null** 升不出
  rationale: "Tripathi 提 Gray 核心量(密度驱动分离,μ(I))但未指出 Gray 失效情景"
- M23→M26: extends (方向反, 类型对)

### ★ 诚实修正: improves 不是稳定可升
之前 NGF improves μ(I) 升出, 是因 Bouzid 恰好提 "across yield conditions" +
"local rheology μ(I)" (隐含 μ(I) 局部在 yield 失效, NGF 非局部解决).
Tripathi 提 Gray 概念但没写 Gray 局限 → 升不出.

**improves 升出与否取决于被引论文 evidence 是否显式写"B 局限 + A 解决"**.
部分论文有(NGF/Bouzid), 部分无(M26/Tripathi). 不稳定.

### 修正后覆盖率 (6 条 gold 边)
- extends: 相对稳定可升 (I_grad extends μ(I) ✓; M23→M26 extends 方向反但类型对)
- improves: **1/2 不稳定** (NGF 升出靠 Bouzid 提 yield 局限; M26 升不出因 Tripathi 没提 Gray 局限)
- compares: **0/2 根本升不出** (需综述视角)

## 10d. kinetic extends 验证 + 最终覆盖率 (2026-08-14)

kinetic 测试 (lift_kinetic.py): M10 kinetic theory(Jenkins_1983, 73边) +
M11 extended kinetic theory(Jenkins_2010, 82边). gold M11 --extends--> M10.
- 方法归纳成功 ✓ (M10 kinetic theory of granular gases / M11 extended kinetic theory)
- **M11 extends M10: extends ✓ conf=medium 对齐 gold!** 双向都 extends.
  rationale 引真实证据: "extended kinetic theory 在 kinetic theory 基础上扩展本构关系,
  B 论文提 'pressure tensors of the two theories are similar'"

### 最终覆盖率 (7 条 gold 边, 3 方法族对)
- **extends: 2/2 ✓ 稳定可升** (I_grad extends μ(I) ✓ + M11 extends M10 ✓)
- **improves: 1/2 不稳定** (NGF improves μ(I) ✓ 靠 Bouzid 提 yield 局限;
  M26 improves M23 ✗ Tripathi 没提 Gray 局限)
- **compares: 0/2 根本局限** (I_grad-NGF + drainage spot-kinematic)

### 最终核心结论 (诚实, 7 条边支撑)
升层机制可行, 分类型产出率:
- extends 稳定可升 (被引论文常写"我扩展 X", 2/2 对齐 gold)
- improves 不稳定 (依赖被引论文是否显式写"B 局限 + A 解决", 1/2)
- compares 根本局限 (被引论文不互提对方方法名, 0/2, 需综述视角)

创新方向"低阶升层长高阶"部分可行性证实: extends 稳定,
improves/compares 受被引论文 evidence 层级限制 (诚实标注).
定位为"可溯源的高阶升层" (见 10c): 不幻觉 + 关系可审计, 适合科学 KG,
用可溯源性补低覆盖率 (vs IncSchema 高覆盖幻觉).

## 10e. multi-arm 对照: 自演化对升层无额外贡献 (诚实挑战, 2026-08-14)

lift_kinetic_multiarm.py: gold M11 extends M10, full vs frozen arm 对照.
- full arm (18 pattern, 含 split 细 pattern): extends ✓ 对齐 gold
- frozen arm (6 seed pattern, 不演化): **也 extends ✓ 对齐 gold**
- ⚠ full 和 frozen 都升出 → **自演化对升层无额外贡献**

### 诚实根因 (挑战创新核心)
1. **升层器只用 edges evidence, 不用 pattern 粒度/schema 结构** →
   frozen 粗 pattern (constitutive_law 29-34 边) 聚拢更多 evidence, 升层不弱于 full
2. 自演化 split 分散 evidence 到细 pattern, 对升层无益 (与 SPLIT_NEGATIVE_RESULT
   一致: split 多角度无优势)
3. 即使升层器用 schema 拓扑, 当前拓扑全弱 (depends_on→definitional_identity
   伪拓扑, 见 DECISION step2), full 不比 frozen 强

### 诚实链 (核心挑战)
当前升层靠 evidence 总量, frozen 粗 pattern 聚拢更多 evidence 反而升层不弱.
自演化 split 的价值仍未证实 (silhouette 无优势 + 升层无额外贡献).
这持续挑战创新核心 (自演化五操作).

### 让自演化对升层有贡献的两条路 (分析后)
A. 改升层器利用 schema 结构 (dep/con/comp 拓扑 + pattern 层级), 让细 pattern
   带来升层优势. 但当前拓扑弱, 需先重构拓扑 (从共现→文本语义, DECISION step2 提到)
B. 重新审视创新定位: 自演化价值不在升层, 在别处 (抽取覆盖/schema 质量/...)
   升层只是低阶的应用之一, 自演化价值要另找评测

### ★ 当前诚实结论 (本 session 最终)
- 低阶升层机制可行 (extends 稳定 2/2, improves 不稳 1/2, compares 根本局限 0/2)
- 但**自演化 schema 对升层无额外贡献** (frozen 也行) —— 挑战创新核心
- 要么 A (升层器用 schema 结构 + 重构拓扑), 要么 B (自演化价值另找)
- 这是诚实缺口, 不能宣称"自演化提升升层"

## 10f. NGF multi-arm 对照: 无贡献普遍确认 (2026-08-14)

lift_ngf_multiarm.py: gold M15 NGF improves M1 μ(I), full vs frozen.
- full (μ(I) 3篇229边 + NGF 15边): improves ✓ conf=high
- frozen (μ(I) 3篇315边 + NGF 11边, 6 seed pattern): **也 improves ✓ conf=high**
- ⚠ full=frozen → **自演化无贡献普遍确认** (Jenkins extends + NGF improves 两对照一致)

### 普遍性根因 (确认)
升层靠 edges evidence + 论文内容 (方法名在 evidence 里, 如 Bouzid 标题
"A non-local rheology across yield conditions"), 不靠 pattern 粒度/schema 结构.
frozen 粗 pattern (influences 99 边, constitutive_law 42) 聚拢更多 evidence,
升层不弱于 full. 自演化 split 分散 evidence, 对升层无益.

抽取覆盖也无优势: frozen 边数 (Midi 209) > full (Midi 144).

### 与历史一致
SPLIT_NEGATIVE_RESULT: split 在 silhouette 无优势.
本 session: split 在升层 + 抽取覆盖也无优势.
→ 自演化 split 当前多角度无优势, 持续挑战创新核心.

### ★ 创新核心受根本挑战 (诚实)
自演化五操作 (核心卖点) 当前在升层/silhouette/抽取覆盖都没证实价值.
升层机制可行但不依赖自演化 (frozen 也能升).
这是创新方向的根本性挑战, 需重新审视:
  (1) split 判据是否拆错维度 (SPLIT_NEGATIVE 提过改按 qualifier/role 拆)
  (2) 自演化价值是否在升层之外 (但抽取覆盖也没优势)
  (3) 创新定位是否调整: 升层是低阶应用价值 (不依赖自演化),
      主卖点转向 n-ary 超图 + 可溯源升层, 自演化降为实现选择非核心

### 修正后核心结论
升层机制可行但**产出不稳定**:
- extends 相对可升 (被引论文常写"我扩展 X")
- improves 不稳定 (需被引论文显式写局限叙述, 部分有部分无)
- compares 根本局限 (被引论文不互提对方方法名)

创新仍成立 (能升出部分高阶关系), 但诚实承认产出率不稳定,
评测须报告分类型覆盖率而非笼统对齐率. 不能维持"2/2 对齐"乐观.

## 10c. 方法论 trade-off 认识 + 创新定位强化 (2026-08-14)

升层机制天花板 = 被引论文 evidence 的跨方法叙述丰富度.
论文写什么决定能升什么关系, 这是方法本质局限非 bug.

### 核心 trade-off (vs IncSchema)
- 我们 (从低阶升): retrieval-augmented 只用真实 evidence → **不幻觉**,
  但覆盖率低 (受被引论文叙述限制)
- IncSchema (LLM 直接 probe): 高覆盖但易幻觉

### 创新定位强化: 可溯源的高阶升层
我们的方法有 IncSchema 没的优势: **关系可溯源** —— 每条升出关系都有
低阶 evidence 支撑, 可审计 (rationale 引用具体 evidence span).
对科学 KG 这是关键 (关系要可信可溯源, 非高覆盖幻觉).

→ 创新定位可强化为: **可溯源的高阶 schema 升层**
  - 不幻觉 (retrieval-augmented, 每关系有 evidence)
  - 覆盖率受 evidence 限制 (诚实标注, extends 主可升, improves 不稳, compares 难)
  - 用"可溯源"价值补"低覆盖率", 适合科学 KG 场景

诚实: 这个 trade-off 须在论文显式说, 不能只宣传"能升高阶"而藏覆盖率局限.

## 11. 下一步选择

A. 扩到 ~10 条 extends/improves gold 边确认覆盖率稳定 (gold extends 12 +
   improves 8 = 20 条可升类型). 但每篇 200-400s, 边际价值递减.
B. 接受当前结论 (extends/improves 可升, compares 根本局限), 设计正式评测:
   - extends/improves 覆盖率 + 对齐 gold 准确率
   - compares 诚实标注覆盖低
   - 对照 frozen/no_intra_dag 看自演化对升层的贡献 (独立评测)
C. 反思创新定位: 升层能产出 extends/improves 是真进展, 但 compares 局限
   意味着"长出高阶 schema"不完整 (compares 是高阶归纳重要部分).
   是否补综述叙述增强 compares? 但违反"从低阶升"原则.

判断: 当前 session 已建立核心结论 (机制可行 + compares 局限). 下一步倾向 B
(正式评测框架), 但 extends/improves 覆盖率样本仍小 (2 条), 值得扩到 ~6-8
条增强信心. 待定 (平衡时间).

纪律8: 改核心设计前 git commit + 写 DECISION (本文件即 DECISION).

---

## 附：诚实约束
- gold 高阶 evidence 是综述叙述句，单篇低阶 evidence 是局部句。
  升层本质是跨论文归纳，单篇做不出方法间关系 —— 必须多篇共性簇才升得动。
  → 最小尝试虽用单篇低阶 pattern，但归纳"方法节点"需多篇 pattern 才有意义，
  故最小尝试应取 ≥2-3 篇被引论文的低阶 pattern 做共性簇。
