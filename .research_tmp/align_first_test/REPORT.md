# 跨篇方法 Concept 对齐首验报告（A2 颗粒流 EVO2 臂，纯分析）

- 日期：2026-08-26
- 数据：`.research_tmp/runs/kernel_v2/A2G_EVO2_PPR_866D169A4642/concept_graph.json`
- 取该 bundle 的依据：10 个 A2G_EVO2 bundle 的 result.json 无 timestamp 字段，改用文件 mtime 排序；mtime 顺序与各 bundle `n_concepts` 单调递增（74→104→139→195→298→317→371→467→524→544）完全一致，确认串行累积。**PPR_866D169A4642 是时间上最后一篇**，其 concept_graph 为 10 篇累积态。
- 脚本：同目录 `analyze.py`；中间 dump：`variants_dump.txt`（30 个跨篇 concept 全量 variant）、`dup_pairs_dump.txt`（45 对重复嫌疑全列表）。
- 论文顺序（供参照）：ED6187DF955A(Kinetic theories 1983) → 99C7FFAD6529(Dense fluid transport) → B0E8916D4E19(New constitutive law) → 8EF563699F2B(Granular fluidity) → AA69AC4BEFFF(Nonlocal continuum/split-bottom) → 88D62082F6B4(Rotating cylinders segregation) → 0375D7D41441(Granular shear flows) → FD0496DE0E8E(Size segregation shallow flows) → 153BD835C9B2(Granular segregation experiments) → 866D169A4642(Granular temperatures segregation)。**10 篇全部是同一子领域（granular flow / kinetic theory / segregation），概念重叠本应非常高。**

---

## 1. 指标汇总（诚实数字，未美化）

### a. 总量与 type 分布

| 指标 | 值 |
|---|---|
| 总 concept 数 | **544** |
| METHOD | 167 (30.7%) |
| PROPERTY | 113 (20.8%) |
| PARAMETER | 95 (17.5%) |
| PHENOMENON | 88 (16.2%) |
| THING | 24 (4.4%) |
| NUMERIC | 19 (3.5%) |
| REGIME | 19 (3.5%) |
| MATERIAL | 19 (3.5%) |

### b. 跨篇合并率（下游命门）

| 指标 | 值 |
|---|---|
| surface_variants 含 ≥2 个不同 paper_id 的 concept | **30 / 544 = 5.5%** |
| 其中 METHOD（下游检索命门） | **7 / 167 = 4.2%** |
| 跨篇程度分布 | 单篇 514 / 2篇 19 / 3篇 10 / 5篇 1 |

**参照系**：这是同一子领域的 10 篇论文（"kinetic theory"、"segregation"、"restitution"、"granular temperature" 是几乎每篇都有的领域公共词汇），5.5% 的跨篇合并率意味着对齐机制基本没有工作。

### c. 跨篇 concept 的表面差异度

- 30 个跨篇 concept 中，**17 个**存在 ≥2 个不同 surface 字符串（其中 METHOD 5 个）。
- 即：合并不是只做"同名精确匹配"，确实抓住了一部分变体（如 CM0086 抓到 "RET"/"revised Enskog theory"/"revised Enskog equation"/"Chapman-Enskog expansion"）。**但样本太少（30 个），且其中混有坏合并（见 §3）**。

### d. 重复嫌疑对（对齐失败信号，token-set Jaccard ≥ 0.6，case-insensitive）

| 类型 | 对数 | 其中跨篇对齐失败（两 concept 锚定不同篇集合） | 其中篇内重复 |
|---|---|---|---|
| METHOD | 13 | 3 | 10 |
| 非 METHOD | 32 | 19 | 13 |
| 合计 | 45 | **22** | 23 |

注意：Jaccard ≥0.6 是**下界**——它只能抓到字面词集接近的对；变体更长的同义对（如 "Chapman–Enskog dense-gas kinetic theory" vs "classical Chapman–Enskog result"）不在此列，实际漏合并远多于 22。

### 交叉验证：AlignmentAgent 每篇 merge 统计（来自各 result.json）

| paper | n_candidates | n_merge | 通过率 |
|---|---|---|---|
| 99C7FFAD6529 | 27 | 2 | 7% |
| B0E8916D4E19 | 58 | 3 | 5% |
| 8EF563699F2B | 92 | 8 | 9% |
| AA69AC4BEFFF | 143 | 12 | 8% |
| 88D62082F6B4 | 154 | 13 | 8% |
| 0375D7D41441 | 146 | 7 | 5% |
| FD0496DE0E8E | 220 | 13 | 6% |
| 153BD835C9B2 | 251 | 8 | 3% |
| 866D169A4642 | 262 | 10 | 4% |

（首篇 955A 为冷启动，n_candidates=0。）对齐 agent 对 90%+ 的 merge 候选说不；candidate 随篇数线性涨（27→262），merge 数几乎不涨（2→10），越到后面欠合并越严重。

---

## 2. 领域核心方法家族碎片化审计（最硬的证据）

这 10 篇是同领域论文，以下方法名理应各合并成 1-2 个跨篇 concept：

| 家族 | concept 数 | 覆盖论文数 | 现状 |
|---|---|---|---|
| kinetic theory | **11 个 METHOD** | 6 篇 | 仅 CM0120("kinetic theory") 跨 3 篇；其余 "kinetic theory formula"/"kinetic theories"/"kinetic theory predictions"/"kinetic theory for binary mixtures"/"kinetic theories for granular flow" 等各自为政 |
| Chapman–Enskog | **7 个 METHOD** | 2 篇 | 其中 6 个来自**同一篇** 955A（篇内都没去重）；跨篇合并 0 |
| segregation | 13 个（PHENOMENON 为主） | 4 篇 | 仅 CF0607("segregation") 与 CF0368 各跨 3 篇 |
| restitution coefficient | **3 个 PARAMETER** | 3 篇 | "coefficient of restitution e"(跨2篇) / "restitution coefficient" / "coefficient of restitution" 三个互不合并 |
| granular temperature | 4 个 | 3 篇 | "granular temperature"(跨2篇) / "granular temperatures" / "anisotropy of the granular temperature" / "each species achieved its own granular temperature" |
| molecular dynamics | 2 个 METHOD | 1 篇 | 同篇内 "molecular dynamics simulations" vs "molecular dynamics type simulations" 未合并 |
| DEM (discrete element) | 2 个 | 1 篇 | 8EF563699F2B 的 "discrete element method (DEM) simulations" 被并进 CM0113("numerical simulations")，AA69AC4BEFFF 的 "discrete element simulations"(CM0289) 是独立 concept——DEM 没有自己的跨篇身份 |

**结论**：本领域第一公共方法 "kinetic theory"（6 篇都谈）碎成 11 个 concept；"restitution coefficient" 这种教科书级公共参数碎成 3 个。这是"几乎不合并"的直接证据。

---

## 3. 跨篇合并 concept 完整列表（30 个 ≤ 40，全列）

格式：cid | type | 跨篇数 | 各 variant（paper 后缀 | surface）。paper 后缀为 paper_id 末 4 位。

### METHOD（7 个）

- **CM0113** | METHOD | 5 篇 | [4E19] numerical simulations / [9F2B] discrete element method (DEM) simulations / [6529] **model** / [0E8E] numerical simulations / [C9B2] **model** —— 坏合并嫌疑：把泛称 "model" 和具体方法 DEM 混进一个 concept
- **CM0004** | METHOD | 3 篇 | [955A] experiments / [EFFF] experiments / [4E19] experimental measurements —— 同义合并，正常
- **CM0120** | METHOD | 3 篇 | [4E19/9F2B/1441] kinetic theory —— 全部同 surface，正常（但见 §2：其他 kinetic theory 变体没并进来）
- **CM0344** | METHOD | 3 篇 | [F6B4] theory / [EFFF] **Nonlocal continuum models** / [C9B2] theory —— 坏合并嫌疑：泛称 "theory" 和具体方法 "Nonlocal continuum models" 合并
- **CM0086** | METHOD | 2 篇 | [6529] RET / revised Enskog theory / revised Enskog equation / Chapman-Enskog expansion / [6462] revised Enskog theory —— 变体合并的正面样本（缩写 RET ↔ 全称）
- **CM0117** | METHOD | 2 篇 | [4E19] new constitutive law for dense granular flows / local rheology / continuum description / [9F2B] constitutive relation between stress and strain-rate / inertial law —— 合并跨度偏大存疑，但语义上同族（μ(I) 流变学的不同叫法），可接受边缘
- **CM0480** | METHOD | 2 篇 | [0E8E] Fig. 3 / [6462] Fig. 3 —— 垃圾：图表引用被抽成 METHOD concept 再跨篇合并

### PARAMETER（13 个）

- **CP0027** | 3 篇 | [955A] R / R / R / [EFFF] r / [6529] **density** —— 符号碰撞 + 语义污染：单字母 R/r 与 "density" 合并
- **CP0164** | 3 篇 | [9F2B] d / [1441] D / [C9B2] D —— 符号合并（粒径 d/D），语义上大概率正确
- **CP0193** | 3 篇 | [9F2B] F / f / [F6B4] f / [C9B2] f —— 单字母合并，语义对错不可判（可能各篇的 f 指不同量）
- **CP0211** | 3 篇 | [9F2B] 66T^9 / [F6B4] T / [EFFF] ( 1 / T ) d T / d θ —— 疑似 OCR 噪声 "66T^9" 与温度 T 合并
- **CP0009** | 2 篇 | [955A/1441] coefficient of restitution e —— 干净的跨篇合并正面样本
- **CP0040** | 2 篇 | [955A] ρ / [F6B4] ρ —— 符号合并（密度），大概率正确
- **CP0041** | 2 篇 | [955A] n / [C9B2] n —— 单字母，不可判
- **CP0170** | 2 篇 | [9F2B/EFFF] velocity fluctuation —— 干净
- **CP0176** | 2 篇 | [9F2B/EFFF] flow field —— 干净
- **CP0209** | 2 篇 | [9F2B] P / $P$ / [F6B4] p —— 符号合并（压力），大概率正确
- **CP0267** | 2 篇 | [EFFF/0E8E] z —— 符号，不可判
- **CP0296** | 2 篇 | [EFFF] intruder depth z_m / distance between the intruder and the primary flow zone / [0E8E] z_m —— 带定义性变体，正常
- **CP0326** | 2 篇 | [EFFF] A / [C9B2] a —— 单字母，不可判

### PROPERTY / PHENOMENON / REGIME（10 个）

- **CC0019** | PROPERTY | 2 篇 | [955A/EFFF] experimental results —— 干净
- **CC0028** | PROPERTY | 2 篇 | [955A/F6B4] temperature —— 干净但过于泛化（普通温度？粒温？）
- **CC0147** | PROPERTY | 2 篇 | [9F2B/1441] granular temperature —— 干净的正面样本
- **CC0152** | PROPERTY | 2 篇 | [9F2B/F6B4] pressure —— 干净
- **CR0178** | REGIME | 2 篇 | [9F2B/C9B2] steady state —— 干净
- **CF0216** | PHENOMENON | 3 篇 | [EFFF] primary flow / [6529] rapid flow granular media / [9F2B] flow —— 过泛：把 "flow" 也吸了进来
- **CF0368** | PHENOMENON | 3 篇 | [F6B4] density segregation / granular segregation / binary mixtures / mixing / segregation of the species / density radial segregation / radially segregated pro6les / [0E8E] particle size segregation / particle-size segregation / [C9B2] granular segregation / mixing —— 过度合并：density segregation 与 size segregation 是本领域两个**不同机制**，被合成一个 concept
- **CF0607** | PHENOMENON | 3 篇 | [C9B2/6462/0E8E] segregation / segregation process —— 干净
- **CF0612** | PHENOMENON | 2 篇 | [C9B2] thermal equilibrium / [6462] failure of equipartition —— 语义可疑（热平衡 vs 能量均分破缺是近义但方向相反）
- **CF0615** | PHENOMENON | 2 篇 | [C9B2] density-difference–driven segregation / [6462] upward segregation of B relative to A / downward segregation / tracer segregation / segregation under gravity of a dilute gas of larger particles... —— 过度合并

**跨篇合并中坏合并/存疑比例**：30 个中明确干净约 17 个，坏/存疑约 13 个（CM0113、CM0344、CM0480、CP0027、CP0211、CF0216、CF0368、CF0612、CF0615 及若干单字母不可判），**约 40% 存疑**。

---

## 4. 重复嫌疑对全列表

### METHOD（13 对；CROSS = 两 concept 锚定不同篇集合 = 跨篇对齐失败；WITHIN = 篇内重复）

| cid | surface | cid | surface | J | 分类 | 人工判读 |
|---|---|---|---|---|---|---|
| CM0003 | Chapman–Enskog dense-gas theory | CM0020 | Chapman–Enskog dense-gas kinetic theory | 0.83 | WITHIN | **真重复**，955A 篇内 6 个 CE 变体全应合并 |
| CM0003 | Chapman–Enskog dense-gas theory | CM0024 | Chapman–Enskog dense-gas analysis | 0.67 | WITHIN | 真重复（"analysis"=theory 的行文变体） |
| CM0003 | Chapman–Enskog dense-gas theory | CM0066 | Chapman–Enskog theory | 0.60 | WITHIN | 真重复 |
| CM0006 | Chapman–Enskog kinetic theory of dense gases | CM0020 | Chapman–Enskog dense-gas kinetic theory | 0.62 | WITHIN | 真重复 |
| CM0006 | Chapman–Enskog kinetic theory of dense gases | CM0064 | Chapman–Enskog hard-sphere kinetic theory of dense gas | 0.60 | WITHIN | 真重复 |
| CM0020 | Chapman–Enskog dense-gas kinetic theory | CM0064 | Chapman–Enskog hard-sphere kinetic theory of dense gas | 0.67 | WITHIN | 真重复 |
| CM0002 | other theories of granular flow | CM0023 | granular-flow theories | 0.60 | WITHIN | 大概率真重复（泛指） |
| CM0005 | kinetic theories for granular flow | CM0023 | granular-flow theories | 0.60 | WITHIN | 大概率真重复 |
| CM0023 | granular-flow theories | CM0057 | recent granular-flow theories | 0.75 | WITHIN | 大概率真重复 |
| CM0377 | molecular dynamics simulations | CM0418 | molecular dynamics type simulations | 0.75 | WITHIN | 真重复 |
| CM0120 | kinetic theory (3篇) | CM0383 | kinetic theory formula | 0.67 | **CROSS** | 真重复，跨篇对齐失败 |
| CM0120 | kinetic theory (3篇) | CM0613 | kinetic theory predictions | 0.67 | **CROSS** | 真重复，跨篇对齐失败 |
| CM0433 | kinetic theory predictions (4.13) | CM0613 | kinetic theory predictions | 0.60 | **CROSS** | 真重复，跨篇对齐失败 |

METHOD 13 对人工判读：**13/13 都是真重复或大概率真重复，0 对是真不同**。（"kinetic theory predictions" 是否应独立于 "kinetic theory" 可 argue，但至少 CM0433 vs CM0613 这对无可争议。）

### 非 METHOD（32 对，摘关键项；全量见 dup_pairs_dump.txt）

**跨篇对齐失败（19 对）中最典型的**：

| cid | surface | cid | surface | 人工判读 |
|---|---|---|---|---|
| CP0009 (955A+1441) | coefficient of restitution e | CP0378 (1441) | coefficient of restitution | **铁证级真重复**：同一篇 1441 里两个都有，且 CP0009 已跨 2 篇 |
| CP0081 (6529) | restitution coefficient | CP0378 (1441) | coefficient of restitution | 真重复（词序倒置变体） |
| CP0122 (4E19) | Inertial number | CP0320 (EFFF) | inertial number I | 真重复（μ(I) 流变学核心参数，两篇各自为政） |
| CR0178 (9F2B+C9B2) | steady state | CR0264 (EFFF) | steady-state | 真重复（连字符变体都没合并） |
| CC0156 (9F2B) | equivalent shear stress | CC0316 (EFFF) | equivalent shear stress τ | 真重复 |
| CP0169 (9F2B) | granular fluidity | CC0317 (EFFF) | granular fluidity g | 真重复 |
| CP0171 (9F2B) | packing fraction | CC0318 (EFFF) | solid packing fraction | 大概率真重复 |
| CC0379 (1441) | wall slip velocity | CC0405 (1441) | slip velocity | 边缘（wall slip 是 slip 的子类） |
| CC0401 (1441) | solids fraction | CP0415/CP0437 (1441) | solids fraction $\nu$ / solids fraction ν | 真重复×2（带符号后缀没去重） |
| CF0444 (0E8E) | inversely graded layers | CF0545 (0E8E) | inversely-graded layers | 真重复（连字符） |

**真不同的代表**（Jaccard 高但语义不同，非对齐失败）：CP0199 vs CP0202（刚度 vs 阻尼系数）、CC0314 vs CC0315（应力偏量 vs 应力张量）、CMAT0359 vs CMAT0369（等密度异尺寸 vs 异尺寸混合）、CMAT0495 vs CMAT0496（小颗粒层 vs 大颗粒层）、CF0463 vs CF0539（shock 的不同部分）。

非 METHOD 32 对人工判读：约 22 对真重复/大概率真重复，约 10 对真不同（多为修饰语不同的物理量近亲，合并需谨慎——这部分不合并是**正确行为**）。

---

## 5. 判读：对齐现状定性与下游风险

### 定性：**"几乎不合并"是主故障，且伴生"符号碰撞型错误合并"——不是"保守但精准"**

三选一的判定：**"几乎不合并"**。证据链：

1. 同子领域 10 篇语料，跨篇合并率仅 5.5%（METHOD 4.2%）。若对齐正常，"kinetic theory/restitution/granular temperature/segregation" 这类领域公共词汇的跨篇合并率应该在百分之几十量级。
2. 领域第一公共方法 "kinetic theory" 碎成 11 个 concept、6 篇各自为政；"restitution coefficient" 碎成 3 个。
3. AlignmentAgent 对候选的 merge 通过率仅 3-9%，且 candidate 越多通过率越低（后面几篇 3-4%）——不是"前面合并过了所以后面没得合"，因为 "inertial number"(EFFF) 在 4E19 已有 CP0122 的情况下仍未合入。
4. 同时**不是精准保守**：已发生的 30 个跨篇合并里约 40% 存疑，包括两类系统性坏合并：
   - **符号碰撞**：单字母参数（R/r、P/p/$P$、A/a、F/f、n）跨篇合并——不同篇里单字母指代不同物理量的概率很高，这是拿"表面相同"当"语义相同"；
   - **泛称吞噬**："model"、"theory"、"flow"、"Fig. 3" 这类泛称/垃圾 surface 被抽成 concept 并参与合并（CM0113 吞了 "model" 和 DEM、CM0344 吞了 "theory" 和 "Nonlocal continuum models"、CF0368 把 density segregation 和 size segregation 两个不同机制合为一体）。

即：**该合的（具体方法名、领域公共参数）基本没合；不该合/需谨慎的（单字母符号、泛称）反而合了。** 这是两个独立故障的叠加：(a) merge 判定对真实方法名变体过严；(b) 抽取层放进了泛称/符号/图表引用这类噪声 surface，对齐层对噪声又过松。

### 下游风险判断

**命门当前不成立。** 下游问答需要"同一方法不同叫法 → 同一 concept"，现状：

- 检索 "DQN" 找不到 "deep Q-network" 这类场景的对应物就是本 audit 的主体：查 "kinetic theory" 只能命中 CM0120 一支（3 篇），其余 8 个 kinetic-theory 系 concept 及其所在论文全部断链；查 "restitution coefficient" 命中 CP0378 则丢掉 955A 那支。
- METHOD 层更严重：167 个 METHOD 只有 7 个跨篇，其中干净的约 3-4 个（CM0120、CM0086、CM0004、CM0117 边缘），垃圾 3 个（CM0480 "Fig. 3"、CM0113、CM0344）。**方法级跨篇对齐实际有效供给 ≈ 2-4%。**
- 反向风险：CF0368（density/size segregation 合一）、CM0113（model+DEM 合一）这类过度合并会让下游检索"命中但答错"（机制混淆），比断链更隐蔽。
- 篇内重复（23 对 WITHIN，Chapman–Enskog 6 连碎是代表）说明问题不只在跨篇对齐层——篇内抽取去重/规范化同样缺位，进入对齐层之前概念已经碎了。

### 一句话结论

**跨篇概念对齐当前处于"几乎不合并"状态（同域 10 篇仅 5.5%，METHOD 4.2%），且少量已发生的合并中约 40% 是符号碰撞或泛称吞噬型坏合并；下游"不同叫法 → 同一方法"的检索链路在 METHOD 层实际可用率约 2-4%。在修复对齐判定（对具体方法名放宽）+ 抽取层噪声 surface 过滤（泛称/单字母/图表引用）之前，A2 颗粒流的共享 A-box 对下游问答不构成可用的跨篇检索基础。**

---

*附：本报告只做分析定诊，不含任何修复决策。原始数据 544 concept 全量、45 对重复嫌疑、家族碎片化明细见同目录 dump 文件。*
