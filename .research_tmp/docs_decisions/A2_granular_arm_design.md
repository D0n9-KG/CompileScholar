# A2 颗粒流域臂选篇设计（v1，基于 ref 对齐表）

日期：2026-08-26。对齐表：`.research_tmp/A2_granular_alignment.json`（15 confirmed + 5 title-only，全部经标题 sanity 验证 5/5）。
对照：ML 臂（DQN→Double→Dueling→PER→A3C→Rainbow，等 MinerU 服务）。

## 选篇原则

1. **构成演替链**：每篇的 gold 方法与前篇有 extends/improves/replaces 关系（gold evolution_edges 里有边）
2. **时间跨度**：1983→2022（比 ML 臂 2013-2018 的跨度长 10 倍——schema 演化在长跨度上的行为本身是有信息量的观测）
3. **双链并行**：rheology 链（连续介质）+ segregation 链（动力学）——两族方法互不依赖，观察 domain 内的 schema 分化
4. 10 篇规模（与 ML 臂对齐，两臂等量）

## rheology 链（5 篇，μ(I) 故事线）

| 序 | PPR | 论文 | gold 方法 | 演替关系（gold 边） |
|---|---|---|---|---|
| 1 | PPR_ED6187DF955A | Jenkins & Savage 1983 | M2 kinetic theory | 链首 |
| 2 | PPR_99C7FFAD6529 | Garzo & Dufty 1999 | M2 精化 | extends M2 |
| 3 | PPR_B0E8916D4E19 | Jop et al. 2006 | M1 μ(I) rheology | 替代/compares M2（连续介质 vs 动力学） |
| 4 | PPR_8EF563699F2B | Kamrin & Koval 2012 | nonlocal fluidity | extends M1（非局部修正） |
| 5 | PPR_AA69AC4BEFFF | Henann & Kamrin 2013 | M42 相关 nonlocal | extends 前篇 |

（MiDi 2004=PPR_88BE57F86CCD 备选替换 3，看 Jop 段落抽取质量定）

## segregation 链（5 篇，drum/segregation 故事线）

| 序 | PPR | 论文 | gold 方法 | 演替关系 |
|---|---|---|---|---|
| 6 | PPR_88D62082F6B4 | Khakhar et al. 1997 | 流化牵引模型 | 链首 |
| 7 | PPR_0375D7D41441 | Savage & Lun 1988 | M18 kinetic sieving | 链首 2（更早） |
| 8 | PPR_FD0496DE0E8E | Gray & Thornton 2005 | 混合物理论 segregation | 改进 Savage-Lun（gold: improves） |
| 9 | PPR_153BD835C9B2 | Sarkar & Khakhar 2008 | 实验证据+流化描述 | compares Khakhar 1997 |
| 10 | PPR_866D169A4642 | Yoon & Jenkins 2006 | segregation fluidity | extends kinetic 线 |

## 执行参数

- 顺序：**按时间序串行**（演化臂必须串行——第 N 篇看到前 N-1 篇的 schema）
- arm：full（演化臂）与 frozen（对照臂）各跑一遍，同篇同 seed
- 观测：跨篇 pattern 复用率（A2 主指标）、演化触发篇数、新 pattern 采纳率、升格条件观察（分层 namespace）
- 数据：全部走 run_kernel.py --tag A2G_{arm}

## 未定项（等 Run 5/ML 语料后一起定）

- seed 数（≥4 是论文要求；先 1 seed 跑通看信号再扩）
- judge 验收口径（Run 5 数字+门柱重校准讨论后定）
