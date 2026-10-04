# 整体审查 + 深化路线图（compact 前定稿）

Date: 2026-08-14

## 一、真实状态审查（不是感觉，是查出来的事实）

### 1.1 代码（扎实，基本可信）
- LogicKG: paper_registry.py / corpus_driver.py / hypergraph_lifter.py(升层+结构信号+并行) / hypergraph_schema.py(from_dict) — 已 commit 到 main，逻辑闭环。
- sci-evo-extract: external_artifacts 表 + store/list + POST API — 已 commit。
- experiments/self_evolution_lift/: 7 个 eval 脚本入库。
- **问题**：judge 并行化是 ThreadPoolExecutor 简单版，无错误隔离/重试，偶发 hang。

### 1.2 ★最严重发现：有完整 gold 却没用
`.research_tmp/gold/ARFM2024_gold.json` 存在且完整：**53 方法(带aliases) + 41 演化边(src/tgt/type/evidence) + 41 law_constraints**。Intern-Atlas 式结构化 gold。
但所有 step7 评测（ablation/complex_q/baseline_PK/NMR）**没有一个用它做边级评测(ERR)**：
- ablation/complex_q/baseline_PK: 只判"gold 类型覆盖"(3/3)，不判具体边对错
- NMR: 用内联的 10方法/6关系简化 gold，没用 53/41 完整 gold
- 之前 eval_gold_41edges.py 跑过完整 gold: **41边只 3 条 coverable，edge_hit 仅 1**（这是诚实承认的覆盖率不足，被后续 step7 的"3/3类型覆盖"掩盖了）

→ **"3/3 gold 类型覆盖"是误导性指标**，掩盖了 41 边只命中 1 条的事实。这是当前最大诚实性问题。

### 1.3 评测数字的真实水分
| 评测 | 报告数 | 真实问题 |
|---|---|---|
| 消融 F 0/3 | 类型覆盖 | 完整 gold 41 边只命中 1，不是 0/3 vs 3/3 那么干净 |
| NMR 0.700 | 10方法简化gold | 没用 53 方法完整 gold；M1/M17 歧义未修 |
| SciREX 1.000 | 8篇宽松召回 | 小样本+子串匹配+只召回无precision |
| SciFact 0.250 | 12 claim | 小样本，诚实劣势但不可靠 |
| baseline PK improves 0→8 | direct 2/3 vs ours 3/3 | direct-probe 是适配版非原IncSchema；LLM非确定性未控 |
| 复杂问题 0→0.5 | N=4 | 题目手写，cherry-pick 风险 |

### 1.4 评测独立性问题（循环论证风险）
- judge=GLM-5，抽取=deepseek，gold=GLM-5.2 — 三模型分离 ✓
- 但 NMR 用 GLM-Embedding-2 做匹配，embedding 是我们方法栈一部分，匹配 gold 用自家 embedding 有轻微循环
- ablation 的 gold 类型覆盖是自评，非外部独立
- ARFM2024 gold 是自建(虽比6条完整)，无标注一致性(kappa)

### 1.5 论文草稿
- .research_tmp/PAPER_draft_step8.md 是草稿，不是终稿
- related work 浅，无 failure case 分析，threat to validity 缺
- 主结果表数字如上含水分

### 1.6 第二领域
- SciREX(NLP/ML域)只做了实体召回，**没评升层**(方法演化关系)
- SciREX gold 是抽取级 n-ary，不是跨论文方法演化级——要在NLP域证升层需再建gold

## 二、深化优先级（先后顺序，按"诚实战力恢复"排）

### P0 必须先做（不做好后面都不可信）
**P0-1. 用完整 gold 重做 ERR 边级评测**
- 用 ARFM2024_gold.json 的 41 演化边，对 lift_corpus 各臂做边级命中(不是类型覆盖)
- 修 NMR 用完整 53 方法(带aliases)做匹配，解决 M1/M17 歧义(用 aliases 区分 μ(I)本体 vs I-gradient扩展)
- 产出诚实的 ERR/PSC 数字，替换"3/3类型覆盖"的误导性指标
- **这是诚实性命门**：现在报告的 3/3 掩盖 1/41，必须纠正

**P0-2. LLM 非确定性控制**
- 每臂跑 3-5 seed，报均值+方差
- paired bootstrap 做显著性检验(结构信号臂 vs 纯文本臂)
- 不然审稿人一句"非确定性"全否

### P1 接着做（让评测站得住）
**P1-1. gold 标注一致性**
- 41 边/53 方法：至少两人独立标一部分，算 Cohen's kappa
- 或者：LLM 辅助标 + 人工核验，报协议
- ARFM2024 gold 现在来源(从综述抽？哪些综述？)要写清

**P1-2. SciREX 全量 + 严格 F1**
- 66 篇全跑，报 Method/Material/Metric/Task 的 precision+recall+F1
- 严格匹配(边界+精确串，非子串)
- 这是把 1.000 水分去掉的唯一办法

**P1-3. baseline 真版至少跑一个**
- IncSchema clone 了但 OpenAI-bound → 找能改 LLM 的 baseline 或适配原版 prompt 到 deepseek 严格复现
- AutoSchemaKG/ASEE 至少 clone 一个跑通
- 现在的"适配 direct-probe"作辅助，不作主 baseline

### P2 然后做（深度和机制性）
**P2-1. 结构信号可解释性消融**
- quals 各维度拆开：evidence_strength 单独 / cited_from 单独 / method 单独
- 看哪个真起作用，给机制解释(不只"加 hint 有效")
- 这把"prompt 工程"升级成"机制分析"

**P2-2. 第二领域升层评测(NLP域)**
- 在 SciREX 之上建跨论文方法演化 gold(NLP综述)
- 或换 NLP 方法演化更主流的设定
- 证升层跨域泛化(不是只抽取跨域)
- **大工程，决定是否做见下**

**P2-3. failure case 分析**
- lift 抽错的边逐条分析(为什么 40 条 gold 边漏 39)
- 写进论文 discussion，诚实展示局限

### P3 最后（写作）
**P3-1. related work 深耕**(6 邻近工作精读，表1对比)
**P3-2. threat to validity**(LLM非确定性/gold自建/域边界)
**P3-3. 完整论文重写**(基于P0-P2真数字)

## 三、两个方向选择(要用户定，影响投入)

### 选择1: 第二领域要不要做(路径A vs B)
- 路径A: 颗粒流做深(完整gold ERR) + SciREX只证抽取通用 — 泛化限到抽取层，1.5域
- 路径B: 颗粒流+NLP都做升层gold — 真泛化但2倍工作，半成品风险
- 倾向A，但用户定

### 选择2: 颗粒流 gold 扩不扩到 Intern-Atlas 式 30 篇综述
- 现在 1 篇综述(ARFM2024) 53方法41边
- 30 篇是 Intern-Atlas 规模，工作量大
- 个人做 3-5 篇综述可能更现实，报"规模小于Intern-Atlas但方法可迁移"
- 用户定规模

## 四、compact 后新 goal 的建议重点
1. P0-1 完整gold ERR(诚实命门)
2. P0-2 多seed统计检验
3. P1-2 SciREX全量F1
4. 其余按优先级

新 goal 应明确：**先恢复诚实性(P0)再扩规模(P1)再深度(P2)再写作(P3)**。现在最大的问题是"3/3类型覆盖"掩盖了"41边只命中1"，必须先纠正这个再谈扩。
