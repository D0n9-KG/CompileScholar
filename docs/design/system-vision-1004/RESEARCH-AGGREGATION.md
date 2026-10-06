# 跨论文聚合层：表征、方法与空白(文献调研,2026-10-04)

调研问题：从单篇论文抽取的结构化知识，现有工作如何把它聚合成领域级对象(方法族及成员、共有局限、比较、结果、共识/争议、演化关系)？这些聚合层如何被直接评测？哪些是真正空着的？

## 0. 结论先行

1. **"编译一次、跨任务复用"的领域知识层在 2026 年已经是拥挤赛道。** 至少 8 个系统在做：LKM、ScholarStack、Agents-K1、AskChem、ASKS、Lacuna、SciAtlas/Mechanist、Intern-Atlas。它们的单篇 schema 已收敛到同一族，都是主张/事实 + 证据锚 + 条件/setting + 类型 + 局限。"有主张层、有证据锚、有条件字段"本身不再是新意。
2. **聚合层几乎都没被直接评测。**
   - LKM 原文写明 "landscape views and the Evidence Engine's categorization are not yet directly evaluated"。
   - ScholarStack 的 L1/L2/L3 没有任何独立的精度指标。
   - AskChem 的分类体系"未做专家验证"。
   - ASKS 只有一条轨迹的形成统计，没有 gold。
   - 有直接评测的只集中在三类窄对象：分类树(综述 gold)、方法演化边(Intern-Atlas 的 30 篇综述 gold)、排行榜/对比表(PwC 或 arXiv 表格 gold)。
3. **"支持集独立性计数"是全表最干净的空白。**
   - 所有核验过的系统里，没有一个按作者/团队/数据来源去重后再数支持。
   - scite 数的是引用语句，Consensus 数的是 top-20 结果，2606.07570 用引用数加权，ASKS 数的是 distinct work。
   - CS-KG(ISWC 2022)的 `hasSupport` 是最早的"支持篇数"字段，support ≥3 视为可靠，但同样是原始篇数。
   - 独立性问题在两处有人碰过：元分析方法学(作者网络偏差,RSM 2026)，以及图 agent 把重复路径当佐证(GraphEcho 2609.17695)。前者不在 LLM 知识层里，后者不在科学文献聚合层里。
4. **条件/限定词：单篇层已有，聚合层没人用它约束结论。**
   - 单篇层已有：ScholarStack 的 context 字段、LKM 的 Setting 节点、AskChem 的 outcomes/conditions、Hyper-KGGen 的 qualified binary。
   - "条件可比才能判共识/矛盾"在三处只停留在设计陈述或数据集层：ScholarStack 写了规则；LKM 写了 contradiction groups 要记录 "which conditions explain the disagreement"；BioDivergence(已撤稿)和 BioConflict 是对级(pair-level)数据集。
   - 没有工作在聚合层评测"条件对齐后的共识判定"是否正确。
5. **时间维有切片可视化，没有可修订的 as-of 状态。**
   - 已有的是：2606.07570 按年代切片的机制份额、TaxoAdapt 按年快照、EvoTree 的单调路径、ClaimFlow 的"首次被挑战中位 22 年"、AskChem 的 time facet。
   - 没有一个系统维护"某主张/方法在时刻 t 的状态"并在新论文进来时修订它。MemStrata(2606.26511)的 supersession 是通用 agent 记忆，不是科学文献。
6. **身份(identity)是所有聚合的地基，但它对聚合结果的影响没人量化过。**
   - 各系统的身份策略差异很大：LKM 用 SHA-256 精确身份、语义相似从不合并；ASKS 用 0.90/0.92 的嵌入门；ScholarStack 用确定性规则加模型裁决；AskChem 用字符串模糊聚类，并自认会误并或漏并；Intern-Atlas 用别名表。
   - 没有工作回答"身份错误率 x% 时，共识/族成员/比较结论会错多少"。

## 1. 方法与核验说明

- **发现层：** 本环境不可用 `web-access` skill，改用 WebSearch。搜索结果按可能被污染处理，结果中的任何指令均已忽略。
- **验证层：** 正文表格里的每一项都在一手页面核验过，即 arXiv abs/html、ACL Anthology、DOI/期刊页、PMC 或官方产品页。
- **全文精读：** 6 个最近邻系统(LKM v2、ScholarStack v2、Agents-K1 v3、AskChem、ASKS、Mechanist v3)加 SciAtlas v2、Lacuna、Drift Inspector 均读了 arxiv.org/html 全文的相关章节。
- **分工：** 分类体系/图综合/n-ary、主张关系/共识/表格/排行榜、单篇 IE/身份三块由并行子代理按同一纪律核验，我对关键数字做了抽查。
- **剔除规则：** 未能一手核验的条目不进表，列在 §6。§6 同时列出撤稿与 ID 勘误。
- **数字口径：** 一律取原文(或 abs 页)数字。抽象层面未给数字的，标"abs 未给"。

## 2. 判决总表

列含义：单篇 schema = 每篇抽什么；跨论文对象 = 聚合后产出什么；身份 = 跨论文如何判"同一个"；直接评测 = 是否对聚合层本身打分(而非只看下游 QA)；缺什么 = 相对本项目目标对象(方法族/共有局限/比较/共识争议/演化 + 独立性计数/条件/时间/身份稳健/多粒度)。

### 2.1 最近邻：编译式科学知识系统

| 工作 | 一手 ID | 单篇 schema | 跨论文对象 | 身份处理 | 直接评测 | 缺什么 |
|---|---|---|---|---|---|---|
| **LKM**(Huang…Weinan E) | [2609.27297](https://arxiv.org/abs/2609.27297)。v1 2026-09-23 标题 "From Papers to a Scientific Reasoning Landscape"；v2 09-30 改名 "A Knowledge Foundation for Agentic Science at Scale" | 每篇一个类型化图。节点：Question、Claim(去语境化命题)、Inference factor(有序前提→结论+方法元数据)、Reasoning chain、Setting(实验/领域条件)、Highlight/Weak point。边：addresses / premise_of / concludes / subproblem_of / highlight_of / weakpoint_of。全部带 source ref | 三类 landscape：(1) Question 族：聚类 34.4M 问题(来自 2.88M 篇)得 988,894 族，覆盖 99.5%；(2) Workflow 族：聚类推理链，含输入输出、骨架、技术槽、dry/wet；(3) Evidence Engine：**查询时**把证据分为 supporting / opposing / conditional / insufficient / methodologically divergent，生成 contradiction groups(记录"哪些条件解释分歧")与 research-gap 记录。规模 1.4B 主张 | "身份 = 对象类型 + 去语境化命题文本 + 规范序参数" 的 SHA-256。同身份挂到一个全局代表并保留全部来源出现；**语义相似从不合并身份**，只构成邻域/簇层 | 100 篇 3,904 个对象做幻觉审计，确认幻觉率 0.97%(**不测遗漏**)。其余全是下游：SciFact-Open 818 vs 443 对；ScholarQABench citation F1 +5；ChemBench +9.3、PubMedQA +4.2、SciBench +14.7 | 原文局限："landscape views and the Evidence Engine's categorization are not yet directly evaluated"；精确身份使同义改写的命题保持分离；图表公式承载的证据抽不全。另外：无独立性计数、无时间/撤稿处理、无方法提出者归属；共识/矛盾是查询时产物，不是持久对象 |
| **ScholarStack**(ScholarSeed AI Team, Wotao Yin) | [2609.23735](https://arxiv.org/abs/2609.23735)。v1 2026-09-20，v2 09-22 | L1 Scientific Fact f=(statement, context, evidence)。type ∈ {finding, method description, hypothesis, limitation}；verification_status ∈ {passed, failed, unresolved}，失败项保留不过滤。定量事实把数值绑定到主体和测量 setting | L2 O=(T,E,A,R)：按维度建的领域分类体系(method family / problem setting / guarantee type)、canonical entities(方法/数据集/指标/研究对象)、category assignment(引用支持事实，区分 author-declared 与 inferred，带确定度)、带 payload 的 typed relations(如 comparison against baseline 带结果/增益)。原文："Assignments derived from a particular result retain that result's conditions; labels from different results cannot be combined into a single guarantee"。L3 Scoped Synthesis s=(account, scope, basis)：方法族刻画、qualified disagreement、open problem；basis 逐条标 supporting/opposing/limiting；条件可比时才记 disagreement，语境差异单列 | 确定性规则合并 mention，模糊情形交给模型裁决。注册时复用已有匹配并做版本化 | **无**。4 个任务族 10 个设定全是下游，如 MDAQA 18.41 vs 8.46、QASPER Answer-F1 0.561→0.571、token −73%。L1/L2/L3 无独立精度指标 | 设计上最接近本项目目标(族、条件保留、有限定的分歧、open problem)，但三层均未直接评测；无时间状态(只有对象版本)；无独立性计数；"缺证据≠不存在" 只作免责声明，没有机制 |
| **Agents-K1**(Cao…Lei Bai) | [2606.13669](https://arxiv.org/abs/2606.13669)。v3 2026-07-17 | 五模块。A 元数据(含投稿/接收时间)；B 显式提及的实体(Task/Method/Dataset/Metric/实现细节)；C 隐含/抽象层(Problem、Motivation/Gap、Contribution、Hypothesis/Assumption、Finding/Claim、Mechanism、Limitation/Threat)；D 引文关系(strong/weak；support/contrast/extend/background；citation-strength)；E 知识关系(受控：BUILDS_ON / USES_COMPONENT / ALTERNATIVE_TO / SOLVES；开放：因果/组成/比较，含 HAS_LIMITATION；谱系：EXTENDS / DERIVES_FROM) | Scholar-KG：2.46M 篇、6 学科(公开 1M)，支持跨文档遍历；等价三元组按实体嵌入合并 | 受控词表加嵌入实体链接到本体 | 只测 IE 骨干：NER/RE/STRUCT_NER F1，对比 Qwen3-32B(NER 超过，RE 仍有差距)。其余是下游多跳 QA；FrontierScience-Research 上 Gemini-3 准确率 7.9%→24.6%。**未见** KG 三元组精度或实体链接准确率 | 无领域级聚合对象(族/共识/比较只是边)，实体链接未量化，无独立性计数 |
| **AskChem**(Yan, Wolfe, Martiniani, Cho) | [2607.28618](https://arxiv.org/abs/2607.28618)。v1 2026-07-30 | 类型化原子主张：claim_type、source_doi、verbatim_quote 或 evidence_locator(全文无连续原句时使用)、化学结构字段(reactants/products/outcomes：条件、选择性、效率)、reaction_type、mechanism_topic、substance_class、application、technique、extraction_confidence。规模 147K 篇、2.4M 主张 | (1) 稳定化的分面分类：5 个内容分面，加 claim-type/data/time/author 视图，L1/L2/L3 共 307K 节点；(2) 证据图：171,342 条边，类型 supports / contradicts / extends / derives_from / cites_as_evidence；(3) Living Taxonomy：4,931 节点的原理/理论/机制/现象树，360K 篇论文放置，可弃权(提议新分支) | 分类路径靠 canonical L1 路由、同义词归一、模糊聚类。主张本身**不去重** | 证据图边类型专家抽检：精度 97.9%(143/146 可判边)。"100% 主张可溯源" 原文自认只是 traceability，不是语义正确性。AskChem-Bench 30 题(条件聚合/时间追踪/冲突呈现)：DOI 可解析 100% vs 88.3%，引用密度 18.1 vs 9.6 | 原文局限：只覆盖化学一部分且多为摘要；分类放置**未做专家验证**；字符串归一 "can merge distinct categories or retain near-duplicates"。另外：无共识聚合、无独立性计数、时间只是导航分面 |
| **ASKS**(Shi-Ju Ran 等) | [2608.29612](https://arxiv.org/abs/2608.29612)。v1 2026-08-30 | Wiki 视图(确定性骨架 + LLM 导航摘要 + 结构化内容)。机器语义槽为 S-P-O 形式，单独校验：谓词注册、重复、引文碎片、裸缩写、描述性短语。GraphDelta 为文档局部的待提交变更：候选边、需要身份消解的边界 mention、规范端点、硬错误 | canonical node；同边再现时把新页加入 origin record("accumulate independent source contributions")；Hub 为持久导航区域，入 Hub 门槛高于留 Hub 门槛(滞回)，有父子谱系 | 嵌入门：label 相似 ≥0.92、语义 ≥0.90、合并门 ≥0.91。命题级：高相似复用；中间带记录语义关系并保留两节点 | 单一研究项目 56 篇按时间顺序编译(2010–2026)：累计复用率 0.1885(204 复用 / 878 新建 / 5 弃权)；18 个 Hub 诞生，0 次分裂/合并/退休；旧节点 churn 均值 0.00467；映射覆盖 89.3%。**无 gold 或人工评测** | 原文自认只评了"这一条轨迹"的形成，乱序重建、语义空模型对照、与其他构图方法比较都留作未来工作。"independent" 只是 distinct work ID，不是作者/团队独立。规模极小 |
| **Mechanist** 的 13k KG(Wang…Huajun Chen) | [2608.12036](https://arxiv.org/abs/2608.12036)。v3 2026-09-06 | 三轴：研究对象(模型/组件/行为现象)、应用场景、方法(可解释性技术)；外加显式陈述的发现、局限、未来方向。"unsupported or ambiguous [attributes] are left unspecified rather than inferred" | 论文/作者/概念/引文图，方法-组件-任务-发现之间的关系；用于找 "sparsely studied combinations"。沿用 SciAtlas 的 OpenAlex 管线 | 同义标签归一、合并等价概念、OpenAlex ID | LLM 评判(Claude Opus 4.7)查相关性、抽取属性的溯源、分类一致性；3 名标注者 100 条记录，准确率 90%+ | 自建的理由是："General-purpose academic taxonomies do not resolve the model components, behavioural settings and analysis methods…"。无冲突处理，未陈述 KG 局限 |
| **SciAtlas**(Qiao…) | [2605.22878](https://arxiv.org/abs/2605.22878)。v1 2026-05-20；v2 08-29 改名 "A Computable Atlas of Science for Knowledge-Grounded AI Research" | 元数据、OpenAlex 主题层级、关键词概念(机制/材料/方法/数据集/任务/模型/实验设置)、同行评审信号。**不抽发现/局限/比较** | 43M 篇、157M 实体、3B 三元组。边：引用、论文-关键词(带权)、关键词共现、主题归属、作者/机构。五层：evidential / conceptual / disciplinary / expertise / normative | 同义词与格式变体映射到规范关键词节点；作者保守消歧 | 只有下游任务 | 原文局限："experimental procedures, conditions, measurements, and outcomes are not yet represented as explicit objects"；周期性更新而非连续 |
| **Lacuna**(Weiss…Larochelle, Rahaman) | [2606.26246](https://arxiv.org/abs/2606.26246)。v1 2026-06-24 | core-idea 摘要(markdown)；concept element = 1–2 句的核心思想、方法、局限或经验观察，共 15.26M 条 | concept element 经嵌入 + HDBSCAN 得 27,017 个 research direction，每个约 2 页综合，内容为 "recurring problem, method family, or opportunity area"；另有 38K 研究提案。规模 733,795 篇(ML) | 以 OpenReview 为身份锚，概念层靠聚类 | 只有下游：ReportBench-ML citation F1 0.052 vs GPT-Researcher 0.039。另有一次 5 条主张的审计 | 原文："batch-built map, not an online updating system"。族是聚类摘要，无成员判定 gold，无共识/独立性/时间 |
| **Intern-Atlas** | [2604.28158](https://arxiv.org/abs/2604.28158) | 从 Method 章节抽方法名(247 种子 → LLM 扩展，至少 3 篇才成立) | 9.4M+ 类型化方法边。强因果：extends / improves / replaces / adapts；语境：uses_component / compares / background。因果边存 bottleneck / mechanism / trade-off 的逐字引文与置信度；确定性检查做子串匹配与年份序 | 别名表：8,155 个规范方法、9,545 个表面形式；最长匹配加词边界，歧义名(如 Mamba)人工处理 | **有**：30 篇综述 gold(2,268 节点 / 1,462 边 / 133 条专家链)。NMR 91.0%、ERR 89.7%、PSC 92.0%；谱系检索节点召回 84.8%、边召回 79.0% | 只有二元边，无 n-ary 条件；无"族"节点与成员；无共有局限/共识；边无独立性计数；时间只是年份；作者自认分不清真研究空白与表示空白 |

### 2.2 单篇抽取 schema 与科学 IE(含早期 KG)

| 工作 | 一手 ID | 单篇 schema | 跨论文对象 | 身份处理 | 直接评测 | 缺什么 |
|---|---|---|---|---|---|---|
| **SciREX** | [2005.00512](https://arxiv.org/abs/2005.00512)，ACL 2020 | 4 类实体(Method / Metric / Task / Material) + 文档内共指成 salient cluster + 文档级 4 元结果关系(含 Score)；438 篇全文，99% 的 4 元关系跨句 | 无 | 仅文档内共指 | 端到端：salient cluster F1 0.307、二元关系 0.096、4 元关系 0.008(给定 gold cluster 时 0.268)；repo 注明约 50% 关系含只在表格出现的实体(被丢弃) | 无跨文档身份、无局限、无条件 |
| **ORKG** contributions / comparisons | [1901.10816](https://arxiv.org/abs/1901.10816)；comparisons：[2006.01747](https://arxiv.org/abs/2006.01747)，JCDL 2020 | research contribution 由属性构成：problem / materials / methods / results，众包加工具 | comparison 表：先找相似 contribution，再对齐属性 | 人工策展；LLM 辅助的属性推荐见 [2405.02105](https://arxiv.org/abs/2405.02105)，属性对齐到已有 ORKG URI 见 [2502.10768](https://arxiv.org/abs/2502.10768) | infrastructure 只有定性用户研究；comparisons 用已发表综述数据评测 | 比较靠策展而非计算；无独立计数、无状态、无时间 |
| **NLPContributionGraph** | SemEval-2021 Task 11，[2106.07385](https://arxiv.org/abs/2106.07385) | contribution 句 → 短语 → 三元组(喂 ORKG) | — | — | 最佳 F1：句 57.27 / 短语 46.41 / 三元组 22.28 | 三元组级精度低 |
| **SciER** | [2410.21155](https://arxiv.org/abs/2410.21155)，EMNLP 2024 | 106 篇全文，实体 3 类(Dataset / Method / Task)，关系 9 类(EVALUATED-WITH、COMPARE-WITH、SUBCLASS-OF、BENCHMARK-FOR、TRAINED-WITH、USED-FOR、SUBTASK-OF、PART-OF、SYNONYM-OF) | — | SYNONYM-OF 只覆盖文档内别名 | 监督：NER 86.85、端到端 RE 61.10(分布内)；Qwen2-72B few-shot：NER 71.44、RE 41.22 | **无"本文提出"标签** |
| **GSAP-ERE** | [2511.09411](https://arxiv.org/abs/2511.09411) | 100 篇 ML 全文，10 类实体、18 类关系(训练、数据使用等) | — | — | 微调：NER 80.6、RE 54.0；LLM 提示：44.4 / 10.1 | LLM 零/少样本在细粒度 RE 上很差 |
| **MASSW** | [2406.06357](https://arxiv.org/abs/2406.06357) | paper card 5 面：context / key idea / method / outcome / projected impact，覆盖 152K 篇 CS(1969–2024) | — | — | GPT-4 与人工(126 篇)余弦约 0.89–0.94 | 无跨论文对象 |
| **LimitGen** | [2507.02694](https://arxiv.org/abs/2507.02694)，ACL 2025 | 局限 4 类：methodology / experimental design / result analysis / literature review；Syn 集 1,000 条 + Human 集 1,000 条(来自 ICLR 2025 审稿) | — | — | GPT-4o 找到 52% 的合成局限，人 86%；RAG +12.2 | 只有单篇局限 |
| **CLAIMCHECK** | [2503.21717](https://arxiv.org/abs/2503.21717) | 41 篇被拒 NeurIPS 论文：168 条 weakness 链接到 154 条目标主张 | — | — | LLM 在 weakness→claim 链接与验证上落后专家 | 单篇 |
| **NSF-SciFy** | [2503.08600](https://arxiv.org/abs/2503.08600) | 400K 份 NSF 摘要中的 2.8M 主张(零样本) | — | — | 精度高、召回较低 | 单篇 |
| **CS-KG** / **AI-KG**(Dessì, Osborne 等) | CS-KG：ISWC 2022；AI-KG：ISWC 2020(官方 PDF) | 从摘要抽 Method / Task / Material / Metric / OtherEntity 实体与陈述(DyGIE++、CSO、OpenIE) | **陈述实体化，`hasSupport` = 抽出该陈述的论文数，并附来源**；support ≥3 视为 reliable，其余交 MLP 判定。规模 41M 陈述、6.7M 篇(门户现为 67M / 14.5M) | 词形还原、缩写消解、sentence-transformer 余弦 ≥0.9 合并、映射 CSO、链接 DBpedia / Wikidata | 1,200 条陈述 3 名专家标注(Fleiss κ 0.435)：P 0.76 / R 0.77 / F1 0.76 | **最早的"支持数"字段，但只是原始共现篇数，不判独立**；无条件、无时间、无争议 |
| **NLP-AKG** / **ResearchPulse** | [2502.14192](https://arxiv.org/abs/2502.14192) / [2509.03565](https://arxiv.org/abs/2509.03565)(ACM MM 2025) | NLP-AKG：60,826 篇 ACL 的 few-shot 抽取；ResearchPulse：跨文档的 motivation → method → experiment 链 | 引文/概念链接；论文簇级链条 | — | 只有下游 | 无聚合对象评测 |

### 2.3 跨论文身份：实体链接、规范化、方法归属

| 工作 | 一手 ID | 解决什么 | 方法 | 精度 | 缺什么 |
|---|---|---|---|---|---|
| **TDMS-IE / AxCell / SciLead** | 见 §2.5 | 把论文结果归到 (Task, Dataset, Metric) 键 | 封闭分类；冷启动聚类 | SciLead 预定义设定：Task 90.86 / Dataset 89.22 / Metric 87.31 / Result 69.26 F1，完整元组 55.27 | 键空间封闭或评测时映射回 gold |
| **Linked Papers With Code** | [2310.20475](https://arxiv.org/abs/2310.20475)，ISWC 2023 | PwC 的 RDF 化：约 376K 篇、8.3K 数据集、2.1K 方法 | — | 只有 158 个数据集(2%)链到 Wikidata | 未见 "introduced-by" 来源属性 |
| **ChatPD** | [2505.22349](https://arxiv.org/abs/2505.22349)，KDD ADS 2025 | 数据集 mention 的跨论文身份 | LLM 抽取 + 图式实体消解 | 约 90% 精度/召回 | 只覆盖数据集 |
| **DMDD** / **SciDMT** | [2305.11779](https://arxiv.org/abs/2305.11779)，TACL 2023 / LREC-COLING 2024 | 数据集(及方法、任务)mention 检测与链接 | 弱监督 | DMDD：450 篇人工评测集；SciDMT 只做检测 | 不涉及方法版本/变体 |
| **SciAD** | COLING 2020(SDU@AAAI-21 的基础) | 科学文本缩写识别与消歧 | — | 62,441 条消歧样本 | 句级，不做跨论文聚合 |
| **CESI** | [1902.00172](https://arxiv.org/abs/1902.00172)，WWW 2018 | 开放 KB 名词/关系短语规范化 | 嵌入 + 侧信息聚类 | — | 通用域 |
| **Intern-Atlas** 别名表 | 见 §2.1 | 方法名规范化 | 8,155 规范方法 / 9,545 表面形式，最长匹配加人工歧义处理 | 未单独报身份精度 | 版本/变体(如 v1/v2、-large)未讨论 |
| **CiteME** | [2407.12861](https://arxiv.org/abs/2407.12861)，NeurIPS 2024 D&B | 由引文语境找被引论文(130 条) | agent 检索 | 人 69.7%；CiteAgent(GPT-4o)35.3% | 是"找被引"而不是"谁提出" |
| **REASONS** | [2405.02228](https://arxiv.org/abs/2405.02228) | 句级引文归属(12,723 条) | RAG | 高级 RAG 仍有 65.4% 幻觉 | 同上 |
| **MIR**(Methodology Adjacency Graph) | ACL 2025 | 沿引文追方法谱系，用于方法论灵感检索 | 图 + 检索 | Recall@3 +5.4、mAP +7.8 | 是检索增益，不是归属准确率 |
| **SciMON** | [2305.14259](https://arxiv.org/abs/2305.14259)，ACL 2024 | 灵感检索用于生成想法 | — | — | 不做归属 |
| **S2AND** | [2103.07534](https://arxiv.org/abs/2103.07534)，JCDL 2021 | 作者消歧(邻近) | — | B³ F1 误差降一半以上 | 作者层，可为"独立性"提供作者身份 |

身份小结：
- **方法提出者归属(谁提出了 X)没有任何已核验工作直接评测过准确率。** 最接近的是 Intern-Atlas 的方法节点溯源、CiteME 的找被引、MIR 的谱系检索。
- 候选数据集 MPR(JCDL 2024，"Method Entities and Scientific Papers" 关系数据集)可能含提出/使用标签，但一手页面被挡，未核实。
- 方法版本/变体(BERT-base vs RoBERTa vs "our BERT variant")的身份稳健性没有工作专门测。

### 2.4 聚合对象 I：主张关系、共识与矛盾

| 工作 | 一手 ID | 单篇单元 | 跨论文对象 | 身份处理 | 直接评测 | 缺什么 |
|---|---|---|---|---|---|---|
| **ClaimFlow**(Pramanick, Hou, Mohammad, Gurevych) | [2603.16073](https://arxiv.org/abs/2603.16073)。v2 2026-06-12 | 摘要、引言、结论中的主张(轻度编辑)。v2 有 5,689 条人工标注，来自 1,617 篇 ACL(1979–2025)；v1 为 304 篇 / 1,084 条 | 被引→引用的主张边 4,871 条：support / extend / qualify / refute / background。自动管线扩到约 13k 篇 | 依赖引文语境定位被引主张；规模化时近重复主张做轻量 canonicalization 聚类(未单独评测) | 人工 gold，κ=0.77；GPT-4.1 关系分类 macro-F1 0.81。领域发现：63.5% 主张从未被复用，11.1% 被挑战过，refute 仅 2.4%，首次被挑战中位 22 年 | 只覆盖被引用到的主张；qualify 只是标签，不是结构化条件；无每主张的共识/争议状态、无独立支持计数、无 as-of 状态；无方法族/局限对象 |
| **SciFact / SciFact-Open** | [2004.14974](https://arxiv.org/abs/2004.14974)(EMNLP 2020)；[2210.13777](https://arxiv.org/abs/2210.13777)(Findings EMNLP 2022) | 主张-摘要对 + rationale 句。标签 SUPPORTS / REFUTES / NOINFO | 无聚合：每对独立标注。Open 版 279 主张对 500K 摘要，IR 式 pooling 构造 gold | — | abstract 级 / sentence 级 F1；Open 设定下系统掉 15–30 F1。分析发现：多证据主张中约 20% 同时有支持与反驳；specificity 错配常见(证据人群更窄或更宽) | 支持与反驳从未合成为共识对象。条件错配只在分析中观察到，gold 按设计不完备(pooling) |
| **2606.07570**(Cheng…Xiao-Gang Wen, Mingda Li) | [2606.07570](https://arxiv.org/abs/2606.07570)，"Can LLMs extract scientific consensus? A case study in high-temperature superconductivity" | 每篇一个 9 维机制意见向量(0–5)，加证据模态(理论 9 / 计算 6 / 实验 22 类)、材料族(9 类)、年份与引用增长 | **计算出来的共识**：意见分按引用加权，指数时间衰减偏向新工作；按材料族的机制画像；按年代切片的机制份额冲积图。范围约 18k 高被引篇(1950–2025)，5,446 篇非零 | 封闭机制分类：BERTopic 初得，专家迭代 | **无定量 gold**。只有"与已知物理理解一致"，加 50 篇的提示/释义/温度/换模型稳健性检验(排序稳定) | 原文局限："does not fully distinguish factual findings from author interpretations, nor does it resolve whether a citation is supportive, critical, partial, or merely contextual"。按引用加权而非按独立团队；单领域、固定分类 |
| **BioConflict** | [ACL 2026.bionlp-1.44](https://aclanthology.org/2026.bionlp-1.44/) | 250 对专家标注论文对(500 篇摘要)，10 个主题 | 三个任务：文档级冲突检测；抽取解释冲突的语境变量(剂量、细胞类型、研究设计)；生成共识综合 | — | 冲突检测 F1 最高 0.89；领域模型在生成任务上更差 | 只有对，不是集合；"共识"是生成的散文，不是计算出的标签；无计数、无时间 |
| **Sosa et al.** | [2212.09867](https://arxiv.org/abs/2212.09867) | CORD-19 药物疗效主张 | 778 对，标签 entailment / contradiction / neutral | 药物/主题过滤 | PubMedBERT macro-F1 0.690，矛盾召回 0.571；专家 κ 0.83–0.84 | 只有成对判断 |
| **Alamri & Stevenson 2016** | [PMC4897929](https://pmc.ncbi.nlm.nih.gov/articles/PMC4897929/)，J. Biomed. Semantics | 心血管摘要主张(YS/NO) | 用 24 篇系统综述的森林图作 gold，每篇综述一个 PICO 问题 | 共享 PICO 问题作"同一问题"锚 | 主张识别一致率 92%，断言值 97% | 二值极性；无条件与时间。"以元分析为 gold" 的思路可借 |
| **scite Smart Citations** | Nicholson et al., QSS 2(3) 2021，[bioRxiv 10.1101/2021.03.15.435418](https://www.biorxiv.org/content/10.1101/2021.03.15.435418v1) | 引用语句 → supporting / contrasting / mentioning | 被引论文上各类引用语句的计数 | 目标粒度是整篇论文，不是主张 | 留出集 F1：supporting 80.98，disputing 58.97(召回 46.36)，mentioning 97.52 | 无主张级身份、不数独立引用方、无条件、无 as-of 状态 |
| **Consensus Meter** | [官方博客 2024-01-15](https://consensus.app/home/blog/introducing-the-consensus-meter/) | 对 yes/no 问题，LLM 给 top-20 结果标 yes / no / possibly | 计数比例 | — | 官方自称误分类约 10%，无评测协议 | 自认不按研究质量加权(元分析与个案同权)、丢失语境(儿科结果被当成人)；无独立性、无条件、无时间 |

### 2.5 聚合对象 II：对比表与排行榜

| 工作 | 一手 ID | 单篇单元 | 跨论文对象 | 身份处理 | 直接评测 | 缺什么 |
|---|---|---|---|---|---|---|
| **ArxivDIGESTables** | [2410.22360](https://arxiv.org/abs/2410.22360)，EMNLP 2024 | 标题 + 摘要 | 论文 × 方面的表(先生成 schema，再填值) | 评测端用 DecontextEval：LLM 生成列描述后做嵌入对齐 | gold 为 2,228 张真实综述表(7,542 篇)；指标为 schema 召回 + 值准确率。值准确率低；人评认为未匹配的生成列与 gold 列同样有用 | 仅摘要；行与行之间无关系(谁胜谁、谁扩展谁)；无支持聚合、无时间 |
| **arXiv2Table** | [2504.10284](https://arxiv.org/abs/2504.10284)，ACL 2026 | 论文全集，含人工核验的干扰论文 | 综述表 | 评测端用双向 QA 对(gold→系统、系统→gold) | 1,957 张表 / 7,158 篇；覆盖 schema、单元格保真、**成对关系一致性**。最好的选文 F1 约 73%，成对一致性 F1 常 <50% | 有成对关系的评分，但无共识/族/演化对象 |
| **TDMS-IE**(Hou et al.) | [ACL P19-1513](https://aclanthology.org/P19-1513/) | (Task, Dataset, Metric, Score) | 排行榜 | 来自 NLP-progress 的封闭分类 | 三元组 micro-F1 66.0；含分数的完整四元组 micro 仅 11.8 | 封闭域 |
| **AxCell** | [2004.14356](https://arxiv.org/abs/2004.14356)，EMNLP 2020 | 结果表单元格 | 把单元格链接到已知 PwC 排行榜 | 链接到已知 (task, dataset, metric) | NLP-TDMS 四元组 micro-F1 25.8；PWC Leaderboards 28.7 | 作者自认封闭域，所有排行榜须预先已知 |
| **SciLead**(Şahinuç et al.) | [ACL 2024.emnlp-main.453](https://aclanthology.org/2024.emnlp-main.453/) | TDM 三元组 + 结果 | 排行榜，分三种设定：全预定义 / 部分预定义 / 冷启动 | 冷启动从空聚类，评测时再映射回 gold(对身份错误偏宽容) | 43 篇 / 27 榜 / 138 三元组；指标为排行榜召回 LR、论文覆盖 PC、结果覆盖 RC、排序重叠 AO。GPT-4 冷启动：LR 81.48 / PC 58.74 / RC 46.13 / AO 48.15 | 数值抽取是瓶颈；规模小 |
| **LEGOBench** | [2401.06233](https://arxiv.org/abs/2401.06233) | — | 排行榜生成与排序 | 以 PwC 的 ⟨D,T,M⟩ 为键 | 11,470 榜、70,559 条目；GPT-4 方法召回 25.24%，Kendall τ 接近 0 | gold 继承 PwC 的不完整 |
| **LAG → League** | [2502.18209](https://arxiv.org/abs/2502.18209)。v2 改名 "League: Leaderboard Generation on Demand" | (title, dataset, metrics, results, settings) 五元组 | 按需排行榜 | 按频次取前 5 数据集、单位换算、保留设置以保证公平 | 无 gold 榜，用 LLM 评判(与人 Pearson 最高 0.76) | 身份靠频次而不是消解 |

### 2.6 聚合对象 III：分类体系归纳与论文归属

| 工作 | 一手 ID | 单篇单元 | 跨论文对象 | 身份处理 | 直接评测 | 缺什么 |
|---|---|---|---|---|---|---|
| **TaxoAdapt** | [2506.10737](https://arxiv.org/abs/2506.10737)，ACL 2025 | 标题 + 摘要，按 5 个维度多标签分类(Task / Methodology / Datasets / Evaluation / Domain) | 每维一棵树，节点带论文成员；按会议年份快照 | 伪标签做 taxonomy-aware 聚类 | **无 gold**，GPT-4o 评粒度/兄弟一致/维度对齐/覆盖，与人一致 70–90% | 节点是主题标签，不是有谱系的方法族；时间只是分离的快照 |
| **HiGTL** | [2410.03761](https://arxiv.org/abs/2410.03761)，题为 "Taxonomy Tree Generation from Citation Graph" | 嵌入 + 引文 | 引文图递归聚类成树，LLM 命名 | 簇级 | 518 篇 arXiv CS 综述 gold；第 1 层聚类 F1 0.7127；BERTScore 0.8694 | 只有主题树 |
| **Chain-of-Layer** | [2402.07386](https://arxiv.org/abs/2402.07386)，CIKM 2024 | 给定的实体集 | is-a 层级 | — | 对 WordNet / DBLP / SemEval-Sci gold 算 Ancestor-F1 与 Edge-F1(WordNet Edge-F1 79.62)；实体多于约 80 时性能显著下降 | 无论文、无成员 |
| **TaxoBench** | [2601.12369](https://arxiv.org/abs/2601.12369)，题为 "Can Deep Research Agents Retrieve and Organize? …" | — | 评测基准 | — | 72 篇 LLM 综述的分类图转录，3,815 篇被引论文映射到**单一**叶节点。指标：ARI / V-measure、US-NTED、Sem-Path。最佳 agent 只召回 20.92%；层级深度 2.99–3.97 vs 专家 4.86；Sem-Path 32–42% vs 人 54.09% | 单标签成员；只有主题；未报抽取的标注者一致性 |
| **TaxoAlign** | [2510.17263](https://arxiv.org/abs/2510.17263)，EMNLP 2025 | 摘要知识切片 | 分类树(**不把论文挂到节点**) | — | CS-TaxoBench(ACM Computing Surveys 标题树)；Node Soft Recall 0.2974 vs AutoSurvey 0.1784 | 无成员 |
| **CHIME** | [2407.16148](https://arxiv.org/abs/2407.16148)，Findings ACL 2024 | 摘要生成主张(98.1% 被摘要 NLI 蕴含) | 每主题至多 5 棵层级，主张多标签挂到节点 | — | 472 篇 Cochrane 综述，专家修正 100 主题：父子链接 99.9% 正确、兄弟组 77% 一致、研究归属 P 0.71 / R 0.53 | 唯一以主张为单元的分类工作；仍无共识/争议标签与独立计数 |
| **Knowledge Navigator** | [ACL 2024.findings-emnlp.516](https://aclanthology.org/2024.findings-emnlp.516/) | — | 检索结果做 2 层主题树 | — | SciTOC(50 篇 Annual Reviews 目录)覆盖 71.6% | 浅层主题 |
| **EvoTree** | [2609.09561](https://arxiv.org/abs/2609.09561)，2026-09-09 | 引文图 | 分类骨干 + 时间微调：把过渡("marginal")论文挂到内部节点，满足单调路径约束 | — | 411 篇综述引文图；352 篇人工标注；NMI 0.526，引文方向准确率 0.893，marginal AUROC 0.726 | 最接近"带时间的成员关系"；边仍是树进展，无类型 |
| **Drift Inspector** | [2609.39710](https://arxiv.org/abs/2609.39710)，2026-09-30 | Atomic Contribution Claims(从摘要抽取的去语境化贡献命题)，346k 条 / 80k 篇 | SPECTER2 + HDBSCAN 跨年份联合聚类；**论文级流行度**("a paper with ten RAG claims counts once") | 聚类即身份 | 人工 136 条：90.5–97.8% Good，κ 0.844；对 SToP gold(653 篇)纯度 0.689 vs BERTopic 0.324 | 只有摘要；"按论文计一次"是最朴素的去重计数，不涉及作者独立 |

### 2.7 图式多文档综合与 n-ary / 超图

| 工作 | 一手 ID | 单元 | 跨文档对象 | 身份 | 直接评测 | 缺什么 |
|---|---|---|---|---|---|---|
| **GraphRAG** | [2404.16130](https://arxiv.org/abs/2404.16130) | 块级实体与关系 | Leiden 社区层级(C0–C3)及预生成的社区摘要 | 精确字符串匹配 | 只有下游 LLM 两两评判(全面性/多样性)，社区结构本身从不打分；语料不是科学文献 | 社区是词汇图划分，不是方法族 |
| **LightRAG** / **HippoRAG 2** | [2410.05779](https://arxiv.org/abs/2410.05779) / [2502.14802](https://arxiv.org/abs/2502.14802) | 块级实体与关系 | 增量并集图 / PPR 记忆 | 去重函数 | 只有下游 | 同上 |
| **HyperGraphRAG** / **Hyper-RAG** | [2503.21322](https://arxiv.org/abs/2503.21322)(NeurIPS 2025) / [2504.08758](https://arxiv.org/abs/2504.08758) | 超边 = 自然语言事实 + 多个实体 | 超图 | 未说明 / LLM 合并 | 只有下游 | 超图本身不评 |
| **Hyper-KGGen** | [2602.19543](https://arxiv.org/abs/2602.19543)，KDD'26 | 二元、带限定的二元(时间/地点/条件)、一般 n-ary 三种粒度 | 文档级超图 | 跨块共指 | HyperDocRED(通用域)：P 0.80 / R 0.43 / F1 0.56 | 无跨文档聚合 |
| **Dagdelen, Dunn et al.** | Nat. Commun. 15:1418(2024)，[doi:10.1038/s41467-024-45563-x](https://doi.org/10.1038/s41467-024-45563-x) | 嵌套 JSON n-ary 记录(主体 + 掺杂 + 修饰 / MOF + 客体 + 应用) | — | 隐式归一 | 掺杂任务 F1 0.849 | 只有单篇 |
| **StarE** | [2009.10847](https://arxiv.org/abs/2009.10847)，EMNLP 2020 | Wikidata 式带限定词陈述 | — | — | 链接预测 MRR 提升最多 25 点 | 条件表示的先例，不做抽取 |
| **Typed Claim Network**(Ding et al.) | [2605.30966](https://arxiv.org/abs/2605.30966)，2026-05-29 | 把引用实例化为对象：源、目标、主张文本，加立场标签 CRITIQUE / ADOPTION / BENCHMARK / NEUTRAL 与态度标签 | 127 篇点云分割、8,260 条 typed claim；支持"最被批评论文""桥接论文"等查询 | — | 150 个窗口单人抽检：立场 macro-F1 0.892、态度 κ 0.735(arxiv html 一手核实)；作者自认是单标注者，且与抽取器共用同一提示 | 节点是论文对，不是主张身份；问答任务结果为零效果 |

### 2.8 与"独立性 / 条件 / 时间"直接相关的邻近工作

| 工作 | 一手 ID | 要点 | 与本项目的关系 |
|---|---|---|---|
| **Authorship network bias in meta-analysis**(Rieck, Mupepele, Dormann) | Research Synthesis Methods 17(4), 2026，[doi:10.1017/rsm.2025.10063](https://doi.org/10.1017/rsm.2025.10063) | 原始研究之间的作者重叠造成效应量不独立。三步法：作者网络距离(测地/Jaccard)对效应相似做相关图；作者相似矩阵作 GLS 方差协方差；复检 | **独立性计数在元分析方法学里有定义与校正法**，但没有进入任何 LLM 知识层 |
| **GraphEcho** | [2609.17695](https://arxiv.org/abs/2609.17695)，2026-09-15 | 受控实验：LLM 图 agent 把重复路径误当额外佐证；provenance-aware 后训练能减少重访，但科学主张上准确率下降 | 直接证明"不做来源去重会被重复证据骗"，但只在合成图与 agent 行为层面 |
| **Nanopublication** / **Cardinal assertions** | Groth, Gibson, Velterop, Inf. Serv. Use 2010，[doi:10.3233/ISU-2010-0613](https://doi.org/10.3233/ISU-2010-0613)；"Towards Computational Evaluation of Evidence for Scientific Assertions with Nanopublications and Cardinal Assertions"(Leiden/Wageningen/Nijmegen，Schultes、Roos、Mons 等)，[CEUR-WS Vol-952](https://ceur-ws.org/Vol-952/paper_26.pdf) | nanopub = 断言 + 出处 + 发布信息。cardinal assertion = 多个 nanopub 中同一断言聚合得到的证据分，并提出"随时间评估证据" | 支持集 + 时间演化的最早**设计**先例(语义网时代、无 LLM、无评测) |
| **EAKR / MetaSynDec**(Li, Mathrani, Susnjak) | [2608.01711](https://arxiv.org/abs/2608.01711)，2026-08-03 | 元分析前把证据分配、对比、结局/时间点对齐、效应量公式显式化为可执行表征。58 个综合单元：完整对象保真 67.9%，证据集完全一致 75.0%，Jaccard 0.909，CI 重叠 98.2% | **以已发表元分析为 gold 直接评测"证据集"**，这是可借的评测范式 |
| **MetaSyn** | [2606.17041](https://arxiv.org/abs/2606.17041) | 422 篇 Nature Portfolio 元分析作 gold(纳入研究 + 资格标准 + 干扰项) | 可作"支持集召回"的 gold 来源 |
| **AutoSynthesis** | [2607.15247](https://arxiv.org/abs/2607.15247) | 抽效应量，做随机效应元分析与调节变量分析，与专家 Hedges' g 对照 | 定量聚合有统计学先例，但限于 RCT 式研究 |
| **Applicability Condition Extraction** | [2606.14031](https://arxiv.org/abs/2606.14031)，Findings ACL 2026 | 药物-疾病-适用条件三元，1,119 对人工标注 | 单篇条件抽取已有数据集 |
| **Tree-of-Concerns** | [2608.20777](https://arxiv.org/abs/2608.20777)，Findings EMNLP 2026 | 未陈述局限抽取；ToC-Bench 414 篇、1,905 条，gold 来自审稿意见与后续引用批评 | 局限的 gold 可从**后续论文的批评**来，可借作"共有局限"gold |
| **What Limits Us?** | [2609.15191](https://arxiv.org/abs/2609.15191)，Findings EMNLP 2026 | ACL/EMNLP 2020–2025 自述局限，人机混合编码，看时间趋势 | 领域级局限聚合的人工分析先例(不是系统) |
| **Time-Aligned Evolving Concept Graphs** | [2609.18163](https://arxiv.org/abs/2609.18163) | 187,848 篇、概念共现图，按论文日期回放，预测首次共现与关系形成；AUROC 0.929→0.972 | 时间回放式评测协议可借，但对象是共现 |
| **MemStrata** | [2606.26511](https://arxiv.org/abs/2606.26511) | 双时态账本 + 确定性 supersession，RAG 有 15–40% 时候给出过期值 | 通用 agent 记忆，不是科学文献 |
| **Knowledge Pull Requests** | [2609.26634](https://arxiv.org/abs/2609.26634)，Van Durme 组 | 新主张路由到章节并标冲突，ChangeLog 把"知识变化"与"文本变化"分开 | 增量更新加冲突标记的形态先例(Wikipedia) |
| **DeepSynth-Eval** | [ACL 2026.findings-acl.1688](https://aclanthology.org/2026.findings-acl.1688/) | 用综述参考文献构造 oracle 上下文，用 checklist 测综合 | 隔离检索、只测综合的评测设计 |

## 3. 聚合层如何被直接评测：gold 来源与指标

| 聚合对象 | 已有直接评测 | gold 来源 | 指标 | 代表 |
|---|---|---|---|---|
| 分类树 + 论文成员 | 有，最成熟 | 综述的分类图或章节树；Cochrane 综述；Annual Reviews 目录 | Edge/Ancestor-F1、ARI/V-measure、NMI、树编辑距离、Sem-Path、节点软召回、聚类纯度 | TaxoBench、HiGTL、TaxoAlign、CHIME、Drift Inspector、EvoTree |
| 方法演化边/链 | 有 | 综述中专家整理的方法链 | NMR / ERR / PSC、谱系节点与边召回 | Intern-Atlas；THE-Tree(gold 由系统输出精修，循环) |
| 排行榜 / 结果聚合 | 有 | Papers with Code、NLP-progress、人工小集 | 四元组 F1、LR / PC / RC / AO、Kendall τ | TDMS-IE、AxCell、SciLead、LEGOBench |
| 对比表 | 有 | arXiv 真实综述表 | schema 召回、值准确率、双向 QA F1、成对一致性 | ArxivDIGESTables、arXiv2Table |
| 主张间关系边 | 有，边级 | 人工标注引文对 | κ、关系 macro-F1 | ClaimFlow、scite、Sosa |
| 证据集(元分析单元) | 有，限生物医学 | 已发表元分析或系统综述的纳入研究 | 证据集一致率、Jaccard、CI 重叠 | EAKR、MetaSyn、Alamri & Stevenson |
| 共识 / 争议状态 | **基本没有** | — | 2606.07570 只有定性一致；Consensus 自报 10% 误差、无协议；BioConflict 是对级加生成 | — |
| 方法族(含成员、共有局限、比较) | **没有** | — | ScholarStack L2/L3、LKM workflow 族、Lacuna directions 均未直接评测 | — |
| 抽取层本身 | 抽样审计 | 人工或 LLM 评判 | LKM 幻觉率 0.97%(不测遗漏)；Mechanist 90%+；AskChem 边精度 97.9% | — |
| 身份/规范化对聚合的影响 | **没有** | — | 只有 SciLead 冷启动做"映射回 gold"，反而对身份错误宽容 | — |

可借的 gold 来源归纳：
1. **综述**：分类图、方法链、对比表都来自这里，已被用透。
2. **系统综述/元分析**：用于证据集与支持集，生物医学已有，CS 领域稀缺。
3. **后续论文的批评引用**：Tree-of-Concerns 用来做局限，ClaimFlow 用来做 refute/qualify。
4. **时间回放**：EvoTree、2609.18163 用；按截止时间冻结，用后来的事实当 gold。
5. **PwC 类结果库**：PwC 已停服，存档不完整。

## 4. 综合

**单篇层已经收敛。** 七个系统的单篇 schema 交集是：主张/事实(去语境化) + 证据锚(逐字或定位符) + 类型(finding / method / hypothesis / limitation) + 条件/setting + 校验状态。各家的差异只在附加件：LKM 的推理链与 Weak point、Agents-K1 的引文意图与谱系关系、AskChem 的领域结构字段、Mechanist 的领域三轴。"提出 vs 使用"在 Agents-K1 中以 BUILDS_ON / USES_COMPONENT 出现，在 Intern-Atlas 中以 extends 与 uses_component 区分。但没有系统报告"方法提出者归属"的准确率(§2.3 见子代理结果)。

**跨论文层分化为三种形态。**
1. **身份合并型**：LKM 精确哈希、ASKS 嵌入门、AskChem 分面归一、Agents-K1 嵌入链接。产物是"同一对象 + 来源出现列表"，聚合到此为止。
2. **聚类摘要型**：LKM 的 Question/Workflow 族、Lacuna 的 research direction、GraphRAG 社区、Drift Inspector 簇。产物是一段综合文字加成员，成员判定没有 gold。
3. **设计上的类型化综合对象**：ScholarStack L3 的 account / scope / basis，basis 标 supporting / opposing / limiting；LKM 的 contradiction group。这一形态最接近"领域状态"，但两家都写明或实际上没有直接评测。

**直接评测与对象价值成反比。** 越是窄、越有现成 gold 的对象(分类树、演化边、排行榜、对比表)，评测越成熟。越是本项目想要的对象(方法族加共有局限、条件化共识、争议、时间状态)，越没有 gold、没有指标。这既是空白，也是风险：做出来如果拿不出直接评测，会重蹈 LKM/ScholarStack"只报下游"的覆辙，而审稿人已经能拿这两篇的自述局限来对照。

**独立性在三个圈子里各说各话。**
- 元分析方法学：作者网络偏差，有统计校正。
- agent 行为研究：GraphEcho，重复路径被当佐证。
- 语义网：cardinal assertion，聚合多来源的证据分。
- LLM 科学知识层里，"支持 = 出现次数"，最好的做法也只是按论文去重(Drift Inspector、ASKS distinct work)。按作者/团队/数据集/代码库去重、区分"复现"与"转述"，没有系统做，也没有评测。

**条件是聚合判定的前提，但没人在聚合层验证它。** SciFact-Open 实测多证据主张中约 20% 同时有支持与反驳，specificity 错配常见；BioDivergence/BioConflict 显示冲突多由语境变量解释。也就是说，不对齐条件的"共识计数"系统性有错。ScholarStack 与 LKM 都把"条件可比才判分歧"写进了设计，但都没有评测这一步的正确率。

**时间是最薄的一维。** 现有做法有四种：年份字段、年代切片可视化(2606.07570)、按年快照(TaxoAdapt)、单调路径(EvoTree)。ClaimFlow 的数据(首次被挑战中位 22 年、refute 仅 2.4%、qualify 更常见)说明：主张状态的变化多是"被限定"而非"被推翻"，而且很慢。按时间截止回放来评测 as-of 状态，需要选在变化足够多的子领域里，否则 gold 几乎全是"未变"。

## 5. 开放问题清单(以上表为依据)

每条给出证据位置与可落地的评测抓手。

1. **支持集独立性计数。**
   - 空白依据：§2.4 所有共识/计数工作(scite、Consensus、2606.07570)与 §2.1 所有系统(ASKS 的 distinct work、LKM 的来源出现列表)都只数出现。
   - 已有素材：作者网络偏差(RSM 2026)给出统计定义；GraphEcho 证明不去重会被骗。
   - 抓手：按作者/团队/数据集/代码库去重的"独立支持数"，以元分析纳入研究(MetaSyn、EAKR 型 gold)或人工标注的"同源重复"为 gold。
2. **条件对齐后的共识/争议判定，以及对它的直接评测。**
   - 空白依据：ScholarStack L3 与 LKM contradiction group 有设计、无评测；BioConflict/BioDivergence 只到对级。
   - 抓手：多论文集合级的标签，即"在条件 C 下 支持/反对/有条件/未定"。gold 可来自系统综述的亚组分析或综述的"争议"段落。
3. **方法族作为一等对象：成员 + 共有局限 + 族内比较。**
   - 空白依据：ScholarStack 的 method family 维度与 L3 族刻画未评测；Lacuna 的 direction 是聚类摘要；Intern-Atlas 只有二元边、无族节点；TaxoBench 是主题而非族、且单标签。
   - 抓手：从综述分类图中取"方法族 + 成员"，从综述局限段取"共有局限"作 gold。成员用多标签 F1，局限用"被族内多少成员共有"的召回。
4. **可修订的 as-of 状态。**
   - 空白依据：所有系统只有年份或快照；MemStrata 是通用记忆。
   - 抓手：按截止时间冻结知识层，用后来的 refute/qualify 事件(ClaimFlow 型关系，或撤稿)当 gold，测状态修订的召回与误报。ClaimFlow 的数据提示要选变化密集的子领域。
5. **身份稳健的聚合。**
   - 空白依据：五种身份策略(§2.1)各有自认的失败方向。LKM 同义改写不合并会低估支持，AskChem 字符串归一误并会制造假共识。没有工作量化身份错误对聚合结论的传播。
   - 抓手：注入受控的身份扰动(拆分或误并 x%)，看族成员/共识标签/比较结论的翻转率。这是一个"聚合层敏感度"指标，与 §3 末行"身份影响无评测"对应。
6. **聚合层的直接评测协议本身。**
   - 空白依据：§3 表中"共识/争议"和"方法族"两行为空；LKM 的局限原文、ScholarStack 的纯下游评测都是现成的审稿对照物。
   - 抓手：把综述(族/局限/比较)、元分析(支持集)、后续批评(局限/争议)、时间回放(状态)四类 gold 统一成一个对象级 benchmark。
7. **多粒度一致性。**
   - 空白依据：ScholarStack L1/L2/L3、AskChem L1/L2/L3、LKM 的主张/族、GraphRAG 的 C0–C3 都有层级，但没有工作检验"上层结论是否被下层事实蕴含且不遗漏反例"。
   - 抓手：上层对象的 basis 完整性，即反证召回：下层存在的 opposing 事实是否在上层 basis 中出现。
8. **方法提出者归属与"提出 vs 使用"的跨论文准确率。**
   - 空白依据(§2.3)：SciER 有 USED-FOR 但没有"本文提出"标签；Linked PwC 没有 introduced-by 属性；Intern-Atlas 有方法节点但未报身份/归属精度；CiteME/REASONS 做的是"找被引"而不是"谁提出"。
   - 抓手：以 PwC 存档的 introduced-in 字段、综述方法表、或人工标注为 gold，测 mention→提出论文的精度与覆盖。本项目 E1 已有 200 条的精度证据，可直接对标。
   - 另需人工补查 MPR(JCDL 2024)是否已占"提出/使用"标注位。

**不建议当卖点的方向**(已被占或已收敛)：
- 单篇主张 + 证据锚 + 条件字段：LKM、ScholarStack、AskChem 均有。
- 主张关系边 support / extend / qualify / refute：ClaimFlow、AskChem、Agents-K1 均有。
- 方法演化边：Intern-Atlas 已有，且有综述 gold。
- 主题分类树加论文成员：TaxoAdapt、TaxoBench 等。
- "缺证据 ≠ 不存在"的免责声明：ScholarStack 已写。

## 6. 未核实、撤稿与勘误

- **BioDivergence(2606.11208)：作者已于 2026-08-30 撤稿**，abs 页可见撤稿记录。表中只作对级条件化矛盾的形态参考，不可当稳定结果引用。
- **ClaimFlow 数字随版本变化**：v1 为 304 篇 / 1,084 主张 / 832 关系 / κ 0.75；v2 为 1,617 篇 / 5,689 主张 / 4,871 关系 / κ 0.77、macro-F1 0.81。引用以 v2 为准。
- **LKM 改名**：v1 名为 "From Papers to a Scientific Reasoning Landscape"，v2 名为 "A Knowledge Foundation for Agentic Science at Scale"。SciAtlas v1 名为 "A Large-Scale Knowledge Graph for Automated Scientific Research"，v2 名为 "A Computable Atlas of Science…"。
- **arXiv 2305.02858** 不是 Newman 等人的表格论文，而是 ReMask(ACL 2023 Findings)，属 ID 勘误。Newman 等人的表格工作是 ArxivDIGESTables(2024)。
- **2410.03761** 实际标题为 "Taxonomy Tree Generation from Citation Graph"，HiGTL 是方法名。**2601.12369** 实际标题为 "Can Deep Research Agents Retrieve and Organize? …"。
- **MPR**(JCDL 2024，方法实体与论文关系数据集，可能含"提出/使用"标签)：ACM、DBLP 页面均被挡，未核实。这是方法归属方向最该人工补查的一项。
- **CS-KG 2.0**(Scientific Data 2025)：页面 403，只核到门户统计数。ORKG 的 K-CAP 2019 出处在 arXiv 页未写明。
- **arXiv 2609.06037**("Visual Analysis of LLM-based Entity Resolution from Scientific Papers")：abs 页描述与 2609 编号不符，存疑未用。
- **Consensus Meter 2.0**：页面返回 403，标签集与加权方式未核实。scite 无 arXiv 版本，以 bioRxiv 加期刊记录为准。
- **未核实、未进表**：TaxoInstruct、SurveyForge、LitLLM、SciClaimHunt(2502.10003，只确认是单篇主张验证数据集)、Cardinal assertions 论文的第一作者姓名(PDF 首页扫描不清)。
- **所有 2026 年条目**均为 arXiv 预印本，除非表中注明会议。会议信息以 ACL Anthology 或 arXiv comments 为准。
