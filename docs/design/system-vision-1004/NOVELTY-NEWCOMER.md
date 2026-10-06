# 查新：面向人（领域初学者）的领域理解工具，尤其利用"他述"的

> 检索时间 2026-10-04/05，覆盖到 2026-10。
> 方法：发现层用 CDP 真实浏览器（Google、arXiv 站内检索、Semantic Scholar 页面、作者主页 josephcc.com/papers、产品官网与帮助中心）；验证层逐条落到一手来源（arXiv abs/html 全文、ACL Anthology PDF、作者主页 PDF、Crossref/S2 元数据、SIGCHI 会议程序页、产品官方文档）。ACM DL 被 Cloudflare 拦截，ACM 论文用 arXiv 版、S2 API 摘要或会议程序页核验。
> 与前几轮的关系：CiteSpace II、Knowledge Navigator、ClaimFlow、Jurgens 2018、scite Smart Citations 的分类部分在 RESEARCH-FIELD-MAP / RESEARCH-AGGREGATION / CDP-SUPPLEMENT-AGGREGATION 里已有，这里只补"面向人、面向新手"的一面。
> "他述"= 后续论文的引用句 / 引用语境（citance），也包括 related work 段落里对别人的描述。

## 1. 判决表

威胁判定针对我们在"帮初学者建立领域认知"这个用途上的叙事：把每篇论文的自述和后续论文的他述按时间汇总成"截至 T 的领域认知"（方法族与成员、演化、对每个工作的定位、共同局限、已知未读），并识别认识变迁（被重新归类、被替代、变成默认组件）。

### A. 阅读界面里展示单篇论文的他述

| 工作 | 一手链接 | 面向谁 | 用了他述/引用语境吗 | 有时间维吗 | 汇总成领域结构吗 | 怎么评测（样本量） | 与我们的关系 |
|---|---|---|---|---|---|---|---|
| CiteRead（IUI 2022，Rachatasumrit, Bragg, Zhang, Weld） | DOI 10.1145/3490099.3511162；摘要核于 [SIGCHI 程序页](https://programs.sigchi.org/iui/2022/program/content/79966) 与 S2 API；机制核于 Semantic Reader 论文 [2303.14334](https://arxiv.org/abs/2303.14334) §2.2 | 读论文的科学家 | **是**：筛选重要的施引论文，把它们的 citances 定位到被引论文的具体段落，放在页边 | 只有"后续工作比原文新"这一层；不按时间展示描述怎么变 | 否，单篇 | 12 名科学家，对照"施引论文列表 + citances"；测对后续工作信息的理解与保持 | **半真**。"读一篇论文时看后续论文怎么说它"已被占；时间变迁、跨论文汇总不占 |
| CiteSee（CHI 2023） | [2302.07302](https://arxiv.org/abs/2302.07302)，DOI 10.1145/3544548.3580847 | 做文献综述的研究者 | 部分：paper card 里显示用户读过的论文中引用该文的句子（个性化） | 无 | 否；按用户阅读/收藏历史标"已知 / 相关但未知"的引用 | 实验室 N=10（对比 3 个基线）；现场部署 N=6 | **假**（他述）/ **半真**（"已知未读"）：它的"未知但相关"是个人阅读史视角，不是领域地图的边界 |
| Semantic Reader Project 总述（arXiv 2023；CACM 2024） | [2303.14334](https://arxiv.org/abs/2303.14334) | 学者与公众 | 两个原型用（CiteSee、CiteRead） | 无 | 否 | 10 个原型，累计 >300 名参与者的可用性研究 | 背景；其中 ScholarPhi、Paper Plain、Scim、Qlarify 都是单篇理解，只用自述，**假** |
| Surf / Beyond the Page（UIST 2025） | DOI 10.1145/3746059.3747647（S2 摘要） | 读论文的研究者 | 用社交媒体讨论（非引用句），聚成线程放进阅读界面 | 无 | 否 | 形成性 8 人；被试内可用性 N=18；理解深度、自我效能、认知负荷 | **假**。说明"把他人评价放进阅读"有正向理解效果，可作动机引用 |
| CitePeek（CUI 2025） | DOI 10.1145/3719160.3737639（S2 摘要） | 读论文的人 | 否，展示被引论文自身信息 | 无 | 否 | 摘要未给样本量 | **假** |

### B. 把多篇论文的他述汇总成主题 / 线索结构

| 工作 | 一手链接 | 面向谁 | 用了他述/引用语境吗 | 有时间维吗 | 汇总成领域结构吗 | 怎么评测（样本量） | 与我们的关系 |
|---|---|---|---|---|---|---|---|
| Action Science Explorer（Dunne, Shneiderman 等，JASIST 2012） | DOI 10.1002/asi.22652；全文 [作者 PDF](https://www.cs.umd.edu/~ben/papers/Dunne2012Rapid.pdf) | 进入陌生领域的研究者（资助项目名即 iOPENER: "Support Rapid Learning in Unfamiliar Research Domains"） | **是**：选中论文后显示其全部 citation sentences；对引用句做多文档摘要 | 按年份排序 + 双端滑块逐步显示论文；不分析描述变化 | 部分：引文网络、社区、排名、多视图协同 | 迭代可用性研究；第二轮 4 人（2 名 CS 博士生、2 名毕业生）；摘要质量被参与者批评 | **半真**。"用他述帮人快速理解陌生领域"14 年前就有系统；时间只是论文出现顺序 |
| Using Citations to Generate Surveys of Scientific Paradigms（Mohammad 等，NAACL 2009） | [N09-1066](https://aclanthology.org/N09-1066.pdf) | 需要快速理解大量技术材料的人 | **是**：直接摘要引用文本生成综述 | 无 | 否，产出是综述文本 | ROUGE + nugget pyramid；QA 与依存句法两个主题 | **半真**（先例）。结论"引用文本里有摘要里没有的综述级信息"已被证明，不能当我们的发现 |
| Surveyor（AAAI 2015）/ Content Models for Survey Generation（Jha 等，ACL 2015） | DOI 10.1609/aaai.v29i1.9495；[P15-1043](https://aclanthology.org/P15-1043.pdf) | 同上 | 用相关论文文本；ACL 2015 的 factoid 取自现有综述与 tutorial | 无 | 否 | AAAI：15 个 CL 主题人评连贯性；ACL：7 主题 3,425 句 factoid 标注、按频次加权的 pyramid | **假**；factoid 评测法可借 |
| Threddy（UIST 2022） | [2208.03455](https://arxiv.org/abs/2208.03455) | 做文献综述的研究者 | **是**：用户从 related work 里剪下引用语境建"线索"（thread） | 无（"长期使用怎么扩展"列为未来工作） | 用户手工建线索层级 | 被试内 9 人 | **半真**偏假：他述成结构，但全手工、个人化 |
| Relatedly（CHI 2023） | [2302.06754](https://arxiv.org/abs/2302.06754)，DOI 10.1145/3544548.3580841 | 想对陌生主题建立概览的研究者（示例用户是"初级 CS 研究者"） | **是**：同一主题下多篇论文的 related work 段落并排，动态重排、高亮未读信息、自动段落标题、淡化冗余 | 只有每段"引用文献年份范围"的时间线小图，用于挑新段或老段 | 段落 + 自动标题，不形成持久地图 | 被试内 n=15（研究员、博后、研究生），两主题各 20 分钟，写综述提纲；两名作者兼领域专家盲评连贯性 / 洞察 / 细节（各 5 分）并数主题数 | **半真**偏真：占了"汇总多篇他述 → 主题概览 → 给新手"。不占时间变迁、定位变化 |
| Synergi（UIST 2023） | [2308.07517](https://arxiv.org/abs/2308.07517)，DOI 10.1145/3586183.3606759 | 做综述 / 综合的学者 | **是**：种子片段 → 2 跳引文图 + loopy BP 排序 → 筛与种子相关的引用语境 → 层次聚类 → GPT-4 递归摘要成线索树 | 无 | **是**，每次查询生成一棵线索 / 子线索树，可编辑成提纲 | 12 人，基线 Threddy 与 GPT-4；20 分钟限时任务；专家评提纲；NASA-TLX。任务设定为"替同事综述相邻问题"以覆盖陌生领域 | **真**（对"他述 → 线索结构"这一步）。不占时间、不持久、无单篇定位 |
| DimInd（arXiv 2025，Fok, Chang 等） | [2504.18496](https://arxiv.org/abs/2504.18496) | 大规模文献综述 | 否，抽取单篇论文信息（自述） | 无 | **是**：多层压缩，论文 → 分面比较表 → 概念 taxonomy → 叙述综合，全程保留出处 | 23 名研究者，对照 ChatGPT 辅助流程 | **半真**（结构）/ **假**（他述） |
| GoAI（arXiv 2025，v2 2025-08-19，自注 work in progress） | [2503.08549](https://arxiv.org/abs/2503.08549) | **AI 专业学生** | 部分：论文间边是 5 类引用语义（基于与扩展 / 支持与补充 / 对比与替代 等）+ 引用所在章节；正文未说明是否由引用句判定 | 由前后向引用扩展出"发展轨迹"路径；无描述随时间变化 | **是**：论文 + 先修知识（概念、技能、工具）知识图；束搜索生成研究趋势与学习路径；Idea Studio | 9 名高年级本科 / 低年级研究生，三组组间（网页 / LLM 助教 / GoAI），2 小时；任务 Frontier Mapping（一页纸：基线、扩展、对比、数据集、先修知识）+ 出点子；SUS、NASA-TLX、点子得分 | **半真**，面向新手里最近邻：引用语义边 + 发展轨迹 + 学习路径。缺时间切片的定位变化、族、共同局限；n=9，未正式发表 |
| CiteSpace（2024–2025 版） | [官方博客](https://citespace.podia.com/blog/ba8c930d-134d-4ecf-8531-7787f0362a71) | 文献计量用户 | 簇摘要会点出关键施引文章 | **是**：时间切片、突现 | 簇（论文集合）+ GPT 生成簇标签与摘要 + 簇间依赖 | 无用户研究 | **半真**：时间加结构都有，但簇级、无方法语义、无单篇定位（前几轮已判） |

### C. 文献发现产品

| 工作 | 一手链接 | 面向谁 | 用了他述/引用语境吗 | 有时间维吗 | 汇总成领域结构吗 | 怎么评测 | 与我们的关系 |
|---|---|---|---|---|---|---|---|
| scite 报告页 + Summarize citations | [llms.txt](https://scite.ai/llms.txt)；[2022 官方教程](https://scite.ai/blog/2022-06-22_Using_scite_to_evaluate_a_key_paper_in_a_given_area_of_research)；报告页实测 | 研究者；教程开头就是新手问题："我发现一篇关键论文，但对这个主题所知甚少，怎么看它在更大文献里的位置？" | **是**：引用语句 + supporting / mentioning / contrasting + 所在章节 + 自引 / 独立；"Summarize citations"对引用语句生成综合 | 报告页有 Year Published 区间筛选；教程用"历年被引数"图判断影响力升降 | 否，以单篇为中心；可沿施引论文 forward chaining | 产品无用户研究；分类器精度见前几轮 | **半真**：单篇他述 + 年份筛选 + LLM 汇总都有，用户可以手工按年筛引用句。但不按时间呈现"描述怎么变"，没有族 / 领域层 |
| Semantic Scholar 论文页 | 页面实测 | 研究者 | citation intent 计数（Background / Methods / Results）、Highly Influential、引用摘录 | 可按 recency 排序 | 否 | — | **假** |
| Connected Papers | [About](https://www.connectedpapers.com/about) | 研究者 | 否：共被引 + 文献耦合相似度，力导向图，"不是引用树" | 弱 | 相似度图 | — | **假** |
| Litmaps | [Visualization 文档](https://docs.litmaps.com/en/articles/9181490-use-and-edit-litmaps-visualization) | 研究者 | 否，只用引用 / 参考关系 | **是**：坐标轴可选 Publication Date、Cite Count、Momentum（按发表时间修正的被引数） | 引文连接图；"找研究空白"= 看哪里没连上 | — | **假** |
| ResearchRabbit | [How ResearchRabbit uses AI](https://learn.researchrabbit.ai/en/articles/13545485)（2026-06） | 研究者、教学 | 否；核心推荐明确不用 LLM，只用引用 / 共被引 / 作者网络 | 弱 | 网络 | — | **假** |
| Inciteful | [首页](https://inciteful.xyz/) | 研究者 | 否；引文网络 + 链接预测，自称更易浮出新文献 | 弱 | 网络 | — | **假** |

### D. 新手入门：阅读清单 / 阅读路径 / 先修 / 概览

| 工作 | 一手链接 | 面向谁 | 用了他述/引用语境吗 | 有时间维吗 | 汇总成领域结构吗 | 怎么评测（样本量） | 与我们的关系 |
|---|---|---|---|---|---|---|---|
| Metro Maps of Science（KDD 2012） | DOI 10.1145/2339530.2339706；[作者 PDF](https://www.hyadatalab.com/papers/kdd2012-shahaf-guestrin-horvitz.pdf) | **进入新领域的人** | 否（内容相干性 + 引用影响） | **是**：多条研究线按时间展开，分叉、交汇 | **是**：研究线（metro line）+ 交汇 | 组间 30 名研究生（有 ML 背景但没做过 RL）；任务：假设自己是 RL 一年级研究生，更新一篇 1996 年综述，找 ≤5 个方向并列论文；专家打分（方向与论文） | **半真**：领域结构 + 时间 + 新手都有；不用他述，没有对单篇的定位及其变化。评测协议可直接借 |
| ACL-rlg（arXiv 2024-12） | [2502.15692](https://arxiv.org/abs/2502.15692) | "熟悉新领域"的人 | 否 | 否 | 否，产出是阅读清单 | 专家标注的阅读清单数据集，按检索任务评测；GPT-4o 疑有数据污染 | **假**；可当外部金标 |
| Reading Path Generation / SurveyBank / RePaGer（2021） | [2110.06354](https://arxiv.org/abs/2110.06354) | 想快速调研一个主题的人 | 否，用综述参考文献推多级阅读清单 | 否 | 先修链 + 阅读路径 | 对 SurveyBank 自动评测 | **假**；SurveyBank 可借 |
| TutorialBank（ACL 2018）/ LectureBank（AAAI 2019） | DOI 10.18653/v1/p18-1057；10.1609/aaai.v33i01.33016674 | NLP 学习者 | 否 | 否 | 概念先修链 | 人工标注先修关系 | **假**（先修关系先例） |
| ACCoRD（2022） | [2205.06982](https://arxiv.org/abs/2205.06982) | 缺先修背景的读者 | **是**（概念级）：利用概念在多篇论文里被描述的多种方式，生成多条描述 | 否 | 否 | 1,275 条标注语境、1,787 条人写描述；用户研究：偏好系统生成、偏好多条描述 | **半真**（概念级他述给新手） |
| DiscipLink（UIST 2024） | [2408.00447](https://arxiv.org/abs/2408.00447)，DOI 10.1145/3654777.3676366 | 跨学科研究者（陌生学科） | 否 | 否 | 按探索问题组织论文、抽主题 | 被试内 12 名研究生 + 开放式案例研究；提纲质量、NASA-TLX、自评知识前后差（7 点）作 knowledge gain | **假**；协议可借 |
| Selenite（CHI 2024） | [2310.02161](https://arxiv.org/abs/2310.02161) | 陌生领域的在线 sensemaking（非学术） | 否 | 否 | LLM 生成选项 × 标准概览 | 三个研究；研究 2：12 人，完成时间快 36.3%，NASA-TLX、SUS | **假** |
| Co-STORM（EMNLP 2024） | [2408.15232](https://arxiv.org/abs/2408.15232) | 探索"未知的未知"的学习者（通用） | 否 | 否 | 动态 mind map + 报告 | WildSeek 自动评测；人评 20 人（对搜索引擎 n=10，对 RAG chatbot n=9），偏好 | **假** |
| LitPivot（UIST 2026） | [2604.02600](https://arxiv.org/abs/2604.02600) | 打磨研究想法的研究者 | 否（检索论文簇） | 否 | 动态论文簇 | 实验室 n=17（对 chat-with-papers 基线）+ 开放式 n=5；专家评想法扎根程度、自评对文献空间的理解 | **假**；"对文献空间的理解"量表可借 |
| SLRMentor（arXiv 2026-06） | [2606.07831](https://arxiv.org/abs/2606.07831) | 学 SLR 方法的新手 | 否 | 否 | 否 | 研究生试点 | **假** |

### E. 分析类工作（不面向人，但直接支撑或威胁我们的说法）

| 工作 | 一手链接 | 内容 | 与我们的关系 |
|---|---|---|---|
| Elkiss 等（JASIST 2008）"Blind men and elephants: What do citation summaries tell us about a research article?" | DOI 10.1002/asi.20707 | 他述汇总（citation summary）与论文自身内容的关系 | 只核到元数据。"他述与自述不同"是 2008 年的问题 |
| Divoli, Nakov, Hearst（2012）"Do Peers See More in a Paper Than Its Authors?" | DOI 10.1155/2012/750214 | 一篇论文的全部 citances 覆盖其摘要的大部分信息，并多出约 20% 的概念；还考察了时间对 citance 内容的影响 | **"他述比自述多"已被量化，时间效应已有分析**（细节未读）；只能作为前提引用 |
| SECite（IEEE CCWC 2026） | [2601.07939](https://arxiv.org/abs/2601.07939) | 明确提出"传统摘要只反映作者自述"；对 9 篇 SE 论文的引用语境做情感分类，再用 LLM 生成"优点 / 局限"摘要，并对照外部引用反馈与作者自述的一致和分歧 | **半真**：单篇级"自述 vs 他述"对照 + 从他述归纳局限已有人做；规模小（9 篇）、无用户研究、无时间维、无族级 |
| The changing role of cited papers over time（arXiv 2025-09） | [2509.04190](https://arxiv.org/abs/2509.04190) | 约 900 篇高被引论文、22 万篇施引全文：论文变老后，被引位置前移、正文提及次数减少、更常与其他文献一起被引、施引论文与它的文本相似度下降，即从直接的方法性使用转向背景式、象征式引用 | 为"变成默认组件"这类认识变迁提供群体层面的经验证据；是分析，不是面向人的工具 |
| Hidden Citations Obscure True Impact in Science（2023） | [2310.16181](https://arxiv.org/abs/2310.16181) | obliteration by incorporation：成为常识后只提名、不引用；有影响力的发现，hidden citation 数超过显式引用数 | 支撑"默认组件"识别，也提示风险：只看引用句会漏掉不带引用的提及 |
| The Noisy Path from Source to Citation（ACL 2025） | [2502.20581](https://arxiv.org/abs/2502.20581) | 约 1,300 万对引用句与原主张的保真度；"传话效应"：低保真引用会向下游传递 | **风险**：他述可能失真，给新手看的他述汇总必须能回到自述原文核对 |

### F. 学习效果的测量方法来源

| 工作 | 一手链接 | 设计 | 可借之处 |
|---|---|---|---|
| Search+Chat（CHIIR 2025，Urgo / Arguello 组） | DOI 10.1145/3698204.3716446（S2 摘要） | 组间 N=40；学习型搜索任务；任务前、任务后、一周后三次选择题测验 | 前测 / 后测 / 延迟测的客观学习测量 |
| Search as Learning（Urgo & Arguello，FnTIR 2025） | DOI 10.1561/1500000084 | 综述"搜索即学习"的学习结果测量法 | 量表与测验类型的总表（未细读） |
| Egusa 等（IIiX 2010）"Using a concept map to evaluate exploratory search" | DOI 10.1145/1840784.1840810 | 用搜索前后的概念图评估探索式搜索（2011 年后续题为"搜索前后用户知识结构的变化"） | 概念图前后对比（只核到书目） |
| Melumad & Yun（PNAS Nexus 2025） | DOI 10.1093/pnasnexus/pgaf316 | 7 个实验 n=10,462：从 LLM 综述学习比从网页链接学习得到的知识更浅，写出的建议更稀疏、更不原创；即使综述附了实时链接也是如此 | **风险加测法**：用"学完写一段建议 / 解释"的产出质量测理解深度；我们给新手的汇总要防"更浅" |
| Learn Your Way（Google，arXiv 2025） | [2509.18664](https://arxiv.org/abs/2509.18664) | 组间 RCT，60 名高中生学一章陌生的神经科学内容；即时测 + 3–7 天延迟测 | 即时与延迟回忆测验 |
| In-situ Thought Exchanges（IUI，2025/26） | [2510.15234](https://arxiv.org/abs/2510.15234)，DOI 10.1145/3742413.3789069 | 46 名初级研究者，两周，三条件（无代理 / 单代理 / 多代理），批判性思维得分 | 多条件、多日的初级研究者研究规模参照 |
| PaperWeaver（CHI 2024） | [2403.02939](https://arxiv.org/abs/2403.02939) | 被试内 N=15（2 名硕士、7 名低年级博士、6 名高年级博士）；基线 = 推荐论文的 related work 段落；笔记按五个维度编码：事实数、论文分组数、关系数、关系细节、好奇心 | 笔记编码表，可直接用于"领域结构是否建立" |

## 2. 判决

**已被占（不要当卖点）**

1. "他述包含自述之外的信息、值得给读者看"：Elkiss 2008、Mohammad 2009、Divoli 2012 已量化；SECite 2026 做了单篇自述与他述的对照。
2. "读一篇论文时看后续论文怎么说它"：CiteRead（2022，12 人研究显示理解与保持更好）、scite 报告页（引用语句 + 立场 + 章节 + 年份筛选 + LLM 汇总，官方教程直接面向"对这个主题所知甚少"的用户）。
3. "把多篇论文的他述汇总成主题 / 线索结构，帮人快速了解陌生主题"：Action Science Explorer（2012）、Relatedly（2023）、Synergi（2023，引用语境 + 引文图 + LLM 递归摘要成线索树）。以按查询生成的形式，这一步已被占。
4. "面向学生：引用语义边 + 发展轨迹 + 学习路径 / 先修"：GoAI（work in progress，n=9）；阅读清单与先修链有 ACL-rlg、SurveyBank、TutorialBank / LectureBank。
5. "给进入新领域的人一张带时间的研究线地图"：Metro Maps of Science（2012）。
6. "时间作为坐标轴、筛选或切片"：Litmaps（Publication Date / Momentum 轴）、scite（Year Published 筛选）、ASE（年份滑块）、Relatedly（引用年份范围）、CiteSpace（时间切片）。
7. 群体层面"老论文被引方式变得背景化、象征化"：2509.04190 与 Hidden Citations 已有分析。

**没找到的（空位，均以本轮检索范围为限）**

1. **按时间呈现"一篇工作被怎样描述"的变化，并对人说出认识变迁**（被重新归类、被替代、变成默认组件）。现有工具最多给年份筛选（scite），让用户自己读引用句去发现；2509.04190 只有群体统计。没有工具自动指出"X 在 2019 年前被当作方法提出来引，之后被当作标准组件引"，也没有工具把它展示给新手。
2. **"截至 T 的领域认知"快照**：没有面向人的工具能回放某个时间点上，领域对各方法的定位、族成员和共同局限。Metro Maps 与 CiteSpace 有时间轴，但不是认识状态。
3. **持久、跨查询累积的领域层**（族、成员、定位、共同局限、已知未读），而非每次查询现生成（Synergi、Relatedly、DimInd）或个人化（Threddy、CiteSee）。CiteSee 的"相关但未读"是个人阅读史视角，算部分先例。
4. **族级共同局限来自多篇他述的汇总**：SECite 只做单篇优缺点，Synergi 只做线索摘要。
5. **评测上的空位**：这一系工具几乎都用专家盲评提纲、自评知识、NASA-TLX；客观前后测只在 search-as-learning 一侧有（Search+Chat）。没有人测"新手是否理解了领域认识怎么变的"（时间性理解）。

**结论**

在"帮初学者"这个次要用途上，"用他述汇总帮新手建领域认知"本身是老题，不能当新颖点。剩下真正的空位与主叙事的时间层重合：认识变迁和截至 T 的状态，对人可见。建议把初学者用途降为"时间层的一个可见演示加小规模用户研究"，主张收窄为：同等材料下，看得到"定位怎么变"的新手对领域现状的判断更准，例如哪些方法已被替代、哪些已成默认组件。

需要正面对比、并在相关工作里引用的：CiteRead、Relatedly、Synergi、Action Science Explorer、Mohammad 2009、Metro Maps、GoAI、scite、SECite、Divoli 2012、2509.04190。

两个要预先回应的风险：Melumad & Yun（LLM 综述让学习变浅）；Noisy Path（他述失真）。回应方式是每条汇总都能回溯到原句，且自述与他述并列。

## 3. 可借的小规模用户研究协议

拼法：任务取 Metro Maps 与 Relatedly，评分取 Relatedly 与 PaperWeaver，客观测验取 Search+Chat，样本量按 HCI 惯例。

- **被试**：12–16 名研究生，有相邻背景但没做过目标子领域（Metro Maps 的筛选法：有 ML 背景、没做过 RL）。被试内设计，两个子领域，系统 × 主题顺序拉丁方平衡（Relatedly n=15 的做法）。若要组间，参照 Metro Maps 30 人、Search+Chat 40 人。
- **条件**：(a) 我们的时间分层视图；(b) 同一语料的"他述无时间汇总"（scite 式引用语句页，或 Synergi 式线索树）；(c) 可选 LLM 对话 / 综述报告（对照 Melumad 效应）。(a) 与 (b) 的差值才是时间层的贡献，这一点决定能否把收益归到我们的独特点上。
- **任务**（每主题 20–30 分钟）：Metro Maps 式的"把一篇 T0 年的旧综述更新到今天"——列出至多 5 个方向及其代表论文，并对指定的 3–5 个方法回答它现在的地位（仍是主流 / 被替代，被谁替代 / 变成默认组件 / 被重新归类到哪个族）。
- **测量**：
  1. 时间性理解客观题（主指标）：前测、后测、一周后延迟测（Search+Chat 设计）。题目金标取自后来的综述与 2509.04190 式的被引方式统计，不由我们系统生成，避免循环论证。
  2. 提纲专家盲评：连贯性、洞察、细节，各 5 分，再数方向数（Relatedly）。评审用两名外部专家，不用作者（Relatedly 用的是作者，是它的弱点），报 Cohen κ。
  3. 笔记编码：事实数、分组数、关系数、关系细节（PaperWeaver 的五维去掉好奇心，加一维"时间性陈述数"）。
  4. 可选：任务前后各画一张概念图（Egusa）。
  5. 次要：完成时间、NASA-TLX、自评知识前后差（DiscipLink 的 7 点量表）。自评只作辅助，不作主证据。
- **样本量与功效**：这类研究在 n≈12–18 被试内设计下，常见报告效应是专家评分差约 2 分（总分 15 分制，Relatedly 10.5 对 8.1，p=0.02）。预期时间性题目差异更大（基线条件下这类信息需要人工拼），n=12–16 够做配对 Wilcoxon 检验；先用 4–6 人做预研究，确定题目难度。

## 4. 未核实清单（不进主表结论）

- CiteRead 全文（任务、理解与保持的具体测法）：ACM DL 被 Cloudflare 拦截，只核到摘要（SIGCHI 程序页、S2 API）和 Semantic Reader 论文中的转述。
- scite "Summarize citations" 的实际输出：报告页点击后未登录不渲染。功能描述来自 Missouri S&T 图书馆指南（二手）。scite Visualizations（2020 年 Medium 文"a new way to look at a field"）未打开。
- Connected Papers 的 Prior works / Derivative works：只见首页的 Google 摘要和 Medium 公告标题；About 页没讲这两个功能。
- Elkiss 2008、Egusa 2010 只核到书目元数据，内容按标题与后续论文标题推断。Divoli 2012 的"时间效应"只核到摘要里的一句话。
- CiteSpace 簇标签如何由施引文章派生：只有官方博客的概括表述。
- Learn Your Way 的"即时测 +9%、延迟测 +11%"：只见 Google 研究博客的搜索摘要，未在论文里核对数字，正文没有引用。
- H2CGL（"用 GNN 整合引用语境的时间演化"做被引预测）：只在 2605.18410 里看到二手提及，未核原文；它不面向人。
- Paper Plain 的正式发表 venue：只核了 arXiv 2203.00130。
- SSRN "Unlearn First: A Meta-Question Protocol for Entering an …"（2026-09）：只见标题，未打开。
- 发现层覆盖的局限：Google Scholar 未用（风控）；ACM DL 站内检索不可用；CHI / UIST 2026 程序只通过 Google 与作者主页间接覆盖，可能漏掉 2026 年刚发表、尚未进 arXiv 的 HCI 工作。
