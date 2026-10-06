# 科学创新类评测调研：新颖性判断、想法生成、研究定位（2026-10-05）

**范围**：哪些"科学创新"任务的成败取决于领域认知，能否接成我们系统的下游评测。"领域认知"指方法族、演化、他述、共同局限、已知未读、可回放到时间 T。时间覆盖到 2026-10。前一轮已经调研过的带截止预测类基准（ForeSci、RAP、IdeaForecastBench、PoT、PreScience、ScholarCatalyst、BackTrend、CKM、HindSight）见 `NOVELTY-AGENT-DOWNSTREAM.md`，这里只在需要时引用，不再展开。

**方法**：
- 发现层：CDP 真实 Google 搜索（中英文）、arXiv 站内检索、GitHub 用户仓库列表。
- 验证层：逐条打开 arXiv abs 页（标题、日期、版本、comments），再 curl `arxiv.org/html` 全文 grep 原文核对数字与引语。许可与可得性查 HF `/api/datasets`、文件树、GitHub API/raw LICENSE，venue 查 ACL Anthology 页面标题。
- 分工：新颖性一块（2.1）由我直接核实。想法生成（2.2）和研究定位（2.3）由两个子代理按同一纪律核实，我对其中关键数字和引语逐条复查过。标 ‡ 的行只核了 ID 和标题，正文数字只有子代理读过。

## 1. 结论先行

1. **新颖性判断的金标已经收敛到 OpenReview 审稿意见。** NovBench、RINoBench、NovGauge、Old Ideas、Beyond "Not Novel Enough"、InnoEval 都是这么做的。不用审稿意见的只有两类：用代理信号（SchNovel 假设"更新的论文更新颖"）或用公理（Axiomatic）。

   LLM 单独判新颖性很差：
   - RINoBench 中 GPT-5 的 macro-F1 最高也只有 17.2，所有模型在"不新颖"一类上的 F1 都是 0。
   - NovGauge 上 GPT-5.5 的 Verified F1 只有 43–72%。
   - RQ-Bench 出现 "novelty mirage"：LLM 评委打高分，专家却偏好原论文的研究问题。
   - Old Ideas 中改一句提示就能翻转过半判决，成对准确率变化超过 50 点。

2. **"失败是不是因为不懂领域"要分两层看。**

   第一层，检索到哪些前人工作，几乎决定了判决：
   - Idea Novelty Checker 识别"不新颖"的准确率：完整流程（按 facet 重排）89.66%，换成通用相关性重排 13.79%，只用关键词检索 5.17%。
   - LimitGen 原文把失败归因于 "a lack of knowledge in related areas"。
   - SciPaths 原文认为 "knowing what scientific building blocks to search for is crucial"。

   第二层，把检索结果直接拼进上下文帮助很小，有时还有害：
   - Old Ideas："Retrieval and larger reasoning budgets help little"，跨生成模型的效应符号都不一致。
   - NovBench：加 RAG 后相关性下降。
   - SCOPE："The core issue is not access to information but how it is integrated"。
   - RINI：直接把原论文给模型，175 例中 137 例认出了相关性，只有 61 例正确归属了贡献。

   所以瓶颈不在"有没有文本"，而在能不能把前人工作组织成可比较的贡献结构。这正是结构化领域认知能补的地方。反过来，我们也必须证明增益来自结构，而不是来自更多的文本。

3. **结构化输入有提升的证据很多，但效应多为小到中等，且大多由 LLM 评委判定**（详见第 3 节）。和我们最贴近的一条是 AgentExpt：去掉"下游论文怎么描述这个基线/数据集"的引文语境后，基线推荐 Recall@20 掉约 17.8%，是所有组件里掉得最多的。这几乎就是我们"他述"字段的直接证据。

   反面信号有三条：
   - Tree-of-Ideas 在 LLM 评委下领先，但在 5 位专家的人评里平均分低于 CoI（6.62 对 6.71）。
   - ResearchAgent 中给随机实体也比不给强，换成较弱的模型后，知识增强的收益变得 "marginal"。
   - GoR 中图结构相对平铺参考文献只多 0.96 Elo。

4. **推荐的 3 个下游**（与前一轮的 PreScience、IdeaForecastBench、PoT 互补）：
   - Old Ideas 的 novelty-judge-bench：新颖性判断，带检索截止，CC-BY-4.0。
   - MasterSet：基线和必引文献，按时间切分，金标是精确 ID。
   - LimitGen：局限识别，原文把失败直接归因于领域知识不足。

   备选：NovGauge、SciPaths、ResearchBench（见第 4 节）。

5. **几处同名，需要先澄清**：
   - "NoveltyBench"（2504.05228）评的是语言模型输出多样性，和科学无关。OpenNovelty 报告里说"将构建"的另一个 NoveltyBench 还没有发布。
   - NovBench（2604.11543）才是论文新颖性基准。
   - LiveIdeaBench 有两个：Ruan et al. 2412.17596 是单关键词发散思维评测；OpenReview KufsJAHeGn 是 Jiaxuan You 组的预测版，疑为 IdeaForecastBench 的 workshop 版。
   - Scientist-Bench 在 HKUDS/AI-Researcher（2505.18705）里，与 Si et al. 的 NoviScl/AI-Researcher 是两个不同的仓库。

## 2. 判决表

图例：
- **时间截止**：是否有显式的文献/模型截止，且金标晚于截止。
- **失败主因**：原文是否把失败归因于不了解相关工作或领域现状，括号里是依据。
- **契合度**：作为"历时领域认知"工具下游评测的合适程度。

### 2.1 新颖性判断 / 新颖性评估

| 基准/工作 | 一手链接 | 任务 | 金标来源 | 时间截止? | 公开与许可 | 失败主因是否领域认知 | 契合度（理由） |
|---|---|---|---|---|---|---|---|
| **Old Ideas, Novel Problems**（novelty-judge-bench） | [2610.02022](https://arxiv.org/abs/2610.02022) 2026-10-01；[HF noystl/novelty-judge-bench](https://huggingface.co/datasets/noystl/novelty-judge-bench)；github noy-sternlicht/novelty-eval | 对新颖性评委做受控研究，有成对和逐点两种形式。设置一 Human-Only：高新颖 vs 低新颖的人类论文。设置二 Human+Generated：高新颖人类论文 vs Claude Sonnet 4.5 不带工具生成的想法。每个成对配置 154 对 | ICLR 2026 审稿意见。高新颖 = 录用、评分在所属 area 前 10%、多数审稿人明确肯定新颖性且无否定；低新颖反之 | **是**。检索截止 2025-03-01（比投稿截止早约 6 个月）；评委模型的训练截止都早于 ICLR 2026 投稿截止 | HF CC-BY-4.0，非 gated | **否，且检索也救不了**。原文："Retrieval and larger reasoning budgets help little"；更强的 agentic RAG（gpt-5.6-sol 实时检索）"does not help"；检索有时会误导评委，让它认为 "does appear novel ... relative to the provided related work"；两个专门的新颖性评估器不如最便宜的提示基线 | **高**。已有"给评委 5 篇相关工作摘要"的检索条件，可直接替换成我们截至 T 的工具输出。检索无效使它成为检验"结构优于文本"的硬测试。缺点：n 小；只覆盖 ICLR；取两极样本，任务偏易；对提示极敏感，必须多提示报告 |
| **NovGauge** | [2609.11234](https://arxiv.org/abs/2609.11234)；[HF ZitaGo/NovGauge](https://huggingface.co/datasets/ZitaGo/NovGauge)；github Zita-Go/NovGauge | 619 对论文和 50 个多论文集合，在 task、problem、method 三个维度上分别判断是否重叠，并级联核验证据是否忠实 | 从 42,682 个 ICLR 2023–2026 forum 中抽取审稿人的"与 X 重叠"声明，加上标注者核验过的综述共被引组 | 否（直接给两篇全文） | HF CC-BY-4.0，非 gated（positives、negatives、grouping 三个 JSON）；代码 MIT；论文 CC BY-NC-ND | **否**。失败主要是证据幻觉（0–39%）和逻辑不支撑（超过 70%）；GPT-5.5 Verified F1 43–72% | **中高**。method 维度和综述共被引组与"方法族"同构，可以测提供族和谱系语境能否提高 Verified F1。没有截止，不测时间 |
| **RINoBench** | [2603.10303](https://arxiv.org/abs/2603.10303)；[LREC 2026.lrec-1.370](https://aclanthology.org/2026.lrec-1.370/)；[HF TimSchopf/RINoBench](https://huggingface.co/datasets/TimSchopf/RINoBench)；github TimSchopf/RINoBench | 1,381 个研究 idea，按 1–5 分 rubric 打新颖性分并写理由；每个 idea 平均附 25.23 篇相关工作摘要（取自引言和相关工作节） | ICLR 2022/2023 审稿新颖性分与理由（由 LLM 综合），只保留理由完全有依据的样本 | 否；ICLR 2022/23 在模型训练截止之前，污染风险高 | HF 非 gated，无 license 标签；仓库无 license | **否**。相关工作已经给定，仍判不准："none of them achieving significant F1"，最高 17.2，"strong bias against predicting ideas as 'not novel'" | **中**。"给定相关工作"这一槽位可以换成我们的结构化视图，但污染严重 |
| **NovBench** | [2604.11543](https://arxiv.org/abs/2604.11543)；[ACL 2026 Findings 1607](https://aclanthology.org/2026.findings-acl.1607/)；github njust-winchy/llm4novelty | 根据引言中的新颖性陈述，生成对新颖性的评价；按 Relevance、Correctness、Coverage、Clarity 四个维度打分 | EMNLP 2023 的 1,684 对论文-审稿（来自 NLPeer/OpenReview）；新颖性陈述由 GPT-5 抽取，审稿中的新颖性评价由 GPT-4o-mini 抽取 | 否；EMNLP 2023 在训练截止之前；RAG 语料只到 2022 年 | 数据在 Google Drive（未打开）；仓库无 license | **否**。原文："limited understanding of scientific novelty"；RAG 提升清晰度但降低相关性："retrieval augmentation or advanced prompting alone is insufficient" | **低中**。金标是审稿文本，用相似度评分；时间设计不适用 |
| **Beyond "Not Novel Enough"** | [2508.10795](https://arxiv.org/abs/2508.10795) v4；[EACL 2026.eacl-long.121](https://aclanthology.org/2026.eacl-long.121/)；github UKPLab/eacl2026-assessing-paper-novelty | 三段式新颖性评估：抽取内容 → 检索并综合相关工作 → 结构化比较 | 182 篇 ICLR 2025 投稿的审稿新颖性评估，由人工标注；原文："we use the GPT-4.1-synthesized assessments as our ground truth" | **是**。检索时过滤掉投稿日期之后的论文；182 篇中只有 11 篇在 GPT-4.1 截止前出现在 arXiv | 代码 Apache-2.0；数据在 TUdatalib 4988 | **部分**。检索消融：以完整流程选出的论文为参照，只用 dense 检索 top-5 召回 0.612，只用关键词 0.300 | **中高**。有截止、数据公开、比较环节可接我们的工具。弱点：金标经 GPT-4.1 综合，结论一致率 75.3% 本身也由 LLM 判定 |
| **Idea Novelty Checker** | [2506.22026](https://arxiv.org/abs/2506.22026)；[SDP 2025.sdp-1.9](https://aclanthology.org/2025.sdp-1.9/)；github simra-shahid/idea_novelty_checker | 检索 → 按 facet 重排 → 判断新颖/不新颖 | 2 位专家标注 51 个 idea（46 个来自 Scideator），Kappa 0.64；限定只看给定论文后升到 0.68 | 否 | 仓库无 license | **是**。消融（58 个"不新颖"样本）准确率：完整 89.66%，通用相关性重排 13.79%，只做 embedding 过滤 10.34%，snippet 8.62%，关键词 5.17%。专家也 "relied on their broader domain knowledge ... the top papers alone were often not sufficient" | **中**。样本太小，不适合当基准，但它是"检索到什么决定判决"最强的一手证据 |
| **SchNovel** | [2409.16605](https://arxiv.org/abs/2409.16605)；[AISD 2025.aisd-main.5](https://aclanthology.org/2025.aisd-main.5/)；github ethannlin/SchNovel | 15,000 对 arXiv 论文（6 个领域，发表相隔 2–10 年），判断哪篇更新颖 | 发表先后（"the more recently published paper is assumed to be more novel"） | 是（只用了发表日期） | 仓库 MIT | **部分**。RAG-Novelty 假设"越新颖，检索到的近邻越新"；GPT-4o-mini 在六个领域的准确率 0.58–0.73，次优的 Self-Consistency 为 0.57–0.66 | **中低**。金标是代理信号。Axiomatic 发现 RAG-Novelty 与 ICLR 分数相关最高（ρ=+0.69），但连"完全复制"探针都没过（0.28），说明它实际上不看检索内容 |
| **Axiomatic Benchmark** | [2604.15145](https://arxiv.org/abs/2604.15145) v2 2026-08-05 | 无标签：改动候选池，看新颖性分数是否按公理变化。三条公理：池覆盖论文内容越多分越低；池越不相关分越高；池在时间上越晚分越低 | 公理本身（构造即真）；候选池取自 pwc-archive/papers-with-abstracts（2025-07） | **时间是被测变量** | 原文说"只发布 arXiv ID、派生文件和脚本"，但 HTML 里没找到仓库链接（未核实） | **部分**。原文："surface redundancy is largely solved but conceptual redundancy is not"；时间公理上 "no system exceeds 0.24"，"no novelty system reliably scores a paper lower against the future slice" | **高（概念上）**。时间公理就是在测"回放到 T"。但它评的是新颖性度量，不是智能体；数据链接没核到 |
| **RQ-Bench** | [2606.12071](https://arxiv.org/abs/2606.12071) | 研究问题的新颖性：比较独立打分、比较式打分与专家评估 | 原论文作者的研究问题（由引文背景重建）；50 例专家评估 | 用的是新近 arXiv 论文，有无显式截止未核 | 没找到数据链接 | 否。原文："novelty mirage"，专家结论与 LLM 评委相反；生成的研究问题 "narrow or source-bound" | 低中，主要作"不能用 LLM 评委判新颖性"的证据 |
| **InnoEval**（ICML 2026） | [2602.14367](https://arxiv.org/abs/2602.14367)；github zjunlp/InnoEval | 逐点、成对、分组三种 idea 评估；用多源知识检索加多人设评审团 | NeurIPS 2025 / ICLR 2025 投稿按最终决定分层抽样 | 未核 | 仓库 MIT | **是**。消融去掉 Web 和代码检索后："search richness can significantly improve evaluation accuracy and emphasizes the critical role of sufficient background knowledge"；成对和分组任务受影响更大。数字只在图里，没抽到 | 中 |
| **ScholarEval / ScholarIdeas** | [2510.16234](https://arxiv.org/abs/2510.16234) v2；github skai-research/ScholarEval | 基于文献评估 idea 的可靠性（soundness）与贡献 | 117 个 idea，覆盖 AI、神经科学、生化、生态 4 个学科，由专家标注评审 rubric；指标是 rubric 覆盖率（LLM 判定） | **是**。原文："annotate each idea with its publication date, which we use as a cutoff date for literature search" | 仓库 MIT | 部分。消融（AI 子集）覆盖率：完整 2.91，去掉"前人方法和结果抽取" 2.47，去掉论文扩充 2.42，去掉成对比较 2.39 | **中高**。按 idea 设截止、跨学科、专家 rubric；"前人方法有没有效"这类结构化信息正好由我们供给 |
| GraphMind（KDD'26，含 SciNova） | [2510.15706](https://arxiv.org/abs/2510.15706)（EMNLP'25 Demo）；KDD 论文 DOI 10.1145/3770855.3818195；github oyarsa/graphmind | 用论文内部层级图和相关论文图做新颖性分数预测，并生成理由 | SciNova：3,063 篇 ICLR+NeurIPS 论文，带全文、参考文献和审稿分 | 未核 | 代码 AGPL-3.0 | Demo 原文："requires extensive knowledge of related work, something not all reviewers have"；KDD 摘要说双层图 "significantly outperforms baseline LLMs" | 中。KDD 全文被 Cloudflare 拦截，**效应量未核实** |
| OpenNovelty | [2601.01576](https://arxiv.org/abs/2601.01576)；github january-blue/OpenNovelty | 系统：为每篇投稿现场构建核心任务相关工作的层级 taxonomy，再逐条贡献比较；部署在 500 多篇 ICLR 2026 投稿上 | 无定量评测（只写了评测计划） | — | Apache-2.0 | — | 低（不是基准）。叙事上要注意：它已经用"按查询现做的 taxonomy"做新颖性分析，但 taxonomy 不持久，也没有时间维 |
| 其他证据 | [2502.16487](https://arxiv.org/abs/2502.16487)（ACL'25）；[2602.06054](https://arxiv.org/abs/2602.06054)；[2502.14297](https://arxiv.org/abs/2502.14297)；[2505.24615](https://arxiv.org/abs/2505.24615) | — | — | — | 2602.06054 和 2505.24615 只有匿名链接 | **是**：13 位专家判定 50 份 LLM 研究文档中 24% 属于改写或大量借用已有工作；LLM agent "systematically overestimate novelty and struggle to detect conceptual plagiarism"；AI Scientist 把 SGD 的 micro-batching 判成新颖 | 只作动机证据 |
| 排除：NoveltyBench | [2504.05228](https://arxiv.org/abs/2504.05228) | 语言模型输出多样性（NB-Curated、NB-WildChat） | — | — | — | — | 与科学无关，同名勿混 |

### 2.2 想法生成与评估

| 基准/工作 | 一手链接 | 任务/规模 | 金标来源 | 时间截止? | 公开与许可 | 文献输入 / 失败主因 | 契合度（理由） |
|---|---|---|---|---|---|---|---|
| **ResearchBench**（ACL'26 Findings） | [2503.21248](https://arxiv.org/abs/2503.21248)；HF ankilok/ResearchBench | 三个子任务：灵感检索、假设构成、假设排序；1,386 篇论文、12 个学科 | LLM 从论文中抽取，专家抽样验证 | **是**，只收 2024 年以后的论文，可滚动更新 | HF CC-BY-NC-4.0，gated:auto；GitHub 为 MIT（子代理核） | 输入是研究问题加背景综述，每题配 75 个候选。GPT-4o 选前 4% 时真灵感入选概率 45.7%。原文："strong retrieval but only moderate composition and ranking" | **高**。"背景综述"这一槽位可以直接替换成我们的编译产物；12 个学科 |
| **IdeaBench**（KDD'25） | [2411.02429](https://arxiv.org/abs/2411.02429)；github amir-hassan25/IdeaBench | 2,374 篇 2024 年生物医学目标论文，29,408 篇参考文献 | 目标论文摘要，用 GPT-4o 排序后算 Insight Score | **是**："cutoff dates before January 1, 2024" | 未见 license | 输入是参考文献摘要。原文："with all references available, unfiltered references produce the most aligned" | 中高。参考文献集合可以直接替换；缺点是对新模型来说截止已过，而且靠 LLM 评委 |
| AI Idea Bench 2025 ‡ | [2504.14191](https://arxiv.org/abs/2504.14191)；HF yanshengqiu/AI_Idea_Bench_2025 | 3,495 篇 AI 会议论文及其灵感来源论文 | 与原论文对齐 | 2023-10-10 以后 | Apache-2.0 | 灵感论文作为输入 | 中高，需要按新模型的截止重新切分 |
| MOOSE-Chem / 2 / 3 ‡ | [2410.07076](https://arxiv.org/abs/2410.07076)（ICLR'25）等 | 化学假设再发现：51 篇，最多 3,000 篇候选语料 | 化学博士标注 | 2024-01 以后 | MIT（Chem/Chem2） | 灵感检索命中率：选 4% 语料时 83.7%（子代理核） | 中，只覆盖化学 |
| **Si et al.** | [2409.04109](https://arxiv.org/abs/2409.04109)；github NoviScl/AI-Researcher（MIT） | 49 名作者、79 名评审，每个条件约 49 个想法 | 专家盲评 | 否 | MIT | AI 想法新颖性 5.64 对人类 4.84（p<0.01）。4000 个种子想法里只有 200 个不重复。评审间一致率 56.1% | 低（成本高）；作动机：LLM 想法同质化严重 |
| **Ideation-Execution Gap** | [2506.20803](https://arxiv.org/abs/2506.20803) | 43 人执行想法，每人超过 100 小时 | 执行后专家盲评 | 否 | 同上 | AI 想法执行后分数降幅：novelty −1.049、excitement −1.760、effectiveness −1.879、overall −1.976；人类想法基本不变。原文："missing baselines and ablations ... almost entirely overlooked during ideation evaluation" | 低；证明 LLM 判定的想法质量不可信，而缺 baseline 本身就是领域认知问题 |
| **Ideation Arena** | [2608.29696](https://arxiv.org/abs/2608.29696)；github foss12138/Research-Ideation-Arena | 14 个 LLM 加 5 种 agent 架构，105 名 CS 研究者完成 6,000 多次双盲对比 | 人类 Elo | 所有系统共享同一封闭文献上下文 | 仓库无 LICENSE 文件 | 原文："shared literature contexts from papers familiar to the participating researchers"。只有 AI-Researcher 框架把 DeepSeek V3.2 从 1099.5 提到 1276.5（+177） | **高（如果能做人评）**。共享上下文就是我们工具的注入点，金标是人类判断；但新系统要重新收集对比 |
| **ResearchAgent**（NAACL'25） | [2404.07738](https://arxiv.org/abs/2404.07738) | 300 篇核心论文（2023-05 以后） | 与人类对齐的 LLM 评委 | 是 | 未见 license | 学术图谱加实体库。原文："random elements is more helpful than providing no elements"；较弱模型下差异 "become marginal" | 中 |
| **CoI**（EMNLP'25 Findings） | [2410.13185](https://arxiv.org/abs/2410.13185)；github DAMO-NLP-SG/CoI-Agent（Apache-2.0） | 50 个话题，Idea Arena（GPT-4o 成对比较） | LLM 评委（与人一致率 70.8%，子代理核） | 是（2024-08 以后的话题） | Apache-2.0 | 原文："a clear developmental trend analysis is more pivotal than the quantity of related literature" | 中高。Arena 协议可以复用 |
| **SciMON**（ACL'24） | [2305.14259](https://arxiv.org/abs/2305.14259) | 根据背景语境生成 idea；用语义邻居、知识图谱邻居、引文邻居作灵感 | 原论文 idea 加专家人评 | **是**：训练/验证/测试分别取 2021 年以前、2021、2022 年 | 原文给了代码链接（未核许可） | GPT4FS+KG 对 GPT4FS：48% 的对比中技术细节更多，45% 更新颖，其余大多打平；但 85% 的对比中原论文明显更好 | 中（较老，可作先例） |
| LiveIdeaBench（Ruan et al.）‡ | [2412.17596](https://arxiv.org/abs/2412.17596)；HF 6cf/liveideabench | 1,180 个关键词、22 个领域 | LLM 评审团 | 否 | Apache-2.0 / MIT | 按设计就排除文献 | 低 |
| Scientist-Bench ‡ | [2505.18705](https://arxiv.org/abs/2505.18705)（HKUDS） | 22 篇 | LLM 成对比较 | 未见 | 未见 license | 只给参考论文 | 低 |
| Nova ‡ / SciPIP ‡ / Scideator | [2410.14255](https://arxiv.org/abs/2410.14255)、[2410.23166](https://arxiv.org/abs/2410.23166)、[2409.14634](https://arxiv.org/abs/2409.14634) | 都是方法或系统 | LLM 评委 / 用户研究 | — | SciPIP 为 MIT | Nova 把问题归因于 "limited ability in acquiring external knowledge"（子代理核） | 低 |

### 2.3 研究定位：相关工作、基线/数据集、已有工作查重、空白与局限

| 基准/工作 | 一手链接 | 任务/规模 | 金标来源 | 时间截止? | 公开与许可 | 失败主因是否领域认知 | 契合度（理由） |
|---|---|---|---|---|---|---|---|
| **MasterSet** | [2604.17680](https://arxiv.org/abs/2604.17680)；HF trratul/MasterSet；github NIU-Graph-Intelligence/MasterSet | 必引文献推荐，其中 Type I 标签是"实验基线"；候选池 67,761 篇（2018–2024），评测 7,028 篇（2025 年 NeurIPS/ICML/ICLR） | 论文真实引用加类型标注（据子代理，由 Gemini 2.5 Flash 标注、人工抽检 510 条） | **是**，按年份切分 | 代码 MIT（README）；HF 非 gated，无 license 标签 | 未归因 | **高**。金标是精确 ID，样本量大；"该比哪些基线"直接取决于方法族和演化知识 |
| **AgentExpt** | [2511.04921](https://arxiv.org/abs/2511.04921) | 约 10 万篇论文，推荐实验基线和数据集 | 论文实验节实际用过的基线和数据集 | 否 | 正文没找到数据链接 | 未归因；但消融显示他述最关键（见第 3 节） | 高（机制同构），数据拿不到 |
| **SCOPE** | [2608.03501](https://arxiv.org/abs/2608.03501) | 300 篇、19 个领域，设计实验方案 | 原论文实验设计，由 GPT-5.2 判分 | **是**，资源限定在某时间点之前 | 原文："Code will be made publicly available upon publication" | 部分。原文："search mode does not improve design quality"；"unstructured retrieval competes with, rather than complements, internal reasoning" | 高，暂时拿不到 |
| **SciPaths** | [2605.14600](https://arxiv.org/abs/2605.14600)；github ericchamoun/scipaths（MIT） | 给定目标贡献，找出实现它所需的前置贡献，并落到具体前作；专家金标 262 条、银标 2,444 条 | 专家标注（必要性准则） | **是**："prior literature available at a specified time" | MIT，含数据划分 | **是**。原文："knowing what scientific building blocks to search for is crucial"；给出金标前置贡献后，Coverage@5 从 0.083 升到 0.357（Gemini），GPT-5.4 从 0.071 升到 0.261 | **高**。谱系和演化关系正好填在"只给主张"与"给金标前置贡献"之间 |
| **TaxoBench** | [2601.12369](https://arxiv.org/abs/2601.12369)；HF konglongge/TaxoBench（CC-BY-NC-4.0）；github KongLongGeFDU/TaxoBench（Apache-2.0） | 72 篇综述、3,815 篇被引文献；先检索，再组织成分类树 | 专家综述的分类树 | 是（以综述发表日期为截止） | 见左 | 部分：最好的 agent 只召回 20.92% 的专家引用；原文："retrieval and hierarchical organization as separate bottlenecks" | 高（方法族组织）。前轮已作为地图类工作列过 |
| **LimitGen**（ACL'25） | [2507.02694](https://arxiv.org/abs/2507.02694)；HF yale-nlp/LimitGen；github yale-nlp/LimitGen | 局限识别。Syn 1,000 条（受控扰动）；Human 1,000 条 | 审稿 weaknesses（经 GPT-4o 过滤）；评测用 LLM | **部分**。原文："We chose ICLR 2025 to mitigate contamination" | HF 非 gated，无 license 标签；仓库无 LICENSE 文件；论文声称 CC BY 4.0（未核） | **是**。原文："LLMs often fail to detect limitations ... due to a lack of knowledge in related areas"；加 RAG 后 GPT-4o 粗粒度准确率 +12.2pp | **高**。"共同局限与开放问题"加他述批评可直接供给；已有 RAG 对照 |
| **ToC-Bench**（EMNLP'26 Findings） | [2608.20777](https://arxiv.org/abs/2608.20777) | 414 篇论文、1,905 条作者没说出的局限 | 审稿 weaknesses 971 条，加后续施引论文中的批评 934 条 | 无；施引批评是事后信息，用作金标时必须加截止 | 原文说会发布，没找到链接 | 未归因（实验设置不允许检索） | 高。金标就是"后人怎么批评"，与我们的他述字段同构；数据暂时拿不到 |
| RINI | [2609.33284](https://arxiv.org/abs/2609.33284) | 审计提案中的贡献声明 | 5 名人工标注 | — | 未核 | **部分**：给了原论文仍然不会归属贡献（137/175 认出相关，61/175 正确归属） | 中（证据） |
| GRASP（ACL'26 Findings） | [2607.03709](https://arxiv.org/abs/2607.03709) | 在 OARelatedWork 上用图规划相关工作 | 作者原相关工作节 | 否 | 未核 | — | 中。ROUGE-1 F1：无图 0.747，双层图 0.771 |
| MIR（ACL'25） | [2506.00249](https://arxiv.org/abs/2506.00249) | 方法灵感检索 | 论文真实引用 | 未核 | 未核 | — | 中。用引文构建的方法谱系图（MAG）使 Recall@3 +5.4、mAP +7.8 |
| CiteBench ‡ / OARelatedWork ‡ / GREP ‡ / AcademicEval ‡ / DataFinder ‡ / BAGELS ‡ | [2212.09577](https://arxiv.org/abs/2212.09577)、[2405.01930](https://arxiv.org/abs/2405.01930)、[2508.07955](https://arxiv.org/abs/2508.07955)、[2510.17725](https://arxiv.org/abs/2510.17725)、[2305.16636](https://arxiv.org/abs/2305.16636)、[2505.18207](https://arxiv.org/abs/2505.18207) | 引用句或相关工作生成、定位偏好、滚动更新的写作任务、数据集推荐、局限抽取 | 原文或专家 | 只有 AcademicEval 滚动防泄漏 | 多为 Apache-2.0 或 MIT（子代理核） | 未归因 | 低到中。多数评字面重合或在给定被引集合上做，不测领域认知 |

## 3. "结构化领域输入能提升创新任务"的证据汇总

| 工作 | 结构化输入 vs 对照 | 效应 | 评判方式 | 我们的读法 |
|---|---|---|---|---|
| AgentExpt [2511.04921](https://arxiv.org/abs/2511.04921) | 第三人称引文语境（下游论文如何描述某基线或数据集）加自述 vs 只有自述 | 去掉引文语境：基线 R@20 0.4523→0.3716（约 −17.8%），数据集 0.3001→0.2493（约 −16.9%）；去掉自述只降 7.2% / 9.7% | 客观（精确 ID） | **最直接对应"他述"字段**，而且是客观指标 |
| SciPaths [2605.14600](https://arxiv.org/abs/2605.14600) | 给金标前置贡献 vs 只给主张 | Coverage@5：0.083→0.357（Gemini），0.071→0.261（GPT-5.4） | 语义匹配 | 是"谱系知识"能带来收益的上界，而且很大 |
| Idea Novelty Checker [2506.22026](https://arxiv.org/abs/2506.22026) | 按 facet（目的、机制、评测、应用）重排 vs 通用相关性重排 | "不新颖"识别准确率 89.66% vs 13.79% | 专家金标（n=58） | 按贡献维度组织比单纯"更相关"重要得多；样本小 |
| MIR [2506.00249](https://arxiv.org/abs/2506.00249) | 方法谱系图（MAG）先验 vs dense 基线 | Recall@3 +5.4，mAP +7.8 | 客观 | 谱系有用，效应中等 |
| ScholarEval [2510.16234](https://arxiv.org/abs/2510.16234) | 抽取前人方法和结果 vs 不抽取 | 覆盖率 2.91→2.47 | LLM 判 rubric 覆盖 | "前人方法效果如何"这类结构化信息有用 |
| Tree-of-Ideas [2608.10740](https://arxiv.org/abs/2608.10740) | 多路径演化树 vs 单链 vs 只给前沿论文；有无 gap 标注 | 6.04 / 5.85 / 4.77；去掉 gap 标注 6.27→5.55 | 3 个 LLM 评委 | **人评反转**：5 位专家给 ToI 6.62，低于 CoI 6.71 |
| CoI [2410.13185](https://arxiv.org/abs/2410.13185) | 按引用链组织 vs 同样 5 篇平铺 | 以完整版为 50 分自比，去掉链式组织后 42.4 | GPT-4o 评委 | 方向一致，属于 LLM 评委下的证据 |
| ResearchAgent [2404.07738](https://arxiv.org/abs/2404.07738) | 参考文献加实体 vs 都不给 | Problem 4.52 vs 4.20（5 分制） | LLM 评委 | 给随机实体也有帮助；较弱模型下收益 "marginal"，说明结构只对强模型有用 |
| SciMON [2305.14259](https://arxiv.org/abs/2305.14259) | GPT4FS+KG vs GPT4FS | 48% 的对比中技术细节更多，45% 更新颖，其余大多打平 | 专家排序 | 有人评，但效应小，且与原论文差距很大（85%） |
| GoR [2605.14790](https://arxiv.org/abs/2605.14790) | 引文演化 DAG vs 平铺参考文献（同为 7B SFT） | Elo 23.42 vs 22.46，差距 0.96 | LLM 评委 | 效应小；作者也承认 SFT 才是主要驱动 |
| Graph2Idea [2606.09105](https://arxiv.org/abs/2606.09105) | 知识图谱上下文 vs 平铺文本 | 新颖性 0.52 vs 0.50 | 自动协议 | 原文："the improvements are modest" |
| GRASP [2607.03709](https://arxiv.org/abs/2607.03709) | 双层图 vs 无图 | ROUGE-1 F1 0.771 vs 0.747 | 字面指标 | 小 |
| IdeaForecastBench [2609.00747](https://arxiv.org/abs/2609.00747) | 领域级 Summary vs Direct | Hit@5 0.487→0.756（GPT-4.1） | LLM 匹配 | 原文自承，部分优势可能来自输出更宽泛："not a control for intrinsic generality" |
| InnoEval [2602.14367](https://arxiv.org/abs/2602.14367) | 多源检索 vs 只用文献 | 显著，但数字只在图里 | 与人类对齐度 | 方向支持；数字未核 |
| GraphMind KDD'26 | 双层图 vs 基线 LLM | "significantly outperforms" | 审稿分 | **效应量未核实** |

**反面证据与警示**：
- Old Ideas：给新颖性评委 5 篇截止前摘要，甚至让它实时检索，效应都小而且符号不一致。
- NovBench：RAG 降低相关性。
- SCOPE：开搜索不改善实验设计。
- RINI：给出原论文，61/175 正确归属贡献。
- RINoBench：已经给了 25 篇相关工作，macro-F1 仍只有 17.2。
- Axiomatic：已有系统都过不了"概念冗余"和"时间"两条公理。
- LiveIdeaBench-Forecasting：据子代理读到，结构化历史摘要的命中，在对照"过去窗口"后只剩 +0.029，95% CI [−0.048, +0.101]，不显著。我复查时 OpenReview 被 Cloudflare 拦截，这条只有单一来源。

**汇总判断**：
- 客观指标上，"他述"和"谱系"有中到大的效应（AgentExpt、SciPaths、Idea Novelty Checker），而且都出现在检索或定位环节。
- 在想法生成环节，几乎所有增益都来自 LLM 评委，效应小。唯一有人评的 Tree-of-Ideas 对 CoI 还反转了。

所以我们的主张更应落在"判断与定位"（新颖性、基线、局限、前置贡献），而不是"让 LLM 评委觉得想法更好"。如果做想法生成，必须配客观结果或人评，并且要设"每次现做的平铺上下文"这一强对照。

## 4. 推荐的 3 个下游评测

评分口径与前一轮一致：(a) 金标客观或来自人类；(b) 现在拿得到，许可可用；(c) 有截止 T；(d) 有现成接入位，可以做有/无配对；(e) 与我们的产出（族、演化、他述、局限、已知未读）对得上。

**1. Old Ideas 的 novelty-judge-bench**（a 强：审稿意见一致的两极样本 / b 强：CC-BY-4.0 / c 强：检索截止 2025-03-01 / d 最强：已有检索条件 / e 中高）
- 做法：把"5 篇截止前摘要"换成我们截至 2025-03-01 的工具输出。对照组有四个：无检索、5 篇摘要、agentic RAG、每次现做的领域摘要。成对和逐点两种形式都报，6 个评委都报。
- 预注册：原文显示一句提示就能使结果变化超过 50 点，所以必须预先固定多个提示变体，按提示分别报告。
- 要求和风险：每个配置只有 154 对，统计力有限；涉及多个 ICLR area，需要先把这些领域编译出来，成本不低。
- 补充：NovGauge（CC-BY-4.0，619 对），按 method 维度做诊断，测族和谱系语境能否提高 Verified F1。

**2. MasterSet 的 Type I（实验基线）子任务**（a 强：精确 ID / b 中强：代码 MIT，HF 无 license 字段 / c 强：候选池 2018–2024，评测 2025 / d 强：候选池固定，可以挂工具 / e 强：基线选择取决于方法族和演化）
- 做法：给 agent 一篇 2025 年论文的问题陈述，让它用或不用我们截至 2024 年末的工具推荐必比基线，看 R@k。
- AgentExpt 的消融说明"他述"对这个任务有约 17% 的贡献，我们可以先复现这一点，再看族和演化能否带来额外收益。
- 风险：类型标签由 LLM 标注（据子代理，抽检 510 条）；只覆盖 AI/ML；2025 年的评测集与现役模型训练截止有重叠，需要按模型截止分层报告。
- 备选：SciPaths（专家金标，MIT，带时间设定，最直接对应谱系），金标前置贡献给出了明确的上界。

**3. LimitGen-Human**（a 中：审稿意见经 GPT-4o 过滤，用 LLM 评测 / b 中：HF 非 gated，许可以论文声明为准 / c 中：特意选 ICLR 2025 防污染 / d 强：已有 RAG 对照 / e 强：对应"共同局限与开放问题"和他述批评）
- 做法：在原 RAG 位置挂我们的工具，比较无 RAG、原 RAG（GPT-4o +12.2pp）和我们三组。
- 原文把失败直接归因于 "a lack of knowledge in related areas"，是三者里对"领域认知是瓶颈"表述最直接的一个。
- 风险：要按模型截止确认 ICLR 2025 是否已被见过。
- 备选：ToC-Bench（金标里有后续施引批评，与他述同构），等数据发布。

想法生成类不作主下游。ResearchBench（背景综述槽可以直接替换，12 个学科，CC-BY-NC）可以作第四个、即"创新生成侧"的代表，但必须同时报原论文匹配度这类客观指标，不能只用 LLM 评委打的新颖性。理由是 Si、Ideation-Execution Gap、RQ-Bench、Old Ideas 四项工作都表明 LLM 评委判新颖性不可靠。

## 5. 对叙事的含义

- **可以引用的动机**：LimitGen 和 SciPaths 原文都写明"缺领域知识"或"不知道该找哪些 building blocks"是失败原因；Idea Novelty Checker 中，检索选择导致 89.66% 对 13.79% 的差距；Axiomatic 发现已有系统都过不了时间公理。
- **必须正面回应的反例**：Old Ideas、SCOPE、RINI 都表明"给更多文本"没用。我们的卖点只能是"结构、他述、截至 T 的状态"，不能是"更多上下文"。每个下游都要设"同预算的平铺检索"和"每次现做的摘要"两个对照。
- **不能声称首个**：用 taxonomy 辅助新颖性分析（OpenNovelty，每次现做）、用引文语境描述 artifact（AgentExpt）、用引文演化图辅助 ideation（GoR、Tree-of-Ideas、CoI、MIR）都有人做过。目前没看到的组合（限于本轮检索）是：把持久的、可按时间切片的领域结构，作为工具接到新颖性判断、基线选择、局限识别这类判断任务上，并在带截止的设置下做配对评测。

## 6. 未核实清单（不进主表结论）

- **GraphMind KDD'26（SciNova）**：KCL 仓库、ACM 和 OpenReview 都被 Cloudflare 拦截，效应量没读到，只核了摘要（KCL 页面）。
- **Axiomatic Benchmark**：原文说发布了 ID 和脚本，但 HTML 里没找到仓库链接。
- **LiveIdeaBench-Forecasting 的过去窗口对照（Δ +0.029）**：只有子代理读到，我复查时 OpenReview 被拦截；它与 2609.00747 是否同一工作只比对了作者和标题。
- **许可未声明或未核**：
  - NovBench：数据在 Google Drive，没打开；仓库无 license。
  - RINoBench、Idea Novelty Checker、Ideation Arena 仓库无 license 文件。
  - LimitGen：HF 无 license 标签，论文声称 CC BY 4.0（未核）。
  - MasterSet：HF 无 license 标签。
  - ScholarCatalyst 的 LICENSE 文件是 CC BY-NC 4.0（已核），与前一轮 HF 的 CC-BY-NC-4.0 一致。
- **数据拿不到**：RQ-Bench；2505.24615 和 2602.06054 只有匿名链接，没打开；AgentExpt、SCOPE、ToC-Bench 都说会发布，暂无链接。
- **只在图中、没抽到的数字**：InnoEval 的消融数字；Scideator、SciPIP 的消融；Nova 完整版的消融（子代理只拿到部分）。
- **只有子代理读过正文**（表中标 ‡）：AI Idea Bench 2025、MOOSE-Chem 系列、LiveIdeaBench（Nature Communications 版）、Scientist-Bench、Nova、SciPIP、CiteBench、OARelatedWork、GREP、AcademicEval、DataFinder、BAGELS；另外 MasterSet 的金标标注方式（Gemini 2.5 Flash + 510 条抽检）。
- **子代理报告的其他问题**：Idea2Plan、IdeationSpace 仓库 404；FutureGen 许可接口异常；GAPMAP 仓库已迁移，新地址没查。2026 年以下新工作只读了摘要：RATIO 2608.27394、LigBench 2608.13136、ProjectionBench 2605.30284、DBench-Bio 2603.03322、IdeaTrail 2607.10144、TF-Bench 2605.06345、IdeaAMBIG 2609.10539。
- **发现层没找到**：独立的"idea 去重"基准、跨论文的研究空白（gap）基准（GAPMAP 只做单篇）。
