# 顶尖科学智能体在"领域认知"环节缺什么（截至 2026-10-05）

调研问题：现有 AI scientist、科研助手和 deep research 产品在"读文献、理解领域"这一步怎么做，哪些失败是领域认知不足造成的，已有的持久领域结构效果如何。目的是判断我们的"历时领域认知"能补哪一块。

核实标记：
- ✓：我本人打开一手来源（arXiv abs/html、官方博客、Nature 文章页、bioRxiv 全文），逐字 grep 到引号内原文。
- ◇：并行子代理打开一手来源核实，我没有二次 grep。

只出现在二手转述里的条目一律进末尾的"未核实清单"，不进主表。

---

## 一、主表

本表用到的我方组件，按以下简称引用：
- 族：方法族与成员
- 演化：演化关系
- 他述：后人如何描述某个工作，区别于作者自述
- 局限：共同局限与开放问题
- 边界：已知但未读的论文
- 回放：回放到任意时间 T

| 系统 | 一手链接 | 文献/领域认知环节怎么做 | 持久知识库? | 自报或被评出的领域认知类失败（含数字） | 我们能补哪一块 |
|---|---|---|---|---|---|
| Sakana AI Scientist v1 | arXiv 2408.06292 ✓ | idea 生成后调 Semantic Scholar API 和 web 做新颖性过滤：10 轮，每轮返回前 10 条及其摘要。写 related work 时再查 20 轮 S2 补引用 | 无，每个 idea 从零检索 | 自报："sometimes struggles to find and cite the most relevant papers"；示例论文"the bibliography is small at only 9 entries"✓。第三方 Beel 等（2502.14297 ✓）："classified all 10 generated ideas and both seed ideas as novel"，归因于"reliance on keyword matching"；"only five (14.7%) of the total 34 references were from 2020 or later"；"failed to cite existing papers on the topic that used exactly the same terminology" | 族+边界：把"检索没命中"换成"在族中的位置+边界内是否有近邻"；回放解决引用陈旧 |
| AI Scientist-v2 | arXiv 2504.08066 ✓；sakana.ai/ai-scientist-first-publication ✓ | 在 idea 阶段把 Semantic Scholar 接进循环，查询"to assess the novelty of a proposed concept and identify relevant prior work" | 无 | 内审："occasionally introduced inaccuracies in citations"。被录用的那篇"omitted key references, notably Hochreiter and Schmidhuber (1997), and instead relied on general textbook citations"。workshop 3 投 1 中，6.33 分 ✓ | 他述+演化：奠基工作的归属（谁提出 LSTM）应来自后人的一致他述，不靠检索碰运气 |
| AI Scientist（Nature 版） | Nature s41586-026-10265-5（2026-03-25）✓；arXiv 2606.15497 ✓ | 同 v1/v2，S2 加 web | 无 | "many types of hallucinations such as inaccurate citations"。"incorrectly attributing the invention of the LSTM to Goodfellow et al."。"it is possible for The AI Scientist to produce a paper on an idea and not identify that this idea has already been explored"。补充材料："sometimes failing to find the most relevant papers, hallucinating references" ✓ | 同上，加"重复研究检测"（族内近邻和他述对照） |
| Agent Laboratory | arXiv 2501.04227 ✓ | PhD agent 调 arXiv API，有三个动作：summary 取"top 20 papers"的摘要、full text 取全文、add paper 加入综述，迭代到 N 篇为止 | 无 | 文献综述阶段成功率 gpt-4o/o1-mini/o1-preview 为 60%/70%/80%。原文措辞是"had a high rate of failure, at 60%, 70%, and 80%"，但用平均成功率 94.3%/92.8%/95.7% 反推，这三个数应是成功率。故障："repeatedly use the summarize command until the maximum phase steps"✓。第三方 Hidden Pitfalls（2509.08713 ✓）："82.4%"的运行选了列表里的前 4 个 benchmark，去掉 SOTA 参考后仍有 79.6% | 族（该任务的标准 benchmark/基线集合）+局限 |
| AgentRxiv | arXiv 2503.18102 ✓ | 多个 Agent Laboratory 共享一个"预印本服务器"，可读彼此此前的研究 | 有，跨实验室共享的自产论文库（不含外部文献结构） | 正面效果：能读到先前研究时"11.4% relative improvement over baseline on MATH-500"，多实验室共享时 13.7% ✓ | 我方是外部领域的持久结构，与它的"自产成果库"互补 |
| Google AI co-scientist | arXiv 2502.18864 v1 ✓（v2 于 2026-06-29 更新）；Nature s41586-026-10644-y（2026-05-19）✓ | Generation agent 用 web search 读文献；Nature 版写作"retrieves and reads relevant research articles … building a knowledge base of scientific facts"。Reflection agent 审新颖性，Proximity agent 去重 | 仅单次运行内的 knowledge base / context memory | v1："may miss critical prior works due to reliance on open-access literature"；"may also omit consideration of prior work on occasions where it has incorrectly reasoned that the work is not relevant"。Nature 版："omission of critical previous work behind paywalls and a systemic lack of access to negative experimental results"，并把"improving citation recall"列为改进方向。专家新颖性评分 3.64/5 ✓ | 边界（付费墙后的已知未读论文显式标出）+局限（负结果、开放问题） |
| FutureHouse PaperQA2 | arXiv 2409.13740 ✓ | Paper Search、Gather Evidence（top-k 稠密检索，再由 LLM 重排并做上下文摘要）、Citation Traversal；读全文 | 无持久领域结构，按问题检索 | LitQA2：precision 85.2%，accuracy 66.0%，平均每题用 14.5 篇。矛盾检测：每篇 2.34 处，其中 70% 经专家确认 ✓。自报的领域认知失败不多，这类系统做得最好的是逐题检索 | 它是逐题检索的上限；我们补的是跨题共享的领域结构（族/演化/局限） |
| FutureHouse 平台（Crow/Falcon/Owl） | futurehouse.org/research-announcements/launching-futurehouse-platform-ai-agents（2025-05-01）✓ | Crow 简答，Falcon 深度综述加 OpenTargets 等库，Owl 专答"Has anyone done X before?"，全文开放获取语料 | 无 | 博客无失败数字。Owl 的存在本身说明"有没有人做过"被当作一类单独的检索问题 | 边界+他述：把"有没有人做过"从一次检索变成可查询的领域状态 |
| FutureHouse Robin | arXiv 2505.13400 ✓ | Crow 先综述"151 papers"提出 10 个机制；再综述"about 400 papers"提出 30 个候选药物；Falcon 为每个候选写评估报告；LLM 锦标赛排序 | 无 | 自称"Robin is the first to propose their application in dry AMD"，同时承认"ROCK inhibitors have been previously suggested for treatment of wet AMD"；最初的 Y-27632 线索追溯到检索中的"a single paper" ✓。没有新颖性失败数字，首创声明未经独立核查 | 演化+边界：给"首个提出"类声明提供可核查依据 |
| FutureHouse/Edison Kosmos | arXiv 2511.02824 ✓ | 每轮并行跑数据分析 agent 和文献 agent，每次运行"read 1,500 full-length scientific papers across 36 literature review agent rollouts" | 有，"structured world model"，在两类 agent 之间共享信息 | 102 条陈述："82.1% of literature review-based statements were validated with primary sources"，"57.9% of synthesis statements were accurate"，总体 79.4%。7 个发现中"Three … independently reproduce findings from preprinted or unpublished manuscripts"。"no automated method to reliably evaluate if a claim is accurate, novel, and significant" ✓ | world model 只管任务上下文，不含领域结构。我们补"综合类"陈述所需的族/局限/他述，以及新颖性判据 |
| Ai2 CodeScientist | arXiv 2503.22708 ✓ | 输入是"a human-curated list of papers"；在论文×代码块组合上做遗传搜索 | 无，人工给定论文清单 | 19 个候选发现中只有 6 个过了"minimally sound and incrementally novel"；"more than half of the potential discoveries were rejected by an internal code review" ✓。新颖性要靠人工逐条论证（附录 G） | 族+演化：替代人工整理的论文清单，并提供"增量新颖"所需的近邻 |
| Ai2 Theorizer | arXiv 2601.16282 ✓ | PaperFinder 检索，取开放获取 PDF 转全文，按查询生成抽取 schema 建证据库（◇：每个查询 K=100 篇，共 13,744 篇） | 每个查询一个临时证据库 | "novelty in this setting reflects absence from the retrieved papers, not empirical validity"。回测精度：accuracy-focused 0.88/0.90，novelty-focused 0.34/0.61（参数知识/文献两种生成方式）。"limited to fields whose scholarly literature is primarily open access" ✓ | 边界：把"检索中缺席"和"领域中缺席"分开 |
| AstaBench（Ai2 Asta 评测） | arXiv 2510.21652（ICLR 2026）✓ | 评测套件：文献理解（PaperFinding、LitQA2、ScholarQA-CS2、表格）、代码、数据分析、端到端 | — | 文献理解最好约 80%，ScholarQA-CS2 最好约 85%，"higher performance is driven by the citation subscores"。综述表格 recall"around 43%"。端到端完成全部步骤最高 5% ✓ | 综述表格（跨论文属性对齐）对应我们的族/成员属性 |
| InternAgent（原名 NovelSeek） | arXiv 2505.16938（v1 标题为 NovelSeek）✓ | Survey Agent 有两种模式：literature review 模式按关键词组合检索、读摘要打相关分；deep research 模式下载全文、扩展关键词再搜 | 无 | 自述难点："ensuring the novelty of proposals often demands a deep understanding of the broader scientific context" ✓。无新颖性失败数字 | 族+局限 |
| InternAgent-1.5 | arXiv 2602.08990 ✓ | 多学科知识图谱（文献/概念/方法/数据集/问题），图搜索加稠密检索 | 有，KG 加 Structured Cognitive Memory | 记忆模块只用"prompt-based case studies"评估，无定量 ✓ | 我们可提供它缺的东西：时间维、他述 vs 自述、定量验证 |
| Dolphin | arXiv 2501.03916 ✓ | S2 API 取标题和摘要，按主题与任务属性相关性排序过滤；新颖性检查"Following AI-Scientist … simply prompt the LLMs" | 有 idea bank，存自产 idea 的 embedding，不存领域结构 | 加入论文排序后"novel ideas significantly increased from 8/20 to 19/20"✓，说明输入文献的相关性直接决定新颖率 | 族：给出"相关论文"的结构化定义 |
| MLR-Copilot | arXiv 2408.14033 ✓ | "We take the research paper as input"：从单篇论文出发，再检索 HF 上的模型和数据 | 无 | 无领域认知失败数字 | 演化：单篇论文之外的上下游 |
| ResearchAgent | arXiv 2404.07738 ✓ | 从一篇核心论文出发，沿引用图扩展参考文献；另有"entity-centric knowledge store"（跨论文概念共现） | 有，实体共现库 | 消融（Problem/Method/Experiment 三项）：完整版 4.52/4.28/4.18，去掉实体 4.35/4.13/4.02，实体和参考文献都去掉 4.20/4.03/3.92 ✓（LLM 评分） | 早期证据：持久结构有增益但幅度小；我们把共现升级为类型化的演化和他述 |
| Chain of Ideas (CoI) | arXiv 2410.13185 ✓ | 从 anchor paper 向前（参考文献）和向后（后续工作）选文献，排成"chain structure to effectively mirror the progressive development"，再让 LLM 预测趋势 | 无，每题临时构造 | 消融（Idea Arena，原版对自己打平记 50）：去掉 CoI 构造后 Novelty 41、平均 42.4；去掉 Future Trend 平均 46.2 ✓。作者承认"there is no guarantee that the generated ideas will be novel" | 演化：CoI 是我方"演化关系"的单链临时版，直接佐证需求 |
| SciMON | arXiv 2305.14259 ✓ | 检索语义近邻、KG 近邻（"constructed on papers before 2021"）和引用近邻作为灵感；迭代 novelty boosting | 有，按时间切分的背景 KG | "ideas still fall far behind scientific papers in terms of novelty, depth and utility"；GPT4FS+KG"often leads to more technical depth" ✓ | 回放：SciMON 已用时间切分防泄漏；我们把它做成通用能力 |
| SciAgents | arXiv 2409.05556 ✓ | 本体知识图谱"developed from around 1,000 scientific papers"，在图上随机路径采样生成假设，再查 S2 判新颖 | 有，静态本体 KG | 无领域认知失败的定量 | 同类先例；差异在于我们是历时的，并区分他述 |
| Virtual Lab（Stanford） | Nature s41586-025-09442-9（vol 646）✓；bioRxiv 10.1101/2024.11.11.623004 v1 全文 ✓ | 多个 LLM 专家 agent 开会，主要靠参数知识，人类给高层反馈 | 无 | 时间错位：agents"did not suggest the very latest machine learning tools (e.g., AlphaFold 3 instead of AlphaFold-Multimer)"，并"wrote code that assumed old function and model names"；原因是"may not be aware of the most up-to-date scientific literature" ✓ | 回放+演化：知道"截至 T 的前沿方法是谁、谁已被谁替代" |
| Biomni（Stanford） | bioRxiv 10.1101/2025.05.30.656746 v1 全文 ✓ | 从"tens of thousands of biomedical research papers"里抽取工具、数据库和 know-how，构成 action space；运行时查 PubMed/Google | 有，抽取出的工具/数据库环境（属于操作知识，不是领域认知） | HLE 子集：Biomni 17.3%，纯"literature agent"12.2%，base LLM 6.0% ✓。无文献类失败数字 | 局限/族：Biomni 证明"离线从论文编译可调用资产"可行，但它编译的是工具，不是认知 |
| AlphaEvolve | arXiv 2506.13131 ✓（标题核实）；2511.02864 ✓ | 文献主要用作题源和已知最优解对照 | 无 | 67 题中"rediscovered the best known solutions in most of the cases" ✓。◇ 有一题由"custom literature search pipeline based on Gemini 2.5"给出建议 | 与本问题关系弱 |
| Gemini Aletheia（Erdős 题） | arXiv 2601.22401 ✓ | 先由 AI 做自然语言验证缩小范围，再由人工判正确性和新颖性 | 无 | 200 个候选中"Meaningfully Correct"占 13 个（6.5%）。13 道"Open"题里"4 through seemingly novel autonomous solutions, and 9 through identification of previous solutions in the existing literature"。"the most challenging step for human experts was not verification, but determining if the solutions already existed in the literature"。"subconscious plagiarism"。1026/397/333/281 号题的 AI 工作"after initial announcements of novelty, be redundant with the literature" ✓ | 边界+他述：最硬的需求证据，判"是否已存在"比验证更难 |
| OpenAI GPT-5 科学加速实验 | arXiv 2511.16072 ✓ | 第 II 章用 GPT-5 做深度文献检索 | 无 | 正面结果："Locating previously published solutions to 10 problems not previously marked as known"；"did not observe any cases in which GPT-5 pretended to have found a correct reference"。局限："limited in perceiving the 'negative space' of mathematics" ✓ | 局限/边界："negative space"（什么已被证明走不通）对应"共同局限与开放问题" |
| DeepScientist | arXiv 2509.26603 ✓ | Findings Memory 是"a list-style database containing thousands of structured records"，含人类前沿知识和自产发现，按 top-K 检索 | 有，列表式发现库，跨轮积累 | "over 5,000 unique ideas … a mere 21 ultimately result in scientific progress"。人类评审："often omitting comparisons to essential baselines or failing to discuss closely related work" ✓ | 族+演化：基线集合与近邻工作；其记忆是扁平列表，没有关系结构 |
| Jr. AI Scientist | arXiv 2511.04583（TMLR 2026 ◇）✓ | 从一篇 baseline 论文的 LaTeX 和代码出发，用 S2 查新颖性、取 BibTeX 和摘要 | 无 | "Because abstracts alone do not contain sufficient information for proper citation, such contextual mismatches frequently occur" ✓；◇ 约 10 个 idea 中 1 个成功 | 他述：引用该说什么，取自后人如何描述该工作，不只看摘要 |
| Denario | arXiv 2510.26887 ✓ | S2 返回标题和摘要，多轮判断；也可调用 FutureHouse Owl | 无 | 判定规则："determine that the idea is new as it has not found relevant papers after multiple iterations"。纯数学论文"Foundational references were often missing"。对一个"has well-established solutions in the literature"的问题，最终"hallucinated an entire paper" ✓ | 族（奠基文献）+边界 |
| AI-Supervisor | arXiv 2603.24402 ✓ | 在多个 venue 并行检索，抽取 OpenReview 分数和审稿人提出的弱点，先摘要过滤再读全文 | 有，"Persistent Research World Model"（KG，跨项目） | 结构指标："16 cross-project connections (vs. 0 for all baselines)" ✓，规模小（三个项目） | 局限：它从审稿意见抽弱点，与我们"共同局限"同向；差在时间维和规模 |
| Intern-Atlas（基础设施） | arXiv 2604.28158 ✓ | 方法演化图，作为 AI scientist 的基础设施 | 有，大规模方法演化图 | 下游效果（作者自报）：idea 评估与专家 Spearman 0.81，纯 LLM 判官 0.58，其中 Novelty 0.84 vs 0.52。idea 生成 Overall 7.20，Semantic Scholar 6.18、BM25 RAG 6.15、No-KB 5.78 ✓ | 直接竞品，证明"演化结构→新颖性判断"有增益；它没有他述/自述、边界、回放 |
| OpenAI deep research | openai.com/index/introducing-deep-research（2025-02-02；2026-02-10 更新）✓ | 浏览大量网页来源；2026-02 起可接 MCP，并"限定在受信任的网站" | 无 | 自报："may struggle with distinguishing authoritative information from rumors"；"minor formatting errors in reports and citations" ✓。◇ system card：PersonQA 幻觉率 0.13 | 作为 MCP 工具接入：提供权威度和领域位置 |
| Anthropic Research | anthropic.com/engineering/multi-agent-research-system ✓ | lead agent 加并行 subagent 检索 | 无（只有计划记忆） | "early agents consistently chose SEO-optimized content farms over authoritative but less highly-ranked sources like academic PDFs" ✓ | 同上 |
| Gemini Deep Research（API） | ◇ 官方文档（2026-09-23 更新） | Google Search、URL Context、Code Execution，可选 File Search/MCP | 无 | ◇ Limitations 一节只列工程限制 | 同上 |
| Claude for Life Sciences / Claude Science | ◇ 官方页（2025-10-20 / 2026-06-30） | PubMed、Wiley 等连接器；◇ "a reviewer agent checks citations" | 无 | ◇ 无失败数字 | 同上 |

---

## 二、失败模式汇总表

根因判定标准："是"表示主要原因是缺少领域级知识（覆盖、位置、时间、归属）；"部分"表示领域认知与判断或执行能力共同作用；"否"表示主要在实现或执行层。

| 失败类型 | 证据（系统 + 数字 + 链接） | 根因是领域认知? |
|---|---|---|
| 1. 重复已有工作 / 误判为新颖 | ① Beel 等：AI Scientist 把 12/12 个 idea 全判 novel，含已知技术（2502.14297 ✓）。② All That Glitters：专家判定"24% of the 50"份 AI 研究文档为改写或大幅借用（2502.16487，ACL 2025 ◇）✓。③ Aletheia：13 道 Open 题有 9 道是文献里早有解；4 道题的 AI 成果宣布后被发现与文献重复（2601.22401 ✓）。④ GPT-5 为 10 道 Erdős 题找到已发表解（2511.16072 ✓），说明这些题的"Open"标记本身就是领域认知缺口。⑤ Kosmos 7 个发现中 3 个与未公开或预印手稿重复（2511.02824 ✓）。⑥ Si 等：298 条评审中 80 条附上已有论文链接说明 idea 不新颖（2409.04109 ✓） | 是 |
| 2. 把"检索没命中"当成"新颖" | Denario："new as it has not found relevant papers"（2510.26887 ✓）。Theorizer："reflects absence from the retrieved papers"，novelty-focused 精度仅 0.34/0.61（2601.16282 ✓）。Glitters：内置检索比对 SSAG 准确率 51.3%/68.5%，给出 oracle 源论文时 88.8%/89.0%，结论是"retrieving relevant papers, not determining similarity, is the bottleneck" ✓。Idea Novelty Checker：识别"not novel"时完整系统 89.66%，只用关键词检索 5.17%（2506.22026 ✓） | 是（边界未知） |
| 3. 新颖性判断器本身不可靠 | Si 等：最好的 LLM 评审准确率 53.3%，低于人类间一致度 56.1% ✓。AI Scientist 的新颖性 prompt：准确率 0.47，κ=0.05，32 次中 18 次默认判"not novel"（2506.22026 ✓）。RQ-Bench："agreement between humans and LLMs dropped to as low as 22%"（2606.12071 ✓）。NovGauge：18 个 LLM 幻觉率 0–39%，正确判断中"over 70%"引用的证据不支持其理由（2609.11234 ✓）。2610.02022：给 gpt-5.4 加检索反而"flips 22 of its correct verdicts to incorrect"，原因之一是"A single close precursor decides the verdict" ✓。RINI：直接给出先前工作仍"no clear aggregate reduction in unsupported novelty"；175 份提案中 137 份认出了相关性，只有 61 份正确归属既有贡献（2609.33284 ✓） | 部分。检索质量和"贡献归属"两者都缺，单靠喂检索结果不够 |
| 4. 引用幻觉 / 错误归属 | OpenScholar：GPT-4o"hallucinates citations 78–90% of the time"（2411.14199，Nature 2026-02-04 ✓）。Agents4Science：只有"approximately 44% of submissions have no hallucinated references"（2511.15534 ✓）。DeepTRACE：deep research 引用准确率 40–80%，PPLX(DR) 无支撑陈述占 97.5%（2509.04499 ✓）。AI Scientist 把 LSTM 归给 Goodfellow（Nature 版 ✓）。Jr. AI Scientist：只读摘要导致"contextual mismatches frequently occur" ✓ | 是（归属和语境）/ 部分（生成层幻觉） |
| 5. 漏引奠基工作 / 相关工作浅 | AI Scientist v1 示例只有 9 条参考文献 ✓；Beel：中位数 5 条，且漏掉"exactly the same terminology"的已有论文 ✓；v2 漏引 Hochreiter & Schmidhuber ✓；Denario"Foundational references were often missing" ✓；DeepScientist"failing to discuss closely related work" ✓；co-scientist"incorrectly reasoned that the work is not relevant" ✓；◇ 2506.01372 用 AI 审稿模型评 28 篇，"Literature Review Deficiencies" 78.6% | 是 |
| 6. 选错或漏掉基线、基准挑软柿子 | Hidden Pitfalls：Agent Laboratory 82.4% 选列表前 4 个 benchmark；AI Scientist v2 给出 SOTA 参考时选 Easy 难度占 47.1%，无参考时 18.0%（2509.08713 ✓）。Ideation-Execution Gap 评审："only compared with the simplest baselines despite well-acknowledged benchmarks"（2506.20803 ✓）。DeepScientist"omitting comparisons to essential baselines" ✓。SCOPE：低层配置（数据集/基线/指标）比高层规划平均低 2.78 分，"search access alone fails to improve plan quality"（2608.03501 ✓）。没有找到专门量化"漏掉公认强基线比例"的工作 | 部分（"该比什么"需要族内标准集合，也受评测投机影响） |
| 7. 时间错位（把旧方法当前沿，或不知道已被替代或已成默认组件） | Virtual Lab：没想到 AlphaFold 3，仍用 AlphaFold-Multimer，代码用旧函数名（bioRxiv v1 ✓）。Beel：34 条参考文献只有 14.7% 是 2020 年及以后 ✓。没有找到直接测量"不知道已被替代或已成默认组件"的研究；SCOPE 评分细则把"adding more recent or stronger SOTAs"当作加分项（◇） | 是。直接测量是空白 |
| 8. 来源权威性判断差 | OpenAI DR："distinguishing authoritative information from rumors" ✓；Anthropic："SEO-optimized content farms over … academic PDFs" ✓ | 部分 |
| 9. 可及性边界：付费墙、开放获取、未发表 | co-scientist 两个版本都自报付费墙导致漏掉关键前作 ✓；Theorizer 限于开放获取领域 ✓；Kosmos 3/7 与未读手稿重复 ✓ | 是（缺少"已知但未读"的边界表示） |
| 10. 综合类陈述准确率低 | Kosmos：综合陈述 57.9%，文献陈述 82.1%，数据分析陈述 85.5% ✓ | 部分 |
| 11. 想法可行性差，执行后掉分 | Ideation-Execution Gap：执行后 AI idea 的新颖性/兴奋度/有效性分别降 1.049/1.760/1.879，人类 idea 几乎不变 ✓。DeepScientist：5000 个 idea 中 21 个带来进展 ✓。AstaBench 端到端完成率 ≤5% ✓。CodeScientist 19 个候选发现中 6 个成立 ✓ | 部分或否（主要在执行层，"局限与开放问题"能提前排除一部分） |
| 12. 想法同质化 | Si 等："out of the 4000 generated seed ideas, there are only 200 non-duplicate unique ideas" ✓ | 否（生成多样性问题） |

---

## 三、持久领域结构：已有做法与效果

| 系统 | 结构形式 | 报告的效果 | 证据强度 |
|---|---|---|---|
| ResearchAgent 2404.07738 ✓ | 实体共现库加引用图 | Problem 维度 4.52 → 去掉实体 4.35 → 都去掉 4.20 | 弱（LLM 评分，增量小） |
| Chain of Ideas 2410.13185 ✓ | 每题临时构造的演化链 | 去掉 CoI 后 Novelty 41（基准 50） | 中（成对比较，自评） |
| SciMON 2305.14259 ✓ | 2021 年前论文的背景 KG | KG 变体技术深度更高，总体仍"far behind"真实论文 | 弱 |
| SciAgents 2409.05556 ✓ | 约 1,000 篇论文的本体 KG | 无定量 | 弱 |
| Kosmos 2511.02824 ✓ | 结构化 world model（任务上下文） | 综合陈述 57.9%，文献陈述 82.1% | 中（专家标注 102 条） |
| DeepScientist 2509.26603 ✓ | 列表式 Findings Memory | 系统内选择机制有效，无领域结构消融 | 弱 |
| InternAgent-1.5 2602.08990 ✓ | 多学科 KG 加三层记忆 | 只有 case study | 弱 |
| AI-Supervisor 2603.24402 ✓ | 持久 KG world model | 16 个跨项目连接，基线为 0 | 弱（规模极小） |
| AgentRxiv 2503.18102 ✓ | 自产论文共享库 | MATH-500 相对提升 11.4%/13.7% | 中（非外部领域知识） |
| Intern-Atlas 2604.28158 ✓ | 大规模方法演化图 | 新颖性 Spearman 0.84 vs 0.52；生成 Overall 7.20 vs S2 6.18 | 中（作者自报，判官含 LLM） |

反向证据（喂结构或检索不等于有用）：
- 2610.02022：检索把 22 个正确判断翻成错误，因为一个近邻就决定了判断 ✓。
- RINI 2609.33284：给出先前工作后，137 份提案认出相关性，只有 61 份正确归属既有贡献 ✓。
- SCOPE：仅有搜索不改善实验计划 ✓。

三条合起来看，缺的不是"更多检索结果"，而是两样东西：一是该工作的贡献在领域里被后人怎样界定（他述），二是它在族里的位置（近邻有哪些、它替代了谁）。这正是我方"他述 vs 自述 + 族/演化"的位置。同时，这也提示我们必须用"喂结构后新颖性/归属判断变好"做下游验证，不能默认有用。

---

## 四、对我们系统的含义（按证据强度排序）

1. **判"是否已有"**：最硬的需求，证据包括 Aletheia、Glitters、Beel、AI Scientist Nature 版的自承。现有系统都把"检索无命中"当新颖。我方的"边界（已知未读）+族内近邻"能把"检索中缺席"和"领域中缺席"分开。
2. **贡献归属与引用语境**：RINI 的 61/137、LSTM 错归、Jr. AI Scientist 的"摘要不够"，都对应"他述 vs 自述"。
3. **该比什么**：族的标准成员集合可以对接 Hidden Pitfalls 和 SCOPE 指出的基线/benchmark 选择问题。
4. **时间错位**：直接测量是空白。我方的"回放到 T"既是能力，也可以做成新的评测（给定 T，问系统"当时的前沿是谁、谁已被替代"）。
5. **局限与 negative space**：GPT-5 论文的"negative space"和 co-scientist 的"lack of access to negative results"对应"共同局限与开放问题"。证据偏定性。

---

## 五、未核实清单（不进主表）

- AstaBench 中 Asta 产品文献 agent 的内部机制，只读到评测数字。
- Kosmos 在 2026 年的后续版本：没有找到一手来源。
- Intology Zochi/Locus 的 2026 动态、Sakana AI Scientist v3 或 RSI 相关工作：只见到搜索结果。
- AlphaEvolve 原论文 2506.13131 正文；Google ERA（Nature）：没有读。
- Gemini 消费端 Deep Research 帮助中心的局限说明；OpenAI Prism 的发布日期（只来自 TechCrunch，属二手）。
- OpenAI 2026 年其他科研页面（GPT-6 相关）：没有读。
- GPT-5.2 Pro"解决 Erdős-333"的原帖：只见到 Aletheia 论文中的转述。
- Bloom 对 GPT-5 Erdős 宣传的批评推文：子代理经 fxtwitter 镜像读取，属半一手。
- DeepScientist 用的检索 API（论文未写）；DeepScientist、Denario、AI-Supervisor 的机构。
- AI-Researcher（HKU，2505.18705）、Medical AI Scientist（2603.28589）、Dr. Claw（2609.00365）：只扫读。
- KG-CoI（2411.02382）的幻觉检测数字：只确认了方法，没有读结果表。
- IdeaBench、LiveIdeaBench、AI Idea Bench 2025：只读摘要，未见"LLM 新颖性判断与人类一致度"的数字。NoveltyBench（2504.05228）◇ 测的是输出多样性，不是科研新颖性，不可引作新颖性判断证据。
- SchNovel（0.54–0.64 / RAG 0.58–0.73）、ResearchBench（45.7%）、SoundnessBench（假阳性 74.0%）：◇，我没有二次 grep。
- 2506.01372 的发表 venue；DeepTRACE 的发表 venue；Idea Novelty Checker 是否发表于 SDP 2025。
- 2605.07723（"146,932 hallucinated citations in 2025"）：只读了摘要。GPTZero 统计的 NeurIPS 2025 幻觉引用（51 篇）：属厂商博客，没有独立复核。
- Nature 版 AI Scientist 的 SI PDF：我读的是 arXiv 2606.15497 版本的补充材料。
- 对 AI Scientist-v2 那篇 workshop 录用论文的第三方复核，以及对 Glitters 的反驳工作：都没有找到。
- Aletheia 的"9 道靠文献"取自我读到的 HTML 摘要。子代理读到的是"8"，可能是版本差异，未逐版核对。
- Virtual Lab 的 Nature 正式版正文在付费墙后，时间错位原文取自 bioRxiv v1。正式版是否保留这段未核。
