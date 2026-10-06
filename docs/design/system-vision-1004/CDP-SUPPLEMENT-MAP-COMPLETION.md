# 领域图谱补全：CDP 发现层补充调研

日期：2026-10-04。范围：截至 2026-10-03。本文补充 `RESEARCH-MAP-COMPLETION.md`（下称"前一轮"），只写前一轮没有覆盖或没有核实的工作。

## 引导段

前一轮只用了 WebSearch 做发现。这一轮改用用户的真实浏览器（CDP），走 Google、DuckDuckGo、HF Papers、alphaXiv、arXiv 站内搜索、GitHub、搜狗微信、百度 site:zhihu，以及几家产品的官方文档。结论有三条：

1. 仍然没有工作同时做到这四件事：持久的领域地图（方法族 + 演化关系）、由地图上的结构缺口驱动外部检索、检索结果写回、估计"还缺多少"。前一轮的核心结论成立。
2. 但前三件事的组合（持久库 → 缺口 → 外检 → 写回）已经有弱形态的先例，前一轮漏掉了。分别是：Consensus 的产品教程 "Find and fill gaps in a Collection"；Karpathy 的 LLM Wiki 范式及其衍生论文，其中 Knowledge Compounding（2604.11243）明确实现了 search write-back，并提出了 knowledge-base coverage rate H(t)；ARIS 的 research-wiki（GitHub 17k stars，带 gap_map，文献检索结果写回）；Elicit 的 Projects + Routines。所以"闭环"本身不能再当卖点。
3. 开放世界的覆盖估计（capture–recapture 等）在 LLM agent 里仍然只有 Undermind 一家，而且只在单次查询内，没有找到新用法。Knowledge Compounding 的 H(t) 是另一个量：它指任务所需信息能被库直接满足的比例，是查询侧命中率的解析模型，不是对"领域里还有多少论文没找到"的估计。

## 0. 方法与核实说明

- web-access skill 用 Skill 工具加载不到，于是直接读取 `SKILL.md` 并照做。全程只用一个自建的后台 tab，结束时已关闭。
- 风控与绕行：Google 中途触发过一次 "unusual traffic" 验证页，我没有硬闯，改走 DuckDuckGo，Google 稍后自行恢复。X 未登录被墙，改用 Google `site:x.com`，信号很弱，没有产出新线索。知乎同样未登录，改用百度 `site:zhihu.com`。asta.allen.ai 和 allenai.org/papers/asta-guide 在 CDP 下都返回 403。
- 验证层规则：凡是写"存在某机制"，都打开了一手页面核实。arXiv 条目全部核对过 abs 页的 citation meta。标 ◆ 的条目用 HTML 或 PDF 全文 grep 过关键机制词，确认的是正文而非参考文献。产品条目一律以官方文档或官方 FAQ 原文为准。
- 前一轮"未核实"清单里的 SurveyAgent-HKA（2609.05938）、DeepSurvey（2605.29522）、ReLTEx（2608.10970）、SGHA（2608.17501）、Paper Circle（2604.06170）这次已核实摘要，均不做持久图上的缺口驱动获取，归入第 2 节的低威胁表。

## 1. 判决表（主表：与"持久图 + 缺口外检 + 写回 + 覆盖估计"相关度高）

威胁等级的含义：真＝四件事基本都做了，直接撞车；半真＝占了其中两到三件，或者机制同构但对象、场景不同；假＝只是表面相似。

| 工作 | 一手链接 | 发现渠道 | 持久结构？ | 缺口驱动外检？ | 写回？ | 估计覆盖？ | 和我们的关系 |
|---|---|---|---|---|---|---|---|
| Consensus "Find and fill gaps in a Collection" | docs.consensus.app/search-basics/fill-collection-gaps（CDP 读原文） | curl 拉 docs.consensus.app/llms.txt 发现 → CDP 读原文 | 是。用户的 Collection（文档示例约 20 篇） | 是。Research Agent 在 Collection 内跑 4 次检索，生成 topics × populations 的 gaps matrix；随后在 400M 全库上"runs one search per gap in parallel with a 5-year recency filter"，178 篇候选中筛出 20 篇 | 半。由用户手动 bookmark 存回 Collection；文档写明 "Repeat this over time to keep a Collection current" | 否 | **半真，产品层面形态撞车**。缺口是 LLM 对小集合即时生成的矩阵，不是结构化的方法族或演化图；写回靠人工；没有估计，也没有评测。但审稿人可以拿它说"缺口 → 外检 → 写回"是现成的产品功能 |
| Elicit Projects + Routines | support.elicit.com/en/articles/15805744-elicit-projects；/17220392-routines-in-elicit | CDP 浏览 Elicit 帮助中心 | 是。Project 关联一个 collection；项目内的会话"prioritize the papers in that collection"；项目内保存的论文默认进入该 collection | 否。Routine 按日或周定时运行，例如 "find relevant new evidence…, update our summary…, tell me what changed"，触发方式是主题加时间，不是缺口 | 是。"Track artifact" 让 artifact 持续更新 | 否 | **半真**。跨会话持久集合加定期更新 artifact，属于 living review 的监视模式，不是由缺口拉取的补全 |
| Undermind（补充前一轮） | undermind.ai 首页 FAQ "It didn't find my paper. Why?"；undermind.ai/mcp 演示 | CDP | 否。"KEEP UP… keeps tabs on your areas of interest and notifies you" 只是主题监视 | 单查询内分阶段（semantic → citation trails → authors），演示里 "Fourth pass added nothing new. Stopping now." | 否 | 是，但限单查询。原文："By statistically modeling this process, we can determine what fraction of relevant papers we have likely found" | 半真（前一轮已列）。新增的一点：它有主题监视，但监视不累积领域结构 |
| ◆Karpathy "LLM Wiki" gist | gist.github.com/karpathy/442a6bf555914893e9891c11519de94f（2026-04-04 创建，2026-09-13 更新） | 知乎（百度 site:）→ GitHub 转录页 → 原 gist | 是。"persistent, compounding artifact" | 半。Lint 环节检查 "data gaps that could be filled with a web search"，并且 "suggesting new questions to investigate and new sources to look for" | 是。"explorations compound in the knowledge base just like ingested sources do" | 否 | **半真，范式级先例**。对象是个人 wiki，不是学术方法族地图；缺口类型是启发式清单；没有估计，也没有评测。传播面极大（已有百度百科词条，知乎和 CSDN 解读大量），审稿人大概率知道 |
| ◆Knowledge Compounding（Qing Claw） | arXiv 2604.11243（Wen & Ku，2026-04-13） | arXiv 站内搜 "LLM wiki" | 是。LLM Wiki 实现，每日 INGEST/LINT、每周 MERGE | 是，但触发条件是查询未命中：wiki 不足以回答时，CEO 才调用 search expert（circuit-breaker 规则："Absolutely forbidden: directly searching without first letting the wiki expert check"） | 是。原文："search results are written back into entity pages, closing the compounding loop"，并声称 "none of these projects implements… search write-back" | 半。定义了 knowledge-base coverage rate H(t)＝任务所需信息能被库直接满足的比例，递推式 H_{i+1}=H_i+α(1−H_i)p_i，属于解析模型，不是对缺失论文的统计估计 | **半真，机制最接近**。"持久库 → 未命中 → 外检 → 写回"加上"coverage"一词都占了。差别在于：通用场景（OpenClaw 开发手册），触发是查询未命中而不是结构缺口，实验只有 4 条顺序查询（Chunk-RAG 13.6K < Compounding 47K < Long-Context 305K tokens，摘要里写的"vs RAG 305K"与正文的排序不一致），没有覆盖估计的校准 |
| ◆ARIS research-wiki | github.com/wanshuiyin/Auto-claude-code-research-in-sleep（skills/research-wiki/SKILL.md；17.0k stars，最近提交 2026-09-29）；论文 arXiv 2605.03042 §4.2 | Google 搜 "llm-wiki" → GitHub | 是。按项目划分，包含 papers/ideas/experiments/claims 四类实体、8 种关系（含 `addresses_gap`、`supersedes`），另有 `gap_map.md`（稳定 ID G1, G2…） | 半。`/idea-creator` "treat top gaps as search seeds"，这里的 gap 是研究空白，用于 ideation；`/research-lit` 是按主题检索 | 是。Hook 1：`/research-lit` 结束后把 top 8–12 篇 `ingest_paper` 写入 wiki，并 `add_edge` | 否。`verify_wiki_coverage.sh` 只对比会话产物里出现的 arXiv ID 与已入库 ID（作者原话 "Diagnostic (NOT a gate)"），记录的是"读过的论文是否入库" | **半真**。持久研究图、缺口表、检索写回都有，ML 研究者圈子里传播很广。但缺口服务的是选题，没有覆盖估计，闭环也没有评测 |
| ◆DeepRefine | arXiv 2605.10488（2026-05-11，v2 2026-08-23） | arXiv 站内搜 "LLM wiki" | 是。可作用于任意已建好的 KG 或 LLM-Wiki | 缺口驱动是，外检否。用用户查询做 Answerability Judgement → Error Abduction，定位缺证据、缺链接、错误、冗余；动作只有 `insert_edge / delete_edge / replace_node`，依据的是原文（"will not change the original text chunks"） | 是（增量更新 KB） | 否 | 半真（只占"诊断 → 写回"）。GBD reward 加 RL 学习修补策略，可以作为"已入库论文方面缺失 → 补抽"这一动作的先例或基线 |
| ◆WikiLoop | arXiv 2607.26604（2026-07-29） | 同上 | 是。agent-native Wiki | 否。Builder 拿到"new source"后提出结构化编辑 | 是。编辑效用由冻结的 Navigator 在下游任务上评分 | 否 | 假到半真。构建与使用的反馈耦合，可以借来作为"写回是否带来增益"的评测思路 |
| ◆SearchOS-V1 | arXiv 2607.15257（2026-07-16） | HF Papers 搜索 | 半。Frontier Task、Evidence Graph、**Coverage Map**、Failure Memory 是任务内的外置状态；跨会话保留的只有 search skills | 是。"continuously refills freed slots with tasks targeting unresolved coverage gaps" | 任务内写回 | 否 | 半真（机制同构，限单任务）。开放域的 schema completion，不是学术领域图 |
| ◆DualGraph（A Tale of Two Graphs） | arXiv 2602.13830（2026-02-14） | Google（CDP） | 否。原文："The KG starts empty and is constructed incrementally from evidence retrieved during the iterative web research process" | 是。"search queries are derived from knowledge graph topology (e.g., missing links, under-supported nodes)" | 任务内写回 | 否。停止条件是四个维度的 LLM 评分超过阈值 | 半真（机制同构，限单任务）。时间早于 SciAtlas，"KG 拓扑缺口 → 定向查询"在 deep research 里已经有了 |
| IBM "Agentic Workflows for Gap-Aware Literature Reviews" | research.ibm.com/publications/agentic-workflows-for-gap-aware-literature-reviews（AGU 2025 会议摘要） | DuckDuckGo（CDP） | 半。"construct topic-specific knowledge graphs from scientific corpora" | 半。"identifying gaps in reasoning, coverage, or evidence using graph traversal and contrastive retrieval" | 未述 | 否 | 半真（只有摘要可读，无全文、无数值）。学术场景里的"KG 上找覆盖缺口"提法 |
| TechRAG | arXiv 2606.01613（2026-06-01） | Google（CDP） | 是。固定的领域语料（约 4 万页）加 Neo4j KG | 半。证据充分性用 100 分量表打分，不足时 "searches external academic databases through optimize–search–vet loops" | 全文未见写回语料 | 否 | 假到半真。"本地不足 → 外检"，但不写回 |

## 2. 已核实、相关度较低的条目（不展开）

| 工作 | 一手 ID | 发现渠道 | 为什么低 |
|---|---|---|---|
| ◆Intern-Atlas | arXiv 2604.28158 | HF Papers | 方法演化图（7 类边、verbatim 引句），有 "Structural gap patterns"，但用于 ideation，图是静态的。前一轮的表里没有列，这里补上（项目记忆里早已登记为竞品） |
| ◆AskChem | arXiv 2607.28618 | HF daily（308 upvotes） | 147K 篇论文、2.4M 条 claim，含 "living taxonomy"（4,931 个节点），批量构建，正文未见缺口驱动获取。是 LKM 同类的持久结构 |
| Scientific Contribution Graph | arXiv 2605.15011 | HF Papers | 655k 篇论文、36M 条 prerequisite 边，批量构建；用 temporally-filtered backtesting 评测 |
| ◆LLM-Wiki（Retrieval as Reasoning） | arXiv 2605.25480 | arXiv 搜索 | 文档编译成 wiki，另有 Error Book 做自纠，不做外部获取 |
| WiCER | arXiv 2605.07068 | arXiv 搜索 | 编译 → 探针评估 → 找出被丢掉的事实 → 重编译（CEGAR 式），限于给定语料。可作为"方面缺失 → 重读"动作的先例 |
| Streaming Knowledge Compilation | arXiv 2606.09877 | arXiv 搜索 | 被动文档流加 materiality pinning，给出 regret bound |
| Memory as Metabolism / Beyond Memory template / Knowledge-Centric IS / WFM | arXiv 2604.12034 / 2607.24759 / 2607.02609 / 2609.18182 | arXiv 搜索 | LLM Wiki 生态里的治理、模板、架构、表示学习，没有缺口获取 |
| Explainable Innovation Engine | arXiv 2603.09192 | 搜狗微信 | 写回的是推理合成出来的方法节点，没有外部获取 |
| litgraph | github.com/gaztrabisme/litgraph（2026-08-03） | DuckDuckGo | 1,285 篇 LLM 文献的 KG，结构欠探索区域 → 研究空白 dossier；"gap"指科学空白，不是库的覆盖缺口；有 temporal backtest |
| GapSpotter | IEEE doi:10.1109/ICANCS65819.2025.11376858 | DuckDuckGo | 语义聚类找研究空白（方法、理论、数据、人群四类） |
| SGHA | arXiv 2608.17501 | 前一轮未核实，本轮补核 | corpus-first，"structural gap" 指"文献里存在但尚未解决的问题"，语料有界 |
| SurveyAgent-HKA / DeepSurvey / ReLTEx / Paper Circle | arXiv 2609.05938 / 2605.29522 / 2608.10970 / 2604.06170 | 前一轮未核实，本轮补核 | 单次综述生成、taxonomy 扩展、单篇 KG，都没有持久的缺口驱动获取 |
| EviMem / HDRI(INFOMINER) / AREX / VeriTrace | arXiv 2604.27695 / 2605.10224 / 2607.21461 / 2605.26081 | arXiv、HF | 单任务内的 gap-driven 迭代检索 |
| EMBL AI Librarian | arXiv 2607.28229 | HF Papers | Europe PMC 之上的逐查询知识层，不持久 |
| MedKGent | arXiv 2508.12393 | HF Papers | PubMed 按日增量构建时间 KG，被动摄入 |
| Benchmark Radar | arXiv 2609.11115 | alphaXiv | 每日从 37 个源被动发现，属于 living database |
| ChatGPT Projects | help.openai.com/en/articles/10169521-projects-in-chatgpt | DuckDuckGo → CDP | "ChatGPT can reference previous chats within a project"；项目里可以用 deep research；没有领域结构，没有覆盖估计 |
| 评测资源：ScholarCatalyst | arXiv 2610.02202（2026-10-01） | HF、alphaXiv | 只能检索项目启动时已有的文献；agentic search 0.42 vs embedding 0.48 R@20。可用作时间切分评测 |
| 评测资源：RAP | arXiv 2609.10092 | HF Papers | 278 个领域的滚动基准；"State carry-forward outperforms direct Forecast"，对"持久状态有用"是间接证据 |

## 3. 前一轮漏掉的最重要的 5 条

1. **Consensus "Find and fill gaps in a Collection"**：持久集合 → 缺口矩阵 → 每个缺口一次全库检索（带时间过滤）→ 存回 → "Repeat this over time"。这是前三件事的产品化形态。本项目的叙事不能再说"首次让文献库由缺口驱动地自我补全"。
2. **LLM Wiki 传播链**，包括 Karpathy gist、Knowledge Compounding 2604.11243、DeepRefine 2605.10488、WikiLoop 2607.26604、WiCER 2605.07068。"编译一次、持久复利、lint 找缺口、外检写回"在 2026 年 4–9 月已成为社区共识范式。其中 Knowledge Compounding 同时占了 search write-back 和 "coverage rate H(t)" 两个说法，相关工作里必须正面区分。
3. **ARIS research-wiki**（17k stars，arXiv 2605.03042）：ML 研究者常用的持久研究图，带 gap_map，文献检索结果自动写回，审稿人可能亲自用过。
4. **DualGraph 2602.13830 与 SearchOS 2607.15257**：deep research 里"KG 拓扑缺口或 coverage map → 定向查询"的机制先例，比 SciAtlas 的 coverage-guided retrieval 更早或者同期，但都限于单任务。
5. **Elicit Projects + Routines**：项目级持久 collection 加定时 agent 更新 artifact，是"持久结构 + 写回"的又一个产品先例，触发方式是时间而不是缺口。

## 4. 对叙事的含义（推断，供讨论）

- 不能当卖点的：持久库、缺口驱动外检、写回，以及这三者的闭环本身（Consensus、Knowledge Compounding、ARIS 都已占）。
- 仍是空位、需要实证支撑的有四点：
  - 缺口定义在方法族和演化结构上，而且是类型化的。现有工作的缺口分别是查询未命中（Knowledge Compounding）、LLM 即时生成的矩阵（Consensus）、研究空白（ARIS、litgraph、Intern-Atlas）。
  - 分区域、开放世界的缺失估计，并且做过校准。Undermind 只在单查询内估计，Knowledge Compounding 的 H(t) 是查询侧的解析模型。
  - 跨任务摊销的实测。Knowledge Compounding 只有 4 条查询，量的是 token，不是召回。
  - 严格评测，例如遮蔽族成员、时间切分（ScholarCatalyst 或 RAP 式）、写回增益（WikiLoop 式）。
- 相关工作要新增两个对照：Knowledge Compounding（区分 coverage 的含义）和 Consensus 的 gap-fill（区分结构化缺口与即时矩阵、自动写回与人工写回）。DeepRefine 可以作为"已入库论文补抽"动作的基线。

## 5. 未核实清单（不进主表）

- MindForge（知乎 2026-08-14，"基于 LLM-Wiki 会思考、探索、规划、生长的知识铸造平台"）：只看到百度摘要，没有打开原文，也没找到 repo。
- SciSpace 的 My Library / Notebook / Library Search：只看到 YouTube 视频标题，scispace.com/agents 页面上未见相关描述。
- Ai2 Asta 有没有持久 library 或项目功能：asta.allen.ai 和 asta-guide 都返回 403；allenai.org/blog/asta 正文未提及。2606.08301（Asta 引用系统）没有读。
- Perplexity（Spaces）和 Google Gemini Deep Research 的记忆或项目功能：本轮没查。
- GitHub 上的研究 wiki 衍生项目（AIwork4me/open-llm-wiki、obsidian-llm-wiki、Alexandre-Tortoza/bastion、elonwoo-02/ReadR）：只看了搜索结果里的描述，没有打开 README。
- 知乎"中国科学院成都文献情报中心杨帅团队 JMI 综述"：只有百度摘要。
- X/Twitter：Google `site:x.com` 返回的都是泛 LLM Wiki 讨论，没有学术地图补全的新线索。
- 前一轮未核实而本轮仍未核实的：Taxonomy Maintenance in the Wild（doi:10.1145/3837127）、"When Should Multi-Round RAG Stop?"（2608.13237）、BibliZap、Thoth 2.0、citationchaser。
- 覆盖估计新用法：在 arXiv 短语检索（"capture-recapture" LLM 返回 0 条）、HF Papers、Google 上都没有找到 LLM agent 用 capture–recapture 或 species-richness 估计文献缺失的工作。这是"未找到"，不等于"不存在"。
