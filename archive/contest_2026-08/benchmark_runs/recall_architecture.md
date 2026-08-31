# 华为赛题三参考系统召回架构调研：PaSa / SPAR / Ai2 Paper Finder

> 调研日期: 2026-08-26。核心问题：在海量论文库里怎么召回符合用户意图的论文。
> 覆盖：a) 召回物理来源 b) 查询→召回链路 c) 迭代策略 d) 成本 e) F1 口径

---

## 1. PaSa (arXiv:2501.10120, ByteDance Seed / 北大, ACL 2025 Main)

**双 agent：Crawler（爬取）+ Selector（筛选），均微调自 7B 模型（SFT+PPO）。**

### a) 召回物理来源
- **搜索引擎**：Google Search API（serper.dev），带参数 `site:arxiv.org` + `before:query_date` —— 即"限定 arXiv 站内的 Google 网页搜索"，不是学术 API。
- **全文获取**：本地 paper_database（cs_paper_2nd.zip，预存 CS 论文库）；无记录时用 **ar5iv** 拉 HTML 全文（含引文列表），解析入库。
- 覆盖面 = Google 对 arxiv.org 的索引 ∩ 预置/可拉取的 arXiv 论文，**只覆盖 AI/CS 领域 arXiv 论文**。

### b) 查询→召回链路
Crawler 只有 3 个动作（Table 3）：
- `[Search]`：LLM 生成搜索词 → 调 Google → 结果全部进 paper queue
- `[Expand]`：LLM 指定论文章节名 → 该章节引文列表里的论文全部进 queue（**引文滚雪球**）
- `[Stop]`：重置上下文回 query + queue 中下一篇

链路：query → Crawler 多轮 [Search]/[Expand]（探索深度上限 3 层）→ paper queue → Selector 逐篇读 **title+abstract** 输出 True/False（决策 token 前置，RL 时兼作 reward model）。

### c) 迭代策略
- 迭代发生在 Crawler 内部：每读 queue 中一篇论文，决定 Expand 哪些章节（引文扩展）还是 Stop。
- **推理时无 Crawler↔Selector 循环**（Selector 只在最后一次性过滤）；但 RL 训练时 Selector 判断作 Crawler 的奖励。
- 深度控制靠 RL 学出（α 成本惩罚项），非规则。

### d) 成本
- 论文未直接报 API/LLM 调用次数与金额。最接近指标是 **crawler action 数**（Search+Expand 合计）：AutoScholarQuery 测试集上 PaSa-7b（α=1.5, c=0.1）**平均 382.4 actions/query**；α=2.0 时 785.5，c=0 时 1296.3 —— 单查询数百次工具调用+LLM 推理。
- 标注成本参考：RealScholarQuery 每条 query 人工审 76 篇候选 × $4 ≈ $304/query。

### e) F1/指标口径
- **RealScholarQuery**：50 条真实研究者 query，gold 由教授级标注者汇总 PaSa/Google/Google Scholar/ChatGPT/GPT-4o 候选池人工审定，平均 **15.82 gold/query**，截止日期 2024-10-01。
- 排序系统（Google 系）报 **Recall@20/50/100**（top-k 命中 gold 比例）；集合系统（PaSa/ChatGPT）报最终集合的 **precision / recall**（不报 F1）。另有 "Crawler recall"（gold 在 queue 中的比例）。
- PaSa-7b：RealScholarQuery 上 precision 0.5146 / recall 0.6111 / R@20 0.5798；比 Google+GPT-4o 的 R@20 高 37.78pt。
- AutoScholarQuery（35k 合成 query，AI 顶会论文）上 recall 0.4834 / precision 0.1448（**precision 很低**，靠 Selector 兜底前的 queue 噪声大）。

---

## 2. SPAR (arXiv:2507.15245)

**多 agent 框架：Query Understanding + Retrieval + Judgement + Query Evolver，核心机制 RefChain（引文链一层扩展）+ 查询演化。**

### a) 召回物理来源
- 固定源集合 **S = {Google, ArXiv, OpenAlex, Semantic Scholar, PubMed}**，按 query 领域/意图选择（如生物医学走 PubMed）。
- **无自建向量库、无 embedding 检索**，纯关键词+引文遍历。
- SPARBench 文档来自 arXiv/PubMed/OpenAlex/S2。

### b) 查询→召回链路
1. Query Understanding Agent：意图分类（综述/最新进展/方法对比）+ 领域识别 + 时间约束抽取，把宽泛 query 展开为结构化子查询列表 Q={q1..qN}（方法/应用/历史/挑战等不同视角）。
2. Retrieval Agent：对每个子查询做**源自适应**检索（S2/OpenAlex 用关键词，Google 用完整 query 字符串），去重合并。
3. Judgement Agent：LLM 对每篇打相关分（0-1），超阈值进 Related Pool。
4. **RefChain**：对 Related Pool 每篇提取参考文献（PDF 解析或元数据），Judgement Agent 再打分，高相关并入 pool。**只扩一层**（引文的引文不再递归）——保 precision 控成本。
5. 最相关 K 篇形成 Paper Cache。
6. Query Evolver Agent：对 Cache 每篇生成 3 条新查询（方法洞察/应用/局限视角），随机子集并入 Q。
7. 循环至 Paper Cache 达预设大小或最大深度。

Rerank 信号（Sec 3.3）：相关分 + 发表权威性（venue 声誉/作者声誉）+ 时间相关性。

### c) 迭代策略
- 双轮驱动：**引文扩展（RefChain，确定性单层）** + **查询演化（LLM 从已命中论文生成新查询）**，循环直到 Cache 满/达最大深度。
- 与 PaSa（RL 学深度控制）不同，SPAR 用**确定性固定深度**策略。

### d) 成本
- 论文无显式成本分析。间接指标：平均原始检索文档数 **569.1（AutoScholar）/ 504.9（SPARBench）**/query；每篇 Cache 论文衍生 3 条新查询。Judgement Agent 对所有候选打分 → LLM 调用量与候选数同量级（数百次/query）。

### e) F1 口径
- **文档级 P/R/F1**：P=TP/(TP+FP)，R=TP/(TP+FN)，F1=调和平均。关键设定：**每个检索系统在各自原生搜索源上评测**（非共享语料库），端到端评估。
- **AutoScholar**（PaSa 的合成基准）：GPT-4o 生成 query，仅 100 对人工复核。
- **SPARBench**：50 query（35 CS + 15 生物医学），~560 篇专家验证相关文档（平均 12/query）。构建：198K 原始候选 → Qwen2.5-7B 粗筛 3K → Qwen2.5-72B 细筛 2K → 研究生级专家审定 ~560。query 含语法错误/拼写错误（模拟真实用户）。
- 主结果：SPAR F1 **0.3843**（AutoScholar，P 0.3612/R 0.4105）、**0.3015**（SPARBench）vs PaSa 0.2449/0.1041。PaperFinder 在此口径下 F1 仅 0.0506/0.0418（因其只在自己 S2 语料上找，跨源 gold 覆盖不足——**注意这是 SPAR 的口径，对 PaperFinder 不公平**）。
- Ablation：RefChain 提 recall（0.41→0.44）但降 precision（0.29→0.19）；rerank 提 Recall@5（0.3146→0.4015）。

---

## 3. Ai2 Paper Finder / asta-bench (arXiv:2510.21652, Allen AI)

**不是自由 agent，而是"手工编码 pipeline + 关键节点 LLM 决策"的工作流编排**（代码名 mabool）。冻结版开源：github.com/allenai/asta-paper-finder。

### a) 召回物理来源
- **Semantic Scholar 生态为主**（需 S2_API_KEY）：
  - **S2 snippet search**（Vespa 检索后端，`/snippet/search` 端点，单次上限 400 条）：~500 词片段级语义检索，覆盖 title/abstract/正文（排除图表标题和参考文献），返回带 refMentions 引文位置标注；
  - S2 relevance search（关键词）、title 精确匹配、citations/references 遍历；
  - **Cohere embedding**（COHERE_API_KEY，用于向量/语义信号）；
  - Google API（GOOGLE_API_KEY）兜底。
- 论文 ID 体系统一为 S2 corpus_id。覆盖面 = S2 全库（约 2 亿+ 篇，全学科）。
- asta-bench 侧另暴露 **Asta MCP 工具**（asta-tools.allen.ai，8 个工具：get_papers / get_citations / search_papers_by_relevance / search_paper_by_title / snippet_search / search_authors_by_name / get_author_papers 等），支持 venue/year/inserted_before 过滤，供被评 agent 统一调用。

### b) 查询→召回链路（论文附录 F.3 + 源码双重确认）
1. **Query Analyzer**：LLM 把 query 解析为结构对象（多个短 prompt 并行，各抽取 1-3 个属性，比单长 prompt 低延迟）：query_type（navigational / semantic / metadata / author）+ 语义准则 + **子准则分解（带重要度权重，供 relevance judgement 逐条判）** + 时间/venue/作者约束 + recency/centrality 修饰词（"classic"/"recent"/"early works"）。
2. **Planner 路由**到工作流：BroadSearch（语义主力）/ SpecificPaperByTitle/Name / SearchByAuthors / MetadataOnly。
3. **Semantic 查询主流程（多轮扩大）**：
   - **Initial search**：LLM 对语义准则生成 k 个改写，k+1 条查询全部调 **S2 snippet search（Vespa 语义检索，单次 ≤400 条）**；
   - **Snippet→paper 聚合**：同一论文的片段按文中出现顺序合并；**片段中提到的被引论文（"Doe et al 2023 show that..."）也关联为候选**——证据可来自论文自身或引用它的论文；
   - **Relevance judgement**：LLM 逐篇判 4 档（perfectly / highly / somewhat / not relevant），逐条对照子准则；
   - **Citation tracking（snowball）**：对前两档论文做前向（被引）+后向（参考文献）滚雪球，新结果再过 relevance judgement；
   - **Followup queries**：**选与 query 在 embedding 空间距离最远的相关论文**（当前查询覆盖边界）作示例，LLM 改写新查询，重复 initial search 流程；
   - **Short-circuit**：高相关论文数够多或 relevance judgement 次数超限即停；预设最大轮数。
4. **Metadata 查询**：LLM 把约束解析成结构化 work-plan → 手工编码 executor 翻译成 API 调用序列（支持嵌套/否定，如"不引用 transformers 的 ACL 2024 论文"）。
5. **Navigational 查询**：三路并行——S2 title API / LLM 直接回忆+title API 锚定 / 关键词→句子搜索→句内引文→高频被引候选。
6. **最终排序**：启发式 = 相关分 + 引用数 + 发表时间 + query 表达的偏好。
- 设计哲学（论文脚注30）：**手工编码 pipeline + 关键点 LLM 决策，优于给 LLM 更多自主权写代码控制流程**——LLM 调用数/token 更少、可并行、更可靠。
- 运行模式：fast（~30 秒）/ diligent（~3 分钟，更穷尽）。

### c) 迭代策略
- 语义查询为**预设最大轮数的"检索→判断→扩大"循环**：每轮检索步比上轮范围更广且以上轮相关文档为信息（snowball + followup query）。
- **多臂老虎机（mabwiser BatchedThompsonSampling）**：不是选检索源，而是**在多个检索 origin（各查询变体）之间分配 LLM relevance judgement 预算**——各 origin 的文档排队，MAB 按"该 origin 已判文档的相关性 reward"决定下一个 batch 判哪个 origin 的文档，配额内最大化找到相关文档的效率。每个 origin 先 uniform preload 一小批作 MAB 初始拟合。
- **Short-circuit**：高相关累计分达 cap 即跳过后续源/提前终止循环。

### d) 成本（asta-bench 论文实测，USD/query）
- **Asta Paper Finder（gemini-2-flash + gpt-4o）：PaperFindingBench 得分 39.7±3.1，成本 $0.063/query**。
- 对比：ReAct (gpt-5) 26.4±3.9 分、$0.428/query；ReAct (o3) 19.3、$0.518；Smolagents Coder (claude-sonnet-4) 22.1、$0.975；You.com Search API 仅 7.2。**PaperFinder 比最强 ReAct 便宜 ~7 倍且高 13 分**——手工 pipeline 的效率优势。
- LitQA2-FullText-Search（recall@30）：PaperFinder 90.7±6.6、$0.112/query。
- 运行时长：fast ~30s / diligent ~3min。

### e) F1 口径（asta-bench paper_finder 任务，论文 E.2 + 代码级确认）
数据集 **PaperFindingBench**：test 267 条 + val 66 条，含 48 navigational + 43 metadata + 242 semantic query；query 来自 PaperFinder/OpenSciLM 用户日志 + LitSearch + PaSa 数据集（专挑系统失败过的难例）；语料截止 2025-06-01。
- **navigational / metadata（gold 完备）**：**standard F1** over result set。metadata 的 gold = 用 S2 API 完整求解该 query 的 python 代码的输出。
- **semantic（gold 天然不完备）**：**adjusted F1** = 调和平均(估计 recall@估计集大小, 下界修正 nDCG)：
  - 相关性由 **LLM judge** 对 agent 提交的 markdown_evidence 逐条对照子准则判定（不给全文，只给 agent 摘的证据——评检索不评长上下文能力）；
  - **估计集大小** = 多个宽松阈值 PaperFinder 变体结果取并集 × 2-10 倍乘数（初始集越小乘数越大）——显式承认 gold 不完备；
  - 用 nDCG 而非 precision 作平衡项（排序质量比集合精度信息量更大）；
- 总分 = 4 类 query 的 per-query 分数平均。agent 输出上限 **250 篇**、须降序 + 每篇附 verbatim evidence、评测带 `inserted_before` 时间冻结。
- 关键设计：**宽泛语义查询不用封闭集合匹配**，用"估计分母 + LLM judge + nDCG"——与 PaSa/SPAR 口径最大的差异。

---

## 横向对比（召回架构视角）

| 维度 | PaSa | SPAR | Ai2 Paper Finder |
|---|---|---|---|
| 检索源 | Google(site:arxiv.org)+ar5iv | Google/ArXiv/OpenAlex/S2/PubMed | S2 snippet(Vespa 语义)+关键词+引文+LLM回忆 |
| 语义/向量 | 无（网页搜索） | 无（关键词） | 有（snippet 语义检索+Cohere） |
| 引文扩展 | Expand 章节（深度≤3，RL 控制） | RefChain 固定 1 层 | snowball 前向+后向+片段内被引关联 |
| 查询演化 | Crawler 自主生成搜索词 | Query Evolver 每篇 3 新查询 | k 个改写+基于 embedding 边界论文的 reformulate |
| 迭代控制 | RL 学出 | 确定性规则 | 预设轮数 + MAB 分配判断预算 + shortcircuit |
| 相关性判断 | Selector 微调 7B（title+abs） | Judgement Agent（LLM 0-1 分） | LLM 4 档（逐子准则）+ MAB 预算分配 |
| 单查询成本 | ~382 crawler actions | ~500+ 文档候选全判 | **$0.063 / 30s（fast）**（vs ReAct gpt-5 $0.428） |
| 代表得分 | RealScholarQuery recall 0.611 | SPARBench F1 0.30 | PaperFindingBench 39.7（adjusted F1×100） |
| F1 口径 | P/R（集合），R@k（排序） | 文档级 P/R/F1，各系统用自己源 | standard F1（封闭gold）+ adjusted F1（LLM judge+估计分母+nDCG） |
| Gold | 人工审定 15.8篇/query | 专家验证 12篇/query | 不完备 gold+归一化估计/LLM judge |

## 对华为赛题三的可借鉴结论

1. **检索源组合**：三系统都非单一源。PaSa 单源但靠引文深挖；SPAR 五源按域路由；APF 用 S2 统一 ID 体系 + snippet 语义检索为主干。赛题若给开放 API 层，"多源+统一 ID 归一"是标配。
2. **召回三板斧**（三系统收敛于同一组合）：查询分解/改写（多视角子查询）+ 引文滚雪球（前向/后向，深度 1-3 层）+ LLM 相关性判断过滤。向量检索只有 APF 有（S2 snippet search 现成可用，不必自建）。
3. **成本控制是分水岭**：PaSa 382 actions/query vs APF **$0.063/query（30 秒）**。APF 的三件套——MAB（判断预算在查询变体间分配）+ shortcircuit（高相关够用即停）+ 手工 pipeline 替代自主 agent（论文实测比 ReAct gpt-5 便宜 7 倍且高 13 分）——是工程上最精细的方案；PaSa 靠 RL 学成本敏感策略，训练成本高但推理架构简单。
4. **F1 口径必须先问 gold 完备性**：封闭 gold（specific/metadata）用标准 F1；宽泛语义查询 gold 天然不完备，APF 用"估计分母 + LLM judge + ndcg 排序分"的 adjusted F1 是目前最成熟的处理。赛题评测若含宽泛查询，建议直接借鉴该口径设计。
5. **超图增强位**（结合本项目）：三系统的 rerank 都只用相关分+权威性+时间；引文网络扩展都是盲扩展（全进 queue 再过滤）。超图/schema 可做两件事：a) 引文扩展的**选择性导航**（按方法演化边选 Expand 目标，替代 PaSa 的 RL）；b) 跨文献关联推理（SPAR 没做、赛题明确要求的部分）。

## 来源
- PaSa: arXiv:2501.10120 (HTML 全文) + github.com/bytedance/pasa README
- SPAR: arXiv:2507.15245 (HTML 全文)
- Ai2 Paper Finder: github.com/allenai/asta-paper-finder (源码级: paper_finder_agent.py / broad_search.py / snowball_agent.py / dense/formulation.py / relevance_loading_optimization.py / vespa.py / adaptive.py) + allenai.org/asta/resources/mcp
- asta-bench: github.com/allenai/asta-bench (源码级: astabench/evals/paper_finder/{task.py, eval.py}) + arXiv:2510.21652 (ICLR 2026, PDF 全文抽取: 附录 E.2 评测口径 / F.3 agent 架构 / Table 5 主结果；本地缓存 .research_tmp/contest_survey/astabench.txt)
