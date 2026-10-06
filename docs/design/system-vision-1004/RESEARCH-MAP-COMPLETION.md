# 领域图谱补全（map completion / growth）文献调研

日期：2026-10-04。范围：截至 2026-10 的工作。

## 引导段

本调研回答一个问题：一个持久的领域知识库（论文 → 结构化记录 → 方法族与演化关系的领域图）要靠外部检索持续发现"图里缺什么"并补进来，这件事前人做到了哪一步。结论先行：

1. 在本次检索并逐条一手核实的范围内，没有找到同时满足以下四点的系统：持久的领域图；由图上检测到的缺口驱动外部定向检索；检索结果写回图；估计"还缺多少"。四点各自都有先例，但分散在不同系统里。
2. 最接近的先例有六个，各占一角。SciAtlas 做了"coverage-guided multi-round retrieval"：诊断缺失的时间窗和方法分支，再定向追问，但只在内部图谱里检索，不写回，也没有消融。Agents-K1 的 O5 算子能检出类型化的结构空洞，但用途是 ideation，不驱动获取。HERO 区分三类缺口，并用产出率早停，但三类缺口共用同一个动作，也只在单次查询内有效。Undermind 用发现曲线饱和来估计某个主题的相关论文总数，同样只在单次查询内。Re:CAP 会探测"可能缺失的主题"，但不面向学术图谱。LKM 能逐篇增量增长并保持对象身份，但增长方式是全量批量摄入。
3. 覆盖率估计在封闭候选池的 TAR（technology-assisted review）里已经成熟，包括 knee/target、quantile、超几何检验、counting process 和 SAFE。开放世界里只有三类做法：系统综述的 capture–recapture、Undermind 的发现曲线、Wikidata 类完整度估计。没有人把估计做到"领域图的每个族或每个区域"这一粒度，也没有人让估计在跨任务累积的检索历史上持续更新。
4. 没有找到把缺口类型（缺已知族成员、缺前沿新作、缺未覆盖区域、缺已知论文的某个方面）分别映射到不同获取动作、再用 value-of-information 排序的工作。
5. 评测有已知的坑：人类引用列表不能当 ground truth（Sahu et al. 2026），自动构建的 gold 也承认会漏（ScholarQuest）。可借用的协议有时间外推验证（CKM、RollingEval）和"探测并判新"（Re:CAP）。

## 0. 方法与核实说明

- `web-access` skill 在本环境不可用（Skill 调用返回 Unknown skill），所以发现层改用 WebSearch。结果视为可能被污染，只当线索。
- 验证层：表中每一项都用 WebFetch 打开过一手页面，包括 arXiv abs/HTML、ACL Anthology、PMC、DBLP 镜像、官方 blog/repo/白皮书 PDF。标 ★ 的条目读过摘要以外的全文（arXiv HTML 或 PDF），重点核查三件事：是否持久、检索是否由结构引导、有无覆盖估计。
- 注意：全文核查是经 WebFetch 的摘要模型完成的，关键处要求它给出原文引句。PDF（HERO、Undermind、West 2014）是直接看页面图像核对的。个别数值标了"待复核"。
- 未能一手核实的条目放在第 6 节，不进主表。

## 1. 判决表

列含义：持久＝是否跨查询/任务保留并增长自己的库；引导＝靠什么决定下一步检索；估计/停止＝有无覆盖估计或停止规则；缺什么＝相对"持久图 + 缺口驱动获取 + 写回 + 覆盖估计"这一目标所缺的部分。

### 1A. 学术检索 agent 与 deep research

| 工作 | 一手 ID | 持久？ | 什么引导检索 | 覆盖估计/停止 | 如何评测 | 缺什么 |
|---|---|---|---|---|---|---|
| ★PaSa | arXiv 2501.10120（ACL 2025） | 否。状态＝当前 LLM context + 单 query 的 paper queue | RL 训练的 Crawler 三动作：[Search]，[Expand] 某 subsection 的全部引文，[Stop]；Selector 判相关并兼作 reward model | 无估计；探索深度上限 3 | AutoScholarQuery（33,551/1,000/1,000）、RealScholarQuery（50 题，平均 15.82 篇答案）；Recall@20/50 | 无跨 query 状态；扩展粒度是 section 级引文；不知道漏了多少 |
| ★SPAR | arXiv 2507.15245 | 否 | 多 agent：意图/领域/时间理解与选源 → 多源检索 → Judgement → Query Evolver（每篇论文生成方法/应用/局限三个新 query）→ 单层 RefChain | 无估计；"until the Paper Cache reaches a predefined size or maximum depth" | SPARBench（50 题/560 篇专家确认）；F1：AutoScholar 0.384 vs PaSa 0.245 | RefChain 只走一层；无持久；无覆盖 |
| PaperScout | arXiv 2601.10029 | 摘要未述 | RL（PSPO）按累积上下文决定何时、如何调 search/expand | 摘要未述 | 合成与真实基准；在 ScholarQuest 上 R@100 0.314，三个 agent 中最高 | 同 PaSa 类：单查询 |
| ★PaSaMaster | arXiv 2605.14306 | 否。只在单会话内迭代 strategy/checklist；后端 1.6 亿论文三层库是静态基础设施 | Navigator/Reflect 根据排序证据识别 "missing coverage, ambiguous constraints, or under-explored directions"，更新下一轮策略；三通道：semantic direct retrieval、citation network expansion、web-to-repository verification | 全文无显式终止规则、预算或饱和检测 | PaSaMaster-Bench（244 任务，38 学科）；F1@20 23.00 vs GPT-5.2 16.69；幻觉降为 0 | 缺口类别只是举例，不映射到不同通道；无停止；无跨任务 |
| ★Crase | arXiv 2608.24809 | 否 | 搜一次拿种子 → 1.5-hop 引文邻域（refs + citing + 邻居间的边）→ claim 级 entailment 剪边 → recency-aware random walk 排序 | 结构边界即停止（固定），无估计 | LitSearch + 500K arXiv；ICLR R@50 0.366 vs 专有 DR agent 0.122；成本约 1/3 | 一次性；覆盖受种子邻域上限约束 |
| PaperPilot（Workflow Induction） | arXiv 2607.00597 | 否。工作流按会话精化 | 可编辑 DAG 算子（keyword search / citation expansion / filter / score）+ 用户反馈 | 无 | anchor-paper 查询 Hit@5 58.0→77.0 | 无跨任务、无覆盖 |
| Ai2 Paper Finder | allenai.org/blog/paper-finder；github.com/allenai/asta-paper-finder | 仅结果文件缓存 | query → 结构化对象 → 按意图路由；对最相关论文做 forward + backward citation tracking；语义准则拆成子准则逐条做 LLM 判定 | 启发式："stops when it either finds enough papers or scans too many candidates"；fast/diligent 模式 | LitSearch（博客口径为"完全/高度相关"比例，非标准 recall@k） | 无覆盖估计；无持久 |
| ★Undermind | 白皮书 undermind.ai/whitepaper.pdf（Hartke & Ramette, 2024-01） | 否 | 迭代：初检 → LLM 三档相关度分类 → 按已发现内容调整再搜 | 有，而且是检索 agent 中唯一的：跟踪相关论文发现率，用指数饱和拟合判断"收敛"，并预测该主题相关论文总数；另用与 Google Scholar 结果重叠的比例（Lincoln–Petersen 逻辑）估计 exhaustiveness | 约 100 个用户查询对比 Google Scholar；分类器与人工标注一致性表 | 估计只在单查询内；厂商自评；隐含两通道独立假设；无持久 |
| ★HERO | ACL Anthology 2026.nslp-1.19（NSLP@LREC 2026） | 否 | 每个 subquery 一条管线；facility-location submodular 选多样 query；Enrichment Agent 读中间综述，找三类缺口：unexplored aspects / insufficient evidence（只有单一来源）/ unresolved contradictions，再生成追问（同样用 submodular 选） | 有启发式早停：一个 batch 中少于 10% 的文档带来新引用即停（允许再多一个低产 batch） | ScholarQABench + DeepResearchGym；DRGym KPR 67.63、Citation F1 91.57 | 三类缺口共用同一动作（追加查询）；无持久；不估总量 |
| ★Sahu et al. "Rethinking Literature Search Evaluation" | arXiv 2605.29234 | 仅 query/result hash 缓存 | LLM 拟关键词 → arXiv/OpenAlex/S2 → 沿种子参考文献 BFS（即 backward snowballing），深度 ≤3、总预算 N_max → embedding/LLM debate 重排 | 深度/预算上限，无估计 | RollingEval-Jun25（250 篇 CS）；召回从 <20% 升到 >80%；人类引用仅 51% 被判中等以上相关，AI 重排为 86–88% | 不取被引方向；无覆盖估计；关键贡献是证明人类引用列表不能当 gold |
| ★ScholarQuest | arXiv 2606.20235 | （benchmark） | — | — | 1000+ CS 主题、四类意图；ScholarBase（S2 快照，BM25/BGE-M3/RRF）；最佳 R@100 0.314；主要失败模式为 off-target exploration；PaperScout 用 19.0 次 expand 胜过 PaSa 的 55.1 次 | gold 自动构建，作者自认 "may still miss some relevant papers" |
| RAAC | arXiv 2608.15191 | 否 | 用 search novelty 和 information coverage 两个无监督信号选动作 | 有：检出停滞后停止 | BrowseComp-Plus，平均少 14 次搜索 | 非学术场景；不估总量 |
| Re:CAP | arXiv 2609.24122 | 周期审计环 | 先识别已覆盖主题 → 生成"可能缺失主题"的问题 → 检索 → LLM 判是否带来新信息 | 审计"缺什么"，不估总量 | 恢复 9–29% 的未标注 gold（TREC-COVID 48%）；人工确认 78.9% 为新信息（κ=0.79） | 面向通用 RAG 语料，不面向结构化领域图；缺口无类型 |
| Elicit | lilly.elicit.com/blog/how-we-evaluated-elicit-systematic-review | 用户 PDF 集合 + Elicit 检索补充 | 检索机制未公开 | 未公开 | 58 篇已发表系统综述上，筛选 recall 93.6%、specificity 62.8% | 搜索端完整性不可核 |

### 1B. 持久的学术知识库与领域图（问题 3 的核心）

| 工作 | 一手 ID | 持久？ | 什么引导检索/增长 | 覆盖估计/停止 | 如何评测 | 缺什么 |
|---|---|---|---|---|---|---|
| ★LKM v2 | arXiv 2609.27297 | 是。逐篇增量，对象身份保持（"Existing bindings are never rewritten"） | 增长＝批量摄入 arXiv/bioRxiv/ChemRxiv 全部预印本 + 多学科期刊；检索返回 claim + 推理链 + 来源 | 无缺口检测；100 篇审计只测了幻觉率 0.97%，明确没测遗漏 | SciFact-Open 证据召回 72.71% vs 39.38%；ScholarQA-CS citation F1；ChemBench/PubMedQA/SciBench +9.3–14.7 | 无定向获取；agent 写回列为未来工作（"The decisive step is to close the loop"） |
| ★ScholarStack v2 | arXiv 2609.23735 | 是（版本化、带溯源的三层资产），但 collection 固定 | task-specific views；open-world 检索时用 view 构造外部搜索 | 无；原文："Missing evidence indicates limited coverage of the available view; it does not establish that no relevant work exists" | 多论文 QA 18.41 vs 全文 8.46；QASPER 用 34% token 达到全文 F1 的约 93% | 外部新发现的论文不登记入库；无增长机制；增益集中在 gold 与编译集合重叠的查询上 |
| ★Agents-K1 | arXiv 2606.13669 | 是。Scholar-KG 覆盖 2.46M 论文、6 学科（批量构建） | GraphAnything 三源（web / multimodal graph / cross-doc traversal）按意图加权；O5 结构空洞算子检出 "orphan methods… singleton datasets… papers disconnected from the main component… sparse cells"；O6 新颖性评分 | 无 | FrontierScience-Research（Gemini-3 7.9→24.6%）、多跳 QA；4B IE 模型 | O5 服务 ideation，不驱动获取；web 结果不自动写回（`graphanything evolve` 只把已答问题折回为 Question 节点） |
| ★SciAtlas | arXiv 2605.22878 | 是。以 OpenAlex 为主，43M 论文、157M 实体、3B 三元组；"updated periodically rather than continuously" | 轨迹重建工作流做 coverage-guided multi-round retrieval：在内部图谱的 citation/concept/discipline/community 邻域里检索 → 诊断缺失的时间窗与方法分支 → 高优先缺口触发定向追问 | 有缺口诊断和停止条件（"until no blocking gap remains or the round limit is reached"；轮次上限值未给）；无总量估计 | latent-relation recovery nDCG@20 近 3 倍；SurveyLens 比对重建轨迹与人写综述 | 是最接近的先例。但缺口只服务单次任务，检索只在内部图谱，不写回，没有单独消融；覆盖受底层记录限制 |
| ★Mechanist | arXiv 2608.12036 | 是。13k 篇 AI 机制研究的 KG（从 OpenAlex 语料按主题选出，另加 Anthropic/Goodfire 博客） | 在 SciAtlas schema 上扩展三套分类：object of study / application scenario / mechanism methods | 无 | 100 条人工抽检准确率 90%+；假设质量对比 Claude Code 等 | 一次性主题选取；无增长或定向补全机制 |
| ★ASKS | arXiv 2608.29612 | 是。每个源对应一个 GraphDelta 状态转移 | 冻结的时间序 manifest（65 个候选取 56 篇） | coverage Q_map=0.893、branch survival、churn 都是事后结构指标，不驱动阅读 | 56 篇张量网络论文；churn 均值 0.00467 | 无获取；单一研究项目，无对比方法 |
| ★Lacuna | arXiv 2606.26246 | 批量构建（733,795 篇论文，15.26M concept elements，27,017 directions） | HDBSCAN 把 concept 聚成 direction 页；proposal 用 "Alien Science" 采样 | 无 | LitSearch R@10 0.538 vs OpenScholar 0.424；ReportBench-ML | 原文："a batch-built map, not an online updating system… incremental updates are future work" |
| ★DAS（Deep Academic Survey） | arXiv 2608.18034 | 部分。DAS-2M 元数据湖（约 2M 篇 arXiv，2020-01 至 2026-06）持久并随新论文更新；单主题的 literature/organization/writing 状态每次新建 | candidate-grounded taxonomy planning + reverse paper-to-section routing（把论文路由到 taxonomy 节点） | 无。候选集一次定死，没有"节点欠覆盖 → 追检" | DAS-Bench 30 主题 16 准则，4.34 vs 4.03 | 无缺口触发检索；无引用召回指标 |
| ★StructSurvey | arXiv 2607.01243；ACL Anthology 2026.surgellm-1.10 | 否。全局领域图每篇 survey 从空开始（"G ← ∅"） | 从摘要抽实体/关系/3–8 类 taxonomy，由 planner 路由 query 函数 | 无；检索数量固定 | 33 篇 ACL 综述；ROUGE-1 recall +2.86pt | 评测限定在 gold 综述的引文池内；无缺口检测 |
| SurveyGen-I | ACL Anthology 2025.ijcnlp-long.193 | 单次写作 memory | 检测到上下文不足时触发 subsection 级细粒度检索 | 无总量估计 | 6 个领域的 content quality / citation coverage | 缺口＝写作上下文不足，单次有效 |
| CKM（Continuous Knowledge Metabolism） | arXiv 2604.12243 | 是。滑动窗口演化的 knowledge state | 按时间被动流入文献 | — | 用后来发表的论文验证生成的假设：72% 主题至少一条被验证 vs 一次性基线 30%，平均提前 404 天 | 被动摄入；可借用的是"时间外推验证"这一评测范式 |

### 1C. KG 构建/补全中的检索与获取

| 工作 | 一手 ID | 持久？ | 什么引导检索 | 覆盖估计/停止 | 如何评测 | 缺什么 |
|---|---|---|---|---|---|---|
| ★West et al. "KB completion via search-based QA" | DBLP conf/www/WestGMSGL14（WWW 2014）；Stanford PDF | 是（Freebase） | 对缺值的 (entity, relation)，学习哪些 Web 搜索 query 模板最可能返回正确值，并聚合多 query 的答案 | — | Freebase 人物属性（出生地等大面积缺失） | 粒度是已知 schema 下的空槽；不涉及"缺哪些实体/论文" |
| Narasimhan et al. | arXiv 1603.07954（EMNLP 2016） | 否 | DQN 决定是否发 query、发哪个、何时停；reward＝抽取准确率增益减去成本 | 有：学习式、cost-aware 的停止 | 枪击事件、食品掺假两个数据集 | 单文档事件槽位；是 VoI 式获取的早期 RL 版本 |
| AgREE | arXiv 2508.04118 | 补全 KG | 针对新兴实体迭代做 web 检索 + 多步推理，生成三元组 | 未述 | 新兴实体 KGC，比有监督方法高最多 13.7% | 实体级、逐个补；不估覆盖 |
| KARMA | arXiv 2502.06472（NeurIPS 2025） | 是（富化已有 KG） | 9 个 agent 处理给定的 1,200 篇 PubMed 文章 | — | 38,230 个新实体，LLM 验证正确率 83.1% | 文档集给定，没有定向获取 |
| ★RAGA | arXiv 2605.17072 | 是。Neo4j KG 随文档增量增长 | Read-Search-Verify-Construct；缺口只在给定文档集内补（search_kg / browse_context / create_todo） | 无 | QASPER Answer F1 0.615 | 不访问外部源 |
| SENATOR | arXiv 2505.07184（NeurIPS 2025） | — | 在 KG 路径上用 structural entropy 量化模型的不确定性，MCTS 探索知识缺陷区域，再合成数据做 SFT | — | LLaMA-3/Qwen2 领域基准 | 修补对象是模型参数，不是知识库；"用结构熵定位缺口区域"的思路可借鉴 |
| TaxoAdapt | arXiv 2506.10737（ACL 2025） | — | 按语料的主题分布迭代扩展 taxonomy 的宽度和深度 | — | LLM 评 granularity/coherence | 语料固定，不去获取新论文 |
| Luggen et al. | arXiv 1909.01109 | — | — | 有：用非参数 species-richness / capture–recapture 类估计器估计 Wikidata 各类的完整度；估计量是否收敛可以指示类是否接近真实规模；插入突发会导致高估 | Wikidata 若干类 | 计数对象是实体；但它是与"每个方法族还缺多少篇"最接近的先例 |
| Razniewski et al.（综述） | arXiv 2305.05403 | — | — | 综述 partial closed-world、recall 的统计估计、负陈述、relative recall | — | 背景框架 |

### 1D. 覆盖估计与停止规则（系统综述、TAR）

| 工作 | 一手 ID | 持久？ | 什么引导检索 | 覆盖估计/停止 | 如何评测 | 缺什么 |
|---|---|---|---|---|---|---|
| Spoor et al. 1996 | BMJ, doi:10.1136/bmj.313.7053.342（White Rose 元数据页） | — | 多数据库检索 | 用 capture–recapture 评估系统检索的完整度（只核到元数据，未读全文） | — | — |
| ★Kastner et al. 2007 | PMC2655834（AMIA） | — | 4 个数据库 | 用 CMR + Poisson 回归估计文献"horizon"：找到 1,246 篇，估计总量 1,838（95% CI 1,749–1,955），即约 68%；查到第 3 个库后估计趋稳 | 回顾性单个综述 | 整综述只有一个标量；假设数据库之间独立 |
| Cormack & Grossman 2016 | doi:10.1145/2911451.2911510（SIGIR；元数据经 ir.webis.de 核实） | — | CAL 主动学习排序 | knee method / target method（方法细节来自已有知识，ACM 页 403 未读原文） | TREC 等 | 封闭候选池 |
| Yang, Lewis, Frieder | arXiv 2106.09871（DocEng 2021） | — | TAR | 借鉴调查抽样估计的 Quant / QuantCI | 多个召回目标与任务 | 封闭候选池 |
| Lewis, Yang, Frieder | arXiv 2108.12746（CIKM 2021） | — | one-phase TAR | 用 quantile 估计给出有统计保证的停止规则；指出超额召回会带来超额成本 | — | 封闭候选池 |
| ★Callaghan & Müller-Hansen 2020 | PMC7700715（Systematic Reviews） | — | ML 优先筛选 | 对未筛文档随机抽样，做超几何假设检验，拒绝"召回 < 目标"即停；启发式规则约 39% 的情况下达不到目标 | 多个综述数据集 | 封闭候选池 |
| Bin-Hezam & Stevenson | ACL Anthology 2023.findings-emnlp.171 | — | TAR | counting process 加分类器信号 | CLEF e-Health、TREC Total Recall/Legal、RCV1 | 封闭候选池 |
| ★SAFE | doi:10.1186/s13643-024-02502-7（PMC10908130） | — | 主动学习筛选 | 四阶段启发式：key papers 全部命中、筛过估计相关数的 2 倍以上、筛过 ≥10%、连续 50 篇无相关；之后换模型复筛并质检 | 无直接对比验证（作者自认） | 封闭候选池 |
| Fletcher & Stevenson | arXiv 2606.15380 | — | TAR | 结论（而非召回）稳定即停 | CLEF DTA 综述 | 封闭候选池；但"信息需求满足即停"的视角可迁移 |

### 1E. "这组论文缺什么"的经典任务：引文推荐与参考文献补全

| 工作 | 一手 ID | 持久？ | 什么引导检索 | 覆盖估计/停止 | 如何评测 | 缺什么 |
|---|---|---|---|---|---|---|
| Bhagavatula et al. | arXiv 1802.08301（NAACL 2018） | — | 嵌入 → kNN 候选 → 判别式重排（global CR） | — | OpenCorpus 7M、PubMed、DBLP；F1@20 相对提升 18% 以上 | — |
| SPECTER | arXiv 2004.07180（ACL 2020） | — | 引文信号预训练的文档嵌入 | — | SciDocs（含 citation prediction） | — |
| SciNCL | arXiv 2202.06671（EMNLP 2022） | — | 在引文图嵌入上做近邻对比采样 | — | SciDocs SOTA | — |
| Nogueira et al. | arXiv 2001.08687 | — | BM25 + BERT 重排 + navigation-based candidate expansion（把候选的引文图邻居加进来） | — | 三个数据集上的最好结果 | — |
| Medić & Šnajder | ACL Anthology 2022.sdp-1.3 | — | — | — | MDCR 多领域大候选池；"BM25 is still very competitive" | — |
| Sjögårde & Ahlgren | arXiv 2403.09295 | — | 从种子出发的 direct citation / bibliographic coupling / co-citation / PubMed RA | — | 以系统综述为 gold；co-citation 是最好的单项方法；三种引文法组合更好；再加文本还能提高 | 强而便宜的结构基线 |
| ★CiteME | arXiv 2407.12861 | — | CiteAgent（GPT-4o，可搜可读） | — | LM 4.2–18.5%，CiteAgent 35.3%，人 69.7% | 单条引文归因，不是集合补全 |
| RMC / CitationR | ACL Anthology 2024.lrec-main.1196（LREC-COLING 2024） | — | RMCNet 用 Attentive Reference Encoder 编码论文已有的引用 | — | 专家标注的"审稿人指出的漏引" | 与"给定集合找缺失"同构，规模小 |
| Lin et al. | arXiv 2210.10073（Scientometrics 2022） | — | 科学实体与引文在同句中的共现 | — | 12,278 篇 CS 论文，475 个漏引实体，原论文中位发表于 8 年前 | 缺失＝漏引实体出处 |
| CiteRAG | arXiv 2601.14949（WWW 2026） | — | 多级混合 RAG + 对比学习微调的嵌入 | — | 引用列表预测 7,267 例 + 位置级 8,541 例；554k 论文语料 | 引用列表生成形态 |
| MasterSet | arXiv 2604.17680（SDM 2026） | — | 只用 title+abstract | — | 15 个 venue、150k+ 候选；must-cite 三层标注；Recall@K | — |
| LitSearch | arXiv 2407.18940（EMNLP 2024） | — | — | — | 597 题；dense 检索比 BM25 R@5 高 24.8pt | 单题检索 |
| LitLLMs | arXiv 2412.15249 | — | LLM 抽关键词 → 外部 KB → 带归因重排 | — | 归一化召回比朴素检索翻倍 | — |
| Färber & Jatowt（综述） | arXiv 2002.06961（IJDL 2020） | — | global vs local CR 分类 | — | — | 背景 |

### 1F. Value-of-information、agent 停止与记忆（问题 5 相关）

| 工作 | 一手 ID | 持久？ | 什么引导检索 | 覆盖估计/停止 | 如何评测 | 缺什么 |
|---|---|---|---|---|---|---|
| BED-LLM | arXiv 2508.21184（ICLR 2026） | — | 用 expected information gain 选下一个问题或查询 | — | 20 Questions、偏好推断 | 未用于文献/领域图 |
| CGDP | arXiv 2605.07042 | — | 把 context gathering 建模为 POMDP；predicate-based belief state | programmatic exhaustion gate | 三个 QA 领域，多跳 +11.4%，最多省 39% token | 非学术 |
| TASR / CoVeR | arXiv 2606.13814 / 2609.26086 | — | — | 分别用 logit margin 和 embedding coverage margin 决定何时停或何时调 verifier | 多跳 QA | 单题证据充分性，不是领域覆盖 |
| GAPMAP | arXiv 2510.25055 | — | — | — | 约 1,500 篇生物医学文档；区分 explicit/implicit 研究空白 | "研究空白"指科学上未知，不是库的覆盖缺口 |
| PARNESS | arXiv 2605.05258（摘要级） | 是。跨 run 的 KG 索引 | scenario-typed retrieval：similar / contradictory / cross-domain / counter-intuitive | — | 端到端生成论文示例 | 未述按缺口获取 |
| AutoSci | arXiv 2605.31468（摘要级） | 是。Long-Term Knowledge Memory + SciEvolve 版本化更新 | — | — | 摘要无数值 | 未述文献缺口驱动获取 |
| Fok, Siu, Weld | arXiv 2502.00881（CHI 2025） | — | — | — | 访谈 11 位综述作者 | 结论："continuous updating remains unmanageable" |
| Riaz et al. | PMC11975841（Mayo Clin Proc Digit Health 2024） | living 框架 | Watcher 模块持续监视新研究 | — | 综述 | LLM 检索牺牲召回、漏同义词 |

## 2. 逐题综合

### Q1 结构引导检索与跨查询持久状态

检索 agent 用的"结构"几乎都是引文图的局部扩展：PaSa 按 section 扩展、SPAR 走一层 RefChain、Crase 走 1.5-hop、Paper Finder 做双向 citation tracking、Sahu 做深度 3 的 BFS。另有两类结构信号：意图/checklist（PaSaMaster），以及子准则分解（Paper Finder）。survey 系统用的是即时构建的 taxonomy：DAS、StructSurvey、SurveyGen-I。

跨查询持久：所有检索 agent 的状态都限于单次查询，最多加一个结果缓存（Paper Finder、Sahu）。PaSaMaster 明确只在会话内演化。持久的只有底层语料库或元数据湖（PaSaMaster 的 1.6 亿论文库、DAS-2M），它们是被动增长的基础设施，不会因为某次查询发现了缺口而改变。

ScholarQuest 的失败分析显示，主要问题是 off-target exploration：读了很多候选，却没有一篇命中 gold。PaperScout 用更少的 expand 拿到更高召回。这说明盲目扩展引文的收益在递减，而"往哪里扩"才是瓶颈。

### Q2 覆盖与停止

有两条成熟的传统，但都不能直接搬过来。

- TAR 停止规则（knee/target、Quant/QuantCI、QBCB 证书、超几何检验、counting process、SAFE）都假设存在一个可枚举的封闭候选池，问题是"排序筛到哪里可以停"。领域图补全面对的是开放世界，没有候选池。
- 系统综述的 capture–recapture（Spoor 1996、Kastner 2007）用多个检索源的重叠来估计总量，可以迁移到开放世界。代价是独立同质的可捕获性假设：检索通道之间一旦相关（共用同一个嵌入，或都偏爱高被引论文），总量会被低估。Luggen 2019 也报告插入突发会造成高估。

开放世界里还有两个做法。Undermind 在单查询内用发现曲线做指数饱和拟合来预测总数，并用与 Google Scholar 重叠的比例作独立核对。agent 侧的停止信号（HERO 的新引用产出率、RAAC 的 novelty/coverage、CGDP 的 exhaustion gate、TASR、CoVeR）只判断"再搜还有没有收益"，不估计"还缺多少"。

citation snowballing：Sahu、PaSa、SPAR、Crase、Paper Finder 都把 backward（以及部分 forward）滚雪球当作召回的主要来源。Sahu 的数据显示，召回从不到 20% 升到 80% 以上，主要靠沿参考文献 BFS。

living review：现有框架（Riaz 的 Watcher 模块、DAS-2M 持续更新、LKM 逐篇增量、CKM 滑动窗口）都是按固定口径推送新论文的"监视"模式，不是由缺口拉取的"补全"模式。Fok et al. 的访谈说明，持续更新综述在人工流程里不可持续。

### Q3 会增长的学术知识库是否做定向检索

- LKM：会增长，但增长方式是全量摄入各大预印本库和期刊，没有缺口检测。它的审计明确只测幻觉、不测遗漏，写回是未来工作。
- ScholarStack：collection 固定；open-world 检索时外部找到的论文不登记入库；作者明确说缺证据不等于不存在。
- Agents-K1：有类型化的结构空洞检测（O5），用于找研究机会；web 检索结果不写回 KG。
- SciAtlas：唯一明确做"覆盖诊断 → 定向追问"的学术知识库，缺口类型是时间窗和方法分支。但追问发生在内部图谱里，服务单次任务，不写回，没有消融；图谱本身是周期性批量更新。
- Mechanist：一次性按主题选语料，再扩展 schema，没有增长机制。
- ASKS：冻结的 manifest；coverage 等指标是事后结构统计。
- Lacuna：批量构建，增量更新列为未来工作。

结论：被检查的学术知识库里，没有一个由检测到的缺口驱动外部获取，并把结果写回库。离这个闭环最近的是通用 KG 侧的槽位级补全：West 2014 学习搜索 query 填缺值，AgREE 为新兴实体检索三元组。它们的粒度是"已知实体的空属性"，不是"缺哪些论文或方法"。

### Q4 最接近的经典任务与强基线

"给定一组论文，还缺什么"在经典任务里对应 global citation recommendation 的集合形态：参考文献列表补全、引用列表预测（CiteRAG）、审稿人指出的漏引（RMC/CitationR），以及以系统综述为 gold 的 seed-based retrieval（Sjögårde & Ahlgren）。CiteME 是单条归因，形态不同。

已核实的强基线：
- 结构类：co-citation、bibliographic coupling 和 direct citation 三者组合，再加文本（Sjögårde & Ahlgren：co-citation 是最好的单项）；候选的引文图邻居扩展（Nogueira 2020）。
- 文本类：BM25 在大候选池上依然很有竞争力（Medić & Šnajder 2022）；SPECTER、SciNCL 嵌入 kNN。在 LitSearch 这类自然语言查询上 dense 检索明显更好。
- LLM 类：Sahu 式"LLM 拟关键词 + 参考文献 BFS + LLM 重排"；PaSa/PaperScout 这类 RL agent；CiteAgent。

评测时要处理 gold 不完整的问题。Sahu 已经证明人类引用列表里只有一半被判为相关，而 AI 列表里的相关文献很多不在人类引用里。如果拿综述的参考文献当"族成员 gold"，召回会被系统性低估，"新发现"也会被误判为错误。

### Q5 缺口类型学与 value-of-information

已有的缺口类型都只服务一种动作：

| 工作 | 缺口类型 | 动作 |
|---|---|---|
| HERO | unexplored aspects / insufficient evidence / unresolved contradictions | 同一种：生成追问 + submodular 选择 |
| PaSaMaster | missing coverage / ambiguous constraints / under-explored directions | 更新 checklist；不映射到三通道中的特定一个 |
| SciAtlas | missing time windows / methodological branches | 同一种：内部图谱定向追问 |
| Agents-K1 O5 | orphan methods / singleton datasets / disconnected papers / sparse cells | 用于 ideation，不获取 |
| Re:CAP | 可能缺失的主题 | 生成问题检索 + 判新 |
| GAPMAP | explicit / implicit 研究空白 | 识别，不获取 |

这些类型和本项目关心的四类缺口有部分对应。"时间窗缺失"大致对应前沿/新作缺口；"方法分支缺失"和"sparse cells"大致对应未覆盖区域；"insufficient evidence"大致对应已知论文的某方面缺失。但没有人把这些类型分别连到不同的获取动作上，例如：缺已知族成员走族内 co-citation/coupling；缺前沿走被引方向 + 时间切片检索；缺未覆盖区域走 query 生成；缺某方面走全文重读。也没有人按类型估计各自的收益。

VoI 方面，BED-LLM（EIG 选问题）、CGDP（POMDP 信念状态）、Narasimhan 2016（准确率增益减成本的 RL reward）、Callaghan/Lewis（筛选成本与召回保证的权衡）提供了框架，但都没有用在"持久领域图上该先补哪个缺口"这个决策上。

## 3. 开放问题（均以表中条目为据）

1. **闭环缺失：缺口 → 外部获取 → 写回 → 再诊断。** LKM 只有批量增长；SciAtlas 和 Agents-K1 能检出缺口但不写回；ScholarStack 明确不登记外部发现。持久领域图上的获取闭环，以及它对下游任务的增益，目前没有被验证过。
2. **分区域、可累积的覆盖估计。** 封闭池停止规则不适用于开放世界。capture–recapture 只做到整综述的一个标量。Undermind 只在单查询内估计。Luggen 只到实体计数。"每个方法族或每个图区域还缺多少，并随跨任务的检索历史持续更新"没有人做。此外，通道相关会让估计量有偏（参见第 2 节 Q2），这是必须正面处理的统计问题，不是工程细节。
3. **缺口类型 → 获取动作 → 优先级。** HERO、PaSaMaster、SciAtlas 都有缺口类型，但一类缺口对应一个动作。BED-LLM、CGDP 有 VoI 框架，但没用在文献上。按类型路由动作、按预期边际收益排序，这件事是空位。
4. **"未找到"与"不存在"的区分要有凭据。** ScholarStack 明确说缺证据不能推出不存在。Undermind 声称收敛后可以"确认什么都没有"，但这个结论只在单查询里成立。持久库中目前没有一种记录形式，能写下"搜过哪些通道、花了多少成本、估计还缺多少、截至何时"，作为可更新的缺席凭据。
5. **已知论文的"方面缺失"作为一种获取动作。** HERO 的 insufficient evidence 和 LKM 自认的图表与公式漏抽都指向这个问题。没有系统把"回到某篇已入库论文读全文或补抽"当成与"找新论文"并列、可以排优先级的动作。
6. **补全评测协议。** gold 本身不完整（Sahu、ScholarQuest），综述引文池又是封闭评测（StructSurvey、SurGE、DAS）。有三种可组合的做法：遮蔽已知族成员再看能否找回；按时间冻结，测能否发现冻结点之后的论文（CKM、RollingEval）；对超出 gold 的发现做 pooled 判新（Re:CAP、Sahu 的相关性评审）。目前没有一个公认的"领域图补全"基准。
7. **跨任务摊销。** 每次查询都从零开始（第 1A 表全部条目）。持久的图可以让后续查询复用前面的覆盖证据，但这种摊销会带来多少召回和成本收益，没有数据。

## 4. 对本项目 map-completion 模块的直接含义（推断，供讨论）

- **新颖性边界。** 单独主张"缺口驱动检索""缺口类型""覆盖估计""持久增长"中的任何一项，都会撞上 SciAtlas、HERO、Undermind 或 LKM。能站住的是这四者的组合在持久领域图上闭环，并且有逐件的消融证据。SciAtlas 的 coverage-guided retrieval 必须作为最近邻先例正面比较。
- **最低基线集。** 至少包括四个：①co-citation + coupling + direct citation 组合加文本（Sjögårde 式）；②Sahu 式 LLM 关键词 + 参考文献 BFS；③SPECTER/SciNCL 以族为中心做 kNN；④PaSa 或 PaperScout 作为 agent 基线。条件允许时，再加一个 SciAtlas 式时间窗/方法分支的覆盖诊断追问。
- **覆盖估计的统计风险。** 用经典 capture–recapture 估计器时，要显式建模或削弱通道相关（让通道在结构上异质，比如引文通道和文本通道），并在遮蔽实验里检验估计是否校准：预测的缺失数与实际找回数之差。

## 5. 一手来源清单（表中所有条目）

arXiv（abs/HTML 可直接访问）：2501.10120, 2507.15245, 2601.10029, 2605.14306, 2608.24809, 2607.00597, 2605.29234, 2606.20235, 2608.15191, 2609.24122, 2609.27297, 2609.23735, 2606.13669, 2605.22878, 2608.12036, 2608.29612, 2606.26246, 2608.18034, 2607.01243, 2604.12243, 1603.07954, 2508.04118, 2502.06472, 2605.17072, 2505.07184, 2506.10737, 1909.01109, 2305.05403, 2106.09871, 2108.12746, 2606.15380, 1802.08301, 2004.07180, 2202.06671, 2001.08687, 2403.09295, 2407.12861, 2210.10073, 2601.14949, 2604.17680, 2407.18940, 2412.15249, 2002.06961, 2508.21184, 2605.07042, 2606.13814, 2609.26086, 2510.25055, 2605.05258, 2605.31468, 2502.00881。

ACL Anthology：2026.nslp-1.19（HERO）, 2025.ijcnlp-long.193（SurveyGen-I）, 2023.findings-emnlp.171, 2022.sdp-1.3, 2024.lrec-main.1196。

PMC / DOI / 其他：PMC2655834（Kastner 2007）, PMC7700715（Callaghan 2020）, PMC10908130（SAFE）, PMC11975841（Riaz 2024）, doi:10.1136/bmj.313.7053.342（Spoor 1996，仅元数据）, doi:10.1145/2911451.2911510（Cormack & Grossman 2016，仅元数据）, DBLP conf/www/WestGMSGL14 + infolab.stanford.edu PDF（West 2014）, undermind.ai/whitepaper.pdf, allenai.org/blog/paper-finder, github.com/allenai/asta-paper-finder, lilly.elicit.com 评测博客。

## 6. 未核实或排除（不进主表）

- **未核实**（只在搜索结果里出现，一手页面打不开或没读）：TAR 的 point-process 停止（ACM TOIS doi:10.1145/3631990，403）；counting-process 停止（SIGIR 2021, doi:10.1145/3404835.3463013）；Decision-theoretic stopping rules（White Rose PDF）；"Taxonomy Maintenance in the Wild over Evolving Scholarly Data"（ACM PACMMOD, doi:10.1145/3837127，403）；BibliZap、Thoth 2.0、citationchaser 等自动引文追溯工具；"Automated citation searching in systematic review production: A simulation study"；Research Square 上的 LLM living SR engine；Cochrane ESM 的 Elicit 与传统检索对比（403）；Helicase、Paper Circle（ACL 2026）、SurveyAgent-HKA（2609.05938）、DeepSurvey（2605.29522）、"When Should Multi-Round RAG Stop?"（2608.13237）、ReLTEx（2608.10970）、SGHA（2608.17501）。
- **已核实但与主题相关度低，未展开**：FlowSearch（ACL 2026 long 971，动态知识流 deep research，通用基准）；H-MAPS（2605.10097，用户画像记忆）；Novelty-Aware Agentic Retrieval（2606.22151，100 篇语料原型）；Agentic AutoSurvey（2509.18661）；ResearchArena（2406.10291）；DeepScholar-Bench（2508.20033）；SurGE（2508.15658）；OpenScholar（2411.14199）；PaperQA2（2409.13740）；ProfOlaf（2510.26750）。其中 SurGE、DeepScholar-Bench、ResearchArena 可作为"引文发现召回"评测的候选来源。
