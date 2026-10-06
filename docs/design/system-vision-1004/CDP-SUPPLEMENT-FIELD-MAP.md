# 领域地图调研 CDP 补充轮（发现层补漏）

日期 2026-10-04，覆盖到 2026-10 初。补的是 RESEARCH-FIELD-MAP.md 那一轮只用了 WebSearch、可能漏掉的同生态位工作。前一轮主表里已核过的条目这里不再重复，只记新发现，以及已核条目的"论文之后"变化。

## 0. 渠道与可信度

- 发现层：用户 Chrome 经 CDP（web-access 2.5.4）。Google 普通搜索可用，中途触发过一次 "异常流量" 验证页，之后放慢节奏恢复；DuckDuckGo 用作备用；百度、搜狗微信用于中文；HF Papers 搜索 API、arXiv 站内搜索、GitHub 站内搜索用于定向扫。
- 被挡的渠道：OpenReview 出 "Verifying your browser" 验证页（2 个 ID 没核成）；xjishu.com 有反爬验证；AMiner 技术树页面渲染出来了，但数据接口报 "系统出错"，并要求登录；Google Scholar 按站点经验会被拦，没有尝试，cited-by 改走 Semantic Scholar 网页版 + OpenAlex。
- 验证层：主表每一行都打开了一手页面，包括 arXiv abs、ACL Anthology、官方 GitHub README/源码、HF 数据集 API、官方产品站与其线上 API、天眼查专利页（专利公布号）。Intern-Atlas 的开源数据另外下载了 method_relations 分片做了计数。
- 判决口径：威胁"真"指在本项目的核心卖点上直接重叠；"半真"指占掉了一个组件或一个概念，但没有占掉组合；"假"指同主题、不同问题。

---

## 1. 判决表

| 工作 | 一手链接 | 发现渠道 | 表示什么 | 怎么建 | 怎么评 | 和我们的关系 |
|---|---|---|---|---|---|---|
| **Intern-Atlas 论文之后的形态**（产品站 + 开源 builder + HF 数据 + 线上 API） | 产品 intern-atlas.opendatalab.org.cn；代码 github.com/OpenRaiser/Intern-Atlas（2026-05-01 建，42★）；数据 huggingface.co/datasets/OpenRaiser/Intern-Atlas（不设 gated）；API intern-atlas.opendatalab.org.cn/api（v2.0，live） | 百度（产品站只在百度前排）→ 产品站 → 代码 / 数据 / API | ① 代码：边类型 extends / improves / replaces / adapts / **combines** / uses_component / compares / background，其中 combines 是论文七类之外新加的；方法间关系 variant_of / component_of / specializes / combines / optimizes / inspired_by。② 每条边的属性有 bottleneck、mechanism、mechanism_type（如 combination、architecture_replacement、component_reuse）、dimensions（如 memory_efficiency、inference_speed）、sacrifice、tradeoff、improvement、scope、confidence、source_year 与 target_year。③ 产品站自述的八段流水线里写了"实体归一""占位节点日后升级为正式节点""方法父子层级 Attention→MHA→GQA""论文身份增量合并""增量同步：只处理新论文，不重跑全量" | 本地 builder：LLM 逐对判定"新论文对旧论文"的关系；SQLite 的 citations 表以 (source_id, target_id) 为主键，写入用 INSERT OR REPLACE | 没有新的评测。API /api/stats 实测：4,203,500 篇（全文 1,030,313 篇、stub 3,173,187 篇），4,143,766 条边，8,155 个方法 | **由半真上调为偏真**。前一轮列为空位的几项里，有三项已经被它在工程层或产品宣称层碰到：combines 类型、方法变体层级、增量同步。但实测有三点：(a) combines 仍是论文对论文的二元边，线上 API 抽查 5 条 combines 边全是一对一，不是 n 元组合对象；(b) HF 上的 method_relations 只有 **60 行**（variant_of 47 / specializes 7 / combines 3 / inspired_by 2 / component_of 1），来源只有 seed 26 行与 legacy_parent 34 行，无证据论文，层级是一份薄薄的手工种子；(c) "增量同步"只停在产品站的文字，代码里是覆盖写，没有版本、as-of 快照、族、分裂合并事件，也没有任何身份错误率报告 |
| **research-genealogy**（NTU Siqiang Group） | github.com/NTU-Siqiang-Group/research-genealogy（2026-09-07 建，09-16 最后推送） | GitHub 站内搜索 | 从一篇种子论文出发，建前向与后向的"技术延续"DAG，允许多个父节点；关系强度分三档：weak（引用或 Related Work 提及）、medium（Introduction/Preliminaries 实质讨论、指出局限、作者重叠）、strong（实验 baseline、隐式 artifact baseline、直接方法依赖） | OpenAlex 取有界邻域 → 多通道取全文 → 按章节定位证据 → 检索完成后的流程是确定性的；LLM 只用来找作者主页；每个结果都保存证据段落、URL、checksum | 无评测，自称 research prototype | **假（低）**。节点是论文，范围限于单个种子的局部，没有方法实体和族。可借的是"按章节区分证据强度"这套判据，它和我们的逐边证据纪律是同一个方向 |
| **LLM Research Lineage** | github.com/llmsresearch/llm-research-lineage（2026-09-18 建，10★）；在线图 llmsresearch.github.io/llm-research-lineage | GitHub 站内搜索 | 人工整理的 LLM 技术谱系图：11 个流水线阶段（tokenization / attention / position / norm / FFN-MoE / 目标 / 系统 / 后训练 / 推理 / 服务等）作横轴、时间作纵轴；150+ 个技术节点、71 个模型投影；边分 direct descent 与 influence 两种；标注 set-aside 与 revived 分支（衰退后复活）；节点 JSON 带 `par`（多父列表）、`fam`、`st`（root/live…）、why/impact | 人工整理，JSON 数据库加校验脚本，数据用 CC BY 4.0 | 无 | **半真（概念层）**。"多维（按流水线阶段）+ 多父 + 分支被搁置后又复活 + 一个模型由多条谱系重组而成"这套表示，已经有人以手工形态做出来、公开发布了。它不是自动系统，规模也小，但削弱了"没人这样表示过"的说法。它也可以直接拿来当 LLM 领域的定性金标 |
| **SciNet / SciNetBench**（清华 FIB Lab，ICML 2026） | arXiv 2601.03260；github.com/tsinghua-fib-lab/SciNet | Google（搜 "scientific lineage"）→ GitHub | 关系感知检索基准，分三层：ego（novelty 与 disruption 指数）、pair（引文情感、共同提及）、path（"从 X 到 Y 的最有影响的引用路径"，即谱系重建） | 2.69 亿篇的元数据库（OpenAlex 2025-07 快照 + arXiv 全文），覆盖 7 个学科、2,640 个子领域、8,940 个经人工审核的任务 | 检索类指标 | **假（威胁）/ 可用（资源）**。Task3 的 path 查询按学科分文件，其中 AI 文件有 1,823 条，可作跨学科演化链的补充评测。答案格式本轮没核 |
| **Inferring Scientific Cross-Document Coreference and Hierarchy with Definition-Augmented Relational Reasoning**（Forer & Hope） | ACL Anthology 2026.tacl-1.36（TACL vol.14） | Google（追 SciCo 的后续） | 科学概念的跨文档共指与层级（SciCo 任务） | 检索全文生成"语境相关定义"与"关系性定义"（说明两个提及怎样相关、怎样不同）+ 高效 re-ranking | 在 SciCo 类数据上，表面形式多、歧义高的子集提升明显 | **半真（身份层组件）**。前一轮说 SciCo 是唯一的专门基准、而且是前 LLM 时代的；这篇就是它在 LLM 时代的方法。方法身份与粒度层要引它当先例，并作为 baseline |
| MIR：Methodology Inspiration Retrieval（ACL 2025） | arXiv 2506.00249 | HF Papers 搜索 | Methodology Adjacency Graph：用引文关系刻画"方法谱系" | 引文关系 + dense retriever 注入先验 | MIR 检索数据集 | 假。谱系只是检索先验 |
| AutoReproduce（ACL 2026 main） | arXiv 2505.20662 | HF Papers 搜索 | "paper lineage"：从被引文献挖隐式实现知识 | 多智能体复现 | PaperBench 等 | 假 |
| INSPIRE | arXiv 2609.33233 | arXiv 站内搜索 | 开放问题的文献检索；金标是目标论文实际谱系里分级的被引前驱 | 目标论文发表前三个月设截止 | 检索 + 轨迹分阶段诊断 | 假。它"按目标论文的实际前驱分级"的做法可作评测参考 |
| ForeSci | arXiv 2606.00644 | Google（追 Intern-Atlas 引用方） | 前瞻性研究判断基准，500 个任务，每个任务配截止日对齐的离线 KB | 任务从截止前的 taxonomy 分支派生 | 截止后的论文只用于验证 | 假。但"截止对齐 KB + 截止后验证"是 as-of 快照评测协议的直接先例 |
| Data-Driven Evolution of LIS Research Methods (1990–2022) | arXiv 2606.25320 | Google（"method evolution" 系列） | 情报学领域 4 类细粒度方法实体（算法/模型、数据资源、软件工具、指标）随时间的演化 | 自动抽取方法实体 + 计量 | 描述统计，"涌现-稳定/实用"周期 | 假。计量派对"方法实体级演化"的先例，没有类型化关系 |
| 华中科大专利：一种基于大语言模型驱动的技术演化路径及技术路线图生成方法和系统 | 申请号 CN202610439158.1，公布号 CN122432705A（申请 2026-04-03，公布 2026-07-21，审中）；天眼查页 zhuanli.tianyancha.com/f7338987217513389b97b70067e2e85d | 百度、百度学术 | 技术主题与技术演化路径，加上从技术报告抽的"共识"节点、从专家言论抽的"非共识"节点与风险，融合成统一 KG | 主题建模 + 大模型实体与关系识别 + 大模型驱动的 KG 链路预测，用于"动态扩展" | 摘要未给 | 假（领域不同：专利与技术路线图）。它是国内产业界的对应物，"共识 vs 非共识"分层可以参考 |
| AMiner 知因分析·技术全景分析（技术树） | vip.aminer.cn/analysis/techtree（trend.aminer.cn 已跳转到 vip.aminer.cn/analysis） | 百度 | 按领域给出技术树（人工智能、大模型等 10 个领域），以及发文趋势、热点、成熟度预测 | 未知 | 未知 | 半真（产品层）。产品存在已核实，树的数据接口报错且要求登录，粒度和是否带证据都没核到。2018 年的老 Trend Analysis 是主题级的"出现-变迁-消亡" |

### 引用追踪（cited-by）结论

Semantic Scholar 网页版与 OpenAlex 双查，日期 2026-10-04：

- Intern-Atlas：S2 记 6 条引用，分别是 PARNESS、IG-Bench、SPARK、BIRD、ForeSci、SciAtlas。我在其中三篇的 arXiv html 正文里 grep 到了引用位置，PARNESS、IG-Bench、ForeSci 都只在相关工作里提一句，没有一篇在它上面建系统或拿它做对比。
- THE-Tree：S2 记 1 条（RINoBench，LREC 2026，研究 idea 新颖度判断），无关。
- SciTraj、Scientific Contribution Graph、EvoTree、IG-Bench：S2 页面没有 Citations 区块，即 0 引用。OpenAlex 上六篇全部是 0。
- 结论：这条链目前追不出后续竞品，生态位里的新工作都还没进入引用网络。这也说明 cited-by 不是本轮的有效发现渠道，有效的是 GitHub 和产品站。

### 中文社区与国内评测

- 知乎、搜狗微信：只搜到 Intern-Atlas 的转载报道（百家号 2026-05-07）和 ResearchArcade 的新智元报道（UIUC，arXiv 2511.22036，是图接口不是演化图），没有发现国内另起炉灶的方法演化图系统。
- 智源"AI 知识树"（hub.baai.ac.cn/knowledge-tree）：是 LLM 撰写的主题百科条目，带发展脉络叙述，不是演化图，假。
- NLPCC 2026：11 个 shared task 已在官方页逐条看过。和本题最近的是 Task 10（AI 辅助科研可靠性：claim 级与 citation 级 faithfulness），没有方法演化或谱系任务。
- CCKS 2026 的评测任务清单页面没能解析出来，见未核实清单。

---

## 2. 前一轮漏掉的最重要 5 条

1. **Intern-Atlas 已经越过了论文**。代码加了 combines 边和 variant_of/specializes 等方法层级关系，边上带 dimensions、mechanism_type、tradeoff；产品站宣称做了增量同步、实体归一和方法父子层级；数据（约 2.5GB 边 parquet）与线上 API 都已开放。前一轮把"组合关系、方法粒度、增量更新"列为它的空位，现在要改写为：在二元标签和产品宣称层面它已经碰到了；在 n 元组合对象、族与族级事件、as-of 快照、身份错误率评测这几层，它仍然是空的。实测数据也支持这个区分：层级只有 60 条种子关系，combines 抽查全是一对一，写库是覆盖式。
2. **LLM Research Lineage**。这是人工整理的多维（11 个阶段）、多父、带"搁置/复活"分支的 LLM 技术谱系图，数据用 CC BY 发布。"多维 + 演化 + 族事件"这一表示在概念上已经有公开实例，论文叙事不能再说没人这样表示过；更稳的说法是"没人能自动地、带证据地、可增量地建出来，也没人评过"。
3. **Forer & Hope, TACL 2026**。这是 LLM 时代的科学概念跨文档共指与层级方法，身份与粒度层必须把它当先例和 baseline。前一轮说 SciCo 是唯一工作，这个判断要更正。
4. **SciNet（ICML 2026）**。7 个学科的 path-wise 谱系重建任务，是 IG-Bench 之外又一个跨学科演化链评测来源。
5. **research-genealogy（NTU）**。按章节区分证据强度、允许多父的论文谱系 DAG。威胁低，但它的证据分档规则是现成的可借组件。

## 3. 可直接用的金标与数据集

| 资源 | 位置 | 内容 | 能怎么用 | 注意 |
|---|---|---|---|---|
| Intern-Atlas 开放数据 | HF OpenRaiser/Intern-Atlas（不设 gated；卡片写 MIT，产品站写 CC-BY） | paper_evolution_edges（9 个分片约 2.55GB）、papers（约 535MB）、paper_methods（约 31MB）、method_relations（60 行） | 作为对照 silver 图：逐边比对类型和证据；拿它的 combines 边测我们 n 元组合的召回 | 论文里的 30 篇综述金标（2268 节点 / 1462 边 / 133 链）**不在**这 4 个 config 里，本轮没找到公开位置。两处许可证写法不一致 |
| IG-Bench 仓库 | github.com/VisionXLab/IdeasHaveGenomes | gene_arena/task 有 30 个谱系文件（每个文件含论文、idea_genome 四槽、edges.taxonomy_type 为 Mutation/Speciation/Hybridization 等）；gene_exam 有 42 组题 | 用 Hybridization 标注测 n 元组合，用 taxonomy_type 测边类型 | License 写的是 TBD；单个谱系很小（cs_LLMReasoning 只有 6 篇 5 边） |
| SciNet Task3 | github.com/tsinghua-fib-lab/SciNet，路径 Queries/Task3/queries_task3_path_{AI_KDD,BIO,CHEM,Geo,MATERIAL,MED,PHY}.json | 起点论文到终点论文的谱系路径查询 | 跨学科演化链评测 | 答案文件的位置和格式没核 |
| LLM Research Lineage 数据库 | github.com/llmsresearch/llm-research-lineage，路径 database/components/*.json | 11 个阶段、150+ 节点，带多父、族、状态、why/impact | LLM 领域族与分支事件的定性金标，也可抽查我们的族划分 | 人工整理，覆盖有意取舍 |
| SciCo + Forer & Hope 2026 | arXiv 2104.08809；ACL 2026.tacl-1.36 | 概念共指 + 层级 | 测方法身份的误合并与误拆分 | 概念不全是方法 |

（THE-Tree 88 棵树、EvoTree 11 棵、SciPaths、PST-Bench 前一轮已列，这里不重复。）

## 4. 未核实清单（不进主表）

- OpenReview IaRkIL0s65（"A Multi-Agent LLM Framework with Hierarchical Citation …"，疑似 SurveyG 的会议版）与 OtfkmjpjqT（"Graph-Grounded Hierarchical Search for Scientific Ideation"）：两个都被浏览器验证页拦下。
- CCKS 2026 评测任务清单：CSDN 转载页与搜狗都没能解析出任务名。
- AMiner 技术树的数据内容与构建机制：接口报错，要求登录。
- NSTL"专利技术演化分析与预测系统""图谱式综述服务"：只见到百度摘要，没打开。
- 另一件专利"一种基于大模型的论文知识图谱构建方法、装置、系统"（xjishu 202610436320，自称"揭示学科演进脉络"）：被反爬验证页拦下，申请号和申请人没核。
- 《融合大语言模型与动态知识图谱的颠覆性技术识别与演化动因研究》（情报理论与实践 2026）：只见到百度学术条目。
- 《基于 PaECTER-BERTopic 与大模型的专利技术主题识别及演化分析》（ISTIC 期刊 PDF）：没打开。
- Intern-Atlas 边数口径不一：百度 AI 卡片转述为"941 万条语义标注边"，线上 /api/stats 实测是 4,143,766 条。差异可能来自是否计入 background 边，没有核清。引用时以 API 实测数为准并注明口径。
- 前一轮未核实里的 "Mapping the evolution of AI: an LLM-driven patent network analysis"：这轮核实了 DOI 10.1007/s11192-026-05827-3（Scientometrics，2026-09-29），书目存在，内容没读。
