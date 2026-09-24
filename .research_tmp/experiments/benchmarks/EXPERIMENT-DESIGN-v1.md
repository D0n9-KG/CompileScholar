# CompileScholar 实验设计方案 v1.0

> 状态：**初步设计（v1.0-draft）**——2026-09-27 经五路 CDP 调研修正、用户多轮质询打磨；用户定位'大概差不多了但还差点意思'。下一 session 将做：地毯式系统审视（设计+实现+论文叙事）+ 实验设计能否支撑顶会论文的重新评估（可重新调研），本设计届时按审视结论修订。执行细节根据实际跑数效果随时调整（用户既定方针）。
> 竞品时钟：ScholarStack（arXiv 2609.23735，2026-09-20）同生态位已上线——静态编译路线被占，我们的差异化=知识模型驱动检索+库生长，须加快
> 调研底座：literature/survey_taskdriven_benchmarks_2026-09-27/（五路报告，全部一手核验）

## 〇、设计原则

1. **任务驱动**（ScholarStack 借鉴不照抄）：任务族选择服从我们系统的能力主张，每族回答一句"我们宣称什么"
2. **主实验只打外部方法**：每 benchmark 配 2 个权威外部系统，同底座自跑；full-text 类自对照全部推消融（系统定稿后另做）
3. **统一协议**：全臂统一底座 Qwen3.8-27B（matched-base-model，ScholarStack 同款协议）；judge 全本地开源（Prometheus/AttrScore/程序化指标）——两代 ScholarQA judge 已分叉的教训
4. **自跑为主、引用为辅**（社区惯例：19 个同生态位工作 12 个全自跑、直接引用零确证）；公开数字只进引用行，分层呈现注明协议
5. **可行性红线**：不部署额外大模型（OpenScholar/PaSa-7B 复现均因部署成本除名）、不预编译大库（SAGE 空白库起步、MDAQA 段落直下、单篇即时编译）、单篇全链 ~30min 在预算内

## 一、系统定位（实验要证明什么）

**主张**：预编译类型化知识模型（谱系/缺口/矩阵/覆盖视图）驱动科学文献的检索、综合与单篇理解，且检索劳动可沉淀为结构化知识实现跨题复利。

**对 ScholarStack 的差异化**（related work 定位句）：分层资产复用已被证明有价值（ScholarStack），我们回答下一个问题——当语料不敷所需时，编译出的知识模型如何驱动库外的检索与库自身的生长。

四支柱 → 四任务族映射：
| 宣称 | 验证场 |
|---|---|
| 知识模型驱动检索（缺口注入词汇/谱系外推） | 族一检索 |
| 类型化知识模型答关系型综合问题 | 族二 QA |
| 编译不损失单篇粒度忠实性 | 族三单篇 QA |
| 检索劳动结构化沉淀、跨题复利 | 族一 SAGE 场纵向分析 |

## 二、任务族与实验矩阵

### 族一 A：固定语料检索 — LitSearch

- **Benchmark**：LitSearch（EMNLP 2024，597 真实检索查询/64,183 篇语料，程序化 Recall@k 判分零 judge）
- **选型理由**：唯一以真实科研者检索意图为题源的固定池基准；官方 BM25/GTR/E5/GRIT 数字公开可校准；语料已本地
- **初始库**：64K 全量 Tier-1 粗抽（摘要→轻记录，12s/篇×32 并发≈**6.7 小时**）+ 语料自带引用图（实测 citation 目标 100% 落在语料内，谱系边免费）——**不做深抽**（64K×30min=32 GPU 天不可行）。外部基线官方口径只索引 title+abstract，我们 Tier-1 已超配。深抽只给检索过程中被判定为核心论文（投资阶梯，分钟级）
- **外部方法**：**BM25 + E5/GRIT**（官方协议复现+官方数字校准）——检索族的权威对手是检索器本身；GRIT/E5 是对比学习训练的 dense 检索器，与我们零训练形态同场对比恰是基准语义（官方表即此形态）；我们赢=零训练结构知识够用，分层看关系型 vs 语义型子集
- **风险预案**：整体打平或略负时按查询类型分层报告（关系型子集应显著胜）

### 族一 B：开放语料检索 — SAGE（web-search 轨）

- **Benchmark**：SAGE open_ended 600 题（arXiv 2602.05975，Yale 团队；HF allenai/sage-retrieval 7MB query+gold，S2 paperId 100% 对齐）
- **选型理由**：推理密集型检索（从研究背景推断该找什么）正对 gap_search 靶心；官方锚点现成（BM25@10 EM 81.2 / GPT-5 71.7 / DR Tulu 42.0）；web-search 轨协议=agent 自由检索、gold 出现在输出/引用即得分（API 检索轨合法同构）
- **已知弱点（接受）**：无 venue、采用度低（仅 ScholarStack 1 篇真评测）、20 万语料未公开（走 API 轨绕开）——采用度低=机会，带官方锚点赢比在卷熟基准挤分好发
- **初始库：空白库起步**（零预编译）。冷启动第一步退化为盲检索是设计内行为；差异从第二步起（粗抽回流挂边→谱系/缺口开始积累→后续检索非盲）
- **外部方法（自跑）**：**ReAct 检索基线**（同底座 Qwen3.8-27B 的标准 agentic web-search 循环，无知识模型——外部方法形态而非我们系统的阉割版，与 Search-o1/AstaBench ReAct 臂同构）；另有 BM25 81.2 / GPT-5 71.7 / DR Tulu 42.0 / PaSa 0.53 官方数字引用行。
  ⚠️ 设计注（09-27 用户裁定）：原"盲缓存对照臂"（我们系统关知识模型组件）本质是消融，已撤——消融统一押后到系统定稿。库生长纵向分析同降级（其"信息量对齐对照"依赖该消融臂），归入未来消融章节
- **纵向分析（机制主张，结果好进主表/差降分析节）**：600 题顺序连跑，按题目序号分层看曲线——我们（结构化沉淀）的后期题增益 vs 盲缓存臂的对应增益，配对检验。预注册判据：后期题相对增益显著大于盲缓存臂。**叙事=同样的检索经历，结构化沉淀跨题复利、原始缓存不会**（非自然规律：信息量对齐后只剩表示形态差异）
- **PaSa-7B**（ACL 2025）作可选臂：需本地部署 2×7B+Google API+定制依赖，时间富余再上；其 RealScholarQuery recall@20 0.5301 作引用行

### 族二 A：开放语料跨论文 QA — ScholarQA-Multi 108

- **Benchmark**：ScholarQA-Multi（Nature 正刊 2026，108 题跨论文综合，441 gold 段落+专家答案全公开）
- **选型理由**：唯一"题+gold+专家答案"全公开的原生开放跨论文综合基准；无同形态替代者（五路调研确认）；闭集成绩 0.6503（ours）/0.5146（LightRAG）/0.4784（PaperQA2）自动成为"检索上界"参照列（零成本复用）
- **初始库：gold 剔除版**。解释：430 篇语料建成的完整 KB 我们已有（闭集 0.6503 用的就是它）。但开集实验里如果留着"本题 gold 论文自己被深抽的记录"，模型答这题时等于从库里直接读答案=循环论证。所以**每题临时剔除该题 gold 论文自己的记录**，库剩余 ~425 篇的谱系/缺口/实体照常可用——模型要答对，必须靠这些"外围知识"驱动检索去库外把 gold 论文找回来。这是防作弊设计，不是常态运行形态
- **外部方法**：**PaperQA2**（闭集 harness 现成改开集）+ **STORM**（NAACL 2024，31.5k★，长文综合生成与 Multi 答案形态最对口，AstaBench CS2 官方数字 78.3 佐证权威性）——两个都是推理时 agentic 系统，同形态公平
- **判分三轨**（全本地）：Prometheus（org/cov/rel 主轨，gold-agnostic 贴真实用户）+ AttrScore-flan-t5-xl（引用质量）+ Citation F1 对 gold（确定性，检索测量）
- **引用行**：OpenScholar Multi LLM 4.51/Cite 37.5、PaperQA2 3.82/47.2、GPT-4o 4.16/0.7（Nature Table 1，judge=Prometheus v2 同口径）

### 族二 B：固定语料跨论文 QA — MDAQA

- **Benchmark**：MDAQA（EMNLP 2025 Findings，797 题、每题强制 2-4 篇 gold+gold 答案，SPIQA 段落级语料直下零解析成本）
- **选型理由**：唯一同时满足"正式 venue+固定 gold 多篇形态+数据公开"的跨论文综合基准（五路调研确认，PaperScope/LitBench/M3SciQA 均有硬伤出局）；ScholarStack 同款考场同款 5 维判分（含 cross-paper synthesis 单列维度）——我们的数字与其 Full-text 8.46/资产层 18.41 引用行隔空同表可比
- **初始库**：gold 语料编译（每题 support 论文，分层子集 ~200-300 题控制建库量）
- **外部方法**：**HippoRAG 2**（ICML 2025；5 篇 25-26 论文基线表实读 3/5 最高频统计判决；语料级多跳 QA 设计目标精确对位；开源成熟）+ **PaperQA2**（gold 论文集给定形态=其单卷宗主形态）
- **判分**：5 维 LLM judge（correctness/completeness/cross-paper synthesis/calibration/directness，ScholarStack 同款匿名化协议）+ 官方自动指标参考轨
- **与 ScholarStack 的错位**：它闭卷跑（gold 给定比表示形态），我们开卷跑（gold 语料编译后开放答题+外扩）——差异如实披露，展示它没有的开集能力

### 族三：单篇论文 QA — PeerQA（单基准）

- **Benchmark**：PeerQA（**NAACL 2025**，579 题/208 篇 ML/NLP+地学+公卫论文，真实审稿人问题+论文原作者作答，官方三轨判分全程序化，数据 tudatalib zip 直下）
- **选型理由**：同族里最新且有正式 venue（QASPER=NAACL 2021 太老出局；QASA=ICML 2023 且无官方判分器出局）；题源最真实（审稿人真实疑问而非标注员出题）；**不可答轨**（answerability）正好测我们的 absence 记录——"库里没有"是诚实答案不是硬编；判分零 judge
- **协议坑（一手实证）**：PeerQA 检索必须全库，oracle 逐篇索引会虚高 R@10 0.011→1.000
- **初始库**：208 篇论文全量深抽（208×30min÷30 并发≈**3.5 小时**，全库可负担）——比"每题即时编译"协议更干净（检索轨全库检索本来就是 PeerQA 官方协议）
- **外部方法**：**PaperQA2**（单篇精读主形态，2025-26 唯二被反复实跑的系统之一）+ **RAPTOR**（ICLR 2024，递归摘要树——"摘要式编译"对照面，与 HippoRAG 2 的图索引、我们的类型化记录构成表示层三向对比：类型化 vs 图 vs 摘要）
- **判分**：官方 answerability F1 + ROUGE-L + 证据检索 nDCG，全程序化

### 库生长 — 押后到消融章节（09-27 用户裁定）

- 纵向连跑的数据在 SAGE 主实验中顺带产生（600 题连跑本身带库生长），但"跨题复利 vs 缓存效应"的分离论证依赖消融臂（信息量对齐的盲缓存对照）——统一归入系统定稿后的消融实验
- Multi 两轮同理撤出当前设计

## 三、主实验总表（一页纸）

| 任务族 | Benchmark | 我们 | 外部方法（自跑） | 引用行 | 判分 |
|---|---|---|---|---|---|
| 固定检索 | LitSearch 597 | 完整系统 | BM25 + E5/GRIT | 官方全套 | Recall@k 程序化 |
| 开放检索 | SAGE open_ended 600 | 完整系统（空白库连跑） | ReAct 检索基线 | BM25 81.2 / GPT-5 71.7 / DR Tulu 42.0 / PaSa 0.53 | Weighted Recall 程序化 |
| 开放 QA | Multi 108 | 完整系统（gold 剔除库） | PaperQA2 + STORM | OpenScholar 4.51 等 Nature 全表 | Prometheus + AttrScore + Cite F1 |
| 固定 QA | MDAQA ~250 分层 | 完整系统（gold 编译） | HippoRAG 2 + PaperQA2 | ScholarStack 8.46/18.41 | 5 维 LLM judge |
| 单篇 QA | PeerQA 579 | 完整系统（208 篇全量深抽 3.5h） | PaperQA2 + RAPTOR | ScholarStack PeerQA 数字 | 官方三轨程序化 |
| （押后）库生长与消融 | 系统定稿后统一做 | — | — | — | — |

外部方法出场账：PaperQA2×3 场（单篇/固定 QA/开放 QA——每场都是它的主形态）、STORM×1、HippoRAG 2×1、RAPTOR×1、BM25/E5/GRIT×1、盲缓存臂×1——无方法刷场，每个都在最对口的场。

## 四、判分与统计纪律

- **judge 全本地**：Prometheus-13B/8x7b（Multi 主判，与 Nature 口径同家族）+ AttrScore-flan-t5-xl + 各基准官方程序化指标；两代 ScholarQA judge 分叉教训=任何引用数字必须钉死 judge 来源
- **judge 盲化**（ScholarStack 协议）：候选匿名编号+条件隐藏+随机呈现序
- **统计**：臂间配对 bootstrap 95% CI（闭集三臂同款协议）；库生长预注册判据=后期题相对增益>盲缓存臂对应增益
- **预注册纪律沿用**：判据先冻结后跑；失败分支如实报告；公开数字永不混表

## 五、排除项与理由（论文 limitations/排除说明素材）

- **SQA2/AstaBench CS2**：schema 适配成本高+rubric held-in 偏置压外部系统分+judge 已三代演变——只引不跑
- **OpenScholar 自跑**：全管线绑定 OSDS 45M DataStore+repo 停更+部署成本——引用行处理（换源复现虽有 PaperQA2 先例但可行性不划算）
- **PaSa-7B 自跑**：2×7B 部署+Google API 依赖——可选臂
- **SciConBench**：clean-room 协议与预编译 KB 根本冲突；**PaperScope**：test 集不公开+我们已跑 dev 判过打平；**MDAQA 官方自动指标**：仪器会低估我们（措辞/长度错位），只作参考轨
- **消融**：系统未定稿，组件级拆解等定稿后另做（用户裁定）
- **撞名预警**：外部已发布同名 LitTraceQA（2608.07370），命名避开

## 六、执行顺序与成本（用户裁）

| 序 | 项 | 成本 | 依赖 |
|---|---|---|---|
| 1 | 判分管线（Prometheus/AttrScore 本地跑通） | 0.5 天 | 无 |
| 2 | LitSearch 场（最便宜出数：语料本地+判分程序化+检索器复现） | 1-2 天 | 无 |
| 3 | SAGE 场（7MB 直下+API 轨+空白库连跑） | 2-3 天 | 检索栈（已就绪） |
| 4 | Multi 开放场（harness 复用度最高） | 2-3 天 | 判分管线 |
| 5 | MDAQA 场（语料直下+gold 编译） | 2-3 天 | 建库管线（已就绪） |
| 6 | PeerQA 场（208 篇深抽 3.5h+harness 适配） | 2-3 天 | 无 |
| 7 | 库生长分析（搭 3/4 便车） | +1 天 | 3/4 完成 |

总计 ~12-15 工作日。机动裁点：STORM/RAPTOR/HippoRAG 2 的适配各含半天环境风险；MDAQA 分层子集量（250 vs 797）视首批结果伸缩；库生长分析节视 SAGE 纵向预注册判据结果升降级。

## 七、已核验事实底账（防再犯）

- SAGE=Yale 团队非 AI2 官方；无 venue；语料 20 万篇未公开不可复现；HF 7MB query+gold 完整；S2 paperId 100% 对齐（DOI 37%/arXiv 11%）
- SAGE 无先验集概念（1,750 篇/域是 ScholarStack 自己的对照设定，已纠正）
- Search-o1/DeepResearcher/WebThinker 全部无学术检索基准数字（只能重跑不能引用）
- Valyu"总榜第一"=厂商 gaming rubric 实锤（rubric 泄漏进生成端）
- Multi 不在 AstaBench（仅 CS 域）；Multi 公开数字只有 Nature 套（judge=Prometheus v2）
- QASA=ICML 2023（非 ACL 2024）；题量 1554+244（本地 1375 为子集）；无官方判分器
- PeerQA oracle 索引虚高实证（R@10 0.011→1.000）
- LitTraceQA 撞名（2608.07370）
- ScholarStack 四支柱核对：无缺口驱动检索/无库生长/无 schema 演化/无 absence 一等记录——差异化全部成立（但注意其 matched-base-model+盲化 judge 协议值得镜像）
