# 领域地图（Field Map）的表征与演化建模：文献调研

调研日期 2026-10-04，覆盖到 2026 年 9 月底。目的：弄清别人如何表征一个研究领域、如何建模方法演化（扩展/改进/替代/组合/迁移/分叉/合并/局限驱动），据此设计本项目的领域地图，并找出真正空着的位置。

## 0. 方法与可信度说明

- 发现层：`web-access` 技能在本环境不可用（Skill 调用返回 Unknown skill），改用 WebSearch。WebSearch 结果只当线索，未发现注入指令。
- 验证层：主表每一行都落到一手来源（arXiv abs/html 页、ACL Anthology、PLOS、Crossref DOI 元数据、官方 GitHub）。Teufel 2006、Jurgens 2018、SciCite 2019、CS-KG 2022 四篇读了 PDF 原文页面。
- 精度提示：WebFetch 由小模型转述页面，数字以 arXiv html 正文为准的条目已标注；Intern-Atlas 的 NR 与 CAS 两项转述数值完全相同（84.8 vs 44.9），疑为转述误差，引用前需回原文表格复核。
- "缺什么"一列：标"自述"的是作者 Limitations 里写的，未标的是我对照本题需求的判断。
- 未能一手核实的条目放在第 5 节，不进主表。

---

## 1. 判决表

### A. 经典科学计量：引文网络结构与主题演化

| 工作 | 一手 ID/链接 | 表征什么 | 怎么建 | 怎么评 | 缺什么 |
|---|---|---|---|---|---|
| Main Path Analysis（Hummon & Doreian 1989） | DOI 10.1016/0378-8733(89)90017-8（Crossref 核实书目） | 引文 DAG 上的"主干路径"，即技术发展主线 | 仅引文图，遍历权重（SPC 等，内容据通识） | 案例（DNA 理论史） | 单条或少数路径；节点=论文；边无类型；不表达分叉合并语义 |
| Semantic MPA（Chen et al. 2022, J. Informetrics） | DOI 10.1016/j.joi.2022.101281 | 多条发展轨迹 | 引文路径 + 主题一致性约束选路 | 锂电池专利案例 | 仍是论文/专利节点；轨迹间关系不类型化 |
| Kleinberg burst detection（KDD 2002） | DOI 10.1145/775047.775061 | 词/主题的突发期 | 文档流上的状态机 | 案例 | 只有"热度"，无结构关系 |
| CiteSpace II（Chen, JASIST 2006） | DOI 10.1002/asi.20317 | 研究前沿（burst 词）与知识基础（共被引簇），时间切片网络 | 共被引 + burst + 中介中心性 | 案例可视化 | 簇=论文集合，无方法级语义，簇间关系靠人读 |
| Boyack & Klavans（JASIST 2010） | DOI 10.1002/asi.21419 | 比较共被引/书目耦合/直接引用三种聚类对研究前沿的代表性 | 大规模引文聚类 | 文本一致性等准确度指标 | 只回答"哪种聚类更准"，不涉及演化关系 |
| Mapping Change in Large Networks（Rosvall & Bergstrom 2010） | PLOS ONE e8694，doi 10.1371/journal.pone.0008694 | 领域簇随时间的合并、分裂、涌现（alluvial diagram） | 期刊引文网络（约 7000 刊，1997–2007），bootstrap 显著性聚类 | 案例（神经科学独立成科） | 期刊级粒度；事件只有拓扑，没有"为什么分/合" |
| Phylomemy（Chavalarias & Cointet 2013） | PLOS ONE，doi 10.1371/journal.pone.0054847 | 领域（关键词簇）的系统发生树：涌现/稳定/衰退节点，分叉、合并事件 | 约 20 万篇胚胎学文献关键词 → 共现聚类 → 跨年 Jaccard 匹配 | 案例 + 统计规律（分叉/合并点密度呈 U 形） | 节点是词簇不是方法；继承边无类型；无证据句 |
| CSO + AUGUR（Salatino et al., ISWC 2018 / JCDL 2018） | DOI 10.1007/978-3-030-00668-6_12；10.1145/3197026.3197052；综述 arXiv 2106.12875 | 计算机科学研究领域本体（层级）+ 新主题涌现预测 | Klink-2 自动建本体；AUGUR 用主题共现网络预测胚胎期主题 | 后验对照真实涌现 | 主题级；无方法间演化边 |
| BERTilda（ECML PKDD 2026） | arXiv 2608.18101 | 主题生命周期：continuation / split / merge / disappearance / unclear | 每时间窗独立 embedding 主题模型 + 相似度与双向文档流建时间图，规则打标签 | 三人标注金标子集，多数一致率最高 87% | 语料是政治推文/演讲，非科学文献；主题级 |
| Topical Phase Transitions in AI（2026） | arXiv 2606.12828 | 主题"相变"式涌现与早期预警特征 | 8.08 万篇五大 AI 会议论文（2017–2025）的发表动态 | 2017–21 定特征，2023–25 验证：P 27%、R 63%（基线 13.5%） | 只有涌现信号，无主题间关系 |
| Unified Subject Map for 130 Years of Physics（2026） | arXiv 2606.14043 | APS 全档（1893–2025）映射到 3000+ PhySH 概念，概念生命周期 | 前沿 LLM 回溯标注现代学科词 | 摘要未给评测 | 用今天的词表回标历史，存在时代错置风险（我判） |
| SciEvo（2024/25） | arXiv 2410.09510 | 200 万篇、30 年的元数据+引文图，供时序计量 | 元数据与引文汇编 | 计量分析（如 LLM 领域引用年龄 2.48 年） | 数据集，非结构化地图 |

### B. 引文功能 / 引用意图（演化关系词表的源头）

| 工作 | 一手 ID/链接 | 表征什么 | 怎么建 | 怎么评 | 缺什么 |
|---|---|---|---|---|---|
| Teufel, Siddharthan & Tidhar（EMNLP 2006） | ACL W06-1613（读 PDF） | 12 类引文功能：Weak；CoCoGM/CoCoRo/CoCo-/CoCoXY（对比类）；PBas/PUse/PModi/PMot/PSim/PSup（正面类）；Neut | 人工标注 + 特征分类器 | 116 篇、2829 处引用，κ=0.72（3 人） | 引用实例级，不落到方法实体；无跨论文聚合 |
| Jurgens et al.（TACL 2018） | ACL Q18-1028（读 PDF） | 6 类：Background / Motivation / Uses / Extension / CompareOrContrast / Future；用引文框架度量 NLP 领域演化 | 约 2000 处引用人工标注 → 分类器标全 ACL | 分类 F1；领域级趋势（共识上升等） | 演化是统计意义的"框架比例变化"，没有方法谱系 |
| SciCite + ACL-ARC（Cohan et al., NAACL 2019） | arXiv 1904.01608（读 PDF） | SciCite 3 类：Background / Method / ResultComparison；ACL-ARC 6 类 | 众包标注 + structural scaffolds | ACL-ARC 1941 例；SciCite 约 1.1 万例；F1 | 粗粒度；没有 extends/replaces 之分 |
| MultiCite（NAACL 2022） | ACL 2022.naacl-main.137；github.com/allenai/multicite | 多句、多标签引文语境；标签 Background/Motivation/Future/Similar/Difference/Uses/Extension | 1.2K 篇 CL 论文、12.6K 语境 | 分类与语境抽取 | 仍是"引用意图"，不是"方法关系" |
| FineCite（Findings ACL 2025） | ACL 2025.findings-acl.1259 | 细粒度引文语境跨度 | 1056 个人工标注语境 | 下游 CCA 基准提升最高 25% | 只解决语境边界 |
| SOFT（TPDL 2025） | arXiv 2601.05103 | 引用意图与被引内容类型分成两个正交维度 | 重标 ACL-ARC + ACT2 跨学科测试集 | 人-LLM 一致性、跨域泛化 | 只做分类框架；但"意图 × 内容"正交化对本项目词表设计有直接借鉴 |
| UniCite（NSLP@LREC 2026） | lrec.elra.info/lrec2026-ws-nslp-28；ACL 2026.nslp-1.28 | 统一三套方案：6 个主功能 × 12 子类 + 2 个正交维度 | 4017 条标注（含 1547 新增） | 多任务学习，子功能分类相对提升 21.1% | 同上，停在引用层 |
| LLM 引文意图实验（Koloveas et al., TPDL 2025） | arXiv 2502.14561 | — | 12 个开源 LLM 变体 zero/few/many-shot + 微调 | SciCite/ACL-ARC F1，微调相对提升 8% / 4.3% | 说明 LLM 能做意图分类，不解决方法关系 |
| LLM for Citation Function（LREC 2026） | arXiv 2607.17738 | 7 类（区分中性与评价性立场：批评/赞扬/矛盾） | 多个 7B LLM 与 SciBERT 比较；新数据集 AC3 | ACL-ARC macro-F1 73.3%（Falcon 7B 微调） | 同上 |

### C. 分类体系 / 综述目录 / 层级组织

| 工作 | 一手 ID/链接 | 表征什么 | 怎么建 | 怎么评 | 缺什么 |
|---|---|---|---|---|---|
| TaxoAdapt（ACL 2025） | arXiv 2506.10737；github.com/pkargupta/taxoadapt | 多维分类：方法 / 任务 / 评测指标 / 数据集各一棵树，按语料分布扩宽扩深 | LLM 先验树 + 迭代层级分类对齐语料 | LLM 判：粒度保持 +26.51%、一致性 +50.41% | 多维但各维独立成树；维间无演化边；"evolving"指适配新语料，不是建演化关系 |
| Context-aware multi-aspect taxonomy（Zhu et al., EMNLP 2025） | arXiv 2509.19125 | 按方法/数据/评测等 aspect 分别摘要、聚类成层级 | LLM 识别 aspect + 聚类 | 156 个专家分类（1.16 万篇）为金标 | 静态；无时间维 |
| TaxoAlign（EMNLP 2025） | arXiv 2510.17263 | 主题层级树 | 三阶段指令式 LLM 生成 | CS-TaxoBench：460 训练 + 80 测试分类，金标取自人写综述 | 只有父子关系 |
| TaxoBench（2026） | arXiv 2601.12369 | 检验 deep research agent 能否找全并组织成专家分类 | 72 篇高被引 LLM 综述、3815 篇被引论文、专家分类 | 最好的 agent 只召回约 21% 专家引用；专家平均深度 4.86 层无一配置达到；人比 agent 高约 13 个百分点 | 揭示"检索"和"层级组织"是两个独立瓶颈 |
| HiCaD（Findings EMNLP 2023） | ACL 2023.findings-emnlp.453 | 综述层级目录生成 | 7.6k 综述目录、38.9 万参考文献 | 语义与结构相似度指标 | 目录即树，无演化 |
| HiGTL（2024/25） | arXiv 2410.03761 | 从引文图生成分类树 | 文本+引文联合层级聚类，LLM 命名簇 | 定性/一致性 | 树；无类型化关系 |
| EvoTree（2026） | arXiv 2609.09561 | "演化树"：父概念早于子概念（单调时间约束），过渡性论文挂内部节点而非强塞叶子 | 引文图编码 + 分布式层级聚类 + 时间微调；LLM 只做最后命名 | 11 棵人工整理参考树（352 篇）；NMI、引用方向准确率、marginal paper 检测 | 自述：仅 AI 领域、金标小、依赖综述章节作弱监督、预印本/会议日期混淆；仍是树，不能表达合并 |
| From Papers to Panoramas（2025/26） | arXiv 2504.13834 | 从大领域到具体研究的多级层级 | embedding 聚类 + LLM 提示混合 | 以 LLM agent 在层级中导航找目标论文的效率评 | 静态层级 |
| CHIME（Findings ACL 2024） | arXiv 2407.16148 | 研究树状分类，研究挂到类别 | LLM 生成 + 专家人在回路修正 | 2174 个层级、472 主题，100 主题专家修正；纠错模型分配 F1 +12.6 | 自述：类别生成尚可，研究归属（placement）最弱 |
| Knowledge Navigator（Findings EMNLP 2024） | arXiv 2408.15836 | 两级主题/子主题导航 | LLM + 聚类 | CLUSTREC-COVID、SCITOC | 浅层、静态 |
| SurveyG（2025/26） | arXiv 2510.07733 | 三层引文图：Foundation / Development / Frontier | 多 agent 横向（层内）+ 纵向（跨层）遍历生成综述 | 专家 + LLM-judge | 三层是粗分期，不是方法关系 |
| Multi-Dimensional Knowledge Profiling（2026） | arXiv 2601.15170 | 主题生命周期、方法转换、数据/模型使用、机构方向等多维画像 | 10 万+ 篇 22 个会议（2020–25），主题聚类 + LLM 解析 | 趋势案例 | 多维描述统计，无类型化演化边 |
| Ideation Space（2026） | arXiv 2601.08901 | 问题 / 方法 / 发现三个解耦表征子空间 | 各维对比学习 | Recall@30 0.329；"ideation transition"检索 HR@30 0.643 | 表征学习，不产出可读地图；但"按维度分解"思路与多维地图一致 |
| TL;DR Progress（EACL 2024 demo） | arXiv 2402.06913 | 文本摘要领域 514 篇的多面检索（指标、范式、挑战、数据集…） | 人工标注 | 系统演示 | 手工、单领域；ORKG 式 |

### D. LLM 时代：方法演化图 / 研究谱系 / 研究轨迹

| 工作 | 一手 ID/链接 | 表征什么 | 怎么建 | 怎么评 | 缺什么 |
|---|---|---|---|---|---|
| **Intern-Atlas**（2026） | arXiv 2604.28158（读 html 正文） | 方法演化图：论文节点 + 方法实体节点 + 317 万个库外 stub 节点；7 类边 extends / improves / replaces / adapts / uses_component / compares / background，前四类为"强因果"；每条因果边带 ρ(e)=(瓶颈, 机制, 取舍, 置信度)，前三项是引用论文的原文片段；瓶颈按 14 维分类 | 103 万篇 AI 论文全文 LLM 抽取；方法表从 247 个手工种子经 LLM 扩展；别名表 8155 个规范方法、9545 个别名（子串查找+大小写/标点归一+版本后缀合并）；确定性后检查：引文片段子串不匹配或违反年份顺序的边丢弃；SGT-MCTS 时间树搜索重建演化链（时间一致性函数峰值在 1–3 年） | 30 篇高影响综述构造 30 个专家演化图（2268 节点、1462 边、133 条链）：NMR 91.0%、ERR 89.7%、PSC 92.0%；链重建 vs Beam@10 大幅领先；idea 评估与 10 名博士打分 Spearman 0.81 | 自述：子组件层级方法边界模糊；范式转移处时间一致性假设失效；MCTS 预算漏罕见轨迹；瓶颈维度只有 14 个；会议/英文偏置。我判：边是二元的，"组合多个父方法"只能拆成多条 extends/uses_component；方法身份靠字符串别名；未见合并/分裂错误率报告；无增量更新机制描述 |
| **SciTraj**（2026） | arXiv 2606.22342（读 html 正文） | 论文级有向类型化引文图（32,559 篇 NLP/ML/CV，2015–2024；573,126 条边），6 类关系：Causal Extension / Limit Addressed / Future Realized / Dispute（claim 驱动）+ Direct Extension / Temporal Semantic（相似度驱动）；每条边配触发标签的 claim 句；约 2.879 亿条长度≥3 的类型化轨迹 | claim 驱动关系用规则词法模式抽取（未用 LLM），DeBERTa-v3-MNLI 验证 claim 被本地语境蕴含；相似度关系用摘要余弦+年差约束 | 3 人 520 项：Fleiss κ=0.74、多数票精度 79.9%；70 条轨迹 70% 判连贯；时间切分链接预测 AUC 0.914，打乱年份 AUC 掉 0.288 | 自述：只覆盖选定会议；部分关系类型边界难一致区分。我判：节点是论文不是方法；"Limit Addressed"只认被引论文的自述局限 |
| **ClaimFlow**（2026） | arXiv 2603.16073（读 html 正文） | claim 节点 + 跨论文 claim 关系 5 类：support / extension / qualification / refutation / background | 1617 篇 ACL 论文（1979–2025）人工标注 5689 claim、4871 关系；再训分类器扩到约 1.3 万篇 | 标注 α=0.75（claim）、κ=0.77（关系）；分类基线 macro-F1 0.81；发现 63.5% claim 从未被复用、11.1% 被质疑，广泛传播的 claim 更多被限定和扩展 | 自述：仅 ACL；只标摘要/引言/结论。我判：claim 层而非方法层，但"qualification"是方法词表里普遍缺的一类 |
| **Ideas Have Genomes / IG-Bench**（2026） | arXiv 2607.08758（读 html 正文） | Idea Genome 对象：角色类型 {niche, mechanism, observation, limitation, delta, claim} + 内容 + 证据指针；GenomeDiff 六类演化动力学：Mutation（机制继承/局部变异，niche 不变）、Adaptive Radiation（机制保留，进入新任务/领域）、Hybridization（从≥2 条谱系引入驱动对象）、Speciation（同 niche，机制被新机制替代）、Niche Competition（同 niche 无继承）、Isolation | 专家提名地标论文 → 引文+语义检索扩展 → LLM 辅助抽取 → 专家审计 → 程序化一致性检查；10 个学科 | 1961 条金标谱系、1085 个 Genome、920 条 Diff；IG-Exam 1029 题，最强 LLM 精确准确率仅 27.3% | 自述：六类是可审计的操作类别，不是完备科学发展理论；真实谱系常混合多种模式，标注只取主驱动。是基准不是建好的地图 |
| **Scientific Contribution Graph**（Findings EMNLP 2026） | arXiv 2605.15011（读 html 正文） | 贡献节点（16 种类型：问题形式化、理论洞见、数据集、模型、算法…）+ prerequisite 边（单一类型）；655k 篇 → 600 万贡献、3600 万边（html 正文另报 230k 篇/200 万节点的子集） | GPT-OSS-120B 全文抽贡献（平均 8.9/篇），再抽 prerequisite 候选并经引文对齐到被引论文的具体贡献节点；约 43 次 LLM 调用/篇 | 25 篇人工参照：贡献召回 91%；prerequisite 核验 93% 为核心；时间回测 prerequisite 预测 MAP 0.48（随机 0.12） | 自述：无规范合并（跨论文同一贡献各成节点）；仅开放获取；模型在训练截止前数据上好 0.02–0.06 MAP，有预训练泄漏。我判：只有"前置依赖"一种关系，没有扩展/替代之分 |
| **THE-Tree**（2025） | arXiv 2506.21763（读 html 正文） | 领域演化树：论文节点 + "因果依赖"边（哪篇的想法使后续成为可能） | MCTS 扩展 + Think-Verbalize-Cite-Verify：LLM 提出进展、引文落地、检索增强 NLI 验证父子逻辑 | 88 棵树经领域专家修订为金标；节点/边 P/R/F1；图补全 hit@1 比引文网络高 8–14%；未来预测 hit@1 约 +10% | 自述：英文同行评议文献；概念定义主观；年分辨率；计算昂贵。我判：边无类型 |
| **EvoNarrator**（ACL 2026） | ACL 2026.acl-long.544；github.com/xiyii-star/EvoNarrator | 从引文网络抽 P-M-L-F 四元组（Problem / Method / Limitation / Future Work）；三种宏观模式：Chain / Divergence / Convergence | LLM 抽取 + SocketMatch 判方法-问题兼容性 | 双盲专家 4.80/5；hindcasting；消融 | 演化表征服务于假设生成，未单独评地图本身；规模未在页面给出 |
| **Tree-of-Ideas / EvoTrace**（2026） | arXiv 2608.10740（读 html 正文） | 引文链上的分支轨迹；边注释 φ(v,u)=进展+缺口转移（addressed / narrowed / inherited / transformed）；节点注释 ψ(u)=残留或新生缺口，缺口类型 assumption_inherited / scope_unclosed / structural_unestablished / self_admitted | 引文对上的逐对 LLM 关系推断 | 未单独评 EvoTrace；去掉缺口注释后 idea 得分 6.27→5.55 | 缺口由 LLM 推断，无原文核验、无金标 |
| **Graphs of Research**（2026） | arXiv 2605.14790（读 html 正文） | 种子论文 2 跳参考文献的论文演化 DAG；边分 explicit_pred / parallel_pred / direct_to_seed | 引文位置（章节权重）、频次、前驱链接、发表时间四信号，严格时间锥 | 用于 SFT idea 生成；LLM-judge 锦标赛 + 5 名博士盲评 | 边类型是结构性的，无语义；中位子图 12 篇 |
| **SciPaths**（2026） | arXiv 2605.14600 | 发现路径：目标工作的使能贡献序列 + 角色 + 依赖理由 + 先前工作落地 | 262 条专家金标 + 2444 条银标（ML/NLP） | 严格语义匹配下最强 LLM 仅 0.189 F1；方法依赖最难 | 基准；说明"逆向追溯前置"远未解决 |
| **Sci-Reasoning**（2026） | arXiv 2601.04577 | NeurIPS/ICML/ICLR 2023–25 oral/spotlight 论文到关键前驱的推理链接；15 种思维模式（Gap-Driven Reframing 24.2%、Cross-Domain Synthesis 18.0%、Representation Shift 10.5%…） | LLM 加速 + 人工核验 | 描述统计 | 创新模式分类，不是领域地图 |
| **GIANTS**（2026） | arXiv 2604.09793 | 父论文集 → 下游论文核心洞见（insight anticipation） | GiantsBench 1.7 万例、8 个领域 | LM judge 相似度（与专家评分相关）；RL 训练 4B 模型相对提升 34% | 父论文集合是给定的，不建图 |
| Chain of Ideas（2024） | arXiv 2410.13185 | 按时间串起的文献链 | LLM agent | Idea Arena | 线性链，正是本题要避免的形态 |
| KnoVo（2025） | arXiv 2506.17508 | 目标论文与相关论文在 LLM 动态抽取的维度（方法、应用、数据集…）上逐对比较，生成新颖度分 | 摘要 + 多层引文网络 | 20 篇演示 | 规模小、无金标；"动态维度"有借鉴价值 |
| ResearchPulse（ACM MM 2025） | arXiv 2509.03565 | 动机-方法思维导图 + 跨论文实验折线图 | 3 个 7B agent，跨文档推理 + 时间对齐方法 | ResearchPulse-Bench，语义对齐/结构一致性 | 论文簇级，不是全领域 |
| PST-Bench（2024） | arXiv 2402.16009；github THUDM/paper-source-trace | 论文的直接"源头"论文（paper source tracing） | 专业研究者标注 | KDD Cup 2024 OAG-Challenge 任务 | 只有"源头"一种关系；可作演化边的二元金标补充 |
| AlgMap（KDD 2019） | DOI 10.1145/3292500.3330913；github.com/zhw12/AlgMap | 算法路线图：算法缩写为节点、比较关系为边 | 从 PDF 表格（弱监督）+ 文本缩写共现抽算法对 | NIPS/ACL/VLDB | 只有"被比较"关系；前 LLM 时代的方法级地图先例 |
| SciCo（AKBC 2021） | arXiv 2104.08809 | 层级跨文档共指：科学概念聚类 + 簇间层级 | 专家标注（比 ECB+ 大 3 倍） | H-CDCR 基线 | 不是演化，但它是"方法身份/粒度层级"问题的唯一专门基准 |

### E. 学术知识图谱 / 领域"世界状态"系统

| 工作 | 一手 ID/链接 | 表征什么 | 怎么建 | 怎么评 | 缺什么 |
|---|---|---|---|---|---|
| ORKG（2019） | arXiv 1901.10816 | 贡献级语义描述 + 跨论文比较表 | 众包 + 自动抽取 | 会议用户评估 | 人工为主、覆盖小；比较表无演化关系 |
| CS-KG（ISWC 2022） | DOI 10.1007/978-3-031-19433-7_39（读 PDF） | 1000 万实体（任务/方法/材料/指标）、179 种语义关系、4100 万 statement、3.5 亿三元组，来自 670 万篇 | 抽取管线 + 定期更新 | 人工标注 statement 基准 | 实体间关系是通用谓词（uses、improves 等从动词归一），不区分演化方向与时间 |
| SciER（EMNLP 2024） | ACL 2024.emnlp-main.726 | 数据集/方法/任务实体与细粒度关系 | 106 篇全文人工标注，2.4 万实体、1.2 万关系 | 抽取基线 | 篇内关系，不跨论文 |
| SciAtlas（2026，v2） | arXiv 2605.22878 | 证据/概念/学科/专长/规范五层共享 schema | 神经符号检索 | 轨迹重建、机会发现、创新评估三工作流案例 | 自述 Ongoing Work；构建细节与规模摘要未给 |
| Agents-K1（2026） | arXiv 2606.13669（读 html 正文） | 实体/claim/证据/机制/方法谱系节点；谱系关系 BUILDS_ON / EXTENDS / DERIVES_FROM / USES_COMPONENT / ALTERNATIVE_TO / DIFFERS_FROM / MOTIVATED_BY 等 | 246 万篇；GRPO 训练的 4B 抽取模型；稳定实体 ID 为连接键，规范名与别名作属性 | 抽取、KG 构建、多跳推理；谱系只经"比较基线检索"算子间接评 | 谱系关系未单独评；无时间一致性检查描述 |
| LKM v2（2026，投 ICLR 2027） | arXiv 2609.27297（读 html 正文） | 推理图：Question / Claim / Inference factor / Reasoning chain / Setting + highlight / weak point；关系 addresses / premise_of / concludes / subproblem_of / highlight_of / weakpoint_of | claim 身份 = SHA-256(类型+去语境化命题+规范参数)；语义相似度"从不合并身份"，只另成邻域层；新论文只新增或匹配对象，不改已有身份（增量） | 约 14 亿向量化 claim、近 100 万跨论文问题族；SciFact-Open 检索 818 vs 443 对；多个 QA 基准；100 篇审计幻觉率 0.97% | 自述：未测遗漏率；图表公式承载的证据抽不全。我判：没有方法演化关系，有"问题族"无"方法族" |
| ASKS（2026） | arXiv 2608.29612（读 html 正文） | 规范知识节点 + Hub（研究区域）+ Wiki 页；每次摄入是可检查的 GraphDelta 状态转移；Hub 分裂时记录父子谱系 | LLM 解读 → 确定性校验 → embedding+图规则合并；身份门：ID/别名 → 分词名匹配 → embedding 门（相似度≥0.90，联合≥0.91） | 单一课题组 56 篇按时间序编译；复用率、多源支持、成员 churn | 自述：只评"形成"，未评顺序鲁棒性与因果分解；分裂/合并/退役从未被触发 |
| Lacuna（ICML 2026） | arXiv 2606.26246 | ML 研究地图：摘要、概念元素、研究方向、拟议研究，节点回指源论文 | LLM 转换论文与元数据；Web/Markdown/MCP 三接口 | LitSearch R@10 0.538；ReportBench-ML 引用 F1 0.052 | 地图结构未作独立评测 |
| Typed Claim Network（2026） | arXiv 2605.30966 | 引用变成带 claim 文本与四类立场标签的类型化边 | 127 篇点云分割论文 → 8260 条类型化 claim | 检索增强、立场汇总、拓扑分析 vs RAG | 单子领域、小规模 |
| Time-Aligned Evolving Concept Graphs（2026） | arXiv 2609.18163 | 18.8 万篇、27 万概念、745 万共现链接的演化概念图；以发表日期为同步更新事件 | 语义状态与图结构按同一发表史重建 | 关系预测 AUROC 0.972；随图更新刷新语境使 AUPRC +16.6% | 关系=共现/形成，无方法语义；但"as-of 时间对齐"做法值得借 |
| Continuous Knowledge Metabolism（ICML 2026 AI4Research WS） | arXiv 2604.12243 | 滑窗持续演化的知识状态 | 滑动窗口文献"代谢" | 以未来论文为金标：50 主题中 36 个假设被验证，平均提前 404 天 | 面向假设生成 |
| PaperAtlas（2026） | arXiv 2609.28275 | 640 万摘要中 107 万计算类论文的算法/软件/Web 服务图谱，1000 簇 | schema 约束抽取 + 聚类 | 对照 bio.tools：61% 未在现有注册库严格匹配；端到端召回 56.6% | 只有工具实体与主题簇，无演化 |
| 技术融合监测（2025/26） | arXiv 2510.25370 | LLM 抽取的技术实体三元组图，检测技术融合 | 27.9 万 arXiv + 9793 USPTO 专利；"noun stapling"合并相近术语 | 案例 | 融合信号，不是谱系 |
| 动态 STI 知识图谱框架（2026） | arXiv 2607.21327 | 五层架构：数据骨干 / 版本化 KG / LLM 增强 / 多层校验 / 分析 | LLM 只产出临时增补，须过结构、证据、比较、专家闸门 | 摘要未给评测 | 框架文，无实证 |
| 概念重组几何信号（2026） | arXiv 2609.14917 | 用反事实消融测某概念对知识 embedding 几何的重组影响，检测"科学革命" | 文档 embedding | 5 个历史案例（狭义相对论、Higgs、深度学习、Transformer 注意力等） | 自述：文档归属与历史数据稀疏问题 |

### F. 增量维护 / 活综述

| 工作 | 一手 ID/链接 | 表征什么 | 怎么建 | 怎么评 | 缺什么 |
|---|---|---|---|---|---|
| GIST（SIGMOD 2027） | arXiv 2607.09149 | 随 arXiv 流持续维护的分类体系 | 从 Related Work 抽局部层级作为专家证据；box embedding 编码 is-a；新颖度感知 coreset 增量更新；假设概念生成器 | Node F1 +11.0%、Edge F1 +13.1%；运行时 9.6%、成本 12.7%（相对最强基线） | 只维护 is-a 树，不维护演化边 |
| DAS（2026） | arXiv 2608.18034 | 有状态综述 agent：文献/组织/写作/定稿四类显式状态，DAS-2M 元数据湖 | 论文→章节反向路由，只重激活受影响状态 | DAS-Bench 30 主题，总分 4.34 vs 4.03 | 状态是写作状态，不是领域结构状态 |
| Agentic Dynamic Survey（2026） | arXiv 2602.04071 | 综述作为活文档，增量并入新工作并尽量少扰动结构 | agentic | 回溯实验 | 同上 |
| Toward Living Narrative Reviews（CHI 2025） | arXiv 2502.00881 | 11 位研究者访谈：综述持续更新与学术激励错位 | 质性研究 | — | 提供需求证据，非系统 |

---

## 2. 综合：表征选择

把上表按"能不能同时表达多维和演化"归类：

1. 树（taxonomy / 综述目录）：TaxoAlign、TaxoBench、HiCaD、HiGTL、CHIME、Knowledge Navigator、GIST。只有父子关系，演化只能靠"新叶子出现"间接看到。EvoTree 往树里加了时间单调约束和"过渡论文挂内部节点"，是树形态里对演化最友好的，但仍表达不了合并。
2. 多维分类：TaxoAdapt（方法/任务/指标/数据集各一棵树）、Zhu et al.（multi-aspect）、KnoVo（动态比较维度）、Multi-Dimensional Profiling、Ideation Space（问题/方法/发现三子空间）。多维有了，但维度之间、时间之间都没有类型化边。
3. 主题/簇的时间演化图：Rosvall alluvial、Phylomemy、BERTilda、CiteSpace。分叉、合并、涌现、衰退这些"族级事件"只在这一派里被正式定义过，但节点是词簇或论文簇，事件没有方法语义，也没有证据句。
4. 论文级类型化引文图：SciTraj、ClaimFlow（claim 级）、Graphs of Research、THE-Tree、EvoTrace、Typed Claim Network。有类型、有时间，但节点是论文（或 claim），"方法 A 被 B 替代"要从论文边间接推。
5. 方法/贡献级演化图：Intern-Atlas（方法实体 + 7 类边 + 原文证据 + 瓶颈）、Scientific Contribution Graph（贡献节点 + prerequisite）、Agents-K1（方法谱系关系）、AlgMap（比较关系）。这是最接近"领域地图"的一派。其中只有 Intern-Atlas 同时做了类型化边、逐边原文证据、年份顺序检查和综述金标评测。
6. 世界状态/知识库：LKM、ASKS、SciAtlas、Lacuna、CS-KG、ORKG。重点在身份、增量、可追溯；LKM 和 ASKS 是仅有的两个把"身份规则 + 增量不改写"写成机制的系统，但 LKM 无方法演化，ASKS 的族分裂/合并从未被触发。

同时做到"多维 + 显式演化"的工作目前没有。最接近的两种组合：一是 Intern-Atlas 的边上挂 14 维瓶颈标签（演化 + 边属性维度）；二是 IG-Bench 的 niche/mechanism 二分（概念上把"任务生态位"和"机制"拆成两个维度，并用它定义演化类型：同机制换生态位 = adaptive radiation，同生态位换机制 = speciation）。后者只是 920 条标注的基准，没有人用它建出全领域地图。

节点定义与身份处理：论文节点（SciTraj、THE-Tree、EvoTree、GoR、EvoTrace）回避了身份问题；贡献节点（SCG）明说不合并；方法节点（Intern-Atlas）用种子表 + 字符串别名；claim 节点（LKM、ClaimFlow）用规范化命题文本，LKM 明确"语义相似不合并身份"；ASKS 用 ID→别名→embedding 门三级。没有任何一篇报告方法身份合并的错误率（误合并/误拆分），SciCo 是唯一专门评测概念共指 + 层级的基准，且是 2021 年前 LLM 时代的。

构建来源：纯引文图（MPA、CiteSpace、Rosvall、EvoTree、GoR、HiGTL）；引文语境（ACL-ARC、SciCite、MultiCite、ClaimFlow、SciTraj 的 claim 句）；Related Work（GIST 从中抽局部层级）；综述（TaxoAlign、TaxoBench、Zhu et al. 用作金标；Intern-Atlas 用 30 篇综述做金标）；全文 LLM 抽取（Intern-Atlas、SCG、Agents-K1、LKM、ResearchPulse）；仅摘要（KnoVo、Eliot、SciTraj 的相似度关系）。

评测来源：综述派生金标（Intern-Atlas、TaxoAlign、TaxoBench、Zhu、HiCaD、EvoTree 弱监督）；专家标注（THE-Tree 88 树、IG-Bench 1961 谱系、SciPaths 262 路径、ClaimFlow 1617 篇、CHIME 100 主题、PST-Bench）；时间回测（SCG MAP、SciTraj 链接预测 + 打乱年份证伪、THE-Tree 未来预测、时间对齐概念图、AUGUR、Topical Phase Transitions）；下游 idea 生成的 LLM-judge（CoI、GoR、ToI、EvoNarrator、GIANTS）；无金标过程指标（ASKS churn/复用）。各基准的最强模型分数都很低：IG-Bench 精确准确率 27.3%、SciPaths 0.189 F1、TaxoBench 召回约 21%，说明"重建谱系"本身远未解决，Intern-Atlas 的 90% 级分数是在"综述方法能否在大图里找到可达路径"这种较宽松定义下得到的。

## 3. 综合：演化关系词表

各家词表并排（只列与演化相关的类）：

| 来源 | 层级 | 关系类 |
|---|---|---|
| Teufel 2006 | 引用 | PBas（以之为基础）、PUse、PModi（修改）、PMot、PSim、PSup、CoCo*（对比，含 CoCoGM 目标/方法对比、CoCoRo 结果对比）、Weak（指出弱点）、Neut |
| ACL-ARC / Jurgens 2018 | 引用 | Background、Motivation、Uses、Extension、CompareOrContrast、Future |
| MultiCite 2022 | 引用 | Background、Motivation、Future、Similar、Difference、Uses、Extension |
| Intern-Atlas 2026 | 方法 | extends、improves、replaces、adapts、uses_component、compares、background；边属性：瓶颈/机制/取舍原文 + 14 维瓶颈 |
| Agents-K1 2026 | 方法 | BUILDS_ON、EXTENDS、DERIVES_FROM、USES_COMPONENT、ALTERNATIVE_TO、DIFFERS_FROM、MOTIVATED_BY |
| SciTraj 2026 | 论文 | Causal Extension、Limit Addressed、Future Realized、Dispute、Direct Extension、Temporal Semantic |
| ClaimFlow 2026 | claim | support、extension、qualification、refutation、background |
| IG-Bench 2026 | 谱系对 | Mutation、Adaptive Radiation、Hybridization、Speciation、Niche Competition、Isolation |
| EvoNarrator 2026 | 宏观 | Chain、Divergence、Convergence（基于 P-M-L-F） |
| EvoTrace 2026 | 边上缺口 | addressed、narrowed、inherited、transformed |
| SCG 2026 | 贡献 | prerequisite（单类） |
| Phylomemy / Rosvall / BERTilda | 族/簇 | emergence、continuation、branching/split、merging、decline/disappearance |

收敛点：

- "在原方法上加东西"（Extension / extends / EXTENDS / Mutation / PBas+PModi）各家都有，是最稳的一类。
- "同一问题换掉核心机制"（replaces / Speciation / ALTERNATIVE_TO）和"同一机制挪到新任务"（adapts / Adaptive Radiation / Extension 中的"新设定"）被 Intern-Atlas 与 IG-Bench 独立提出，且定义可互相对齐。
- "用作组件"（Uses / uses_component / USES_COMPONENT / PUse）是稳定的一类，并且大家都把它和"演化"分开。
- "比较/竞争"（CompareOrContrast / compares / Niche Competition / DIFFERS_FROM）也稳定，属于非继承关系。
- "触发原因"是第二个维度，不是第一维的又一个类：SciTraj 把 Limit Addressed、Future Realized 做成关系类，Intern-Atlas 把瓶颈做成边属性，EvoTrace 把缺口转移做成边注释，EvoNarrator 把 L、F 做成四元组槽位。SOFT 和 UniCite 在引用层也主张"意图"和"被引内容"正交拆维。
- "组合多个父方法"只在 IG-Bench（Hybridization）、EvoNarrator（Convergence）、Sci-Reasoning（Cross-Domain Synthesis 占 18%）里作为一等概念出现；Intern-Atlas 和 Agents-K1 都是二元边，只能拆成多条边，丢掉"这几个父方法是一起被组合的"这一信息。
- 族级事件（分叉、合并、涌现、衰退）只在主题演化派里有定义，和方法级边词表没有连接。

建议采用的词表（证据最充分的组合，属于我的综合，不是任何一篇的原样）：

1. 主关系（方法→方法，有方向，带时间先后）：extends（加能力/组件）、improves（同一形式化在某一维变好）、replaces（同一问题，承重组件被不同机制替代）、adapts（同一机制进入新任务/领域/模态）、combines（n 元：从≥2 个父方法引入驱动组件，对应 Hybridization/Convergence）、uses_component（作为非核心模块复用）、competes（同一问题的并行替代方案，无继承）。前五类为"谱系边"，后两类不进谱系。依据：Intern-Atlas 七类 + IG-Bench 六类对齐 + ACL-ARC/MultiCite 的 Extension/Uses/Compare 的长期一致性证据。background 不必建边。
2. 触发属性（正交维度，挂在谱系边上）：addresses_limitation（区分被引方自述局限与引用方指认的局限，EvoTrace 的 self_admitted 已有这个区分）、realizes_future_work、responds_to_dispute/qualification（来自 ClaimFlow 和 SciTraj 的 Dispute）、无明确触发。每个触发值都要有原文片段（Intern-Atlas 的子串校验做法）。
3. 维度属性：每条谱系边记录"变的是哪一维"（机制 / 任务 / 数据 / 评测 / 效率等），对应 IG-Bench 的 niche-vs-mechanism 拆分和 Intern-Atlas 的 14 维瓶颈。adapts 与 replaces 的区别本身就是"变的是任务维还是机制维"。
4. 族级事件（由边和族成员随时间派生，不直接标注）：emerge、continue、split、merge、decline，定义沿用 phylomemy/Rosvall/BERTilda 的拓扑判据，但族成员是方法实体而非词簇。

一致性风险：细粒度演化类型的标注一致性未经验证。已有的一致性数字都来自较粗的词表（Teufel 12 类 κ=0.72、SciTraj 6 类 κ=0.74、ClaimFlow 5 类 κ=0.77），Intern-Atlas 的 7 类我没有看到逐类一致性或逐类精度报告；IG-Bench 的六类上 LLM 只有 27.3% 精确准确率。上面这套词表落地前需要先做小规模双人标注测一致性，不行就合并 improves 进 extends、competes 进 replaces 的"非继承"侧。

## 4. 开放问题（逐条对应主表证据）

1. 方法身份与粒度没人认真评过。Intern-Atlas 靠种子表 + 子串别名，自述子组件边界模糊；SCG 明说不做规范合并；ASKS 用固定 embedding 阈值；LKM 刻意不按语义相似合并。没有一篇报告误合并/误拆分率，也没有人把 SciCo 式的"概念层级共指"用于方法演化图。方法的"变体是同一节点还是子节点"（如 BERT → RoBERTa 是 improves 边还是同族子节点）没有统一规则。
2. 多维与演化没有同时做到。多维派（TaxoAdapt、Zhu、KnoVo、Profiling）没有类型化演化边；演化派（Intern-Atlas、SciTraj、SCG）基本是单维（方法）或论文级。IG-Bench 的 niche/mechanism 拆分给出了把"维度"用来定义演化类型的思路，但只停在 920 条基准标注。
3. 非树、非二元结构缺位。分类树（TaxoAlign、TaxoBench、CHIME、EvoTree）无法表达合并，CHIME 自述研究归属最弱，EvoTree 只能把过渡论文挂到内部节点。方法级图都是二元边，组合（Hybridization 一类）只能拆边。族级分叉/合并只在主题簇层有定义，ASKS 的族分裂/合并在实测中从未触发。
4. 时间一致性只有最低限度的检查。Intern-Atlas 只保证年份先后，自述在范式转移处失效；EvoTree 自述预印本与正式发表日期混淆；THE-Tree 只有年分辨率；SCG 发现预训练泄漏（截止前数据好 0.02–0.06 MAP）。SciTraj 的打乱年份证伪实验是唯一把"时间是否真起作用"做成检验的。"在时间 t 时领域看起来什么样"（as-of 快照）只有时间对齐概念图（2609.18163）和滑窗代谢（2604.12243）在预测任务里做过，没有人把它用于方法演化地图。
5. 边证据强弱不一，"缺口/局限"类信息几乎都没核验。逐边原文子串校验只有 Intern-Atlas 做了；SciTraj 用 NLI 验证 claim 被本地语境蕴含；ClaimFlow 用引文语境；EvoTrace 的缺口全靠 LLM 推断，未单独评测；SCG 的边只有自然语言解释。"A 的局限被 B 解决"这件事：SciTraj 只认 A 的自述局限，EvoTrace 区分了自述与他述但没核验，没有人检验被指认的局限是否属实、被声称解决的局限是否真被解决。
6. 增量更新与演化边是两条没交汇的线。LKM（身份哈希、只增不改）、ASKS（GraphDelta）、GIST（分类树增量，Node/Edge F1 有金标）、DAS（写作状态）做了增量；Intern-Atlas、SciTraj、SCG 都是批量构建。没有人评过演化边在增量摄入下的顺序鲁棒性（ASKS 自述明确排除了这一项），也没有人处理"新论文到来后旧边类型要改"（如某方法后来被证明只是 improves 而非 replaces）。
7. 金标的来源偏窄且有继承偏差。综述派生金标（Intern-Atlas 30 篇、TaxoBench 72 篇、TaxoAlign 540 个）继承了综述作者的组织方式；EvoTree 自述以综述章节作弱监督。领域高度集中在 AI/NLP（Intern-Atlas、SciTraj、ClaimFlow、EvoTree、TaxoBench、SciPaths），跨学科的只有 IG-Bench（10 学科）、SCG、LKM。可直接拿来评"方法演化地图"的现成金标：Intern-Atlas 的 30 个综述演化图（若公开）、IG-Bench 的谱系与 GenomeDiff、THE-Tree 88 棵树、SciPaths 262 条路径、PST-Bench 源头标注、EvoTree 11 棵树。
8. 族（family）的定义没有落到方法语义上。主题演化派的族是词簇或引文簇，方法演化派没有族概念（Intern-Atlas 有链无族），LKM 有"问题族"但无"方法族"。"一个方法族的边界、族内主干、族间迁移"这一层，目前没有工作同时给出定义、构建和评测。

对本项目的直接含义（简述）：主关系词表沿用 Intern-Atlas 七类可直接对齐最强竞品，但应补 n 元 combines、正交的触发维度和"变化维度"属性；身份层可借 LKM 的"精确身份 + 语义邻域分层"与 ASKS 的分级身份门，并补上误合并率评测；族层用方法实体重做 phylomemy 式的分叉/合并判据；评测优先用 IG-Bench 与 Intern-Atlas 的综述演化图，并加 SciTraj 式打乱年份证伪。真正空着的位置是第 2、3、6、8 条的组合：方法级、多维、带族级事件、可增量更新且逐边有证据的领域地图。

## 5. 未核实条目（仅作线索，不进主表）

- PI-Embedding: Scientific Idea Representation Learning via Citation Intent and Paper Provenance：搜索结果给出 arXiv 2603.27435，但该 ID 实际对应 "Improving Attributed Long-form Question Answering with Intent Awareness"（ICLR 2026），ID 与标题不符，未核实。
- Metadata Meets LLMs: Constructing Knowledge-Rich Citation Networks with CoT-Enhanced Representations（仅 exa.ai 条目）。
- LLM-Enabled Scientific Knowledge Diffusion Analysis（仅 exa.ai 条目）。
- LLMs in Citation Intent Classification: Progress, Precision, and Reproducibility Challenges（仅 exa.ai 条目）。
- Identifying Meaningful Citations（Valenzuela, Ha & Etzioni, AAAI-15 Workshop）：只见 researchr 等二手书目页。
- An Entity-Based Main Path Analysis Method（Atlantis Press 2023/24）、A multi-entity reinforced main path analysis（J. Informetrics 2024，仅 RePEc 线索）。
- Mapping the evolution of AI: an LLM-driven patent network analysis（Scientometrics 2026，未打开）。
- Research on the evolutionary trajectories of intelligent energy technology based on MPA and SAO semantic analysis（Sci. Rep. 2026）：Crossref 有该 DOI 10.1038/s41598-026-64126-2，内容未读。
- Exploiting Contextual Embeddings to Extract Topic Genealogy from Scientific Literature（polito 仓储 PDF，未打开）。
- SurveyG 的 ICML 2026 录用信息（只见 icml.cc 检索结果，未打开核对；arXiv 本身已核实）。
- CiteSpace、MPA、burst 的方法内容细节按领域通识描述，只核实了书目（Crossref），未重读原文。
