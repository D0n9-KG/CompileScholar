# 智能体侧查新与下游评测资源（2026-10-04）

范围：一是"领域认知"在智能体侧怎么评测，二是有没有人已经把"带时间的领域认知"作为工具交给智能体用。时间覆盖到 2026-10。

方法：发现层用 CDP 真实 Google 搜索、arXiv 站内检索、HF/GitHub 站内页面；验证层逐条打开 arXiv abs（标题、日期、版本、comments），关键结论再到 arxiv.org/html 全文里 grep 原文，数据与许可查 HF `/api/datasets` 和 GitHub 仓库页。下表每一行都打开过一手页面。前几轮已列过的（Intern-Atlas、SciTraj、TaxoBench、ClaimFlow 等地图类工作）这里不再重复，只补智能体评测这一侧。

## 1. 结论先行

1. **"截止时间对齐的领域判断"基准在 2026 年已经形成一小簇**：ForeSci、RAP、IdeaForecastBench、Proof of Time、PreScience、BackTrend、ScholarCatalyst、INSPIRE、CKM、HindSight。协议做法收敛为三件：冻结截止前语料、截止后论文当金标、只在截止后窗口报告"真预测"分数。但它们**评的都是"预测/检索"，没有一个直接问"截至 T，领域对方法 X 怎么看"**。这一格在我的检索范围内没有找到（这只说明没搜到，不能证明不存在）。
2. **对我们动机最有力的外部证据来自 RAP（2609.10092）**。原文："recovering and preserving a time-local state is a major bottleneck"。在累积历史检索下，State carry-forward 对四个模型都优于直接 Forecast。IdeaForecastBench 也报告 Summary（一段领域级压缩）在四个 backbone 上都优于 Direct。这两条说明"给智能体一个截至 T 的领域状态"有实测收益，但它们给的都是临时压缩，不是持久结构。
3. **"同一知识层接到多种 harness 上做配对评测"的方法论已经有人做过**：SkillsBench 在 OpenHands、Claude Code、Gemini CLI、Codex CLI 上对 18 个模型-harness 配置做有/无 Skills 配对评测，+16.6pp，并发现"harness 的选择会实质改变同一模型怎么用 Skills"。所以"即插即用评测"本身不能当贡献，只能当实验设计，而且必须引用它。科学文献这边，LKM、ScholarStack、EMBL AI Librarian、Lacuna、Agents-K1 都把知识层做成 API/CLI/MCP，但都没有把**同一个**知识层接到多个 harness 上配对评测，也都没有时间维度。
4. **时间泄漏的实证已经很充分，可以直接引用**：搜索引擎日期过滤有 71%/81% 的题目泄漏（2602.00758，ACL 2026）；用提示词"假装不知道"做不到，SI 与 TI 差 52%（2601.13717）；工具级时间屏蔽之后仍有约 10 倍于内容级检测的残留泄漏（OracleProto）。这是"带 as-of 的离线知识层"在方法上的必要性论据。
5. **能马上拿来用、金标客观、带时间的下游**：PreScience（ODC-BY）、Proof of Time（MIT）、IdeaForecastBench（代码 MIT，语料按 arXiv 逐篇许可）。最贴合但还拿不到数据的是 RAP 和 ForeSci（两者都写明发表后才放出）。

## 2. 判决表

图例：时间相关 = 有显式截止 T 且金标来自 T 之后；"威胁"指与"给智能体提供带时间的领域认知 + 即插即用评测"这一主张的重合度。

### 2.1 带截止时间的领域判断/预测基准

| 工作 | 一手链接 | 测什么/做什么 | 时间相关 | 数据公开与许可 | 金标来源 | 适合作我们下游吗 | 与我们的关系 |
|---|---|---|---|---|---|---|---|
| **ForeSci** | [2606.00644](https://arxiv.org/abs/2606.00644) v3 2026-09-26 | 500 题、4 个 AI 领域、4 类决策：方向预测、瓶颈-机会发现、战略规划、按会场定位。每题配截止对齐的离线 KB，禁用 web 检索；比较 native LLM、Hybrid RAG 与三种研究智能体 | 是（截止对齐 KB；截止后论文只用于验证） | 未公开。原文："Code and supplementary materials will be released in anonymized form for review and made publicly available upon publication"。论文 CC BY 4.0 | 截止后论文派生的隐藏目标（题目由截止前的 taxonomy 分支与方法演化信号构造） | **最贴题但现在拿不到**。离线 KB 就是"接入点"：用我们的 as-of 状态替换或增强它的 KB 即可。发现的"evidence-decision decoupling"（引用对了、预测对象错了）正是领域结构可能帮上忙的地方 | **半真，中**。KB 构建里有"temporal taxonomy induction、evidence and evolution asset construction"，但这些结构只用来出题，原文："public questions do not expose internal taxonomy information"。没有把结构作为工具交给智能体 |
| **RAP**（Research Attention Prediction） | [2609.10092](https://arxiv.org/abs/2609.10092) 2026-09-09 | 278 个 AI/ML 领域、1,390 个 episode；智能体检索时间受限的 arXiv 语料，预测未来 6 个月 8 个冻结方向的论文占比 | 是（滚动截止；Forecast-Strict 只用模型文档截止之后的窗口） | 未公开。原文："We plan to release the benchmark data and evaluation code upon acceptance" | 截止后真实论文计数（方向归属由冻结的抽取器完成，有人工审计） | **概念上最适合，等数据放出**。它的 State 条件和 Oracle historical-state 消融就是"给不给智能体截至 T 的领域状态"的现成对照位 | **半真，低**（对我们是支持证据）。所有模型都输给 EWMA；给了精确历史状态后残差方向对齐转正，但仍很难超过 EWMA。原文："recovering and preserving a time-local state is a major bottleneck"。State 是一次查询，不是持久结构 |
| **IdeaForecastBench**（EMNLP 2026） | [2609.00747](https://arxiv.org/abs/2609.00747)；代码 [social-world-model/idea-forecast-bench](https://github.com/social-world-model/idea-forecast-bench)；语料 [HF 4R5T/idea-forecast-bench](https://huggingface.co/datasets/4R5T/idea-forecast-bench) | 给定截止前某社区文献，输出至多 5 个排序 idea，与之后论文匹配；624 个滚动 episode、52 个主题、12 个月度截止 | 是 | 代码 MIT；语料 108,768 篇 cs 论文**全文** markdown（MinerU 转换），license 为 arxiv-per-paper，非 gated | 截止后论文，用 retrieve-then-judge（LLM judge）判匹配；原文自己报告了 judge 与阈值敏感性 | **适合（推荐 2）**。基准本身就在比较"历史表示"：Direct、Retrieval、Summary、Topic Trend、Memory，外加学习型 MDF。把我们的 as-of 状态当第六种表示接进去，是天然的即插即用位。全文语料与我们的管线一致 | **半真，中低**。已测"领域级压缩表示"（Summary 在四个 backbone 上都优于 Direct；Topic Trend = 簇级轨迹），审稿人可能拿来说"领域摘要当上下文已经有人测过"。但都是每次查询现做的压缩，没有方法族、演化、他述 |
| **Proof of Time (PoT)** | [2601.07606](https://arxiv.org/abs/2601.07606)；代码 [shan23chen/proof_of_time](https://github.com/shan23chen/proof_of_time)；数据 [HF AIM-Harvard/proof-of-time](https://huggingface.co/datasets/AIM-Harvard/proof-of-time) | 冻结截止前证据，放进离线 sandbox，预测截止后结果。四个领域：影响力（引用）、科学价值（获奖）、研究演化（教授未来方向）、技术前沿（SOTA 轨迹）；比较用工具的智能体与非智能体基线；共 30K+ 实例 | 是 | 代码 MIT；HF 数据 MIT，非 gated（任务 3.8 MB + sandbox 66 MB）；基于 Inspect AI | 截止后可观测信号：引用、获奖、教授后续论文、SOTA 数值。客观，但部分是代理指标（引用≠价值） | **适合（推荐 3）**。sandbox 就是现成的接入点，Inspect 框架与 AstaBench 同源，挂 MCP 成本低。"研究演化"与"技术前沿"两族最贴近领域认知；获奖/引用两族与领域结构关系弱 | **半真，低**。Faculty 族已在试"智能体能否从截止前出版物的结构化中间产物受益"（关键词与领域描述），但规模很小。结论是工具收益"strongly task-dependent" |
| **PreScience** | [2602.20459](https://arxiv.org/abs/2602.20459)；数据 [HF allenai/prescience](https://huggingface.co/datasets/allenai/prescience)；代码 [allenai/prescience](https://github.com/allenai/prescience) | 98K 篇目标论文（共 502K）上的七个任务：贡献生成、合作者预测、**前置工作选择**、引用数预测、**未来组合预测**、两种**主题趋势预测**；有工具型 GPT-5 智能体基线 | 是（元数据按截止快照，防泄漏） | 数据 ODC-BY，非 gated；代码 Apache-2.0。只有标题+摘要，无全文 | 真实论文的 key references、作者、引用、主题标签。客观，按 ID 精确匹配 | **适合（推荐 1）**。"前置工作选择"直接对应"已知但未读的论文"与谱系；"未来组合预测"对应演化；金标是精确 ID，不需要 LLM judge。代价：只有 AI 领域；测试窗口 2024-10 到 2025-10 与现役模型的训练截止重叠，必须分层报告 | **假（威胁）**，是资源 |
| **BackTrend**（EMNLP 2026 Findings） | [2609.24921](https://arxiv.org/abs/2609.24921)；数据 [HF rebeccazzzz/BackTrend](https://huggingface.co/datasets/rebeccazzzz/BackTrend) | 给定成熟主题和时间证据约束，找回早期的"弱信号"前驱（问题侧/方案侧）；最强系统 F1 10.1% | 是（2019–2023 证据、2024 采纳验证） | CC BY 4.0，非 gated；66 个信号、832 篇 grounding 论文 | 发文频率轨迹 + 四道闸门 + 专家筛选。半客观；匹配用 LLM judge | **可用，小**。最贴"对认识变迁的识别"，但只有 66 条，统计力弱。论文写 25 个主题，HF 卡写 18 个父主题，口径未核 | **假** |
| **ScholarCatalyst** | [2610.02202](https://arxiv.org/abs/2610.02202) 2026-10-01；数据 [HF ScholarCatalyst/ScholarCatalyst](https://huggingface.co/datasets/ScholarCatalyst/ScholarCatalyst)；代码 stanford-iris-lab/ScholarCatalyst | 184 位一作为 207 篇论文标注"哪些候选推进了/本可推进其项目"；只检索项目开始时已有的文献 | 是（截止 = 项目开始时） | CC-BY-NC-4.0；corpus 190,896、queries 894（core 207 + subfield 687）；代码许可 NOASSERTION | **作者本人标注 + 理由**，金标质量高 | **适合（推荐 4）**。测"领域里该读哪些前人工作"，金标不靠综述。原文："Agentic search does no better than embedding retrieval (0.42 vs. 0.48 Recall@20)"；Claude Fable 5.1 可能见过完成后的论文，也只到 0.51 | **假** |
| **INSPIRE** | [2609.33233](https://arxiv.org/abs/2609.33233) 2026-09-27 | 去掉解法的研究简报，截止为目标论文前 3 个月，按目标论文真实引用谱系分级（T3/T2/T1）评检索；476 个评测目标；用轨迹拆出"暴露/选择/排序"三段 | 是 | 文中有 release manifest，但没找到公开链接 | 目标论文真实被引前驱（自动分级） | 可用，但数据未找到 | **假** |
| **CKM**（ICML 2026 AI4Research WS） | [2604.12243](https://arxiv.org/abs/2604.12243)；代码 [TaoJinkai/ckm-hypogen](https://github.com/TaoJinkai/ckm-hypogen) | 滑动窗口维护演化的知识状态并生成假设；以后来的论文验证；50 个主题，72% vs 一次性基线 30%，平均提前 404 天 | 是 | 代码 MIT，含主题表、运行日志、judge 代码 | 截止后论文 + LLM 判匹配 | 协议可借，规模小 | **半真，中**。"持续演化的知识状态"是同一直觉，但被动摄入，只服务假设生成，不是可查询工具 |
| **HindSight** | [2603.15164](https://arxiv.org/abs/2603.15164) | 时间切分的 idea 评测：截止 T 前生成，与之后 30 个月论文匹配，按引用/会场打分；LLM-judge 判不出检索增强的差异（p=0.584），HindSight 判出 2.5 倍 | 是 | 未找到代码或数据 | 未来论文 + 影响力 | 证据可引，数据未找到 | **假**。可引用的一点："LLM 判的新颖性"与"未来影响力"负相关（ρ=−0.29） |
| **Scientific Contribution Graph**（EMNLP 2026 Findings） | [2605.15011](https://arxiv.org/abs/2605.15011)；[cognitiveailab/scientific-contribution-graph](https://github.com/cognitiveailab/scientific-contribution-graph) | 前置依赖预测，时间过滤回测，MAP 0.48 | 是 | 仓库 Apache-2.0（含 API、数据） | 后来论文的真实前置依赖边 | 可作组件级评测；不是智能体任务 | 已在前轮列过（半真）；它自报截止前数据有预训练泄漏（0.02–0.06 MAP） |
| **SciNet**（ICML 2026） | [2601.03260](https://arxiv.org/abs/2601.03260)；[tsinghua-fib-lab/SciNet](https://github.com/tsinghua-fib-lab/SciNet) | 关系感知检索：ego（新颖性/颠覆性）、pair（引文情感、共同提及）、path（谱系路径重建）；8,940 题、7 个学科 | 否（无截止） | README 写 MIT。Task3 每学科一个 JSON，AI 文件 1,823 题，只有问题文本（例："most influential citation path from X to Y"）；评测脚本有 path-connectivity 与 path-llm | 引文网络与科学计量指标（可计算），路径质量部分靠 LLM | 演化链的补充评测；依赖其 2.69 亿篇元数据库 | **半真，低**。原文："agents empowered with SciNet achieve a 25.3% improvement in review quality"，是"关系数据层接到智能体上"的一个先例，但没有时间维 |
| **IG-Bench** | [2607.08758](https://arxiv.org/abs/2607.08758)；[VisionXLab/IdeasHaveGenomes](https://github.com/VisionXLab/IdeasHaveGenomes) | 谱系推理 IG-Exam（1,029 题、42 类）+ 谱系条件生成 IG-Arena | 部分（谱系有先后，没有截止协议） | README 写 "License TBD"；没找到 HF 数据集（README 的 Hugging Face 链接只指向论文页） | 专家审计谱系 | 组件评测可用，许可不明 | 前轮已列（半真，中高） |
| **DeepScholar-Bench** | [2508.20033](https://arxiv.org/abs/2508.20033)；[HF deepscholar-bench/DeepScholarBench](https://huggingface.co/datasets/deepscholar-bench/DeepScholarBench)；仓库已迁到 guestrin-lab/deepscholar | live related-work 生成；HF 快照 63 篇、1,630 条引文 | 弱（取新近 arXiv v1 以避开污染，没有显式 as-of 检索约束） | MIT | 人写 related work 与引文 + LLM nugget 判分 | 已在用（DSB）；不测领域时间认知 | **假** |
| **AstaBench**（ICLR 2026） | [2510.21652](https://arxiv.org/abs/2510.21652)；[allenai/asta-bench](https://github.com/allenai/asta-bench) | 2,400+ 题科研智能体套件；文献任务含 PaperFindingBench、ScholarQABench2（即我们的 CS2）、LitQA2-FT、ArxivDIGESTables | 是（工具层）。README："Information access is limited to date-restricted usage of the Asta MCP tools" | 代码 Apache-2.0；HF 数据 gated:auto，cardData 无 license 字段 | 各子任务 rubric/金标 | **作为评测框架最合适**：标准 MCP 工具 + 日期限制 + Inspect，可以把我们的工具和 Asta 工具并列挂上做配对评测 | **假** |

### 2.2 新颖性/定位/idea 类基准（与"对每个工作的定位"相关）

| 工作 | 一手链接 | 测什么 | 时间相关 | 数据与许可 | 金标 | 适合度 | 关系 |
|---|---|---|---|---|---|---|---|
| RWGBench | [2606.24894](https://arxiv.org/abs/2606.24894)；[BFTree/RWGBench](https://github.com/BFTree/RWGBench) | 把 related work 当"引文级学术定位"来评：选引、语境、组织、话语结构；100 篇测试 + 1.09M 检索库 | 否 | MIT | 已发表 related work | 与 DSB 同类，可测"定位"；不测时间 | 假 |
| NovGauge | [2609.11234](https://arxiv.org/abs/2609.11234) | 新颖性分维度诊断（task/problem/method），619 对 + 50 组；带证据忠实性核验 | 否 | 未核 | ICLR 审稿人重叠声明 + 综述共被引 | 可测"差异化定位"，数据未核 | 假 |
| SchNovel | [2409.16605](https://arxiv.org/abs/2409.16605) | 15,000 对论文，相隔 2–10 年，假设较新者更新颖 | 是（只用发表先后） | 未核 | 发表日期（假设很强） | 弱 | 假 |
| RQ-Bench | [2606.12071](https://arxiv.org/abs/2606.12071) | 研究问题新颖性：LLM judge 出现"novelty mirage"，专家结论相反 | 否 | 未核 | 作者锚定 RQ + 专家 | 只作"不能用 LLM judge 判新颖"的证据 | 假 |
| Old Ideas, Novel Problems | [2610.02022](https://arxiv.org/abs/2610.02022) 2026-10-01 | LLM 新颖性 judge 不稳定：一句提示可改变过半判决，成对准确率变化超过 50 点 | 否 | 未核 | OpenReview 一致意见 | 同上，方法论警示 | 假 |
| Reconstruction | [2608.16645](https://arxiv.org/abs/2608.16645) | 只给发表前参考文献，复原论文的研究 idea；643 篇、6 个领域 | 是（时间引文截止 + 匿名引用 ID） | 未找到 | 原论文 idea + LLM judge | 弱 | 假 |
| AgentIdeaBench | [2609.07611](https://arxiv.org/abs/2609.07611)；HKUST-KnowComp/AgentIdeaBench | 静态观察 vs 主动探索两种设置下的 ideation；33 个模型、40 个子领域 | 否 | 仓库 NOASSERTION | 文献核验的 critic 打分 | 低 | 假 |
| FIRE-Bench（ICML 2026） | [2602.02905](https://arxiv.org/abs/2602.02905) | 全流程重发现已知结论 | 否 | 未核 | 已发表结论 | 不对口（实验执行） | 假 |
| 比较式成功预测 | [2605.21491](https://arxiv.org/abs/2605.21491)（ACL 2026 Findings） | 两个 idea 中谁在基准上更好；11,488 对；含跨域时间切分测试集 | 部分 | 未核 | PapersWithCode 客观结果 | 低 | 假 |

### 2.3 LLM 在科学知识上的时间问题

| 工作 | 一手链接 | 发现 | 对我们的用处 |
|---|---|---|---|
| Temporal Leakage in Date-Filtered Retrieval（ACL 2026） | [2602.00758](https://arxiv.org/abs/2602.00758) | Google `before:` 有 71%、DDG 有 81% 的题目至少检出一页重大泄漏；直接泄露答案的分别为 41%/55%；Brier 0.10 vs 无泄漏 0.24；建议用冻结的带时间戳快照 | "截止过滤靠搜索引擎不可信"的直接证据；支持离线 as-of 知识层 |
| Simulated Ignorance Fails | [2601.13717](https://arxiv.org/abs/2601.13717) | 用提示词要求模型压住截止后知识：SI 与 TI 差 52%；CoT 压不住；推理型模型更差 | 支持"不能靠提示让模型回到 T"，需要外部按时间裁剪的知识 |
| ExAnte | [2505.19533](https://arxiv.org/abs/2505.19533) | 含"科学论文预测"子任务；显式截止提示下 LLM 仍持续泄漏 | 同上，覆盖科学域 |
| OracleProto | [2605.03762](https://arxiv.org/abs/2605.03762)；[HF MaYiding/OracleProto](https://huggingface.co/datasets/MaYiding/OracleProto)（MIT） | 按模型截止准入样本 + 工具级时间屏蔽 + 内容级泄漏检测，残留泄漏降到约 1%，比只做工具过滤低一个数量级 | 我们做截止评测时可照搬的泄漏控制流程（数据是通用事件，不是科学） |
| All Leaks Count（Shapley-DCLR / TimeSPEC） | [2602.17234](https://arxiv.org/abs/2602.17234) | 把预测理由拆成原子主张，Shapley 加权算"决策关键泄漏率"；时间过滤检索 + 主张级监督两者缺一不可 | 可作答案级泄漏审计指标 |
| Pitfalls in Evaluating LM Forecasters | [2506.00723](https://arxiv.org/abs/2506.00723) | 系统梳理时间泄漏的多种形态 | 引用 |
| Dated Data | [2403.12958](https://arxiv.org/abs/2403.12958) | 有效截止 ≠ 声称截止，且随资源/主题不同 | 解释为什么要按模型分层报告 |
| Do LMs Encode the Current Year?（COLM 2026） | [2608.15507](https://arxiv.org/abs/2608.15507) | 联想式与陈述式两种"当前年"机制分离；提示、SFT、编辑都无法同时改动两者 | "让模型以为现在是 T"在机制上就做不到 |
| ChroKnowledge（ICLR 2025） | [2410.09870](https://arxiv.org/abs/2410.09870) | 演化型知识（含科学发现）上，LLM 在时间边界处只部分回忆或出现截断 | "模型答不好截至某年怎么看"最接近的实证，但题目是事实级，不是领域级 |
| 时间对齐预训练模型：ChronoGPT / DatedGPT / Scaling PiT LMs | [2502.21206](https://arxiv.org/abs/2502.21206)、[2603.11838](https://arxiv.org/abs/2603.11838)、[2607.11889](https://arxiv.org/abs/2607.11889) | 按年/月截止训练的模型族（金融、社科），用来消除前视偏差 | 规模小，不适合当科研智能体 backbone；只作"从模型侧解决"的对照路线 |
| TIDE | [2608.08512](https://arxiv.org/abs/2608.08512) | 版本化文档（海关法规）上判断"查询日期时哪个版本有效"：隐式日期 59.7%，识别"给的版本不适用"只有 26.7%；模型倾向跟随自信的参数知识而不是给定上下文 | 非科学域，但"as-of 版本解析"失败模式与我们的叙事同构，可作旁证 |
| StateMemBench | [2608.19652](https://arxiv.org/abs/2608.19652) | 智能体记忆要区分"当前状态"与"已被取代的状态"；显式跟踪取代关系的方法有 1.6–1.8 倍提升 | 旁证："取代"需要显式建模 |
| SciConBench | [2606.11337](https://arxiv.org/abs/2606.11337) | 加 clean-room 后分数一致下降，说明泄漏抬高了估计 | 旁证（医学域） |

### 2.4 把持久知识/领域结构作为工具接到智能体上

| 工作 | 一手链接 | 接口 | 评测方式与结论 | 跨 harness？ | 有时间？ | 与我们的关系 |
|---|---|---|---|---|---|---|
| **SkillsBench** | [2602.12670](https://arxiv.org/abs/2602.12670) | Agent Skills | 87 个任务，有/无 Skills 配对，18 个模型-harness 配置（OpenHands、Claude Code、Gemini CLI、Codex CLI）；+16.6pp；"Harness choice materially changes how the same model uses Skills"；自生成 Skills 替代不了人工整理的 | **是** | 否 | **方法论先例，必须引用**。领域是通用操作型任务，Skill 是过程性知识，不是领域认知 |
| EMBL AI Librarian | [2607.28229](https://arxiv.org/abs/2607.28229)；petroni-lab/librarian（MIT） | 自然语言知识层（Europe PMC） | ScholarQABench Citation F1 +16；LitQA2 上 GPT-5.4 接 Librarian 比 web search 高约 8 分；LAB-Bench 上 GPT-4o/GPT-5.4 有/无 Librarian 对比 | 否（换 backbone，不换 harness） | 否 | **半真，中**。"knowledge layer for AI agents"的说法和有/无对比都已被用过；但它是每次查询现场检索编排，没有持久领域结构 |
| LKM v2 | [2609.27297](https://arxiv.org/abs/2609.27297) | OpenAPI、CLI、Bohrium skills | 固定答题模型，有/无 LKM 检索：ChemBench +9.3、PubMedQA +4.2、SciBench +14.7 | 否 | 否 | 前轮已列；在"即插即用"上是半真 |
| ScholarStack v2 | [2609.23735](https://arxiv.org/abs/2609.23735) | 技能 + 知识视图 | 原文："we use Qwen3.8-Max as the base model and Codex as the agent framework"；综述任务拿 Claude Code DeepResearch 当对照 | 否（单框架，Claude Code 只是对手） | 否（全文 grep 无 as-of/cutoff） | 前轮已列；这里补一点：它没做跨 harness 配对 |
| Lacuna | [2606.26246](https://arxiv.org/abs/2606.26246) | Web、/md、MCP | 拿 Lacuna Deep Research 与 GPT-Researcher、STORM、ODR 比；**没有**把 Lacuna 接进 GPT-Researcher 做有/无 | 否 | 否 | 前轮已列 |
| Agents-K1 | [2606.13669](https://arxiv.org/abs/2606.13669) | Python API、CLI、MCP；Claude Code skill | GraphRAG vs 基线 LLM（地学 QA） | 否 | 否 | 前轮已列 |
| SciNet | 见 2.1 | 数据层 | 下游综述质量 +25.3% | 否 | 否 | 半真，低 |
| BioContextAI | [bioRxiv 10.1101/2025.07.21.665729](https://www.biorxiv.org/content/10.1101/2025.07.21.665729v1) | MCP 服务注册中心 | 摘要里没有给出效果数字；"MCP 访问比 web search 答得更好"是 EMBL 论文的转述 | 否 | 否 | 假；数字未核 |
| ToolUniverse | [2509.23426](https://arxiv.org/abs/2509.23426) | 2,700+ 科学工具标准 | 案例研究 | 否 | 否 | 假 |
| DeepXiv-SDK | [2603.00084](https://arxiv.org/abs/2603.00084) | CLI、MCP、SDK；arXiv 每日同步 | 未核其评测 | — | 只有同步，没有 as-of | 假 |
| **Reconcile Once, Write Anytime** | [2608.12984](https://arxiv.org/abs/2608.12984) | 维护的 point-in-time 知识库 + 多智能体写作 | 原文："compose ... report at any knowledge cutoff T, reading only evidence with as_of <= T (no look-ahead)"；金融/统计源；6,845 处跨章节矛盾降到 0 | 否 | **是** | **半真，中**。"持久、可修订、按 as_of 裁剪的知识库 + 写作端"这一机制在金融域已有人做；它不涉及科学文献、方法族、演化与他述。叙事里不能声称"首个 as-of 知识库" |
| Sola Security Brain | [2609.30345](https://arxiv.org/abs/2609.30345) | 离线解析的安全上下文层 | 与 Claude Code、Codex 在同一 AWS 环境比较：0.549 vs 0.340/0.281 | 是（作对手，不是插件） | 否 | 旁证（非科学域）："离线预解析的结构层"胜过让编码智能体现场探查 |
| PoT 的 Faculty 族 | 见 2.1 | 截止前关键词/领域描述 | 小规模"结构化中间产物"实验 | 否 | 是 | 低 |

## 3. 下游评测推荐排序

评分口径：(a) 金标客观、不靠 LLM judge；(b) 现在就能拿到、许可可用；(c) 有截止 T，能测"截至 T 的领域认知"；(d) 有自然的接入位（工具/语料/表示可替换），能做有/无配对；(e) 与我们系统的产出（族、演化、定位、已知未读）对得上。

1. **PreScience**（a 强 / b 强 / c 强 / d 中 / e 强）。用"前置工作选择"和"未来组合预测"两个任务：金标是精确 ID，ODC-BY。接法：给它的 GPT-5 工具型智能体基线加上我们的 as-of 工具。风险：只有摘要；测试期 2024-10 到 2025-10 与 2026 年模型的训练截止重叠，必须按模型截止分层（照 RAP 的 Forecast-Strict 做法）。
2. **IdeaForecastBench**（a 中，LLM 判匹配 / b 强 / c 强 / d **最强** / e 中）。基准本身就是"历史表示"的对照实验，我们是第六种表示，与 Summary、Topic Trend、Memory 同协议比较；全文语料与我们的管线一致。要预先声明用它的两个 judge 都报。
3. **Proof of Time**（a 中强 / b 强 / c 强 / d 强 / e 中）。Inspect sandbox 加挂 MCP 即可；只取"研究演化"和"技术前沿"两族作主报，获奖/引用族作次报。
4. **ScholarCatalyst**（a **最强**，作者标注 / b 中，CC-BY-NC / c 强 / d 强 / e 中强）。测"该读哪些前人工作"，对应"已知但未读"。注意 NC 许可只能用于研究。
5. **AstaBench（框架）+ CS2/PaperFindingBench**（a 中 / b 强 / c 工具级日期限制 / d **最强** / e 弱）。不测领域时间认知，但它是"同一套标准 MCP 工具 + 日期限制"下最规范的配对评测框架，可以把我们的工具与 Asta 工具并列挂上，作为即插即用实验的通用底座。
6. **RAP**、**ForeSci**（概念上 e 最强，b 现在为零）。两者都写明接收/发表后才放出，最好现在就联系作者或盯着发布。RAP 的 State/Oracle-state 消融与我们的主张一一对应；ForeSci 的离线 KB 是现成接入点。
7. **BackTrend**（CC BY 4.0，66 条），用于"认识变迁/弱信号"的小样本补充。
8. **SciNet Task3**、**IG-Bench**、**SCG**：组件级（谱系/演化/前置依赖）评测，不作智能体主下游。IG-Bench 许可 TBD，用前需联系作者。

即插即用实验的设计建议（来自上述工作的共同做法）：
- 照 SkillsBench：至少两个 harness（如 Claude Code 类 + GPT-Researcher/自家管线）× 两个 backbone，有/无我们的工具配对，报配置级增益分布，不只报平均。
- 照 RAP / OracleProto：主报只用模型文档截止之后的窗口；本地 Qwen3.6-27B 的截止没有文档（RAP 表中标 "not documented"），不能进严格轨。
- 照 2602.00758：对照组的 web 检索不要用搜索引擎日期过滤，统一用冻结快照或 AstaBench 式受限工具，否则对照组会因泄漏被抬高。
- 照 IdeaForecastBench：把"每次查询现做的领域摘要"（Summary/Topic Trend）作为强对照，单独证明持久结构比临时压缩多出来的那部分。

## 4. 对叙事的含义

- 不能声称首个：截止对齐的离线 KB（ForeSci、PoT）、as-of 知识库 + 写作端（2608.12984，金融域）、知识层有/无评测（LKM、Librarian）、跨 harness 配对评测（SkillsBench）。
- 目前没看到有人做的组合（限于本轮检索）：把**持久的、按时间可切片的领域结构**（族、演化、他述定位、已知未读）作为**同一个**工具接到**多个**科学智能体 harness 上，在**带截止的领域判断基准**上配对评测。RAP 与 IdeaForecastBench 的结果给这个组合提供了"为什么会有用"的外部证据。
- 风险：IdeaForecastBench 里 Summary 已经有效，审稿人会要求证明"持久结构 > 每次现做的压缩"。这必须作为一条专门的消融，不能省。

## 5. 未核实清单（不进主表结论）

- ForeSci、RAP 数据：两者都写明发表/接收后才放出，现在不可得；发布时间未知。
- INSPIRE：文中提到 release manifest，但没有找到公开数据链接。
- HindSight、Reconstruction：没找到代码或数据仓库。
- NovGauge、SchNovel、RQ-Bench、比较式成功预测（2605.21491）：数据可得性与许可没核。
- SciNet Task3 金标答案格式：仓库里只有问题文本与评测脚本，金标路径按推断是用其元数据库现场计算的，没跑脚本确认。
- IG-Bench：README 写 "License TBD"，没找到 HF 数据集。
- AstaBench HF 数据集 license 字段为空（仓库是 Apache-2.0），数据许可以 gating 协议为准，本轮没读协议原文。
- BackTrend 主题数口径：论文写 25 个成熟主题，HF 卡写 18 个父主题，没核对差异原因。
- OpenReview 上的 "LiveIdeaBench: Forecasting Emerging Research Ideas"（forum KufsJAHeGn）：被 Cloudflare 验证页拦截，没打开。作者组（Fenghai Li、Zihan Tang、Jiaxuan You）与 2609.00747 相同，推断是同一工作的早期版本，没核实。
- BioContextAI "MCP 访问比 web search 答得更好"：只来自 EMBL Librarian 论文的转述，bioRxiv 摘要里没有这一数字。
- 2608.23058（LLM 预测智能体综述）、MCP-Universe、Holistic Agent Leaderboard、Repo-To-Skill 等通用智能体/MCP 评测：只看到标题，与科学领域认知关系弱，没展开。
- 本轮 HF Papers 站内检索（`/papers?q=`）返回的是当日 trending，没有按关键词过滤，因此 HF 渠道的长尾发现基本无效，靠 Google 与 arXiv 站内检索补齐。
