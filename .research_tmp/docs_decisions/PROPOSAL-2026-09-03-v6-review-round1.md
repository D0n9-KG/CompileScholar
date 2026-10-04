# PROPOSAL v6 审稿第一轮：五席报告全文归档（2026-09-03）

> 审查对象：PROPOSAL-2026-09-03-v6-knowledge-compilation.md
> 五席独立审稿（互相不可见），顺序：R1 方法学 / R3 数据库跨域 / DA 魔鬼代言人 / EIC 领域主席 / R2 领域专家。
> 编辑部综合判决见同目录 PROPOSAL-2026-09-03-v6-editorial-decision.md。

---

# R1（方法学席位）判决：大修（Major Revision）

## 总评

这份提案在"问题定义的诚实度"上高于平均：作者自陈了物化视图质疑、列出了风险、区分了证据类型（人类 vs agent）。但从评测方法学的角度，**推导链的每一环都有构念效度断层，而评测设计（第 5 节）目前不足以闭合这些断层**。核心问题一句话：提案用"现有范式失效"的证据（E2、负载目录）支撑"编译层是解"的声明（C1-C4），中间缺一整段因果论证；而设计来补这段论证的层消融实验，自身携带至少三个混淆变量。

## Findings

### F1 [CRITICAL] 负载驱动推导链（C5）存在双重构念断层，且"负载驱动"是选择性的

**问题一：人类负载 → agent 表示的构念效度**。Asta 20 万条是人类用户分布，目录自己标注为"代理真实需求的上界参考"——但"上界"在构念上不成立：人类查询与 agent 查询不是子集关系。agent 侧证据（§1.3）是**代码行为挖掘，不是查询日志**——它告诉你七个系统"现在怎么做"，推不出"它们需要什么"。现有系统全用关键词入口，存在无法被代码挖掘排除的替代解释：当前工具循环协议下结构化接口本来就用不起来。观测行为 ≠ 潜在需求。真正承载"agent 需要什么"因果证据的只有 E2，而 E2 是 12 个任务（见 F3）。

**问题二：选择性负载驱动**。Q1（宽泛主题探索）占 51.6%/65.0%，是分布的绝对主体，却被标记"唯一被充分服务的类型"后**排除在设计驱动之外**；驱动 schema 的是失效集中的约 30%。C5 的实际声明不是"schema 由实测负载反推"，而是"schema 由实测负载中**现有系统失效的子集**反推"——前者是分布跟随，后者是缺口填充，两者的查新表述不同。目录 §5 的定位表述用的是前者，有过证之嫌。

**建议**：(a) C5 改写为显式的"失效子集驱动"；(b) agent 侧行为证据降级为"成本与访问模式观测"，需求侧因果证据的担子全部压在 E2 扩版上——E2 扩版从"待做"升为 gating；(c) 从开源 deep research 系统收集真实 agent 查询日志（哪怕几百条）作为桥接证据。

### F2 [CRITICAL] 层消融的三个混淆变量，"L2 拉开差距 → 创新在编译层"推理不封闭

**混淆一：L0 不是平面 RAG 的等价物**。L0 = "verbatim span + 表格单元格（带坐标）+ provenance"——已是抽取管线的产物，与朴素 RAG 的"PDF 文本切块"在检索单元粒度、结构、噪声谱上全不同。要么给 L0 一个"未投资抽取"的退化版本（L0'）作真正的 RAG 等价臂，要么承认层消融是内部消融，外部比较交给受控 PK。

**混淆二：L2 的提升可能来自对齐键投资而非视图结构**。需要在消融中冻结对齐键版本，让 L1 臂与 L2 臂共享同一份对齐产物，否则"L2 = L1 + 编译"的可加性假设不成立。

**混淆三：任务-视图循环设计**。任务类型与视图类型一一对应=为 L2 臂创造量身定做优势。解法：任务集含"非甜区探针"（目录中存在、视图未覆盖的查询类型，预期 L2 不优于 L1），甜区/非甜区落差本身是诚实信号。

**另有等价性问题**：结构化接口单次调用信息量远大于关键词检索。同"工具预算"下比较的可能是接口信息密度而非表示质量。需预注册预算计量口径，报告实际信息摄入差。

**建议**：层消融重写为：冻结抽取栈与对齐键 → L0'/L0/L1/L2 四臂 → 甜区与非甜区探针 → 预算口径预注册 → 主要对比单点预注册，其余探索性。

### F3 [CRITICAL] E2 证据的方向性缺口：证明"失效"≠证明"编译是解"

"独立问题"需要证明：(a) 失效存在（E2 有）；(b) 失效不可由更好的检索修复（**PaperQA2/OpenScholar 臂还没跑——要从 optional 升为 blocking**）；(c) 编译视图能修复（完全没有）。目前只覆盖 (a)，行文有把 (a) 的证据滑向 (b)(c) 的修辞倾向。

细节缺口：12 任务上 2.83 vs 2.75 的差距是否在单 judge 方差带内？"失效按分散度分层、aggregation 最差 2.33"是事后归因还是预注册分层？图臂 1.58 需排除"该图构建质量差"的替代解释。

扩版缺三样：(a) **每任务证据集 gold 标注**（区分检索失败 vs 聚合失败）；(b) 失效模式分类**预注册**；(c) judge 盲评协议。

**建议**：C1 证据表述降级为"现有范式失效已证，编译作为解的验证由层消融+扩版 E2 承担"。

### F4 [MAJOR] C6 保真度评测：gold 构造的同源循环风险与对齐判定操作化缺失

原四道闸是为方法演化关系边设计的；单元格是"带条件的主张簇"，簇边界判定本身就是开放的标注歧义。需要：gold 由独立于编译栈的流程产生，先报告**构造者间信度**（≥2 名独立构造者，ICR/κ），信度不足的单元格类型不进主指标。对齐判定完全未操作化——需三层判定规范：值层（数值容差规则）、条件层（限定词规范化词表+等价类）、簇层（成员判定 ICR 阈值），每个视图类型各定一个主指标。

### F5 [MAJOR] C2 冲突感知不可验证

抽取噪声（把 0.84 抽成 0.48）与真冲突（两篇论文真的报了 0.84 和 0.48）在观测信号上**同构**——区分两类错误（误报/漏检）天然是 ROC 曲线。建议构造**对抗性注入测试集**：注入已知冲突，操纵冲突幅度与噪声幅度，测检出率/误报率随幅度变化的敏感性曲线。测试集本身可成为可复用贡献物。

### F6 [MAJOR] 层消融与受控 PK 分工不清；统计设计不足

合理切分：层消融=内部归因，PK=外部校准；L0+L1"≈GraphRAG"的等价性声明是自造负担。n≥50 按 9 查询类型分层每格 n≈5-8，多臂全矩阵 power 不足——须预注册单一主要对比+效应量（paired bootstrap CI），其余明标探索性。

### F7 [MINOR] 声明缺评测出口

C3 失效代数无任何测试（需事件式评测：摄入序列→注入 split/rename/撤稿→测失效传播正确率与重算完整性）；C7 覆盖地图的 false-gap/false-covered 需单独 gold；成本指标必须与 D2 召回红线联合报告（成本-召回 Pareto）。

## 真实优点

1. 问题选择的证据密度高于同领域多数提案（六方向查新+代码级负载挖掘+实测分布+试点实验，四路证据互相咬合）。
2. 诚实度纪律（自陈物化视图质疑、pilot 地位、认识论陷阱）。
3. 证据类型标注纪律（人类分布与 agent 行为分开标注）。
4. 召回红线（D2）从 Lacuna 死因学到正确教训。
5. 失效代数三类失效的概念分类干净（问题在验证缺失，不在概念）。

## 裁决：大修。建议 Phase 1（视图代数形式化）之前先冻结评测协议设计。

---

# R3（数据库跨域席位）

## 总评

自我批判意识远超平均水平，负载侧实证罕见地扎实。但从数据库席位看，提案存在一个结构性问题：**它借用的 DB 概念，恰恰在被借用的那一刻丢掉了使这些概念成为"方法"的形式化内核**。"workload-driven"在 DB 里不是修辞，而是可形式化、可求解、可证明近似最优的问题。结论预告：C5 的方法论创新声明在跨域检验下大部分不成立；**C2 的真正近邻被查新遗漏（truth discovery / data fusion，可能是致命的先例问题）**；C3 的"超出 IVM"部分成立但表述过度；**C4 是唯一觉得真正新鲜的东西**。

## Findings

### F1 [CRITICAL] "负载驱动 schema"不是 workload-driven design 的概念层提升，而是形式化内核被抽空后的松散类比

DB 的 workload-driven design = 代价模型 + 选择算法 + 近似最优保证。本提案的实际操作是：定性查询分类 + 频次统计 → **人工设计出六个视图**。这在 DB 术语里不叫 view selection，叫 requirements engineering informed by workload analysis（CQ 传统的加强版）。"对比矩阵"本质是 OLAP cube，条件掩码就是 dice/slice——形状是 40 年前的；创新被正确地放在单元格内容，但那是另一个问题（F2）。"负载驱动"的全部因果链是"负载目录启发了六个手工视图"，不足以支撑方法论声明。

**建议（二选一）**：实质化路线——定义真的选择问题：维护代价可测量（新论文摄入→失效→重算的 token 成本，有全部 instrumentation）；查询收益可测量（每视图族查询频次×每查询节省）；展示预算约束下**哪些视图值得物化是非平凡决策**（消融单视图子集，证明 selection matters）。降格路线——如实表述为"负载分析驱动的表示设计"（workload-informed）。

### F2 [CRITICAL] C2 的真正近邻是 truth discovery / data fusion / entity resolution，而先例查新完全遗漏了这条线

C2 描述的问题——"多个冲突源、保留出处与条件、不静默合并"——正是 data fusion / truth discovery 文献的核心领地：Dong, Srivastava, Berti-Equille（VLDB 2009 起）、Li et al. truth discovery 综述（TKDE 2016）、Dong & Srivastava《Data Cleaning》教材。SIGMOD/VLDB/TKDE 不在 arXiv/NLP 检索空间里。对应关系：认识状态 ≈ source reliability 手工分类版；对齐键 ≈ entity resolution（Fellegi-Sunter 1969 起）；"带条件的主张簇" ≈ incomplete databases 的 c-tables（Imielinski & Lipski, JACM 1984）——条件表就是"值带条件标记、查询时按条件切"。

**但有一个提案没有自觉到、形式化后反而是真贡献的点**：truth discovery 假设冲突值同构，而文献域的冲突大量是**伪冲突**（条件不同且常未声明）。C2 的真问题不是"从冲突中找真值"，而是"**判断两个分歧值是否构成真冲突，需要条件等价性检查（conditional equivalence checking），而条件本身是从文本里带噪声抽出来的**"。truth discovery 没做过且确实难。

**建议**：(1) 必须补 truth discovery/data fusion/ER/c-tables 引用与差异化；(2) 差异化轴从"编译 vs SQL 视图"（错误轴）转向"真值判定 vs 条件等价性下的冲突检测"（正确轴）；(3) 可考虑用 truth discovery 的 source-accuracy joint inference 作为视图单元格置信度标注的实质借用。

### F3 [MAJOR] C3"失效代数"：三类失效各有 DB 对应物，复合是域特有的；但"algebra"撑不起来

轨迹失效=标准 IVM；对齐漂移失效=schema evolution + view adaptation（1995-2005 活跃文献）+ ER 修复传播（collective/temporal ER）；认识状态失效=temporal DB 的 valid time + KR 的 belief revision（AGM 1985）。三类没有一类概念上是新的；**组合**是域特有的。

真正的超出在提案没指出的地方：**DB 的视图维护是可靠的——V = Q(D) 可证明；本提案的编译产物是启发式抽取的结果，"失效检测"本身不保证召回（派生轨迹记录不全就漏标失效），重算也不保证恢复一致**。视图没有 Q——编译器不是可重放的函数。诚实的名字应该是"**best-effort invalidation semantics under noisy compilation**"。

**建议**：(1) 改名并给形式化骨架：视图元素状态机（fresh/stale/invalid/orphaned）+失效判据+传播规则，**把"失效检测召回率"作为 C6 保真度评测的一等指标**；(2) 引用并差异化 view adaptation 与 valid-time 文献。

### F4 [MAJOR] 负载刻画与 DB 的 trace/benchmark 标准差距过大，且存在未识别的负载迁移假设

Asta 日志是人在搜索平台的关键词查询，不是智能体在已编译知识模型上的类型化操作——**把人类搜索负载当智能体操作负载的设计输入，是一次未验证的分布迁移假设**。且没有负载漂移的故事：提案的维护语义全部针对数据更新，没有一个字针对负载更新——对以"负载驱动"为方法论核心的提案是结构性缺口。

### F5 [MINOR] C7 在 DB 席位下是 closed-world assumption 的直接应用（Reiter 1978），作为创新行偏薄——降格为接口性质。

### F6 [MINOR] 对齐键=entity resolution；"视图质量上限"表述暗示了 50 年 ER 文献却零引用。

### F7 [MINOR] L2 视图与 semantic caching / GPTCache 的近邻关系未被讨论。

## 真实优点

1. 层消融定位创新承载位置——DB 审稿人最欣赏的实验设计纪律。
2. 需求侧负结果真实且反直觉（推翻"小规模直读够用"）。
3. **C4 delta 继承解析是全篇最新鲜的东西**：给定正确抽取，展开用递归 CTE 二十行 SQL 就能表达，**但"给定正确抽取"不成立——省略式引用的解析是从文本恢复视图定义的问题，DB 里视图定义总是显式给出的**。建议 C4 权重上升、C5 权重下降。
4. 召回红线对应 DB"视图不替代基表"铁律。
5. 自陈风险质量高。

## 对审查焦点 (a)(b) 的直接回答

(a) 物化/预计算动作本身无新意，且差异化不应打在"预计算"上。可防守的创新内核是三个：冲突簇语义（必须先过 truth discovery 关）、失效语义的域特有复合（表述需降格）、delta 继承解析（最干净）。(b) vs Lacuna/SciAtlas 成立；vs KG 成立但让步很大（表示层贡献让位给编译层贡献，而编译层恰是打击最重的位置）；**vs materialized views / truth discovery 的差异化目前不成立或未做**。

## 若投 DB venue：大概率被拒且死因可预测（查询语言/代价模型/选择问题形式化/失效可靠性度量四个"没有"+玩具规模）。现实归宿=NLP/IR 侧 agent 系统方向。若要 DB 出线：完成 F1 实质化路线+F2 文献补课，约一年以上增量。

## 最后的建设性总结

**把 DB 在"不可靠编译"上的空白变成贡献**：DB 的视图维护之所以是科学，因为 Q 可靠且可重放；你们的一切"失效"难题的根源是 Q 不存在。与其声称超出 IVM，不如宣称"我们研究视图维护在编译器不可靠时的退化形式"——定义失效检测召回率、重算收敛性这些可测量，把 DB 形式化当作靶子而不是帽子。

---

# DA（魔鬼代言人）

## 1. 最强反驳

本提案对"物化视图"质疑的防御（C2：单元格保留冲突簇而非合并单值）恰恰暴露了一个它自己没有命名的两难。走第一条分支——保留冲突簇——那么格子不是答案，是预组装的证据：消解工作被推回查询时的 agent，而 E2 的诊断（强模型修不好分散且条件互斥的跨篇聚合）恰好预测 agent 会死在这个残余任务上，整个评测计划没有任何环节单独测量"消解冲突簇"这一能力。走第二条分支——视图消解冲突——则是在已知不低的抽取噪声上做有损物化：delta 继承链把单条误解析放大成整张方法卡，而类型化 API 的存在恰恰抑制 agent 回 L0 复核；对实验规划 agent，**一个自信地错的结构化格子比 RAG 模糊地错更危险**——它权威、整洁、且以唯一姿态出现。一条分支是切块更好的 RAG，另一条是正确性更差的缓存：C1 在两条分支上都得不到支撑。更糟的是，作者自家实验室唯一一次结构化对照（E2 图臂 1.58 对直读 2.83）是 0 胜 1 负——"这次结构与负载匹配了"正是待检验的假设，却被当成了设计前提。

## 2. 问题清单

**CRITICAL**

- **C-A【评测设计】编译与缓存不可区分**。消融各臂只变表示、不变成本与能力配额。"L2 拉开差距"可能全部来自 token 预算/注意力腾挪（Lacuna 4 倍效应即此类），那是成本转移不是能力增益。缺"能力配对"对照（给 L1 臂无限预算看是否收敛到 L2）、缺等成本对比、缺视图优势随模型强度变化的趋势测试。C1 是全篇承重墙，这一漏即塌。
- **C-B【核心机制】抽取噪声经编译被放大而非缩小，且无任何量化**。本团队自己的验收轨迹：方法演化边 37.4%、外推 50-58%——这个数字做谱系图勉强，做数值格是灾难（15% 错格率的对比矩阵对实验规划 agent 是对抗性错误信息）。没有 per-field error budget、没有 delta 链的复合误差传播分析（k 跳继承后方法卡正确率的下界）、没有把"视图置信度"作为一等字段暴露给 agent。风险清单列了对齐键，却没列误差传播——比对齐更深一层的结构性沉默。
- **C-C【方法论】负载的循环性与测量对象错位**。Q1-Q9 分布的主体是人类用户经搜索框表达的需求；agent 侧只有访问行为与成本的机械观测，没有 agent 的信息需求分布。两总体被 D1-D10 混合消费。更根本：负载由现有接口塑形——用被批评系统的失败模式反推表示，是在现状的局部最优点上做设计；真正创新的负载（agent 还不会问的问题）在目录里结构性不可见。C5 被内生性一击即溃。
- **C-D【竞争】SciAtlas 窗口期 + 组合收益无法归因**。Ongoing work、大团队、两月一版、四空轴是他们自知的 roadmap。本方从骨架稿到含 E2 扩版五要件现实 6-12 个月，中间隔着 3-6 个 SciAtlas 版本，且无 scoop 应急预案。差异化=四组件组合，按作者自己的铁律"组合创新必须证明组合>各部分"，**没有任何实验能把增益归因到组合而非单组件**——一张做得好的对比矩阵（ORKG 已验证其价值形态）可能拿走全部收益。
- **C-E【核心机制】对齐键是自认的单点失效，且最可能运行点是"半可信死区"**。不是几乎全对也不是明显崩坏，而是 ~85% 对齐：视图既不能弃用也不能信任，降级通道吃满后 L2 退化为 L1。而跨篇实体对齐是本团队已知未解难题（split 命名跨篇不收敛、NMR 跨语言失败史）。全部 L2 价值以一个自家解决不了的问题为质量上限。

**MAJOR**

- **M-A** E2 pilot 被过度杠杆化：12 任务/单 judge/无出题人分离，三臂绝对分全 <3/5——与"任务过难或评分噪声"的替代解释完全兼容。
- **M-B** L0 消融臂被污染：L0 含表格单元格+坐标，不是 vanilla RAG 等价物——表格理解本身是已知增益来源。
- **M-C** 无盈亏平衡核算：两遍 harness×全语料×多视图编译的 build cost 与摊销查询量缺失；pilot 规模下摊销几乎必然为负。
- **M-D** 缺失性查询的行为风险：对齐漏检产生的 false gap 与 real gap 不可区分，且 absence 无法用 provenance 抽查；agent 不会把"编译语料内未见"限定词带进新颖性结论。**让缺口查询更容易可能比基线制造更多假新颖性判断——结构化把 AI Scientist 的拍脑袋升级成了自信地拍脑袋**。
- **M-E** 提案正文漏引 Eigenius 与 AI-Supervisor：L1 的认识状态四分类与 Eigenius 的 declared/observed/derived/verified **几乎逐字相同**（report_4 自己查明的）。按 SYNTHESIS 自己的红线"漏一家审稿人一查即死"，§6 目前踩在自己画的红线上。
- **M-F** C1 证据形态："2024-2026 无此任务"是弱证据（benchmark 滞后于问题）；E2 扩版"任务类型与视图类型一一对应"=按系统出题，self-serving benchmark 的味道。
- **M-G** 认识状态失效的负载证据是全目录最薄的一格：撤稿/结果被质疑是罕见事件，常态是共识渐移=普通增量更新，periodic rebuild 对之已是合理解。失效代数可能是用重机械解罕见事件，而它恰恰是 C3 的核心。

**MINOR**

- m-A "视图代数"名不副实：当前是六张 schema，无组合算子、无完备性/最小性论证——六个 ER 图的事后升格。
- m-B 数值计算失效穿过表示层：对比矩阵仍要求 agent 在格子上做算术，该失效模式视图不修复。
- m-C C4 rebranding 风险："配置属性上的共指消解+属性继承"；零命中是关键词层面的，不是概念层面的。
- m-D 消费者架构赌注："API 即知识模型"预设 tool-loop 范式持续。
- m-E 2,355 篇单域对 ML 场是 toy scale kill。

## 3. 被忽略的替代路径

1. **Lazy compilation（查询时视图构造+缓存）**：失效代数存在的唯一理由是选了 eager。提案从未论证 eager over lazy——全篇最大的未审视分岔。Asta 日志恰恰显示查询重尾分布。
2. **社区 curation 路线（PapersWithCode/ORKG）**：PwC 就是已部署、负载验证过的对比矩阵+覆盖地图。人/社区维护在格子正确性上完胜 LLM 抽取——既是竞品也是"配置轴天花板已被人类方案占据"的反例。
3. **Benchmark-first 论文**：先把 C1+C6 做成问题定义+保真度基准的论文，系统后置 follow-up。风险砍半、吃存量优势、绕开与 SciAtlas 的正面试竞速——且被 scoop 后仍可存活。
4. **专责聚合模型**：小型 specialist aggregation agent 查询时在 L0 verbatim 证据上做 join——保 grounding、无 stale 视图、无维护语义，是"RAG 堆片段"与"离线物化"之间提案从未考虑的中点。
5. **Scaling baseline（必须先跑）**：给 RAG 臂 10 倍 token 预算+前沿模型。若闭合差距，编译视图只剩成本故事——最便宜也最致命的对照，放在 Phase 1 之前比任何视图 schema 都值钱。

## 4. 利益相关者盲区

真正的消费者（agent 开发者）从未被直接征询；无部署路径。Venue 身份未决：DB 审稿人要代价模型与形式化代数，ML 审稿人要规模与 SoTA 表——两头不靠。被抽取论文的作者：把他人结果标"共识 vs 声明"是对他人学术主张的再分类，误标是声誉行为，无治理/申诉机制。用户自己的时间线：与 SciAtlas 竞速无 scoop 触发条件——"何时降级为 benchmark 论文"应预先写成硬阈值（如 SciAtlas v3 出现内容轴即切换）。

## 5. 非缺陷观察

- 自我怀疑的质量高于平均——但这反而抬高了剩余未处理意见的杀伤权重：**被自己的诚实清单漏掉的，都是承重级的**。
- **隐藏的王冠是 C4**：零先例、可单独验证、是关于科学写作体裁的内容洞察而非系统声明。Eigenius 占掉认识状态四分类后，持久差异化集中在 C4+C6——一篇更紧的论文可以以这两点为主、把"编译层"降为基础设施叙事，反而更抗撞车。
- **"对 agent 而言，知识模型的 API 就是知识模型"是全提案最好的定位句**，值得当论文题眼；它比"knowledge compilation"这个带 DB 历史包袱的词更能承载 C1。
- 负载目录的标注纪律是罕见的诚实——作者对循环性有直觉但没把它推到结论。

---

# EIC（领域主席席位）

## 一、总体评价与两道裁决题

**裁决 (a)**：作者的辩护策略（把创新从"预计算"下放到编译层 C2 与维护语义 C3）**必要但不充分**：C2 与 data fusion（Dong et al.）、provenance semirings（Green et al. 2007）、uncertain/lineage databases（Trio/ULDB 系）高度相邻；C3 两类"新"失效各有部分先例（incremental ER / view adaptation / ontology evolution）。六方向查新覆盖的是 AI/agent 侧近邻，**恰好没有覆盖作者自己提出的质疑所指向的领域——DB 理论**。这是 novelty 论证的结构性空洞。逐机制看，只有 C4 大概率是干净的新机制。**正式回答：按当前论证不足以支撑；补齐 DB 轴查新并给出逐机制 delta 表之后，有真实的成立可能。**

**裁决 (b)**：vs Lacuna 成立（但 Lacuna 是同期演进系统，差距是"程度+机制"而非"物种"）；vs KG 诚实但把整篇论文押在系统层声明上（期票）；**vs materialized views 当前不成立，且 C1 存在被直接反例的过度声明**——MS GraphRAG 的 global search 用离线预计算的 community summaries，RAPTOR 是离线预计算的层级摘要树。真正的 delta 必须重述为：物化的不是通用内容摘要，而是**负载类型化的答案结构（对齐后、带冲突簇、带维护语义）**。

## 二、真实优点

1. §2 的核心设计裁决（"物化的对象是负载的答案结构"）——全文最锋利的一句话，正确绕开与 KG 比表达力的必败之仗。
2. §5.1 层消融敢证伪自己——同类论文极少敢于设计能杀死自己主卖点的实验。
3. 召回红线作为设计不变量——把竞争情报转化为架构约束的正确做法。
4. 声明纪律（分层表格带边界与依据、风险自陈准确无粉饰）。
5. 需求侧三角证据的方法论结构正确。
6. **C6 的定位（第五空轴）可能是全文最坚实的贡献位**，且正好落在团队既有方法论（评测工具先于被测系统审计）的顺风位。

## 三、问题清单

**CRITICAL-1：novelty 辩护缺少 DB 理论轴**——C2 的主张簇 = data fusion + provenance + 条件维度；C3 = ER maintenance + 撤稿/ontology evolution 的变体。每条承重声明都需要对这批文献给出精确 delta，目前一条都没有。建议补专门查新方向（data fusion/conflict resolution、provenance semirings、uncertain databases、view adaptation、incremental ER、ontology evolution、OLAP 条件切片），产出"机制 × 最近先例 × 我们的 delta"对照表写进论文。**这一步不做完，(a) 的答案永远是"未证"，且在 DB 背景审稿人手里是一击必杀**。

**CRITICAL-2：两条承重声明（C3、C4）在评测设计里零挂钩**。§5 至少加三项：(i) 增量摄入实验（流式加入 N 篇，测失效检出率、误报率、局部重算成本 vs 全量重建）；(ii) 对齐漂移注入（人为 split/rename，验证级联失效传播与恢复）；(iii) delta 继承保真度（含省略式引用的方法卡 vs 人工展开 gold）。

**MAJOR-1**：C1 全称否定被 GraphRAG community summaries/RAPTOR 反例。重写。
**MAJOR-2**：L2 必要性缺"theory of the case"——为什么 agent + L1 查询时 join 不够？需显式成本模型论证（编译摊销条件）或能力论证，且 MAJOR-2/7 合并。
**MAJOR-3**："knowledge compilation"撞 Darwiche & Marquis 2002（NNF/d-DNNF 线，30 年既定含义）——改名（answer-structure compilation / demand-driven materialization）或引言显式划界。倾向前者。
**MAJOR-4**：需求证据链与"agent 需求"错位——"need-driven"目前实为"人类负载+代码检视驱动"。E2 扩版不是投稿前加固，而是主轨道资格线。
**MAJOR-5**：C7 未对三类近邻核查（research-gap detection/literature-based discovery、incomplete databases 的 closed-world 形式化）。降格并引 incomplete-DB 托底。
**MAJOR-6**：C5 是凑数声明，降级为方法论一节；把 WORKLOAD_CATALOG 本身作为 resource/artifact 贡献——其独立价值比"方法论声明"更硬。砍掉后剩五条声明，论文反而更强。
**MAJOR-7**：编译摊销成本模型缺席——对位 40 年物化视图传统的论文没有 cost model 会被 DB 视角审稿人视为不严肃。
**MAJOR-8**：规模与覆盖地图认识论限定互相放大——2,355 篇单域下"编译语料内未见"的实用价值很弱；第二域验证从"加固项"提为"缺口查询演示的前置条件"。

**MINOR**：认识状态四分类引 evidence grading（GRADE/循证医学）；六视图按"四核心+两常规件"分层表述；need-driven/workload-driven/demand-driven 统一；认识状态失效引撤稿文献处理实践；七基线分层报告（Tier 1 全跑，Tier 2 引用已发表数字）。

## 四、分量与 venue 判断

**承重墙**：C1（改写后）、C2、C3、C6。**半承重**：C4（最具体、查新最干净，但宽度不足以独立承重，作为 C2 的支撑机制定位更准）。**凑数**：C5、C7——砍掉后五条声明的论文比七条的更强。

**分量裁决**：不是"一个 workshop 论文的想法"；当前是"**一篇扎实的 specialty 论文的底子，附带条件性的顶会主轨潜力**"。升档条件四个：(i) DB 轴查新闭合且逐机制 delta 精确；(ii) 层消融真的把 L2 分离出来；(iii) C3/C4 评测补齐；(iv) E2 扩版+第二域。四缺一落回 SIGIR/WWW 级 specialty。

**Venue**：1) 形式化主轴→SIGMOD/VLDB（无人占位但 CRITICAL-1 必须先做完）；2) agent 任务增益决定性→NeurIPS/ICLR 主轨；3) **最稳对口径：NeurIPS Datasets & Benchmarks（C6+负载目录 artifact+层消融协议打包，与团队评测先行传统完全同构）**；4) 保底 COLM/WWW/SIGIR；5) ACL 主轨不推荐。

**时机风险**：被窗口焦虑推着跳过 CRITICAL-1/2 的版本会在顶会被打死，比晚投半年更伤。先闭合两个 CRITICAL，再谈投稿节奏。

---

# R2（领域专家席位）判决：Major Revision

## 一、事实核查

**通过项**：SciAtlas 四轴自认引文核实一致；Asta 分布数字一致；七系统行为代码级核实一致；LitBench 判断一致；C5 四要件真空查证属实；SciAtlas 元数据级/periodic rebuild 一致。

**问题项**：
- **F6 [MAJOR] Lacuna recall 0.028 的口径与选择性引用**：该数字来自 ReportBench-ML 的下游 agent 引用召回，不是知识模型自身召回；同一系统在 LitSearch 上 Recall@10=0.538 **高于 OpenScholar 的 0.424**。只引最差不引有利数字，查到原文后会被读成 cherry-picking。改写为"Lacuna 证明预计算对效率有效（4 倍）、对下游 prose 引用召回有代价，但从未测量过知识模型保真度"。
- **F7 [MAJOR] C1"全部评测协议无聚合"限定不足**：report_2 的共性结论限定于 GraphRAG 家族；AutoSynthesis/SLIDERS/EviSearch 做带统计协议的元分析聚合，MVSS 把"结构化对比表"显式物件化。全称量词在循证医学背景审稿人手里是子弹。限定为"KG-RAG/multi-hop QA 评测协议"。
- **F8 [MAJOR]** "稀疏图 1.58"臂配置归因缺失：臂 C 用的是词面查询，惨败恰恰"归因精确化了知识模型的前提"。不交代查询接口配置=把接口错配记成表示失败。
- F13-F17 [MINOR]：Asta 数字未注模式；"停止条件全是计数器"过度（实为 5/7）；"需求侧实证（已完成）"标签过强（pilot n=12 应行内标注）；HippoRAG 2 自认有断章风险（保留条件从句）；SciAtlas v2 引文需投稿前复核。

## 二、差异化裁决

**vs Lacuna**：四点差异成立三点半，前提修 F6 口径+(b) 维护语义需精读原文确认。
**vs KG**："一等公民与访问路径，非表达力"成立且是全文最清醒的设计判断。防两个后续攻击：①"弃图"趋势攻击（T²RAG/LinearRAG——答案已在 D2 里，接到论证位置）；② L0+L1≈GraphRAG 等式不严格（GraphRAG 是社区摘要非字段化记录）——L1 消融臂自己实现，GraphRAG 作外部 PK。
**vs materialized views**：方向成立表述需精确化，缺三组引用：probabilistic/uncertain databases（Trio/MayBMS）处理不确定数据的代数存在——真正差异是"输入是非结构化文本+抽取噪声+实体身份漂移，无 schema 保证、无正确性下界"；view adaptation under schema evolution；AGM belief revision。

## 三、漏竞品扫描

- **F2 [CRITICAL] 循证医学 evidence synthesis / 元分析自动化线在提案正文整体缺席**（调研有、论文没写）。三重压力：C2——meta-analysis 的 heterogeneity（I²统计量、亚组分析）就是"冲突的统计形式"；**C7——evidence gap maps（3ie/Campbell，Snilstveit 2016 系统化）："干预×维度矩阵，有证据=格子填充，空白=研究缺口"，与覆盖地图定义几乎逐字对应，是成熟方法学名词**；维护语义——Cochrane Living Systematic Reviews（持续更新+失效概念，人类流程版）。可检索：Trialstreamer/RobotReviewer/evidence gap maps 3ie/living systematic review。域限 RCT/临床、窄 schema、人类策划、无 agent 评测、无失效代数——**恰是"四轴更通用+agent 原生"叙事的完美踏脚石而非威胁，但前提是正面引用划界。不处理就是死罪**。
- **F3 [MAJOR] Intern-Atlas（2604.28158）与谱系 DAG 边类型几乎重合**：{extends/improves/compares/replaces/adapts}（9.4M 边）vs"取代/改进/组件/复用"一字之差。差异存在（binary、static、non-maintained、非视图、无 delta 解析）但必须写出。
- **F4 [MAJOR] ORKG comparisons 在 §6 缺席**（report_4 自己判定"表示撞车"）。
- **F5 [MAJOR] "knowledge compilation"术语撞车**（同 EIC MAJOR-3；R2 认为这个类比其实有利，可先试正面划界）。
- **F9 [MAJOR] 语义查询引擎线未覆盖**：DocETL/Palimpzest/Lotus（semantic operators）——"查询时拼装"的学术化形态，比 GraphRAG 更精确的对照极，纳入后 C1 论证反而完整。
- **F11 [MAJOR] C6 需与 KG 质量评测传统划界**：OAEI/GERBIL/KGBench 2024——"评测工具先于被测系统"的先例。真空在"agent 构建的科学文献知识"对象上成立，但句式不加域限定会被打。
- F18/F19 [MINOR]：FinQA/ConvFinQA/MultiHiertt（表格数值推理）；CoreSC/argumentative zoning 的 Gap 标签、scite.ai smart citations。

## 四、C4"零先例"专项裁决：按当前表述不成立；降级重写后核心空位仍成立

1. 方法学层：report_3 原话有双重保留（WebSearch 弃用、arXiv-only、自警"ACL 需二次扫描"），提案写成无保留的"查新零先例"——**过度转述自己的查新**。
2. citation function classification 三十年线（Teufel argumentative zoning → ACL-ARC → SciCite → 3Cext；综述 arXiv 2402.01605）：**分类引用的功能但不解析继承的内容**——恰好是 C4 的空位所在，但不引这条线会被立刻拿出来。
3. **化学域同构先例（真正威胁）**："same procedure as [ref]"在合成化学是明确讨论过的经典 IE 挑战（RSC Faraday Discussions 2019 "Same procedure as? — Leveraging heterogeneous data in chemistry" c8fd00153a；Vaucher automated extraction chemical synthesis actions Nature Communications）。与"same setup as [14] but X"同构。查新完全没扫化学/材料域。

**重写表述**："省略式配置继承的解析在 NLP/ML 文献域无系统工作（arXiv+ACL 覆盖内确认，化学域有同构现象的讨论先例），citation intent 分类识别但不解析，化学程序 IE 触及但不针对方法学配置"——完全站得住且更经得起攻击。

## 五、领域贡献判断

叙事清晰可信（elevator test 通过）。三个结构性风险：**效应量风险（最大）**——直读 2.83 vs RAG 2.75 差距 0.08，编译层要在 n≥50 上从 2.8 拉到 4+，12 任务 pilot 无法支撑外推；论文成败押在§5.1 层消融上，而这是全文唯一没做过任何先导版本的核心实验——**建议 Phase 1 前用现有 L1 资产做一个最小对比矩阵视图的 smoke test**。声明数量风险——七条太多，主叙事收缩为 C1+C2/C3+C6。可辩护性依赖 C1 修正。

## 修改要求汇总（优先级）

1. F2 evidence synthesis 一节+三处划界；C1 限定
2. F1(C4) "零先例"降级重写；引 citation intent 线+化学先例
3. F3 Intern-Atlas 正面引用
4. F4/F5 ORKG 进 §6；术语与 Darwiche 线区分
5. F6/F8 Lacuna 双口径；1.58 臂配置交代
6. F9 DocETL/Palimpzest/Lotus 纳入 C1
7. F10/F11 C3 补 view adaptation+AGM；C2 补 probabilistic DB；C6 补 OAEI/GERBIL/KGBench
8. F13-F17/F18/F19 口径与引用精度逐项修

**总评一句话**：骨架是对的，最锋利的几句话已达顶会水准；但"零先例"式声明的检索纪律和 evidence synthesis 线的缺席，是当前版本与"能过审"之间最大的两块补丁——而这两块补丁的材料你们自己的调研里已经有一半。
