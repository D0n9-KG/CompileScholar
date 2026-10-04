# 五席复审 · EIC 席报告（Journal-Fit / Area Chair 视角）

**席位**：EIC（WWW/ACL 研究轨资深 AC，知识系统+科学文献挖掘方向）。角色分离：本报告只写本席判断，不推测、不引用其他评审人意见。
**评审对象**：`IDEA-REVIEW-PACKAGE-2026-09-08.md`（下称"材料包"）及其指定补充档案。只读评审，未修改任何材料文件。
**复审基线**：v6 五席判决（memory `v6-review-round1-verdict.md`，2026-09-03，判决=大修）。
**日期**：2026-09-08。

---

## 0. 总体判决倾向

**Borderline — 方向可投，当前证据形态不可投。** 以 2026-09-08 的材料状态投 WWW/ACL/EMNLP 研究轨，我作为 AC 会给 "Reject & Resubmit / Major Revision" 档：核心实验（B3 round-2 非劣 PASS + 耦合交互项）是真进展且方法论纪律罕见地好，但（a）头条动机主张 C1 引用了自家档案已推翻的结论，（b）外部基准证据处于 oracle-scoped 降级状态、R2 未跑，（c）最危险竞品 ASKS 在材料包 §4 生态位清单里整体缺席。这三件都是可在投稿前修复的，没有一件是方向性死刑。修复清单见 §5。

本席认为该项目**最强的可发表资产不是"需求驱动知识底座"这个叙事，而是 B3 的耦合交互项实验设计 + 预注册治理纪律**——贡献声明应当倒过来写（详见 §1）。

---

## 1. Venue 匹配与贡献定位（席位任务 1）

### 1.1 venue 匹配判断

**WWW 研究轨：首选，但有硬性前置条件。**
- 匹配面：科学文献知识系统 + agent 访问层 + 评测治理，正对 WWW "Semantics and Knowledge" / "Systems and Infrastructure" 轨道口味；SciAtlas（浙大+UCL）、ASKS 都把这个生态位当 WWW/SIGIR 系地盘在占。
- 不匹配面（当前状态）：WWW 审稿人对规模敏感，40 篇档 + 50 自建题是全场最小考场；而 MERGED-VERDICT.md 判决一明确记录 **SciAtlas v1→v2 三个月补齐全套自跑基线（4 基准+~20 基线+统一明文）已成"生态位事实准入标准"**（MERGED-VERDICT.md L10）。在自家 50 题上只比自家 flat/RAG 臂、具名基线一个没跑的状态下投 WWW，desk-reject 不至于，但 "insufficient external validation" 会出现在每个审稿意见里。
- **前置条件**（缺一即降档）：①PaperScope R2 第一批（含 LightRAG/PaperQA2/官方复刻臂，全自跑）完成且非劣结论在外部考场复现；②H7(d) one-shot 强化臂补跑（B3 报告披露 5 自己挂账，~600k，便宜到没有理由不带进投稿）；③材料包 §1 的 C1/C4 重锚（见 §4 逐条）。

**ACL/EMNLP 研究轨：次选，且需要重写重心。**
- NLP 系审稿人会把火力集中在两处：LLM 裁判依赖（材料包 §5.8 自认官方人机一致性 r≈0.62 是天花板参照）与 gold 出题人=作者（§5.4）。这两条在 WWW 是弱点，在 ACL 是结构性怀疑（"你的主结果测量仪器本身不可靠"）。
- 若走 ACL/EMNLP，裁判偏置研究（材料包 §3 "可独立成文"那条：换裁判臂均分移 0.9/κ=0.39 排序稳健、"编造"判词 96-100% 失实、J1 核验附录协议）应升格为正文一等贡献而非附录补丁——这是 NLP 系真正稀缺的东西。当前材料把它埋在证据链第三行，是浪费。
- **严重度 MINOR（定位建议，非缺陷）**：修改位置=论文结构层；怎么改=WWW 版把裁判研究放"评测协议"节+附录，ACL/EMNLP 版把它升为第二贡献并加 κ/一致性全表。

### 1.2 贡献声明该怎么写才立得住（可直接采用的措辞骨架）

按当前证据强度排序，**只有下面这个顺序是立得住的**：

**第一贡献（主结果，措辞必须用非劣性语言）**：
> "在 40 篇语料档上，类型化记录层 × 类型化工具 ReAct 访问达到整文档直读的非劣（3.42 vs 3.50，预注册阈值 −0.10 内，单次实测过线、两轮实录全披露），token 成本约为直读的 1/7；在同一循环下把类型化视图换成原文检索的受控对照中，交互项 +0.50（Δtyped +0.69 vs Δraw +0.19），为'收益来自表示语义而非通用 agent 循环'提供了首个受控证据。"

三个措辞纪律：①"非劣"不许写成"匹配或超越"（闸 3 FAIL，追平未超越，B3 报告 §二自己判的）；②+0.02 余量与 ±0.1 单跑方差带必须与主结果同句出现（B3 报告 §八披露 1 的原话"单次实测过线，非稳定超过阈值"就是审稿人语言，直接抄进 limitation）；③交互项自称"受控证据/初步因果证据"，不许写"归因证明"（该闸在预注册里标注"探索性，无通过闸"，见 §4-O3）。

**第二贡献（方法论，收窄口径照抄查新档案）**：
> "据我们所知，首个把预注册验收门柱贯穿 IE/KG 系统多轮构建程序、并完整披露迭代日志（含 FAIL 轮与 harness 缺陷的逐轮拦截实录）与修改仲裁记录的工作。"

这句必须按 novelty_surfaces.md surface_A 的"可防守写法"（L59）原样收窄——**禁写"首次预注册"**，必引 zemhp（Cochran 2026，同域！）/Vaccaro/Thomas/EMSE RR 体系/Donoho 2025。IL-C4/C5/C6 三缺陷"验证环节三轮各拦一个"+ 拒写分带因果归因（螺旋带 6 题翻身、靶题 T08 1→4）是这条贡献的实证肉身，这是我在全部材料里看到的、其他任何竞品档案里都没有的东西——**建议把 B3 报告 §三的缺陷带归因表直接搬进正文**，它是"预注册治理真的在干活"的最硬展示。

**第三贡献（需求侧动机，重锚后才可用）**：
> "整读路线的失效是规模带的：同一强模型直读从 166k 字符档 3.50 退化到 339k 档 3.00，93 篇档撞上 ~800k 字符的窗口算术墙；5 篇小档上前沿模型直读反而最强（聚合 4.33），因此本工作的主张限定在规模档。"

注意这个措辞把 E2-R2 修正（直读失效=中端模型/规模档产物）**吸收成了主张的边界条件而非隐藏它**——这是唯一能同时满足"诚实"与"动机成立"的写法（详见 §4-O1）。

**不能承载头条的**："需求驱动知识底座"作为命名贡献。理由见 1.3。

### 1.3 竞品格局下"需求驱动"差异化还剩多少

逐项盘点（全部有档案锚点）：

| 占位者 | 占掉了什么 | 我方剩余 | 锚点 |
|---|---|---|---|
| SciAtlas v2（§4.3 workflow-specific context assembly） | **"demand-driven access" 字面被占**；v2 同时把"全套自跑基线"立为准入标准 | "demand-driven content"（内容四轴：配置/数值/取代/条件对象化——v2 Limitations 原文自认仍缺） | sciatlas-v2-mechanist-recheck.md L15 |
| Mechanist（同团队） | **"需求驱动 schema 扩展 + 保守抽取纪律"已小规模演练**（13k 专域 KG，DeepSeek 抽取+留空不猜+Claude grounding 质检） | 逐字引文锚/条件维度化/失效语义它没有；但它在自证"浅元数据 KG 撑不起深需求"——这只能当我方动机引用，不能当我方独占 | 同上 L17 |
| ASKS（2608.29612，08-30） | **"scientific knowledge compilation" 术语 + 定义句 + 治理闭环概念（GraphDelta/事务融合/可检查状态转移/replay/bounded authority）双撞车** | 它零基准评测、无需求驱动查询负载、无 typed tools、56 篇单案例——我方全部实证差异化轴对它成立，但**命名权已失** | asks_note.md §3/§5；competitor_audit_2026-09-06.md §三 |
| PaperScope/PaperArena | 基准位（考场owner，非竞争者） | 用其考场是加分项不是威胁 | MERGED-VERDICT.md 判决一 |
| LedgerMind/Doctor-RAG/CEL/SCAIR | 三条红线各自点位 | 材料包 §4 已正确处置（必引区分不宣称首创） | 材料包 L34 |

**结论（MAJOR）**："需求驱动"作为形容词的独占性已经塌了一半——access 半被 SciAtlas v2 占、schema 扩展半被 Mechanist 演练、"编译"命名被 ASKS 抢注。剩余可辩护的是 **"demand-driven content × 语料尺度 × 三臂实证闭环"的交点**，而且这个交点的实证支撑目前只有耦合交互项一条腿（内容层直接证据=C2 的代理指标，见 §4-O4）。修改建议：论文标题与摘要**不出现裸的 "knowledge compilation"**（asks_note.md §5 行动项原话），"需求驱动"每次出现必须带 content/access 限定词切割句（sciatlas recheck L15 已给出切割公式）。

### 1.4 最可能被打成"工程组合无新意"的主张

**首要靶子 = C2（记录层升级有效）**，材料包 L11。理由：
1. 它给出的证据全是代理指标——条数 1010→6691（条数多≠好，可以是膨胀）、引文验证 99% vs 0、年份元数据 100% vs 0。审稿人一句话："schema 设计 + 确定性校验管线是 KB 构建的标准工程实践（EDC/DySECT/AutoSchemaKG 谱系），你证明的是'认真做工程比不认真做工程好'。"
2. 项目自己的查新档案已经预判了这个判决：novelty_surfaces.md surface_B L89——"必须有对照实验……否则顶会审稿大概率判 'engineering choice, not a research contribution'"。而 C2 恰恰**没有**同语料同抽取器的"记录层形态"消融（typed records vs 自由文本记录，喂同一个 ReAct 循环）——B3 的 rawtext 对照臂换掉的是**检索对象**（原文 chunk vs 类型化视图），不完全等价于记录层消融。
3. 修法（具体）：把 C2 的证据基座从代理指标**换成 B3 耦合交互项**（那才是"记录语义在下游被消费到"的行为学证据），代理指标降为构建质量描述；或者在 PaperScope R2 主场补臂里加一条"flat 记录+同循环"消融（材料包 §6 预算内可挤）。**C2 与耦合实验绑定陈述，禁止独立成条。**

**次要靶子 = 九件工具+七招动作手册**（材料包 §2.3）。"七招类型化动作手册"若无每招的消融或至少逐招消费频次证据，就是 prompt engineering。B3 报告 §六已有工具调用分布（findings 143/card 76/config 21/…/search 兜底仅 6）——**把"兜底 search 仅 6/286 次"这个数搬进正文**，它是"类型化工具接住了需求"的最直观证据；七招则建议降格为实现细节+附录，不进贡献列表。

---

## 2. 上轮遗留裁定：两条查新盲区轴（席位任务 2）

**先纠一个材料包自身的档案错误（MINOR）**：材料包 §7 L60 称两轴"未重查……自 v6 后未再扫"。这不准确。v6 判决当日的 P0-1 补课**已经产出**两轴的完整查新报告：DB 理论轴 = `.research_tmp/literature/survey_agent_knowledge_model_2026-09-03/report_7_db_theory_axis.md`（8 条线机制×先例×delta 对照表 + rebuttal 措辞），循证医学轴 = 同目录 `report_8_evidence_synthesis_axis.md`（v6 判决档 L21 明确记录"循证医学轴补课完成，划界段落草稿已备好可直接进论文"）。"未重查"的真实含义是**09-03 之后没有对这两轴做增量扫描**——这与"盲区未处理"是两回事，材料包把自己的补课成果漏报了。

### 2.1 DB 理论轴：裁定 = **缓解（可管理的 related work 义务），不阻断**

理由三条：
1. **查新义务已履行且质量高**。report_7 把 8 条线全部落到一手文献并给出精确 delta：c-tables（Imielinski & Lipski, JACM 1984）= 条件掩码的表示层完全占位（report_7 §3 原话"审稿人会拿这一点打'结构不新'"——伏兵已识别）；truth discovery 全线把冲突定义为待裁决误差、无"条件等价性判定"先例（§1）；view adaptation（Gupta SIGMOD 1993/1995）的定义精确给定 vs 我方漂移推断（§4）；语义缓存（Dar VLDB 1996/GPTCache）无证据变化驱动的失效语义（§7）；AGM/KM 二分完美映射失效三分类（§6）。**rebuttal 措辞和两个"审稿伏兵"（c-tables 专家、SOLARIS）都已预案**（report_7 L78-82）。
2. **主张迁移降低了暴露面**。v6 的 C2（冲突感知编译）/C3（失效语义）是 DB 轴的主要撞击点；当前材料包已把治理环自我降格为"工程实践定位，非论文卖点"（§2.4），头条主张移到了访问层非劣+耦合。DB 轴剩余的暴露点只有三个且全部有现成 delta 措辞：条件 dims 分带↔c-tables、as_of 工具↔valid-time DB（Jensen & Snodgrass 1996）、编译视图↔view maintenance。
3. **残余风险与条件**。D 面查新显示该生态 2026-05→09 占位速度以月计（MemTX/TOKI/sciltp 五连发，novelty_surfaces.md §四），"LLM × uncertain DB / c-tables / valid-time" 交点 09-03 后可能已有新入场者。**条件：投稿前做一次窄口径增量扫描**（检索式三条即够：`LLM "c-tables"`、`"uncertain database" LLM agent`、`"valid time" / "bitemporal" LLM knowledge`——注意 TOKI 已占 bitemporal 命名，见 surface_D L145），半天工作量。行文红线照抄 report_7 L80："切不可说'首次保留分歧/首次条件标注'——秒杀"。

### 2.2 循证医学轴：裁定 = **缓解（可管理的 related work 义务），不阻断**

理由三条：
1. **划界段落已成稿**。report_8 产出物在 v6 判决档 L21 记录为"可直接进论文"：evidence gap maps（Snilstveit 2016/White 2020，3ie/Campbell）与覆盖地图概念同构但纯人工流程（358 篇文献在用、社区刚开始张望 AI readiness）；Shojania 2007 证据老化生存分析=失效代数的人类流程版；SLIDERS/EviSearch 已占冲突消解与 per-cell provenance——"冲突处理"不能声称首创的边界已画好。
2. **09-06 查新给了独立交叉印证**。surface_C 一手核验 Cochrane Handbook v6.5 §5.4.3 原文："Include 'not applicable', 'not reported' and 'cannot tell' options as needed"（novelty_surfaces.md L113）——**缺席三态是 Handbook 逐字指令**，这比 v6 轮的判断更严：三态连"域迁移式新颖"都只剩半个（surface_C 判定：嫁接只能作 design lineage 不能作 claim，L127）。同时 3ie 的"干预×维度矩阵，空白=研究缺口"与 B3 最强单元 coverage 3.8（缺席三分+档案跟线索的语义红利）概念同构——**我方 coverage 类胜绩越亮，这条引用义务越重**：不引 3ie，审稿人替我引，性质就从"致敬先例"变成"隐瞒先例"。
3. **条件**：①report_8 划界段落必须实际织入 related work（当前只在档案里，材料包 §4 未提 EBM 轴一个字）；②"缺席三态""覆盖缺口"两处措辞全文禁用"首次/一等公民首创"，改为"Cochrane Handbook §5.4.3 的三态指令与 3ie evidence gap maps 的空白语义，首次在 agent 可消费的类型化记录层上自动化并实证其下游收益（coverage 带 +0.2 超直读）"——把新颖性从概念转移到"自动化+agent 消费+实测收益"三件套上，这是唯一守得住的表述。

**两轴合并裁定：均从 v6 的"CRITICAL 盲区"降级为"带条件的 related work 义务"。条件汇总：织入 report_7/8 成稿段落 + 遵守禁语清单 + DB 轴投稿前窄口径增量扫描一次。**

---

## 3. 标题/摘要级攻击面：审稿人最可能写进 meta-review 的三句话（席位任务 3）

**攻击句 1（打主结果的强度，最可能出现在 meta-review 第一段）**：
> "The headline result is a non-inferiority margin of +0.02 on 50 author-constructed questions in a single run, inside the authors' own disclosed ±0.1 per-run variance band, against an author-implemented reading baseline — and the paper's own preregistered Gate 3 (upgrade effectiveness) fails: the system ties full-document reading at 1/7 the cost but never beats it."
（锚点：B3 报告 Round-2 §二闸 2/闸 3、§八披露 1；材料包 §5.1 自认。防御=非劣性语言+两轮实录+成本 domination 叙事前置，1.2 节措辞已内置。）

**攻击句 2（打新颖性，引三件现成占位）**：
> "ASKS (arXiv:2608.29612) already defines 'scientific knowledge compilation' with provenance-preserving transactional ingest six days before this work's implementation phase; SciAtlas v2 §4.3 already assembles workflow-specific context on demand; and the absence tri-state is a verbatim Cochrane Handbook §5.4.3 instruction while 3ie evidence gap maps prefigure the coverage matrix by a decade — what remains is a well-engineered integration of known components, evaluated on the authors' own benchmark."
（锚点：asks_note.md §3/§5；sciatlas recheck L15；novelty_surfaces.md L113/L127。防御=1.3 节切割公式 + 2.2 节措辞替换 + 把差异化重量全压到耦合交互项和评测治理两条腿上——这两条上述三个占位者确实都没有。）

**攻击句 3（打外部效度，最致命因为目前无法完全防御）**：
> "The only external-benchmark evidence is oracle-scoped — all three arms consumed question-bound paper groups, as the authors themselves disclose — and is declared 'indeterminate' against direct reading; the scale claim (C4) rests on arithmetic extrapolation plus cited official numbers from a different harness and judge, with the authors' own largest self-measured corpus still in progress."
（锚点：材料包 §3 泄漏审计行 L28、§1 C4 L13；MERGED-VERDICT.md 判决一"官方数字禁止裸引"L8。**这条在 PaperScope R2 完成前无解**，是投稿时序必须等 R2 的根本原因。）

---

## 4. 过度声明逐条点名（席位任务 4）

### O1【CRITICAL】§3 E2 行与 §1 C1：引用了自家档案已推翻的结论
- **哪里错**：材料包 §3 L24 写 E2 结论为"直读也失效、失效按分散度分层、聚合最差、强模型修不好聚合"，§1 C1（L10）写"强模型修不好聚合类失效（E2，五轮）"。但项目自己的 E2-R2 修正档案（memory `e2-r2-scaling-verdict.md`，2026-09-03，预注册对照矩阵）**明文推翻了这条**："直读失效(2.83/5)是中端模型(deepseek)产物；Qwen3.8-Max 思考关直读 4.00/聚合 4.33"，并立下写作红线（L26）："论文里'直读失效'必须限定为'中端模型/规模档'"。材料包把修正前的结论当现役主张写进了头条 C1。
- **后果**：动机章建立在无效证据上；任何拿到 E2 全档案的审稿人（或复现者）会发现作者引用了自己已作废的实验——这比主张弱更糟，是可信度伤害。
- **怎么改**：C1 重锚为规模带主张（措辞见 1.2 第三贡献）："直读失效是规模带现象：同模型直读 166k 档 3.50 → 339k 档 3.00（B3 报告 §一 new flat v1.2 行，一手现役数据），93 篇档 ~800k 字符算术墙；5 篇档前沿模型直读最强（E2-R2，如实报）。"E2 原 12 任务数据只保留"失效按分散度分层"中在修正后仍成立的分带观察，并标注模型档位。§3 E2 行加一句"E2-R2 修正见 P0-VERDICT.md"。

### O2【MAJOR】§1 C4："官方 PaperScope 2000 篇档 16 个 agent 实测最高 40.95" 违反自家引用纪律
- **哪里错**：MERGED-VERDICT.md 判决一（L8）明文："PaperScope 官方数字（16 agent）**禁止裸引**：统一 Qwen-Agent harness+GPT-5 judge+200 题子集产物，无外部对比协议表述"；合法引用点只有 Gold Context 消融结论。材料包 C4 把这批数字与自有实测并列作规模主张的支撑，恰是被禁的裸引形态。
- **怎么改**：C4 拆两半：(a) 自有部分="token 预算每题 O(1) 是设计性质（固定步数帽），质量-规模关系自有实测最高 53 篇档（进行中）"；(b) 外部部分只引 Gold Context 消融结论（"直供 gold 文档仍不强"）作动机，16-agent 分数若要出现必须带 harness/judge 不可比限定句且不得与我方数字同表。

### O3【MAJOR】§1 C3 尾句与一句话方向："收益可归因到表示语义（非通用 agent 技巧）"
- **哪里错**：耦合交互项 +0.50 出自预注册闸 5，B3 报告 §二/§Round-2 二均标注"**探索性，无通过闸**"；对照臂 n=26 且沿用 round-1 产物（披露 3）、rag20 列为跨口径近似（披露 4）。用探索性单次子集实验支撑"可归因"的因果措辞，超出证据等级。
- **怎么改**："归因"→"提供受控初步证据（consistent with）"；正文引用交互项时四项披露（子集/单轮/跨口径/探索性闸）必须同页出现。1.2 节第一贡献措辞已按此写。

### O4【MAJOR】§4 "四空轴对 SciAtlas 仍成立（其 v2 自认 Limitations）"
- **哪里错**：与自家深查档案矛盾。sciatlas-v2-mechanist-recheck.md L15 判定四轴中 **(c) 需求驱动轴"部分转向（本次唯一实质修正）"**——v2 §4.3 已做访问层需求条件化投影，档案给出的指令是"写论文须切割 demand-driven access（已被占）vs demand-driven content（我独占）"。材料包写"四空轴仍成立"等于把 3.5 轴报成 4 轴，且漏掉了切割义务。
- **怎么改**：§4 该句改为"三轴半仍成立：(a) 内容对象化/(b) 演化/(d) 全文深抽取+逐字锚 v2 原文自认仍缺；(c) 需求驱动被 v2 §4.3 占据 access 半轴，我方主张限定 content 半轴并全文执行 access/content 切割"。同时（同节）补 ASKS 进生态位清单——见 O5。

### O5【MAJOR】§4 生态位清单整体遗漏 ASKS（本项目自查新档案判定的高威胁头号撞车者）
- **哪里错**：材料包 §4 L35 同生态位名单=SciAtlas/Mechanist/Lacuna/Agents-K1/PaperArena/PaperScope，**无 ASKS**；三红线（L34）也无 ASKS。而 competitor_audit_2026-09-06.md §三把 ASKS 列为"头号发现——与我们蓝图形态最趋同的新工作"，asks_note.md §5 判"高威胁（术语与治理闭环双撞车）"，§5.四行动项明文"Stage B 命名与论文叙事需重新锚定（避开裸用 knowledge compilation）+相关工作必引"。材料包 §2.2 仍在使用"编译视图"叙事而未挂 ASKS 划界。
- **怎么改**：§4 加第四条红线："ASKS（2608.29612）= scientific knowledge compilation 术语与事务性摄取闭环先占者，必引划界（它：图路线+embedding 几何整合+零基准评测；我：类型化记录+注册表治理+三臂实证）；全文禁用裸 'knowledge compilation' 措辞，或每处使用随引 ASKS。"§2.2 "编译视图"首次出现处加脚注划界。

### O6【MINOR】§1 C2 证据形态：条数当优势
- **哪里错**：C2 括号内首项证据"1010→6691 条"——记录条数增长在无质量分母时不构成优势主张（可以是切分粒度变化或膨胀），审稿人会要求 per-field 精度而不只是 first_pass 0.879。
- **怎么改**：C2 证据序重排：耦合交互项（下游行为学证据）> 引文验证 99% vs 0（可核验质量证据）> first_pass/残差率 > 条数（降为构建规模描述，不作优势证据）。与 1.4 节"C2 禁止独立成条"联动。

### O7【MINOR】§1 C4 "每题预算 O(1) 与语料规模无关"的隐含质量承诺
- **哪里错**：token 预算 O(1) 是设计事实，但句子与 C4 的规模主张连读会暗示"质量也与规模无关"。B3 报告 §四：aggregation −0.33 是最大缺口、触顶率 36%——固定步数帽下质量对广度的压力有实测证据（材料包 §5.2 自己列为头号风险）。
- **怎么改**：O(1) 句后紧跟"质量-规模关系未验证；40 篇档已实测广度题对固定步数帽的压力（aggregation −0.33/触顶 36%）"。一句话内自我限定，堵攻击句 3 的半个枪口。

### O8【MINOR】§1 C3 / §3 "成本 1/7"
- **哪里错**：分母 old flat "~9M/轮" 在 B3 报告 §五标注为 **est** tokens（估算），非直测；分子 1.277M 是直测。比值精度不该超过输入精度。
- **怎么改**：写"约 1/7（flat 侧为估算口径，B3 报告 §五）"或补 flat 臂直测 token 数（判分日志里应有）。

### O9【MINOR】§7 表"查新盲区：未重查"
- **哪里错**：漏报 report_7/report_8 两份已成稿的补课档案（见 §2 开头）。向复审面板低估自家档案完备度，会换来不必要的"仍阻断"裁定。
- **怎么改**：该行改为"v6 当日已补课成稿（report_7/8，含划界段落与 rebuttal 措辞）；09-03 后未做增量扫描，投稿前补窄口径扫描一次（DB 轴）"。

### 非缺陷但必须记录的诚实项（不构成 overclaim，材料包 §5 已自查到位）
oracle-scoped 泄漏降级披露（§3 L28）、非劣余量薄（§5.1）、出题人=我（§5.4）、单域（§5.5）、H7(d) 挂账（§5.6）、裁判协议补丁体系（§5.8）、抽取质量天花板（§5.9）——这份自查清单的完备性本身是我给"方法论贡献"加分的依据，保持原样进论文 limitation 节。

---

## 5. 修复清单（按投稿阻塞性排序）

| # | 动作 | 阻塞级别 | 工作量 | 对应发现 |
|---|---|---|---|---|
| 1 | C1/E2 重锚到规模带（措辞已给，1.2 节） | **投稿阻塞** | 半天改写 | O1 |
| 2 | PaperScope R2 第一批完成（外部考场+全自跑具名基线） | **投稿阻塞** | 已预注册，预算内 | 攻击句 3、1.1 |
| 3 | ASKS 进红线清单+全文措辞排雷（裸 "knowledge compilation" 禁用） | **投稿阻塞** | 1 天 | O5、攻击句 2 |
| 4 | C4 官方数字裸引清除，换 Gold Context 消融引用 | **投稿阻塞** | 2 小时 | O2 |
| 5 | "四空轴"改"三轴半"+access/content 切割句全文执行 | 高 | 半天 | O4 |
| 6 | one-shot 强化臂补跑（~600k） | 高（审稿必问） | 已挂账 | 1.1 前置② |
| 7 | report_7/8 划界段落织入 related work + 禁语清单执行 | 高 | 1 天（稿已有） | §2 两轴条件 |
| 8 | DB 轴窄口径增量扫描（三条检索式） | 中 | 半天 | §2.1 条件 |
| 9 | C2 证据重排并与耦合实验绑定；工具调用分布（search 兜底 6/286）进正文 | 中 | 半天 | O6、1.4 |
| 10 | "归因"→"受控初步证据"；O(1) 句加质量限定；成本比值标估算口径 | 低 | 1 小时 | O3/O7/O8 |

---

## 6. 本席判决

**判决倾向：Major Revision（对 idea）/ 条件放行（对投稿计划）。** 方向、证据弧线（B flat→B′→B3 瓶颈迁移链）与治理纪律达到我见过的同类投稿的前 10%；但头条主张 C1 引用已作废结论、外部效度悬空、最危险竞品缺席于自我定位——三者叠加意味着**现在投出去会以我最不愿看到的死法死掉：不是被证明错，而是被证明不诚实与不充分**。修复清单 1-4 完成 + R2 第一批数据在手后，WWW 2027 研究轨是本席认可的目标 venue，贡献声明按 1.2 节骨架写，主结果押耦合交互项与非劣+成本 domination，方法论押预注册治理闭环——这两条是全部竞品档案里查不到先例的东西，其余一律让渡给 related work。

*—— EIC 席，2026-09-08*
