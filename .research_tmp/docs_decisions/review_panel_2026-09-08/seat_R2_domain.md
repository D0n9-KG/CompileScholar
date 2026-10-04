# R2 席（Domain Reviewer）评审报告 — 2026-09-08

**席位人设**：科学文献 KG / 信息抽取领域资深研究者（schema 归纳 EDC/AutoSchemaKG 系、超图表示 Hyper-KGGen/HGNet/SCION 系、KG 演化 DIAL-KG/EvoGraph 系均亲手做过或审过）。
**评审对象**：`docs_decisions/IDEA-REVIEW-PACKAGE-2026-09-08.md`（只读）。
**判决倾向**：**大修后接收方向（Major Revision → 偏 Accept）**。上轮"编译 vs 缓存"CRITICAL 裁定**有条件解除**；核心证据链的实验设计在领域内属于正确且高于生态位平均水准的做法；遗留义务全部可在论文阶段补齐，无一构成方向级阻断。以下按席位任务五条逐一裁定，发现汇总见文末。

---

## 一、上轮遗留裁定："编译 vs 缓存" CRITICAL

### 裁定：解除（有条件——三项义务须在论文投稿前完成，见 F1-F3）

**理由（领域视角）**：上轮 CRITICAL 的实质是"你的视图与 DB 物化视图/预计算缓存不可区分，须能力配对对照+摊销模型"。现在的三轮证据链恰好从两个方向把主张移出了缓存解释空间：

1. **缓存解释预言的内容等价性被证伪**。若编译视图只是缓存（内容预存、访问形式无关），则同一循环下 raw-text 检索应拿到相近增益。实测耦合交互项 +0.50（Δtyped +0.69 vs Δraw +0.19，B3 round-2 终判闸5），且方向跨两轮一致（round-1 +0.423 → round-2 +0.500）、同轮配对 +0.462（11胜12平3负）。缓存无法解释"同样的循环、同样的检索预算，换表示就换增益量级"。
2. **flat 臂的算术墙是反缓存的第二证**：把缓存内容全量渲染（findings 覆盖仅 6.4%，EVIDENCE-B-REPORT）两轮 FAIL（−0.92/−0.60）——若价值在预存内容，渲染路径应当工作；它撞窗失败，说明价值在**类型化选择性访问 + 表示语义驱动的动作**，即"动作编译"而非"内容缓存"。
3. **能力配对对照已按上轮要求做了**：rawtext 对照臂 = 同 ReAct 循环、同笔记闸（chunk_id 天然合法回指）、同预算量级——这正是"能力配对"的领域标准形态。摊销模型以 quality-cost frontier 呈现（3.42 分 / 1.277M tokens vs old flat 3.50 / ~9M，成本 1/7）。

**耦合交互项在领域内够不够硬？——"及格偏上，可支撑主张，不可支撑强因果措辞"**：

- **够硬的部分**：2×2 交互设计（循环×表示）是本领域证明"组合>各部分"的正确实验形态，且比生态位平均水准高——part_A 审计显示 EvoGraph-R1/AutoSchemaKG-QA/Agents-K1 表7 的基线数字**来源悬空**（无重跑声明），而本工作有预注册冻结闸、双口径（adj/raw）、分带归因、逐题配对、两轮迭代全披露。交互项还有题型级旁证交叉（aggregation +0.73 vs tools、coverage 3.8 首次单题型超整读、拒写带翻身 3.42>3.33）——多路独立证据同向，这是审稿人最难打翻的形态。
- **不够硬的部分（四点，每点都是现成攻击面）**：
  - (a) **n=26 子集、单跑**。披露的单跑方差带 ±0.1 是单臂量级；交互项是差之差，方差复合后 ~±0.2 量级，+0.50 约 2.5σ——方向可信，量级置信区间宽。
  - (b) **跨轮跨口径拼合**：Δtyped 用 round-2 子集数字，Δraw 用 round-1 对照臂（沿用，未过 IL-C4/C5/C6 修复轮），rag20 列为 stageA-kimi 口径（无 J1）。B3 报告披露项 2/4 已如实标注，但论文若直接引用 +0.50 而不重算，等于把披露义务转嫁给读者。
  - (c) **弱单轮臂质疑（H7(d) 义务未消）**：交互项的分母格（round-2 tools，adj 2.84）本身被自家档案标注"弱单轮臂"——若该格欠调优，Δtyped 被系统性抬高。one-shot 强化臂（~600k）不跑，交互项的四格矩阵有一格是软的。
  - (d) **摊销账未含编译成本**：frontier 只算答题期 tokens，KB 构建成本（7162 条抽取+五闸+registry 治理）不在 1/7 的账里。DB 背景的审稿人（物化视图批评的原发人群）第一问必然是 total cost of ownership：建库一次+查询 N 次的摊销曲线，N 多大时 domination 成立。

### 发现

- **F1【MAJOR】交互项须同轮同口径重算**。证据锚点：B3 报告披露项 2/4（rag20=stageA-kimi 口径跨列近似；对照臂=round-1 产物）。修法：论文阶段用同一裁判协议（J1+附录）对 26 题子集的 round-2 typed / round-2 rawtext 重跑或重判一次，交互项只报同轮同口径数字；跨轮版本降级进附录作稳健性佐证。若预算允许，子集扩到全 50 题（对照臂补跑 24 题，~300k）。
- **F2【MAJOR】one-shot 强化臂（H7(d)）从"挂账"升为"耦合主张的前置义务"**。证据锚点：B3 披露项 5；pack §5.6。修法：~600k 已预估，跑完后交互项矩阵四格全部同强度；在此之前，论文措辞用"收益与表示类型耦合（coupled）"而非"收益**来自**表示语义"的强因果句。
- **F3【MAJOR】摊销模型补编译成本账**。证据锚点：frontier 表（B3 §五）只含答题成本；EVIDENCE-A（7162 输入→6691 过检）证明建库规模。修法：报"建库 tokens/篇 + 每题 tokens"双栏，给 domination 成立的查询次数阈值 N*；40 篇档下预计 N* 很小（答题一轮即省 7.7M），这反而是免费加分项，不报是浪费。
- **F4【MINOR】"编译 vs 缓存"解除后，防线要前移一格**：下一轮审稿人的同构攻击是"编译 vs 人写 SOP"（见第五条 F12）。上轮 CRITICAL 的解除不等于"编译"一词安全——ASKS 已先 6 天占用 "scientific knowledge compilation" 术语（new_entrants H2），ISC（2608.20845）标题即主张 ingest-time compilation。修法：论文命名避开裸用 "knowledge compilation"，或首现处引 ASKS 划界（图路线+零基准+56 篇案例 vs 我们类型化记录+预注册验证+需求驱动评测）。

---

## 二、记录层贡献的领域定位

### 与四家正面对照后的存活新颖面（按经得起 ablation 追问的程度排序）

| # | 新颖面 | 对照占位情况 | 下游证据现状 | ablation 耐受力 |
|---|---|---|---|---|
| 1 | **缺席三态一等公民**（not_reported 带穷尽性举证 / explicitly_stated / cannot_tell，作为可查询 KG 内容） | Cochrane Handbook §5.4.3 三态是**表单选项**先例（C面判决：必引作 design lineage 不可作 claim）；Intern-Atlas/EDC/AutoSchemaKG/DIAL-KG **均无 absence 语义**；Materials Explorer 的覆盖缺口是统计层非记录层 | **两个独立证据点**：Stage A coverage 类 4.10（首次验证缺失性信息可表达）+ B3 coverage 3.8=全表最高单元、**首次单题型超整读**（+0.2 vs old flat）；A15 教科书轨迹（find_gap 三态确认"直接对比不存在"）；记录层 not_reported 119 条带穷尽性 evidence | **最强**。机制→工具→题型增益的因果链完整，单条轨迹可演示，塌缩消融（三态→二值/删 absence 记录→重跑 coverage 10 题）成本约一个臂的 1/5 |
| 2 | **n-ary 类型化记录 + 条件 dims 词表** | new_entrants §五.3 判决：typed cross-paper relations ≥5 组独立出现（AskChem/ClaimFlow/MUSES/Intern-Atlas/Evidence-Based QD）但**全部二元**——"n-ary/条件化记录仍是空位"；XBRL→科学域双引擎零先例（C面） | conditional 题型 +0.2 vs old flat（弱正）；但 **typed 六维命中仅 13%**（EVIDENCE-A + pack §5.9 自认） | **中**。真空成立但我方填充率低——审稿人会用 13% 反打"名义 n-ary，实际二元+属性"。claim 必须与实际覆盖率对齐：写"支持 n-ary 条件化的记录模型+13% 实测填充+328 条仲裁队列可见通道"，不写"条件化记录已实现" |
| 3 | **epistemic 分轨**（stated/demonstrated/cited；二手数字可审计） | 四家均无此字段；最近邻=Mechanist 的 "unsupported/ambiguous 留空不猜"（**抽取时纪律，非记录级类型字段**——须引并划界，见 memory sciatlas-v2-mechanist-recheck）；LedgerMind ECC/NCC 是运行时接地非编译期分轨 | **零下游 ablation 证据**。当前价值是协议性的（材料源=gold源 的记录层基础） | **最弱但最便宜补**。去 epistemic 字段消融臂（预测：追源类/材料源≠gold源题型下降）一臂可证；不补则只能作协议贡献写，不得进贡献句 |
| 4 | **quote-first 数值纪律 + 确定性五闸后检链（99%/6691 规模）** | **非新颖面**：ATIBA（verbatim 定位失败即丢弃）、HGNet（人工审计 96.5%/94.2%）、AskChem（schema 校验+DOI 接地）、Citation Faithfulness 2607.20527（verifier 严格度使 unsupported 率 3%↔18%）多先例已占 | 引文验证 99% vs 0、年份 100% vs 0（EVIDENCE-A） | 作 C2 的支撑数字与工程硬度，**不作新颖性主张**；引用上述先例站肩上反而加分 |
| 5 | **注册表治理（零发明权+人工闸+版本化）** | B面判决：交点空（DIAL-KG 有环无闸 / kgg 有闸无环 / AutoSchemaKG+EDC 全自动扩），但哲学层死刑（TEIRESIAS 1979/Text2Onto 2005/Codd 1979）且**收益对照缺失** | registry v3.1 落地实例（218 表面名仲裁、DROP 列表全是引用残渣/泛称）但无对照数字 | 见第三条专节 |

### 发现

- **F5【MAJOR】C2 的归因混杂：schema 升级与管线成熟不可分**。证据锚点：EVIDENCE-A-COMPARE 诚实限定节**自认**"本对比是'带验证管线 vs 无验证管线'的整体对比，不能把差异全归因 schema"；而 pack §1 C2 的表述"类型化记录……对旧自由文本记录有可测优势（1010→6691、99% vs 0、100% vs 0）"把这些数字放在了 schema 名下。领域审稿人（EDC/AutoSchemaKG 背景）一眼看穿：v2 是单遍全文抽取无后检，三个对比数字里至少引文验证与年份两项是**管线**贡献非 **schema** 贡献。修法（二选一）：(a) C2 措辞改为"类型化记录层+验证管线作为整体对旧形态有可测优势"，并在 EVIDENCE-A 呈现时保留原诚实限定句进论文；(b) 补同管线 schema 消融臂（同一抽取器+同一后检链，v1.3 schema vs 自由文本 schema，跑抽取质量+下游两层）——成本较高，(a) 已足够诚实。
- **F6【MINOR】记录层贡献句应显式让渡谱系**： Intern-Atlas（binary+static+9.4M 边，方法演化关系类型与我方 lineage 边几乎一字不差——memory 已判"发明关系类型不能作卖点"）、EDC（None-of-the-above 自动增补=同一步骤相反权力结构）、AutoSchemaKG（zero manual intervention 卖点，ACL 2026 正会）。修法：相关工作按"占位→让渡→存活面"三段写，存活面收敛到上表 #1-#3；这恰好也是项目自家纪律（paper-novelty-must-prove：组合创新须证明真空+收益）的执行。
- **F7【MAJOR】组件级 ablation 矩阵缺失——当前只有 bundle 级证据**。证据锚点：B3 耦合实验的对照单元是"整个类型化视图 vs 原文检索"；题型级归因（coverage→缺席三态、conditional→条件带、temporal→年份双轨）是**代理归因**非干预归因。ablation 追问是本项目自家铁律（"每借鉴组件想 ablation 不可省"），也是审稿人必问题："六记录类型+四个语义特性，哪个在赚钱？temporal 至今 −0.2，是不是年份双轨根本没收益？"修法：论文阶段按上表 #1/#2/#3 各设计一个塌缩臂（absence 三态→二值；dims 条件→自由文本；epistemic→删除），每臂只跑对应敏感题型 10 题，三臂合计 ~400k 可完成；预测方向写进预注册。这是把"最经得起 ablation 追问的面"从推断变成实测的唯一通路。

---

## 三、治理环降级是否明智

### 裁定：降级方向正确，执行有三处划界风险（F8-F10）

**方向正确的领域依据**：B面查新档案的判决原文就是"收益是另一半，目前缺——必须有对照实验，否则顶会审稿大概率判 engineering choice, not a research contribution"。B臂被砍后，把治理环从卖点降为工程实践描述是**与自家查新结论一致**的诚实处置；part_A 的惯例审计也显示低 tier venue（DIAL-KG@DASFAA，2 基线零 backbone 说明）才放行无对照的机制声明，顶会线（Youtu/Hyper-KGGen/SciAtlas v2）全是 (b) 形态自跑对照。不带对照数据硬撑卖点，攻击句 1（"No controlled experiment shows the arbitration gate beats automatic expansion"）必中。

**但"审稿人会放过吗"要分句回答**：
- **会放过**：治理环只出现在方法描述+可审计工件发布（仲裁账本、版本记录、registry v3.1 对照表），不承载任何收益暗示句。A面的真空要素（迭代全日志+修改仲裁治理，四路检索零命中+Donoho 2025 权威引证）支撑的是**方法学主张**（"首个把预注册验收门柱贯穿多轮构建程序并披露仲裁记录的工作"），该方法学主张**不需要 B臂收益对照**即成立——这是降级后仍能保留的发表面，pack 目前没把它捡起来，可惜。
- **不会放过**：任何暗示"治理让 KB 更好"的句子。当前有一处隐性越线：C2 的记录层优势数字里 canonical 解析 85%、实体对齐（v2 表面名各活各的 vs v1.2 注册表）是**治理驱动的**——治理通过 C2 隐式承载了收益 claim。审稿人追问"实体对齐的收益 ablation 呢"时，降级声明救不了这句。

### 发现

- **F8【MAJOR】claim 卫生：把治理从 C2 归因链中显式摘除或显式合并**。修法：C2 措辞二选一——(a) 列优势数字时删去/脚注实体对齐项，注明"注册表治理为管线组成部分，未单独消融"；(b) 把治理并入"管线整体"表述（与 F5 修法 (a) 合并执行，一次解决两个归因混杂）。
- **F9【MINOR→机会】免费的准对照就在下一步计划里，pack 没有认领**。证据锚点：registry_v31_application.md L234 明文预测"下一个实验启动时把 registry 路径换成 registry_v31.json……预期接地噪声下降"。修法：PaperScope R2 或 one-shot-v2 强化臂启动时，做 registry v3 vs v3.1 的**接地噪声配对指标**（同题同臂只换注册表）——这不能证明"人工闸>LLM 自动扩"（那需要 EDC 式自动扩臂），但能证明"治理动作有可测下游效应"，把治理环从纯描述升为带一个实测数字的工程章节。成本≈零（本来就要换 registry）。
- **F10【MINOR】把 A面方法学主张捡回来**。修法：仲裁账本+失败轮全日志作为随论文发布的可核查工件，绑定 A面收窄口径的防御写法（必引 zemhp/Vaccaro/Thomas/EMSE RR/Donoho/Søgaard）。注意 A面时间警告：zemhp 型工作加一轮迭代日志即正面覆盖，**写作窗口以月计**——这条与投稿时机绑定（见 F14）。

---

## 四、竞品威胁重估

### 裁定：生存窗口仍成立，但材料包 §4 的必引清单对自家查新档案**不完整**（F11），且三条趋势在收窄窗口

**窗口判定（逐家）**：
- **SciAtlas v2**：四空轴判定维持——(a) 条件/配置/测量对象化仍缺（v2 Limitations 原文自认 "not yet represented as explicit objects"）；(b) 演化=periodic 非 continuous；(d) 无全文深抽取/verbatim（数据源=OpenAlex title/abstract）。**但 (c) 需求驱动轴已被部分侵占**：v2 §4.3 workflow-specific context assembly = 访问层需求条件化投影。pack §4 写"四空轴对 SciAtlas 仍成立"**过强**——准确表述是"内容层四轴仍空，访问层需求条件化已被 §4.3 部分占位"。我方主张边界必须钉死在 demand-driven **content** + 表示语义**动作编译**（表9 交集），C1-C4 目前没有踩线，但相关工作切割句（memory 09-06 已备好）必须进论文。v2 补齐全套自跑基线=生态位事实准入标准，我方 PaperScope 计划的 (b) 形态（pack §6）与之对齐，正确。
- **Mechanist**：威胁级维持高。其"需求驱动 schema 扩展（三 taxonomy）+保守抽取（unsupported/ambiguous 留空不猜）+grounding 质检"是我方 epistemic 分轨与治理叙事的**小规模同生态演练**——若该团队把 experimental layer 照 Mechanist 配方放大，(a) 轴空位直接关闭。同时它是我方动机叙事的免费弹药（原文抱怨 SciAtlas 粒度不解决细粒度概念=浅元数据 KG 撑不起深需求的同生态自证）。**必引双用**。
- **表9 交集（表示语义 pre-emptive 动作编译 × 科学文献 KB × 前瞻触发）**：09-07 playbook 判决真空仍成立（8 组 arXiv 短语阴性+Bing 双确认），按其诚实限定写 "to our knowledge"。

**2026-08/09 材料包漏掉的同生态位工作**——不是"需要新查"，而是**自家 09-06 查新档案已判定高威胁/必引、但 pack §4 未列**：

### 发现

- **F11【MAJOR】pack §4 同生态位清单缺五家自家档案已判必引者**：
  1. **ASKS（2608.29612）**——"scientific knowledge compilation" 术语占位先我方 6 天+治理闭环同源（ingest 状态机/replay/impact analysis）；零基准是它的软肋，但术语划界义务在我方（new_entrants H2 行动项原文："Stage B 命名避开裸用 knowledge compilation 或引用划界"）。
  2. **AskChem（2607.28618，NYU）**——我方记录层哲学的化学域全尺寸实现（2.4M claims/147K 篇、verbatim+DOI 接地、schema 校验、MCP agent 接口、自带基准）。new_entrants 判"必须作主对照引用"；我方五张差异牌（跨域/演化谱系/覆盖缺口/需求驱动查询/维护）要逐条写。
  3. **Materials Explorer（2606.27384）**——schema 升格（catchall>5%→人工批准命名列）/覆盖缺口可见化/跨论文数值对比**三机制先例**，与治理环叙事正面相邻；相关工作必引。
  4. **Intern-Atlas（2604.28158）**——lineage 边类型几乎一字不差+9.4M 边 binary static 大规模先例；memory 已判"必须论文显式 head-to-head，否则审稿问'为什么不是 Intern-Atlas 超集扩展'"。pack §4 完全未提，这是记录层定位的最大单点遗漏。
  5. **DIAL-KG（2603.20059，DASFAA 2026）**——in-loop schema 演化声明（无 ablation）+有环无闸；我方"首次严格验证 in-loop 收益"贡献位（innovation-final-positioning memory）的正面对照物。
  另加两条**审稿人弹药**级必引：**Fidelity Before Structure（2601.00821 v4）**——受控消融 verbatim 块胜抽取类型化工件 15.9-22.0 分的**反方实证**（对话记忆域；我方"类型化记录+原文 verbatim 双存"与其 augment-not-replace 结论兼容，须主动引用化解）；**IBM M20（VLDB AgentGraph26 workshop）**——"每层结构须证明下游收益"的结构配方文，出自 Text2KGBench 作者，审稿人大概率引用质询——好在耦合交互项正是它要求的那种证明，主动引用可转守为攻。
  修法：pack §4 与论文相关工作按"访问层红线（现有三条+LedgerMind）/记录层红线（Intern-Atlas/DIAL-KG/AskChem/ASKS/Materials Explorer）/反方与质询弹药（Fidelity Before Structure/IBM M20/Citation Faithfulness/Ranked by the Matcher）"三层重组。
- **F12【该查方向（领域嗅觉，不联网）】**：09-06 扫描的覆盖缺口自己列了三条未送达通道（HF papers/GitHub/X + Bing/DDG 补充轮），叠加我的领域判断，投稿前值得再扫一次的方向按优先级：(a) **ACL/EMNLP 2026 industry track camera-up**——SCAIR 本身就是 ACL 2026 Industry，arXiv-first 的扫描系统性漏 industry track，而"schema×agent"正是工业界主场；(b) **ICLR 2027 OpenReview 投稿季动向**（9 月底开闸）——SciAtlas v2 的去处与同生态位投稿潮；(c) **FutureHouse 线**——PaperQA2 全文至今档案缺失（part_A 证据缺口表明列），其后续（LitQA 系/新工具）是我方 PaperScope 计划的直接对照臂；(d) **ScholarQABench 同基准后续**——SciAtlas v2 已在该基准放数字（引文质量 biomed 49.7→56.2），任何跟进者都会把跨论文对比强加给我方；(e) **agent-memory 工业线**（Mem0/Zep/Letta）加科学文献模板的产品动向；(f) **NeurIPS 2026 D&B track 决定**（9 月下旬）——本生态位基准论文集中地；(g) 2609.02129（Persistent Discovery Context，编译上下文持久复用胜重复检索=我方 F12"检索式访问有损"同构外部佐证）所在集群的后续。
- **F13【MINOR】窗口收窄的速度证据应进投稿时机决策**：D面五连发以月计、ASKS 术语占位以天计、"compilation" 叙事 ≥4 组独立提出（new_entrants §五.1）、A面 zemhp 型演进一轮即覆盖。领域判断：**内容层四轴的窗口比访问层表9 交集的窗口关得快**（Mechanist 配方放大即关 (a) 轴）。修法：论文优先级=记录层内容主张先行锁定，访问层主张次之；投稿不晚于 2026 年内。

---

## 五、三红线区分表述精度

### 裁定：档案层精确，pack 层过简；一条真实的反打路径未被现有红线覆盖（F14）

逐条核对（对照 VERDICT_playbook_lineages 判决原文）：

1. **Doctor-RAG（2604.00865）**——pack 只写"error-type→fix 映射"。档案层的区分是精确的：其 𝒮:𝒞→ℱ 映射的**键=自身轨迹的失败类型，时机=post-hoc 修复**（原文自述 "post-hoc repair setting"）；我方键=KB 记录类型、时机=pre-emptive 前瞻引导。**风险**：pack 级表述若直接进论文，审稿人一句"同一个映射换了个键"就能打——区分必须把三轴（规则来源×触发时机×载体）同时写全，判决原文的"三点差异化必须同时守住"是硬要求。
2. **CEL（2509.25052）+MS Foundry**——档案层区分正确：CEL 的 playbook 由 post-episode 轨迹反思蒸馏（试错累积，grid-world 域），MS Foundry procedural memory 来源=轨迹审计；我方来源=表示语义。**但这条红线旁边站着一个 pack 和判决表都没有正面处理的更近邻居：Agent-S 类"人写 SOP"**——见 F14。
3. **SCAIR（2607.22571）**——档案层区分精确：schema 作**负向约束**（relation paths 白名单过滤遍历）非正向动作指令，域=企业 CMDB。附加纪律：**"schema-conditioned agentic reasoning" 这个短语我方 claim 句禁用**（名字已被占），pack 未写明这条，补上。
4. **LedgerMind（2607.28374）**——"运行期账本 vs 编译期记录层"方向正确，且 verifier 档案已给出更细的存活位（宿主空白=预编译记录层×运行时 record_id 闸；耦合空白=证据覆盖定停止+记录回指定可答合一）。**但档案同时点名两个"必直面挑战"pack 未提**：Parsing the Stream（裸 scratchpad 能以更低成本匹配 fold 准确率⇒须 record_id vs scratchpad 对照）与 ECHO（provenance 可审计≠答题更好）。现有 rawtext 对照臂只部分回应前者（chunk_id 合法回指=ID 供给对照，但非"无回指自由笔记"对照）。

### 发现

- **F14【MAJOR，本轮新识别的最深反打路径】"你的动作手册就是人写 SOP——'编译'在哪里？"** 七招动作手册是设计者手写的（react_loop.md，每招指认消费的表示语义=设计理据），不存在一个从 schema v1.3 机械派生动作规则的编译过程。审稿人（尤其 DB/IE 背景）会把上轮"编译 vs 缓存"的攻击同构平移过来："Agent-S 人写 SOP、你的七招也是人写，'表示语义 pre-emptive 动作编译'的'编译'是隐喻不是机制——表9 的真空主张建立在一个修辞区分上。"这是表9 交集主张的**承重墙**，当前无预置回应。修法（按强度排序）：(a) **把 type→action 映射表做成一等工件**：schema 记录类型/标记（cited、absence 三态、dims、record_id）→ 工具可供性 → 手册招式的显式映射表，随 schema 版本化，并演示一次机械再推导（schema v1.1→v1.3 升版时映射表哪些行随之变更——registry v3.1 与三次 schema 升版的档案里应能重建至少一个案例）；这让"编译"从隐喻变成可核查的派生关系，且与治理环工件发布（F10）共用基建。(b) 退一步的措辞防线：claim 句用"representation-informed action compilation（设计期从表示语义系统派生并随 schema 共同版本化）"，明示编译发生在设计期+维护期而非运行时——诚实且仍可防守三轴区分。(c) 不可选的退路：删"编译"改"设计"——那会连带弱化 C3 与表9 交集，等于自撤主卖点。
- **F15【MINOR】红线表述的执行清单**：①三条红线区分一律带三轴全写（来源×时机×载体）；②"schema-conditioned agentic reasoning" 短语禁用；③LedgerMind 节补 Parsing the Stream/ECHO 两挑战的正面回应段（rawtext 对照=部分回应的声明+record_id vs 无回指 scratchpad 的挂账或补跑）；④MS Foundry 作谱系②工业证引用时注明其来源=轨迹审计（与CEL 同侧），不与我方同侧。

---

## 六、发现汇总

| # | 严重度 | 一句话 | 锚点 |
|---|---|---|---|
| F1 | MAJOR | 耦合交互项须同轮同口径重算（跨轮+跨裁判拼合不可直接进论文） | B3 披露项 2/4 |
| F2 | MAJOR | one-shot 强化臂升为耦合主张前置义务；此前措辞用 coupled 不用"收益来自" | B3 披露项 5 |
| F3 | MAJOR | 摊销账补编译成本（TOO 是 DB 背景审稿人第一问） | B3 §五 frontier |
| F4 | MINOR | "编译"术语避撞 ASKS/ISC，首现划界 | new_entrants H2/§五.1 |
| F5 | MAJOR | C2 归因混杂（schema vs 管线），措辞改"记录层+验证管线整体" | EVIDENCE-A 诚实限定节 |
| F6 | MINOR | 记录层贡献句显式让渡 Intern-Atlas/EDC/AutoSchemaKG 谱系 | memory 定位史+B面档案 |
| F7 | MAJOR | 组件级 ablation 矩阵缺失（absence 塌缩/dims 塌缩/epistemic 删除三臂，~400k） | B3 题型归因=代理非干预 |
| F8 | MAJOR | 治理从 C2 归因链显式摘除或并入管线整体表述 | EVIDENCE-A 实体对齐节 |
| F9 | MINOR | registry v3 vs v3.1 接地噪声配对=免费准对照，下一步实验顺手做 | registry_v31 L234 |
| F10 | MINOR | A面方法学主张捡回（仲裁账本+迭代日志作可核查工件），窗口以月计 | surface_A 判决 |
| F11 | MAJOR | pack §4 必引清单缺五家自家已判高威胁（ASKS/AskChem/Materials Explorer/Intern-Atlas/DIAL-KG）+两条弹药（Fidelity Before Structure/IBM M20），按三层重组 | new_entrants H1-H3+§五 |
| F12 | — | 投稿前再扫方向：industry track camera-up/ICLR27 投稿季/FutureHouse 线/ScholarQABench 后续/memory 工业线/NeurIPS D&B | part_A 缺口表+领域判断 |
| F13 | MINOR | 内容层窗口比访问层关得快，记录层主张先锁定，投稿不晚于年内 | new_entrants §五.1+D面 |
| F14 | MAJOR | 新识别最深反打："动作手册=人写 SOP，编译是隐喻"——type→action 映射表做成版本化工件+机械再推导案例，或措辞降为设计期编译 | Agent-S 类+表9 判决 |
| F15 | MINOR | 三红线执行清单（三轴全写/SCAIR 短语禁用/Parsing the Stream+ECHO 正面回应） | playbook 判决+verifier 档案 |

**遗留 CRITICAL 三条对照本轮裁定**：编译 vs 缓存=**解除（有条件，F1-F3）**；噪声放大=维持"部分缓解"（材料覆盖率 50% 审计+绑定错误 J1 实锤已把噪声量化为覆盖缺口，但端到端传播系数仍无系统实验——与本席 F7 组件消融可合并推进）；负载循环性=维持"部分缓解"（PaperScope 外部题+官方复刻臂是正确的锚，泄漏审计的 oracle-scoped 降级披露与 R2 全语料形态改造是领域内标准的协议卫生，出题人=我 的残留由外部基准稀释后可接受）。

*R2 席完毕。本报告仅含本席意见。*
