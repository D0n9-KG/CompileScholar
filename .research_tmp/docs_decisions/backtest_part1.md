# Schema v1 表达力回测——批次1（A01–A15）

- 回测对象：`STAGEB-EXTRACTOR-SPEC-v1.md`（2026-09-05 定稿）§1 记录类型 + §2 维度词表 + §3 Stage 0 manifest
- gold：`experiments/e2_need_gap/gold_work/gold_batch1_calibrated.json`（15 题，全部 aggregation 型；must 64 点 + bonus 24 点）
- 判定口径：must 逐点 E/P/G/U；bonus 全 must 完成后同样过一遍并单独标记。组装级需求（跨篇对比/排序/聚合/计数/时间线）按框架规则1归编译层，只要每个原子事实有记录落位即判 E。年份按规则2（manifest 双年份 + lineage 时序）。条件限定词按规则3对五维。negative relation 按规则4逐条专列。缺席断言按规则5对 absence 三态。
- 纪律：宁严勿松；每条 P/U 附 gold 摘录（≤30字）。

## ① 逐题判定表

must 计数格式 E/P/G/U；bonus 单列。

| qid | type | must点数 | must E/P/G/U | bonus E/P/G/U | 缺口一句话 |
|---|---|---|---|---|---|
| A01 | aggregation | 4 | 4/0/0/0 | 3/0/0/0 | 无（二手数字=epistemic:cited；双协议=dims.setup；漂移检测归编译层） |
| A02 | aggregation | 4 | 4/0/0/0 | 1/0/0/0 | 无（机制自述/前作批评=finding.stated+scope_ref；改进链=lineage improves） |
| A03 | aggregation | 4 | 2/2/0/0 | 1/0/0/0 | **主张级冲突关系无类型化表达**（"Fedus点名反对Z&S"只能落 finding 文本） |
| A04 | aggregation | 4 | 4/0/0/0 | 2/0/0/0 | 无（组件映射=lineage component_of+citation_context；语料级缺口=编译层派生absence） |
| A05 | aggregation | 7 | 7/0/0/0 | 1/0/0/1 | must 无缺口（ICM缺席=absence not_reported 正中设计）；bonus 表头错位警示的记录层传播存疑 |
| A06 | aggregation | 6 | 6/0/0/0 | 2/0/0/0 | 无（吞吐口径=measure.unit+_dims；冠军判定归编译层） |
| A07 | aggregation | 3 | 1/0/0/2 | 1/0/0/0 | **field-level 批评的 finding.scope_ref 绑定未定义**（"RL惯例实践"不是可注册方法） |
| A08 | aggregation | 5 | 5/0/0/0 | 1/0/0/0 | 无（表内二手基线=cited；语料外方法=registry citation_context） |
| A09 | aggregation | 3 | 3/0/0/0 | 1/0/0/0 | 无（"超人类"三操作化=定义性 finding；年份=manifest；协议=dims） |
| A10 | aggregation | 3 | 3/0/0/0 | 2/0/0/0 | 无（建议清单=finding.stated；采纳时间线=setup维度值共享+manifest年份 join） |
| A11 | aggregation | 5 | 5/0/0/0 | 2/0/0/0 | 无（条件性null结果=finding+condition dims；KL vs |TD|信号=config item） |
| A12 | aggregation | 5 | 5/0/0/0 | 2/0/0/0 | 无（数值消融=role:ablation+variant+delta；定性排序=finding；图only数字=absence） |
| A13 | aggregation | 3 | 3/0/0/0 | 2/0/0/0 | 无（协议表=config/dims 逐篇；可比组分带=编译层 dims 等价类） |
| A14 | aggregation | 4 | 4/0/0/0 | 1/0/0/0 | 无（"替代ε-greedy"=lineage replaces，机制实体需入 registry） |
| A15 | aggregation | 4 | 4/0/0/0 | 1/0/0/0 | 无（head-to-head 缺失=编译层派生 absence+单篇 not_reported absence） |
| **合计** | | **64** | **60/2/0/2** | **23/0/0/1** | |

## ② GAP + PARTIAL + UNCERTAIN 明细清单

### DET-1（PARTIAL，根因：主张级冲突/对话关系）——A03 MUST[2]、MUST[3]

- **gold 摘录**：M2 "直接不同意Zhang & Sutton（原文点名）"；M3 "显式对话关系：Fedus原文引用并反对Z&S"
- **现状**：事实内容可保存——finding{claim:"fixed replay ratio result disagrees with Zhang & Sutton (2017)…", strength:stated, paper_id:Fedus, quote 锚定}；但 lineage 是**方法级**且七词全是正向词，**finding↔finding 的"反驳/不同意"无类型化边**。编译层要渲染"三家对话关系"（gold gates 自述这是"检索最难部分"）只能对 claim 文本做 NLP，无结构 join 键。
- **丢失的部分**：机器可 join 的冲突关系结构（谁的哪条结论反对谁的哪条结论）。
- **建议归属**：**新 relation 词（主张级）**——如 contradicts/disputes，端点允许指向 finding 记录（或 lineage 增设 claim 级端点类型）。与规格 §7 预留的 negative relation 仲裁合并处理（见③结论）。

### DET-2（UNCERTAIN，根因：finding.scope_ref 指称类型未定义）——A07 MUST[0]、MUST[1]

- **gold 摘录**：M0 "实现细节差异对性能影响巨大"（领域级批评）；M1 "RL惯例训练与测试用同一环境"（实践级批评）
- **两种理解**：(a) 规格对 scope_ref 未定义指称类型 → 可空/可指领域概念（"deep RL 评测实践"）→ E；(b) 按 lineage method_ref 类比，scope_ref 须指向 registry 注册实体 → 领域级批评无绑定对象 → PARTIAL。批评性论文（Henderson/Cobbe 型）的核心产出恰是领域级主张，此绑定决定 finding 类型能否承载整个"评测方法论批评"体裁。
- **实验 scoped 部分不受影响**：HalfCheetah 双5-run不同分布=finding+condition(repeats:5 runs, subject:HalfCheetah)；CoinRun 过拟合=result/finding(subject:CoinRun)；正则化开关=dims.variant。
- **建议归属**：**字段扩展/规格澄清**——v1.x 明确 scope_ref 可空 + 可指 registry 概念实体（概念实体经 Stage 1.5 注册），或增设 scope_level ∈ method/family/practice。连带影响：A07 M2 的"重叠"综合（判 E，规则1）其 join 键依赖本项裁定；A02 M3（FQF 批三个前作）、A06 B1（R2D2 对比 IMPALA）的多目标主张可按目标拆成多条 finding 共享 quote，不受阻。

### DET-3（UNCERTAIN，根因：解析质量警示无记录层字段）——A05 BONUS[1]

- **gold 摘录**："表头解析有错位，引用须谨慎标注"
- **两种理解**：(a) quote-first + Stage 3 loc 回源 + 审计日志已承担数据质量追踪，记录层无需新字段 → E；(b) 审计日志不随记录走，编译/渲染层引用该数字时"表头错位"警示丢失，gold 要求的"谨慎标注"无法自动传播 → PARTIAL。数值本身（11,539.69±1,227.71）可落 result，无争议。
- **建议归属**：**字段扩展（轻量）**——公共字段增设可选 `quality_flag`（如 suspect_binding），或明确由编译层从 Stage 3 日志回查（需在块2规格接住）。

### GAP：0 条

批次1未发现 schema 完全无法表达的原子事实。

## ③ negative relation 案例专列

七词封闭集 {extends, improves, uses, component_of, replaces, compares_with, motivated_by} 无否定词。批次1所有"负向"表述逐条如下：

| # | qid·点 | gold 摘录 | 性质 | 判定与编码 |
|---|---|---|---|---|
| N1 | A03·M1 | "a large replay buffer can significantly hurt the performance" | 组件负效应（buffer大小→性能），非方法-方法边 | **E**：finding{strength:demonstrated}+config(item:buffer size)+dims.budget；不需否定 lineage 词 |
| N2 | A03·M2 / A11·M2 / A15·M2 | "prioritized experience replay does not significantly affect…" | **条件性 null 结果**（最接近"A fails to improve B"的压力案例） | **E**：finding{claim 原文, strength:demonstrated, condition:dims(budget:10M buffer)}；编码为实验 null 而非 lineage 否定边。若仲裁坚持 lineage 级否定表达，此条是代表案例 |
| N3 | A03·M2 / A12·M2 | "去掉n-step的Rainbow不再受益，中位数-2.3%" | 负 delta 消融 | **E**：result{role:ablation, dims.variant:"−n-step", delta:−2.3%}——拍板A直接覆盖 |
| N4 | A04·M2 / A14·M3 | "移除NoisyNets（退回ε-greedy）聚合性能变差" | 定性负消融（正文无数值） | **E**：finding{claim 原文, condition:dims(variant:"−NoisyNets")}；注：拍板A的 role:ablation+delta 预设数值，定性消融回落 finding，variant 绑定经 condition.dims 保留，不丢内容 |
| N5 | A05·M0 | "Pitfall仍为负（-155.97）" | 负分数（非负关系） | **E**：result{measure.value:−155.97} |
| N6 | A12·M2 | "去PER的Rainbow仍+17.3%" | 移除组件仍正增益（削弱组件必要性） | **E**：result{role:ablation, variant:"−PER", delta:+17.3%} |
| N7 | A11·M4 | "其边际作用被后续证据削弱" | 跨篇综合判断 | **E（编译层）**：对 N2/N4 类记录的组装，非记录级需求 |
| N8 | A03·M2/M3 | "直接不同意 Zhang & Sutton" | **主张级否定（论文结论反对论文结论）** | **PARTIAL**——见 DET-1，批次1唯一真触发的否定表达缺口，且不在 lineage 方法边上而在主张边上 |

**专列结论**：批次1未触发方法-方法级否定 lineage 边需求（如演化题"X 未能改进 Y"）；负向表达全部是 null/负实验结果，finding/result+dims 承载得住。真正的否定性结构压力出现在**主张级冲突**（N8/DET-1）。警告：本批 15/15 全是 aggregation 型，方法间否定断言天然稀少，七词封闭集的否定缺口**不能被本批清零**，须待批次2/3（对比/演化/配置型题）复测。

## ④ 统计汇总

| 层 | 点数 | E | P | G | U |
|---|---|---|---|---|---|
| must | 64 | 60（93.8%） | 2（3.1%） | 0 | 2（3.1%） |
| bonus | 24 | 23 | 0 | 0 | 1 |
| **合计** | **88** | **83** | **2** | **0** | **3** |

- 题级：15 题中 13 题 must 全 E；A03 是唯一含 PARTIAL 的题（2点，同一根因）；A07 是唯一 must 含 UNCERTAIN 的题（2点，同一根因）；A05 的 U 在 bonus。
- 根因去重后实际结构问题只有 **3 个**：DET-1 主张级冲突关系（P）、DET-2 finding.scope_ref 语义（U）、DET-3 解析质量警示传播（U）。
- 高频承重落位（E 判定依赖）：result+dims（协议/数值事实主力）、epistemic:cited（二手数字：A01/A04/A06/A08/A09 共 5 题出现，设计正中）、absence 三态（A05 ICM/A15 head-to-head/A04+A12 图only，3 种缺席形态全部落位）、finding.stated（机制自述/定义/建议/批评——**系统性过载**，见⑤）、config（优先级信号/协议字段/硬件项）、lineage citation_context（组件溯源/语料外引用）。

## ⑤ 诚实注记

1. **样本偏置（最重要）**：批次1 全部是 aggregation 型——数值+协议事实密集，恰是 Cochrane 五元组+dims 的主场。G=0 **不可外推**到 50 题全集；negative lineage 边、演化叙事（shift 类型在本批**零次被 must 点直接调用**）、配置细节型缺口在本批代表性不足。
2. **finding 类型过载（观察级，非缺口）**：机制自述（A02 全部、A06 架构创新）、前作批评（A02/A08）、定义操作化（A09"超人类"三口径）、建议清单（A10）、定性消融（N4）、研究主题自述（A06 R2D2 drift/staleness）全落 finding.stated。表达力成立，但若四视图编译需要区分这些子体裁（如"批评"视图 vs "机制"视图），建议在 v1.x 考虑 claim_type 细分或靠编译层分类——需在块2规格核对。
3. **编译层义务清单（本批 E 判定的依赖项，回测只测记录层）**：①协议可比性判定与分带（A01 M3/A09 M2/A10 M1-2/A13 M1-2）②派生 absence 与来源标记区分（A04 M3/A15 M1）③时间线合成（A05 M6/A09/A10 B0 采纳链）④跨篇数值漂移检测（A01 B2/NGU 摘要1344.0 vs 表1354.4 同篇双值——去重指纹不同两条都留，冲突检测归编译层）⑤排序/冠军判定（A06 M5/A12 M3）⑥主张相似度重叠判定（A07 M2）。若块2规格未接住，相应 E 判定贬值。
4. **维度值归一化是关键路径**：no-op starts vs "up to 30 no-ops"、Atari-57/55/49/26@100k 四档 subject 值、human-starts 双协议——A01/A10/A13 的可比组划分全部依赖 Stage 1.5 词表 vN 把这些表面变体归一到同一 explicit 值。schema 结构够，但词表质量决定这些题的成败（与冒烟验收联动）。
5. **registry 须容纳机制级实体**：ε-greedy、熵正则、n-step returns、distortion risk measures、sticky actions 等非"方法"实体是 lineage（replaces/uses）与 config 的端点/项——paper_card 的 related_methods 已预留入口，但 Stage 1.5 注册规格宜明示"机制/协议实体可注册"。
6. **语料外引用年份仅非结构化保留**：A04 M0 的"Sutton 1988 教科书"年份存在于 method_ref.surface 与 quote 中，无结构化字段（manifest 只覆盖语料内论文）。按框架规则2判非缺口（gold 答题依赖的是 Rainbow 的引文原文），但若未来题要求"按提出年份排序语料外祖先"，此处会升级为缺口。
7. **absence 三态对"图 only"内容的语义弹性**：A12 M4/A04 B0"数字在图3不在正文"——论文其实报告了（图里），只是文本抽取不可达。判 E 靠 not_reported+evidence 说明承载，三态枚举没有"已报告但抽取不可达"专态；图表数字化资产若接入，此处语义需复核。
8. delta 字段的基线模糊案例（"previous research" A06 M2、"previous SOTA" A08 M1/M3）依赖 registry canonical 化压力测试，quote 保底不丢内容，判 E 但列入冒烟抽检关注项。

## 附：must 点逐条映射（E 点的 kind+关键字段一行记录）

<details>
<summary>A01–A15 逐点映射（88 行）</summary>

- A01 M0 **E**：result×N{measure{metric:median human-normalized, value, unit:%}, dims.subject=Atari-57, setup=no-op/human-starts, budget=帧数}；二手（R2D2/IMPALA）=epistemic:cited+paper_id=NGU；硬件（376核/8TPU/5天）=budget(算力)+setup(定性硬件)；NGU 摘要/表双值=两条 result（值指纹不同，去重规则保双条）；CHNS 100%=result{metric:局超人类比例}
- A01 M1 **E**：result(dims.subject=Atari-49)；DQN 79%/DDQN 118% 自 QR-DQN 表=epistemic:cited
- A01 M2 **E**：PER 两组 result（subject:49/57 各一档，delta.baseline=DQN/DDQN）
- A01 M3 **E**：编译层对 dims(setup/subject/budget) 的可比性推导（规则1）
- A02 M0 **E**：finding(stated, 分布式视角/Bellman用于分布) + config(item:参数化, value:固定支撑+可变概率) + finding(distributional instability, scope_ref=C51, condition)
- A02 M1 **E**：lineage(QR-DQN improves C51, explicit_claim) + finding(C51不保证最小化Wasserstein, scope_ref=C51, paper_id=QR-DQN, stated) + config(转置参数化) + result(Huber 33% delta, baseline=C51)
- A02 M2 **E**：lineage(IQN improves QR-DQN) + finding(离散分位数限制批评, scope_ref=QR-DQN) + finding(τ~U([0,1])重参数化/样本数可调/风险敏感策略类, stated)
- A02 M3 **E**：finding×3(单侧参数化批评, scope_ref=C51/QR-DQN/IQN 各一条, 共享quote) + lineage(FQF improves ×3) + finding(双轴全参数化) + result(55游戏, subject=Atari-55)
- A03 M0 **E**：finding(问题重构=采样, stated) + config(PER item:优先级信号, value:|TD error|) + result(41/49局, 48%→106%, delta.baseline=uniform DQN)
- A03 M1 **E**：finding(大buffer伤害, demonstrated) + result/finding(setup维:tabular/linear/非线性FA) + config(CER机制)
- A03 M2 **P**：容量×算法效应=finding+result(Fedus数字)✓；"直接不同意Z&S（原文点名）"仅 finding 文本无类型化冲突边→DET-1
- A03 M3 **P**：三家问题重构=各自 finding✓、不一致综合=编译层✓；"Fedus引用并反对Z&S"对话关系→DET-1
- A04 M0 **E**：lineage(component_of ×6, from=各组件, to=Rainbow, evidence_basis=citation_context, paper_id=Rainbow)；Sutton 1988 年份在 quote/surface（注记6）
- A04 M1 **E**：result×N(各源论文自报, dims 带协议)；DDQN 118%=cited(paper_id=QR-DQN)
- A04 M2 **E**：finding(排序+"removing either caused a large drop"定性, quote) + result(metric:全Rainbow优于消融版局数, value:53/57) + finding.condition(variant:−组件)保留消融绑定
- A04 M3 **E**：编译层派生 absence（method=n-step 的独立增益 result 记录不存在；规格§1.5边界明示）
- A05 M0 **E**：result(A3C+伪计数, subject=Montezuma, budget=200M帧, mean 142.50) + result×2(baseline_comparison: A3C 0.06/DQN 0.02) + result(Pitfall −155.97)
- A05 M1 **E**：absence{subject:ICM, missing:Montezuma/Pitfall评测, type:not_reported, evidence:全文扫描+实评游戏清单}（举证责任正中设计）+ finding(好奇心=特征空间预测误差)
- A05 M2 **E**：result(22/24房间; best return 17,500, aggregation=best) + finding(SOTA声明, stated/demonstrated) + finding(RND机制)
- A05 M3 **E**：result(43,000+, variant:无领域知识; 65万+, variant:+领域知识) + delta(≈4×前SOTA) + finding(Pitfall首破零声明) + finding(记忆+回访机制)
- A05 M4 **E**：result(8,400 mean; 16.8k±6.8k, aggregation=mean±std, repeats) + finding(首个非零, condition:无演示/手工特征) + finding(两级内在奖励)
- A05 M5 **E**：result(9352.01±2939.78; baseline_comparison:Human 4753.30) + finding(全57首超人类声明) + result(Skiing, budget=78B帧) + lineage(Agent57 extends NGU)+config(bandit自适应)
- A05 M6 **E**：编译层时间线（manifest年份+lineage+各first声明finding；规则1+2）
- A06 M0 **E**：finding(异步梯度+并行actor-learner稳定效应/无需回放) + result(训练时间减半, setup:单机多核CPU) + result(16线程≥1个量级加速, metric:speedup)
- A06 M1 **E**：finding(actor-learner分离+V-trace) + result(250k帧/秒; 21B帧/天, measure.unit) + delta(30×, baseline=单机A3C)
- A06 M2 **E**：finding(中央推理) + result(millions frames/sec; 11× vs IMPALA, setup:8 TPU v3) + finding(80× vs "previous research", stated——基线模糊见注记8)
- A06 M3 **E**：finding(分布式PER架构) + budget(376核+1GPU/5天/22800M帧) + result(434% no-op)
- A06 M4 **E**：finding(RNN+存序列回放/burn-in/drift与staleness研究主题) + config(burn-in) + result(1920.6%, epistemic:cited, paper_id=NGU) + finding(quadruples SOTA声明)
- A06 M5 **E**：编译层——measure.unit(帧/秒 vs 加速比 vs 总帧/天)+dims 口径对比与冠军判定（规则1）
- A07 M0 **U**：HalfCheetah双5-run=finding(condition:repeats=5runs, subject=HalfCheetah)✓；领域级批评簇（可复现性/实现细节/报告规范）=finding 但 scope_ref 绑定未定→DET-2
- A07 M1 **U**：CoinRun实验=result/finding(subject=CoinRun, variant=正则化开关)✓；"惯例同环境训练测试"实践级批评→DET-2
- A07 M2 **E**：编译层对两组 finding 的重叠综合（规则1）；join 键依赖 DET-2 裁定（注记于明细）
- A08 M0 **E**：finding(对比目标机制) + result(1.2×; 19/26局超基线; median 0.175, epistemic:cited, paper_id=SPR)
- A08 M1 **E**：finding(target projection+多步潜在动力学) + result(0.415; no-aug 0.307, variant维) + delta(+55% vs 前SOTA) + result(超人类7局) + budget(100k步=400k帧≈2小时)
- A08 M2 **E**：result(mean 1.904/median 1.160; 14/26超人类) + delta(+170%/+180%, baseline=SPR) + finding(首次2小时数据超人类声明) + lineage(uses/component_of: MCTS, value prefix, consistency loss, reanalyze)
- A08 M3 **E**：编译层演进链（lineage improves/extends citation_context + manifest年份）+ result(DQN 2.20/0.959, cited, budget=200M帧——500倍比值归编译层)
- A08 M4 **E**：result×5(SimPLe/DER/OTRainbow/CURL/DrQ, epistemic:cited, paper_id=SPR)；语料外方法入 registry(citation_context)；归属修正部分=gold 校准元数据，非记录层需求
- A09 M0 **E**：finding×3(操作化定义: ≥75%专业测试者/中位数>100%/全57超基线, stated, 各带quote) + result(29/49局) + budget(50M/78B帧)
- A09 M1 **E**：result×N(各方法中位数+dims协议) + finding(各声明如NoisyNets sub→super) + manifest(年份) + cited(二手1920.6%)
- A09 M2 **E**：编译层对 dims(budget 50M→22800M/subject 49-57/setup 起手/基线定义finding)的不可比推导（规则1）
- A10 M0 **E**：finding×5(Machado建议清单, stated, paper_id=ale_revisit)；sticky机制=finding(driver式内容)；④报告不同间隔=measure.timepoint 已预留
- A10 M1 **E**：编译层——两系 result 记录按 dims.setup(no-op/human-starts)分组对比（双协议数字 A01 已录）
- A10 M2 **E**：编译层跨带配对无效性推导（dims 不等价）
- A11 M0 **E**：result(41/49, 48→106 delta; 111→128 delta, subject:57)
- A11 M1 **E**：finding(two most crucial, quote) + result(53/57) + role:ablation 系记录（paper_id=Rainbow）
- A11 M2 **E**：finding(PER不显著@大buffer, demonstrated, condition:budget/setup=10M) + result(去PER仍+17.3%, role:ablation, delta)——negative 专列 N2
- A11 M3 **E**：finding(Z&S对PER的预期陈述, stated, paper_id=replay_analysis)
- A11 M4 **E**：编译层证据状态综合；config(Rainbow变体 item:优先级信号, value:KL loss) 提供与 PER 原作 config(|TD|) 的结构化对照键
- A12 M0 **E**：finding(定性排序+quote) + result(53/57)；正文无数字的诚实性由 absence 承载（同 M4）
- A12 M1 **E**：result×5(role:ablation, variant:−projection/1-step/non-temporal/no-aug, delta 对 0.415)
- A12 M2 **E**：result(role:ablation, variant:−n-step, delta:−2.3%, dims.budget:大buffer) + result(−PER +17.3%)
- A12 M3 **E**：编译层分协议带内排序（规则1；dims 提供协议带键）
- A12 M4 **E**：absence{subject:Rainbow各消融精确中位数, missing:正文数值, type:not_reported, evidence:仅图3}（注记7语义弹性）
- A13 M0 **E**：config/result dims 逐篇（subject=游戏集, setup=起手/sticky, budget=帧/步）；QR-DQN调参5游戏=config(item:调参集)；PER双臂=两组dims
- A13 M1 **E**：编译层 dims 等价类分组（四带）
- A13 M2 **E**：编译层跨带不可比推导
- A14 M0 **E**：finding(NoisyNets原作独立实验, paper_id=noisy_nets) + result×3(对DQN/Dueling/A3C)
- A14 M1 **E**：lineage{from:NoisyNets, to:ε-greedy, relation:replaces, scope:DQN/Dueling} + lineage(to:熵正则, scope:A3C)（机制实体入registry，注记5）+ finding(状态相关扰动 vs dithering 机制差异)
- A14 M2 **E**：result×3(83→123/132→172/80→94, delta+baseline) + finding(sub→super声明) + finding(免ε调参)
- A14 M3 **E**：finding(Rainbow消融定性, condition:variant=−NoisyNets)——negative 专列 N4
- A15 M0 **E**：config×2(item:优先级信号; PER=|TD error| / Rainbow变体=KL loss, 各带quote) + finding(KL噪声鲁棒论证, stated)
- A15 M1 **E**：编译层派生 absence（无 KL×TD head-to-head result 记录）+ absence{subject:Rainbow, missing:信号类型对比, type:not_reported, evidence:仅ω鲁棒性实验} + finding(ω robust, stated)
- A15 M2 **E**：finding(Fedus null 含 Dopamine Rainbow=KL 优先级——config归属) + config/lineage(Ape-X/R2D2 uses TD系优先级, citation_context)
- A15 M3 **E**：编译层综合（config 对照+动机 finding×2+派生 absence）
- bonus 各点：A01 B0(subject=Atari-55) B1(subject=Atari-26+budget=100k步+setup) B2(双记录 cited vs stated, 漂移检测归编译层)；A02 B0(finding 统一统计框架)；A03 B0(config:item=replay ratio+finding)；A04 B0(absence not_reported 图only) B1(finding+result A3C自述)；A05 B0(result cited R2D2 2.3k, paper_id=NGU) **B1(U→DET-3)**；A06 B0(lineage citation_context Gorila先行) B1(finding 架构对比)；A07 B0(编译层叙事)；A08 B0(result×3 role:ablation)；A09 B0(编译层跨轴对比)；A10 B0(config setup=sticky+quote含Machado引文→setup值共享join) B1(finding+setup维ALE版本)；A11 B0/B1(lineage uses, citation_context)；A12 B0(result delta 加组件方向) B1(编译层分协议)；A13 B0(budget维) B1(finding 作者自述近似可比+subject清单差异)；A14 B0(finding stated 哲学声明)；A15 B0(finding/config 附录A非对称变体讨论)——除 A05 B1 外全部 E。

</details>

---
*回测人：schema 表达力回测分析员（批次1）。只读任务，未修改任何已有文件。判定依据 spec v1 文本，未参考实现代码。*
