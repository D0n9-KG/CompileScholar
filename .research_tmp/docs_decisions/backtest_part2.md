# Schema v1 表达力回测 · 批次2（时序题 T01–T10）

- 回测对象：`STAGEB-EXTRACTOR-SPEC-v1.md`（记录层 schema v1，2026-09-05 定稿）
- gold 来源：`experiments/e2_need_gap/gold_work/gold_batch2_calibrated.json`（批次2时序题校准版，2026-09-04）
- 判定纪律：宁严勿松；组装级需求（跨篇对比/排序/聚合/计数/时点过滤）依框架规则1归编译层，不算记录级缺口；每条 GAP/PARTIAL 附 gold 摘录 ≤30 字。
- 分析员：批次2回测（只读任务，未改动任何既有文件）。日期：2026-09-05。

## ① 逐题判定表

| qid | type | must点数 | E | P | G | U | 缺口一句话 |
|---|---|---|---|---|---|---|---|
| T01 | temporal | 4 | 4 | 0 | 0 | 0 | 无缺口；基线地位=后继行为记录的编译层组装 |
| T02 | temporal | 4 | 4 | 0 | 0 | 0 | 无缺口；分口径由 measure.aggregation + dims 承载 |
| T03 | temporal | 3 | 2 | 1 | 0 | 0 | 文外事件年份（Machado 2018 提出 sticky actions）无结构化落位 |
| T04 | temporal | 5 | 5 | 0 | 0 | 0 | 无缺口；"静默沿用/无修改"正是 absence 三态设计用例 |
| T05 | temporal | 5 | 4 | 1 | 0 | 0 | lineage 七词缺"理论统一/收进框架"关系义（drl-stats 收编三法） |
| T06 | temporal | 4 | 2 | 2 | 0 | 0 | Atari-100k 引入者与 OTRainbow 的文外引用年份均无结构化落位 |
| T07 | temporal | 6 | 6 | 0 | 0 | 0 | 无缺口；"未被替代而是被分化"= scoped 关系并存的编译层组装 |
| T08 | temporal | 5 | 4 | 1 | 0 | 0 | 源头 Sutton 1988（文外引文年份）无结构化落位 |
| T09 | temporal | 3 | 3 | 0 | 0 | 0 | 无缺口；49/57 口径分裂由 subject 维度结构化承载 |
| T10 | temporal | 4 | 4 | 0 | 0 | 0 | 无缺口；"改了什么"经机制实体 lineage + finding/config 承载 |
| **合计** | | **43** | **38** | **5** | **0** | **0** | 根因仅 2 个（R1 文外年份 ×4 点；R2 统一关系词 ×1 点） |

bonus 要点 12 个：**12 E**（其中 T05-b1 附条件注记，见 ⑤-6）。

## 逐点判定明细（must，kind+关键字段映射一行式）

### T01（2018 时点默认强基线）
- m1 E — Rainbow 身份/arXiv 1710.02298/AAAI 2018=Stage 0 manifest；六组件=lineage `component_of`（from=DDQN/PER/Dueling/C51/n-step/NoisyNets, to=Rainbow, evidence_basis=explicit_claim）×6。
- m2 E — result ×2（method_ref=Rainbow, measure{metric=median human-normalized, value=223%/153%, aggregation=median}, dims{setup=no-op/human-starts, subject=Atari-57, budget=200M frames}）。
- m3 E — Ape-X Table 1 对照行=result（paper=ape_x, role=baseline_comparison, epistemic=cited, dims.setup 两口径）；IQN 比较=result cited 或 lineage `compares_with`；DER='data-efficient Rainbow'=lineage `extends`（evidence_basis=citation_context）；"被改造对象而非平等基线"=编译层对 lineage-被指 vs result-被引 模式的统计。
- m4 E — manifest arxiv_year/arxiv_id 过滤（Agent57/NGU/R2D2/Ape-X 均在语料）。

### T02（2016 时点构建建议）
- m1 E — manifest arxiv_id YYMM 时序排列 + Stage 1.5 registry（方法池全部在语料）；"count_explore 属探索线"=其记录指向（config item=内在奖励）编译层归类。
- m2 E — Duel Clip 172%=result（dims.variant=Duel Clip, subject=57游戏, setup=no-op）；PR.DUEL 592%/172%=result ×2（paper=qr_dqn, role=baseline_comparison, epistemic=cited, aggregation=mean/median 区分）；A3C SOTA+半时间单机CPU=finding（strength=stated, quote 逐字）+ dims.setup 含定性硬件条件。
- m3 E — result（172% median）+ manifest 年份过滤（Rainbow/C51 不可见）。
- m4 E — 组合建议=组装；"三件 2015 已公开"=manifest；"Rainbow 六组件中四件"=component_of 记录对照；"n-step 当时在 A3C 系"=lineage `uses`/A3C variant 记录；"NoisyNets 未见"=年份过滤。

### T03（人类水平里程碑时间线）
- m1 E — 每个时间线点=result（measure{metric, value, aggregation}+dims{subject=7/49/57游戏乃至 Skiing 成员级, budget=50M/200M/78B frames, setup=协议}）；"6局超前人/3局超专家"=result（metric=局数, value=6/3, unit=games）；"75%×29局"=result（metric=达人类75%局数, value=29）；R2D2 1920.6% 二手=result（paper=ngu, epistemic=cited）——epistemic 三值直接承载"二手"标记；年份=manifest。
- **m2 PARTIAL** — no-op→human starts 演变=各 result 的 dims.setup+manifest 年份编译；RND 采纳 sticky actions=lineage `uses`（citation_context）或 RND 记录 dims.setup=sticky-actions；**丢失部分：Machado 2018 这一文外提出事件的年份只能留在 quote 逐字文本里，无结构化字段**（详见 R1）。
- m3 E — 三个操作化各自活在对应论文 result.measure.metric（+quote 逐字）："75%专业测试者(2015)"/"中位%人类(2017-)"/"全57局超基线(2020)"；漂移结论=编译层按时间对比 metric 定义。注记：metric 归一化/注册机制规格未明说（见 ⑤-2）。

### T04（target network 各家处理）
- m1 E — DQN 提出=config（method_ref=DQN, item=target network 更新, value=τ=10,000 周期复制, role=default）+manifest 年份；PER 转述句=finding/lineage（paper=seed_PER, epistemic=cited, evidence_basis=citation_context）。
- m2 E — "stays unchanged from DQN"=absence（absence_type=explicitly_stated, quote 必填正好承载原句）或 finding stated；"online 选动作"=config/finding；"10k→30k"=config（dims.variant=tuned）；"以降低过估计"驱动=finding（claim 含 because 语义）。
- m3 E — "PER 全文仅一处提及且无修改"=absence（absence_type=not_reported, evidence=穷尽性依据"全文仅一处提及且为复述"）——正是拍板 B 的设计用例；"继承 DQN 基础设施"=lineage `extends`/`uses`。
- m4 E — "绕开"=lineage `replaces`（from=A3C 并行机制, to=experience replay, quote="Instead of experience replay..."）；"target network 不再必需"=absence explicitly_stated（quote="we do not use a replay memory"）+编译层解读。
- m5 E — R2D2 config（item=target network update interval, value=2500 updates）；"帧数→更新次数"单位演变=编译层跨篇对比；"均保留"=config/lineage `uses` 存在性。

### T05（分布式 RL 提出-质疑-完善）
- m1 E — C51 主张=finding（strength=stated）；自曝 instability=finding（strength=demonstrated, quote="exposing a significant distributional instability"）；年份=manifest。
- m2 E — C51 入 Rainbow=lineage `component_of`；消融排位=result（role=ablation, dims.variant="−distributional", delta）+finding（排位主张 demonstrated）；排序=编译层。
- m3 E — "指出前作局限"=finding（claim=C51 不保证最小化 p-Wasserstein, scope_ref=C51, strength=stated）；修正关系=lineage `improves`（QR-DQN→C51, IQN→QR-DQN, explicit_claim）。批评≠关系否定，finding 足够（见 ③-7）。
- **m4 PARTIAL** — 统一框架主张=finding ✓；"认识深化"叙事=shift（from_state/to_state/driver, source_type=discussion）✓；**丢失部分："把 C51/QR-DQN/IQN 收进框架"的收编关系在七词封闭集无对应词**——`extends` 义为方法构建非理论收编，`motivated_by` 方向反；变通=3 条 finding（scope_ref=各被收编方法），但降级为自由文本 claim，编译层失去可 join 的"框架收编了哪些方法"关系结构（详见 R2）。
- m5 E — 正/修贡献分类=编译层对 lineage+finding 记录的组装；"每篇点名前作局限"=各 finding 已在 m3/m4 覆盖。

### T06（2019 时点样本高效最好成绩）
- **m1 PARTIAL** — Atari-100k 作为协议=dims.setup 注册值 ✓；"100k步=400k帧≈2小时"换算=config/finding ✓；**丢失部分：'由 Kaiser et al. 2019 + van Hasselt 2019 引入'的引入者年份是文外引文年份（SimPLe/DER 不在语料，gold 自己标二手），无结构化字段**（R1）——as-of 判定"该设定 2019 年确立"依赖它。
- **m2 PARTIAL** — DER/SimPLe 四数值=result ×4（paper=spr, role=baseline_comparison, epistemic=cited, aggregation=mean/median, dims.budget=100k steps）✓；mean/median 口径分裂由 aggregation 结构化 ✓；**丢失部分：'OTRainbow 0.204 是 Kielak 2020，超出 2019 时点'的文外实体年份仅在 quote 文本内**（R1），时点排除判定需编译层解析引文文本。
- m3 E — 500×差距=dims.budget（400k帧 vs 200M帧）编译层求比；EZ 原句"500 times more data"=finding（epistemic 视转述对象, strength=stated）；"2.5个数量级"=编译层计算。
- m4 E — SPR 0.415/EZ 1.160=result（各自 paper 主结果）+manifest 年份过滤"不可见的未来"。

### T07（ε-greedy 的命运）
- m1 E — DQN 用 ε-greedy=config（item=exploration, value=ε-greedy 退火）+lineage `uses`；NoisyNets 转述=epistemic=cited / citation_context。
- m2 E — boot-dqn 批判=finding（scope_ref=ε-greedy, strength=stated）；随机化值函数替代=lineage `replaces`。
- m3 E — NoisyNets 替代声明=lineage `replaces`（quote="ε-greedy is no longer used..."逐字）。
- m4 E — 内在奖励系三法=config（item=内在奖励, value=伪计数/ICM/RND）+manifest 年份；"不完全替代 ε"=uses/config 记录与 replaces 记录并存的编译层组装。
- m5 E — Agent57 策略族=dims.variant（族成员，family-scoped 合法域）+config（族内 ε 值）；Ape-X 分层常数 ε 公式=config（item=ε 调度, value=ε_i=0.4^{1+iα/(N-1)}, ε=0.4, α=7，quote 锚定）；"不退火"=quote/config 语义；R2D2 沿 Ape-X=lineage `uses`（explicit_claim, quote="All missing parameters follow..."）。
- m6 E — "未被替代而是被分化"=对 scoped 关系（replaces scope=学习式探索场景；uses scope=分布式设置=dims.setup）并存模式的编译层结论，无需否定词（见 ③-4）。

### T08（n-step returns 进入 DQN 系）
- **m1 PARTIAL** — A3C 四变体之一=dims.variant（A3C 族合法域：one-step Sarsa/one-step Q/n-step Q/A2C）✓；Ape-X 引文原句=lineage `uses`（citation_context）✓；**丢失部分：'源头：Sutton 1988'的文外提出年份无结构化落位**（R1）——"源头归属+年份"只剩 quote 逐字。
- m2 E — Rainbow 纳入=lineage `component_of`（from=n-step, to=Rainbow）；消融=result（role=ablation, dims.variant="−multi-step", delta）——拍板 A 设计用例；"最关键组件"排位=finding demonstrated；Rainbow 早于 Ape-X=manifest arxiv_id 排序（1710.02298 < 1803.00933）。
- m3 E — Ape-X n=3=config（item=multi-step n, value=3）；IMPALA V-trace=lineage `component_of`+config；"偏差被显式处理"=finding。
- m4 E — R2D2 n=5=config；优先级混合公式 p=η·max δ_i+(1-η)·δ̄=config（公式型 value，quote 逐字锚定，见 ⑤-4）；"built upon"=lineage `uses`/`extends`（explicit_claim）。
- m5 E — Fedus 张力表述=finding（claim="理论无根基但实证独特受益", strength=demonstrated）；"V-trace 是校正机制"=lineage `motivated_by`（from=V-trace, to=多步 off-policy 偏差问题）——七词中 motivated_by 恰好承载。

### T09（前五个推过 100% 中位）
- m1 E — 五方法各=result（measure{metric=median human-normalized %, value=118/106/151/123/178, aggregation=median}, dims{subject=49/57游戏, setup=no-op, variant=Duel Clip/NoisyNet-DQN/NoisyNet-Dueling}）；"前五/第六"排序+">100% 过滤"=manifest arxiv_id 时序+编译层；cited 值（QR-DQN 表）=epistemic=cited。
- m2 E — DQN-nature 79%=result（cited, QR-DQN 表）；"未过"=79<100 编译层数值判定；"其人类水平是 75%×29局口径"=result（metric 定义不同）+编译层口径对照。
- m3 E — 49/57 口径不等价=dims.subject 结构化差异；组合成绩（PRIOR./PR.DUEL.）时间归属=result.paper_id→manifest 年份（归属报告论文）。

### T10（2015–2016 直接改进 DQN 的论文）
- m1 E — 时间窗清单=manifest arxiv_id 过滤（六篇+count_explore 全在语料）；"直接改进"=lineage `improves`/`extends`（to=DQN-nature）。
- m2 E — "各自改了什么"=以机制实体为 lineage 端点（PER→experience-replay `replaces` 均匀采样；A3C→replay `replaces`；boot-dqn→dithering `replaces`）+每篇 finding/config 承载具体内容（双流分解、|TD|优先、10k→30k、异步并行半时间、随机化值函数头）；"Gorila first massively distributed"=finding（strength=stated/demonstrated）。前提：registry 收机制级实体（见 ⑤-1）。
- m3 E — 窗口边界=manifest 过滤；最优组合=组装（同 T02-m4）。
- m4 E — count_explore 边界判断=其记录指向奖励信号（config item=伪计数 bonus）而非架构/训练机制→编译层分类；DDPG 排除=manifest 完备清单（不在语料=可判定）+若在文内被引述则 lineage `extends`（scope=连续控制, subject 维度）/finding cited。

## ② GAP + PARTIAL + UNCERTAIN 明细清单

**GAP：0 条。UNCERTAIN：0 条。PARTIAL：5 条，归并为 2 个根因。**

### R1（字段扩展类）：文外实体的引文年份无结构化落位 —— 4 个 must 点命中

批次2 gold 头部校准项明言"时间索引的证据来源（arXiv编号/**文内引用年份**）须显式"。语料内论文年份由 Stage 0 manifest 双年份承载（零缺口）；但**语料外实体/事件**（Sutton 1988、Machado 2018、Kaiser 2019/SimPLe、van Hasselt 2019/DER、Kielak 2020/OTRainbow、DDPG 1509）的提出年份只存在于引用它们的 quote 逐字文本（"(Kielak, 2020)"式作者-年份），schema 无任何结构化字段安放。

| qid·点 | gold 要点摘录（≤30字） | 丢失能力 | 影响 |
|---|---|---|---|
| T03-m2 | "Machado 2018建议sticky actions（RND采纳）" | 文外协议提出事件的年份结构化 | 协议演变时间线断一环 |
| T05-m4* | （不属 R1，见 R2） | — | — |
| T06-m1 | "由Kaiser et al. 2019+van Hasselt et al. 2019引入" | 文外基准引入者年份结构化 | as-of 判定"设定2019年确立"依赖引文文本解析 |
| T06-m2 | "OTRainbow 0.204是Kielak 2020，超出2019时点" | 文外方法年份结构化 | 时点排除判定不可机器执行 |
| T08-m1 | "源头：Sutton 1988（引文原句）" | 文外源头归属年份结构化 | "源头+年份"只剩 quote |

判定依据（框架规则2）：这正是"gold 需要文内事件时间点且无处安放"的情形——quote 是溯源字段非内容字段，内容字段（method_ref/relation/dims/measure）均无年份位。信息未彻底丢失（quote 逐字保真，编译层可文本解析作者-年份），故判 PARTIAL 非 GAP。

**建议归属**：字段扩展——Stage 1.5 method registry 实体增可选 `origin_year_cited{value, source_paper_id, quote}`（值来源=引文逐字作者-年份，非 LLM 猜测，不违反"记录层无 LLM 猜的 year"纪律——该纪律约束的是语料论文年份的 manifest 权威性）；或 lineage/finding 增可选 `cited_entity_year`。次选：仲裁明确"编译层从 quote 确定性解析作者-年份"为正式通道并接受其脆弱性。**注意 as-of 题是本批次题型核心，此根因直接压在题型要害上。**

### R2（新 relation 词类）：lineage 七词封闭集缺"理论统一/收编/泛化"关系义 —— 1 个 must 点命中

| qid·点 | gold 要点摘录（≤30字） | 缺失能力 | 建议归属 |
|---|---|---|---|
| T05-m4 | "把C51/QR-DQN/IQN全部收进…框架" | 理论框架对多个前作的收编/泛化关系（generalizes/unifies/subsumes 义）；`extends`=方法构建、`motivated_by`=方向反、`component_of`=部分-整体，均不匹配 | 新 relation 词（如 `generalizes`）走仲裁升版；或仲裁认定 finding（scope_ref=被收编方法）降级表达已够用并接受关系结构损失 |

当前变通及其损失：3 条 finding（claim="C51 可视为统计估计量+注入策略的特例"等，scope_ref 各方法）可保事实不丢，但"框架收编了哪些方法"从可 join 的关系记录降级为需读 claim 文本——恰是记录层要消除的负担。shift 记录可承载"认识深化"叙事但不承载逐方法收编关系。

### UNCERTAIN：0 条

两个曾考虑标 U 的点经内部一致性论证后落定：①机制级实体可注册性——T01-m1 的 `component_of`（from=n-step, to=Rainbow）在 schema 内部已强制 registry 必须收技术/机制级实体（n-step 非论文级方法），故非存疑而是规格隐含承诺（转 ⑤-1 显式化建议）；②T05-m4 的 extends 宽读——按宁严勿松判 PARTIAL 并在 R2 保留两种理解的仲裁选项。

## ③ negative relation 案例专列

批次2 全量扫描否定性表述 8 处，**无一需要 relation 否定词**，全部落入三条既有通道：

| # | qid·点 | 表述 | 通道 | 判定 |
|---|---|---|---|---|
| 1 | T04-m3 | "PER…无任何修改（absence级证据）" | absence `not_reported` + evidence 穷尽性依据 | E（设计用例正命中） |
| 2 | T04-m2 | "stays unchanged from DQN"（明示未改） | absence `explicitly_stated`（quote 必填承载原句）或 finding stated | E |
| 3 | T04-m4 | A3C 绕开 replay（"we do not use a replay memory"） | absence `explicitly_stated` + lineage `replaces` | E |
| 4 | T07-m6 | "ε-greedy未被替代而是被分化" | scoped 关系并存组装：replaces（scope=学习式探索）与 uses（scope=分布式 setup）共存→编译层结论 | E（建议作为否定结论编译推导的 canonical 案例） |
| 5 | T07-m4 | "不完全替代ε" | 同 #4（uses/补充记录并存） | E |
| 6 | T09-m2 | "DQN-nature未过（中位79%）" | result 数值比较（79<100）编译层推导 | E |
| 7 | T05-m3 | "指出C51'not guaranteed to…'" | 批评=finding（scope_ref=前作）；修正关系仍是 `improves`——批评≠关系否定 | E |
| 8 | T02-b1 | "boot-dqn…中位数字不及Dueling系" | 跨篇 result 数值对比编译层推导 | E |

另 T08-m5 "theoretically ungrounded…uniquely beneficial" 张力表述=finding（demonstrated），非关系否定。

**专列结论**：时序批次未暴露 "A fails to improve B" 型硬案例（实证性关系否定）——批次2 的否定性全部是 absence 型、数值比较型、scoped 组装型。**negative relation 是否加否定词的仲裁不应基于批次2 证据，须等 config/result 密集批次（其他批次）的暴露情况再定**；若其他批次也只出现上述三类，则七词封闭集+absence 三态+编译层组装可能已完备。

## ④ 统计汇总

- **must 43 点：E=38（88.4%）/ P=5（11.6%）/ G=0 / U=0**
- **bonus 12 点：E=12（100%，其中 T05-b1 附条件注记）**
- 合计 55 点：E=50（90.9%）/ P=5（9.1%）/ G=0 / U=0
- 根因收敛：5 个 PARTIAL 归并为 **2 个根因**——R1 文外实体引文年份（字段扩展类，4 点）、R2 理论统一关系词（新 relation 词类，1 点）
- **维度词表零缺口**：批次2 全部条件限定词均映射五维——no-op/human-starts/sticky-actions/分布式/单机CPU→`setup`（含定性硬件，双落位生效）；49/57/26/7游戏、Skiing→`subject`（family→member 两级正好用上）；200M/78B/50M帧、100k步、10 days、1 GPU→`budget`（typed 含算力/墙钟）；Duel Clip/NoisyNet-DQN/NoisyNet-Dueling/PR.DUEL/DDQN-tuned/A3C四变体/Agent57策略族→`variant`（family-scoped 合法域）；mean/median→measure.aggregation。无一映射失败。
- **absence 三态充分**：not_reported ×1（T04-m3）、explicitly_stated ×3（T04-m2/m4、T07 辅助）、cannot_tell ×0（未用到，无语料证据支持其必要性——保留判断）。
- **时序承重结构**：manifest 双年份+arxiv_id（YYMM+月内序号）承载 10/10 题的全部时序判定，语料内零缺口（含 T09-b1 月级同月排序 1710.10044 vs 1710.02298）；唯一漏洞即 R1 文外实体。
- **epistemic 三值高负荷**：批次2 大量"二手数字"（R2D2 1920.6%[NGU表]、DER/SimPLe[SPR表]、DDQN 118%[QR-DQN表]）全部由 epistemic=cited + paper_id 承载，"材料源=gold 源"审计链完整。
- **消融并入（拍板 A）直接命中**：T05-m2（Rainbow 组件排位）、T08-m2/b1（−multi-step 及 measure.timepoint 早/晚期）为 role=ablation+dims.variant+delta 的设计用例，零缺口。

## ⑤ 诚实注记

1. **最大解释性风险（非 gold 判定缺口，但须显式化）**：本批次 T04/T07/T08/T10 共约 17 个 must 点的 E 判定承重于一个前提——**Stage 1.5 method registry 的实体粒度包含机制/技术级实体**（target network、ε-greedy、n-step returns、experience replay、sticky-actions、V-trace）。该前提由 schema 内部强制（T01-m1 的 component_of from=n-step 要求 n-step 是注册实体；paper_card 的 related_methods 字段支持），但规格文本未明写。若仲裁认定 registry 只收论文级方法，则上述 17 点的机制指向关系全部退化为 quote 自由文本，批次2 的 E 率将从 88.4% 跌至约 50%。**建议 v1.1 显式写明实体粒度条款**——这是本次回测最重要的单条建议。
2. measure.metric 的归一化/注册机制规格未明说（维度词表有 Stage 1.5 注册纪律，metric 无对应条款）。T03-m3 的"操作化漂移"判定依赖各记录 metric 文本+quote 保真，呈现级可行；但跨时间自动检测"定义变了"需要 metric 归一化，建议规格补一句归属（编译层或注册表）。
3. 文外实体（Sutton 1988、DDPG、OTRainbow、Machado 2018）的 canonical 注册依赖 Stage 1.5 接受 citation_context 实体——规格写了 related_methods 进 registry，未明写文外实体处置，与注记1同源，建议一并显式化。
4. 公式型 config value（ε_i=0.4^{1+iα/(N-1)}、p=η·max δ_i+(1-η)·δ̄）：value 字段语义为"带单位"，公式是合法边界情形——quote-first 纪律保证逐字锚定，判 E；建议 Stage 3 后检对公式型 value 明确走"quote 逐字包含"而非数值归一化通道。
5. 组合方法（PR.DUEL./PRIOR.）的 canonical 归属是 registry 生长问题（组合体是否独立实体），非表达力缺口；T09-m3 的"时间归属看组合论文"由 result.paper_id→manifest 承载，无风险。
6. T05-b1（FQF 2019-11）判 E 附条件：若 FQF 论文在语料则 manifest+finding 全承载；若不在语料，其年份归属落入 R1 同一根因（bonus 级，不影响 must 统计）。
7. **本回测判定的是表达力（schema 能否承载），不是抽取可行性（Stage 2 低成本档能否稳定抽出这些记录）**——E 判定不构成对冒烟阶段通过率的任何预期；尤其 absence 型记录（穷尽性依据）与 citation_context 型 lineage 的抽取难度显著高于主结果记录，冒烟时应单列观察。
8. 时序口径纪律建议：批次2 gold 全程以 arXiv 时序裁定（校准头明言），manifest 双年份中编译层应固定"时序索引=arxiv_id"、venue_year 仅作渲染——建议写入编译层规格防两口径混用。
9. 判定者单人单遍，无第二判定者交叉；PARTIAL/E 边界（尤其"组装级 vs 记录级"的划界）依赖框架规则1的授权，若仲裁对规则1采更窄解读，T01-m3/T07-m6/T10-m4 等组装重灾区判定需复核。
