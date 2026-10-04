# Schema v1 表达力回测——批次3 报告

- 回测对象：`STAGEB-EXTRACTOR-SPEC-v1.md`（2026-09-05 定稿）§1 记录类型 + §2 维度词表 + §3 Stage 0 manifest
- gold：`gold_batch3_calibrated.json`（25 题：coverage C01–C10 / conditional D01–D10 / method_config G01–G05）
- 判定口径：must 要点逐条判 E/P/G/U；bonus 全量过一遍单独标记。组装级需求（跨篇对比/排序/计数/直方图）按编译层分工算 E，只要每个原子事实有记录归属。判定宁严勿松。
- 日期：2026-09-05；分析员：批次3 回测（只读）

## ① 逐题判定表（must 级）

| qid | type | must点数 | E/P/G/U | 缺口一句话 |
|---|---|---|---|---|
| C01 | coverage | 3 | 3/0/0/0 | 无；对照角色=result role:baseline_comparison，"无Procgen评测"=编译层推导absence（must3为gold勘误元注记，判E-N/A） |
| C02 | coverage | 3 | 3/0/0/0 | 无；"引用≠评测"=epistemic:cited+evidence_basis:citation_context 原生区分 |
| C03 | coverage | 3 | 3/0/0/0 | 无；DER 二手数字=result+epistemic:cited；benchmark 交集为空=编译层推导absence |
| C04 | coverage | 3 | 3/0/0/0 | 无；复现vs引用=role+epistemic+evidence_basis 三字段合力；≥2复现者计数=编译层 |
| C05 | coverage | 4 | 4/0/0/0 | 无；引入年份=manifest双年份（月精度可由arxiv_id推）；采纳=config/setup维度值+编译层时序 |
| C06 | coverage | 3 | 3/0/0/0 | 无；方法-环境对=dims.subject（family→member）矩阵+推导absence |
| C07 | coverage | 4 | 4/0/0/0 | must级无；确定性变体消融=variant维+finding(demonstrated)；bonus"concurrent work"暴露relation缺 concurrent 词（见②P-3） |
| C08 | coverage | 3 | 1/2/0/0 | P×2：①"DeepMind系"机构归属无字段（manifest缺authors/affiliations）②图-only置信区间无记录载体 |
| C09 | coverage | 3 | 3/0/0/0 | 无；空白年/爆发年=manifest年份+编译层直方图；arXiv vs 发表年口径=双年份设计showcase |
| C10 | coverage | 4 | 4/0/0/0 | 无；sweet spot=config role:best_reported+applicability；"按任务调节空白"=推导absence |
| D01 | conditional | 4 | 3/0/0/1 | U×1：buffer容量(1M→10M)作为实验条件的维度归属不明（根因①） |
| D02 | conditional | 4 | 4/0/0/0 | 无；机制主张=finding(stated)；"大动作空间"类任务性质走setup枚举值生长（注记） |
| D03 | conditional | 3 | 3/0/0/0 | 无；mean 228% vs median 79%=measure.aggregation 两条result记录 |
| D04 | conditional | 3 | 3/0/0/0 | 无；per-game 负增益=result.delta负值+dims.subject逐游戏 |
| D05 | conditional | 3 | 3/0/0/0 | 无；policy-lag 档位=setup枚举值；"3 out of 5 tasks"=可计数measure |
| D06 | conditional | 4 | 1/0/0/3 | U×3：buffer容量/数值级条件维度归属不明（根因①，must1/2/3）；must4"否定Z&S"=relation词表压力案例（finding承载，见③） |
| D07 | conditional | 4 | 4/0/0/0 | 无；理论性质主张（Wasserstein度量错配）=finding(stated)；参数化演化链=lineage |
| D08 | conditional | 4 | 4/0/0/0 | 无；actor规模=setup"含硬件条件"读法（若仲裁收窄setup则滑入根因①，见⑤）；Figure 4数据由正文句承载 |
| D09 | conditional | 3 | 3/0/0/0 | 无；"纯探索设置"=variant组件开关（外在流关闭）；房间数/回报=result |
| D10 | conditional | 4 | 4/0/0/0 | 无；n取值谱=各篇config记录；early/final=measure.timepoint；讨论者清单=编译层枚举 |
| G01 | method_config | 4 | 4/0/0/0 | 无；配置谱系=config记录+编译层时序；相对值α/4=字段保真注记（见⑤） |
| G02 | method_config | 4 | 4/0/0/0 | 无；无head-to-head=推导absence；信号差异=config跨篇diff+finding(动机) |
| G03 | method_config | 4 | 4/0/0/0 | 无；l=40实证vs 20备选=config.role区分；超参表=行头+列头+单元格quote规则 |
| G04 | method_config | 4 | 3/0/0/1 | U×1：atom数值级（N=32/51/…）作消融条件的维度归属（根因①）；must4"无51最优证明"=absence not_reported ✓ |
| G05 | method_config | 4 | 4/0/0/0 | 无；bandit机制=finding+config(τ值/(β,γ)族)；vs NGU差异=原文自述finding+编译层对比 |

**must 合计：89 点 → E 82 / P 2 / G 0 / U 5**

bonus 级（26 点）：E 25 / P 1（C07 bonus "concurrent work"，见②P-3），其余全部可表达（明细见⑤注记与③专列）。

## ② GAP + PARTIAL + UNCERTAIN 明细

本批 **GAP=0**。P/U 共 8 条（must 7 + bonus 1），收敛为 **4 个根因**。

### 根因①（最高优先）：数值级超参作为实验条件时，五维无明确归属 —— U×5

维度词表现状：`variant`=组件**开关**（−n-step、−priority 式）；`budget`=运行规模（frames/steps/hours/FLOPs/算力）；`setup`=协议/环境配置条件（含**定性**硬件条件）。当研究把某个**超参取值**当自变量系统扫描时（Fedus 的 buffer 1M→10M、Z&S 的 10²→10⁵、C51 的 atom 数 N），该条件轴落在三者的语义缝隙里：它不是开关、不是运行预算、也不是"定性"协议条件。

两种理解（均成立，故记 UNCERTAIN）：
- 理解A：`variant` 的"closed hypercube（按方法族 scoped）"可扩展出取值级轴（如 Fedus 研究族的 buffer-size 轴、C51 族的 atom-N 轴）；或 `setup` 枚举值可注册 "buffer-10M" 类离散配置条件 → 全 E。
- 理解B：按字面语义（开关/运行规模/定性协议），取值级条件无处安放 → 条件只能退入 finding.claim 自由文本，**失去 typed 条件化查询能力**——而"条件性一等公民"正是 D 类题（本 schema 核心负载）的立身之本。

| # | qid | gold要点摘录（≤30字） | 缺失能力 | 建议归属 |
|---|---|---|---|---|
| U-1 | D01 must2 | "大容量回放（1M→10M）下 PER 不显著" | buffer容量作为result/finding的条件维度 | 三选一仲裁：variant语义扩至取值级轴 / setup开放注册配置条件枚举值 / budget扩为"任何量化资源设置"；影响面=全部条件类题 |
| U-2 | D06 must1 | "grid world 10^2→10^5 单调下降实验" | 同上（buffer容量扫描档位） | 同 U-1 |
| U-3 | D06 must2 | "容量 1M→10M 显著提升" | 同上（−n-step部分=variant开关✓，容量轴不确定） | 同 U-1 |
| U-4 | D06 must3 | "容量效应=组件×逼近×任务三元交互" | 三元交互的容量轴无法typed条件化（合成结论本身=编译层✓） | 同 U-1 |
| U-5 | G04 must2 | "Varying the Number of Atoms 实验" | atom数取值级作为消融条件（单调性结论=finding✓） | 同 U-1 |

关联（未降级但同根因邻域）：D06 must4 的 replay ratio（判E：靠"fixed-ratio"作 setup 协议限定词承载；若需 typed 比值档位则同根因）；D08 的 actor 数 8→360（判E：靠 setup"含硬件条件"读法；若仲裁收窄 setup 语义则滑入本根因）。

### 根因②：Stage 0 manifest 无作者/机构字段 —— P×1

| # | qid | gold要点摘录 | 缺失能力 | 建议归属 |
|---|---|---|---|---|
| P-1 | C08 must2 | "2018后的DeepMind系论文普遍±" | 机构归属无系统化 typed 访问：manifest 字段（title/arxiv_id/双年份/venue/openreview_id/source_*）不含 authors/affiliations；paper_card 亦无。"DeepMind系"分组只能靠 quote 文本偶然出现 | **字段扩展**：Stage 0 manifest 加 authors/affiliations（非LLM、arXiv/S2 元数据零成本可得）。附带收益：C05"Machado et al./Machado 协议"式 eponym 归属（当前靠引文 quote 承载，可接受但不系统） |

### 根因③：图形层数据无记录载体 —— P×1

quote/loc 锚定 mineru 解析**文本**；表格行有显式规则（行头+列头+单元格），但纯图形数据（曲线点值、CI 阴影、saliency 热图）无类型、无 quote 可锚。本批多数图形依赖被正文句/caption 救回（C07 消融图有文字 caption、D08 Figure 4 有正文结论句、D02 saliency 有正文描述），唯一裸露处：

| # | qid | gold要点摘录 | 缺失能力 | 建议归属 |
|---|---|---|---|---|
| P-2 | C08 must3 | "图-only的置信区间（如IMPALA阴影）不计入" | 图内数据（学习曲线阴影=方差报告的行为证据）无处安放，导致"是否报方差"分类对图-only论文系统性盲区——恰是本题 trap_view 点名的元数据级属性 | **字段扩展/新类型**：最低配=paper_card 实验矩阵扫描扩"图表清单"（哪些图存在、caption 逐字）；完整配=figure 级记录（caption+loc，数据本体不强求）。或编译层 caveat 模板兜底（答案声明文本口径）——但事实本体仍建议有家 |

### 根因④：relation 七词封闭集的两个词表 miss（非否定词方向）—— bonus P×1 + 专列观察

| # | qid | gold要点摘录 | 缺失能力 | 建议归属 |
|---|---|---|---|---|
| P-3 | C07 bonus | "TD3 与 SAC 互为 concurrent work" | 七词 {extends,improves,uses,component_of,replaces,compares_with,motivated_by} 无 **concurrent_with**；且"同期平行而非互相响应"恰恰排除全部有向影响词。finding 可承载该 claim（stated+quote），但 lineage 图失去 typed 同期边（影响编译层时序叙事） | **新relation词**候选：concurrent_with（无向/对称，evidence_basis=explicit_claim）；或接受 finding 兜底。交仲裁 |
| （观察） | D06 must4/证据集 | "fixed ratio 下才观察到对 Z&S 结论的否定" | 跨篇**结论反驳**（disagrees-with）无 typed 边；注意 lineage 的 from/to 是 method_ref，即使加 contradicts 词也连不了 finding↔finding——这是结构性错位非单纯加词 | **编译层可推导非缺口（建议）**：Fedus 单篇 finding（claim 含分歧陈述+quote）已承载事实；矛盾检测=三层验证第三层（跨记录一致性/条件等价检测）已点名职责。低风险，不建议动 lineage |

## ③ negative relation 案例专列

七词封闭集无否定词是规格已知压力案例（§1.3 留给回测暴露）。本批扫描结果：**未出现必须以否定 lineage 边为唯一表达的 gold 要点**——批次3 的否定性事实全部是"条件化 null/负结果"或"缺席"，另有归属路径：

| # | qid | 案例摘录 | 形态 | 归属路径 | 词表压力 |
|---|---|---|---|---|---|
| N-1 | D06 must1 | "large replay buffer can significantly hurt" | 条件化负效应 | result（delta负值）+dims条件+finding(机制:陈旧数据) | 无（结果层非关系层） |
| N-2 | D06 must4 | "对 Zhang & Sutton 结论的否定" | **跨篇结论反驳** | finding(Fedus, stated, quote含disagrees)+编译层一致性验证 | **有**：contradicts 缺词，且 lineage from/to 类型（method_ref）结构上连不了 finding——建议留编译层（见②根因④） |
| N-3 | C07 must1 | "deterministic variant ... substantially worse stability" | 变体级负面消融 | dims.variant="deterministic-actor"+finding(demonstrated)；数值化则 result.delta | 无 |
| N-4 | D04 must2 | "表中 per-game 有降有升"（NoisyNets 部分游戏不如ε-greedy） | 逐subject负增益 | result.delta（负）+dims.subject 逐游戏记录 | 无 |
| N-5 | D01 must2 / C10 must3 | "PER 不显著"/"优先级整体不显著" | 条件化 null 结果 | result（delta≈0/不显著+条件dims）或 finding(demonstrated)。若硬编码成 "PER fails_to_improve Rainbow@大buffer" lineage 边则七词无解——但 gold 原子均在结果层，未逼出该需求 | 潜在（本批未触发） |
| N-6 | D05 must3 | "lag 可忽略时 V-trace 无增益" | 条件化 null | finding/result+setup条件（lag-negligible） | 无 |
| N-7 | C02/C03/G02/C10/G04 | "没有系统评测"/"无head-to-head"/"按任务调节空白"/"无51最优证明" | 缺席性断言 | absence 三态（not_reported+evidence穷尽性依据）+编译层推导absence，全部对位 | 无 |
| N-8 | C07 bonus | "同期平行而非互相响应"（concurrent work） | **非否定但反向**：明确否认影响关系 | finding 承载 claim；typed 边缺 concurrent_with | **有**（见②P-3） |

小结：negative relation 压力在本批的真实形态不是 "fails to improve" 边，而是 **(a) 跨篇反驳（N-2）与 (b) 同期非影响（N-8）** 两个词表/结构 miss；条件化负结果与缺席全部被 result.delta / finding / absence 三态接住。规格 §1.3 预留的仲裁问题可据此收窄。

## ④ 统计汇总

**must 级（89 点）**

| 判定 | 数量 | 占比 |
|---|---|---|
| EXPRESSIBLE | 82 | 92.1% |
| PARTIAL | 2 | 2.2% |
| GAP | 0 | 0% |
| UNCERTAIN | 5 | 5.6% |

**bonus 级（26 点）**：E 25 / P 1 / G 0 / U 0

**总计（115 点）**：E 107（93.0%）/ P 3（2.6%）/ G 0 / U 5（4.3%）

**按题型（must）**：coverage 33 点 E31/P2；conditional 36 点 E32/U4；method_config 20 点 E19/U1。

**根因收敛（8 条 P/U → 4 根因）**：
1. 数值级超参作为实验条件的维度归属（U×5，波及 D01/D06/G04，D06-must4、D08 为邻域）——**唯一影响核心负载（条件类题）的根因，建议最优先仲裁**
2. manifest 缺 authors/affiliations（P×1，C08）——非LLM可得，修复成本最低
3. 图形层数据无记录载体（P×1，C08；本批其余图形依赖均被正文/caption 救回）
4. relation 词表 concurrent/contradicts 两个 miss（bonus P×1 + 观察×1）——contradicts 建议留编译层；concurrent_with 交仲裁

absence 三态、epistemic 三值（stated/demonstrated/cited）、evidence_basis 双值、config.role 三值、measure 五元组（含 timepoint/aggregation）、manifest 双年份在本批均有正面对位案例，无一被击穿。

## ⑤ 诚实注记

1. **C01 must3 是 gold 内部勘误元注记**（"初版判'无'是假阴性"），非语料事实，判 E-N/A 计入分母——它检验的是 gold 建造流程不是 schema。若剔除，must E 率为 81/88=92.0%，结论不变。
2. **提及计数类证据不是记录层内容**：gold 大量使用扫描口径（"±392次""DMLab 46次提及""VizDoom仅1次提及""D4RL×22"）。这些语料文本统计本质是扫描工具/编译层能力；对应记录层事实（存在带std聚合的result、存在DMLab评测result）均可表达。回测未把计数本身当 schema 要求。
3. **定性实验结论的 kind 路由边界需澄清**：本批出现大量定性比较结论（"much more consistently""worse stability""performed worse overall""more robust"），全部判 E 的前提是走 finding（strength=demonstrated）承载；result.measure 面向数值（quote 须逐字含数值）。若抽取规格不明确"定性结论→finding、数值→result"的路由规则，Stage 2 会出现路由方差——建议写进抽取器 prompt 规格，非 schema 缺口。
4. **符号/相对配置值的字段保真**：α/4、η/4、ε_i=0.4^{1+iα/(N-1)}、p=η·max δ_i+(1-η)·δ̄ 等要求 config.value 允许表达式字符串（quote 逐字兜底+编译层可解析）；D08 actor 数判 E 依赖 setup"含定性硬件条件"的宽读法。两处均建议仲裁时一并明确。
5. **抽象任务性质靠 setup 枚举值生长承载**：稀疏奖励/部分可观测（D01）、大动作空间/状态价值主导（D02）、高 off-policyness（D10）、奖励延迟程度（D10）——本批均判 E，前提是 setup 枚举域按语料生长接纳这类值。与规格自己的预判一致（"第二域 setup 词表预期最先崩"），本批确认了该压力方向：生长纪律（Stage 1.5 注册+仲裁）将频繁被触发。
6. **C04 的"重跑 vs 抄数字"是可表达但难抽取的**：epistemic demonstrated/cited 区分 schema 上完备，但判断 QR-DQN Table 1 的 DQN 行是重跑还是引用，需要抽取器读实验设置节交叉验证——保真风险，非表达力缺口；恰是本题 trap_view 的实现难点。
7. **C05 时间口径小漂移**：gold 同题内 "arXiv 1709.06009"（=2017-09）与 "Machado et al. 2018""sticky（2018-09 推荐）"并存。判 E 按年份粒度（manifest 双年份足以支撑"采纳者均晚于引入"的结论）；月精度可由 arxiv_id YYMM 推导，但"哪个版本的哪个月"gold 自身口径未定——不影响判定，提请 gold 侧留意。
8. **零 GAP 与 92% E 率的诚实限定**：批次3 题型分布对 schema 友好（覆盖/配置是六类记录+编译层分工的主场；条件类的条件化恰好压在唯一的 U 根因上）；且 gold 由我方基于语料扫描构造，出题视角与记录层设计同源，存在自洽偏置。本批结论应读作"schema v1 对批次3 负载充分，唯一结构性悬案是取值级条件维度"，不宜外推为全域充分——第二域（PaperScope）回测与批次1/2/4 汇总前不下总判。
9. 本报告只读产出，未修改任何既有文件；所有摘录以 `gold_batch3_calibrated.json` 原文为准。
