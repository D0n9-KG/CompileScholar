# 系统设计评审 · R3 席（跨学科：数据库系统 + IR + agent 记忆系统）

- 日期：2026-09-17 · 评审对象：DESIGN-DECISIONS.md（A1-A3/B/D1-D3/F5 重点）+ DIRECTION.md + ASSET-STATE.md
- 弹药：AGENT-D 主报告及 part1/2/4、AGENT-A、v6 R3 席旧判决（PROPOSAL-2026-09-03-v6-review-round1.md）
- 辅助取证：实读 `src/kb_compiler/views/compiler.py`（575 行）与 `tools.py`（526 行）确认视图编译为纯 Python 确定性投影（build_matrix/genealogy/coverage/cards/narrative/notation_index 全为 records 上的纯函数，零 LLM 调用）——本报告所有"确定性重编译成本≈0"的判断以此为工程事实基础。
- 纪律自缚：每条 DB 概念借用附形式化内核；内核达不到的地方明说达不到，不送帽子。五席隔离，未见其他席位输出。

---

## 0. 总评（一段话）

这份设计清单在 2026-09 的文献坐标系下站位出奇地好：保真三件套（B3/B4/C3-canary）从"癖好"升格为有六个独立基准背书的领域正确答案（Mem0 每条记忆 38.1 个源外数字、DreamBench-SWE literal-storage 全场最高、AuthMem 16.9%→0%），冻结+仲裁（B2）拿到 MAGG +47% F1 的量化背书，"视图=确定性投影"与 CortexDB/ESAA 同构且更干净（他们的源是事件日志，你们的源是逐字锚记录）。但清单里有一条被自己标为"缺口"的项（F5 负载遥测未标准化）实际上是全局承重墙：**A1 的方法论声明、D1/D3 的静态弱点、v6 R3-F1/F4 两条未闭批评、以及用户"随使用进化"提案的可行性，全部收敛到同一个前置条件——把散落的遥测统一成 workload ledger，并在其上建一个带验证门的派生层适应回路**。该回路恰好落在四路检索确认的 T1∩T2 真空上，且实现成本近零 token（确定性投影+本地模型 replay）。我的总裁决：内容层与记录层设计全部保持；派生层（视图参数+路由）应从"冻结"改为"门控自适应"；这不是推翻 A2/B2，而是把它们没说完的另一半补上。

---

## 1. 任务 1【本席最重要交付】：用户"随使用进化"提案的正式裁决

### 1.1 提案重述与裁决

用户原话大意：负载驱动设计只预设了通用需求模式，应在数据应用阶段根据查询等新需求不断自进化。

**裁决：该做，但必须做对分解。** 提案的合法性有四路独立证据支撑：
1. **需求侧**：representation-inference gap（ACE-GraphRAG 2608.01269 + MOSAIC 2609.11065，同图同生成器受控实验证明无固定检索策略一致最优）是对"冻结四视图"的直接学术质疑；D1 自己承认"四张=预设负载假设，未覆盖形态落 search 兜底"；D3 承认"工具集静态、NAV 是补丁非通解"。
2. **信号侧**：NAV miss/usage_audit/auto_rows/PROJ_LEDGER 已在产真实 agent 负载信号，只是散落（F5）——演化的触发数据已经存在，缺的是统一 schema 和消费回路。
3. **空位侧**：T1（immutable 源+确定性编译+负载触发重编译+收益验证 四要件交集）与 T2（两速演化显式建模）经四路检索确认真空；但两条腿已分别被 ACC（2608.31082，Idreos 本人，2026-08-31）与 VikingRAG（2609.11390，2026-09-10）+ CortexDB 占领，**窗口以月计**。
4. **回撤史侧**：五大记忆旗舰全部停止运行时自主演化（A-MEM 一作自我反转做版本控制 ChronoMem 2607.27773、Mem0 插件 Dream 半年撤回 PR#7203、Letta 2026 最激进演化零受控 ablation）；演化加门成潮（TOKI gated 0/6 vs ungated 6/6、Quipu 2608.16813、TRUSTMEM、Recuris validation-gated）。**教训不是"不演化"，是"无门演化必回撤"。**

**关键分解（对用户提案的精确化）**："随使用进化" ≠ "schema 自演化"。系统状态应分解为四层，各层演化速率与门控完全不同：

```
Σ = (S, R, θ, ρ)
  S = schema（词表/记录类型/协议，v1.4）        —— 慢速道：仲裁+版本迁移，保持 B2 冻结，不动
  R = 记录库（逐字锚，frozen per build）          —— 慢速道：canary+postcheck 押运的新建库批次，不做在线变异
  θ = 视图定义参数（模板选择/分带阈值/分组键/工具注册表）—— 中速道：负载触发+验证门+确定性重编译，本提案主战场
  ρ = 路由策略（查询→视图/工具分派）              —— 快速道：在线、确定性计数、零 LLM 调用
```

不变式（这是与一切无门演化系统的形式化分界）：
- **I1（源不可变）**：ρ、θ 的任何演化算子不得触碰 R；R 只能经慢速道（现有建库管线全门）增长。
- **I2（可重导）**：V = Q_θ(R)，Q 为确定性纯函数（已核实：compiler.py 零 LLM）；任何 θ 变更的回滚 = drop V + recompile，代价≈秒级零 token。回滚面平凡是自动化的安全性来源——**正因为记录层冻结且逐字锚定，派生层才有资格自动演化**。这句话同时是 B2 与用户提案的和解书。
- **I3（轨迹正确性）**：借 GEM（2605.26252）自己的词汇——"correctness is a property of the state trajectory"——每次 (θ,ρ) 转移必须过门并落 adoption ledger，演化历史本身可审计可回放。

### 1.2 架构：演化面放哪层、验证门怎么设、与 B2 怎么共存

**演化面 = θ（中速）+ ρ（快速），S/R 不动。** 与 B2 的共存不是妥协而是互证：B2 的 C 线查新价值（"冻结+仲裁+canary 最接近真空"）由慢速道原样保留；DIRECTION §2 的三层阴性证据（Stage A 采纳 0/5、novel 边 6 runs 全 0、方差主导）继续作为"schema 层不上自主演化"的判决依据，且现在多了 D 线回撤史作外部佐证——论文里阴性结果转资产的写法不变。

**触发语义（ECA 规则，active DB 的形式化内核，不许只借词）**：
```
ON  miss_class(k, window w) 计数 ≥ τ_k        -- 确定性谓词，来自 workload ledger
DO  propose(θ')                                -- 提案生成：结构参数用确定性规则；
                                               --   语义分类（miss 归因）允许 LLM（C1 三明治管辖内）
```
触发是确定性的；LLM 只做"这个 miss 属于哪类需求"的语义判断——与 C1 铁律（规则只用于结构/确定性）严格一致。

**验证门四道（G1-G4，按序，任一 FAIL 即 revert）**：
- **G1 确定性重放**：recompile 两次哈希一致（F13 已有，ESAA 2602.23193 哈希重放同构，直接复用）。
- **G2 源忠实抽检（=T5 真空的首个实现）**：视图元素级——数值格逐字回锚 quote（e5_v2 机器已现成），非数值格抽样 LLM 审计+确定性规则先筛。注意：G2 是两层门（确定性可判部分全覆盖，语义部分抽检），我不送"视图整体语义一致性已验证"的帽子——那是开放问题，只能报抽检覆盖率与通过率。
- **G3 收益配对回放**：新旧 θ 在同一 replay 题集上配对 A/B（本地 3.8-27B，近零成本），主指标=端答案分（E2 判分纪律），辅指标=NAV miss 率/工具调用数。预注册主对比，其余探索性——沿用 G1 实验纪律。
- **G4 自动回滚护栏**：adopt 后在线监测窗口内指标回归即 revert（EvolveMem 2605.13941 的 revert-on-regression 是机器仲裁最保守形态的先例，直接借件）。
- **稳定性约束（防 flapping）**：最小支持度 τ、最小变更间隔、滞回带。形式化内核诚实声明：这是**带回归护栏的测量驱动爬山（empirical hill-climbing）**，不是 Harinarayan 意义下的代价模型优化——utility 无法解析预测只能 replay 测量，所以没有近似最优保证。**命名必须用 "measurement-driven view adaptation with verification gates"，不许写 "workload-driven view selection"**（后者有代价模型+选择算法的既定含义，见 §4 F1 更新——上次我批的就是借概念丢内核，这次不许自己犯）。

**采纳分级（自动化程度按影响面递减）**：
- ρ 级（路由表项）：全自动，确定性计数+滞回，零 LLM 写路径（MemCon 2607.13591 先例：tabular bandit 零额外调用；我们从更保守的计数规则起步，不上 learned controller——与"规则只用于结构"纪律一致，bandit 留作消融对照）。
- θ 参数级（分带阈值/分组键/top-k）：过 G1-G4 自动采纳，落 ledger，人**事后**批量抽审。
- θ 模板级（新视图类型/新工具）：过 G1-G3 后人仲裁准入（MAGG 模式）——这一级不自动，因为模板级变更等效于扩展负载分类学本身，属于慢速道的边缘。

**与 GEM C6（检索必须是状态转移算子）的正面交锋升级**：D 线报告现在的回应是"治理立场差异"。有了本架构，回应可以升级为机制性的：把 Σ=(R,V,ρ) 拆分后，**检索对 R 保持纯函数，对 ρ 是状态转移算子**——retrieval-induced adaptation 在控制层真实发生（每次 miss 改变路由状态），且状态轨迹过门可审计，恰好满足 GEM 自己提的 transition soundness + provenance preservation + bounded active state（滞回=bounded）。GEM 的 Observation 1 点名"caches/views/triggers 只能记录访问发生过"——对的，但我们的 triggers 接的是带 G3 收益验证的重编译回路，不是缓存。这是从"立场辩护"到"机制回应"的实质升级，论文该这么写。

### 1.3 新颖性声明能写多强（逐一划界后剩什么）

| 对手 | 它占了的 | 划界后我们剩的 |
|---|---|---|
| CoEvoKG 2608.01904 | 成功轨迹证据运行时写回 KG | 它原地变异图、无门、无源不可变；我们写回对象是派生层，源永不动，每次转移过 G1-G4 |
| EvoGraph-R1 2607.12764 | RL agent 的 GraphEdit 在线重塑超图 | learned in-loop 变异 vs 确定性门控离线重编译；它无保真门无回滚面 |
| Libra Healer 2607.00016 | "训环境不训模型"口号+失败驱动 catalog 重写 | **结构上最近亲，必须诚实**：口号不能主张新颖；我们剩的是它没有的三件——确定性重编译（它是 LLM 重写 Markdown）、验证门栈（它无 G2/G3）、adoption ledger+revert（它无） |
| EviGraph 2608.04738 | 弱节点定位+下游子图再生成+checkpoint | 会话内运行态修复 vs 跨会话门控适应；其 checkpoint/下游再生成必引，是我们增量维护的运行时同构物 |
| ACC 2608.31082（Idreos） | 查询驱动增量物化、投机式 cracking 进 agent | "query-driven materialization"不能主张；我们剩：它只增不重构、无重编译、无逐字锚、无质量门（它的 validate 是结构校验非源忠实）；我们做负载统计触发的**整体重编译+收益验证**（T1 第四要件它只有 token 侧无质量侧） |
| VikingRAG 2609.11390 | usage-derived experience edges + Harinarayan 引用权 | 它占了谱系引用位（related work 声明），但 experience edges 是无失效语义的只增启发式缓存；我们剩：带 G2 源忠实+G4 回滚的验证过视图，且它无两速建模 |
| CortexDB（商业） | "derived views = deterministic functions of (events, derivation_version)" 命题 | 确定性重编译腿被占；我们剩：负载统计触发（它是运维触发版本升级）、G2/G3（它无源忠实门与收益验证）、逐字锚+epistemic 的记录层（它的抽取器是 LLM 且 byte-identical 机制未解释）、学术可复现形态 |
| KBGym 2608.21829 | "需求驱动建库"首篇量化 | 它用监督 gold 训 curator；我们无 gold、用负载遥测+确定性门——信号源不同，且它动的是存储本体，我们动的是派生层 |
| LLM-Wiki 2605.25480 | 编译式检索 SOTA + Error Book 自动演化 | 最强同向竞争者：它是 LLM 编译+自动纠错，我们是确定性编译+门控演化+逐字锚记录层；论文必须正面对比定位（它的域是多跳 QA，可能无法直接跑我们的题，退而以鉴别轴对照） |
| FluxMem 2605.28773（zjunlp） | 运行时拓扑演化+feedback refinement 三基准 SOTA | 最危险邻居：无门全自动、无源忠实、对话/任务域；我们的差异=治理栈整体，须持续盯 |

**划界后能写的声明（精确措辞，可直接进论文）**：
> 首个在科学文献 KB 上实现「不可变逐字锚记录层之上、由查询负载统计触发、经确定性重编译与多级验证门（重放一致性/源忠实抽检/配对收益回放/自动回归回滚）约束、并以两速治理（schema 慢速人仲裁 / 派生层快速门控自动）显式分层」的适应回路。

**不能写的**（每条都有占位者）：query-driven materialization、usage-derived structure、deterministic recompilation（单独）、train-the-environment、需求驱动建库量化、编译式检索 SOTA、"compiled memory"作标题术语（2026 话语场里它多数指行为编译——Atlas 指令重写/Muscle Memory/MemCompiler/PMMC，术语已被劫持，用了就是给审稿人递刀）。

**声明强度的诚实上限**：这是一个**系统/架构贡献**，够论文一个主 section 或线 3 的机制章，不够独立承重一篇论文——承重仍应按 DIRECTION 排线 1（治理实证，永不塌）与线 2（规模定律）。G3 的收益数字是声明的生死线：没有配对 replay 的量化收益，上述声明退化为 GEM 式 vision paper。MAGG +47% 是"人仲裁"的数字；我们中速道参数级**自动**采纳的合法性完全依赖自己的 G3 数字，引不到别人的。

### 1.4 最小可行实现路径与成本

| 阶段 | 内容 | 成本 | 前置 |
|---|---|---|---|
| P0 | **workload ledger**（=F5 重构）：统一 usage_audit/NAV miss/auto_rows/PROJ_LEDGER 为单一事件 schema `{query_id, class, tool, view_hit, miss_reason, tokens}`；类目标借意图分类学收敛成果（PaperPilot 五方向×ScholarQuest 四意图）做外部锚定 | 零 token，~3-5 天 | 无（PS-53 等待期即可做，与 F32 审计协议整理并行） |
| P1 | **ρ 路由层**：确定性计数+滞回的查询类→视图/工具分派；在既有冻结题集（dev30/AirQA20）上离线 replay 评测 | 零 API token（本地 3.8-27B），~1-2 周 | P0 |
| P2 | **一个完整门控采纳循环**：选 matrix 分带阈值或 coverage 分组键作首个 θ；G1 复用 F13，G2 复用 e5_v2 数值回锚机器+抽检，G3 配对 replay，G4 回归回滚；落 adoption ledger（注意：legacy 的 Mutation+Contract+Ledger 内核在 archive/ 死区且 import 闸禁 granular_agent——**只能借设计模式，实现必须 fresh 过闸**） | 近零 token，~1-2 周 | P1 |
| P3 | **负载迁移消融（论文实验）**：题集日志对半分（适应半/评测半），或 PS 建库 KB × AirQA 负载 replay；预注册两种结局（KBGym 的 overlap 发现预测收益∝分布重叠度，负结果同样是资产——与 DIRECTION §2 阴性转资产写法同构） | 判分 ~1-2M（或本地判+Kimi 抽检） | P2 |

合计 ~4-6 周、<2M 远端 token，符合预算纪律。**一个必须预注册的诚实风险**：现有评测负载全是冻结题集，"随使用进化"在体外没有真实分布漂移可适应——P3 的漂移是合成的；部署态的真实漂移故事只能靠 AirQA（跨域）与 PS-53（跨规模）两个天然 shift 载体讲。

---

## 2. 任务 2：A2 eager vs lazy vs cracking 的跨域判决

**判决：全量 eager 编译保持，但 A2 的表述必须重写为分层声明，并补一个路由层；不引入 cracking 式物化。**

分层拆开看，eager-vs-lazy 在两个层的经济学完全不同：

**视图层（θ 的产物）：eager 无条件成立，且这不是一个分岔。** DB 内核：物化的价值 = 避免重复计算 Q(D)。我们的 Q 是 575 行纯 Python（实读核实），对 4k-18k 记录的重编译是秒级零 token——**lazy 没有任何东西可以省**（没有昂贵计算可推迟），只引入延迟与不确定性。DA 把 A2 称为"全篇最大未审视分岔"在视图层是打错了靶：真正的分岔在记录层。cracking 同理不适用：cracking 的内核是"重组成本与查询成本同量级、部分状态可用、逐查询渐进收敛"（Idreos 2007/Salles 2007），前提是被物化的东西贵到只能摊着建。视图不贵，cracking 解决的是我们没有的问题。ACC 把 cracking 搬进 agent 恰是因为它的结构是 LLM 现场抽取（贵）；我们的结构是确定性投影（免费）——**同一术语在两个成本体制下指向相反的最优解**，这个对照本身值得写进论文（也是对 ACC 的精确划界）。

**记录层（R 的生产）：eager 是有条件的，条件必须显式写出来。** 这才是与 Novelty-Aware（在线即时构建）、ACC（投机物化）、KBGym（需求驱动建库）真正竞争的层。~280k tok/篇的抽取成本下，eager 的合法性 = 摊销条件：**有界语料 × 预期多次查询 × 建库成本可回收**。Lacuna 4× 与 ACC 28× 上界是同方向的摊销证据；Dosu 的缓存三条件（irreplaceable/expensive to recompute/used all the time）是视图选择代价模型的散文版，建议直接采纳为未来每个新物化决策的显式判据。诚实限定：论文的摊销主张在冻结题集上是**实验设计事实**（建一次答 N 题）而非部署事实，措辞要写成"bounded-corpus amortized compilation"，与在线构建系的对照臂（Novelty-Aware 是天然对照）留给审稿人问题"为什么不在线建"。

**查询时视图路由：引入（=任务 1 的 ρ 层）。** 这是 representation-inference gap 的对症药，也是 D 线报告自己给出的最低成本回应（把"视图选择"从静态变遥测驱动）。注意与 lazy 的本质区别：路由不推迟物化，只推迟**分派决策**——四张视图全在，选哪张接哪类查询由负载数据定。迁移成本 ~1-2 周（P1）。

**CRITICAL①（编译 vs 缓存不可区分）仍未闭**：预算配对臂（给对照臂无限预算看是否收敛）依旧欠着，DA C-A 与 EIC MAJOR-2 的批评在 2026-09 现状下原样成立。若审稿人把视图定性为"语义缓存"，可用的辩护是：缓存按查询键控、命中率依赖查询重复；视图按 schema 键控、效用不依赖重复，且有缓存没有的 G2 源忠实门（FinCacheServe 2607.26076 是最严格的缓存失效先例——它守 freshness 不守 faithfulness，恰好衬托我们的门管的是 faithfulness）。但辩护替代不了实验，配对臂该跑。

**IVM 谱系的一句了断**：我们的维护不是增量视图维护，是全量确定性重编译，且在此成本体制下严格优于 IVM（IVM 的存在理由是重算贵；我们不贵）。PVLDB 2026 十二系统评测"局部维护比全局重组省钱"不构成反证——他们的全局重组是 LLM 重组；我们的义务是把"确定性重编译成本≈0"变成可测主张：**报 build_views 在 4k/18k 记录两档的 wall-clock**（一行实验，零 token），从此这句话有数字。

---

## 3. 任务 3：A3"不做图"复审

**判决：内容层弃图维持（2026 证据加强了它）；访问层补一条显式注记——确定性元数据上的图形态访问算子该做（条件：Type B 激活时），LLM 关系抽取图永不该做。A3 弱点里"弃图与图降维使用的边界没系统论证" hereby 闭合，边界判据如下。**

2026 证据核对：
- **弃图趋势是真的且指向重型图**：MS GraphRAG 官方降级 maintenance（README 一手实锤，LazyGraphRAG 0.1% 成本数字是 MS 自证"贵的部分不必要"）；LinearRAG relation-free 拿 ICLR 2026；LlamaIndex graph 索引全年 3 commit；LangChain 官方叙事 graph 词汇消失。重 LLM 抽取图已死，这一半 A3 判对了，且比 09-03 判的时候更对。
- **但 citation 结构的实证价值集中在"组织/重排"通道而非"扩召回"通道**：SciRAG contribution chains（不用相似度不用中心性，2026 最强设计方向）；PaperPilot 把 citation expansion 做成带方向参数的原子算子；SPAR 刻意单层 RefChain（"deterministic ensures reliability"）；我们自己的 contest 通道消融同判——引用通道对召回是小杠杆，query-decomp 才是大杠杆；MDAQA 教训一致。EviGraph 证明类型化证据图作**运行态组织**有 +40% 级收益。
- **保图侧的赢法全是"便宜+schema 约束"**：HCG-RAG（"what is placed in the graph matters more than how many nodes"，8-135× 省构建）、TIGRAG（共现图零 LLM 免费建）。Neo4j NICD +80% 那张牌带着赞助自披露与"对照轴不含非图结构记录"两个限定，part4 已录，不赘述。

**形式化边界判据（代替"图灵完备表示间无表达力差异"这句正确但太抽象的话）**：一条边该不该进访问层，看它的生产成本与确定性——
1. **确定性元数据边**（citation 邻接、registry 实体链接、lineage 关系）：抽取成本≈0（manifest/registry 里已经有）、无 LLM 方差、可回放 → **作为 typed tools 的图形态算子暴露**（引文邻域扩展带方向参数、共被引聚类），存储形态就是邻接表，不引入图数据库、不引入图本体。genealogy 视图已局部这么做，缺的是把"引用邻域扩展"做成 Type B 的一等操作子。
2. **LLM 关系抽取边**（开放语义关系、共现推断）：贵、方差主导、2026 实证死亡带 → 永不进。这正是 Stage A 超图路线的教训与 EvoGraph-R1 们的赌注，我们有一手阴性数据（DIRECTION §2）证明这条路在我们的门下过不了。

成本：判据 1 的算子化在 Type B 激活时约数天工作量（邻接数据现成）；现在只做文档化（A3 条目加边界判据），不预建。

---

## 4. 任务 4：v6 R3 席四条批评的更新判决

**F1（workload-driven 形式化内核被抽空）：仍成立，但首次出现可负担的治愈路径。**
现状核对：四视图仍是手工设计（D1 自认"预设负载假设"），无代价模型、无选择算法、无"哪些视图值得物化是非平凡决策"的消融——09-03 的批评字面上原样成立，且 VikingRAG 已把 Harinarayan 引用权占走，"负载驱动"话语在 2026 被建制化（CIDR/VLDB/Tutorial），空洞引用的代价比 09-03 更高了。但三件新事实改变了预后：(i) T3 检索确认**无人完成完整方法论移植**——做一个测量驱动的最小内核仍是第一；(ii) 遥测信号已实际存在（散落），P0 ledger 零成本统一；(iii) 效用函数可测——冻结题集上的配对 replay 使 b(θ)=Σw(q)·Δutility 成为可计算量，09-03 时这没有载体。**判决：批评未过时；执行任务 1 的 P0-P3 即构成"实质化路线"的最低完成态；若不执行，论文里 A1 必须降格为 "workload-informed design"（09-03 给的降格路线原样有效）。** 附加外部效度：Agent-Native Memory（2606.24775，12 系统×5 workload）"有效性取决于记忆结构与 workload bottleneck 的对齐"是 A1 前提的大规模独立实证背书，论文必引——它把 A1 从我们的方法论偏好升格为领域实测结论。

**F2（truth discovery/c-tables 划界）：大半已回应，残留一处真缺口+一笔引用债。**
现状核对：v6 C2 的"冲突簇单元格"没有按原样建；实际建成的是 D2 分带（budget/条件不同不进同格）+B5 维度坐标+B4 epistemic。这在架构上恰好走了 09-03 我指出的"正确轴"：**不做 truth discovery（从冲突中找真值），做条件等价性划分（让伪冲突根本不进同格）**——c-tables 的条件标记思想以"确定性抽取的维度坐标"形态落地，比条件表更保守（条件是显式字段非查询时解释）。残留缺口：**同带内真冲突无机制**——两篇论文在相同条件报不同数值时，现系统只有 epistemic 标记没有冲突检出与并列呈现（DA C-A 的"消解推回 agent"两难在此重现，但发生率低、且 matrix 分带使爆炸半径受控）。修法：同带同 subject 数值分歧的确定性检出（规则可判，C1 管辖内）+格内并列双值带各自 quote——成本一周内，属 D2 的优化项非重构。引用债：Imielinski-Lipski c-tables、Dong et al. data fusion、Fellegi-Sunter ER 三笔写作期必引划界照旧（DIRECTION §3 债务清单该补这三条，现在没列）。

**F3（失效代数应改名 best-effort invalidation）：目标已消失，实质被架构消解，残留记录层一处。**
现状核对："失效代数"作为声明已不在清单里（A/B/C/D 无此条目）——批评对象不存在了。更重要的架构事实：视图=全量确定性重编译意味着**视图层 staleness by construction 不可能**（V 永远是 Q(R) 的当前值，没有需要传播的失效）；09-03 我要求的形式化骨架（fresh/stale/invalid 状态机）在视图层被"永不 stale"平凡满足。残留在记录层：R 对现实世界的失效（撤稿/结果被质疑/对齐漂移如 F34 "+"变体）仍是 best-effort——as_of+epistemic+仲裁队列处理它，无失效检出召回率指标。2026 领域收敛（标记不删除：Mem0 Supersede/bi-temporal/TOKI 审计行）证明我们的仲裁台账风格是主流正确形态。**判决：已回应（以架构消解而非以代数实现——这是比 09-03 提案更好的结局）；若论文提及记录层失效，名字用 "best-effort invalidation under noisy compilation" 并给检出召回率数字，不许复活"代数"一词。**

**F4（负载迁移假设：人类查询→agent 操作未验证+无漂移故事）：仍成立，修复路径已具体化。**
现状核对：A1 弱点栏自己承认"负载目录主体是人类查询（Asta），agent 侧只有代码行为挖掘"——09-03 的批评原样在册。新事实：(i) 真实 agent 侧负载信号已在积累（NAV miss/usage_audit/overflow 1294 条画像），但散落无统一 ledger（F5）；(ii) PS-53（规模 shift）与 AirQA（域 shift）是两个天然分布迁移载体；(iii) 任务 1 的 P3 负载迁移消融正是 09-03 要的"负载更新故事"的最小实现。**判决：仍成立（无 ledger=迁移不可测量，无 P3=漂移无回应）；P0+P3 执行后转"已回应"；Asta 人类分布降级为设计灵感来源的表述修正照旧必要。**

四条小结：F1 仍成立（可治愈）、F2 大半已回应（残留同带冲突缺口）、F3 已回应（架构消解）、F4 仍成立（修复路径具体）。**共同前置=P0 workload ledger，这是四条批评与任务 1 的汇合点。**

---

## 5. 任务 5：A/B/D/F5 逐条判决

| 条 | 判决 | 证据锚 | 修法 | 成本 |
|---|---|---|---|---|
| **A1 负载驱动** | **优化** | Agent-Native Memory 2606.24775 独立实证背书；但 v6-F1/F4 未闭，T3 真空=机会也是义务 | P0 ledger+意图分类学外部锚定映射表（PaperPilot×ScholarQuest×D1-D10）；不执行则论文措辞降格 workload-informed | 零 token，3-5 天 |
| **A2 eager 优先** | **优化**（保持行为，重写声明） | compiler.py 实读=纯函数投影；ACC/CrackIVF 谱系对照；Dosu 三条件 | 声明分层：视图 eager 无条件（补 wall-clock 可测数字）+记录 eager 限 bounded-corpus 摊销显式化+引入 ρ 路由；预算配对臂补上闭 CRITICAL① | 路由 1-2 周；配对臂一次判分预算 |
| **A3 不做图** | **保持**+边界判据入册 | MS GraphRAG maintenance/LinearRAG ICLR/HCG-RAG；SciRAG 组织通道 vs MDAQA+自家消融召回小杠杆 | §3 判据写进 A3 条目：确定性元数据边=访问算子（Type B 激活时），LLM 关系抽取边=永不 | 文档化零成本 |
| **B1 类型化记录** | **保持** | 四轴空位仍在（DIRECTION 线3）；Mechanist 第三方背书 | 表达力边界（公式/图）走 F2/F3 排队通道，不因 AirQA 弃分动摇记录层设计 | — |
| **B2 冻结 schema+仲裁** | **保持（加强）** | MAGG +47% F1 首个量化背书；D 线回撤史（ChronoMem/Mem0 PR#7203/Letta 零 ablation）；Stage A 一手阴性数据 | 任务 1 架构把它从"保守选择"升格为"派生层自动化的安全性前提"（I1/I2 不变式）；治理吞吐真瓶颈走 F35 模式自动仲裁，触发条件照 DIRECTION §2 | 零 |
| **B3 quote-first** | **保持** | C3 共识六个独立基准：DreamBench-SWE literal-storage 全场最高/Mem0 38.1 源外数字/FACTWASH | 论文写法从"我们的设计选择"升格为"2026 已量化的领域结论，我们是全程贯彻者" | 零 |
| **B4 epistemic 三态** | **保持** | AuthMem 16.9%→0%；Explicit-Not-Longer +15pt（NOTES_SPEC 修复的独立外部验证）；T4 真空（epistemic 进编译层无人做） | 抽取端自判 epistemic 的准确率单测一次（B4 自己列的弱点，仪器先审纪律适用于自家字段） | 近零 |
| **B5 维度坐标** | **保持，F35 提优先级** | 56% 单层表头残留是 D2 分带质量的直接上游（弱点传导链已在清单自认） | F35 排队现状合理，但若 θ 适应回路（任务1 P2）选 matrix 分带做首个循环，F35 先行——分带阈值调优在 56% 维度缺失的数据上是给噪声调参 | F35 已估 1-2M |
| **B6 absence 三态** | **保持+持续审计** | F32 伪缺口 321→8 活体取证是线 1 核心资产；DA M-D（false gap 不可区分）风险仍在 | F32 审计协议整理（已在执行队列）；derived absence 每批建库后强制跑可达性审计，写进 C3 canary 押运清单 | 零 token |
| **D1 四视图** | **优化** | ACE-GraphRAG/MOSAIC representation-inference gap 直接质疑冻结视图；v6 m-A"无组合代数"未全应 | 不建视图代数（那是过度工程，m-A 的正确回应是最小交叉引用而非完备算子集）；把"四张=预设假设"改为"四张=θ 的初始值"，模板级变更走人仲裁、参数级走门控自动（任务 1 中速道）| 含在 P1/P2 |
| **D2 分带** | **保持+补缺口** | 分带=条件等价性预防（v6-F2 正确轴的落地）；带间趋势需求自认无表达 | 同带冲突确定性检出+格内并列双值（§4-F2 修法）；分带阈值列为 θ 适应首批候选；带间趋势查询由 genealogy as_of 部分承接，缺口如实标注 | ~1 周 |
| **D3 typed tools+search 兜底** | **优化** | 工具集静态=清单自认；NAV 是补丁；自家 Type B 消融 bm25 0.610>vector 0.530 | NAV 升格为 ρ 路由层（补丁变一等组件）；KB 内 search 兜底加词法通道（bm25 已证明词面优势在自家数据成立）；工具生长走 θ 模板级人仲裁 | 路由含在 P1；词法通道数天 |
| **F5 负载遥测** | **重构（升格一等组件）** | 任务 1 四要件之三、v6-F1/F4 治愈前置、D1/D3 弱点共同根因，三线汇于此处 | =P0 workload ledger：统一事件 schema+类目外部锚定；它不是"缺口清单第 5 项"，是下一阶段的承重墙 | 零 token，3-5 天 |
| F1-F4/F6 | 保持排队（非本席焦点） | — | 仅注：F1 Type B 激活时把 §3 判据 1 的引文算子一并做；AWS retrieval-generation gap（2606.25656）再次确认 E2"端答案判分才是终审"纪律正确 | — |

---

## 6. 真实优点（如实记）

1. **不变式思维已经在系统里**：I1/I2（源不可变+可重导）不是我送的理想化——records 冻结、视图纯函数、F13 双臂 PASS、canary 押运，架构事实先于我的形式化。形式化只是给既有事实命名。
2. **保真栈的外部验证密度全领域罕见**：B3/B4/C3 每一条都能在 2026 文献找到独立定量背书（§5 表内锚），这在 09-03 时不成立——时间站在了这边。
3. **阴性结果资产化纪律**（Stage A 自演化失败数据）恰好是任务 1 声明的反面证据基座："我们知道无门演化会怎样，因为我们测过"——这是 FluxMem/LLM-Wiki 们没有的。
4. **G2 仪器先审纪律**与领域元观察（TIAP/MirageBench Self-Monitoring Inversion/Commercial Tax）同构，方法论上领先。
5. compiler.py/tools.py 的代码形态（纯函数、无全局态、exclude canary 显式参数）使 θ 参数化的改造面很小——工程债况好于文档给人的印象。

## 7. 判决统计

保持 9（A3/B1/B2/B3/B4/B5/B6/D2 + E 区冻结不审）· 优化 4（A1/A2/D1/D3）· 重构升格 1（F5）· 替换 0 · 专项裁决 3（任务1：该做-θ/ρ 层门控演化，schema 层维持冻结；任务2：视图 eager 无条件+记录 eager 限摊销域+引入路由+不上 cracking；任务3：内容层弃图维持+访问层确定性元数据算子有条件引入）· v6 四条更新：F1 仍成立可治愈 / F2 大半已回应 / F3 已回应（架构消解）/ F4 仍成立路径具体。

**一句话收尾**：这副设计 09-03 时欠 DB 一个形式化内核，2026-09 它欠的只剩一个 ledger——把 F5 从缺口清单第 5 项提为下一战役承重墙，四条旧批评、三个新威胁（representation-inference gap/GEM C6/静态工具面）和用户提案在同一个 ~4-6 周近零 token 的回路里一起清账。
