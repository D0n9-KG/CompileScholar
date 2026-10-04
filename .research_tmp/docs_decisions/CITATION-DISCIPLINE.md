# 引用纪律：禁语清单 + 必引清单（三层）

> 2026-09-08 立档，执行 EDITORIAL-DECISION 桶A-3/A-4。适用范围：论文稿、abstract、对外报告、评审回复、记忆档案 headline。所有 ID 均一手核验（核验档案：literature/novelty_recheck_2026-09-06/new_entrants.md、literature/eval_practice_and_baselines_2026-09-08/、literature/survey_agent_knowledge_model_2026-09-03/surface_C.md）。

## 一、禁语清单（出现即违例，写作与审稿回复前逐条对照）

| # | 禁语 | 违例原因 | 许可替换形态 |
|---|---|---|---|
| F1 | 无限定"直读也失效"/"强模型修不好聚合" | E2-R2 修正档案明文推翻：那是中端 deepseek 产物；Qwen3.8-Max 关思考直读 4.00/聚合 4.33 | "中端模型档直读失效按分散度分层（E2 原口径限定）+ 规模档退化实测（166k→339k 档 3.50→3.00）+ 93 篇算术墙"（C1 重锚形态） |
| F2 | "首次三态"/"首次覆盖缺口" | Cochrane Handbook 三态逐字指令在档（surface_C 09-06）；3ie evidence gap maps 同族 | "缺席三态作为记录层一等公民+工具化（find_gap）+题型增益因果链"；coverage 3.8 胜绩须**同页引 3ie/Cochrane** |
| F3 | 裸引官方数字（PaperScope 40.95 等） | part_A §2.5/§三.3：官方数字基于 200 题分层子集、四维不可比、禁做主对比 | 标注引用="难度参照行"（引用时写明子集口径与不可比性），永不进排序/对比表 |
| F4 | 裸用 "knowledge compilation" | ASKS（2608.29612）已正式定义并命名该术语；RAGFlow/Pinecone/Meta 产业占用——叙事窗口关闭中（new_entrants §结论1） | 引用 ASKS 划界后使用，或改用自有术语（"需求驱动的类型化文献知识底座/记录层编译"）；差异化钉死在"持久知识层+类型化记录+验证维护+需求驱动评测"组合而非"编译"一词 |
| F5 | "非劣达成"/"非劣闭环"/"追平直读"作 headline/abstract 句 | B3 余量 +0.02=1 题分 ≪ 方差带 ±0.1；配对 CI [−0.34,+0.18] 跨 0；余量全由修复靶题带贡献（B3 报告 §10.1/10.2） | "测量分辨率内不可区分（tie）"；冻结闸字面 PASS 只在实验档案层陈述 |
| F6 | "收益来自表示语义而非通用 ReAct"归因句 | 耦合两腿：token 不对称 2.2× 未排除（B3 §10.3）；等预算对照（桶B7 token-matched flat ~1M / 桶B6 H7(d) ~600k）未跑 | "耦合收益有受控初步证据（同轮配对 +0.46 主证/交互项 +0.50 次证，探索性闸）"；归因句待 R2 耦合闸（A1 vs A9）+桶B7 落地 |
| F7 | "循环结构增益（而非 token 预算增益）"排他表述 | 同 F6：每题 12.2k→25.5k（2.09×）换 +0.58，无等预算对照前逻辑不可辨认 | 报双口径+挂账披露 |
| F8 | 全成本口径"成本 1/7" | 只计查询侧：记录层构建（抽取 token+治理人工时）与 prompt caching（flat 有效成本可压数倍）均未入；v6 CRITICAL#1 未解除 | "查询侧口径 1/7；摊销+缓存 break-even 模型（桶B8）交付前不作完整成本主张" |
| F9 | "三场证据战收官/闭环"式权重表述 | DA-CR1：B3 PASS 证据学上是临时的（bug-free n=1 在样本内），R2 才是确证；框架不得颠倒证据权重 | "瓶颈迁移链完整；确证 deferred 至 R2 CI 判据" |
| F10 | abstract 归因句（任何形态） | R3 条款+席间分歧裁定2：负载循环性未解除（R2 主闸+耦合闸落地前） | abstract 只陈述可验证读数与协议，不陈述归因 |
| F11 | 引用禁引名单：CILK/SciShin/CorNorm/PIE | 假引文/不可核验前科（react_loop 排雷清单） | 不引；如需同族论点换用已核验一手文献 |
| F12 | PaperQA2 误标 2409.13731 | 2409.13731=KAG；PaperQA2=**2409.13740**（勘误已入排雷清单） | 引前核 ID |

## 二、必引清单（三层重组）

### 层1 对标（同生态位系统/基准——定位句与对比表义务）

| 工作 | 一手 ID | 必引理由 | 关系 |
|---|---|---|---|
| SciAtlas（浙大+UCL） | arXiv 2605.22878 | 最危险主对照：动机句同构；v2 补齐全套自跑基线=生态位事实准入标准 | 主对照；三轴半真空对其仍成立（(c) access 半轴被其 v2 §4.3 占） |
| Mechanist（SciAtlas 同团队） | arXiv 2608.12036 | 抱怨 SciAtlas 粒度不足被迫自建 13k 专域 KG=我方粒度论点同生态自证 | 佐证+威胁（需求驱动 schema 扩展演练） |
| ASKS | arXiv 2608.29612 | "scientific knowledge compilation" 术语正式定义方；治理闭环撞车（GraphDelta/事务融合/可重放） | 术语划界（F4）；概念邻居非系统竞争者（零基准/无 agent 消费层/56 篇） |
| AskChem | arXiv 2607.28618 | claim-centered 基础设施最完整先例：类型化 claim+verbatim 接地+MCP agent 接口+自带基准；2.4M claims/147K 篇 | 对标；差异=无演化谱系/覆盖/维护治理，化学单域 |
| Materials Explorer | arXiv 2606.27384 | schema 演化机制最直接先例（catchall→显式升格=我们同款设计）+覆盖缺口可见化+跨论文数值对比 | 对标；差异=面向科学家浏览器非 agent 原生、无方法演化边、无基准 |
| Intern-Atlas | arXiv 2604.28158 | 方法演化关系类型占位（extends/improves/compares/replaces/adapts/background 几乎一字不差）+9.4M edges | 正面引作 binary/static 先例；差异=n-ary+mutable+schema 并进演化 |
| DIAL-KG | DASFAA 2026 | in-loop 闭环声明先例（无 ablation 验证） | 差异=首次严格验证 in-loop/耦合收益（我们贡献位） |
| Lacuna | arXiv 2606.26246 | 同生态位（recall 0.028 警训出处） | 对标 |
| Agents-K1 | arXiv 2606.13669 | 同生态位占位（2026-06） | 对标 |
| PaperScope（基准方） | arXiv 2604.11307 | R2 外部考场：120 题/53 篇精简版+官方源码复刻臂（A9） | 基准；官方数字遵守 F3 |
| PaperArena（基准方） | arXiv 2510.10909 | 跨论文 agent 基准同族（9 LLM×工作流+PhD 人类基线） | 基准生态定位 |
| 基线四件：LightRAG / PaperQA2 / HippoRAG 2 / OpenScholar | 2410.05779（EMNLP25 Findings）/ **2409.13740** / 2502.14802（ICML25）/ 2411.14199 | R2 对比臂 A5-A8 | 全部自跑（part_A 惯例判决：cite-official=0/19），统一性明文+判分协议披露 |

### 层2 区分（红线划界——机制撞车点，不宣称首创）

| 工作 | 一手 ID | 撞车机制 | 划界句义务 |
|---|---|---|---|
| Doctor-RAG | arXiv 2604.00865 | error-type→fix 映射 | 我们的 type→action 手册须引其为先例，差异=表示语义 pre-emptive 编译（非运行期诊断） |
| CEL（+MS Foundry） | arXiv 2509.25052 | KB→playbook 编译 | 引为先例；差异=编译产物消费方是文献 QA 负载+缺席/条件语义 |
| SCAIR | arXiv 2607.22571 | schema-conditioned agentic reasoning | 引为先例；差异=schema 与抽取并进演化+治理环 |
| LedgerMind | arXiv 2607.28374 | 运行期账本（威胁极高） | 划界=我们编译期记录层 vs 其运行期账本；必引 |
| Cochrane Handbook + 3ie evidence gap maps | 机构文档（surface_C 09-06 核验，三态逐字指令在档） | 缺席三态/覆盖地图的领域先例（循证医学轴） | F2 义务：coverage 胜绩同页引用；差异=计算化记录层+agent 工具消费 |
| compilation 叙事占用方：CCA / mimeo / IdeaForecastBench / ISC | 2609.00759 / 2609.00453 / 2609.00747 / 2608.20845 | "编译优于直读"叙事 ≥4 组独立提出 | F4 义务：术语窗口关闭的证据；定位靠四件套组合非叙事词 |

### 层3 反方弹药（审稿人必用——主动引用+预先划界，不引=被动挨打）

| 工作 | 一手 ID | 攻击形态 | 我方预先处置 |
|---|---|---|---|
| Fidelity Before Structure | arXiv 2601.00821（v4 07-22） | 受控消融：verbatim 块胜抽取类型化工件 15.9-22.0 分（对话记忆域）——"结构化记忆应增强而非取代原文" | 主动引+划界：我们"类型化记录+原文 verbatim 双存"恰与其 augment-not-replace 结论兼容；域差异（对话记忆 vs 文献）如实标注 |
| IBM M20（How Much Structure Should Agentic Graph Memory Build?） | VLDB 2026 AgentGraph workshop PDF（vldb.org，GET 206 核验；会期未核验——引用时注明） | "该建多少结构"实证配方：每层结构须证明下游收益+质量-成本前沿——出自 Text2KGBench 作者，审稿人大概率引用质询 | 主动引+对接桶B义务：组件消融三臂（桶B5）=逐层结构收益证明；frontier 双档（桶B8）=质量-成本前沿 |
| hybrid 先例（自家） | memory/hybrid-experiment-verdict | 开发集对齐效应实测 +0.96→−0.45 反转（出题人=记录设计者循环） | 论文如实呈现（DA-CR1 第5点）；R2 外部题=解药 |
| 验证期 oracle 泄漏（自家） | STAGE0-FINDINGS §11 | 历史 +0.83 为 oracle-scoped 条件 | 时间线如实呈现（DA-CR5 处置）：字面触发→审计揭混杂→闸前瞻化；R2 全语料形态 |

## 三、执行钩子

1. 论文稿每节写完 → 对照 F1-F12 逐条扫（grep 级自查）；
2. related work 骨架 = 层1 表格行序；方法节每个机制首次出现 → 查层2 是否有划界义务；limitation 节 → 层3 全表逐条给处置段；
3. 新调研发现的高威胁工作 → 先归档核验 ID，再判入哪层（节拍同 registry 仲裁：发现→核验→归层→冻结）；
4. 投稿前按 EIC 条款补一次三条检索式增量扫描（DB 理论轴/EBM 轴/agent memory 轴），新命中入层。
