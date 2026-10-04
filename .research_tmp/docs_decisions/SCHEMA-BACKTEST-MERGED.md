# Schema v1 表达力回测——三批合并判决 + v1.1 仲裁清单（2026-09-05）

材料：backtest_part1.md（A01-A15 聚合）/ part2.md（T01-T10 时序）/ part3.md（C/D/G 25题 覆盖·条件·配置）。判定者=三个独立子代理（同一 rubric，无交叉复核）。

## 总统计

| 层 | 点数 | E | P | G | U |
|---|---|---|---|---|---|
| must | 196 | 180（91.8%） | 9 | **0** | 7 |
| bonus | 62 | 60 | 1 | 0 | 1 |
| **合计** | **258** | **240（93.0%）** | **10** | **0** | **8** |

P/U 共 18 条 → 跨批去重后 **8 个根因**。三批独立确认的正面结论：absence 三态、epistemic 三值（cited 高负荷命中）、measure 五元组、manifest 双年份+arxiv_id 时序、拍板 A 消融并入、拍板 D 维度词表（批次2 全部条件限定词零映射失败）——无一被击穿。

## 三批一致的重要收敛信号

1. **negative lineage 边三批零需求**——"A fails to improve B" 型硬案例未出现；全部否定性事实被 result.delta 负值 / finding / absence 三态三通道接住（批次1 专列8例、批次2 专列8例、批次3 专列8例，独立得出同结论）。**规格 §7 预留的否定词悬案可关闭（高置信）**。真实的词表压力在别处（见 RC5）。
2. **主张级冲突是真的，但它是跨篇边**——批次1（A03）与批次3（D06-must4）从不同题型独立命中同一结构：finding↔finding 的"反对/不同意"。批次3 补充了关键结构分析：lineage from/to 是 method_ref，加 contradicts 词也连不了 finding；且**记录层抽取单位是单篇，抽 Fedus 时 Z&S 的 finding id 不存在——跨篇冲突边在记录层原理上造不出来**（见 RC1 仲裁）。
3. **G=0 不可外推**（三批各自声明）：批次1 全聚合题型=五元组主场；批次3 明言 gold 与记录层设计**同源自洽偏置**；shift 类型全 50 题 must 层零直接调用（批次1 注记）；表达力≠抽取可行性（批次2 注记7：absence 穷尽性举证与 citation_context lineage 的抽取难度显著高于主结果记录）。第二域（PaperScope 53篇）回测前不下全域总判。

## v1.1 仲裁清单（8 项，按优先级）

### RC3【最高优先·核心负载】数值级超参作为实验条件时五维无家

- 来源：批次3 U×5（D01/D06×3/G04——buffer 1M→10M、10²→10⁵、atom 数 N）；邻域 D06-must4 replay ratio、D08 actor 数。
- 问题：不是 variant（组件开关）、不是 budget（运行规模）、不是 setup（定性协议词）。条件类题（D 类）是视图立身负载，退入 finding 自由文本 = typed 条件化查询失效。
- 候选修法：(a) variant 语义扩至取值级轴；(b) setup 开放注册配置枚举值；(c) budget 扩为任何量化资源设置；(d) **新增第六维 `hyperparam`（typed）：{item, value, unit}，item 名走 Stage 1.5 注册（explicit item + typed value），按方法族 scoped**。
- **推荐 (d)**：(a)(b)(c) 都破坏既有维度的校验语义（variant/setup 是 explicit 枚举，塞数值进去合法性闸就废了）；(d) 是 XBRL typed dimension 的正统翻译（维度名注册、取值开放），且与 config 记录 (item,value) 同构——编译层可交叉验证"result 的条件轴 vs config 的推荐值"，顺带喂条件等价检测。

### RC2【承重前提显式化】registry 实体粒度条款 + finding.scope_ref 指称

- 来源：批次2 注记1（**单条最重要建议**：T04/T07/T08/T10 约 17 个 must 点承重于 registry 收机制级实体；若裁定只收论文级方法，批次2 E 率 88.4%→约50%）+ 批次2 注记3（文外实体处置未明写）+ 批次1 DET-2（A07 U×2：领域级批评无绑定对象）+ 批次1 注记5。
- **推荐修法**：v1.1 显式写入——①registry 实体带 `entity_type ∈ {method, mechanism, practice, out_of_corpus}`（target network/ε-greedy/sticky-actions=mechanism；"RL 评测惯例"=practice；Sutton 1988/DDPG/OTRainbow=out_of_corpus 经 citation_context 注册）；②finding.scope_ref **可空，可指向任何粒度的注册实体**；③lineage 端点同样放开到任何 entity_type（component_of from=n-step 本来就强制了这一点）。

### RC5【词表扩充】relation 七词 → 九词：+generalizes +concurrent_with；否定词悬案关闭

- 来源：批次2 R2（T05-m4 "把C51/QR-DQN/IQN全部收进框架"——extends=方法构建、motivated_by=方向反，均不匹配）+ 批次3 P-3（C07 "TD3 与 SAC 互为 concurrent work"——同期平行恰恰排除全部有向影响词）。两批零散命中但性质同：非否定方向的词表 miss。
- **推荐**：加 `generalizes`（理论收编/统一/泛化，保"框架收编了哪些方法"的可 join 结构=家谱视图核心）+ `concurrent_with`（对称边，时序叙事的关键类型：同期平行 vs 响应影响）。**同时正式关闭否定词悬案**（三批独立零需求，收敛信号2）。词表仍封闭，扩充走的就是本次仲裁通道——这本身是治理机制的首次演练。

### RC4【as-of 要害】文外实体引文年份结构化

- 来源：批次2 R1（PARTIAL×4：Machado 2018/Kaiser+van Hasselt 2019/Kielak 2020/Sutton 1988——as-of 时点边界判定恰是时序题型要害）+ 批次1 注记6（同根因观察级）。
- **推荐采纳**：registry 实体（entity_type=out_of_corpus 及任何文外端点）增可选 `origin_year_cited{value, source_paper_id, quote}`。值来源=引文逐字作者-年份（"(Kielak, 2020)"式），**非 LLM 猜测**——不违反"记录层无 LLM 猜 year"纪律（该纪律约束的是语料论文年份的 manifest 权威性）。

### RC6【零成本】manifest 加 authors/affiliations

- 来源：批次3 P-1（C08 "2018后的DeepMind系论文"机构分组无系统化访问；附带 C05 eponym 归属）。
- **推荐采纳**：Stage 0 manifest 增 authors/affiliations（arXiv/S2 API 非 LLM 元数据，零成本）。

### RC1【结构裁定】主张级冲突边：记录层不加，编译层显式义务

- 来源：批次1 DET-1（建议加主张级 relation 或 lineage 端点指 finding）vs 批次3 N-2（建议留编译层：一致性检测是三层验证第三层已点名职责）。**两报告建议冲突，需仲裁。**
- **推荐批次3 方案 + 一个记录层补强**：跨篇冲突边在单篇抽取的记录层原理上造不出来（收敛信号2）；修法=①finding 增可选 `target_ref`（被批评/反驳的注册实体，RC2 之后 practice/method 都可指）——给编译层提供可 join 的指向键，替代自由文本 NLP；②"主张级冲突检测（contradicts 边）"写进块2 三层验证第三层的显式义务清单。**记录层不动 lineage。**

### RC7【最低配】图形层数据载体

- 来源：批次3 P-2（C08 图-only 置信区间无家，"是否报方差"分类对图-only论文系统性盲区）+ 批次1 注记7（"已报告但文本抽取不可达"在三态里没有专态，靠 not_reported+evidence 弹性承载）。
- **推荐最低配**：①paper_card 实验矩阵扫描扩为"**图表清单**"（哪些图/表存在 + caption 逐字），caption 内容可路由进 finding/result（有正文对应数值时）；②absence 不加第四态，但把"图-only"路由规则写进抽取 prompt 规格（evidence 里明示"仅图X，文本不可达"）；③图形数据本体（曲线点值/CI 阴影）**记为已知限制**——图表数字化资产是独立线，接入与否后置仲裁，不阻塞 Stage B。

### RC8【轻量】quality_flag 公共字段

- 来源：批次1 DET-3（A05 bonus：表头解析错位警示不随记录走，gold 要求的"谨慎标注"无法自动传播）。
- **推荐采纳**：公共字段增可选 `quality_flag`（枚举：suspect_binding / parse_warning / figure_only …），Stage 3 后检与抽取器可打标，渲染层必须带出。

## 观察级（不进 v1.1 仲裁，转三个后续落点）

**→ 块2（编译层）规格义务清单**：协议可比性判定与分带 / 派生 absence 与来源标记区分 / 时间线合成 / 跨篇数值漂移检测（同篇双值两条都留）/ 排序冠军判定 / 主张重叠与冲突检测（含 RC1 的 contradicts 边）/ metric 归一化归属（批次2 注记2）/ 时序口径固定=arxiv_id、venue_year 仅渲染（批次2 注记8）/ 提及计数类=扫描工具能力非记录层（批次3 注记2）/ finding 子体裁分类 claim_type（批次1 注记2，四视图需要时再加）。
**→ Stage 2 抽取 prompt 规格**：定性结论→finding、数值→result 的路由规则（批次3 注记3）；公式型 config value 合法（α/4、ε_i=0.4^{...}），Stage 3 对公式走 quote 逐字包含通道而非数值归一化（批次2 注记4 + 批次3 注记4）；setup 枚举生长将被频繁触发（稀疏奖励/部分可观测/大动作空间类抽象任务性质，批次3 注记5）——仲裁队列要有吞吐预案。
**→ 冒烟验收关注项**：absence 型与 citation_context 型 lineage 的抽取难度单列观察（批次2 注记7）；delta 基线模糊案例（"previous SOTA"）依赖 registry canonical 化压力测试（批次1 注记8）；维度值归一化质量（no-op 变体、Atari-57/55/49/26 四档）决定聚合题成败（批次1 注记4）。

## 诚实限定（合并级）

1. 三批判定者互不交叉、无第二判定者复核；P/E 边界依赖"组装级归编译层"的规则授权，若仲裁采更窄解读，批次2 的 T01-m3/T07-m6/T10-m4 等需复核。
2. **同源偏置不可由回测自身消除**：gold 出题视角与记录层设计同源（批次3 注记8），E93%/G0 应读作"schema v1 对已校准负载充分"，不是全域充分证明。独立出题人+第二域回测（PaperScope）是仅有的解毒剂，都在协议补课清单里。
3. shift 类型全 50 题 must 层零直接调用——它的价值证据来自 PaperScope trend 类失分（外部基准），本回测既不能证实也不能证伪，冒烟+第二域时重点看。
