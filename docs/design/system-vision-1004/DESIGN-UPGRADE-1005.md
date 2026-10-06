# 整体升级设计：历时领域认知系统（DESIGN-UPGRADE-1005，v1 草案）

> 用户裁定（10-05）："直接开干"——先把整体升级方案设计清楚，再全面升级，用实验迭代，最后成文。
> 叙事：NARRATIVE-V9-1005.md（历时领域认知）。评测：EVAL-PLAN-1005.md（现成基准，不做大规模人工标注）。
> 本档每个数字都是 10-05 实测的，出处在 §11。基准格式细节等 BENCHMARK-FORMATS.md 回来后补进 §7。

## 0. 一句话

在大规模 CS 论文语料上，对每篇论文抽**自述**（它说自己做了什么），对每个引用标记抽**他述**（后来的论文说被引者做了什么）。两者确定性地落到"论文 + 首版日期"上，再编译成一个**截至任意 T 都能重算**的领域认知：方法族、演化、每个工作的定位、常用基线、公认局限、已知但没读的论文。这份认知以带 `as_of` 参数的工具形式交给科学智能体，用于新颖性判断、基线选择、前置工作、局限识别这类任务。

## 1. 设计原则（从这轮审查和调研里来）

1. **一条数据流，一套 schema。** 四套抽取、两套状态、六套演化词表全部收成一套（INTERNAL-AUDIT 断点 1、10、40）。旧 KB v2 冻结，不再改。
2. **身份和时间全部确定性处理。** 引用标记 → 参考文献条目 → 论文 id 这一段不用 LLM，日期一律取 arXiv 首版日期。LLM 只做"判断"。对比：Arnaout 的引用年份归属准确率只有 0.59。
3. **表述是原子单位，领域认知是表述的纯函数。** `cognition(T) = f({表述 | date ≤ T})`，不存快照，任何 T 都能重算。
4. **不判对错，只记差异。** 他述和自述不一致就并列给出（用户裁定：领域认识在发展，不叫错误）。
5. **他述稀疏时有后备。** 发表不到一年的工作几乎没有他述（ScholarCatalyst 正例中位年龄 17 个月，35% 不到 1 年）。这时用"自述 + 它引了谁 + 它在族里的位置"顶上，并标明证据量。
6. **增益必须来自结构，不是文本量。** 每个工具返回紧凑的结构化结果加原句出处，篇幅预算固定。评测必须设"同预算平铺检索"和"每次现做摘要"两个对照（Old Ideas、SCOPE、RINI）。
7. **原有能力不退化。** 新 schema 是在原有内容上加维度、加一遍引用句抽取，不替换原内容。CS2 dev 回归非劣，界 −0.02。

## 2. 总体架构

```
L0 语料与元数据     arXiv OAI-PMH（cs，带 created 日期）+ 本地 OAI 快照（到 2404）+ OpenAlex 补洞
                    → papers 表：arxiv_id, v1_date, title, abstract, authors, categories
L1 全文与引用解析   arXiv HTML（LaTeXML）为主；LaTeX 源码 / IdeaForecastBench Markdown / 共享盘 PDF 兜底
                    → 句子切分 → 引用标记 → 参考文献条目 → 论文 id（确定性）
L2 统一抽取（LLM）  (a) 自述：每篇论文的贡献、提出的方法（含 role）、实验设置/基线/数据集、自述局限
                    (b) 他述：每个"引用句 × 被引论文"一条：被引者做了什么、关系类型、被指出的局限、并列分组
L3 表述库           statements 表：speaker, about, date, kind, relation, text, quote, span（全部带出处）
L4 编译 cognition(T) 实体与身份（方法 ← 提出论文）→ 族（多来源成员证据）→ 演化边 → 定位画像
                    → 族级事实（支持集 / 独立数 / 条件）→ 认识变迁事件 → 边界（已知未读）
L5 消费             工具 API（as_of 参数）→ MCP server / Python 接口；答题管线；人读视图
L6 补全             缺口诊断 → 分型获取（未读成员 / 时间前沿 / 未覆盖区域 / 补读全文）→ 回到 L1
```

## 3. L0 语料与元数据

- **范围**：先覆盖评测基准用到的 CS 子领域，也就是 cs.AI / LG / CL / CV / IR / RO / NE / MA 加 stat.ML，时间 2018-01 至今。之后按补全逐步扩张。
- **来源**：
  - arXiv OAI-PMH（`oaipmh.arxiv.org/oai`，set=cs，已实测可用，带 `<created>`）做增量全量拉取。
  - 本地 OAI 快照只到 2404.03658（2024-04），够历史部分用。
  - OpenAlex（arXiv 源 S4306400194）补 DOI 和引用数。
  - 非 arXiv 论文（会议版、期刊）作为**桩节点**，只要标题、年份、DOI，不抽内容。
- **日期**：v1_date 取 arXiv 首版 created，精确到天。非 arXiv 论文按出版年月，标注粒度。
- **规模估计**：上述类别 2018 年以来大约几十万篇。抽取不全量做，见 §5 的分层预算。

## 4. L1 全文与引用解析（确定性，不用 LLM）

| 来源 | 引用标记 | 参考文献→论文 | 实测 |
|---|---|---|---|
| arXiv HTML（LaTeXML，主路径） | `<cite class="ltx_cite">` 内含 `href="#bib.bibN"`，标记到条目是锚点直连 | 条目文本 → 显式 arXiv 号，否则标题解析（本地快照 → OAI-PMH 标题 → OpenAlex） | 2404.01039：238 个 cite，177 个 bibitem，18 个带 arXiv 号 |
| LaTeX 源码 e-print | `\cite{key}` + .bbl/.bib | 同上 | 可下载，2404.01039 共 118 个文件 |
| IdeaForecastBench Markdown（2023-01..2025-10，108,768 篇） | `[n]` 数字式 | 文末编号列表 → 标题解析 | 样本：65 个数字引用，文末有编号参考文献 |
| 共享盘 arXiv PDF（4.68 TB） | 解析难度最高，只作最后兜底 | — | 有 cs 分包 |

- **复用**：`compile/skeleton/bib.py`（书目解析与标记，E1 精度 200/200）和 `resolve.py`（条目→论文）推广到所有论文；新增 HTML 解析器 `sources/arxiv_html.py`。
- **限速**：arXiv 要求 15 s/请求（robots），单路下载每天约 5,700 篇。对策：
  - 优先用 IdeaForecastBench 已有全文和共享盘资产；
  - HTML 只拉"引用句来源论文"（见 §5 的抽样策略），不全量拉；
  - 是否走 arXiv 官方批量源（S3，付费）待用户决定。
- **产出**：
  - `citations` 表：citing_id, citing_date, sentence, section, marker, bib_raw, cited_id, resolve_method；
  - 一句同时引用多篇时，额外产出一条 `group` 记录（成员列表，即"这几篇被并列"）。

## 5. L2 统一抽取（LLM，本地 27B）

**一套 schema、两遍调用**，原有内容全部保留：

- **自述遍**（每篇论文一次，读摘要加引言和方法节的首段）：
  - `contribution[]`：本文做了什么；
  - `method_proposed[]`：名字 + 一句话 + 原句；
  - `relations_self[]`：(role ∈ extends/improves/replaces/adapts/combines/uses/compares, target 引用标记, 原句)，target 已由 L1 解析到论文；
  - `baselines[]`、`datasets[]`、`metrics[]`（来自实验节）；
  - `limitations_self[]`；
  - 保留旧 schema 的 finding / result / config。
- **他述遍**（每个 citation 一次，按批处理）：
  - `cited_did`：被引者做了什么；
  - `relation`（同一词表，从施引者角度）；
  - `limitation_of_cited`：可为空；
  - `function` ∈ background/basis/baseline/contrast/data/tool。
- **身份**：方法名 → 提出论文。依据是自述"本文提出 X"加若干条他述"[n] 提出了 X"，即 P0-3 的推广。泛名（transformer、LSTM）标为 generic，不强行落到某篇论文。
- **预算**（10-05 实测，修正此前按"调用次数"的估算）：瓶颈是解码 token，不是调用次数。
  - 本地 27B 的聚合解码速度：并发 16 时约 460 tok/s，并发 48 时约 720 tok/s；单流约 30 tok/s。
  - 他述遍：24 对一批，耗时 58 s（单流），每对约 120 输出 token。按约 700 tok/s 算，一小时约 2.1 万对。
  - 自述遍：每篇约 18 s（单流），输出约 600 token，一小时约 4,000 篇。
  - IdeaForecastBench 全量：他述约 560 万对（每篇平均 51.5 对 × 108,768 篇），约 270 小时，**不全量跑**。
  - 他述只抽"被引的目标论文"：每篇被引论文最多 N=30 对，按施引日期分层；候选池为基准正例与候选、族成员、高被引论文。先精算规模再开跑。
  - 自述：只跑被引的目标论文和施引来源论文，不跑全部语料。
- **质量闸门**（不做大规模人工，见 [[no-large-scale-human-annotation]]）：
  - 原句必须是原文子串；
  - 关系与功能抽 200 条，两个模型独立标注（E1 方式），只看分歧项；
  - 50 篇论文做新旧 schema 覆盖率对照。

## 6. L3 表述库 + L4 编译 cognition(T)

**表述（statement）**：`{id, speaker_paper, about_paper|about_method|about_family, date, kind ∈ self|other, relation, text, quote, source_span}`。

**cognition(T)** 只用 date ≤ T 的表述，由编译器产出：
1. **方法与身份**：method = (规范名, 提出论文, 别名)，别名来自他述里对同一提出论文的不同叫法。
2. **族**：多来源成员证据加权。
   - 来源包括：并列引用的分组（同一句并列，确定性）、综述章节的归属（综述也是一种他述）、自述里的 extends/adapts 关系、引文耦合和共被引、LLM 对成员描述的归纳（field_state 降为其中一个来源）。
   - 输出族、成员（带置信度）、定义、代表工作。
3. **演化边**：方法→方法，带类型、自述/他述的出处、日期。n 元 combines 保留为一条超边。
4. **定位画像**（每篇论文 × T）：
   - 自述贡献；
   - 他述汇总：被认为做了什么、通常被当作什么（basis / baseline / component / contrast 的比例）、被指出的局限；
   - 自述与他述的差异；
   - 证据量（他述条数、施引论文数、时间跨度）。
5. **族级事实**：性质、局限、比较，各带支持集、独立论文数（去掉同作者）、条件原句；状态 ∈ consensus / contested / qualified / single-source，由支持集算出。
6. **认识变迁事件**：基于定位画像随 T 的变化检测，类型包括：
   - 被重新归类（族归属改变）；
   - 范围被收窄（他述里出现限定条件）；
   - 暴露新局限；
   - 被替代（出现 replaces 边且其后他述转为 contrast / baseline）；
   - 成为默认组件（uses / component 占比越过阈值，P1 的信号）。

   每个事件带日期和证据句。
7. **边界**：被引 ≥ 2 次但未抽取（没有全文）的论文，以及他述指向的未读论文；每族报"已知 N、已读 n"。

实现：编译器是批处理，按 T 网格（月）缓存族划分；定位画像和事实按需计算（纯函数加缓存）。

## 7. L5 工具 API（给智能体）

所有工具都带 `as_of`（日期），返回紧凑的结构化结果，每条附 1–2 句原句和论文 id，返回长度有上限（同预算原则）。

| 工具 | 输入 | 输出 | 主要服务的基准 |
|---|---|---|---|
| `field_map(topic, as_of)` | 方向描述 | 相关族、定义、代表工作、族间演化、边界计数 | 通用；ScholarCatalyst、TaxoBench |
| `paper_profile(paper, as_of)` | 论文 id 或标题 | 自述、他述汇总、角色比例、局限、变迁事件、证据量 | PreScience、ScholarCatalyst、LimitGen |
| `closest_prior(idea, as_of)` | 想法描述 | 最相近的已有工作，按贡献维度（目的/机制/评测/应用）对齐，附其贡献 | Old Ideas、NovGauge |
| `baselines_for(problem, as_of)` | 问题描述 | 该方向常用的对比基线与数据集，附被当作基线的频次和时间趋势 | MasterSet |
| `open_issues(topic, as_of)` | 方向描述 | 族级局限与开放问题，附支持集和状态 | LimitGen |
| `frontier(topic, as_of)` | 方向描述 | 最近的工作、正在被替代的方法、正在成为组件的方法 | Proof of Time、IdeaForecastBench |

- **接入方式**：
  - MCP server（复用 `baselines/harness/mcp_server.py` 的框架），给 Claude Code 类 harness 和 ScholarCatalyst 的 toolcall agent 用；
  - Python 接口，给自家管线和 IdeaForecastBench 的"历史压缩"位用。
- **as_of 的强制方式**：工具层按 T 过滤表述，基准每题的截止由适配器传入，智能体不能改。

### 7.1 各基准接入点（BENCHMARK-FORMATS.md，10-05）

| 基准 | 论文 ID → 我们 | as_of | 接入点 | 注意 |
|---|---|---|---|---|
| ScholarCatalyst | `arxiv_YYMM.NNNNN` 去前缀即 arXiv id（语料 190,543/190,896）；`oa_W…`/`s2_…` 共 353 条按标题对齐 | 官方按月（同月放行）；主口径取 `paper_published` 所在月末，严格口径取前一天，两者都报 | 新 agent 模块注册进 `agentic_search.py::AGENTS`，以 `toolcall.py` 为模板加工具；返回的 id 必须是语料 id | 官方只有标题+摘要，`expand`/`read` 依赖的引文图和全文未发布 → 他述必须由我们自带；`llm_rerank.OPENAI_BASE` 写死，接兼容端点要改常量 |
| IdeaForecastBench | 全是 arXiv id | 按月（截止月 1 号） | `IdeaStrategy.generate(train_papers, cutoff_month, top_k)` 子类，注册进 `strategy/registry.py::create_strategy` | `generate()` 拿不到 topic id；基线只见本 topic 2024-04 后的论文 → 我们用更广的历史就是"信息条件不同"的一臂，单列报告；judge 默认 gpt-4.1-mini + Voyage，换本地后不可与论文数字比 |
| PreScience prior work | S2 corpusId，每行带 `arxiv_id` | 按天（`date < cutoff`） | 新 baseline `predict_references(author_ids, cutoff_date, …)`，输出表内 corpus_id | **输入只有作者 id 和日期，没有目标论文的标题和摘要** → 我们的工具只能从作者历史出发，契合度低于预期，降为次要 |
| MasterSet Type I | 内部 UUID，只有 title/venue/year → 标题对齐 | 按年（池 2018–2024，题 2025） | 输出池内 paper_id 排序，复用 `compute_all_metrics` | 与 AgentExpt 是两篇不同论文（AgentExpt 数据未发布）；约 16% 金标不在池内，Recall 上限 ≈0.84 |
| Old Ideas | 只有标题（ICLR 2026 投稿）→ 标题对齐 | 全局 2025-03-01 | 自写 `retrieval_cache.json` 的 `related_work` 字段 | 评委走 OpenAI Responses API，兼容端点若不支持要改代码；必须预注册多提示 |
| NovGauge | 431/926 篇有 arXiv id | 只有年份，任务无截止 | 替换 `run_bench.py` 组装论文内容那一步 | 测的是"两篇论文在任务/问题/方法上是否相似"，不是新颖性打分 → 作为 `closest_prior` 的诊断 |
| LimitGen-Human | 疑似 OpenReview forum id → 经 OpenReview 转标题 | 原无截止；我们取 ICLR 2025 截稿前 | 替换 `main_human.py::paper_retrieval()` 的返回 | 需要全文（已提供）；官方代码有 bug（literature 用错 prompt、`rating.py` 只评前 10 篇），修了要披露 |
| TaxoBench Bottom-Up | 只有标题 → 标题对齐（阈值 0.92） | 无（综述年份未知，要按标题查） | 写 `{"id","hierarchy_tree"}`，跑 `taxobench-score` | 记忆泄漏风险高 |

### 7.2 对 L1 的直接要求（来自 IdeaForecastBench 40 篇抽样）
- 两种引用风格都要支持：数字式 21/40 篇（标记→条目 89%），作者-年份式 19/40 篇（99%）。`compile/skeleton/bib.py` 已支持两种（E1 两类各 100/100），直接复用。
- 要处理的噪声：alpha 标签 `[AZLS19]`；MinerU 丢失方括号；LaTeX 下标 `w_{2}[0]` 误识别为标记；参考块标题不规范，位置在全文 27%–96% 之间（有附录在后），不能简单取文末。
- 条目 → arXiv id 主要靠标题匹配（自带 arXiv 号的条目只占一部分）→ 需要 2024-04 之后的标题索引（OAI-PMH 补齐）。

## 8. L6 补全

缺口分型，各自走不同动作：
- 未读成员：按 id 拉全文；
- 时间前沿：走 OAI-PMH 增量，加正向引用；
- 未覆盖区域：先找该区域的综述，再做语义检索；
- 已读论文缺某方面：补读全文。

离线跑，不看题目；评测期间不写回共享库（防止按题补库）。闭环本身不当卖点（Consensus、LLM Wiki 已有）。

## 9. 代码组织（新包内）

```
src/compilescholar/
  sources/  arxiv_oai.py(新)  arxiv_html.py(新)  arxiv_snapshot.py  refgraph.py  sciverse.py
  corpus/   (新) papers 表构建、全文获取与缓存、句切分
  citations/(新) markers + bib 解析（迁入 skeleton/bib.py、resolve.py）、group 记录
  extract/  (新) self_pass.py  other_pass.py  schema.py（唯一 schema）  validate.py
  cognition/(新) identity.py  families.py  lineage.py  profiles.py  facts.py  shifts.py  boundary.py  asof.py
  tools/    (新) api.py（六个工具）  mcp.py
  answer/   pipeline.py（改为调用 tools，保留旧通道作对照）
  eval/     现有 + adapters/{scholarcatalyst,prescience,masterset,oldideas,limitgen,ideaforecast,taxobench}.py
```

- 存储：SQLite（papers / citations / statements / methods / families / edges / facts / events），加向量（qwen3-embedding-8b，本地）。
- 旧 `kb/`、`compile/state/`、`compile/skeleton/` 迁入新位置，旧接口保留到所有对照臂迁完。
- 表征测试照旧：迁移不改行为，新增的部分各自带单测。

## 10. 实施顺序与迭代

1. **L0 + L1**：语料表、HTML 解析、引用句抽取与解析，带单测。验收：在 IdeaForecastBench 和 HTML 两条路径上，各抽 200 条"标记→论文"做双模型核对，精度 ≥ 0.95。
2. **L2 两遍抽取**：先跑基准候选池加高被引论文。验收：§5 的质量闸门。
3. **L3/L4 编译器**：cognition(T)。验收：TaxoBench Bottom-Up（只评地图）、P2 协议复跑（定位画像替换原句）。
4. **L5 工具 + 基准适配器**：先接 ScholarCatalyst、MasterSet，再接 Old Ideas、PreScience、LimitGen、IdeaForecastBench。
5. **主实验**：四条件配对，按被引工作年龄分层报告，预注册写进 PREREG 修订 2。
6. **迭代**：只在 dev 切分上迭代，每个基准切固定 dev/test；test 只跑一次。
7. **成文**。

## 11. 本档用到的实测（10-05）

- 本地 27B 短调用吞吐：并发 32 → 10.7 次/s，64 → 14.8，96 → 15.6。
- arXiv HTML 2404.01039：238 个 `ltx_cite`（带 `#bib.bibN` 锚点），177 个 `ltx_bibitem`，其中 18 个带 arXiv 号、0 个带 DOI。
- 本地 OAI 快照最新 id 为 2404.03658；arXiv OAI-PMH set=cs 可用（带 `<created>`）；OpenAlex arXiv 源 2026-09 起 46,980 条。
- IdeaForecastBench 行字段：arxiv_id / month / title / text；样本全文 50,207 字符，65 个数字引用，文末有编号参考文献。
- ScholarCatalyst core：207 题，661 个正例；正例年龄中位 17 个月，35% 不到 1 年。
- P1：384 万条边上，同一被引论文晚期比早期 uses_component +0.079、lineage −0.031（扣施引年份基线）。
- P2：他述 − 自述，族级性质 +0.084 [+0.039, +0.131]；局限一项不支持。

## 12a. 系统一体化契约（10-05，用户要求"当成一个系统完整升级，不要模块割裂"）

**一条主流程，一个存储，一套身份。**
- 存储：`data/dfc/`，每层一组 SQLite 表。每层只读上一层的表，原始文件（parquet、HTML、OAI）只由 L0/L1 读。
- 身份：论文 = `arxiv_id`；非 arXiv 文献 = `stub:<norm_title>`。日期 = `v1_date`（精确到天）。所有基准适配器都映射到这套身份，不允许另起 id。
- 原子单位：`statement`（谁、何时、关于什么、说了什么、原句、出处）。L4 及以后的一切都是 statement 的函数，L4 不回头读全文。
- 唯一 schema：`extract/schema.py` 定义 statement 类型和**唯一的关系词表**，两遍抽取、编译器、工具共用。旧的 6 套演化词表全部作废。

**流程编排**：`compilescholar build <stage>`，stage ∈ papers → citations → extract → cognition → tools。每个 stage 写一份 manifest（输入表哈希 → 输出表哈希 + 计数），可续跑、幂等；任何一层改了，下游自动失效、重算。

**构建顺序：先打通全链路，再加量。**
1. 设计是完整的，不做"最小版"。
2. 第一批数据（一个领域、约 2,000 篇）先从 L0 一路走到 L5 和一个基准适配器，用它暴露接口对不上的地方；之后扩量只是让更多数据走同一套代码。

**系统不变量（端到端测试强制检查）**：
- 每条 statement 的原句都是其来源全文的子串；
- 每条领域级事实都能追溯到 statement，且全部满足 `date ≤ T`；
- as_of 单调：cognition(T) 不使用任何 `date > T` 的表述，T 更大时只增不减（族划分重算除外，需记录）；
- 每个工具返回的论文 id 都在 papers 表里，或是带标记的 stub；
- 基准适配器传入的截止日期由适配器决定，智能体改不了。

**新旧模块去留**：

| 现有模块 | 去留 |
|---|---|
| `compile/skeleton/bib.py`、`resolve.py` | 并入 `citations/`（已复用，并完成迁移） |
| `compile/skeleton/proposes.py` | 并入 `extract/` 自述遍（"本文提出"是 role=proposes 的一种） |
| `compile/state/field_state.py`、`coarse.py` | 族归纳逻辑并入 `cognition/families.py`，降为族成员证据的来源之一；旧接口只保留给留出检验复现 |
| `kb/`（KB v2 + 混合检索） | 冻结，作为 v9b 记录与"平铺检索"对照臂；新系统的检索索引建在 statement 上 |
| `answer/pipeline.py` | 改为调用 `tools/`（规划用 field_map，取证用 paper_profile / closest_prior）；旧通道只作对照臂 |
| `sources/refgraph.py`、`sciverse.py` | 留在 L6 补全，作为外部获取通道，不再在答题期临时调用 |
| `eval/*`、`baselines/*` | 保留；新增 `eval/adapters/`，所有基准经同一套 tools 接入 |
| `.research_tmp` 旧 KB 构建链、views、growth 脚本 | 退役，不迁移，只留归档 |

## 12. 用户裁定（10-05）

1. 不付费使用 arXiv S3。全文来源：IdeaForecastBench 自带全文（2023-01..2025-10，108,768 篇）+ 按需拉 arXiv HTML（遵守 15 s/请求）；2025-10 之后的覆盖较慢，不够再议。
2. 语料从 2018 年起，他述每篇被引论文最多取 N=30 条，按施引时间分层抽样；跑之前先精算总量。
