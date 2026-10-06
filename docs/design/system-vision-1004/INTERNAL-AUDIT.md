# 知识层内部审查（INTERNAL-AUDIT，2026-10-04，只读）

范围：答题 KB（`cs2/base_kb_v2`）、它的构建链、新包 `src/compilescholar`、新骨架数据 `data/state/`、超图时代存档。
约定：除特别标注「文档值」外，本档数字都是我用 Python 只读脚本从盘上数据算的。
路径缩写如下：
- `KB2` = `.research_tmp/experiments/benchmarks/cs2/base_kb_v2/`
- `KB1` = `.research_tmp/experiments/benchmarks/cs2/base_kb/`
- `LB` = `legacy/benchmarks/cs2/`
- `CS` = `src/compilescholar/`
- `KC` = `src/kb_compiler/`
- `DD` = `.research_tmp/docs_decisions/`

## 0. 一页结论

1. 答题 KB 有 1,262 篇论文、29,007 条记录，是三套互不相同的抽取拼起来的：
   - 粗抽取：只读摘要，1,105 篇；
   - 深抽取：全文，21 篇 hub 加上混进 bulk 的 4 篇；
   - 综述专用抽取：136 篇综述。

   没有统一 schema。三套的 kind 集合、关系词表、实体载体各不相同（§A）。
2. 跨论文信息约 87% 是"综述作者说的"：survey 层占 18,748/29,007 条记录。系统自己从多篇原文合出来的跨论文对象，在答题 KB 里一个都没有：
   - 族 = 综述 taxonomy 节点名经嵌入合并，989 个，其中 933 个只来自一篇综述；
   - 族的 4,164 条证据 100% 是 KB 记录原文复用（record_id 与 quote 全部一致）。
3. 论文、方法、实体三套身份空间互不相连：
   - registry 的 15,100 个实体中 `in_corpus_paper_id` 填了的是 0 个；
   - 10,099 个 `*_ref` 槽位里只有 1 个带 paper_id，而且那篇不在 KB 里；
   - 谱系 2,422 条，端点落到论文的为 0。

   新骨架（`data/state/`）把综述书目解析到论文 id 的比例做到 79%，但只有 95 条落在 KB 内的 34 篇上。谱系落地（P0-4）没写。骨架数据还没有被任何消费者读取。
4. 领域地图（非线性演化）目前不存在。KB 里的谱系边按 entity_id 计有 1,998 个不同的端点对，其中得到 ≥2 篇不同论文佐证的只有 6 对。端点大量是泛词，例如 lstm、large language model、cnn。
5. 答题管线实际消费的只有 records 全文检索，加上把 state_merged 作为"带族名前缀的同一批 quote"。v9b test 第一轮的引用里：
   - 外检 ext 占 61.9%（2,095/3,386）；
   - KB 加 state 占 28.8%；
   - views、实体 id、谱系结构、absence 类型、骨架数据都不被消费。
6. 历史上的生长和补全机制都没有沉淀进答题 KB：
   - backflow、growth_demo、growth_marathon（实测 1/24 个缺口被解决）都只留在副本里；
   - 外检和 refgraph 只在答题期临时使用，不回写；
   - 答题期 admit 进来的 3 篇论文、加上 1 篇运行期深读论文，混进了 KB v2 的 bulk 层，共 446 条深读记录。
7. 时间维几乎为空：
   - 论文只有年份，没有日期；
   - 谱系、族、事实都没有日期；
   - 只有 domain_snapshot 带 `as_of`，取的是综述年份；
   - 唯一真正生效的时间机制是按年粒度的知识截止过滤。
8. 超图时代（n-ary 超图、自演化 schema、富拓扑、演化 intent 边、前端图浏览器）只存在于 `archive/` 和 `.research_tmp/runs/` 下。对当前 KB 和新包，代码和数据的承接都是零。

---

## A. 单篇抽取

### A.1 产出过 KB 内容的抽取器

| 抽取器 | 代码 / prompt 位置 | 输入 | 记录类型与字段 | 跑在哪些论文上（KB2 内实测） | 模型 | 质量（我算的 quote 逐字率*） |
|---|---|---|---|---|---|---|
| 粗抽取 Tier-1 | `KC/records/coarse_extract.py:47-69`（prompt），`:94-104`（记录形状）；新包副本 `CS/compile/state/coarse.py` | 标题加摘要（≤6,000 字符；短于 150 字符跳过） | kind ∈ {finding, method, limitation}；字段 subject、claim、claim_type（仅 finding）、quote（摘要原句）、mentions[]，provenance=coarse，id 前缀 `coarse:` | bulk 层 1,102 篇，5,073 条：finding 2,247 / method 2,106 / limitation 720（limitation 出现在 583 篇）；每篇中位 5 条 | `local:Qwen3.8-27B`（`KB1/ledger_coarse.jsonl` 4,770 次调用全是该模型） | 对摘要：字母数字归一后 99.7%；按空白和大小写的严格口径 99.4%（29 条未命中，分布在 23 篇） |
| StageB 深抽取（槽位遍 + absence 遍） | `KC/records/deep_extract.py:47-78`（KIND_SLICES），`:79-89`（quote-first 纪律），`:91-130`（CHUNK / ABSENCE prompt）；冻结 schema 见 `KC/records/schema.py` v1.4 | 全文 markdown，按节切块；注入 registry 和 vocab | result{method_ref, measure{metric, value, unit, direction, aggregation, timepoint}, role, delta, dims}；config{item, value, applicability, role}；lineage{from_method_ref, relation, to_method_ref, scope, evidence_basis}；finding{claim, scope_ref, target_ref, condition, strength, claim_type}；absence{subject, missing, absence_type, evidence}；shift；notation | hub 21 篇 4,740 条（finding 1,892 / result 1,443 / config 1,146 / absence 136 / lineage 112 / notation 10 / shift 1）；另有 bulk 层 4 篇 446 条：`arxiv_2101.01910` 和 3 篇 `ext_*`，都是运行期深读产物，见 §E | hub 卡片记录为 Qwen3.8-27B（`KB1/hub_cards.json`，21/21）；docstring 默认值是 DeepSeek-V4-Flash | hub 对全文 95.9%（finding 97.6、result 97.0、lineage 98.2、config 91.9、absence 92.5、notation 70）；bulk 4 篇对各自 deep_read_texts 为 98.2%（429/437） |
| 综述专用 S1–S4 | `KC/records/survey_extract.py:47-93`（四个模板），`:95-106`（纪律：转述不进事实层、方法名不得是作者引用）；适配层 `LB/base_kb_build/survey_views_adapter.py:24-79` | 综述全文；先过语义分节卡（taxonomy / comparison / chronology / challenges / method_entry / other），再按标签路由允许的 kind | survey_claim{claims_about, claim, conditions, semantic_label}；domain_snapshot{subject, snapshot_type ∈ taxonomy_node / comparison_table / timeline, claims[{claim, claims_about, conditions}], as_of}；survey_lineage 被适配成 lineage（端点 entity_id 置 None，`adapter:38-43`；evidence_basis=survey_claim）；survey_gap 被适配成 absence（absence_type=survey_claimed，gap_type ∈ open_problem / stated_future_work / noted_deficiency） | 136 篇综述，18,748 条：claim 12,712 / lineage 2,294（分布在 120 篇）/ absence 2,162 / snapshot 1,580（taxonomy_node 1,215、comparison_table 323、timeline 42） | 盘上未找到模型记录（未核实） | lineage 98.3%、absence 98.3%、survey_claim 93.2%、domain_snapshot 87.0%（多为表格行和跨句） |
| P0-3 提出者（新） | `CS/compile/skeleton/proposes.py:23-38`（prompt），`:58-71`（确定性校验：证据句逐字在摘要中、命中提出句式、排除依赖句式），`:49-51`（GENERIC 名单） | 摘要 | 独立文件 `data/state/methods.jsonl`：{paper_id, method, aliases, evidence, generic} | 已处理 1,126 篇非综述论文；657 篇有提出记录，共 850 条（hub 12 篇、bulk 645 篇）；generic 4 条 | local Qwen3.8-27B，temp 0（`experiments/skeleton/p0_3_proposes.py:2`） | 证据逐字由构造保证；E2 精度和召回闸门未做 |
| field_state 编译器内置的逐篇抽取 | `CS/compile/state/field_state.py:44-49`，复用 coarse prompt | 留出综述的引用论文摘要（S2 来源） | 同粗抽取 | 不在 KB 内：18 篇留出综述的引用集，`KB2/heldout_refs/` 共 2,957 条，与 KB 标题重合仅 29 篇；每次运行约 9,740 条记录 | local Qwen3.8-27B | 未单独核 |
| 超图抽取器（8 月） | `archive/legacy/granular_agent/hypergraph_extractor.py` 等（子代理核） | 颗粒流和 few-shot NLP 论文全文 | InstanceHypergraph（HGNode / Hyperedge{pattern_type, node_ids, node_roles, qualifiers, evidence_span}） | 不在 KB 内（§C.4） | DeepSeek / Paratera 一类（文档值） | — |

\*逐字率口径：小写、只保留字母数字 token 后做子串匹配；quote 含 "…" 时，每段长度大于 10 的片段都必须命中。口径偏宽，属于上界。

更正一条交接里的数字。handoff-1004 说"粗抽取 8.5% quote 不在摘要"，但越源的并不是粗抽记录。真正原因是 bulk 层混入了 4 篇论文的全文深读记录（446 条，其中 437 条带 quote），拿它们去对摘要当然对不上；如果对各自的全文，命中率是 98.2%（429/437）。`use_cached False` 这条出自 `arxiv_2101.01910` 的 config 记录。粗抽记录本身对摘要的逐字率是 99.4–99.7%。

### A.2 schema 统一性

**没有统一 schema。** 实际在用的至少有五套：
1. StageB v1.4，`schema.py:34-35`：RECORD_KINDS 为 result / config / lineage / finding / absence / shift / notation。
2. 粗抽取：finding / method / limitation。其中 method 和 limitation 不在 RECORD_KINDS 里。
3. 综述 S1–S4，适配后变成 lineage / absence / survey_claim / domain_snapshot。absence_type 用的是 `survey_claimed`，不在 `schema.py:59` 的 ABSENCE_TYPES 里。
4. P0-3 的 methods.jsonl，不在 records 里。
5. field_state 的输出：families / properties / limitations / problems 加 support，只落在 `KB2/heldout_eval/*.state*.json`。

关系词表互相冲突：
- `schema.py:45-47` 的 9 个词：extends, improves, uses, component_of, replaces, compares_with, motivated_by, generalizes, concurrent_with。
- `survey_extract.py:69` 的 6 个词：extends, improves, replaces, uses, compares, combines。

KB2 实测出现了 13 种关系标签，见 §C.2。

**各类信息的捕获情况**（KB2 实测）：
- **贡献 / 提出**：records 里没有"本文提出"字段。粗抽 method 记录不区分提出和使用：2,106 条中只有 884 条（42.0%）的 quote 带 we / our / this paper … propose / introduce / present / develop / design 句式。P0-3 另立文件，657/1,126 篇有提出记录。
- **方法**：粗抽 method 2,106 条，只有 subject 加一句 claim，没有机制、组件、输入输出结构。hub 深抽通过 config 和 finding 提供细节。
- **结果**：result 1,510 条，只来自 20 篇论文（hub 18 篇加 bulk 2 篇）。metric 表面形式有 222 种；dims.subject 有 286 种，其中 197 条为空。dims 里没有 dataset 槽位，见 `schema.py:85-94`，所以跨论文的结果单元格组不起来。
- **局限**：粗抽 limitation 720 条（583 篇），hub absence 136 条，综述缺口 2,162 条，finding 中 criticism 200 条。
- **关系**：lineage 2,422 条，其中 2,294 条是综述断言，见 §C.2。粗抽 mentions 后来被转成 entity_refs，只是共现，不带关系。
- **综述层没有摘要**：papers.json 里 abstract 非空的只有 bulk 1,105 篇和 hub 21 篇，136 篇综述的 abstract 全为 None。

缺口：抽取层只有 1.7%（21/1,262）的论文覆盖了全文。"提出 vs 使用"这层角色信息只在 P0-3 侧链里有。结果层不足以支撑任何跨论文比较。

---

## B. 身份与归一

### B.1 论文身份
- 主键用 `arxiv_<id>`（1,238 篇）、`hub_W…`（21 篇，OpenAlex id）、`ext_<标题 slug>`（3 篇）。1,259/1,262 篇有 arxiv_id，只有 3 篇有 DOI。年份只有整数 year，没有日期（§F）。
- 层的划分由 `LB/base_kb_build/build_kb_v2.py:66-76` 决定，依据是 survey / hub manifest 和 primary_category。`ext_*` 因为 primary_category 为空，被归进了 bulk_category，见 §E 的污染一条。

### B.2 实体身份（方法、数据集、任务、指标）
- 实体表是 `KB1/registry_v2.json`，15,100 个实体，provenance 全部是 `round2_growth`。它由 `KC/records/registry_growth.py` 用 27B 一次性把记录表面名映射成 match 或 new，结果 matched 0 / new 15,100，dup_guard_folds 4,051。类型分布：method 9,357 / out_of_corpus 3,574 / mechanism 1,399 / practice 770。**没有 dataset、metric、task 类型**（`schema.py:80` ENTITY_TYPES）。
- 绑定由 `LB/normalize_entities.py` 确定性完成，零 LLM，分三段：
  - A：*_ref 槽位查 surface_index；
  - B：mentions，4,174 次；
  - C：文本 n-gram 扫描，21,359 次，限制为 ≥4 字符、DF ≤5%（`:45-46`）、每条记录最多 6 个。

  KB2 共 34,604 个 entity_ref，引用到 10,301 个实体：
  - 只出现在 1 篇论文的 8,464 个（82%）；
  - 出现在 2–4 篇的 1,314 个；
  - 出现在 ≥20 篇的 111 个。
- 被最多论文共享的实体以泛词和误绑为主，比如 deep learning 101、large language model 94、strategies 76、benchmark 70、loss 67、study 65、step 59、score 55、`LEADING` 53、`WILL` 47、`issue tracker labels` 47。其中 `LEADING` 是把普通词 "leading" 绑成了实体，`issue tracker labels` 是把 "issues" 误绑。
- 其他失败样例：
  - **引用标记被当成实体**：实体 `ref [ 227 ]` 有 29 个别名，都是不同编号的 "Ref. [201]"、"Ref. [206]" 等。
  - **过合并**：`large language model` 的别名里有 "pre-trained language models"、"Transformer-based Language Models"、"Large-scale models"；`deep learning` 吞掉了 "Deep neural networks"；`self-supervised learning` 吞掉了 "existing surveys in the field of self-supervised graph learning"。
  - **欠合并**：只做小写、去标点和复数的平凡归一，就能再找到 43 组碰撞，例如 vit/ViTs、GRU/GRUs、MS MARCO/MS-MARCO、VAE/VAEs。
- **实体和论文没有连接**：`in_corpus_paper_id` 0/15,100，`origin_year_cited` 0/15,100。用这两个字段的设计（`registry.py:5-15`）在 CS2 KB 上从未生效。
- 方法族的身份见 §C.1：族名嵌入余弦 ≥0.86 且共享内容词即合并（`LB/base_kb_build/merge_state_v2.py:28-29`）。merge_audit 样例里能看到错并，例如 "loss functions of GNN-based SocialRS" 并进了 "decoders of GNN-based SocialRS"，"unsupervised learning algorithms in SNN" 并进了 "Supervised learning algorithms for SNN"。
- 新骨架的方法身份：methods.jsonl 有 850 条，用名字加别名，没有和 registry 或 entity_id 对接。谱系端点名能精确匹配某个提出方法名的，一端匹配 90/2,422，两端都匹配 2/2,422。

缺口：论文、实体、方法三套主键之间没有任何外键。数据集和指标没有身份。实体表里混着泛词、普通词和引用标记。

---

## C. 跨论文组织与编译

### C.1 现存的跨论文对象（逐一）

| 对象 | 位置 / 构建 | 规模（实测） | 端点能否落到论文 | 成员 / 支持集 / 时间 |
|---|---|---|---|---|
| 方法族 `state_merged.json`（答题在用） | `LB/base_kb_build/build_state_v2.py:44-56`：按规范化名把 taxonomy_node 分组；`:58-109`：按名字挂载 challenges、absence、comparison、lineage；然后 `merge_state_v2.py` 做嵌入合并和挂载（T_MERGE 0.86 / T_ATTACH 0.80） | 989 族；n_sources=1 的 933 族，=2 的 45，≥3 的 11；props 2,424 / limits 1,211 / compares 529；未挂载的 limits 1,850、compares 2,285 | 族的 `members` 字段是规范化名字，平均 1.19 个，不是论文 | 证据 4,164 条全部是 KB 记录（record_id 和 quote 4,164/4,164 一致）；来源层：survey 4,135 / hub 27 / bulk 2；只有 24 族含非综述证据；722 族的证据只来自 1 篇论文；无时间 |
| 未挂载项 | 同上，`state_merged.unattached` | limits 1,850 / compares 2,285 | — | 答题端不读 |
| lineage 记录 | records（kind=lineage） | 2,422 条，来自 144 篇论文 | 10,099 个 ref 槽中 paper_id 1 个（不在 KB）；端点是引文串（"Wang et al. (2018)" 等）的 336 条 | 无日期；evidence_basis 见 §C.2 |
| 谱系视图 `views_cs2.genealogy`（不在 KB2） | `KC/views/compiler.py`；`KB1/views_cs2.json` | 节点 15,100，边 2,406，ancestor_closure 1,600；边的 year 2,406/2,406 是 `backfill_years.py` 回填的；stats 里 nodes_with_year=0 | 节点是 entity_id，不是论文 | 只有 legacy KBTools 消费 |
| 比较矩阵 `views_cs2.matrix`（不在 KB2） | 同上 | 534 张表，涉及 18 篇论文；含 ≥2 篇论文的表只有 3 张（531 张是单篇） | — | `pair_deltas` 324 条，全是单篇内部推导 |
| coverage / absence 视图 | 同上 | absences_extracted 2,298，absences_derived 0，coverage families 0 | — | — |
| absence / 缺口记录 | records | 2,318 条：survey_claimed 2,162 / explicitly_stated 80 / not_reported 74 / cannot_tell 2 | 无 | `resolved_by` 15 条（09-30 规则配对加 LLM 核；文档称 9/15 是同篇自解）；没有 gap_status 层；`missing_translated` 275 条是布尔值 |
| survey_claim | records | 12,712 条；claims_about 去重后 7,893 种；在 ≥2 篇综述出现的 354 种；带 [n] 的 436 条，带 et al. 的 1,551 条 | `about_paper_id` 只有 22 条，其中 1 条在 KB | `paraphrase_rel`（转述四分类）只有 1 条 |
| domain_snapshot | records | 1,580 条（taxonomy_node 1,215 / comparison_table 323 / timeline 42），claims 4,920 条 | 无 | `as_of` 1,580/1,580，取综述年份 |
| 共识视图 `views_consensus.json`（不在 KB2） | `survey_views_adapter.py:82-112`：按 claims_about 字符串分桶 | 11,620 个"实体"桶 | 无 | 未被消费 |
| backflow 边（不在 KB2） | `KC/records/backflow.py`；`KB1/backflow_edges.jsonl` | 74 个外部论文节点，436 条边，全是 `external_mention` / coarse | 外部节点 `ext:` 不对应 KB 论文 | 见 §E |
| 引用桥 `cite_bridge_ledger.jsonl` | `LB/resolve_citations.py` | 23 条；第一条就把 "Papernot et al. [21]" 落到一个叫 "security and privacy" 的条目，是错的 | — | 已被骨架取代 |
| **骨架 CITES**（新） | `CS/compile/skeleton/bib.py`（书目切分加标记）+ `resolve.py:4-10,82-106`（显式 id → OAI 快照 → KB 标题 → OpenAlex → Crossref） | bib_entries 21,153 条，来自 **130** 篇综述（文档写的是 135 篇有参考文献节）：数字式 13,115 / 作者-年份式 7,894 / 裸数字式 144；被记录 quote 消费的 9,287 条；`cites.jsonl` 9,287 行，解析率 79.0%（oai_snapshot 3,172 / openalex 2,387 / explicit_arxiv 1,138 / explicit_doi 633 / kb 9 / stub 1,948） | 落到论文 id 的 7,339 条：arXiv 4,253 / DOI 2,796 / OpenAlex 195 / **KB 95，对应 34 篇**（hub 20、survey 7、bulk 7）；被消费的不同论文 6,648 篇，被 ≥2 篇综述引用的 385 篇，≥3 篇的 120 篇 | 年份来自书目条目（328 条为空）；E1 精度 200/200（两个模型标注，`results/e1/`）；数字式覆盖 79.7%，没过 85% 的闸门（文档值） |
| **骨架 PROPOSES**（新） | `proposes.py` | 850 条 / 657 篇 | 方法 → KB 论文 | 未与 CITES 或谱系连接；被 KB 内被引论文命中的只有 17 篇 |
| **骨架 LINEAGE 落地**（P0-4） | 设计在 `DD/upgrade-1004/DESIGN-W2.md:21`；`skeleton/lineage.py` **不存在** | `data/state/` 下没有 lineage.jsonl / markers.jsonl / papers_stub.jsonl | 潜力：2,294 条综述谱系里 1,060 条 quote 带数字标记，881 条至少有 1 个标记已解析到论文 id，其中只有 16 条落到 KB 论文。注意"标记已解析"不等于"端点已对齐" | — |
| field_state 族 / 事实 / 问题（新，**不在答题 KB**） | `CS/compile/state/field_state.py`：族归纳 `:85-134`（80 篇一批提出族，再合并）、族级事实 `:164-196`（支持集逐字、n_papers）、领域问题 `:217-247` | 18 篇留出综述各跑两次，合计：族 362，性质 2,513，局限 1,444，领域问题 885 | 成员 = 论文 key（真成员） | 性质 n_papers ≥2 的占 52.9%（1,329/2,513）；局限只有 12.0%（173/1,444）；问题 38.0%（336/885）；无时间 |

另外两点：
- `build_state_v2.py:21-22` 的关系归一表 REL 里没有 `uses`。KB2 中 uses 关系 1,198 条，占谱系的 49%，在进入族的 compares 时被整体丢掉。
- 20 篇留出金标综述仍在 KB2 里：2,360 条记录，state_merged 中有 259 个族引用了它们。目前的留出检验不读 KB，所以检验本身没有泄漏；但只要把编译器跑到全库，就必须先剔除这 20 篇（`DESIGN-CROSSPAPER.md:51-54` 已列为硬约束）。

### C.2 "演化关系"词表沿革与数据实际内容

| 阶段 | 词表 | 位置 |
|---|---|---|
| 8 月超图 evolution 族 | extends / improves / compares / replaces / adapts / background（角色 from/to，带 semantic_boundary） | `archive/legacy/granular_agent/hypergraph_schema.py` 物理 seed L1100-1160、通用 seed L1232-1283；`citation_intent.py:145` INTENT_TYPES 是同一组 |
| 8 月富拓扑类型 | evolution / law_parameter / composition / definition / nary / method_parameter / method_phenomenon / method_regime / other | `hypergraph_evolution.py:1809`（infer_rich_topology_direct），`:1855-1887` |
| 8 月 meta-edge | subclass_of / type_relation / pattern_dependency（细分 depends_on / constrains / composes） | `hypergraph_schema.py:271`，`hypergraph_evolution.py:1343/1421/1463` |
| StageB 冻结 schema v1.1+ | extends, improves, uses, component_of, replaces, compares_with, motivated_by, generalizes, concurrent_with；evidence_basis ∈ explicit_claim / citation_context | `KC/records/schema.py:45-47, 69` |
| 综述 S1 | extends / improves / replaces / uses / compares / combines | `KC/records/survey_extract.py:69` |
| backflow | external_mention（弱边） | `KC/records/backflow.py:1-27` |
| lineage_walk | 后继方向 extends / improves / replaces / generalizes；前驱方向 motivated_by / component_of；另外认 external_mention | `src/retrieval/lineage_walk.py:34-35, 153` |
| build_state_v2 归一 | compares_with→compares；保留 extends / improves / replaces / combines；**丢弃 uses** | `LB/base_kb_build/build_state_v2.py:21-22` |
| 转述四分类 | restate / extend / qualify / dispute | `LB/classify_paraphrases.py:1-13` |
| upgrade-1004 设计 | LINEAGE ∈ {extends, improves, replaces, combines, uses, compares} 加 PROPOSES / CITES / MEMBER_OF / SUPPORTS / CONTRADICTS / QUALIFIES / REPORTS_RESULT；状态取 consensus / contested / qualified / single-source / open | `DD/upgrade-1004/DESIGN-CROSSPAPER.md:96-105, 223-244` |

**KB2 实际数据**（2,422 条 lineage）：
- 关系分布：uses 1,198、extends 469、improves 364、compares 165、combines 106、replaces 66、compares_with 35、motivated_by 10、concurrent_with 3、component_of 3、formalized_by / proposed_by / proposes 各 1。共 13 种标签，没有统一归一。
- evidence_basis：survey_claim 2,294 / explicit_claim 115 / citation_context 13。
- 两端都有 entity_id 的 2,030 条，其中自环（from = to）66 条。
- 不同 entity 对有 1,998 个，涉及 2,801 个节点。被 ≥2 条记录支持的对有 24 个，被 ≥2 篇不同论文佐证的只有 **6** 个。
- 端点名去重后 3,388 个，频次最高的是 lstm 37、large language model 33、mixup 32、cnn 26、transformer 21、rnn 19。

可见这是一张以泛方法为枢纽、几乎没有重复佐证的稀疏星形图，算不上演化网络。演化里的分支、合并、替代这类非线性结构，在数据里没有被单独表示：combines 只有 106 条，而且端点是名字。

### C.3 比较 / 结果
- 跨论文结果矩阵实质上为零：3/534 张表含 ≥2 篇论文。
- 综述比较表是现成的"多篇并排"材料（comparison_table 323 条），目前只当文本片段使用。
- field_state 在设计上不编译跨族比较关系（`field_state.py:19`）。

### C.4 超图时代资产（子代理核，数字为它实算）
- **代码**：`archive/legacy/granular_agent/`，20 个模块、11,880 行，README 标注 2026-09-05 起 FROZEN。比赛副本在 `archive/contest_2026-08/dist_build/ScholarGraph/`。
- **数据**：
  - `archive/pre_stageB_root_2026-09-16/data/graphs/llm_domain/kb.json`（16.4 MB）：50 篇 few-shot NLP 论文；3,827 个 concept；3,912 条超边，其中元数 ≥3 的 376 条；tbox 有 9 个 meta_node、218 个 pattern，split_from 为 0。
  - `.research_tmp/runs/kernel_v2/`（608 MB）：358 个 run，81 篇颗粒流论文。
  - `.research_tmp/runs/ARFM2024/`：11 个 instance，994 条超边。
- **承接**：没有任何一项喂给 KB2 或新包。新包对超图的唯一渊源是 `CS/llm/client.py`，它通过 kb_infra 间接源自 granular_agent 的 llm_client。`tests/test_kb_compiler_import_gate.py:35` 明文禁止导入 granular_agent。

缺口：跨论文层没有"系统自己编译、带成员和支持集"的对象进入答题 KB。演化关系散在至少 6 套词表里，数据里的关系没有落到论文，也基本没有重复佐证。

---

## D. 渲染与消费

| 视图 / 产物 | 谁消费 | 现状 |
|---|---|---|
| `KB2/records.json` + `record_vecs.f32`（29,007 × 4096，qwen3-embedding-8b） | `CS/kb/index.py` HybridIndex（BM25 + 向量，RRF，每篇最多 2 条）← `CS/kb/store.py:99-115` `KB.search` ← `CS/answer/pipeline.py:189-230` gather | 文本检索。`record_text`（`index.py:27-35`）只拼 subject、claim、quote、方法表面名、relation、conditions、标题；entity_id、kind 结构、absence_type 都不参与 |
| `KB2/state_merged.json` | `store.py:41-85` state_search：查询嵌入对族名（取前 3 个名字）求余弦，阈值 0.55，展开 property / limitation / comparison 原文，snippet 前加 `[property of 'X']` | 证据和 KB 记录一致（§C.1）。在 `_interleave`（`pipeline.py:273`）的来源轮转里 state 排在最后。规划（`plan`，`pipeline.py:166`）只看 probe 的检索结果，不看族 |
| `views_cs2.json`（matrix / genealogy / coverage / cards / narrative） | 只有 legacy：`KC/views/tools.py`（KBTools：lineage / compare / find_gap / card / as_of / ppr），供 `legacy/benchmarks/_shared/tools/evidence_gate2r_harness.py` 和 growth 脚本使用 | 新包不读。v9b 答题路径不读 |
| `data/state/*`（骨架） | 只有 `experiments/skeleton/*.py` 和 tests | 不被 KB 或答题读取 |
| field_state 输出 | `CS/eval/field.py`（留出综述检验），输入是 `KB2/heldout_refs/` | 不进答题 |
| 前端 | `archive/pre_stageB_root_2026-09-16/demo/{app.py, index.html}`：FastAPI，vis-network 画超图 ego 网络，n-ary 边画成 ◆，cites 按 intent 着色 | 只吃 `kb.json{tbox, abox}`，和 KB2 格式没有适配层；仓库根目录的副本已无法运行（src/granular_agent 已删） |
| 卡片 | `KB1/hub_cards.json`（21 篇结构通读卡）；`views_cs2.cards` 3,303 张实体卡 | 只有 legacy 消费 |

**实际消费量**：v9b test，`arm_vnext/answers_test100_r1.json`。
- 检索到的证据：kb 7,104 / state 4,586 / ext 6,703 / cite 1,222。
- 写作引用的证据：ext 2,095（61.9%）/ kb 659 / state 315 / cite 317。r2 基本相同。
- 被引 KB 记录按 kind 分：survey_claim 238、finding 165、absence 64、method 61、domain_snapshot 44、lineage 34、limitation 31、config 16、result 6。
- 被引 KB 和 state 证据按层分：survey 691（71%）/ bulk 170 / hub 113。71 题至少引用了 1 条 state。
- 消融结论来自文档值：state 通道 Δ≈0（`DESIGN-CROSSPAPER.md:44`）。§C.1 的 4,164/4,164 原文复用给了它一个更强的解释：state 通道里没有任何检索不到的新信息。

**领域层直接检验**：`CS/eval/field.py`，18 篇综述。我按 `heldout_eval/*.scores.json` 重算的 recall 均值如下：

| | state | state_v2 | direct | flat | memory |
|---|---|---|---|---|---|
| 族 | .630 | .542 | .465 | .596（recall@K .163） | .317 |
| 性质 | .110 | .093 | .077 | .049 | .065 |
| 局限 | .221 | .276 | .209 | .256 | .112 |

state 与 state_v2 的均值减去 direct：族的 recall 差 +0.121，与文档一致。性质的 recall 差只有 +0.025；文档报的 +0.035 对应的是 recall@K 的差（+0.036）。**同一张表里，族用的是 recall，性质用的是 recall@K，口径不一，需要注明。**
绝对水平上，系统只覆盖了综述作者约 10% 的族级性质陈述。scores 文件是否已经按 W1-4 修订后的协议重判，我未核实。

缺口：所有结构化视图要么没人消费，要么只被 legacy 消费。答题端把 KB 当成一个带标签的片段库来用。没有面向人的领域地图视图。

---

## E. 生长、外部检索与地图补全

| 机制 | 做了什么 | 实测结果 | 去向 / 是否进 KB2 |
|---|---|---|---|
| bulk 层选取 | `LB/base_kb_build/snapshot_fill.py:9-11`：从本地 arXiv OAI 快照里，按 6 个类别 × 2021–2025-04 分层随机抽 184 篇 | 1,105 篇；论文年份 2021–2024 共 1,216/1,262 篇 | 进了 KB2。按题目盲原则是合规的，但它是**随机抽样，不构成"领域"**，与领域地图的前提不符 |
| 综述池 / hub 池 | `survey_pool.py`：OpenAlex 取 type=review、按概念 × 年代高被引；`hub_consensus.py`：按 referenced_works 跨综述计数 | 136 篇综述；hub 21 篇（共识统计里被 ≥3 篇综述引用的 94 篇，`KB1/consensus_stats.json`） | 进了 KB2 |
| demand_core | 按 CS2 dev 题面逐题用 Sciverse 检索 top-10，再粗抽 | 793 篇 / 2,589 条，外加深读 46 篇 / 5,686 条 | 用户判定为作弊；KB2 构建时已剔除（`build_report.json`） |
| backflow（mentions 回流） | `KC/records/backflow.py`：外部论文粗抽 → mentions 匹配 registry → `external_mention` 弱边 | 74 个节点，436 条边；文档称深回流在 CS2 上实际 0 篇成功，补回流后 62/66 篇，关系升级抽检错 17/20 | 留在 `KB1/backflow_edges.jsonl`，**不是 KB2 的输入** |
| 运行期 admit / 深读 | `legacy/.../_shared/tools/external_tools.py:289`（admitted_from=external） | manifest_all 中 6 篇 admitted_from=external | **其中 3 篇 `ext_*` 和 1 篇运行期深读的 `arxiv_2101.01910` 仍在 KB2 的 bulk 层，共 446 条深读记录**（build_report 中 `deep:bulk_category` 4 篇 / 446 条）。这些是题目驱动产生的残留，规模小，但违反题目盲 |
| growth_demo | `LB/growth_demo.py`：缺口 → 检索词 → Sciverse → admit → 深读 → backflow | ledger 共 12 轮：resolved 2、ingested_not_resolving 6、no_relevant 3、admit_failed 1 | 只存在于 `cs2/demo_growth_kb/` |
| growth_marathon | `LB/growth_marathon.py` | `growth_marathon_ledger.json`：24 轮，ingested 23，**resolved 1/24**，新增 11 篇 / 477 条，17.8 分钟；矩阵表 1,170→1,189，跨论文表 0→1，桥实体 2,588→2,590 | 只存在于 `demo_growth_kb2/`；用户裁定"生长不押分数" |
| ext_candidates | 答题期外部候选落盘 | 14 行 | 只服务答题期 |
| Sciverse | `CS/sources/sciverse.py`：语义检索，服务端截止，30 req/min | 答题期主力：v9b 引用的 61.9% | 不回写 |
| refgraph 引文扩展 | `CS/sources/refgraph.py`：OpenAlex 批量 → S2 → Crossref；`pipeline.py:89-130` 按共引排序 | 缓存 `_shared/refgraph_cache` 7,488 个文件（320 MB），`s2_cache` 2,984 个；v9b−v8 +0.010，不显著（文档值） | 只存 HTTP 缓存，不回写 KB |
| OAI 快照标题索引 | `CS/sources/arxiv_snapshot.py`；`cache/arxiv_snapshot_title_index.tsv`（338 MB，文档称 245.5 万条） | 骨架 P0-2 主力：3,172 条命中 | 只给骨架用 |
| OpenAlex / Crossref 解析 | `resolve.py` 的网络阶段 | openalex 2,387 条命中；offline 版本 stub 4,315 条，online 版本 1,948 条 | 落在 cites.jsonl |
| 留出引用集 | `LB/base_kb_build/fetch_heldout_refs.py`：S2 references 加 arXiv 摘要 | 2,957 篇，与 KB 标题重合 29 篇 | 只给 field 检验 |
| `src/retrieval/`（sci-evo 收编） | 有 gap_search、lineage_walk、citation_graph、coarse_store（TierStore）、registry（SQLite）等 | — | 新包零 import，只被 legacy 脚本经 `src/sci_evo_extract.py` 垫片使用 |

结论：所有补全机制都是在"答题期间"或"副本 KB 上"运行的，没有一条进入答题 KB 的增量通路。
"缺口"的定义（综述缺口文本或 explicitly_stated absence）太宽，检索词又是缺口原文拼接（marathon ledger 第 1 轮的检索词就是 subject 和 missing 直接拼起来的），所以 22/24 轮都是"入库了但没解决"。
骨架给出了第一个可以算的"地图边界"：被消费的 6,648 篇被引论文中，KB 内只有 34 篇；KB 外被 ≥2 篇综述引用的有 385 篇。但目前没有任何机制去取这些论文。

---

## F. 时间

| 对象 | 时间字段 | 实测 |
|---|---|---|
| papers.json | `year`（int） | 1,262/1,262 有年份，**没有日期 / 月份**；分布：2013–2019 共 14 篇，2020 年 32 篇，2021–2024 年 1,216 篇 |
| domain_snapshot | `as_of` | 1,580/1,580，取综述年份（2020 / 2022 / 2024） |
| absence.resolved_by | `year`、`verified_at` | 15 条 |
| shift | `time_range` | 1 条（"Recently"） |
| lineage、族、族级事实、field_state 输出 | 无 | — |
| registry | `origin_year_cited` | 0/15,100 |
| views_cs2.genealogy 边 | `year`（回填） | 2,406/2,406。文档称一重编译就会丢；这份文件不在 KB2 |
| cites.jsonl | `year`（从书目条目启发式提取） | 9,287 条中 328 条为空 |
| 知识截止 | `CS/core/cutoff.py:38-61` | 只有年份时同年整年排除；KB 通道已接入（`store.py:87-97`，W1-12）。`DESIGN-CROSSPAPER.md:270` 说"KB 不按年份过滤"，已过时 |

**没有任何 as_of 切片查询**，新包里不存在。legacy KBTools 有 `as_of(year)`（`KC/views/tools.py:880`），但文档称调用次数为 0。
upgrade-1004 设计了 `status(fact, T)` 这种纯函数切片（`DESIGN-CROSSPAPER.md:107-113, 266-278`），属于 W4，尚未开工。

---

## G. 历史教训：跨论文 / 领域层试过的主要思路（文档值，详见各档）

| 思路 | 判决 | 原因（一句） | 出处 |
|---|---|---|---|
| 8 月 n-ary 超图 + 自演化 schema（split / lift / retire）+ 富拓扑 | 2026-09-05 冻结、归档 | split 的价值只体现在 schema 层，NMR 等无差别；被 StageB 记录层重写取代 | MEMORY「pilot 实测：split 价值只在 schema 层」；`archive/legacy/granular_agent/README.md` |
| 世界状态框架 v5（可比性分带、类型化未知、as_of、五个检验槽） | 被 v6 取代 | 五个槽里只有 CS2 真跑过；新颖性被 LKM / ScholarStack 挤压 | `DD/EXPERIMENT-DESIGN-WORLDSTATE-0926.md:33-110` |
| 跨论文比较矩阵 | 放弃通用版 | 18/2,099 篇有可比结果，跨论文表 3/534 | `DD/FIX-PLAN-P0-0929.md:70-75`；`DESIGN-CROSSPAPER.md:246-264` |
| 综述骨干 + survey_extract + 共识视图 | 保留为 KB 主体，但被判定为"族 = 综述目录搬运" | 跨论文判断都是综述作者的 | `DD/REBUILD-PLAN-1003.md`；handoff-1004 |
| state_v2 / merged 族视图进入答题 | 失败 | 消融 Δ≈0，状态证据 841/842 与 KB 逐字相同（本档实测 4,164/4,164）；用户否定"纯靠综述分族" | `DESIGN-CROSSPAPER.md:39, 44` |
| typed absence 三态 + gap_status 动态差集 | 暂停 | "认领"只存在于提示文案；absences_derived=0；分解层增益在噪声带内 | `DD/REBUILD-PLAN-1003.md:74`；`DD/RESEARCH-KB-ASSISTED-RETRIEVAL-0929.md:46-48` |
| 意图过滤的定向引文遍历 / PPR 谱系重排 | 放弃 | S2 intent 覆盖 0/55；lineage 回放 41/41 为空；31a 批级无增益 | `DD/NARRATIVE-GROWING-STATE-0929.md:48-66`；`DD/NARRATIVE-COVERAGE-1003.md:14` |
| 跨论文五件套（实体归一、引用桥、共享实体群、别名、转述四分类） | 无增益 | 引用桥 23 条且有错；桥实体前 50 个是泛词；四分类只产出 1 条；31b−31a −0.005 | `DD/FIX-PLAN-P0-0929.md:194-252`；`DD/REBUILD-PLAN-1003.md:46` |
| v7 覆盖估计（书目当捕获-再捕获） | 降为备选 | 门槛 A 过（一跳池召回 0.423 vs 0.209），门槛 B 未过（Pearson 0.13–0.39）；用户评价"只是增强召回" | `DD/NARRATIVE-COVERAGE-1003.md:199-253` |
| 缺失主张认证、实验网络、证据许可域、DSpace、类别枚举、主题聚类等试点 | 全部判死或打平 | 例如缺失认证每份报告 <0.3 条、判官精度 19%；DSpace −0.110 | `DD/NARRATIVE-COVERAGE-1003.md:35-44` |
| v8 field_state 编译器 | 采纳为主证据候选；效应小，未进 KB | 族 +0.121、性质 +0.035（口径见 §D），局限不显著 | `DD/PAPER-DRAFT-EXPERIMENTS-1003.md:140-158` |
| upgrade-1004 骨架（CITES / PROPOSES / LINEAGE 落地） | 进行中，暂停 | E1 数字式覆盖未过线；P0-4 未写；KB 内落地只有 95 条 | `DD/upgrade-1004/DESIGN-W2.md`；handoff-1004 |
| "没读过的论文的领域地图"（综述书目指挥外检） | 被用户否定，要求重新构想 | 只靠综述，分析不够透彻 | handoff-1004 |
| 生长循环（demo / marathon） | 不押分数；审查要求要么出可度量结果，要么从方法章删除 | 1/24 | `DD/upgrade-1004/AUDIT-VENUE-READINESS.md:189-191` |

---

## 断点清单

| # | 层 | 断点 / 缺失能力 | 证据 |
|---|---|---|---|
| 1 | 抽取 | 没有统一 schema：五套 kind 集合、两套关系词表并存 | `KC/records/schema.py:34-35,45-47,59`；`survey_extract.py:47,69`；`coarse_extract.py:55`；`proposes.py:13` |
| 2 | 抽取 | 98% 论文只读摘要：全文覆盖 21 hub + 4 篇 / 1,262 | papers.json 层计数；records 中 provenance 分布 |
| 3 | 抽取 | 粗抽 method 不区分提出和使用，"本文贡献"不是一等字段 | 2,106 条中 884 条（42%）带提出句式；`coarse_extract.py:65` |
| 4 | 抽取 | 结果层只来自 20 篇，dims 没有 dataset 槽 | result 1,510 条；`schema.py:85-94` |
| 5 | 抽取 → 跨论文 | P0-3 提出者未回写 records，也没被任何消费者读取 | `data/state/methods.jsonl` 850 条；新包 src 中没有对 data/state 的读取 |
| 6 | 身份 | 实体与论文没有外键 | registry `in_corpus_paper_id` 0/15,100；ref 槽 paper_id 1/10,099 |
| 7 | 身份 | 实体表混入泛词、普通词、引用标记 | `LEADING` 53 篇、`WILL` 47、`issue tracker labels` 47、`ref [ 227 ]` 29 个别名；8,464/10,301 个实体只出现在 1 篇 |
| 8 | 身份 | 过合并与欠合并并存，归一由一次性 LLM 映射加 DF 阈值决定 | LLM 别名合并样例；43 组平凡碰撞；`normalize_entities.py:45-46` |
| 9 | 身份 | 没有 dataset、metric、task 实体 | `schema.py:80`；metric 表面形式 222 种 |
| 10 | 身份 | 方法族靠名字嵌入合并，存在错并 | `merge_state_v2.py:28-29`；`merge_audit.json` 样例 |
| 11 | 跨论文 | 族没有成员论文；933/989 只来自一篇综述；722 族只有 1 篇论文的证据 | `KB2/state_merged.json` |
| 12 | 跨论文 | 状态层没有新信息：证据 100% 是 KB 原文 | 4,164/4,164 |
| 13 | 跨论文 | 族编译时丢弃 uses 关系（占谱系 49%） | `build_state_v2.py:21-22` |
| 14 | 跨论文 / 地图 | 谱系端点未落到论文；端点 336 条是引文串 | 2,422 条中 0 条落地 |
| 15 | 跨论文 / 地图 | 谱系图稀疏，几乎没有重复佐证，枢纽是泛词 | 1,998 个 entity 对中，被 ≥2 篇论文佐证的 6 个；端点频次前列是 lstm / LLM / cnn |
| 16 | 跨论文 / 地图 | 非线性演化（分支、合并、替代链、族内谱系）没有表示 | combines 106 条且只是名字；没有族—族、族—谱系关系 |
| 17 | 骨架 | P0-4 谱系落地未实现；markers、lineage、papers_stub 未产出 | `CS/compile/skeleton/` 下没有 lineage.py；`data/state/` 文件清单 |
| 18 | 骨架 | CITES 几乎全部落在 KB 外：KB 内只有 95 条、34 篇 | `data/state/cites.jsonl` |
| 19 | 骨架 | 书目只覆盖 130 篇综述（文档说 135）；数字式覆盖未过 85% 闸门 | bib_entries 的 survey 去重数；handoff-1004 E1 |
| 20 | 骨架 → 消费 | 骨架数据没有被 KB 加载器或答题读取 | `CS/kb/store.py`、`answer/pipeline.py` 中没有相关引用 |
| 21 | 聚合 | field_state 编译器不进答题 KB，只跑在留出引用集上（2,957 篇，与 KB 交集 29） | `CS/eval/field.py:27-29`；`KB2/heldout_refs` |
| 22 | 聚合 | 族级局限跨论文成立的比例低 | 173/1,444 = 12.0% |
| 23 | 聚合 | 没有共识、争议、限定状态；转述四分类只有 1 条 | KB2 中 `paraphrase_rel` 1 条 |
| 24 | 聚合 | 跨论文结果矩阵为零；323 张综述比较表只当文本 | `views_cs2.matrix` 3/534；domain_snapshot comparison_table 323 |
| 25 | 缺口 | 缺口只是文本：没有 gap_status，没有解决链（15 条 resolved_by） | absence 2,318 条；absences_derived 0 |
| 26 | 渲染 | 答题只用 records 文本和 state 片段，不用 entity_id、谱系、absence 类型、views | `CS/kb/index.py:27-35`；`store.py:41-115` |
| 27 | 渲染 | 规划不看状态；state 在证据轮转里排最后 | `pipeline.py:166-186, 273` |
| 28 | 渲染 | 没有可用的领域地图视图或前端：存档前端只认 kb.json 的 tbox / abox | `archive/pre_stageB_root_2026-09-16/demo/app.py` |
| 29 | 渲染 / 评测 | 领域层检验的族用 recall、性质用 recall@K，口径不一 | 本档 §D 重算 |
| 30 | 生长 | 外检和引文扩展结果不回写 KB，只在答题期使用 | `pipeline.py:189-230` 没有写操作；refgraph 只有 HTTP 缓存 |
| 31 | 生长 | 生长循环 1/24，产物只在副本里 | `growth_marathon_ledger.json`；`demo_growth_kb2/` |
| 32 | 生长 | backflow 边不是 KB2 的输入 | `KB2/build_report.json` 的 inputs_sha16 只有 5 个文件 |
| 33 | 生长 / 地图 | 库外高频被引论文（≥2 篇综述引用的 385 篇）没有桩节点，也没有获取机制 | `cites.jsonl` |
| 34 | 生长 | 缺口定义过宽，检索词是缺口原文拼接，22/24 轮入库后没有解决问题 | marathon ledger 第 1 轮的 queries |
| 35 | 语料 | bulk 是 arXiv 六类随机抽样，不构成"领域"；年份集中在 2021–2024 这 4 年 | `snapshot_fill.py:9-11`；年份分布 |
| 36 | KB 卫生 | 运行期 admit / 深读的 4 篇（446 条）以 bulk 身份留在 KB2 | build_report `deep:bulk_category` 4 篇 / 446 条；manifest_all `admitted_from=external` |
| 37 | KB 卫生 | 20 篇留出金标综述仍在 KB2（2,360 条，259 个族引用） | `KB2/survey_gold.json` ∩ papers.json |
| 38 | 时间 | 论文没有日期；谱系、族、事实没有时间；没有 as_of 切片 | papers.json；§F |
| 39 | 时间 | 截止按年粒度，同年整年排除 | `CS/core/cutoff.py:38-61` |
| 40 | 历史资产 | 超图、自演化 schema、富拓扑、intent 边、前端零承接；演化词表在 6 处各自定义 | `archive/legacy/granular_agent/`；§C.2 |

## 附：本档用到的数据文件（只读）
`KB2/{papers,records,state_merged,build_report,merge_audit,survey_gold}.json`、`KB2/heldout_eval/*.json`、`KB2/heldout_refs/*.json`；
`KB1/{registry_v2,registry_growth_report_v2,views_cs2,views_consensus,deep_read_records,records_survey,records_hub,manifest_all,hub_manifest,survey_manifest,consensus_stats,snapshot_fill_stats}.json`、`KB1/{backflow_edges,cite_bridge_ledger,ext_candidates,entity_norm_ledger,ledger_coarse}.jsonl`、`KB1/{survey,hub,deep_read}_texts/`；
`data/state/{bib_entries,cites,cites.offline,methods}.jsonl`；`cs2/{growth_marathon_ledger,growth_demo_ledger}.json`；`cs2/arm_vnext/answers_test100_r{1,2}.json`；`results/e1/e1_agreement.json`。
