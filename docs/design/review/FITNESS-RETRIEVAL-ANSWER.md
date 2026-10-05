# 检索 / 答题 / 工具 / 评测层适用性审查（FITNESS-RETRIEVAL-ANSWER，2026-10-05）

> 审查对象：旧检索（`kb/`、`kb_compiler/retrieve`、`kb_compiler/views/search_text`、`retrieval/{search_service,query_understanding,rerank,circuit}`）、新写的 `index/build.py`、`tools/*`、`cognition/*`、`answer/pipeline.py`、`baselines/harness/*`、`eval/*`。标准是 INTEGRATED-SYSTEM-1005 的新系统硬要求：工具带 as_of（按天，含当天）；论文级数百万、段落级千万行、表述级千万级；每次工具调用在亚秒到数秒内返回。
>
> 做法：逐个读代码；在 `%TEMP%\fitret_bench` 下用合成数据实测（结束已删除）；对 `data/dfc/{papers,citations}.sqlite` 只读打开计时；不改仓库代码，不调用 LLM，不访问网络。
>
> 合成数据说明：
> - 段落文本是按 20 万条真实引用句的词频分布独立抽取的随机词，日期按真实 arXiv v1_date 分布抽样；
> - 合成表述库（`extract.sqlite`）用的是真实 cites 表里的 id、日期和共引结构（每个被引对象最多 30 句，与 `N_OTHER=30` 一致），另加每篇 10 条 self 表述，共 1,120,223 条；文本是随机词。
> - 因此**延迟和内存是可信的，聚类、族的质量不可信**，族的质量不在本档结论之内。
>
> 机器：RAM 29.8 GB（测试时空闲 10.7 GB），16 逻辑核，Python 3.13.12，SQLite 3.50.4。

## 0. 结论先行

1. **段落 / 表述检索不能用 FTS5，换 tantivy。**
   - FTS5 在 `UNINDEXED` 列上做 `date <= ?`，不走索引：先对全部 MATCH 行算 bm25，再逐行读 content 过滤。
   - 100 万行时中位 1,066 ms；42 词的长查询（DSB 题面式）8.94 s；千万行按线性外推约 10 s。
   - 本机已装的 tantivy 0.26.0 测得：100 万行中位 11 ms，1,000 万行 41 ms；长查询在 100 万行上 78 ms；1,000 万行索引 2.5 GB，建索引 57 s。
2. **向量部分不能整表读进 numpy。**
   - 100 万 × 4096 维 float32 需要 16.4 GB；千万级表述需要 164 GB。
   - 每个 `claude -p` 会话会拉起一个 stdio MCP 进程，N 路并发就有 N 份内存拷贝。
   - 替代方案（真实 KB v2 向量实测）：符号位二值码 512 B/行，Hamming 粗召回 200 条后用 float 精排，recall@10 0.995、recall@50 0.997。100 万条二值码占 0.51 GB，扫描 75 万条 135 ms，按日期排序后 as_of 就是前缀切片。
   - 千万级需要 ANN：chromadb 已装（真实向量带日期过滤 recall@50 0.976，53 ms）；faiss / hnswlib / usearch 未装。
3. **cognition "现算" 在真实规模下不成立。**
   - 112 万条合成表述上，`Identity(view)` 10.4 s，`lineage.edges` 29.6 s，`paper_profile` 63.7 s，`frontier` 37.7 s，每次 RSS 增加约 1.9 GB。这些都要改成编译阶段物化（§4）。
   - `field_map` 在真实 citations.sqlite 上会跑约 3 小时：`cites.sentence_id` 没有索引，`co_cited` 每次全表扫描 152 ms，而 298 篇的 scope 要调 74,562 次。加一个索引（0.7 s 建成）后降到 1.9 s。
   - `_brief` 和 AsOf 新建连接**不是**瓶颈：每个候选 0.5 ms，每次新建连接 0.18 ms，因为有 30 句的上限兜底。但 `n_citing` 也因此被截在 ≤30（§8 第 6 条）。
4. **answer_lit 复用 v9b 的 screen / write / assemble 时有三处静默出错。**
   - snippet 被 `_t()` 截到 260–300 字符，加了 "…" 并压缩了空白，已经不是逐字原文；
   - lit_other 把引用方的句子挂在被引论文名下，CS2 要求 snippet "取自被引文献"；
   - `arxiv` 字段直接取 `id[6:]`，DOI 优先身份落地后会写进 PPR_ 号。
   - 另外，接入外检后同一篇论文会因 identity 不统一拿到两个编号（W1-13 回归）。
5. **截止语义有三个硬冲突。**
   - 外检按年（Sciverse `year ≤ cut−1`）、本地按天，DSB 的月粒度还会丢天；
   - 日期是混合精度的字符串比较（实测 `'2024' <= '2024-03-01'` 为真，会泄漏）；
   - 引用句标的是 v1 日期，但文本取自后续版本：0.52% 的已解析引用指向比引用方 v1 更新的论文，T=2023-06-30 时有 761 个对象提前可见。

## 1. 实测数字

### 1.1 段落检索（合成，平均 120 词/行，OR 查询，k=400）

| 实现 | 行数 | 建索引 | 大小 | as_of=2025-04-30（中位/最大） | as_of=2020-12-31 | 无过滤 |
|---|---|---|---|---|---|---|
| FTS5，`date` UNINDEXED 列过滤（现行写法） | 10 万 | 3.1 s | 144 MB | 108 / 137 ms | 95 / 125 ms | 38 / 55 ms |
| 同上 | 100 万 | 44.9 s + optimize 7.4 s | 1,400 MB | **1,066 / 1,391 ms** | 916 / 1,187 ms | 334 / 493 ms |
| FTS5，按日期顺序插入、`rowid <= ?` 过滤 | 100 万 | 同上 | 同上 | 277 / 404 ms | 137 / 196 ms | — |
| FTS5，AND 语义（作对照，语义不同） | 100 万 | — | — | 6 / 31 ms | — | — |
| **tantivy**，`day` 整数 fast field 做 range | 100 万 | 9.6 s | 251 MB | **11 / 15 ms** | 6 / 8 ms | 3 / 6 ms |
| **tantivy** | 1,000 万 | 56.7 s | 2,498 MB | **41 / 66 ms** | 31 / 44 ms | 37 / 58 ms |

- 查询计划：`SCAN passages_fts VIRTUAL TABLE INDEX 0:M5` + `USE TEMP B-TREE FOR ORDER BY`，date 不参与索引。rowid 版的计划是 `0:M5<`，rowid 约束下推到了 FTS5。
- OR 查询在 100 万行里中位命中 476,750 行（47.7%），慢就慢在这里：高频词让 bm25 对一半的库逐行打分。
- 42 词长查询（DSB 题面去停用词）：FTS5 date 列 8.94 s，rowid 1.42 s，tantivy 78 ms。

### 1.2 向量

| 实现 | 规模 | 内存 / 磁盘 | 查询 | 召回 |
|---|---|---|---|---|
| numpy 全量 float32（现行 `Index._dense`） | 10 万 × 4096 | 1.64 GB RAM | 36 ms | 精确 |
| 同上，外推 | 100 万 × 4096 | **16.4 GB RAM** | 约 360 ms | — |
| sqlite-vec 0.1.9 `float[1024]` + day 元数据列 | 10 万 | 415 MB | 534 ms | 暴力精确 |
| sqlite-vec `bit[4096]` | 10 万 | 55 MB | 103 ms | — |
| chromadb 1.5.9 HNSW + `where day<=` | 10 万 × 1024（随机向量） | 470 MB | 120–195 ms | 随机向量是 HNSW 最坏情况，不作数 |
| chromadb，真实 KB v2 向量 | 29,007 × 4096 | 建 15 s | 53 ms | 带过滤 recall@50 **0.976** |
| numpy 二值码（`bitwise_count`），as_of = 前缀 | 100 万 × 4096 bit | 0.51 GB | 75 万条前缀 Hamming top-1000 135 ms | — |
| 二值码 → top-200 → float 精排（真实 KB v2，300 条留一查询） | 29,007 | — | — | recall@10 0.995 / @50 0.997 |
| 截到 1024 维、不精排 | 同上 | — | — | recall@10 0.870 |
| int8 逐维标量量化 | 同上 | — | — | recall@10 0.880 |

已装 / 未装（py3.13；py3.11 环境里这些全没有）：
- 已装：sqlite-vec 0.1.9、tantivy 0.26.0、chromadb 1.5.9、qdrant-client 1.19.1（local 模式只能暴力检索，server 模式要 docker，本机 docker daemon 未运行）、numpy 2.4.6、torch 2.13 cpu；
- 未装：faiss、hnswlib、usearch、duckdb、lancedb、annoy、bm25s、rank_bm25。

### 1.3 真实库只读计时（data/dfc）

| 查询 | 计划 | 耗时 |
|---|---|---|
| `AsOf.co_cited`：`cites WHERE sentence_id=? AND date<=?` | **SCAN cites**（223 万行） | 150–154 ms/次 |
| 同上，在临时副本上加 `INDEX(sentence_id)`（建索引 0.7 s） | SEARCH | ≈0 ms |
| `cited_by`：`cites WHERE cited=? AND date<=?` | 走 `ix_cites_cited` | 头部论文 2103.00020（9,319 行）24.7 ms |
| `references`：`entries WHERE citing=?` | 走主键 | 6 ms/次 |
| `AsOf.paper`（104.6 万篇） | 走主键 | 0.037 ms/次 |
| 每次新建 2 个只读连接 | — | 0.18 ms |

库规模：
- papers 1,045,961；docs 28,450；entries 1,372,649；sentences 1,118,931；cites 2,233,249。
- cites 中 stub 675,106 行；不同被引对象 372,462 个，其中解析到论文的 134,026 个。

### 1.4 cognition 与工具（合成表述库，T=2025-04-30，接真实 papers / citations）

| 调用 | 耗时 | 备注 |
|---|---|---|
| `view.statements(kind="self")` / `(kind="other")` | 1.85 s / 7.08 s | 674,480 行 other 读进内存，RSS +1.78 GB |
| `Identity(view)` | **10.4 s** | 21,452 个名字 |
| `ident.aliases(p)` | 100 ms/次 | 每次遍历全部名字再逐个 resolve |
| `lineage.edges(view, ident)` | **29.6 s** | 237,716 条边 |
| `profile` / `_brief` | ≤1 ms / 8 个 4 ms | 有 30 句上限兜底 |
| `paper_profile` | **63.7 s**，RSS 峰值 +1.95 GB | Identity + 全量 lineage |
| `frontier` | **37.7 s** | `L.edges(view)` 自建 Identity |
| `field_map`（原库，无 sentence_id 索引） | 约 **193 min**（估算） | 298 篇 scope → 74,562 个引用句 × 155 ms |
| `field_map`（加索引后） | 1.9 s | 合成类别使族链成一个 297 篇的巨族，族质量不作数 |
| `open_issues` | 0.1 s | |
| `what_is_missing` | 382 ms / 139 ms（加索引后） | 每次 `set(Documents().ids())` |
| `baselines_for` | 7–9 ms | |
| `citations_of`（头部论文） | 52 ms | |
| `find_evidence`（假 index） | 74 ms | 不含检索本身 |
| 共引对表 `cocite(a,b,citing,date)` 物化 | 建表 19 s，1,649,169 行 | 298 篇 scope 的全部对支持 52 ms |

v9b `HybridIndex`（真实 KB v2，只用 BM25）：29,007 条记录、110 万 token，建索引 1.0 s，RSS +196 MB，查询 2–19 ms。纯 Python dict-of-Counter 约 180 B/token，千万条表述 × 40 token 约需 72 GB。

## 2. 判定总表

工作量：S ≤ 0.5 天，M 1–2 天，L 3–5 天。

| 模块 / 功能 | 判定 | 证据 | 具体改法 | 工作量 |
|---|---|---|---|---|
| `kb/index.py` HybridIndex（BM25 内存实现） | **冻结**（v9b 臂保留，不进新 index） | 全内存 dict-of-Counter（`kb/index.py:39-64`），约 180 B/token；KB v2 +196 MB | 原样留给冻结 v9b 路径（表征测试逐字节）；新 index 不复用代码 | — |
| ↳ RRF k=60 | 进新 index（逻辑） | `kb/index.py:121-125` | tantivy 与向量两路排名做 RRF，k=60 不变 | S |
| ↳ 每篇上限（per_paper） | 进新 index（逻辑），修 bug | 新 `index/build.py:178-186` 里只有向量命中的条目 `about` 为 None，**绕过了上限**；dense 路也不按 kind/facet 过滤 | 先从行键取 paper_id，再按 paper_id 截断；dense 路和 BM25 路用同一个过滤谓词 | S |
| ↳ 命中强弱（strong/weak/none） | 进新 index，**定义要换** | `kb/index.py:135`：只要有一条 BM25 命中就是 strong。百万级 OR 查询中位命中 47.7% 的行（§1.1），几乎恒为 strong | 改为结构化信号：查询里 IDF 最高的 m 个词在 top-1 中的覆盖率、top-1 与第 k 名的分差、命中所在层级（表述 / 段落 / 只有元数据）；阈值用 dev 题标定后冻结 | M |
| `kb/store.py` KB + state_search | **冻结** | 加载时嵌入全部族名（`store.py:45-48`）；截止按年（`store.py:87-97`） | 只服务 v9b 臂 | — |
| `kb_compiler/retrieve/hybrid.py` | **退役** | 与 `kb/index.py` 只差 `pid_filter`（diff 实测），是重复实现 | 删除 | S |
| `kb_compiler/views/search_text.py` | **退役** | 每次查询对每个 chunk 重算 tf（`search_text.py:78-103`）；纯 Python 余弦遍历全部向量（`:118`）；`append_paper` 线性查重（`:183`） | 删除。确定性词形扩展（`_expand_query :124`）由 tantivy 的 `en_stem` 分词器取代（`en_stem` 未实测） | S |
| `retrieval/search_service.py` | **退役** | v9b 已因"15 s 时隙 + 熔断把配额排队当成失败，45 路并发里 15 路为空"而绕开（`answer/pipeline.py:48-51`） | 外检工具直接用 `sources/sciverse.py` | S |
| `retrieval/query_understanding.py` | **退役** | A6 召回 0.062 判 FAIL；plan 阶段已经生成子查询（INTEGRATED §3 第 96 行） | 删除 | S |
| `retrieval/rerank.py` | **需优化**（并入 `sources/`） | 纯 Python 余弦（`rerank.py:80`），≤64 个候选可以接受；自带一套 embedding（`:60-77`），违反"每项能力一份实现"；`candidate_text` 重建 OpenAlex 摘要，与 `refgraph._oa_abstract` 重复 | 改用 `llm.embedding` 和 numpy；只在融合本地与外部结果时作可选的同模型重排，默认用 RRF；摘要重建留一份 | S |
| `retrieval/circuit.py` | **直接可用**（移入 `sources/`） | 确定性、线程安全（`circuit.py:36-90`） | 只包住网络错误和 5xx，429 交给 sciverse 自带的跨进程令牌桶排队（`sciverse.py:108-135`）。注意状态是进程内的，harness 每个会话一个进程，彼此不共享熔断状态 | S |
| `index/build.py` FTS5 三套表 | **替换** | §1.1：100 万行 1.07 s，长查询 8.9 s；`build()` 把全部表述 fetchall 进内存（`:76`），每次 DROP 全量重建（`:57-59`），grow 追加只能全量重建 | 改用 tantivy 三个索引（papers 覆盖 registry 全部论文的元数据，带 tier 字段；passages；statements）。fast field：`day_hi`（整数 yyyymmdd，取保守上界）、`paper_id`（raw）、`kind`/`facet`/`section`；按 segment 增量写；建索引改成流式 | L |
| `index/build.py` dense（全量 numpy） | **替换** | §1.2：100 万 × 4096 = 16.4 GB，每个 MCP 进程一份；`dense` 表没有模型列（INTEGRATED §3 缺陷） | 按日期排序存储，二值码 memmap 粗召回，float16/32 memmap 精排 top-200；表头记模型标签和维度；千万级表述先只给 self 表述和论文建向量，全量另评估 ANN（§5.3） | L |
| `index/build.py` papers 索引范围 | **需优化** | 只收"系统知道点什么"的论文（`:44-48`），不满足"论文级数百万元数据可检索" | 收录 registry 全部论文（104.6 万），每条带 tier（T0/T1/T2/has_text），让智能体知道读到了多深 | S（随替换） |
| `tools/api.py` find / read / compare 类 | **需优化** | 单次 1–74 ms（§1.4）；但每个返回都过 `_t()`（`:63-65`），截断加 "…" 并压缩空白，破坏了文档声明的逐字原文（`:5-6`） | 返回 `quote` 原文，另给 `display` 截断字段；`what_is_missing` 改为 registry 的 `has_text` 列，不再每次加载全部 doc id（`:116`）；AsOf 改用进程级只读连接池（当前连接从不关闭） | M |
| `tools/api.py` field 类（field_map / paper_profile / frontier / open_issues） | **需重构** | 见 §3 | 读物化表（§4） | L |
| `tools/mcp_server.py` | **需优化** | FastMCP 对同步工具直接在事件循环里调用（`func_metadata.call_fn_with_arg_validation` 源码：`return fn(**args)`），一次 60 s 的调用会堵住整个服务；截止由服务端固定（`:20-22`），这一点正确；没有外检工具 | 工具改 async，用 `anyio.to_thread.run_sync` 包同步实现；单次调用设硬超时，超时返回结构化错误；新增 `search_external` / `expand_citations`；工具描述标明"本地 / 外部" | M |
| `cognition/asof.py` | **需优化** | 只读视图本身正确；`co_cited` 缺索引（§1.3）；`visible()` 对 stub 恒为真（`:34-38`） | 立即给 citations 加 `INDEX(sentence_id)`；日期过滤改用保守上界（§7）；连接池化 | S |
| `cognition/identity.py` | **需重构**（物化） | 每次全表扫描两遍（`identity.py:32-48`）10.4 s；`aliases()` 每次 O(全部名字)（`:59-60`） | 物化 `name_evidence`，按名字和论文查（§4） | M |
| `cognition/lineage.py` | **需重构**（物化） | 全量扫描 other 表述（`lineage.py:43`）29.6 s；回退解析时在循环里调 `ident.aliases(g)`（`:54`），每次 100 ms；`lineage_of` 每次都重算全量（`:63-64`） | 物化 `lineage_edge`，带解析区间（§4） | M |
| `cognition/families.py` | **需重构**（物化） | 每个 scope 句子调一次 `co_cited`（`families.py:35-39`）；`support` 推导式复杂度为成员数 × 对数（`:98`）；族随查询 scope 变化，同一 T 下不同问题会得到不同的族 | 物化 `cocite` 和 `category_member`；族在离线快照里算（季度快照，可接 LLM 命名 / 裁定，INTEGRATED §4），读时取 ≤T 的最近快照 | L |
| `cognition/facts.py` | **需重构** | Jaccard 规则聚类（`facts.py:61-75`），INTEGRATED §4 要求改成"向量粗聚类 + LLM 裁定"，只能离线做；`independent()` 是贪心作者合并，结果依赖顺序（`:47-58`） | 物化 `fact` 和 `fact_support`，再加 `author_group`（离线 union-find）；status 在读时由 ≤T 的支持行算出 | L |
| `cognition/profiles.py` | **需优化**（在线） | 有 `N_OTHER=30` 兜底，≤1 ms；但 `n_citing` 取的是被截过的表述的说话人数（`profiles.py:59`），恒 ≤30（头部论文实际 11,263 次引用） | `n_citing` 改读 `cite_count_monthly`（零 LLM，来自 cites）；表述只作样本，标明 `sample=k/n` | S |
| `cognition/shifts.py`、`comparisons.py` | **直接可用**（在线） | 单对象有界：`events` 18 ms，`compared_with` 1 ms | `superseded` 改读 `lineage_edge(parent=?, relation='replaces')` | S |
| `answer/pipeline.py` v9b `answer` | **冻结** | 表征测试 | 不动 | — |
| `answer/pipeline.py` `answer_lit` / `lit_gather` | **需重构** | 见 §5 | 修 snippet / identity / 归属，接外检，探测阶段也走外检，并行取证 | M |
| `answer/pipeline.py` `ext_search` / `cite_expand` | **需优化**（移入 tools） | 依赖线程局部截止（`:76,109,124`）；`answer_lit` 不设截止（`:443-466`） | 改为显式传 `as_of`；共引排序（≥2 个种子共同引用，`:115`）保留 | M |
| `baselines/harness/runner.py` | **需优化** | `ALLOWED_TOOLS` 写死两个 retrieval 工具（`runner.py:48`）；`mcp_config` 只注册 retrieval 服务，传的是月粒度 `KNOWLEDGE_CUTOFF`（`:72-83`） | 见 §6.1 | M |
| `baselines/harness/mcp_server.py` open 模式 | **冻结**（harness-open 臂） | Sciverse 包装，服务端强制按年截止（`:180-209`） | 不动，作为"开网 harness"对照 | — |
| `baselines/harness/mcp_server.py` corpus 模式 | **退役** | 每次查询纯 Python 扫全部 chunk（`:99-110`） | 固定语料臂改用 lit 工具加 scope 限定 | S |
| `eval/cs2/{judge,scoring,format_align}`、`stats`、`length` | **直接可用** | 按 qid 汇总（`scoring.py:75`），判分只读 sections / title / snippets，与身份无关 | 只迁出 `.research_tmp` 路径（`scoring.py:45`、`judge.py:60`） | S |
| `eval/dsb.py` | **需优化** | 判分只看文本；题集读 `.research_tmp/review_1002/p6/oracle_inputs.json`（`dsb.py:105`）；跑题脚本用 `published_date[:7]` 月粒度截止（`legacy/benchmarks/cs2/run_vnext_dsb.py:49`） | DSB 适配器给 `as_of = published_date 的前一天`（官方规则是 "published before {cutoff_date}"） | S |
| `eval/field.py` | **需重构** | 只比较旧的 `compile.state.field_state`（`field.py:22,230`），新 cognition 没接进来；heldout_refs 2,957 条里有 675 条 `doi == "None"` 字符串；综述截止只有年（`:67`） | 新增 cognition 臂：refs 经 DOI / arXiv 映射到 paper_id（把 "None" 当缺失），`AsOf(综述发表日)` 下跑 families / facts；综述日期补到日 | M |

## 3. 工具逐个代价与归属

"现状"取 §1.4 的合成实测，"千万级"按线性外推。

| 工具 | 主要代价 | 现状 | 千万级表述 / 百万全文 | 归属 |
|---|---|---|---|---|
| search_papers / closest_prior | index.papers + k 次 `_brief` | 4 ms + 检索 | 检索换 tantivy 后 <100 ms；`_brief` 有界 | 在线 |
| citations_of | cited_by 全部行，在 Python 里去重排序 | 52 ms（9,319 行） | 头部论文行数随全文规模增长，可达 10 万+ | 在线；SQL 里 `GROUP BY citing ORDER BY min(date) DESC LIMIT k` |
| references_of | entries + 每条一次 paper() | 1–6 ms | 不变 | 在线 |
| what_is_missing | `Documents().ids()` 全量 + 30 个种子的参考文献 | 139–382 ms | 100 万全文时每次读 100 万个 id | 在线；改 `has_text` |
| paper_card | statements(about, self) | 10 ms | 有界 | 在线 |
| read | 整篇解压 units JSON | 未测 | 单篇有界 | 在线 |
| find_evidence | 表述检索 + 段落检索 | 74 ms + 检索（FTS5 100 万行 1.07 s） | 换 tantivy 后 <100 ms | 在线 |
| field_map | `_scope` → families → co_cited × N | 约 3 h（无索引）/ 1.9 s（有索引） | 共引对随 scope 增长；族随查询漂移 | **物化**：cocite、family_snapshot、fact |
| paper_profile | Identity 全量 + lineage 全量 | 63.7 s | 约 10 分钟、>15 GB（外推） | **物化**：name_evidence、lineage_edge |
| baselines_for | 30 个种子 × statements(speaker) | 7–9 ms | 有界 | 在线 |
| open_issues | scope + facts(200) | 0.1 s | Jaccard 改成 LLM 后只能离线 | **物化**：fact、fact_support；读时算 status |
| frontier | 40 次 `_brief` + `L.edges(view)` 全量 + 40 次 events | 37.7 s | 同 paper_profile | **物化**：lineage_edge |
| compared_with | statements(about) | 1 ms | 有界 | 在线 |

结论：凡是"对全体表述做一次全局计算"的地方（Identity、lineage、families、facts），必须编译时物化；"对单个对象读有界行"的地方（profile、events、card、compare）可以继续现算。前提是有界性要么来自 `N_OTHER` 上限，要么来自物化表上的索引。

## 4. 物化表设计（stage `cognition`，派生库）

原则：
- 物化表存的是**带日期的证据行或事件行**，不存某个 T 下的结论。
- 读时用 `WHERE date_hi <= :as_of` 走 (key, date) 复合索引，对单个对象做有界聚合。as_of 单调性由此仍然只依赖行日期。
- 非局部的结构（族、事实聚类）用离线快照，读时取 `snapshot_date <= :as_of` 的最近一份。这样不会泄漏未来信息，代价是最多落后一个快照周期。

```sql
-- 名字证据：取代 Identity(view) 的全量扫描
CREATE TABLE name_evidence(norm_name TEXT, paper_id TEXT, date_hi TEXT, weight INT,  -- self proposes 3 / alias 2 / other 1
                           kind TEXT, statement_id INT);
CREATE INDEX ix_ne_name  ON name_evidence(norm_name, date_hi);
CREATE INDEX ix_ne_paper ON name_evidence(paper_id, date_hi);
-- resolve(name, T) = SELECT paper_id, sum(weight) ... WHERE norm_name=? AND date_hi<=? GROUP BY paper_id，读时套 2 倍规则
-- LLM 裁定结果（INTEGRATED §4：规则召回候选、LLM 裁定）
CREATE TABLE name_verdict(norm_name TEXT, paper_id TEXT, verdict TEXT, model TEXT, prompt_sha TEXT, decided_at TEXT);

-- 谱系边：取代 lineage.edges(view) 的全量扫描
CREATE TABLE lineage_edge(child TEXT, parent TEXT, relation TEXT, kind TEXT,   -- self | third
                          speaker TEXT, date_hi TEXT, statement_id INT, sentence_id INT,
                          parent_name TEXT,                  -- third 边上的原始名字
                          resolvable_from TEXT, resolvable_until TEXT);  -- 由 name_evidence 按时间回放得到的区间
CREATE INDEX ix_le_child  ON lineage_edge(child, date_hi);
CREATE INDEX ix_le_parent ON lineage_edge(parent, relation, date_hi);
-- 可见条件：date_hi<=T AND (kind='self' OR (resolvable_from<=T AND (resolvable_until IS NULL OR resolvable_until>T)))
-- 时间不一致的边（child 早于 parent）在构建时丢弃并计数；combines 超边用 sentence_id 分组

-- 共引对：取代 families.edges 里逐句调用的 co_cited（实测建表 19 s，165 万行；298 篇 scope 查询 52 ms）
CREATE TABLE cocite(a TEXT, b TEXT, citing TEXT, sentence_id INT, date_hi TEXT);  -- a<b，组大小 2..MAX_GROUP
CREATE INDEX ix_cc_a ON cocite(a, date_hi);
CREATE INDEX ix_cc_b ON cocite(b, date_hi);
CREATE TABLE category_member(category TEXT, paper_id TEXT, speaker TEXT, date_hi TEXT);
CREATE INDEX ix_cm_cat ON category_member(category, date_hi);
CREATE INDEX ix_cm_paper ON category_member(paper_id, date_hi);

-- 被引计数：修正 n_citing 被截在 ≤30（每篇引用方只有一个日期，所以按月的不同引用方数可以直接相加）
CREATE TABLE cite_count_monthly(paper_id TEXT, month TEXT, n_citing_new INT, n_sentences INT,
                                PRIMARY KEY(paper_id, month));

-- 独立性：离线 union-find（作者重合 ≥ 半数），取代读时的贪心 independent()
CREATE TABLE author_group(paper_id TEXT PRIMARY KEY, group_id INT);

-- 族快照：离线（季度；LLM 命名 / 裁定只在这里做）
CREATE TABLE family_snapshot(snapshot_date TEXT, family_id TEXT, paper_id TEXT, support INT,
                             PRIMARY KEY(snapshot_date, family_id, paper_id));
CREATE INDEX ix_fs_paper ON family_snapshot(paper_id, snapshot_date);
CREATE TABLE family_name(snapshot_date TEXT, family_id TEXT, name TEXT, source TEXT, model TEXT);

-- 族级事实：离线聚类（向量粗聚类 + LLM 裁定）；status 由读时 ≤T 的支持行算出
CREATE TABLE fact(fact_id TEXT PRIMARY KEY, family_id TEXT, facet TEXT, text TEXT, model TEXT, prompt_sha TEXT);
CREATE TABLE fact_support(fact_id TEXT, statement_id INT, speaker TEXT, about TEXT, date_hi TEXT, polarity TEXT);
CREATE INDEX ix_fsu_fact ON fact_support(fact_id, date_hi);
CREATE INDEX ix_fsu_about ON fact_support(about, date_hi);
-- status(T)：按 author_group 数 count(DISTINCT group) 和 count(DISTINCT about) 判定；
--            contested 由 LLM 写入的 polarity 决定，不再用否定词正则
```

registry 补两列：`has_text`（取代 `Documents().ids()`）和 `date_lo` / `date_hi` / `date_precision`（§7）。

citations 立即补一个索引，不用等重构：`CREATE INDEX ix_cites_sid ON cites(sentence_id)`。实测建索引 0.7 s，`co_cited` 从 152 ms 降到约 0，field_map 从约 3 h 降到 1.9 s。

## 5. answer_lit：证据形态核对与外检接入

### 5.1 screen / write / assemble 的隐含假设，逐条核对

| 假设（v9b） | 位置 | lit 证据实际情况 | 后果 |
|---|---|---|---|
| snippet 是逐字原文，长度 800–1200 | kb 900（`kb/store.py:114`），ext / cite 1200（`pipeline.py:82,128`） | lit_self 的引语是 `_t(quote, 260)`，lit_other / lit_passage 是 `_t(…, 300)`（`tools/api.py:139,180,183`），带 "…" 且压缩了空白；只有摘要回退是 `[:1200]` | **静默**：snippet 不再逐字、明显变短。format_align 的背景就是 snippet 长度会影响 CS2 引用 facet（`format_align.py:3-6`，ours 中位 1,140 vs harness 240）；据 10-04 交接，格式对齐后 ours 相对 harness 掉了 0.117 |
| `paper_key` 格式 kb:/ext: | `pipeline.py:81,111,126` | `paper:<id>` | 去重键 `(paper_key, snippet[:160])` 照常可用 |
| `_identity` 先看 arxiv，再看规范化标题 | `pipeline.py:320-331` | lit 的 `arxiv = b["id"][6:]`（`:428,437`） | 现在能用。DOI 优先身份落地后 arxiv 字段会写进 `PPR_…`。接外检后，同一篇论文本地走 `arxiv:` 键、外部走 `title:` 键，**会拿到两个编号**（W1-13 回归） |
| year 是 int | `_fmt_ev`（`:241`）只用于展示 | str（`"2023"`） | 无害 |
| src 名字 | `_interleave` 的优先序写死为 target/ext/kb/cite/state（`:277`）；`assemble` 只特殊处理 target（`:347`） | lit_self / lit_other / lit_passage 落在队尾，仍然轮转 | 无错，但 v9b"外部优先"的调优没有对应物 |
| snippet 取自被引文献 | CS2 prompt（`runner.py:25`）、citation 判分 | lit_other 的 `paper_key` 是被引论文（about），snippet 却是**引用方**的句子（`pipeline.py:430-437`），常带 "[12, 15]" 一类标记 | **静默归属错位**，需要决定：要么挂在说话方名下（"Y 说 X…"），要么保留挂在 X 名下但在 metadata 标 `described_by` |
| trace 可审计 | `answer()` 写出 evidence / used_eids（`:496-497`） | `answer_lit` 不写（`:463-466`） | 事后无法做来源统计和外检占比分析 |
| 规划前先探测 | v9b 用外部 + KB 探测，修了 20/74 节跑题（`:133-148`） | 只用本地 search_papers（`:452`） | 本地覆盖不到的题，规划等于盲拆 |
| field_block 很便宜 | — | `field_map` 在真实库上约 3 h（§1.3） | **当前 answer_lit 跑真实库会卡死在规划前** |
| 取证并行 | `gather` 6 线程（`:197`） | `lit_gather` 串行，每条查询约 24 次工具调用（`:414-437`） | 工具变快后问题不大；外检加进来后必须并行 |

### 5.2 改法

1. 工具层把 `quote` 原文和 `display` 分开。evidence 只用原文；长度对照仍走 `format_align`，不在工具里截。
2. identity 一律取 `paper_id`。外检命中在 gather 时就近解析：DOI → arXiv → `title_prefix`（papers 表已有 `ix_prefix`），解析到了就并入本地 paper_id；解析不到用 `ext:<来源>:<doc_id>`。`arxiv` / `doi` 字段从标识表取，不再切字符串。
3. lit_other 的归属按上表裁定（需用户定）。
4. 探测和每条查询都并行跑本地 + 外检；trace 补上 evidence 和 used_eids。

### 5.3 外检接入（复用现有函数，结果标"外部"，评测期不写回）

- `tools.search_external(query, as_of, k)`：
  - 复用 `SciverseClient.semantic_search`（`sources/sciverse.py:198-235`），把截止改成显式参数：服务端过滤 `year ≤ as_of.year`；
  - 年份等于 `as_of.year` 的命中，先解析到本地 paper_id 取日级日期，解析不到再查 OpenAlex `publication_date`，仍拿不到就排除（与现行"同年排除"一样保守）；
  - snippet 取 `raw.abstract`，可选 `raw.chunk`；返回 `{"external": true, ...}`。
- `tools.expand_citations(seeds, as_of)`：
  - 本地种子走 `references_of` + `cocite`，零网络；
  - 外部种子走 `refgraph.prefetch` → `refgraph.references` → `refgraph.resolve_abstracts`（`sources/refgraph.py:346,390,440`）；
  - 共引排序搬自 `pipeline.cite_expand`（`:89-129`，n≥2）。
- `circuit.SourceCircuit` 只包网络错误；配额（30 次/分钟，`pipeline.py:31`）由跨进程令牌桶排队。harness 与内置管线并发时共用这一个桶，吞吐上限是硬约束，需要在实验计划里算账。
- refgraph 缓存目录目前指向 `.research_tmp`（`refgraph.py:29`），迁到 `cache/refgraph/`。
- 不写回：外检结果只进本次答案的 trace；grow 只能离线读 trace 里的候选，走 acquire 全链路后才进入文库。

## 6. harness 与评测

### 6.1 新工具层接到 harness 臂上要改的

1. `runner.mcp_config` 增加 `literature` 服务（`compilescholar.tools.mcp_server`），`env.DFC_AS_OF` 与内置管线用同一个适配函数算出；`ALLOWED_TOOLS` 加上 `mcp__literature__*`。
2. 臂定义分开：harness-open（现 Sciverse 两工具，冻结）和 harness-lit（lit 工具 + search_external / expand_citations）。不要往同一个服务器里混两套 `search_papers`。
3. 每个 `claude -p` 会话都会拉起一个 stdio 服务进程。所以服务启动必须轻：只打开 tantivy 和 memmap，不预载矩阵，不建 Identity；并发 N 路靠操作系统页缓存共享。
4. 工具要 async，并设单次硬超时。Claude Code 的 MCP 工具超时默认值本次未核实。
5. 返回里带展示用的 id（DOI / arXiv）和原文 quote。CS2 prompt 要求 snippets "extracted from the reference document"，并且 metadata 里要有 arxiv（`runner.py:25-46`）。
6. corpus 模式退役，固定语料臂改用 lit 加 `DFC_SCOPE`。

### 6.2 评测代码对身份和截止的假设

| 代码 | 身份假设 | 截止假设 | 要改的 |
|---|---|---|---|
| `eval/cs2/scoring.py`、`stats.py` | 只用 qid | 无 | 只改路径 |
| `eval/cs2/judge.py` | 判分读 title + snippets，与身份无关 | 无 | 只改路径；snippet 截断会影响 citation facet（§5.1） |
| `eval/cs2/format_align.py` | `abstracts` 按小写标题做键（`format_align.py:62`） | 无 | 无 |
| `eval/dsb.py` | 纯文本 nugget | 题集文件里有 `published_date` 时间戳；旧跑题脚本取到月 | 适配器给日级 `as_of` |
| `eval/field.py` | 金标 refs 用标题 + 摘要；新 cognition 臂需要 refs → paper_id | 综述只有年 | 映射时把 `doi == "None"` 当缺失（实测 675/2,957 条）；补综述发表日 |

## 7. 截止语义：使用点与冲突

年 / 月粒度（`core/cutoff.py`，排他月，同年排除）的使用点：
- `kb/store.py:87-103`（v9b KB）
- `answer/pipeline.py:23,76,109,124,137,197,477`（ext / cite / probe / gather / answer）
- `sources/sciverse.py:208-210`（服务端 `year ≤ cut_year−1`）
- `baselines/harness/mcp_server.py:182-194`
- `baselines/harness/runner.py:76` 与 `cli.py:57`（环境变量 `KNOWLEDGE_CUTOFF`）
- `core/manifest.py:22`（记录环境）
- `retrieval/sources.py:34-43,816,853`：**另一套实现**，查 `sys.modules["cutoff"]`，看不到 compilescholar 的线程局部截止
- `kb_compiler/views/tools.py:326-336,880-899`：`as_of_year`，年粒度，旧系统

日粒度（as_of，含当天）的使用点：
- `tools/mcp_server.py:20-22`（`DFC_AS_OF`）、`tools/api.py` 全部
- `cognition/asof.py:26-90`
- `index/build.py:134,147,152,169`
- `answer/pipeline.py:379-386`（月 → 前一天）、`:450`

冲突点：

1. **外检与本地粒度不一致。** CS2 cutoff `2025-05` 时本地 as_of 为 2025-04-30，Sciverse 只给 ≤2024。外检丢掉 2025 年 1–4 月，本地却能看到。修法：按 §5.3 把同年命中解析到日级日期。
2. **环境截止与显式 as_of 并存。** `answer_lit` 没有调 `_set_cut`。如果直接把 v9b 的 `ext_search` 接进来，它会读 `cli.py:57` 设的全局环境变量；DSB 每题截止不同、又并发作答，就会用错截止。修法：外检函数只接收显式 `as_of`，删掉线程局部状态。
3. **月 → 日转换丢天。** DSB 的 `published_date` 是时间戳；旧脚本取 `[:7]` 之后 `as_of_date` 得到上月最后一天，最多少看 30 天。方向保守，不泄漏，但和官方口径不一致。
4. **混合精度日期的字符串比较。** 实测 `'2024' <= '2024-03-01'` 为 1，`'2024-05' <= '2024-05-15'` 为 1。registry 引入"日期精度（日 / 月 / 年）"后，只有年份的论文会在当年 1 月 1 日就可见，提前泄漏。修法：存 `date_lo` / `date_hi`，可见当且仅当 `date_hi <= as_of`（与 `cutoff.allowed` 同年排除的保守口径一致）；tantivy 和向量都按 `day_hi` 过滤。
5. **v1 日期与文本版本错配**，这是数据层泄漏，工具层修不了。引用句标的是引用方的 v1 日期（`citations/build.py:44-45` 的 `_write_doc` 用 `papers.v1_date`），文本却来自 HTML / ideaforecast 的后续版本。实测 1,552,983 条已解析引用里有 8,142 条（0.52%）指向比引用方 v1 更新的论文，其中 1,139 条相差超过 180 天；在 T=2023-06-30，有 761 个对象经引用句提前可见（1,205 行）。修法：documents 记录文本版本号和版本日期，引用句与表述的日期取版本日期，不取 v1。
6. harness 的 `KNOWLEDGE_CUTOFF`（月）与 lit 服务的 `DFC_AS_OF`（日）必须由同一个适配函数从基准配置推出，不能各自手填。
7. `retrieval/sources.py` 的截止查找随包一起退役。

## 8. 之前被默认可用、实际不满足要求的点

1. "date 过滤在查询内部完成"（`index/build.py:11`）：确实在查询里，但不走索引。100 万行 1.07 s，长查询 8.9 s。
2. "沿用旧的 RRF + 每篇上限"（`index/build.py:10-11`）：只有向量命中的条目绕过每篇上限，也绕过 kind/facet 过滤（`:177-186`）。
3. "命中强弱信号可以直接并入"（INTEGRATED §2 index 行）：百万级 OR 查询下几乎恒为 strong，定义必须换。
4. "cognition 现算"（ARCHITECTURE.md:20）：paper_profile 64 s、frontier 38 s，每次 RSS 增加约 2 GB。
5. field_map 只差一个索引：在真实库上约 3 h / 次，answer_lit 会卡死在规划之前。
6. `n_citing` 被 `N_OTHER=30` 截断：头部论文实际 11,263 次引用，工具报 ≤30。
7. as_of 单调不变式在日期层面成立，在文本版本层面不成立（§7 第 5 条）。
8. 工具返回"逐字原文"（`tools/api.py:5-6`）：`_t()` 截断加 "…" 并压缩空白之后已经不是原文；answer_lit 和 harness 都会把它当 snippet 用。
9. lit_other 的 snippet 归属是引用方而不是被引文献。
10. 17 处 `[6:]` 把 `paper:` 后面的部分当 arXiv id，用来查 `arxiv_id` 列；`pipeline.py:428,437` 还把它写进 citation 的 `arxiv` 字段。
11. MCP 同步工具在事件循环里执行，一次慢调用会堵住整个服务。
12. 向量"按需全量载入"：每个 harness 会话一个进程，N 路并发就是 N 份 16 GB。
13. "全量规模时换 Qdrant 即可"（`search_text.py:21-23`）：本机 docker daemon 未运行，qdrant-client 的 local 模式是暴力检索。
14. answer_lit 没有外检，也没有外部探测；v9b 有 61.9% 的引用来自外检（INTEGRATED §2 answer 行）。
15. `retrieval/sources.py` 的截止查找看不到包内线程局部截止。
16. heldout_refs 里 675 条 `doi == "None"` 字符串，DOI 优先映射时会被当成真 DOI。
17. `index.build` 一次性把全部表述读进内存，每次全量重建，grow 追加无法增量。

## 9. 未核实 / 待定

- tantivy 的 `en_stem` 分词器和中文分词没有实测；合成段落是 120 词的随机词，真实段落长度分布和短语查询没测。
- 本地 qwen3-embedding-8b 嵌入千万条表述的吞吐和耗时没测。这决定了表述级向量是全量做，还是只做 self 表述。
- ANN（usearch / faiss）未装，加依赖需用户同意（锁定版本）；chromadb 只在 2.9 万条真实向量上测了召回。
- Claude Code 的 MCP 工具超时与结果大小上限未核实。
- lit_other 的归属（挂在说话方名下还是被引方名下）需要用户裁定，它直接影响 CS2 的 citation facet。
- 族 / 事实的质量不在本次合成实测范围内。
