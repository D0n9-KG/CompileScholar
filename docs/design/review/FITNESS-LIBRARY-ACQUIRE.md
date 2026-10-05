# 底座模块适用性审查：文库 / 来源 / 获取 / 解析 / 截止（FITNESS-LIBRARY-ACQUIRE，2026-10-05）

> 对象：INTEGRATED-SYSTEM-1005 §2–§3 里准备"保留或迁移"的底座模块。
> 方法：逐个读源码；用合成数据在临时目录实测；现有库一律只读打开；MinerU 9605 做了 2 次解析；没有调用 LLM。
> 实测环境：本机 Windows 11 + Python 3.13；Sci-Hub 索引在本地盘，PDF 档案在 UNC 共享盘。临时文件都在 `%TEMP%\fitness_review_1005`。
> 结论先行：这批模块没有一个能"原样迁入"。身份层的 merge/去重逻辑有正确性缺陷，规模层是逐条事务加全表扫描，MinerU 客户端整套过时，截止规则在 4 处各写一套。可以复用的是**表结构的思路**和少数函数，不是代码本身。

## 0. 先厘清两个事实

1. **`src/retrieval` 实际没有被新系统使用。** `compilescholar` 里没有任何 `from retrieval` 的 import，只在 `sources/sciverse.py:3` 的 docstring 里出现过。retrieval 也不在打包范围内（pyproject）。它和 `sci-evo-extract/src/sci_evo_extract/library/` 是两份并行演化的拷贝：registry 去掉 import 名后 diff 为 0；sources.py 差 90 行；acquisition.py 差 5 行。所以 sci-evo 那 24 个测试测的是 sci-evo 自己的拷贝，不是 CompileScholar 的这份。
2. **sci-evo registry 的实测规模很小**（只读打开 `scievo_registry.sqlite`）：490 篇论文、1,681 个标识、420 个 PDF、415 个 MinerU 作业（407 个 ready、8 个 failed）、46,174 个 evidence_units。`extraction_runs`、`paper_compile_states`、`paper_compile_log`、`paper_citations`、`paper_collections`、`upstream_asset_links` 这几张表都是 0 行。"Multi 语料取到 431/436（98.9%）"这类旧指标都是在这个规模、以 CS 为主的语料上测的，**条件不同，不能直接沿用**到百万级、跨领域的场景。

## 1. 总表

| 模块 / 功能 | 判定 | 证据（file:line、实测） | 具体改法 | 工作量 |
|---|---|---|---|---|
| registry：`papers` / `paper_identifiers` 表结构 | 需重构 | 标识表 `UNIQUE(scheme,value)` 方向是对的（registry.py:3477-3485）。缺少：日期精度字段，`published_date` 是自由文本，实测数据里同时有 `'2022'` 和 `'2019-01-01'` 两种格式（13 条中 1 条只有年）；版本 / 预印本关系；作者；`paper_identifiers.paper_id` 索引（EXPLAIN 结果是 SCAN）。`year`/`published_year` 两列语义重复。62/490 篇没有标题 | 新表：`papers(paper_id, canonical_doi, title, norm_title, first_public_date, date_precision{day,month,year}, date_source, published_date, ...)` + `paper_identifiers(scheme,value,paper_id,source,role{self,version_of,preprint_of})` + `idx(paper_id)`；数据从 sci-evo 迁移，不沿用旧 DDL | 中 |
| registry：`paper_id` 生成 | 需重构 | `stable_id("PPR_", normalized_doi or title or uuid4)`（registry.py:164）有三个问题：① 用**原始标题**做哈希，大小写不同就得到不同 id，实测 "Granular Flow" 和 "granular flow" 是两个 id；② 没有 DOI 也没有标题时用随机 uuid，**不幂等**；③ 预印本先入库时，DataCite DOI `10.48550/arxiv.*` 成为主键，实测 sci-evo 里有 260/490 篇的 `normalized_doi` 就是这种 DOI，之后正式版 DOI 不会替换它（C1'） | 规则：有正式版 DOI 用正式版 DOI；只有 arXiv 时用 `arxiv:<id>`；两者都没有用 `title:<norm_title>|<year>`。paper_id 一旦发出就不再改，换"首选 DOI"只改 `canonical_doi` 列，不改 id | 小 |
| registry：`normalize_doi` | 需重构 | registry.py:18-35。实测问题：`http://www.doi.org/…` 原样保留；`%28…%29` 和 `%2F` 不解码；末尾 `.` 保留；`(10.x)` 括号保留；`arXiv:2101.00001` 被当成 DOI 放行。同一个 DOI 的 7 种写法 upsert 后变成 **4 篇**论文（C4）。全仓至少 3 套规则互不一致：refgraph 只做 replace，大小写敏感；skeleton `bib.DOI` 会在 `;` 处截断 SICI DOI，见子审查 A | 一份 `core/ids.py`：unquote → 剥掉 `doi:` 前缀和任意 `(dx.\|www.)?doi.org/` → strip 末尾标点和成对括号 → lower；`10.48550/arxiv.X` 识别为 arXiv 标识，不当作独立 DOI。这些写法在 Sci-Hub 索引上全部要命中（见 §3） | 小 |
| registry：`normalize_arxiv_id` | 需重构 | registry.py:116-123 不剥 `arXiv:` 前缀，结果是 `arxiv:2101.00001`。旧式 id 只做 lower，`math.GT/0309136` 被改坏。同一个 arXiv id 的 5 种写法变成 **3 篇**（C10）。citations/resolve 的正则 `\d{4}\.\d{4,5}`（resolve.py:20）完全不认旧式 id，实测 `hep-th/9711200` 和 `arXiv:cond-mat/0507321` 都没识别出来 | 并入 `core/ids.py`；旧式 id 保留子类大小写（arXiv 官方 id 里 `math.GT` 是大写） | 小 |
| registry：upsert 的合并逻辑 | 需重构 | 用构造用例实测：<br>**C1** 预印本 DOI + 正式 DOI，arXiv id 用 `arXiv:…v2` 写法 → 分裂成 2 篇；<br>**C2** 先有正式版（无 arXiv id）再来预印本，同标题同年、置信度 0.99 → 2 篇，原因是 DOI 不同时 `_find_paper_by_title_year` 跳过（:2748-2750）；<br>**预印本 2014 / 正式版 2015** → 2 篇，年份必须完全相等（:2743）；<br>**C7** 先按标题入库、后带 DOI 入库 → 2 篇，标题匹配要求 `identity_confidence ≥ 0.92`（:2737），调用方大多不传这个参数；<br>**C5** 两篇不同的 "Editorial"（1990 年 JFM、2010 年 Nature）无 DOI → 被**合成 1 篇**，原因是 paper_id 来自标题哈希，走了 ON CONFLICT 分支；<br>**C8** 带上别人的 DOI 作为标识 → **被并入那篇论文**；<br>sci-evo 里真实存在的重复对：PPR_1E5D…（arXiv）和 PPR_792C…（PRE），是同一篇论文的两个条目 | 拆成"身份解析"和"写入"两步。① 先按全部标识查询，命中多个 paper_id 时进入合并队列，不在写入时偷偷选一个。② 标题回退只在双方都没有冲突的强标识时用，年份允许 ±2，同时用作者姓氏核对；是否合并由 LLM 裁定，符合设计 §4 "规则召回、LLM 裁定"。③ 短 / 泛化标题（Editorial、Reply、Erratum…）禁止只凭标题合并 | 中 |
| registry：标识污染（真实数据） | 需重构（数据迁移前必须清洗） | sci-evo 里 13 篇论文各挂着 2–12 个**互不相干的 DOI**。比如 PaperQA（PPR_10ED…）挂了 ssrn、techrxiv、NIST、EMNLP 等 10 个 DOI。原因是 `source_candidate_identifiers`（acquisition.py:1499-1520）把所有 `ready` 的**检索候选**的 DOI 当作本篇的标识写进去了（:346、:486）。66 条 doi 标识和本篇 `normalized_doi` 不一致；276 条 DataCite DOI 混在里面。C11 实测：被污染之后，真正拥有这个 DOI 的论文入库时会拿到新 id，但 `find_paper_by_identifier('doi', …)` 仍然返回被污染的那篇，新论文自己连一条 doi 标识行都插不进去（`ON CONFLICT DO NOTHING`，:2695） | 标识只来自"已确认指向本篇"的来源（DOI 精确查询的返回、arXiv 元数据里的 doi 字段）；检索候选只进 `source_candidates`。迁移时只保留 `source='user_input'` 的标识，以及与本篇 DOI 指向同一实体的 openalex/arxiv 标识，其余重新解析 | 中 |
| registry：最早公开日期 | 需重构 | upsert 用 `COALESCE(old, new)`（:183-186、:235-237），**谁先写谁留**。C3 实测：先导入正式版（2016-03-01）、再导入预印本（2015-11-01），结果留下的是 2016。`backfill_paper_year_from_candidates` 只处理年份，候选里没有 `published_date` 字段（:1157） | `first_public_date = min(各版本日期)`，精度取得出该日期的那个来源的精度；每次写入都重新算 min，每个来源的日期留存到 `paper_dates(paper_id, source, date, precision, kind{v1,online,issue})` | 小 |
| registry：标题规范化 | 需重构 | `normalize_title` 只保留 `[a-z0-9]`（:38-42）。实测：中文标题变成 None，所有 CJK 标题键相同；`α-synuclein` 和 `β-synuclein` 归一后相同；`Étude` 变成 `tude`。dfc 的 `norm`（arxiv_snapshot）也是同一套规则 | NFKC + casefold，保留 Unicode 字母和数字（`\w`），希腊字母转写或保留；空键不参与匹配 | 小 |
| registry：批量写入性能（R2） | 需重构 | 每个方法都是 `with self._connect()`，即每次调用一个连接、一次 commit（:2657-2661）。没有 WAL，没有 busy_timeout，`synchronous` 用默认值。实测：<br>10k 篇（DOI 路径）100.9 s，10.1 ms/篇；<br>100k 篇 1,095 s，10.95 ms/篇，外推 **100 万篇约 3.0 h、870 万篇约 1 天**；<br>无 DOI 走标题路径 10k 篇 141 s（14.1 ms/篇），每次都扫一个年份分区（`COALESCE(published_year,year)=?` 无索引，1M 规模实测 187 ms/次），1M 外推至少 52 h（线性下界，实际是超线性）。<br>对照：同样的表用 WAL + 每批 2 万条 `executemany`，**1M 篇 + 2M 标识用了 117 s**，快约 90 倍 | 新增 `library.bulk_import(rows)`：分批、单事务、预解析标识冲突；连接统一开 WAL、`busy_timeout=30000`、`synchronous=NORMAL`；单条 upsert 只留给交互路径 | 中 |
| registry：查询与全表扫描（R2） | 需优化 | DOI 和标识的点查有唯一索引，100k 规模下 0.36–0.38 ms。全表扫描的地方：<br>`list_papers()` 100k 规模 1,409 ms，而 `acquire_by_doi`/`acquire_by_title` **每次调用都要先跑一遍**来判断"是不是已有身份"（acquisition.py:347-349、:487-489），`pdf_resolver.py:234-239` 也一样；<br>`list_papers(query=)` 是 `LIKE '%q%'`，100k 规模 52 ms，1M 外推约 0.5 s；<br>`paper_identifiers(paper_id)`、`pdf_assets(paper_id)`、`processing_jobs(paper_id)`、`extraction_runs(paper_id)`、`paper_compile_states(level,kb_name)` 都没有索引（EXPLAIN 都是 SCAN；在 2M 标识上按 paper_id 查要 175 ms）；<br>`list_provenance_events(paper_ids)` 取出整张表后在 Python 里做子串过滤（:2431-2453）；<br>`find_duplicate_asset_diagnostics` 会对全部 pdf_assets 做标题比较（:507-523）；<br>`paper_readiness` 一次调用十几个查询，`list_papers(readiness filter)` 再对每篇调用一次（:1008-1020） | 补上上述 5 个索引；标题检索改走 FTS5（`norm_title`）；"是否已有"直接用 upsert 的返回值判断；provenance 加 `paper_id` 列和索引；readiness 改成 SQL 视图 | 中 |
| registry：并发写（R7） | 需重构 | 4 个进程各写 300 条，无错误（18 s），因为写入很短。但**一个读事务持续 8 s 时，写入在 5.5 s 后报 `database is locked`**（rollback-journal 模式下读者阻塞写者）。`store_pdf_asset` 是先 SELECT 再拷贝文件再 INSERT（:380-418），没有 BEGIN IMMEDIATE，两个进程同时写同一个 sha 时，一个会拿到 UNIQUE 异常，文件已经拷了一半 | WAL + `BEGIN IMMEDIATE` + busy_timeout；资产先写到 `.part`，INSERT 成功后再原子改名；写入集中到单个 writer 进程（获取 worker 只提交结果） | 小 |
| registry：`merge_paper_assets` | 需重构 | C9 实测：合并后 `paper_references` 还留在源论文上，没有迁移；`paper_compile_states`、`extraction_runs`、`paper_citations`、`external_artifacts` 也都不迁移；目标论文的 `published_year` 不取更早的值。文件路径保留在源论文目录下（:612）。DOI 冲突只处理"目标没有 DOI"的情况。只有 API 路由调用它，CompileScholar 里没有调用方 | 合并要覆盖所有挂 paper_id 的表，并在 `paper_aliases(old_id→new_id)` 里留下别名（派生库和工具按别名解析，旧 id 不会失效）；日期取 min；在一个事务里完成 | 中 |
| registry：`paper_compile_states` / `paper_compile_log` | 需重构 | 主键是 `(paper_id, kb_name)`，只有一个 `level ∈ {none,shallow,deep}`（:2179、:3692-3702）。新系统的抽取状态是"每篇 × 每遍（self-T1/T2、result、other）× prompt_sha × 模型"，还要区分成功、失败、可重试。`set_compile_state` 不检查 status，失败也能写成 deep。0 行，没有调用方 | 新表 `extraction_state(paper_id, pass, prompt_sha, model, status{ok,failed,retry}, attempts, last_error, next_retry_at, output_ref, updated_at)`，主键 `(paper_id, pass, prompt_sha)`。只有 ok 才算完成；failed 按指数退避重试，最多 N 次。`compile_log` 的思路保留，作为状态变更日志 | 中 |
| registry：`extraction_runs` | 需重构 | 没有 `pass`、`prompt_sha`、`model`、`attempts`、`started_at` 字段，`output_artifacts` 必须是库内路径（:2596-2598）。0 行。唯一的写入方 `coarse_store.TierStore` 只被 legacy 调用，而且它把 COARSE 无条件写成 `ready`（coarse_store.py:94-101） | 并入上面的 `extraction_state`，或者作为它的运行明细表（每次尝试一行） | 小 |
| registry：资产与解析产物（R8） | 需优化 | `pdf_assets` 有 sha256、size、source_kind、source_uri、license（:3508-3520），好用。`processing_jobs` 有 `tool_config_hash`，但 hash 里**没有解析器版本**（processing.py:35-38）；`mineru_artifacts.checksum` 和 manifest 有用。`register_mineru_artifact` 要求文件放在库根目录内 | 解析产物加 `parser_name`、`parser_version`（MinerU 返回 `parse.parser_version="4.0.5"`）、`tier`、`output_formats`、`router_job_id` 字段；图片资产单独存 sha256 | 小 |
| registry：死表 / 死方法 | 退役 | **死表**（sci-evo 里 0 行，或新系统用不到）：`upstream_asset_links`（Sci-MKG 上游）、`paper_collections`/`_members`、`api_jobs`（UI 作业，1,290 行，只有 API 用）、`remote_parsed_assets`（8 行，Sciverse AI-ready）、`external_artifacts`（1 行，LogicKG 超图）、`paper_citations`（0 行）。<br>**两个仓库都没有调用方**：`find_pdf_asset_by_sha256`、`paper_identity_chain`、`list_external_artifacts`。<br>**只有 sci-evo API/CLI 在调**：`merge_paper_assets`、`backfill_paper_year_from_candidates`、`delete_paper`、collections/tags 系列、`paper_sciverse_ai_ready`、compile_state 系列、api_jobs 系列。<br>3,806 行里，约 40% 是 Sciverse AI-ready / readiness / manifest 的 UI 辅助函数（:1447-1943、:3009-3433） | 按设计 §2 拆成 `library/{identity,assets,parse,state,provenance}.py`；死表不建；UI 辅助逻辑放到 service/，基于 SQL 视图重写 | 中 |
| registry：`evidence_units` 表 | 退役 | 新系统把全文单元放在派生库（设计 §1：派生库可重建），而这张表在权威库里。46,174 行单元的模型过时（见 evidence_units 一行） | 不迁移；documents/ 的派生表取代它 | 小 |
| `LocalDoiArchive.lookup`（Sci-Hub） | 需优化（改一行，收益最大） | 用 `WHERE lower(items.doi_norm) = ?`（sources.py:138），EXPLAIN 结果是 **SCAN items**（8,697 万行，13.6 GB），实测**单次查询 22.8 s**。索引里 `doi_norm` 本来就是小写且已解码：随机 2 万条中大写 0 条、含 `%` 0 条，所以 `lower()` 完全多余。去掉后走 `idx_items_doi_norm`，**20 个随机 DOI 中位 1.3 ms、最大 3.6 ms，20/20 命中**；从 UNC 共享盘解 zip 取 PDF，中位 0.10 s、最大 0.20 s，**20/20 是有效 PDF**（样本跨医学、材料、地学、经济、生物、电力，大多没有 arXiv）。另外每次查询都新开连接并重新做 schema 自省（:121-132），`with sqlite3.connect` 不关连接；索引用读写方式打开（应改为 `mode=ro`）；`_lookup_generic_hit` 会扫所有表（:151-177） | 去掉 `lower()`；连接复用并用只读 URI；schema 只探测一次；DOI 先经统一的 `core/ids.normalize_doi`。注意：用 `%28…` 编码、`http://www.doi.org/` 前缀、末尾 `.` 这类写法查询，在现行 normalize_doi 下实测**全部 miss**；`10.1016%2F…` 也 miss。改完规范化后要复测 | 小 |
| `LocalDoiArchive.extract_pdf` | 需优化 | 只检查 `%PDF-` magic（:113）；`write_bytes` 不是原子写（:115）；不算 sha256，也不记 zip 的 crc 和 size（索引里本来就有 `items.crc/size` 可以对照） | 解出后校验 crc 和 size，计算 sha256，原子写入；来源记录写 `{archive_rel_path, inner_path, crc, index_built_at}` | 小 |
| sources.py 的各 API 客户端 / citation_graph / refgraph | 需重构（以 compilescholar/sources 为底合并） | OpenAlex、S2、Crossref、Sciverse、arXiv 各有 2–4 份实现。限流和熔断只在进程内有效，或只在本机跨进程（`~/.pace_*`）；读 Retry-After 的只有 S2 和 arxiv_oai；有 7 个各自写的重试循环；失败被当成完成（refgraph 没配 key 时把空路由永久缓存、只拿到部分结果也缓存、prefetch 在请求之前就做标记；citation_graph 把 error 行缓存 7 天）。详见子审查 A | 统一 http 层：按主机的跨进程 pacer、Retry-After、熔断（429 不计为故障）、transient 不进缓存；客户端各保留一份；mailto、速率、UA 移到 configs/ | 中 |
| `acquisition_chain.acquire_fulltext` | 需重构（骨架可留） | 通道顺序是 arxiv → oa → sciverse → local（:293-298），没有 arXiv HTML。`_ensure_env` 按"上溯四级目录"找 `.env`（:310），在 CompileScholar 里解析到 `Desktop\.env`，**实测不存在**，于是 Sci-Hub 通道静默返回 "not configured"。下载直接写到最终路径；全局替换 `_download` 不是线程安全的；没有节流也没有重试。详见子审查 B | 首通道用 `sources/arxiv_html`；路径走 core/paths，配置走 configs；`.part` + 原子改名；每次尝试都记录 sha256、url、通道、HTTP 状态 | 中 |
| 身份核对 `verify_pdf` / `titles_match` | 需重构 | `acquisition_chain.py:71-84`、`:52-58`：<br>pdftotext 返回空（扫描件）时**直接放行**；<br>词表只取 `[a-z]{3,}`，`CO2 capture in MOFs` 实测判为不匹配，中文标题必然不匹配；<br>匹配阈值是 `max(2, 60%)` 个词，用的是子串匹配，实测 "Stroke outcomes" 对上一篇无关的康复综述、"Deep learning" 对上 protein folding 论文、`α-Synuclein` 对上 `beta-synuclein`、"Granular flow on inclined planes" 对上一篇湍流论文，**都判为 True**；<br>registry 那条路径（`acquire_by_doi`）完全不做身份核对；<br>pdftotext 依赖 Git for Windows 自带的版本，项目里没有声明 | 核对改成两层。① 确定性的层：DOI 写在首页 / XMP 里，或者标题 token 覆盖率高、且 Unicode 规范化后作者姓氏也命中 → 判通过。② 介于两者之间的交给 LLM 裁定。PDF 文本用 MinerU 输出的 `doc_title` 块（实测两篇都能正确抽出），不依赖 pdftotext | 中 |
| 隔离 | 需重构 | docstring 写的是 "quarantined (deleted)"（:17），实际就是 `dest.unlink()`（:146、:204、:211、:286），既不保留文件也不留记录 | 不匹配的 PDF 移到 `library/quarantine/<sha256>.pdf`，并写一条 `asset_attempts(status=identity_mismatch, observed_title, expected_title)`，可以人工复核或 LLM 复判 | 小 |
| `acquisition.py` 的其余部分 / `pdf_resolver.py` / `search_resolve.py` / `process_workflow.py` | 退役 | 本地档案在远程之前尝试（acquisition.py:361-408），顺序和设计相反；远程 PDF 不做身份核对；`acquire_by_title` 每次调用都会写 papers 和 candidates（:490-499）；pdf_resolver 的 unpaywall 分支没有对应 client，永远 miss；search_resolve 只做 dry-run，链接指向 sci-evo 的 `/api/library`；process_workflow 是 UI 状态机。CompileScholar 里都没有调用方 | 删除；sha256 去重和 provenance 事件的思路并入 acquire/ | 小 |
| MinerU 客户端（`_see_upstream.py:548-958`、`processing.py`） | 替换 | 用的是同步 multipart 的 `/file_parse`（:621-631），期望返回 zip 或 `results.*.content_list`。9605 Router 上没有这个端点，openapi 实测只有 `/v1/*`。另外还有两套 legacy 客户端（`parse_mineru.py:54`、`promote_to_deep.py:93`），一共三套。作业配置 hash 里没有版本；单次同步请求，超时 600 s | 新写 `sources/mineru.py`：uploads → PUT → complete → `POST /v1/parse/jobs`（**返回 202，字段是 `job_id` 不是 `id`**）→ 轮询 → `GET /v1/files/{id}/content`。实测注意：要绕开系统代理（ProxyHandler({})）；job 里 `parse.parser_version` 要记进来源记录；按 `(pdf_sha256, tier, formats, parser_version)` 做幂等键 | 中 |
| MinerU 输出格式（R4，新发现） | — | 设计里写的"只有 markdown、内嵌 base64"不完整。`/v1/health` 实测 `output_formats: [markdown, middle_json, structured_content, zip]`。`structured_content` 实测按页给出 `blocks[{type, bbox(相对坐标 0–1), content}]`：<br>类型有 doc_title / paragraph_title(level) / text / table（HTML，含 rowspan/colspan）/ chart(captions, image_source=base64) / ref_text / header / footer / page_number / page_footnote；<br>metadata 里有 `page_count`、`parse_mode=ocr` 和每页的 `width_pt/height_pt`。<br>两次实测：<br>① 1994 年 Stroke 期刊，8 页 OCR 扫描件，standard 档，服务端用时 7.9 s，md 95,946 B，其中 base64 占 51,967 B（54%）；<br>② 1987 年药理期刊，15 页，8.5 s，17 张 chart，md 里有 17 张 base64 图。<br>两篇都没有公式，**equation 块的形态未验证**。`middle_json` 在这个版本里结构是 `{pages, schema, ...}`，不是旧的 `pdf_info` | **documents/ 以 structured_content 为主要来源**（页码、bbox、表格 HTML、图注和图片、参考文献块都有），markdown 只作阅读视图。header / footer / page_number 不进单元。base64 先剥离，按 sha256 落盘为图片资产 | 中 |
| `evidence_units.py`（content_list 分块） | 退役（page_idx 定位思路移植） | 只读旧格式 content_list（:132-133）；bbox 没有放进 locator（:148-152）；`table_caption` 是 list 时会被 `first_text` 跳过（:429-436）；chart 不算视觉块（:388）；header 和页码会混进证据；KG / Sciverse 那几类来源属于旧系统（:186-289）。新 Router 已经不产 content_list | 删除；按 structured_content 的块模型在 documents/ 重写 | 小 |
| `documents/units.py`（新的单元模型） | 需优化（作为统一模型保留） | 单元只有 `uid/kind/section/text/start`（:28-35），没有 page 和 bbox；uid 用 `arxiv_id`（:48）；只认 `#` 标题和 `<table>`。子审查 B 用实测 markdown 跑过：base64 图片变成 para，`Fig. 1 \|` 不识别为 caption，pipe 表变成 para，短公式被 20 字下限滤掉 | Unit 加 `page`、`bbox`、`block_type`、`asset_ref` 字段；新增 `from_structured_content()`（块到单元一对一：table / figure / caption / equation / para / section）；uid 改用 paper_id；markdown 和 HTML 两条路径继续保留给 arXiv HTML 来源 | 中 |
| `documents/build.py` | 需优化 | 来源只有 ideaforecast parquet 和 arXiv HTML（:56、:71）；必须在 papers 表里有 `v1_date` 才能入库（:62-65），**非 arXiv 论文根本进不来**；docs 表没有 sha256 和解析器字段 | 输入改成"registry 的 paper_id + 解析产物"；docs 表加 `source_asset_sha256`、`parser`、`parser_version` 字段 | 中 |
| `corpus/papers.py`（104.6 万篇 arXiv 元数据） | 需重构（变成导入器） | 主键是 `arxiv_id`（:44）。实测：169,209 篇（16%）有 DOI；69,121 条 DOI **不是小写**；230 条是多个 DOI 用空格拼在一起（如 `10.1109/TIT… 10.1109/ISIT…`）；358 组 DOI 重复；入库时不做任何 DOI 规范化（:63-66、:75-77）。80% 是 cs.*。`v1_date` 是天精度（100% 长度为 10），可以直接作为 `first_public_date(precision=day)`。`resolve_title` / `title_prefix` 只覆盖 arXiv | 改成 `library/import_arxiv.py`：读同一个快照加 OAI，经统一规范化后批量写入 registry（arXiv 标识 + DOI 标识，多 DOI 拆开，标记为 `role=published_version`）。papers.sqlite 退化为 OAI / 快照的缓存，不再承担身份职责 | 中 |
| `citations/resolve.py` 标题解析 | 需重构 | 只解析到 arXiv id（resolve.py:5-9）。`title_year` 的启发式在非 CS 引文格式上实测：<br>APS / ACS 风格本来就没有标题，取出的"标题"是 `86, 5886 (2001)`、`2010, 132, 1234–1240`；<br>Nature 风格把作者串当成标题；<br>Vancouver 风格（医学）把 "Smith J, Jones K. Effect of…" 整段当成标题；<br>带 LaTeX 的标题（`La$_3$Ni$_2$O$_7$`）只取出 `S`；<br>中文 GB/T 7714 格式因为 `norm` 只保留 ASCII 而失效；<br>旧式 arXiv id 不识别；<br>Vancouver 条目里明明有 `doi:10.1056/…`，这一层也没用上（DOI 阶段在 skeleton 里，没接进来）。<br>前缀 40 字符要求标题开头一致，年份窗 −1…+4，姓氏要求出现在条目里，这三条对 CS 是合理的；但非 CS 的大部分引文没有标题，前缀法从源头上就用不了。旧 DOI + OpenAlex 阶段的解析率 79.0% 是 **CS 语料上测的，条件不同** | 解析顺序：① 条目里的 DOI 或 arXiv 标识（统一的 ids 规则）；② **结构化引文解析**：期刊缩写 + 卷 + 页 + 年 → Crossref `query.bibliographic` 批量查询，或用 OpenAlex 的 filter 批量查询（APS / ACS / Nature 这类无标题格式只能走这条路）；③ 文库内的 norm_title FTS 匹配（Unicode 规范化后），并核对年份窗和姓氏；④ 剩下的才是 stub。被引对象统一解析到 paper_id；只有 DOI 的直接建为"仅元数据"的论文 | 大 |
| `core/cutoff.py` 与 as_of | 需重构（统一成一份 `core/asof.py`） | 见 §2 | 见 §2 | 中 |

## 2. 截止语义：现状是 4 套规则，统一方案

**现状**（全部来自读源码，外加一项字符串比较的实测）

| 位置 | 输入 | 粒度 / 包含性 | 缺少日期时 |
|---|---|---|---|
| `core/cutoff.py:40-63` | `KNOWLEDGE_CUTOFF=YYYY-MM`（env 或线程局部变量） | 月份，排除截止月；只有年份时，同年一律排除（保守） | 排除 |
| `retrieval/sources.py:34-43` `_knowledge_cutoff` | 读 `sys.modules["cutoff"]`，也就是 legacy 的 `_shared/tools/cutoff.py`，不是 core | 推到 Sciverse 时是 `year ≤ Y-1` | 由 `apply_discovery_filters` 处理，见下 |
| `acquisition.apply_discovery_filters`（acquisition.py:1135-1138） | 同上 | 年份 | **放行**（和 core 的规则相反） |
| `sources/sciverse.py:208-210`、`baselines/harness/mcp_server.py:182-196`、`kb/store.py:88-102`、`answer/pipeline.py`（v9b 路径） | core.cutoff | 年份 / 月份 | 排除 |
| `cognition/asof.py:16-30`、`index/build.py:134,148`、`tools/mcp_server.py:20-22`（`DFC_AS_OF`） | `YYYY-MM-DD` | 天，**包含当天**，用字符串比较 `v1_date <= T` | 不可见 |
| `answer/pipeline.py:379-386` `as_of_date` | 把 `YYYY-MM` 转成"截止月前一天" | 唯一一处在两种语义之间换算 | — |
| `kb_compiler/views/tools.py:880` `as_of(year)` | 年份 | 旧栈 | — |

**冲突点**
- **用字符串比较时，精度不同会出错**：实测 `'2015' <= '2015-01-01'` 为 True，即只有年份的日期会被当成当年 1 月 1 日之前；而 `'2015-06' <= '2015-01-01'` 为 False。新系统一旦引入月、年精度的日期（非 arXiv 论文大多只有年份），AsOf 和 index 的字符串过滤会**让只有年份的论文在当年第一天就可见**，造成泄漏。
- 现在的 `allowed()` 只在 `cm > 12` 时才放行同年（:63），同年任何月份都排除；as_of 按天时，没有对应的规则。
- 同一次评测里，外检走"年份 ≤ Y-1"（Sciverse 服务端），库内走"按天 ≤ T"，两者的可见集合不一致。
- 截止日期通过 env 和线程局部变量传递，`cli.py:56-57` 会改写 `os.environ`；建库阶段如果同一进程里设了这个 env，会泄漏进 Sciverse 的通道（子审查 B 的 R9 风险）。

**统一方案**
1. 只保留一份 `core/asof.py`：`AsOf(T: date)`，按天、包含当天。各来源的日期统一存成 `(date_lo, date_hi, precision)`：日精度 lo=hi；月精度为当月第一天到最后一天；年精度为 1 月 1 日到 12 月 31 日。
2. **可见性规则**：`date_hi ≤ T` 为确定可见；`date_lo > T` 为确定不可见；落在中间（精度不足、无法判定）的**默认排除**，计数并在结果里报告（沿用现行 core 规则"无法判定就排除"的保守立场）。没有日期的一律排除。
3. 所有 SQL 和 numpy 过滤只比较 `date_hi`（统一的 10 位 ISO 日期），不再直接拿原始日期字符串比较。
4. 外部适配器（Sciverse、OpenAlex、Crossref、S2）**由调用方显式传入 AsOf**，在适配器内部换算成该服务能用的最细粒度：Sciverse 只能按年，就推 `year ≤ year(T)−1`，再按 date_hi 在客户端二次过滤；OpenAlex 可以推 `to_publication_date:T`。不再读 env 或线程局部变量。
5. 基准的截止由 `eval/adapters/<bench>` 换算成 AsOf，例如 CS2 `inserted_before=2025-05` 换算成 T=2025-04-30，并写进 MCP server 的启动参数（沿用 `DFC_AS_OF` 的做法，模型不能修改）。
6. 需要改的调用方：`kb/store.py`、`sources/sciverse.py`、`baselines/harness/mcp_server.py`、`answer/pipeline.py`（两条路径）、`cli.py`（删除写 env 的那段）、`cognition/asof.py`（换成区间规则）、`index/build.py`（存 date_hi）、`retrieval/sources.py` 和 `acquisition.py`（随退役一起删掉）。**v9b 冻结路径**为了保持表征一致，保留一个 `legacy_month_rule` 适配器，即 `AsOf.from_month_exclusive("2025-05")`，再按 v9b 的规则做年份判定，靠表征测试锁定结果一致。

## 3. 关键实测数字一览

| 项 | 数字 | 条件 |
|---|---|---|
| registry 单条 upsert（有 DOI + 2 个标识） | 10.1 ms/篇（10k）、10.95 ms/篇（100k） | 本机 SSD，journal=delete，每条一次 commit |
| registry 单条 upsert（无 DOI，走标题路径） | 14.1 ms/篇（10k），随规模增长 | 同上 |
| 外推 100 万篇（现行 API） | ≥ 3.0 h（有 DOI）、≥ 52 h（无 DOI，线性下界） | 线性外推 |
| 批量导入（WAL + executemany，每批 2 万条） | 1M 篇 + 2M 标识共 117 s，库 0.49 GB | 同一套表结构 |
| DOI / 标识点查 | 0.36–0.38 ms（100k）、0.04 ms（1M，批量库） | 有唯一索引 |
| `list_papers()` 全表 | 53 ms（10k）、1,409 ms（100k）、367 ms 只取 id（1M） | acquire 每次调用都会触发 |
| 读者持锁 8 s 时写入 | 5.5 s 后报 `database is locked` | journal=delete，默认 timeout |
| Sci-Hub `lookup`（现行代码） | 22.8 s/次 | 8,697 万行，SCAN |
| Sci-Hub 走索引查询 | 中位 1.3 ms、最大 3.6 ms，20/20 命中 | 去掉 `lower()` |
| Sci-Hub 取 PDF | 中位 0.10 s、最大 0.20 s，20/20 是有效 PDF | UNC 共享盘 |
| MinerU 9605 standard 档 | 8 页 OCR 扫描件 7.9 s；15 页 8.5 s（服务端 duration_ms） | 两篇非 CS 老期刊 |
| MinerU markdown 里 base64 的占比 | 54%（8 页那篇）；15 页那篇有 17 张图 | — |

## 4. 之前被默认可用、实际不满足要求的点

1. **"registry 已经是 DOI 优先的身份权威"**：实际上 DataCite 预印本 DOI 成了 53% 论文的主键，同一个 DOI 的 7 种写法会拆成 4 篇，同一个 arXiv id 的 5 种写法拆成 3 篇，预印本和正式版在常见场景下（年份差 1、arXiv id 带版本号、先导入正式版）不会合并，两篇不同的 "Editorial" 会被合成一篇。
2. **"标识表可以直接迁移"**：sci-evo 里 13 篇论文挂着别人的 DOI（检索候选被当成本篇的标识写进去），而且这种污染会让后来入库的真正拥有者查不到自己。迁移前必须清洗。
3. **"有 published_date，可以做最早公开日期"**：实际是先写者赢，没有精度字段，格式也不统一。
4. **"registry 能承载百万级"**：每条一个连接、一次 commit，没有 WAL，多个 `paper_id` 列没有索引，acquire 每次调用都扫全表。现行 API 导入 100 万篇需要 3 h 以上，870 万篇需要 1 天以上；批量路径约 2 分钟。
5. **"compile_states 已经为 CompileScholar 扩展好了，可以承载分层状态"**：只有一个三档 level，不区分遍、prompt、成功或失败，0 行，没有调用方。
6. **"Sci-Hub 本地库已接入"**：在 CompileScholar 里因为 `.env` 路径错，被静默关闭；接上以后每次查询要 22.8 s。改一行之后是 1.3 ms。
7. **"获取链有首页身份核对，不匹配会隔离"**：扫描件直接放行，非 ASCII 标题必定判失败，短标题和泛化标题会误放行，registry 那条路径根本不做核对，所谓"隔离"其实是直接删除。
8. **"MinerU 只给 markdown"**（设计 §1、§2 的前提）：Router 还能给 `structured_content`，里面按页有块级 bbox、表格 HTML、图注和图片。documents 应该以它为主要来源，而不是从 markdown 里反推结构。旧客户端（三套）一律不能用。
9. **"evidence_units 的 content_list 分块可以并入 documents"**：新服务已经不产 content_list，而且旧分块会丢 bbox、丢图注、把页眉页码混进证据。
10. **"citations 补上 DOI 和 OpenAlex 阶段就够了"**：非 CS 引文（APS、ACS、Nature 无标题风格）的标题提取本身就失败，必须加一条按期刊、卷、页、年的结构化查询路径，这项工作量大，不只是"补两个阶段"。
11. **"cutoff 是单一实现"**：一共有 4 套规则（legacy 的 cutoff 模块、core 的月份规则、AsOf 的按天字符串比较、旧栈的年份规则），而且在缺少日期时的处理方向相反。字符串比较遇到年精度日期会泄漏。
12. **"sci-evo 的测试覆盖了 retrieval"**：测的是 sci-evo 自己的那份拷贝。CompileScholar 的 tests 目录里一个 retrieval 测试都没有。

## 附录 A：来源层子审查要点（sources / citation_graph / refgraph / circuit）

- **DOI 规范化有 3 套规则**。
  - registry 版：保留 `www.doi.org`、末尾 `.`、括号；把 `arXiv:` 当成 DOI。
  - skeleton 版（`bib.DOI` + lower，bib.py:31、resolve.py:90）：会在 `;` 处截断 SICI DOI，把 `;2-g` 丢掉。
  - refgraph 版：只做 replace（refgraph.py:214、:372），大小写敏感。
  - 三套都不解码 `%xx`。`corpus/papers.py` 入库时不做任何规范化。
- **arXiv id 规范化**：四个正则（resolve.py:20、bib.ARXIV、refgraph `_ARXIV_IN_REF`、kb_compiler manifest.py:81）都只认新式 `\d{4}\.\d{4,5}`。`…/pdf/…v2.pdf` 在 bib 和 refgraph 两处都解析不出。`s2_paper_id_param` 把带版本号的 id 或整段 URL 直接当作 S2 的 paperId 发出去。
- **OpenAlex**：两份实现。
  - A（sources.py:180-268）逐个 DOI 请求，不读 Retry-After。
  - B（refgraph.py:194-247）按 50 个 DOI 或 W-id 一批，key 从 secrets 取，有 credit 守卫，429 时把剩余额度置 0。
  - 结论：以 B 为底，移植 A 的 `fetch_by_doi` 和 `pdf_candidates`。
- **S2**：A 读 Retry-After，支持分页和 contexts/intents（sources.py:294-321、:480-514）。B 只取单页、没有 key，429 后冷却 1,200 s，冷却状态写在 `~/.pace_s2_cooldown.json`。结论：合成一份。
- **Crossref**：两份实现，mailto 都是占位邮箱（example.org/.com）。
- **Sciverse**：`compilescholar/sources/sciverse.py` 是从 A 移植来的，保留这份；A 退役。B 还有三处要修：429 不读 Retry-After（:174-176）；令牌拿不到也照发（:160）；`capacity=rate`，第一分钟可以发出 2 倍额度（:148）。
- **arXiv**：同一主机上有 3–4 个节流器互不协调。
  - A 的 `ArxivClient` 用进程内全局变量节流，不加锁（:522-531）。
  - refgraph 用文件锁，间隔 3.1 s / 15.5 s。
  - `arxiv_html` 用线程锁，只在进程内生效（:279-305）。
  - kb_compiler 的 `_arxiv_batch` 完全不节流。
  - 另外，`arxiv_snapshot` 的默认路径写死成一个 UNC 路径（:20）。
- **失败被当成完成**：
  - refgraph 在没配 OpenAlex key 时，把空路由永久缓存（refgraph.py:196-197、:432-434）。
  - 只拿到部分结果也照样缓存，`test_refgraph_cache.py:50-56` 是刻意这样断言的。
  - prefetch 在发请求之前就把标题记为已取（:351），后面失败了也不会再单篇重取（:416）。
  - citation_graph 把 `[{"error":…}]` 当作数据缓存 7 天（:180-189、:217-229）。
  - arxiv_html 遇到 404 时写 `.none`，以后永不重试（:311-317）。
- **R6 路径与配置**：
  - refgraph 缓存目录由 env `CS_REFGRAPH_CACHE` 决定，默认在 legacy_bench，而且 import 时就会 makedirs（:30-31）。
  - 节流状态写在 `~/.pace_*`；Sciverse 共享桶写在 `~/.sciverse_bucket.json`（pipeline.py:33）。
  - configs 里只有 `sciverse_max_wait_s` 一项；速率、mailto、重试次数、UA 都散落在代码或 env 里。
- **R9 截止与回写**：refgraph 适配器里没有截止逻辑，由调用方事后过滤（pipeline.py:109、:121）。评测时还会写共享的 `refgraph_cache`，后续答题会读到这份缓存。缓存内容本身与截止无关，但不符合"截止由适配器决定、评测不回写"的纪律。

## 附录 B：获取层子审查要点（acquisition / chain / resolver / processing / evidence_units）

- 七个模块在 `src/compilescholar` 和 `tests` 里都没有调用方，只有 legacy 脚本通过 `src/sci_evo_extract.py` 垫片在用。被新代码间接用到的只有 `acquisition.discover_tiered`（经 search_service.py:26、:98）。
- `acquisition_chain` 的下载没有 `.part` 临时文件（:87-100）；全局替换 `_download` 不是线程安全的（:343-345）；没有节流，也没有重试。
- MinerU 失败时会新建一条 failed 作业（id 里带 uuid，registry.py:1270），复用时只认 ready（:1300-1323）。所以失败可以重试，不会被当成完成，但重试没有上限，也没有退避。
- `processing.py` 的作业 hash 只包含旧表单字段和 URL（:35-38），不包含解析器版本。
- `UpstreamConfig.from_env` 用 `load_dotenv` 改写了 `os.environ`（_see_upstream.py:84-86），和 `core/secrets.py` 并存，成了第二个 env 读取器。
- `apply_discovery_filters` 对没有年份的候选放行（acquisition.py:1135-1138），而 core 的规则是排除。
- 完全死掉的代码：`acquisition.first_safe_remote_pdf_candidate`（:1357）；`pdf_resolver` 的 `local_upload` 和 `unpaywall_pdf` 分支（:55-62）；`registry.update_source_candidate_status`（没有调用方，失败状态因此不会落库）。

## 5. 没有验证的部分

- MinerU 对含公式论文（equation 块的形态、LaTeX 是否保留）、`advanced` 档、大文件（> 50 页）、并发队列的表现：本次只做了 2 次解析，两篇都没有公式。
- `structured_content` 的块顺序是否就是阅读顺序（双栏版面）：看了样例，没有系统核对。
- Sci-Hub 档案在**多进程并发**读 UNC 时的吞吐和失败率：只测了单进程串行 20 篇。
- 合并方案里 LLM 裁定的准确率：设计阶段待定，本次没有调用 LLM。
- 子审查 A（来源层）和 B（获取层）里标为"按代码推断"的条目，比如旧式 arXiv id 在 `arxiv_html` 里拼路径会失败：没有逐条实测。
