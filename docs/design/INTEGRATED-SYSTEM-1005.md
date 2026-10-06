# 一体化系统与工作区设计（INTEGRATED-SYSTEM-1005，v2.3）

> **v2.3（10-06 晚，阶段 B 实现与闸门）**
>
> 1. **文库建成**：`data/library/registry.sqlite`，约 119 万篇。
>    - arXiv cs 104.8 万篇，每个版本的日期都存；
>    - 四个基准语料：只入成员和元数据，不入题目和标签；
>    - sci-evo 480 篇，清洗后迁入。
>    - 写入时一律不合并：共享 DOI、标识冲突、标题相同都进合并队列（54,416 条），由两个模型裁定；两边都说"是同一篇"才合并。
>    - 两个不同的 arXiv 号永不自动合并。200 对审计里的错误合并全是这一类：同一作者的后续研究用了新号。
> 2. **取全文**：
>    - 按通道依次尝试：arXiv 分版本 NAS 镜像 → GCS 公开桶；只有 DOI 的走 OpenAlex OA → Sci-Hub 本地库。
>    - 身份核对先走确定性规则，拿不准的再交给 LLM。
>    - 资产只存指针；核对不一致的进 quarantine，不删除。
> 3. **文档两档**：
>    - 解析产物存进权威库 `parses` 表，按 PDF 的 sha256 存；
>    - 快速档 `documents/tei` 在 61 篇金标上召回 0.930、精度 0.930，表格行不算引用句；
>    - 精读档 `documents/mineru` 产出表格 HTML、公式、图注和参考文献表。
> 4. **版本感知** `documents/versions`：
>    - v1 全读，最新版只取增量，用模糊匹配比对。
>    - 74 篇双版本论文里，逐字不同的句子占 48%，真正新增的只有 13%。
>    - 190 篇实测：v1 的引用没有一条晚于 v1 的日期；最新版里晚于 v1 的 41 条参考文献，全部落在增量里。
> 5. **跨领域闸门**（`experiments/library/gate_b_crossdomain.py`）：从 Sci-Hub 索引按出版社前缀随机抽 24 个 DOI，覆盖 Nature、Nat Commun、Science、PRL、PRB、JACS、Angew、NEJM、Lancet、Cell Signal、JGR、Acta Mater。
>    - 22 篇走通了全流程：registry → acquire → 两档解析 → 文本单元。
>    - 2 篇如实判为不一致：一篇是 Nature 新闻页"无 PDF"的提示页；另一篇是 PRB 1979，原因是 Crossref 标题里带 MathML，已修。
>    - 较新的期刊论文（2002 年以后）引用句都正常。
>    - **1990 年以前的扫描件**（PRB 1979、JACS 1988/1991、Angew 1960s，以及 NEJM/Lancet 19 世纪档案）文本层有，但用的是上标数字引用，或者没有成形的参考文献表。GROBID 能拿到条目，连接却是 0。非 CS 领域纳入构建时，要补"上标引用识别"（PyMuPDF 的 span flags），或者对这类老论文只做他述，不做引用句。当前 CS 主线不受影响。
>    - Crossref 投稿日（received）的覆盖：Springer、Nature、ACM、Wiley 有；Elsevier、APS、CUP、IEEE 没有（256 篇里 39 篇有）。

> **v2.2 修订（10-06，阶段 B 第一件事的实测结果；改写 v2.1 第 2、3 条）**
>
> 1. **快速档改用 GROBID，再用 PDF 自带的引用链接补漏。** 不再自己适配 PyMuPDF 文本层的引用切分。
>    - 样本：2018–2025 年 cs 论文每年随机 9 篇，共 72 篇（固定种子）。其中 61 篇同时有 NAS 上的 v1 PDF 和 arXiv 的 v1 HTML。金标取自 HTML：LaTeXML 把每个引用做成指向条目的锚点，"哪句话引了哪条"是确定的。共 4,066 个正文引用对、2,424 条参考文献。脚本在 `experiments/documents/fasttier_{sample,compare}.py`。
>    - 结果（召回 / 精度 / 条目找全率 / 速度）：
>      - PyMuPDF + 现有切分：0.636 / 0.872 / 0.729，15/61 篇一条都没抽到；
>      - GROBID CRF 版：0.869 / 0.912 / 0.977；
>      - GROBID full 版（参考文献用深度模型）：0.873 / 0.913 / 0.994；
>      - **GROBID full + PDF 引用链接：0.934 / 0.919 / 0.994**，被引条目找全率 0.955，召回低于一半的论文只剩 2 篇。
>    - PDF 引用链接：71 篇里有 58 篇带 hyperref 链接（LaTeX 生成的）。按"链接落点最近的条目"认条目，对照落点处的文字核对，准确率 93.7%。规则是：带链接的论文以链接为准，不带的用 GROBID 自己的连接。
>    - 选 full 版不选 CRF 版：条目质量高一截（标题能解析到 arXiv 论文的比例 0.407 对 0.379），代价是慢。有一个已知的毛病：author-year 条目带方括号标签（`[Cao et al., 2024] …`）时，full 版偶尔会把标签单独切成一条（2,480 条里有 56 条不到 30 字，集中在 2 篇论文）。documents 层可以确定性地并回下一条。本机 14 路并发下，full 版约 0.8 s/篇（约 11 万篇/天），CRF 版约 0.29 s/篇（约 30 万篇/天）。吞吐成了瓶颈时再退回 CRF 版。
>    - 部署：`grobid/grobid:0.9.1-full`（digest `sha256:35c83afe…`），Docker 只绑 127.0.0.1:8071，不联网整合（consolidate 全关）。httpx 要设 `trust_env=False`，否则系统代理会把 localhost 请求劫走，返回 502。
>    - 精度的去向：抽查 40 个"误报"，约 17 个是表格或图里的文字（引用是真的，但不是正文句）；约 10 个其实是对的，金标匹配没认出来；约 4 个是句子碎片；真正连错条目的约 3 个。所以下一步要在 documents 层把表格和图的文字从句子里剔掉。
>    - 非 CS 论文（审查报告用过的 3 篇，取自本地 Sci-Hub 库，仅内部使用）：PRL 无参考文献标题，得到 30 条条目、22 句；Nat Commun 上标引用，得到 53 条、33 句，这两篇都正常。JACS 的 ACS 上标加字母子引用（`4a,4d`）只得到 5 句，而且连错条目；出版社 PDF 也没有引用链接。非 CS 进入范围时要补上标识别（PyMuPDF 能看到上标 span，这篇有 31 个）。
>    - 金标的局限：只覆盖有 LaTeX 源、arXiv 能出 HTML 的论文。72 篇中有 11 篇没有 v1 HTML，被排除。
>    - 顺带发现：现有 `citations/html.py` 只认 `bib.bibN`，有 4/57 篇 HTML 的条目 id 是 `bib.bibxN`，这几篇整篇解析为空。重建 citations 阶段时一起改。
> 2. **MinerU 9602 的实际吞吐远低于 18 路的标称。**
>    - 6 路并发：12/12 成功，172 页用 393 s，即 0.44 页/s，约 2,600 篇/天；单篇耗时 57–261 s。
>    - 18 路并发：worker 超时，6 个 GPU worker 中有 2–3 个被路由标为不健康；23 篇中 6 篇重试 3 次后仍然失败，每篇耗时约 25 分钟。我停掉客户端之后，路由队列仍然涨到 48，说明这台服务可能还有别人在用。
>    - 9602 用的是 vlm 后端，比 9605 慢：9605 单作业串行，flash 档 1.14 页/s，standard 档 0.84 页/s。
>    - 用户确认 9602 有别人在共用。
>    - 结论：精读档 9602 最多开 6 路，再加上 9605 一路作为补充，合计每天几千篇；共用时实际更少。深抽子集的规模按这个上限来定，客户端要看 `/health` 的排队数自动降速。

> **v2.1 修订（10-06，用户裁定 + 实测）**
>
> 1. **构建范围由基准驱动，不做全量离线构建。**
>    - 基准自带固定库的（ScholarCatalyst 19.1 万篇、MasterSet 6.8 万篇、PreScience 伴随集、IdeaForecastBench 10.9 万篇），以该语料为库全量抽取。四个语料并入同一个库，每个基准按自己的 scope 取子集。
>    - 开放式基准（CS2、DSB、Old Ideas、LimitGen）打开外检，复用已建好的领域认知，评测期不写回。
>    - 题目盲规则：库 = 基准的候选语料；深抽按库内被引数挑选；题面和金标一律不用。
> 2. **文档层分两档。**
>    - 快速档：PDF 文本层，用 PyMuPDF，每篇约 0.2 s。负责全库的引用句、v1 摘要、检索段落和版本差分。
>    - 精读档：MinerU。只处理深抽子集（表格、公式、版面）。
>    - 现有的引用解析在 PDF 文本层上只有 7/20 篇抽得出引用句，需要专门适配。GROBID 作为候选方案，各跑 50 篇对照。
> 3. **MinerU 的实际部署在 192.168.199.73。**
>    - 9605（V1 Router）：`max_concurrent_jobs=1`，作业内文件串行，实测 0.84–1.14 页/s。
>    - 9602（`/file_parse` + task router）：6 个 GPU worker × 3 路，共 18 路，不需要权限。
>    - 精读档走 9602，客户端按空闲槽位限流，遇到 409/503 排队重试。上一轮并发 16 路时 12 路返回 409/503，原因是没有限流，需要重测。
>    - sci-evo 旧客户端 `_see_upstream` 调的正是 9602 的 file_parse，判定从"替换"改为"需优化"（加限流、重试、版本记录）。9605 的客户端保留为备选。
> 4. **表述向量全量做。** 本地 embedding 实测 174 条/s，200 万条约 3 h。
> 5. 不再按截稿日期压缩范围或调整排期，投哪个会由用户定。
> 6. **阶段 E（服务与前端）移出当前计划**（用户 10-06）：先全力完成论文，前端网页放到论文之后单独做，用于改善用户体验。当前只走命令行和 MCP 工具；sci-evo 的 `api/` 与 `frontend/` 暂不迁入，仍在原项目里。

> v1（commit 778ac58）是在"已有模块大体可用"的前提下写的。四份适用性审查逐模块读了代码，并做了小规模实测，结论是这个前提不成立。v2 按审查结论重写。v1 的原文保留在 git 历史里。
>
> **依据**
> - 四份适用性审查（`docs/design/review/`）：
>   - FITNESS-LIBRARY-ACQUIRE：文库、来源、获取、解析、截止；
>   - FITNESS-EXTRACTION：3 篇非 CS 论文经 Sci-Hub 和 MinerU 实测，新旧抽取逐篇对比；
>   - FITNESS-RETRIEVAL-ANSWER：检索、工具、答题、评测；
>   - FITNESS-INFRA-COGNITION：LLM 客户端、配置、阶段契约、编译层，8 个契约探针。
> - 10-05 晚补充分析，共 17 条跨模块缺口。
> - 10-05 晚实测：
>   - Sci-Hub 各年份覆盖率；
>   - 期刊投稿日能否取到；
>   - NAS 上 arXiv 的分版本镜像；
>   - GCS 公开桶。
>
> **用户裁定**
>
> 10-05 已定：
> - 底座用 sci-evo，身份 DOI 优先；
> - Sci-Hub 用于系统内部建设，发表时不写这个来源；
> - MinerU 用 9605；
> - 旧系统逐项审视并升级；
> - 整个项目当一个完整系统来做，工作区要条理清晰；
> - 复杂语义交给 LLM；
> - sci-evo 的 API 和前端并入（只绑 localhost）；
> - 工作区的移动和冷存方案同意；
> - 旧包打 tag 后从 src 删除。
>
> 10-05 晚新定：
> - 阶段 E（服务与前端）挪到 F 之后；
> - CS 主实验不用 Sci-Hub，其他领域用；
> - 抽取 schema 的四项改动同意（§7.1）；
> - 编译层三项同意：shifts 改为时间窗加统计检验；新增依赖 leidenalg 和 usearch 或 faiss，均锁定版本；field_state 降为评测对照组；
> - 他述证据挂在说话方名下；
> - **v1 全文完整抽取；有后续版本的再抽最新版，只留新增内容，日期记最新版的日期**（§2）。
>
> 叙事（NARRATIVE-V9）与评测（EVAL-PLAN）不变。

## 0. v2 相对 v1 改了什么

1. **没有模块可以原样搬进来。** sci-evo 的 registry 要重构：
   - 身份合并有错，同一个 DOI 换种写法就被当成 4 篇论文；
   - 13 篇论文挂着别人的 DOI；
   - 日期是"先写先赢"；
   - 写入是逐条事务，100 万篇要 3 小时以上。

   新写的 dfc 主线也有问题：
   - 阶段契约改了代码再重建时一条都不重算，manifest 却被标为新鲜；
   - 编译层每次调用都现算，一次画像要 64 s；
   - 抽取在非 CS 论文上，主路径基本走不通。
2. **新增时间模型（§2），作为全系统的一节。** 版本泄漏已经在真实数据里坐实：6.6% 的施引论文引用了晚于自己 v1 的论文。日期有区间和精度，截止规则合并成一处。
3. **全文来源换掉。** CS 改用 NAS 上分版本的 arXiv PDF 镜像（v1 覆盖约 99%），2025-09 之后用 GCS 公开桶。不再依赖"HTML 加 IdeaForecast"，这两者都读的是最新版。非 CS 按年份分通道（§4）。
4. **文档层以 MinerU 的 structured_content 为主**，这种输出带页码、bbox、表格 HTML 和参考文献块。不再从 markdown 里反推结构。
5. **编译层离线物化**：按事件时间追加，as_of 查询是精确的；族存快照。LLM 裁定也只看日期不晚于生效日的证据（§8）。
6. **检索换成 tantivy 加二值码向量**，FTS5 和整表载入内存都撑不住真实规模（§9）。
7. **基础设施排在最前面。** 阶段契约、唯一的 LLM 客户端、配置与路径、ids、asof 都放进阶段 A，因为后面每一步的可信度都建立在它们之上。

10-05 晚那 17 条补充分析分别落在：

| 条目 | 位置 |
|---|---|
| 1 版本泄漏；4 日期精度 | §2 |
| 2 LLM 裁定与 as_of 冲突 | §2.4、§8 |
| 3 他述抽样偏早期 | §7.3 |
| 5 arXiv 自身的 DOI；6 id 稳定性；7 基准 id | §3 |
| 8 全文覆盖；9 存储；10 MinerU 吞吐 | §4、§12 阶段 B |
| 11 改一点就全量重跑 | §10.1 |
| 12 题目盲范围规则 | §13 待定 |
| 13 对照臂与消融开关 | §9.4 |
| 14 自引 | §7.3、§8 |
| 15 不带引用的提及 | §8（mention_link 覆盖全部句子） |
| 16 Sci-Hub 来源标记 | §2.5 |
| 17 备份 | §10.4 |

## 1. 系统全景

```
 sources ─► acquire ─► library（权威库 data/library/）
 arXiv 元数据/分版本 PDF   registry.sqlite：论文身份 · 标识 · 日期（区间+精度，各来源各版本）·
 GCS 桶 / arXiv HTML       资产指针 · 解析产物 · 抽取状态 · 来源记录 · 合并别名
 OpenAlex/Crossref/S2      papers/PPR_*/：解析产物（structured_content、markdown，已剥图）
 Sciverse / Sci-Hub
 MinerU 9605                                │ paper_id + 文本版本
                                            ▼
          派生库 data/derived/（可重建；阶段 manifest；条目级状态）
          documents ─► citations ─► extract ─► cognition（物化） ─► index
          单元/句号/     引用句→条目     T1/T2/结果/他述   身份·谱系·共引·     tantivy +
          页码 bbox      →paper_id      统一终检          接受·事实·变迁·族快照  二值码向量
                                            ▼
          tools（as_of 由服务端固定；本地/外部分标；预算记账；对照臂与消融开关）
             ├─► MCP（harness 臂）   ├─► answer（lit 模式 + 冻结 v9b）   ├─► eval   └─► service/web（阶段 E）
          grow（缺口诊断 → acquire → 同一条流水线；评测期不写回）
```

- 权威库只放不能重建的东西：身份、标识、各来源日期、资产指针、解析产物、抽取状态、来源记录。派生库都可以重建。LLM 产物虽然理论上能重建，但重建一次要几百 GPU 小时，所以它和响应缓存一起备份（§10.4）。
- 派生库和工具一律用 `paper_id` 作键。对外展示时，从标识表取 DOI 和 arXiv 号，不再从 id 里切字符串（审查发现有 17 处 `[6:]`）。

## 2. 时间模型（新增，贯穿全系统）

### 2.1 三种日期

| 日期 | 管什么 | 来源 |
|---|---|---|
| **首次公开日** `first_public` | 这项工作从哪天起能被领域看到。决定截至 T 时它在不在 | arXiv v1 提交日（arXiv 没有单独的投稿日，v1 提交日就是投稿日）；期刊取在线首发日（Crossref `published-online`，没有就取 `issued`）；多个来源时取最早的 |
| **投稿日** `received` | 谁先做出来，只用于判断先后 | Crossref 的 assertion 字段（Springer、Nature 有，Elsevier、APS 没有，覆盖率待测）；拿不到就留空，不用来判断可见性 |
| **文本版本日** `text_date` | 某条表述、某个引用句从哪天起存在 | 抽取时读的那一版全文的日期（arXiv 第 N 版的提交日；期刊正式版取在线首发日） |

### 2.2 日期区间与可见性

- 每个日期都存成 `(date_lo, date_hi, precision ∈ {day, month, year})`：
  - 日精度时 lo = hi；
  - 月精度取当月第一天到最后一天；
  - 年精度取 1 月 1 日到 12 月 31 日。
- 可见性规则：`date_hi ≤ T` 时可见；`date_lo > T` 时不可见；落在两者之间的**一律排除**，并在结果里计数。没有日期的也排除。所有过滤都只比较 `date_hi`，不再拿原始字符串比较。实测 `'2015' <= '2015-01-01'` 为真，会造成泄漏。
- 可见性对表述和返回对象都生效：一条表述要可见，它本身的 `text_date` 和它指向的对象的 `first_public` 都要 ≤ T。stub（还没解析出身份的被引文献）以首次被引的日期为准。
- 以上规则只有一份实现：`core/asof.py`。外部适配器由调用方显式传入 AsOf，适配器内部换算成该服务能支持的最细粒度：Sciverse 只能按年，同年的命中再解析到具体日期判断；OpenAlex 直接推 `to_publication_date`。不再用环境变量或线程局部变量。v9b 冻结路径保留一个 `legacy_month_rule` 适配器，结果由表征测试锁定。

### 2.3 版本：v1 全抽，最新版补抽增量

- **v1 全文完整抽取**，表述日期记为 v1 提交日。方法什么时候提出、论文什么时候首次可见，都以 v1 为准。
- **有后续版本时，再抽最新版**。只保留 v1 里没有的内容：新增的结果、改写过的局限、新增的参考文献和引用句。日期记最新版的日期。**中间版本跳过**，它们的新增内容会落在最新版的日期上，只会偏晚，不会偏早，所以不泄漏。
- 判断"v1 里没有"分两步：
  1. 先用结构信号找候选：句级对齐，加上引用条目的差集；
  2. 只有改写过的段落交给 LLM，判断是不是新主张。
- 元数据同样分版本存。OAI 缓存里的标题和摘要是最新版的，所以 T1 读的是 v1 全文里的摘要，不读缓存。各版本的日期从 arXivRaw 的版本列表取，`sources/arxiv_oai.py` 要改成把每一版的日期都存下来，现在只存了版本数。
- 期刊论文：预印本与正式版合并成一篇（§3）。预印本的文本按预印本日期记，正式版的增量按正式版的在线日期记。

### 2.4 LLM 裁定的生效日

每个 LLM 裁定只看日期 ≤ d 的证据，并以 d 作为生效日。查询截至 T 时，只用生效日 ≤ T 的裁定。
- 名字到论文的链接：候选只取提及日期时已经可见的论文。这样裁定结果与查询时间无关，每条链接只算一次。
- 两条说法是否等价或矛盾：生效日取两条说法里较晚的日期。
- 族的命名和边界：只用快照日期之前的证据。
- 认识变迁是否成立：只用事件日期之前的证据。

做不到的一点：模型的参数记忆本身可能带有后见之明。"原句必须存在"这条约束可以挡住大部分，剩下的在论文里写明。

### 2.5 来源标记

每份解析产物都记录 `source_channel`，取值为 arxiv_nas_v / arxiv_gcs / arxiv_html / oa_pdf / sciverse / scihub_local 之一。这个字段一路传到 documents、statements 和工具返回。CS 主实验与公开发布的数据只用不含 `scihub_local` 的产物，由派生库的过滤视图保证，并在 manifest 里统计各来源占比。

## 3. 身份

- **`paper_id` 一旦发出就不再改。** 生成规则：有正式版 DOI 就用它；只有 arXiv 的用 `arxiv:<id>`；两者都没有的用 `title:<norm_title>|<year>`；不再用随机 uuid。后来查到"首选 DOI"时，只改 `canonical_doi` 列，id 不变。
- **arXiv 自己的 DOI**（`10.48550/arXiv.*`）一律识别为 arXiv 标识，不当作独立 DOI。sci-evo 里有 260/490 篇的主键就是这种 DOI，迁移时要重新生成。
- **规范化只有一份 `core/ids.py`**：
  - DOI 依次做 URL 解码、去掉 `doi:` 前缀和任意 doi.org 前缀、去掉末尾标点和成对括号、转小写；
  - arXiv 号去掉 `arXiv:` 前缀，支持旧式 id（如 `math.GT/0309136`，子类保留大小写）；
  - 标题做 NFKC 和 casefold，保留 Unicode 字母和数字。

  全仓现有的 3 套 DOI 规则和 4 个 arXiv 正则全部删掉。上线前要用多种写法在 Sci-Hub 索引上复测命中。
- **先解析，再写入，合并走队列。**
  1. 先按全部标识去查；命中多个 paper_id 时进入合并队列，不在写入时随手挑一个。
  2. 标题回退只在双方都没有互相冲突的强标识时才用：年份允许相差 ±2，作者姓氏要对上，是否合并交给 LLM 裁定。
  3. 短标题和泛化标题（Editorial、Reply、Erratum 等）不允许只凭标题合并。
  4. 合并会覆盖所有挂着 paper_id 的表；`paper_aliases(old→new)` 保留旧 id，派生库按别名解析，旧 id 不会失效。日期取各来源中最早的。
- **标识只来自已确认指向本篇的来源**，即 DOI 精确查询的结果和 arXiv 元数据里的 doi 字段。检索候选只进 source_candidates，不进标识表。sci-evo 的数据迁移前先做清洗：13 篇论文挂着 2–12 个不属于它的 DOI。
- **标识种类**：doi / arxiv / openalex / s2 / openreview / pmid / pmcid / sciverse。Old Ideas 和 LimitGen 的论文来自 OpenReview，很多不在 arXiv 上，需要 openreview 标识和 OpenReview 元数据来源。`eval/adapters/` 负责把各个基准的 id 映射到 paper_id。
- **作者**：registry 存作者列表，可能时附 OpenAlex author id。自引判断和事实的独立性计数都用作者 id 的连通分量；只看姓氏的话，两篇随机 cs.LG 论文有 1.5% 会被误判为同一作者组。作者缺失时，独立性标为"未知"，不计入"共识"。

## 4. 全文来源（按领域和年份）

| 范围 | 首选 | 备选 | 实测 |
|---|---|---|---|
| CS，2007-04 至 2025-09 | NAS 上分版本的 arXiv PDF 镜像：`JournalPapers\arxiv\arxiv\pdf\<yymm>\<id>vN.pdf`（来自官方 GCS 数据集，每个版本一个文件） | arXiv HTML `/html/<id>v1`（每次请求间隔 15 s） | cs 论文的 v1 覆盖率：1803 99.8%、2006 99.2%、2203 99.9%、2406 100%、2502 99.7%；2509 只有 31.5%（这个月只同步了一部分） |
| CS，2025-09 之后 | GCS 公开桶 `storage.googleapis.com/arxiv-dataset/arxiv/arxiv/pdf/<yymm>/<id>vN.pdf`（不需要鉴权、不付费、不经过 arxiv.org 的限速） | 同上 | 已有到 2610 的前缀；单篇 0.8 MB 约 1.9 s |
| 非 CS，2021 年及以前 | 本地 Sci-Hub 档案（只存指针，不复制 PDF） | OpenAlex OA PDF、Crossref | 随机期刊文章命中率：2016–2020 年 45–59%，五大出版商 86–100%；2021 年 21%（五大出版商 32–72%） |
| 非 CS，2022 年及以后 | OpenAlex OA PDF、PMC OA、arXiv 物理和数学、bioRxiv、medRxiv | Sciverse | Sci-Hub 2022 年以后为 **0**（档案最晚写入于 2025-03）；OpenAlex 显示 2023 年 is_oa 72.7%，有 pdf_url 的 59.8% |
| 拿不到全文 | 只有元数据和摘要（T0/T1）；他述照样可以从施引论文里抽 | — | — |

- 废弃的路径：
  - `arxiv-papers\cs-*.zip`：版本混放。抽样 60 篇里 v1 只有 26 篇，v2 到 v5 共 18 篇；
  - IdeaForecast 的 Markdown：600 篇抽样都没有版本戳，审查也证实读的是最新版。它不再作为表述来源，只作为该基准自己的输入。
- **存储**：
  - NAS 和 Sci-Hub 上的 PDF 只记指针 `{channel, archive/path, inner_path, size, crc, sha256}`，不复制；
  - MinerU 产物剥掉 base64 后才入库（base64 占 markdown 的 54–98%）；
  - 批量构建时不存图片，只存 `(pdf 指针, page, bbox)`，需要时从 PDF 重新裁出来。
- **获取时核对身份**分两层：
  1. 确定性判断：首页或 XMP 里写着 DOI，或者标题词覆盖率高并且作者姓氏命中，就判为一致。标题取 MinerU 的 `doc_title` 块，不依赖 pdftotext。
  2. 介于两者之间的交给 LLM 裁定。

  判为不一致的移到 `library/quarantine/<sha256>.pdf`，并记录一次失败尝试。现在的代码是直接删掉，要改。
- **MinerU 吞吐要先测再定规模**（§12 阶段 B 的第一件事）。他述需要所有施引论文的全文：单是 cs.LG/CL/CV/AI/IR，2018 年以后就有约 43 万篇。现在的实测只有串行几篇，9 页的论文 standard 档要 8–15 s。要测的内容：并发 4/8/16 下的吞吐、flash 档的质量、服务是否与他人共用。如果吞吐撑不住，引用句这一步改用快速文本层（PyMuPDF）专门切句，MinerU 只给要深抽的论文用。这样会多一条文档路径，到时候先报给用户再决定。

## 5. 包结构（唯一的包 `compilescholar`）

| 模块 | 职责 |
|---|---|
| `core/` | 配置（唯一来源是 `configs/`）、路径（唯一入口，外部资源一律走 `paths.resource(key)`，没配置就报错，不设默认值）、密钥、**ids**、**asof**、运行清单 |
| `llm/` | 唯一的 LLM 客户端（§10.2）、JSON 解析与截断抢救、embedding（带模型标签）、响应缓存、台账、arm purity、视觉调用 |
| `library/` | registry：身份、标识、日期、资产、解析产物、抽取状态、来源记录、别名；身份解析与合并队列；批量导入 |
| `sources/` | 统一的 http 层（按主机跨进程限速、读 Retry-After、熔断、瞬时失败不进缓存）；arXiv（OAI、快照、分版本 PDF、GCS、HTML）、OpenAlex、Crossref、S2、OpenReview、Sciverse、Sci-Hub 本地库、MinerU V1 Router 的客户端各留一份 |
| `acquire/` | 按领域和年份选通道（§4）、身份核对、隔离、写资产指针与来源记录 |
| `documents/` | 从 structured_content 生成单元（markdown 和 HTML 两条路径保留）；剥图、分离浮动体、拼回断句、句号、页码和 bbox；匹配视图；确定性表格解析 |
| `citations/` | 参考文献块定位（没有标题也能找到）、引用标记解析（方括号数字、上标、作者-年份、字母标签、ACS 行内）、条目到 paper_id 的解析（§7.2） |
| `extract/` | schema v2、T1/T2/结果遍/他述遍、版本增量、统一终检、哨兵用例、work 表 |
| `cognition/` | 物化：mention_link、lineage_edge、cocite、reception_daily、category、fact、shift_event、族快照；在线查询层 AsOf |
| `index/` | tantivy（论文、段落、表述）加二值码和 float 的 memmap 向量；按 `day_hi` 过滤 |
| `tools/` | 全部工具（含外检）；quote 与 display 分开；预算记账；对照臂和消融开关；异步 MCP |
| `answer/` | lit 模式；冻结的 v9b |
| `grow/` | 缺口诊断，接着走 acquire 全链路 |
| `eval/`、`baselines/` | 评测、判分、对照系统；`eval/adapters/`；`eval/arms/field_state_llm.py`（冻结的对照组） |
| `service/`（阶段 E） | HTTP API（只绑 localhost，删除类和读文件类接口要二次确认） |
| `cli.py` | 唯一的命令行：build（`--rebuild` / `--resume`）、acquire、library、answer、judge、score、serve |

`web/` 放在仓库根目录（阶段 E）。`kb_compiler/`、`kb_infra/`、`retrieval/`、`sci_evo_extract.py` 的能力迁出后，先打 tag `pre-integration-20261005`，再从 src 删除；`legacy/INDEX.md` 记录每项能力的去向。

## 6. 旧模块逐项判定（汇总四份审查）

"直接可用"只有少数几项。大多数模块是"思路保留、代码重写"。证据和改法见各报告里对应的行。

| 层 | 直接可用 | 需优化 | 需重构 | 替换 | 退役 / 冻结 |
|---|---|---|---|---|---|
| 文库 registry | — | 资产与解析产物表（补解析器版本）；查询索引 | 表结构、id 生成、DOI / arXiv / 标题规范化、合并逻辑、日期、批量写入、并发写、资产合并、抽取状态表 | — | 死表和死方法（约 40% 是 UI 辅助）、evidence_units |
| 来源 | 熔断 circuit | Sci-Hub 查询（去掉一个 `lower()`，从 22.8 s 降到 1.3 ms）、PDF 解包（校验 crc、原子写）、rerank（改用统一 embedding）、MinerU `/file_parse` 客户端（9602，v2.1 改判：加限流、重试、版本记录） | OpenAlex / S2 / Crossref / Sciverse / arXiv 各有 2–4 份，合并成统一 http 层；refgraph 失败被当成完成 | 另两套 legacy MinerU 客户端 | search_service、query_understanding、pdf_resolver、search_resolve、process_workflow |
| 获取 | — | — | acquisition_chain（骨架保留）、身份核对、隔离 | — | acquisition.py 其余部分 |
| 文档 | — | documents/tables（指数、科学计数法、pipe 表首列序号） | documents/units（剥图、浮动体、句号、页码 bbox、structured_content）、documents/build（从 registry 输入） | — | evidence_units 分块 |
| 引用 | skeleton/bib（移到 citations/） | — | citations/markdown（上标、无标题参考文献块、DOI 尾标点）、citations/resolve（结构化引文查询，工作量大） | — | skeleton/resolve 的 Resolver |
| 抽取 | salvage 截断抢救（迁到 llm/） | T1（回传句号）、result_pass（顺序和 own_methods）、other_pass（按句分组，0.6 阈值降为信号）、notation_harvest、statements 溯源字段 | T2（全文分块）、postcheck（改为统一终检，取消 fuzzy 放行）、canary、table_semantic（跨领域轴角色）、proposes（拆开）、done 标记 | in_text、中文提示词 | 路由正则、单独的 absence 遍、注册表注入、coarse 两个模块、figure_channel（冻结）、arbitration、chunk_retry；survey_extract 暂缓 |
| 编译 | — | profiles（n_citing 改用全量计数）、comparisons（补可见性过滤） | asof（可见性和版本）、lineage、families（Leiden 加快照）、shifts（时间窗）、identity（提及级链接） | facts（向量召回加 LLM 裁定） | compile/state/coarse、kb_compiler/views/field_state；compile/state/field_state 移到 eval 作对照组 |
| 检索 | — | 论文索引范围（覆盖全部论文，并带层级） | — | FTS5 换 tantivy、整表 numpy 换二值码 memmap | kb/index.py、kb/store.py 冻结给 v9b；kb_compiler/retrieve、views/search_text 退役 |
| 工具与答题 | — | find / read / compare 类工具（quote 与 display 分开）、mcp_server（改异步、加硬超时、加外检）、ext_search / cite_expand（移入 tools，截止显式传入） | field 类工具（读物化表）、answer_lit（snippet、身份、归属、外检、探测、并行） | — | v9b answer 冻结；harness 的 corpus 模式退役，open 模式冻结 |
| 评测 | eval/cs2（只改路径） | eval/dsb（日级 as_of）、harness runner | eval/field（接 cognition 臂，"None" 字符串 DOI 按缺失处理） | — | — |
| 基础设施 | core/secrets | core/paths、core/manifest（原子写）、llm/embedding、llm/jsonparse | dfc/store 阶段契约、llm/client（本地和 Paratera）、调用台账、core/config、core/cutoff（并入 asof） | — | kb_infra/llm（先迁出视觉、purity、退避、CST 写法）、kb_infra/embedding、_see_llm、_see_config、conf/、根目录 build.py |

**之前被默认可用、实际不满足要求的主要几点**（完整清单见各报告"之前被默认可用"一节，共约 60 条）：

1. 我们以为 registry 已经是 DOI 优先的身份权威，实际上：
   - 预印本 DOI 成了 53% 论文的主键；
   - 同一个 DOI 写法不同会拆成 4 篇；
   - 两篇不同的 Editorial 被合成了一篇；
   - 有标识污染。
2. Sci-Hub 本以为已经接好，实际上：
   - 在本仓里因为 `.env` 路径错，一直被静默关闭；
   - 接上以后每次查询要 22.8 s；
   - 2022 年以后的论文覆盖率是 0。
3. MinerU 的输出：
   - 我们以为只有 markdown，其实还能给 structured_content；
   - 三套旧客户端都调不通。
4. 新抽取层在非 CS 论文上：
   - 引用句 0 条；
   - 方法和实验窗口 0 字符；
   - 同一篇论文，旧深抽 35–150 条，新抽取只有 4–7 条。
5. 阶段契约：
   - 改了代码重建时一条都不重算，失败也标完成；
   - manifest 不是原子写；
   - 没有锁，两个进程并跑会写出重复数据。
6. as_of：
   - 文本版本和日期错配；
   - 只有年份的日期用字符串比较会泄漏；
   - 截止规则有 4 套，缺日期时的处理方向相反。
7. 编译层"现算"在真实规模下跑不动，单次要几十秒到几小时；规则做语义判断时出错（"not slow"和"slow"被归为一类、变迁误报约 75%、族塌缩成 1 个）。
8. 检索：
   - FTS5 加日期过滤，100 万行要 1 s 以上；
   - 向量整表载入内存，每个 MCP 进程 16 GB。
9. LLM 客户端：
   - build 实际只跑 4 路；
   - 超时后服务端仍在生成；
   - 温度为 0 也不可复现；
   - 台账不落盘。
10. 工具返回的"原文"其实被截断过；他述的归属错位；外检没有接入。

## 7. 抽取层

### 7.1 schema v2（用户已同意）

- FACETS 新增 `config`（`meta.config{item, value, unit, applies_to}`）、`absence`（只收论文明确写出的缺失，不收猜测）、`definition`（可选）。
- Statement 新增一级字段：
  - `epistemic ∈ {demonstrated, stated, hypothesized, cited}`；
  - `condition`：原文中的限定词片段；
  - `loc{unit_id, sent_id, char_start, char_end}`；
  - `schema_version`；
  - 溯源字段 `pass / run_id / model / prompt_sha`。确定性产物的 model 记为 `deterministic:<代码哈希>`。
- `kind=other` 的表述可以带 `target`，用来表达第三方的说法，如"A 扩展了 B"。
- 旧的单篇 `shift` 记录退役。认识变迁由编译层跨时间计算。

### 7.2 引用

- 参考文献块：从文末往前找连续的、形似条目的行。标题有没有都行，允许 `■` 这类装饰符。
- 引用标记支持方括号数字、上标（`<sup>` 内是数字、范围或字母后缀的，整体当作一个标记）、作者-年份、字母标签、ACS 行内 `(n)`。
- 条目解析的顺序：
  1. 条目里自带的 DOI 或 arXiv 号；
  2. 结构化引文查询：期刊缩写 + 卷 + 页 + 年，走 Crossref `query.bibliographic` 或 OpenAlex filter 批量查。APS、ACS、Nature 这类条目不带标题，只能走这一步；
  3. 文库内按 norm_title 全文检索，同时核对年份窗口和作者姓氏；
  4. 以上都不行，记为 stub。

  只有 DOI 的条目直接建成"只有元数据"的论文。
- 解析率要分开报：CS 的 79.0% 是旧条件下测的；非 CS 要单独测。

### 7.3 四遍抽取与统一终检

| 遍 | 读什么 | 输出 |
|---|---|---|
| T1 自述 | v1 全文的标题加摘要句（编好句号） | 按句号回传 facet / role / proposes{name, aliases, relation: proposes\|uses}；quote 就是那一句，天然是原文子串 |
| T2 自述 | 全文单元打包成 ≤ 8k 字符的块；排除参考文献、致谢、作者、页脚 | 一套英文提示词覆盖全部 facet，每条带 epistemic / condition / sent_id；转述他人工作的句子交给他述遍；块调用的 max_tokens ≤ 6000，避免掉进宽度只有 6 的大调用通道 |
| 结果遍 | 修好后的表格网格，加 caption 和上下文句 | LLM 只提议轴角色（对象/样品/组别、条件、测量量和单位），结构门决定写不写入；数值只取自单元格；own_methods 来自 T2 |
| 他述遍 | 引用句，一句一项，带该句的 cited keys | 对每个被引 key 给出 role / function / facet；quote 就是这一句；带 `meta.self_cite`（按作者 id 判断是否自引） |

- **他述抽样**：在每篇被引论文的被引寿命内**按时间分层配额**，例如按年均分，或者按引用量开方分配。目的是让早期和后期都有足够样本。现在的轮转从最早的月份开始取，模拟数据下 30 对全部落在前 2.5 年，后期一条都取不到。另外，被引计数不再受 30 对抽样的上限截断，改由 `cite_count_monthly` 统计全量。
- **统一终检**只有一处实现：
  1. 用"匹配视图"定位原句。视图依次做：NFKC、去空白、去 `$`、LaTeX 折叠、去 HTML 标签、去引用号、MinerU 字形映射表。它只用来匹配，入库的 quote 保持原文。
  2. 数值必须逐字出现在原文里。
  3. 检查枚举值和必填字段。
  4. 不合格的给 LLM 修复一次，修完仍不合格就丢弃，并记下违规码。

  fuzzy 匹配只用来报告"差在哪"，不放行任何一条。
- **哨兵用例**：原有的 7 个事实和 6 个陷阱，按新 schema 重写打分器。再加一个 MinerU 形态的化学或医学变体，覆盖上标、无标题参考文献、base64、科学计数法表格，并新增两类陷阱："quote 不是原文子串"和"数值被改写"。每次改提示词都必须跑。
- **work 表**：`work(paper_id, pass, item_id, prompt_sha, model, status ∈ {pending, ok, failed, gave_up}, attempts, last_error, updated_at)`。只有 ok 算完成；失败的重试，最多 3 次。论文级状态写进 registry。

## 8. 编译层：离线物化 + LLM 裁定

- **按事件时间追加的表**（截至 T 的查询结果是精确的，只是 `WHERE date_hi ≤ T` 走复合索引）：
  - `mention_link`：覆盖**全部句子**里的方法名提及，包括不带引用标记的。"变成默认组件之后，引用里就看不到了"要靠它来补；
  - `lineage_edge` 和 n 元的 `lineage_hyper`：带 `valid_from = max(date, child, parent)`；
  - `cocite`：一条 SQL 自连接生成，实测 19 s；
  - `reception_daily` / `cite_count_monthly`；
  - `category_canon` 加 `category_daily`；
  - `fact_member` 加 `fact_status_event`；
  - `shift_event`；
  - `comparison_edge`；
  - `author_link`。
- **按网格存快照的只有族**。最近 3 年按月，更早的按季，另外加上各基准的截止日，约 80 份。查询取 ≤ T 的最近一份快照，再把快照之后新出现的强连接对象并进来，输出时注明快照日期。聚类用固定种子的 Leiden，并按度数给枢纽节点降权。族名和边界成员交给 LLM 判定。
- **LLM 化**：先用规则、统计或向量召回候选，再交 LLM 批量裁定。裁定结果按 §2.4 带上生效日，并缓存起来。

  | 原来的规则 | 改成 LLM 判什么 |
  |---|---|
  | Jaccard + 否定词正则 | 族级事实：same / opposite / narrower / broader |
  | 写死的伞形词阈值 | 类别短语规范化，并标出伞形词 |
  | 计数权重 | 方法身份：一个名字有 ≥ 2 篇候选时归到哪篇（18.3% 的命名论文与别人重名） |
  | — | 族命名 |
  | 前后两半切分 | 认识变迁候选先过时间窗内的 Fisher/二项检验并做多重比较校正，通过的再由 LLM 对照证据句判断是否成立 |

  每类裁定先抽 200 条，用双模型核对，一致率达到阈值才全量启用。成本估算见 FITNESS-INFRA §4：现有范围不到 2 小时；千万级规模约 30 小时，只算一次，之后只算增量。
- **bitemporal**：每行带 `build_id`。补进一篇旧论文后，过去某个 T 的结论会变，这是对的。但为了让旧的运行能复现，查询时可以加过滤条件 `ingested_build ≤ B`。
- `cites(sentence_id)` 的索引现在就补上：建一次只要 0.7 s，field_map 从约 3 h 降到 1.9 s。

## 9. 检索、工具、答题

### 9.1 索引

- 改用 tantivy（本机已装 0.26，要锁定版本写进依赖）。实测 1,000 万行查询 41 ms，100 万行 11 ms；FTS5 在 100 万行上要 1,066 ms。
- 三个索引：
  - papers：registry 里的全部论文，带层级字段（T0 / T1 / T2 / 有全文）；
  - passages；
  - statements。
- fast field 设 `day_hi`、`paper_id`、`kind`、`facet`、`section`，按 segment 增量写入。
- 向量：
  - 按日期排序存放；用二值码 memmap 粗召回 200 条，再用 float memmap 精排。实测 recall@10 为 0.995。
  - as_of 过滤就是取前缀切片。
  - 表头记录模型标签和维度，查询时模型对不上直接报错。
- 千万级表述：先给 self 表述和论文建向量。全量做不做，等实测 embedding 吞吐之后再定。ANN 在 usearch 和 faiss 中选一个，阶段 D 做小规模对比后决定。
- RRF 的 k=60 不变。每篇上限的 bug 要修：只有向量命中的条目会绕过这个上限。命中强弱改用结构化信号定义，阈值在 dev 题上标定后冻结。

### 9.2 工具

- 返回值把 `quote`（原文）和 `display`（可截断）分开，证据一律用 quote。
- 各工具的归属：
  - field 类工具（field_map / paper_profile / frontier / open_issues）读物化表；
  - 有界的工具（card / profile / compare / events）继续在线现算；
  - 目标是单次调用 < 200 ms。
- 新增两个外检工具：
  - `search_external(query, as_of, k)`：Sciverse 语义检索；同年的命中先解析成具体日期，再判断是否可见；
  - `expand_citations(seeds, as_of)`：本地用 references 加 cocite，外部走 refgraph，按共引排序。

  外检结果都标"外部"，评测期间不写回；外部命中如果能解析到本地 paper_id，就并入本地。
- MCP：工具改成 async，用 `to_thread` 包住同步实现，单次调用设硬超时。启动只打开 tantivy 和 memmap，不预载任何东西，因为每个 harness 会话都会起一个进程。

### 9.3 答题（lit 模式）

- 他述证据挂在**说话方**名下，写成"Y 说 X……"，snippet 就是 Y 的原句；X 作为被描述对象写进 metadata（用户已裁定）。
- 身份一律用 paper_id，展示用的 DOI 和 arXiv 号从标识表取。
- 探测和每一条子查询都并行跑本地加外检。
- trace 写出 evidence 和 used_eids。

### 9.4 对照臂、消融、预算

- 工具层把对照臂做成一等开关，由服务端配置固定，智能体改不了。四个对照臂：
  - `none`：不接工具；
  - `flat`：同预算的平铺检索，只返回段落；
  - `summary`：每次调用现场压缩，对检索结果做摘要；
  - `full`：完整工具。
- 消融开关：
  - `−time`：不做历时分层，但截止仍然生效；
  - `−reception`：去掉他述；
  - `self_only`。
- 每次工具调用都记下返回的字符数和 token 数，同一实验各臂的预算上限相同。台账写进运行目录，供"同预算"核账。

## 10. 基础设施

### 10.1 阶段契约（阶段 A 最先做）

- 代码哈希改成按阶段入口自动计算 import 闭包，不再手工维护 CODE 表。
- 参数里加上 provider、model、模板 sha、采样参数、embedding 标签。
- **过期判断分两级**：
  - 配置级：参数、代码闭包、模型或模板变了，受影响的那一遍全部重算。改一个提示词只让这一遍过期，因为 work 表按 prompt_sha 记。
  - 数据级：上游新增或变更了条目，只算这些条目（按条目的输入指纹判断）。语料增长是追加，不会让整个阶段过期。只有族快照这类全局对象，要按受影响的范围重算。
- `build --rebuild` 先写到新文件，完成后原子替换。`build --resume` 只在配置级 digest 不变时允许。
- 失败的条目不标完成；连续失败到阈值就熔断，阶段报错，不写 manifest。manifest 的 counts 里包含 failed 数。
- manifest 先写临时文件再 `os.replace`。每个阶段加文件锁。statements 加自然键唯一索引。设 `busy_timeout=30000`。数据目录只放本地盘。
- 8 个探针改写成正式测试，断言修复后的行为。

### 10.2 唯一的 LLM 客户端

- 传输用 httpx 流式。墙钟超时一到就关闭连接，让服务端中止生成（GPUStack 上是否真会中止，要实测确认）。车道一直持有到连接真正关闭。
- 车道宽度、超时、重试次数从 configs 读，build 和 answer 用同一份配置。
- 每个 provider 配一个令牌桶（跨进程共享）。退避用指数加抖动，并遵守 Retry-After。
- 熔断：连续失败抛 `ChannelDead`。
- 台账常开，写到 `runs/<run_id>/llm_calls.jsonl`，字段包括 http_status、error_class、finish_reason、template_sha、served_model 等。
- **用响应缓存保证可复现**，因为实测温度为 0 时输出也不稳定。缓存键是 `sha(provider, model, messages, sampling, schema)`。
- 从 kb_infra 迁出视觉调用、arm purity 检查和 CST 的关思考写法。`call_json` 带 schema 校验和截断抢救。
- embedding 只用本地模型，返回 `(vectors, model_label, dim)`，不允许跨模型回落。

### 10.3 配置与路径

- 配置分层，后者覆盖前者：`configs/base.yaml` < `configs/local.yaml`（本机，gitignore，放外部资源路径和车道宽度）< 实验配置 < `--set`。解析后的配置写进运行目录和阶段 manifest。
- 硬编码清单：新包里有 13 处，另有 38 个环境变量当配置在用（FITNESS-INFRA §3）。全部改成走配置。
- 加一个闸门测试：`src/compilescholar` 里不允许出现 `.research_tmp`、盘符路径、UNC 路径；`os.environ` 只能出现在 core/ 和登记过的第三方注入点。

### 10.4 备份

以下内容每天增量同步到 NAS 个人目录（`\\192.168.199.138\Share400T\personal\jhd0n9\`，10-04 已建好）：
- 权威库：registry、解析产物；
- `cache/llm/responses.sqlite`；
- 派生库里的抽取产物。

同步脚本放在 `tools/`。另外每个阶段闸门通过时做一次快照。

### 10.5 测试

需要补上的测试：
- 规模冒烟：10 万条表述，检查耗时预算，并断言查询计划走索引；
- 并发与契约；
- 失败与重试：用本地假 HTTP 服务模拟 429、5xx、挂起、截断；
- as_of 泄漏：参考文献晚于施引论文的文本版本；只有年份的日期；
- 跨领域：只有 DOI、作者缺失、非 arXiv 的 id。

另外两项处理：
- 干净 clone 下 `test_config.py` 失败，随迁移表一起修。
- golden 视为冻结数据，记录 sha256 并加测试断言不变。kb_compiler 要等 salvage 迁到 `llm/jsonparse` 之后才能删，否则表征测试会挂。

## 11. 工作区目标布局

与 v1 一致：`src/compilescholar/`、`web/`、`tests/`、`configs/`、`experiments/<主题>/`、`runs/<run_id>/`、`results/<冻结名>/`、`data/{library, derived, benchmarks, external}/`、`cache/`、`docs/{design, archive}/`、`third_party/`、`tools/`、`legacy/`；`.research_tmp/` 只作草稿区。v1 §5 里的迁移表、根目录清理和 `.research_tmp` 处理方案都不变，用户已同意。

新增：
- `cache/llm/`（响应缓存）；
- `library/quarantine/`；
- 依赖要按实际 import 声明并锁定版本，新增的有 httpx、tantivy、leidenalg（含 igraph）、usearch 或 faiss、pyarrow、mcp、pydantic、fastapi、typer、uvicorn、python-multipart。

## 12. 实施阶段与闸门（E 挪到 F 之后）

| 阶段 | 内容 | 闸门 |
|---|---|---|
| **A 基础** | ① 阶段契约修复（§10.1）；② 唯一的 LLM 客户端加配置与路径（§10.2–10.3）；③ `core/ids.py`、`core/asof.py`（v9b 走兼容适配器）；④ 把隐藏依赖迁出 `.research_tmp`；⑤ 旧包的能力迁出（先迁 salvage），sci-evo 代码作为待重构的代码迁入；⑥ 根目录清理与打包；⑦ `cites(sentence_id)` 索引 | v9b 表征测试逐字节一致；全部测试通过；契约探针测试、路径闸门测试、客户端故障测试通过；干净 clone 上测试能跑 |
| **B 底座**（v2.3：已建成，闸门见 v2.3 第 5 条） | ① ~~9602 加限流和重试后重测 MinerU 吞吐；快速档与 GROBID 对照~~（v2.2 已完成：快速档 = GROBID full + PDF 引用链接，MinerU 9602 最多 6 路）；② 按新 schema 重构 registry，批量导入 104.6 万篇 arXiv 元数据（含各版本日期），并导入四个固定库基准的语料；③ 清洗 sci-evo 数据后迁入；④ 合并队列加 LLM 裁定；⑤ sources 统一 http 层，Sci-Hub 修好；⑥ acquire 按通道接入，加身份核对和隔离；⑦ MinerU 客户端（9602 file_parse 为主，9605 V1 备选），精读档 documents 从 content_list / structured_content 生成，快速档 documents 从 GROBID TEI 生成（句、引用、条目带坐标，PDF 引用链接补连接，剔除表格和图里的文字，PyMuPDF 只用于读链接和上标）；⑧ 版本感知（v1 加最新版增量） | 身份合并抽查，双模型核对；20 篇跨领域 PDF（Nature、物理、化学、医学、含公式）走通 acquire → MinerU → units；版本泄漏测试（参考文献不晚于文本版本日）；equation 块形态核实 |
| **C 引用与抽取** | ① citations 重构加结构化引文查询；② schema v2；③ T1 / T2 / 结果遍 / 他述遍；④ 统一终检；⑤ 哨兵用例；⑥ work 表；⑦ 他述按时间分层抽样；⑧ 自引标记 | 哨兵通过；抽样 200 条双模型核对 ≥ 0.95；50 篇新旧覆盖率对照（旧深抽的信息要被新记录覆盖）；引用解析分 CS 和非 CS 各复测一次 |
| **D 编译、索引、工具** | ① 物化各表（§8）；② 五类 LLM 裁定，各先做 200 条试点；③ tantivy 加向量；④ 工具改造、外检、对照臂和消融开关、预算记账；⑤ answer_lit 修复 | 端到端不变量（含泄漏和规模冒烟）；裁定一致率达到阈值；工具单次调用 < 200 ms |
| **F 实测与打磨** | ① 写下题目盲的建库范围规则，并预注册；② 真实构建；③ 基准适配器；④ 真实智能体经 MCP 在 dev 上跑，看轨迹迭代 harness | CS2 dev 非劣（界 −0.02）；主基准 dev 结果 |
| **E 服务与前端**（论文之后） | 不在当前计划内（v2.1 第 6 条）。论文完成后另做前端网页，届时在统一文库和工具之上重写 API（只绑 localhost） | — |

每个阶段结束都提交一次，同时更新 `docs/ARCHITECTURE.md`。

## 13. 仍需用户决定或尚未核实

**需要用户决定**（到对应阶段再问，不影响阶段 A 开工）：
1. ~~题目盲的建库范围规则~~：v2.1 已定为"库 = 基准的候选语料；深抽按库内被引数挑选；题面和金标永远不用"，预注册时写明。
2. ~~表述向量范围~~：v2.1 已定为全量。
3. ~~快速文本层~~：v2.1 已定为两档。

**尚未核实**：
- MinerU 的 equation 块、advanced 档、50 页以上的大文件、并发；
- Crossref 投稿日的覆盖率；
- GPUStack 在客户端断开时是否真的中止生成；
- tantivy 的 en_stem 分词器；
- Claude Code MCP 工具的超时和结果大小上限；
- Sci-Hub 档案多进程读 UNC 时的吞吐；
- 合并、聚类、身份三类 LLM 裁定的准确率；
- OpenReview 元数据的来源。
