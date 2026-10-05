# 一体化系统与工作区设计（INTEGRATED-SYSTEM-1005，v1，待用户确认）

> 用户裁定（2026-10-05）：
> - 底座用 sci-evo-extract，直接搬进来合并进系统；身份改为 DOI 优先；
> - Sci-Hub 本地库用于系统内部建设；MinerU 先试原地址，不行用 9605 端口（实测 9605 可用）；
> - 系统更新不能只做新功能，旧系统也要完整审视并升级迭代；
> - 把项目当成一个完整系统来打磨，模块浑然一体、成体系，不是零碎拼接；
> - 工作区条理清晰，不乱放代码和文件；
> - 复杂语义交给 LLM，规则只做结构性、确定性的事。
>
> 依据（10-05 三路只读盘点 + 实测）：sci-evo 全项目盘点、旧模块到新系统的映射、工作区盘点；
> Sci-Hub 索引与档案实测、MinerU 9605 实测（单篇 9 页 standard 档 15 s）。
> 本档取代 DESIGN-UPGRADE-1005 §9（代码组织）和 DESIGN-LITERATURE-LAYER-1005 §3（新旧判定）。
> 叙事（NARRATIVE-V9）与评测（EVAL-PLAN）不变。

## 0. 现状一句话

现在仓库里实际有三套系统叠在一起：
- 旧栈：kb_compiler + kb_infra，12,137 + 1,195 行，答题用的 KB v2 靠它建成；
- sci-evo：retrieval，13,714 行，包含文库、获取、MinerU、检索服务，现行路径一行都没用；
- 新栈：compilescholar 里的 dfc 主线，9,365 行，以 arXiv id 为身份，另建了一套存储。

三套之间，LLM 客户端、embedding、配置、检索、存储、Sciverse、引用图各有 2–4 份实现。现行代码还从 `.research_tmp`（24 GB 的实验工作区）里读 KB、rubric 和金标，干净 clone 跑不起来。

这次升级的目标是把它们合成**一个系统**：一个包、一套身份、一个权威库、一条数据流。每项能力只有一处实现，每个文件只有一个归处。

## 1. 系统全景

```
                    ┌──────────────────── 权威库 data/library/ ────────────────────┐
 sources ──► acquire ──► library: registry.sqlite（身份 DOI 优先 · 标识 · 资产 · 解析产物 ·
 (arXiv OAI/HTML,        编译状态 · 抽取运行 · 来源记录）+ papers/PPR_*/（PDF、MinerU 输出）
  OpenAlex/S2/Crossref,  └──────────────────────────────┬──────────────────────────────┘
  Sciverse, Sci-Hub,                                    │ paper_id
  MinerU)                                               ▼
                    ┌─────────────── 派生库 data/derived/（全部可重建，阶段 manifest）──────────────┐
                    │ documents ──► citations ──► extract ──► index                           │
                    │ 节/段/表/图注   引用句→论文   自述·结果·他述   论文/段落/表述 检索              │
                    └──────────────────────────────┬──────────────────────────────────────────┘
                                                   ▼
                              cognition（AsOf(T) 现算：身份 · 族 · 谱系 · 画像 · 事实 · 变迁 · 比较）
                                                   ▼
                       tools（找 · 读 · 证据 · 领域 · 比较 · 外检）──► MCP │ answer │ service(API) │ eval
                                                   ▲
                                 grow（缺口诊断 → acquire → 走同一条流水线）
```

- **权威库与派生库分工。** 权威库只放不可重建的东西：身份、标识、资产文件、解析产物、编译状态、来源记录。派生库放的都是能从权威库和原始来源重算的东西，包括全文单元、引用句、表述、索引。派生库删了能重建，权威库不能删。
- **一个身份。** 论文 = `paper_id`（registry 的 `PPR_`+哈希，DOI 优先生成，没有 DOI 就用规范化标题），挂一张标识表：DOI / arXiv / OpenAlex / S2 / Sciverse。
  - 预印本与正式版按 DOI↔arXiv 合并为一篇；
  - 日期 = 最早公开日期（arXiv v1 或出版日期，取早的），并标明精度（日 / 月 / 年）；
  - 派生库与工具一律用 `paper_id`；对外展示时附带 DOI 和 arXiv 号；
  - 解析不到的被引文献仍是 `stub:<规范化标题>`，有 DOI 的则直接建为"只有元数据"的论文。
- **一条数据流。** 每个阶段只读权威库和上游派生库，阶段 manifest 记录参数、上游版本和代码哈希；上游一变，下游自动判过期并拒绝运行（沿用已写好的 `dfc/store.py`）。

## 2. 目标包结构（唯一的包 `compilescholar`）

| 模块 | 职责 | 来源（合并自） |
|---|---|---|
| `core/` | 配置（唯一：`configs/`）、路径（唯一入口）、密钥、截止日期、运行清单 | 现 core/；吸收 `kb_compiler/config.py`、`_see_config.py`；`conf/` 退役 |
| `llm/` | 唯一的 LLM 客户端（本地 / Paratera / CST / 视觉）、JSON 修复、embedding（带模型标签）、调用台账、arm purity 检查 | 现 llm/；吸收 kb_infra 的视觉调用、purity、提供方标签、调用台账，`kb_compiler.records.common.salvage_json_records`；`_see_llm.py` 退役 |
| `library/` | 文库：registry（DOI 优先身份、标识、资产、解析产物、编译状态、抽取运行、来源记录）、身份归一与去重合并、大批量导入 | sci-evo `registry.py` 原样迁入后按职责拆成几个文件；`corpus/papers.py` 的 104.6 万篇 arXiv 元数据改为批量导入 registry |
| `sources/` | 外部来源客户端：arXiv（OAI-PMH、快照、HTML）、OpenAlex / S2 / Crossref（合并为一套）、Sciverse（一份）、Sci-Hub 本地库、MinerU、熔断、重排 | 现 sources/ + retrieval 的 sources / citation_graph / circuit / rerank / pdf_resolver；refgraph 与 citation_graph 合并 |
| `acquire/` | 取全文链：arXiv HTML → arXiv PDF → OA PDF → Sciverse → Sci-Hub 本地库；首页标题核对身份，不符就隔离；PDF 经 MinerU 解析后写入文库资产 | retrieval 的 acquisition / acquisition_chain / search_resolve / processing / process_workflow |
| `documents/` | 全文 → 单元（Markdown / HTML / MinerU content_list，带页码和 bbox；剥离内嵌 base64 图片）、确定性表格解析 | 现 documents/；吸收 retrieval `evidence_units.py` 的 MinerU 分块逻辑 |
| `citations/` | 引用句 → 参考文献条目 → 论文（DOI → arXiv → 文库标题 → OpenAlex 批量），支持方括号数字、作者-年份、字母标签、**上标数字**（Nature / 物理 / 生物期刊） | 现 citations/ + compile/skeleton 的 bib / resolve；补回旧骨架的 DOI 与 OpenAlex 阶段（旧解析率 79.0%，现只认 arXiv） |
| `extract/` | 一套 schema、三遍抽取（自述 T1 / T2，结果，他述）+ 统一终检 + 哨兵回归 | 现 extract/ + 旧 deep_extract v1.4（见 §3）+ postcheck + canary + table_semantic + notation_harvest（可选）+ skeleton/proposes |
| `index/` | 论文 / 段落 / 表述三级检索，BM25 + 向量（带模型标签），as_of 过滤，命中强弱信号 | 现 index/；吸收 kb/index.py 的强弱信号；`kb_compiler/retrieve/hybrid.py`、`views/search_text.py` 退役 |
| `cognition/` | AsOf(T) 编译：身份、族、谱系、画像、族级事实、认识变迁、定性比较 | 现 cognition/；field_state 降为族命名；语义判断改用 LLM（§4） |
| `tools/` | 找 / 读 / 证据 / 领域 / 比较 / **外检**，全部带 as_of；MCP 服务 | 现 tools/；吸收旧 KBTools 的 config()、基于 absence 的 find_gap；新增外检工具（Sciverse、引文扩展，结果标"外部"，不写回） |
| `answer/` | 内置答题管线：lit 模式（工具驱动）+ 冻结的 v9b 路径（对照组） | 现 answer/；lit 模式接入外检工具（v9b 有 61.9% 的引用来自外检，不接会掉分） |
| `grow/` | 缺口诊断 → acquire → 同一条流水线 | 现 grow/；获取动作走 acquire/ 全链路，不再只取 HTML |
| `service/` | HTTP API：文库管理（导入、获取、解析、浏览、编译状态）+ 工具 | sci-evo `api/`（45 条路由）迁入，改建在统一文库与工具之上；默认只绑 localhost，删除与读文件接口加确认 |
| `eval/`、`baselines/` | 评测、判分、对照系统；新增 `eval/adapters/`（ScholarCatalyst、MasterSet 等） | 现有，不动 |
| `cli.py` | 唯一命令行：build / acquire / library / answer / judge / score / serve | 现 cli.py + sci-evo `cli.py`（typer 命令并入）；根目录 `build.py` 退役 |
| `web/`（仓库根，非包） | 文库前端 | sci-evo `frontend/`（React）迁入，通过 service API 访问 |

**退出 `src/`**：`kb_compiler/`、`kb_infra/`、`retrieval/`、`sci_evo_extract.py`。各自的能力按上表并入后，先打 tag `pre-integration-20261005`，再从 src 删除；历史留在 git 里，`legacy/INDEX.md` 记录"旧能力 → 新位置"。

## 3. 旧系统逐项升级（不只是新功能）

| 旧能力 | 实测质量 | 并入后的做法 |
|---|---|---|
| deep_extract v1.4（全文分块 ≤8k 字符、上限 500k；result / config / lineage / finding / absence / shift / notation；先引原文） | 21 篇 hub 共 4,740 条，逐字率 95.9%；config 1,146 条 91.9% | **T2 自述遍改为全文分块**，取代现在"引言 ≤7k + 方法实验 ≤9k"的截断读法。补上 config{项, 取值}、明确写出的 absence、条件与认识状态字段；沿用"先引原文"规则。旧的实体注册表注入不搬 |
| postcheck（五道确定性门：数值逐字、LaTeX 折叠、表格宽松匹配，修复一次，仍坏就丢） | 只检过深抽取产物 | 三遍抽取汇合后的**统一终检**，取代现在只做"空白归一后子串匹配"的 `in_text` |
| canary（合成哨兵论文：7 个事实 + 6 个陷阱） | 无入口 | 按新 schema 重写，作为抽取回归测试（每次改提示词必跑） |
| table_semantic（F35：LLM 只提议行列角色，确定性门决定写入） | subject 填充率 0 → 43% | 结果遍的第二阶段：先跑自述遍，把它抽出的"本文提出"作为 own_methods 传给结果遍（现在的顺序反了，本文方法锚定失效）；门里的注册表匹配换成 Identity |
| notation_harvest（公式定义） | 93% 的论文有公式，约 34 个/篇 | 可选的 T2 子遍（facet=definition），按领域开关 |
| figure_channel（视觉模型读图） | GLM-4.6V 20/20 | MinerU 现在能输出图片，接入视觉调用后作为可选 T2 子遍；先冻结，等主线稳定 |
| arbitration_list（junk_latex / citation_ref / generic_label 启发式） | — | Identity 的名字过滤（结构性判断，规则可以做） |
| 旧 KBTools 的 config()、find_gap（基于 absence）、检索命中强弱信号 | 接口方向对，旧数据断 | 并入 tools 与 index |
| sci-evo 获取链 + MinerU | Multi 语料取到 431/436（98.9%） | acquire/，加 Sci-Hub 本地库与 MinerU 9605 |
| sci-evo 熔断 + 重排 | 重排后目标论文从第 4 升到第 1；熔断 3 次失败 / 120 s → 断开 600 s | sources/，外检工具与 acquire 共用 |
| sci-evo coarse_store 两档 + registry compile_states（none / shallow / deep） | 表已为 CompileScholar 扩展，但一直是空表 | **抽取分层状态记在 registry 的 compile_states 里**，取代现在派生库里的 done_self |
| 旧检索的"每篇上限、RRF" | v9b CS2 0.829 | index/ 已沿用 |
| query_understanding（单发改写） | A6 召回 0.062，FAIL | 不并入；规划阶段本来就会生成子查询 |
| 实体注册表、综述专用抽取、视图编译器、backflow、生长马拉松、sci-evo 导出与演化案例 | 见 INTERNAL-AUDIT | 退役（能力已由新模块覆盖，或已判死） |

**要修的新代码缺陷**（盘点 3 指出）：
- 抽取的"已完成"标记不分成功失败一律写入，失败的论文和引用对永远不会重跑 → 改为只标记成功，失败的记录进重试队列；
- 向量表没有模型列 → 加模型标签，不同模型的向量不混算；
- 引用解析只认 arXiv → 补 DOI 与 OpenAlex；
- 不支持上标引用 → 补上；
- 内嵌 base64 图片 → 入库前剥离，图片单独存为资产。

## 4. 复杂语义交给 LLM

| 判断 | 现做法（规则） | 改为 |
|---|---|---|
| 两条说法是否讲同一件事（族级事实聚类） | 内容词 Jaccard ≥ 0.5 | 向量粗聚类 + LLM 裁定 |
| 他述描述是否忠实于引用句 | 词面覆盖 ≥ 0.6 | 保留"原句必须存在"这条确定性约束；忠实度由 LLM 判（批量） |
| 两条说法是否矛盾、条件是否一致 | 否定词正则 | LLM 判 |
| 两个名字是否同一方法（变体、同名异义） | 归一化名字 + 计数权重 | 规则召回候选，LLM 裁定 |
| 是否被重新归类、是否被替代 | 写死阈值 | 统计信号保留作候选；事件是否成立由 LLM 对证据句判断 |

规则仍然做的事：引用标记到参考文献条目的对应、DOI 和 arXiv 号识别、表格网格展开、日期过滤、阶段契约、原句子串检查。

## 5. 工作区目标布局

```
CompileScholar/
  src/compilescholar/     唯一的包
  web/                    前端（sci-evo frontend）
  tests/                  与包结构对应；sci-evo 的 24 个测试按新位置迁入（14 个改 import 即可，10 个随 api/cli 一起迁）
  configs/                唯一配置（base / local / bench）
  experiments/<主题>/     一次性实验脚本，只写 runs/
  runs/<run_id>/          运行产物（gitignore），根部不留散文件
  results/<冻结名>/       只放冻结、被论文引用的结果，由 `compilescholar freeze` 写入，不由脚本直写
  data/                   gitignore
    library/              权威库：registry.sqlite + papers/PPR_*/
    derived/              派生库（各阶段 sqlite + manifests）
    benchmarks/<名>/      基准数据、rubric、金标、存档答案、冻结的 KB v2（从 .research_tmp 迁出，大文件记 sha256）
    external/             第三方原始数据（IdeaForecastBench、ScholarCatalyst、Intern-Atlas、Asta 日志）
  cache/                  gitignore（OAI、HTML、refgraph 缓存）
  docs/                   ARCHITECTURE / DATA / EXPERIMENTS / RESULTS / DECISIONS
    design/               现行设计（本档 + system-vision-1004 下的现行文档迁来）
    archive/              过期设计
  third_party/            第三方 checkout 说明与补丁（asta-bench checkout 指向此处，gitignore）
  tools/                  仓库工具（pre-commit）
  legacy/                 旧脚本 + INDEX（旧能力 → 新位置）
  .research_tmp/          只当草稿区：现行代码不读它，不再强制跟踪新文件
```

**迁移表（现行代码读 `.research_tmp` 的隐藏依赖 → 新位置）**

| 现在 | 迁到 |
|---|---|
| `cs2/base_kb_v2`（KB v2，含 475 MB 向量） | `data/benchmarks/cs2/kb_v2/` |
| `scholarqa_multi/sqa2_rubrics_v{1,2}_recomputed.json` | `data/benchmarks/cs2/rubrics/` |
| `cs2/arm_memorized`、`cs2/arm_harness` 存档答案 | `data/benchmarks/cs2/arms/` |
| `_shared/refgraph_cache` | `cache/refgraph/` |
| `cs2/base_kb_v2` 里的 survey_gold、heldout_refs | `data/benchmarks/field/` |
| `deepscholar/dsb`、`review_1002/p6/oracle_inputs.json` | `data/benchmarks/dsb/` |
| `scratch/.../asta-bench-main` | `third_party/asta-bench/`（gitignore，config 指路径） |
| `cs2/arm_vnext/config_test100_r1.json`、FREEZE 文件（test_config 依赖） | `results/cs2-test100-v9b-20261004/` |
| `docs_decisions/` 现行设计 | `docs/design/`；过期的进 `docs/archive/` |

`core/paths.py` 成为唯一入口，所有路径都由它给出。

**根目录与杂物**
- `build.py`、`build_manifest.json`、`conf/` 退役，由 `compilescholar build` 取代；
- `CHANGELOG.md` 更新；`.gitignore` 删除死规则；
- `runs/` 根部散落的日志归入对应的 run 目录；
- `ccfa-workfiles/`（101 MB，ppt 等）、`.playwright-mcp/` 移出仓库（待用户定去处）。

**`.research_tmp` 本身**：迁出隐藏依赖后，
- 已强制跟踪的 4,181 个文件里，设计文档迁到 docs/，其余取消跟踪（文件留在盘上）；
- 嵌套的 `.research_tmp/.research_tmp`、`_trash_pending`、根部 67 个散文件，核对后冷存到 NAS（沿用 10-04 的冷存流程：先复制、逐文件校验，再删本地）。

**打包**：只打包 `compilescholar`；按实际 import 声明依赖（补 pyarrow、mcp、pydantic、fastapi、typer、uvicorn、python-multipart；删掉没用到的 extra）；测试不进 wheel。

## 6. 实施阶段与闸门

| 阶段 | 内容 | 闸门（不过不进下一阶段） |
|---|---|---|
| A 工作区与包 | 隐藏依赖迁出 `.research_tmp`；三个旧包的能力并入 compilescholar（LLM、embedding、配置、Sciverse、引用图各留一份）；sci-evo 代码与测试迁入；根目录清理；打包修正 | v9b 表征测试逐字节一致；全部测试通过；干净 clone 上测试能跑 |
| B 底座 | registry 成为身份权威（DOI 优先、多 id、最早公开日期 + 精度）；迁入 sci-evo 数据（490 篇、2.6 GB 资产）；104.6 万篇 arXiv 元数据批量导入；派生库改以 paper_id 为键；acquire 接通 Sci-Hub 与 MinerU 9605；documents 支持 MinerU 输出 | 身份合并抽查（预印本 / 正式版）双模型核对；20 篇跨领域 PDF（含 Nature、物理、生物）走通 acquire → MinerU → units |
| C 抽取 | T2 全文分块并入 deep_extract v1.4 能力；结果遍接 table_semantic 与 own_methods；统一终检 + 哨兵；失败重试；上标引用；DOI / OpenAlex 解析 | 哨兵通过；抽样 200 条双模型核对 ≥ 0.95；50 篇新旧覆盖率对照（旧深抽信息被新记录覆盖）；引用解析 E1 式复测 |
| D 编译与工具 | §4 五处改用 LLM；工具补 config / absence 缺口 / 命中强弱 / 外检；向量带模型标签；lit 答题接外检 | 端到端不变量测试；聚类与身份裁定抽样核对 |
| E 服务 | sci-evo API 建在统一文库与工具之上（只绑 localhost，危险接口加确认）；前端迁入 web/；命令行合一 | API 测试迁入并通过；前端能浏览文库与工具结果 |
| F 实测与打磨 | 真实构建（题目盲范围）；基准适配器；真实智能体经 MCP 在 dev 上跑，看轨迹迭代 harness | CS2 dev 非劣（界 −0.02）；主基准 dev 结果 |

每个阶段结束都会提交并更新 `docs/ARCHITECTURE.md`，不留"文档描述的系统和代码不一致"的状态。

## 7. 待用户确认

1. **sci-evo 的 API 服务和前端**是否并入（阶段 E）。建议并入：它是文库管理的界面，改建在统一文库上；默认只绑 localhost。
2. **工作区里有移动或删除性质的操作**，需要逐项确认：
   - `ccfa-workfiles/`、`.playwright-mcp/` 移到哪里（建议移到 NAS 个人目录）；
   - `.research_tmp` 已跟踪文件除设计文档外取消跟踪；
   - 嵌套目录、`_trash_pending`、散文件冷存到 NAS 后删本地。
3. **旧包退出 src 的方式**：建议打 tag 后从 src 删除（历史在 git 里，INDEX 记录去向），不在 legacy/ 里再留一份副本。
