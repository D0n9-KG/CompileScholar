# 基础设施与编译层适用性审查（FITNESS-INFRA-COGNITION，2026-10-05）

> 审查对象：LLM 客户端（三套）、配置与路径、阶段契约 `dfc/store.py`、编译层 `cognition/*`、`compile/state` 与 `compile/skeleton`、测试。新写代码与旧代码同等对待，不默认可用。
> 标准：INTEGRATED-SYSTEM-1005 §1–§4，以及任务书列出的硬要求（一个包一份实现；路径只经 core/paths；配置只经 configs/；密钥只经 core/secrets；LLM 重试、超时、限流、台账、可复现；千万级规模；as_of 按天、含当天、单调；复杂语义交给 LLM；可续跑、SQLite 并发安全、manifest 正确传播过期）。
>
> 做法：逐文件读代码；在真实库 `data/dfc/{papers,citations}.sqlite` 上只读查询；在 `%TEMP%\fitinfra_review` 下用合成数据（1 万篇、17.5 万条表述、23.3 万条引用行）实测编译层；写 8 个阶段契约探针测试；调用本地 Qwen3.8-27B 共 18 次（并发 15 + 超时探针 1 + 确定性 2）。没有改仓库代码，没有读 .env 内容（只经 `core.secrets` 判断键是否存在），没有访问外网，临时目录已删除。
>
> 证据格式：`file:line`（相对 `src/compilescholar/`，旧包写全路径）或实测数字。工作量：S ≤ 0.5 人日，M 1–2 人日，L 3–5 人日。

## 0. 结论先行

- **阶段契约 `dfc/store.py` 现在不能保证"上游一变，下游必重建"，需重构。** 8 个探针全部复现问题：
  - 改了代码再 rebuild，实际一条都不重算（各阶段都按"已完成集合"跳过），却把 manifest 盖成新的、判为新鲜；
  - 上游数据变了而参数没变，digest 不变，下游不知道；
  - extract 的代码哈希没覆盖 `llm/client.py`、`llm/jsonparse.py`、`compile/skeleton/proposes.py`，index 的没覆盖 `llm/embedding.py`；
  - LLM 全部失败时，3 篇论文全标"已完成"、0 条表述，阶段仍判新鲜；
  - manifest 写入不是原子操作，并发读时 816 次里 318 次 JSONDecodeError；
  - 两个进程同时构建同一阶段，结果整套重复（14 条表述里 7 组重复）。
  - 真实库现在就处在这种状态：citations 判为过期（上游 documents 未建、代码已改），这时执行 `build all`，已有的 28,450 篇文档一篇都不会重算，citations 却会被重新盖成新鲜。
- **编译层逻辑上能用，但三个硬要求都没达到：规模、as_of 正确性、语义判断方式。**
  - 规模：每次工具调用都现算、全表扫描。在 17.5 万条表述上，`paper_profile` 一次 7.0 s，`Identity` 1.35 s，`lineage.edges` 2.74 s。按线性外推，千万级时 `paper_profile` 一次约 400 s，单次读表述的内存约 20 GB。真实库的 `cites` 缺 `sentence_id` 索引，`co_cited` 每次 0.15 s，一次 `field_map` 的范围要查约 2 万次，约 52 分钟。
  - as_of：真实库有 1,867 / 28,450 篇（6.6%）施引论文的参考文献里出现 v1 之后 31 天以上才发表的论文（3,754 条，其中 97 条超过一年）。原因是正文取的是最新版本，日期却按 v1 记。`references()`、`baselines_in()`、`find_evidence` 因此会返回 as_of 时还不可见的 id。端到端测试没有发现，是因为合成数据里没有这种情况。
  - 语义：facts 的 Jaccard 聚类在实测中把"not slow"和"slow"并成一类（0.80），却把"slow"和"does not scale"分开（0.43）。shifts 在零变化的随机数据上，`became_component` 误报 30%（n=8），`recategorized` 误报 75%（n≥40）。label propagation 在有明确社区结构的图上塌缩成 1 个族。identity 的 GENERIC 过滤在归一化之后失效，比如 "diffusion models" 归一化成 "diffusion"，不在名单里。cs 标题前缀中，18.3% 已命名的论文和别的论文重名。
- **LLM 客户端：新版 `llm/client.py` 在传输层可靠**。实测并发 15 时零失败、零 JSON 解析失败，每路 23–32 tok/s，聚合 228 tok/s。但它不能作为唯一客户端，缺的有：
  - Paratera 没有退避、没有限流；
  - 超时后服务端仍在生成，车道却已释放（实测 3 s 墙后后台线程还活着）；
  - 台账默认不写盘，缺失败原因和 finish_reason；
  - 温度 0 不可复现（同一提示两次输出不同）；
  - 没有视觉调用和 arm purity 检查（这两项只在旧 kb_infra 里有）；
  - `compilescholar build` 不设车道宽度，`--workers 48` 实际只跑 4 路（`client.py:154` 默认值），other pass 的预计耗时从 14 h 变成约 87 h。
- **配置与路径：两套配置系统的取值互相矛盾**（conf/ 写本地并发 96、墙 900 s；configs/ 写 48；客户端默认 4、240 s）。新包里有 13 处硬编码路径或 `.research_tmp` 依赖，其中一处默认指向 NAS 的 IP。约 38 个行为参数绕过 configs/、直接读环境变量。干净 clone 下 `tests/test_config.py` 会失败，因为它读的 `config_test100_r1.json` 没有纳入 git 跟踪。

## 1. 模块判定表

### 1a. LLM 客户端与 JSON 修复

| 模块/功能 | 判定 | 证据 | 具体改法 | 工作量 |
|---|---|---|---|---|
| `llm/client.py` 本地调用 | 需重构 | 优点：并发 15 实测 0 失败，0/15 JSON 失败（95% 上界约 20%）；思考关闭写法正确（`client.py:175`）。问题：①车道默认 4（`:154`），`cli.py:144-173` 的 build 不设该值，只有 answer 在 `cli.py:54` 设；②超时后 daemon 线程不取消请求（`:97-140`），实测墙 3 s 返回 None，5 s 后线程仍在，而车道已经在 `:185` 释放；之后最多重试 4 次（`:178`），服务端可能同时跑同一提示的 4 份；③温度 0 下同一提示两次输出不同（2,820 / 2,825 字符） | 并入第 2 节的统一客户端：传输改用 httpx 并启用流式，墙到点就关连接，让服务端中止生成；车道持有到连接关闭为止；车道宽度、超时、重试次数全部从 configs 读，不经环境变量；加响应缓存作为可复现的手段 | L |
| `llm/client.py` Paratera | 需重构 | 两次尝试之间没有 sleep，不识别 429，不读 Retry-After（`:231-242`）；没有并发上限；没有限流 | 统一客户端里给每个 provider 配令牌桶（跨进程可选，复用 `sources/sciverse.py:108-135` 的 `_FileTokenBucket`）和指数退避加抖动（从 `kb_infra/llm.py:548` 的 `_cst_backoff_sleep` 迁来） | 含上 |
| 调用台账（新） | 需重构 | 只有设置 `LLM_CALL_LOG` 才写盘（`:69-75`），新包里没有任何地方设它；全局列表 `CALL_LOG` 只增不减（`:27,68`），千万级调用约占 5 GB 内存；`prompt_hash` 是整个 payload 的哈希（`:49-55`），拿不到模板身份；没有 http_status、error_class、finish_reason、reasoning_tokens、served model、stage/item | 台账始终开启，写到 `runs/<run_id>/llm_calls.jsonl`；字段见 §2；内存里只保留汇总计数 | 含上 |
| `llm/embedding.py` | 需优化 | 读响应体在调用线程里做，墙只在两个 chunk 之间检查（`embedding.py:42-44`），正是 09-29 在 chat 路径上修掉的那类挂死；失败时 attempt 写死为 3（`:59`）；返回值不带模型标签，`index/build.py:31` 的 dense 表也没有模型列 | 走统一传输；返回 `(vectors, model_label, dim)`；dense 表加 `model`、`dim` 列，查询时模型不一致直接报错 | M |
| `llm/jsonparse.py` | 需优化 | 只取第一个平衡括号块：输入 `'As in [1], output: {…}'` 时返回 `[1]`，错得悄无声息；`'Note {x}. {…}'` 返回 None；截断、尾逗号都返回 None（实测） | 依次尝试所有平衡块，取第一个能通过调用方 schema 校验的；把截断抢救（salvage）并进来；修尾逗号；可选启用服务端 guided JSON（GPUStack/vLLM 的 `response_format`，未验证） | S |
| `kb_compiler/records/common.salvage_json_records` / `call_json` | 迁入后退役 | 截断抢救有效：截断输出能救回 1 个完整对象（实测）；新代码 `compile/state/coarse.py:21` 仍然从 kb_compiler import 它；`call_json` 的重试加抢救是新 self/other pass 没有的（`extract/self_pass.py:108-112` 只调用一次，解析失败直接放弃） | salvage 迁到 `llm/jsonparse.py`；`call_json(validate=, salvage=)` 成为统一客户端的方法；`route_model` 改成 configs 里的 `provider:model` 写法 | S |
| `kb_infra/llm.py` | 退役（先迁出能力） | 值得迁出的：视觉调用（`:385`）、`check_arm_purity`（`:837`，含 min_calls）、`call_log_summary`（`:125`）、CST 的思考关闭写法（`:562` 文档串：只认 chat_template_kwargs，传 `thinking` 会报 422）、退避（`:548`）。必须丢掉的：TLS 全局关闭（`:31-33`）；`.env` 优先于进程环境（`:764`）；DeepSeek 失败时默认静默回落到 GLM-5-Turbo（`:261`），破坏 arm purity；`load_env` 有一段重复的死循环（`:36-52`）；Paratera 读响应体在主线程（`:357-366`） | 视觉、purity、汇总、CST 写法、退避迁入统一客户端；DeepSeek 直连和 Intern 通道不迁（新系统硬要求里没有它们） | M |
| `kb_infra/embedding.py` | 退役 | `embed_texts_robust` 失败时跨模型回落（GLM 1024 维 → CST 4096 维，`:170-195`）；Paratera 的 `embed_batch` 失败时填零向量（`kb_infra/llm.py:461`）。两者都是静默降级 | 不迁；embedding 只保留本地模型，带标签 | — |
| `retrieval/_see_llm.py` + `_see_llm_cases.py` + `extraction_runs.py` | 退役 | 用 openai SDK；provider 按"哪个环境变量先出现"自动选（`_see_llm.py:12-16`，`PERATERA` 拼写错误）；默认模型 gpt-4.1-mini（`:41`）；`load_dotenv` 会改写全局环境（`:36-39`）；没有台账；它服务的 sci-evo 案例抽取按 INTEGRATED §3 已退役 | 删除 | S |
| `retrieval/rerank.py:60-77`、`query_understanding.py:79-100`、`kb_compiler/records/figure_channel.py:55-75` | 改调统一客户端 | 各自直接调 urlopen，自己读环境变量，没有墙、没有台账 | rerank 随 sources/ 迁入后改调 `llm.embed`；figure_channel 改调 `llm.vision`；query_understanding 不并入（INTEGRATED §3） | S |

### 1b. 配置、路径、密钥

| 模块/功能 | 判定 | 证据 | 具体改法 | 工作量 |
|---|---|---|---|---|
| `core/secrets.py` | 直接可用 | 进程环境优先于 .env（`secrets.py:29-34`）；不记录值 | 保持；补一个闸门测试：除 `core/secrets.py` 和第三方适配器注入点外，禁止读 `*_API_KEY`、`*_TOKEN` | S |
| `core/paths.py` | 需优化 | 只有 REPO/data/cache/runs 四个根；`legacy_bench()` 指向 `.research_tmp`（`paths.py:28-30`），被 8 处调用 | 按 INTEGRATED §5 增加 `library()`、`derived()`、`benchmarks(name)`、`external(name)`、`results(name)`、`third_party(name)`、`cache(sub)`、`runs(run_id)`；外部资源（NAS 快照、asta-bench checkout、harness 工作目录、Sci-Hub 档案根、MinerU URL）一律走 `paths.resource(key)`，从 `configs/local.yaml` 的 `paths:` 读，没配就报错，不设默认值；迁移完成后删掉 `legacy_bench()` | M |
| `core/config.py` + `configs/` | 需重构 | 只有 answer/runtime/bench 三段（`config.py:21-55`）；`cmd_build` 完全不加载配置（`cli.py:144-173`）；LLM 的超时、车道、重试、种子、allowlist、台账，以及 Sciverse 限速，都只能经环境变量设置（新包里直接读 38 个环境变量名，见 §3） | 增加 `llm:`（providers、lanes、timeouts、retries、rate、cache、ledger）和 `build:`（各阶段参数）两段；`RunConfig` 作为对象传给客户端和各阶段，不再投影到 `os.environ`（`cli.py:54-58` 删除）；解析后的配置写进运行目录和阶段 manifest | M |
| `core/manifest.py` | 需优化 | `json.dump(…, open(p, "w"))` 不是原子写，文件句柄也没显式关闭（`manifest.py:70`）；`_ENV_KEYS`（`:22-25`）的存在说明行为参数都在环境变量里 | 改为临时文件加 `os.replace`；环境参数改由配置记录以后，`_ENV_KEYS` 只保留位置覆盖类变量 | S |
| `core/cutoff.py` | 需重构 | 语义是"严格早于某月"（`cutoff.py:5-9`），同年但月份未知的一律排除（`:63`）；`AsOf` 的语义是"按天、含当天"（`cognition/asof.py:6`）。两套时间语义并存，答题路径用前者，工具层用后者 | 统一用 `as_of: YYYY-MM-DD`（含当天）；旧的 `cutoff="2025-05"` 换算成 `as_of=2025-04-30`；可见性规则改为"日期区间的上界 ≤ T"，配合 registry 的日期精度（日/月/年，INTEGRATED §1）：只知道年份的论文在当年 12-31 之后才可见；线程局部变量加环境变量的传值方式改为显式参数 | M |
| `kb_compiler/config.py` + `conf/` | 退役 | 和 configs/ 取值冲突：`conf/base.yaml:24` 本地并发 96，`configs/base.yaml:21` 写 48，客户端默认 4；墙 900 s（`conf/base.yaml:27`），客户端默认 240 s；`conf/base.yaml:31-33` 指向 `.research_tmp` | 删除；有用的容量数字（`conf/capacity.yaml` 的 64/96/128 路实测）以注释形式迁进 `configs/base.yaml` 的 `llm.lanes` | S |
| `retrieval/_see_config.py` | 退役 | 这是 sci-evo 单篇抽取运行的 pydantic 配置（`_see_config.py:8-36`），和系统配置不是一个概念 | 随 extraction_runs 一起删除 | S |
| 根目录 `build.py` + `build_manifest.json` | 退役 | 12 处 `.research_tmp`（`build.py:44,74-165`）；LLM 阶段"产物存在即新鲜"（`build.py:257-260`），代码变了也不会重跑 | 删除，由 `compilescholar build` 取代（INTEGRATED §5） | S |
| 硬编码路径清单 | 需修 | 见 §3 | 见 §3 | 含上 |

### 1c. 阶段契约

| 模块/功能 | 判定 | 证据 | 具体改法 | 工作量 |
|---|---|---|---|---|
| `dfc/store.py` digest 与过期判定 | 需重构 | ①代码哈希只按 CODE 表列的文件算（`store.py:28-36`）。探针按 import 闭包比对，漏掉的有：extract 缺 `llm/client.py`、`llm/jsonparse.py`、`compile/skeleton/proposes.py`；index 缺 `llm/client.py`、`llm/embedding.py`；citations 缺 `sources/refgraph.py`；documents 缺 `compile/skeleton/bib.py`。②digest 由参数、上游 digest、代码三部分算出（`:81-83`），不含数据内容：探针新增一个 parquet 后 documents 新存 1 篇，digest 不变，citations 仍判新鲜。③embedding 模型名不在 index 的参数里。schema 变动经 extract digest 能传到 index，这一点没问题 | 代码哈希改成从阶段入口模块自动算 import 闭包（AST 静态分析，探针脚本约 30 行可直接复用），不再手工维护 CODE 表；参数里加上 provider、model、模板 sha、采样参数、embedding 模型标签；digest 加入输入指纹：上游表的行数加一个内容摘要（如 `sum(hash(id, updated_at))`），原始输入则用文件清单加 mtime/size/sha | M |
| 续跑与重建语义 | 需重构 | 各阶段都按"已完成集合"跳过（`citations/build.py:80-84`、`documents/build.py:54`、`extract/build.py:103,125,133`）。探针：标记 `citations/resolve.py` 已改、再执行 build，重新解析 0 条，但 manifest 判为新鲜。改参数续跑（n_deep 2→0）时，manifest 写 0，而 2 条 T2 结果仍在表里 | 已完成标记带上 `(digest, item_key)`：digest 一变，旧标记作废，不能靠 `INSERT OR REPLACE` 续写；默认是全量重建到新文件后原子替换；增量续跑只允许在 digest 不变时进行；把"重建"和"续跑"拆成 `build --rebuild` 和 `build --resume` 两个明确的命令 | M |
| 失败标记 | 需重构 | `extract/build.py:119-120,153-154` 不管成功失败都写 done；探针：LLM 全失败 → done_self=3、done_pairs=4、statements=0，阶段新鲜 | 只有成功才标 done；失败写入 `failures(item, error_class, attempts, last_ts)`；连续失败达到阈值（熔断）时阶段报错退出，不写 manifest；manifest 的 counts 必须包含 failed，failed > 0 时 status 显示"不完整" | S |
| manifest 原子性 | 需修 | `store.py:103` 直接 `json.dump(open(p, "w"))`；探针：一个进程循环写，另一个循环读，816 次读里 318 次 JSONDecodeError | 临时文件加 `os.replace`（Windows 下同一卷内是原子的）；读失败时重试一次 | S |
| 多进程同时构建 | 需修 | 没有阶段锁；`statements` 没有唯一键（`extract/schema.py:92-94`）；探针两进程并跑同一 extract：返回码都是 0，14 条表述中 7 组重复。`connect` 没设 busy_timeout，用的是 Python 默认 5 s（`store.py:49-57`） | 每个阶段加一个文件锁（filelock 已在依赖里，`sources/sciverse.py:115`），拿不到锁就报错退出；statements 加自然键唯一索引 `(speaker, kind, about, facet, quote_sha, pass)`；`PRAGMA busy_timeout=30000`；数据目录只允许放本地盘（WAL 不支持网络盘） | S |
| status 递归 | 直接可用（顺手简化） | 探针：`require_fresh` 检查 4 个阶段，共调用 15 次 status()、15 轮代码哈希、26 次读 manifest，耗时 9 ms，不是性能问题；`require_fresh` 对每个阶段算两次 status（`store.py:130`） | 在一次调用内做记忆化即可 | S |
| `cognition` 阶段占位 | 需修 | `STAGES`、`UPSTREAM`、`CODE` 里列了 cognition（`store.py:22-35`），但没有任何代码会构建它；`build status` 显示"not built" | 按 §5 物化后成为真正的阶段；物化之前先从 STAGES 去掉 | S |

### 1d. 编译层 `cognition/*`

| 模块/功能 | 判定 | 证据 | 具体改法 | 工作量 |
|---|---|---|---|---|
| `asof.py` | 需重构 | ①`references()` 只检查施引论文是否可见（`asof.py:82-86`）。真实库里 1,867 篇的参考文献中有 3,754 条比施引论文的 v1 晚 31 天以上，原因是 IdeaForecast 的 Markdown 和 `sources/arxiv_html.py:47` 的无版本号 URL 取到的都是最新版本，日期却按 v1 记。②`co_cited` 的日期过滤是多余的（同一句子所有行的日期相同），它真正的问题是返回的被引对象可能尚不可见；`cites` 缺 `sentence_id` 索引（`citations/build.py:35-36`），查询计划是 SCAN，真实库每次 0.15 s（冷启动 4.6 s）。③`T` 只检查长度为 10（`:18-19`），"2025/01/01"、"2025-13-45"都能通过。④每个工具调用都新开 3 个连接（`tools/api.py` 中每个工具都执行 `AsOf(as_of)`）。⑤id 格式写死为 `paper:<arxiv_id>`，代码里用 `obj[6:]` 切片（`:37` 等），要换成 `paper_id` | AsOf 改成物化表（§5）之上的查询层；所有返回对象 id 的方法都加"被引对象在 T 可见"过滤（stub 也要带 `first_cited_date`，只在首次被引日期 ≤ T 后可见）；`cites(sentence_id)` 加索引；T 用 `date.fromisoformat` 校验；连接池按进程复用；版本问题从源头修：引用句的日期取它所在文本版本的日期（registry 记录版本日期），或者只用 v1 文本（INTEGRATED 阶段 B 的 acquire 负责） | M |
| `identity.py` | 替换 | ①GENERIC 检查放在 `norm_name` 之后（`identity.py:36,46`），而 `norm_name` 会去掉 model/network 等词（`:23`），结果实测 "diffusion models"→"diffusion"、"graph neural networks"→"graph neural"、"large language models"→"large language" 都不在 GENERIC 里。②真实库 cs.LG/CL/CV/AI 有 108,618 篇用"名字: 副标题"格式命名，其中 6,143 个名字被 2 篇以上使用，涉及 19,857 篇（18.3%），比如 PRISM 约 100 篇，SAGE、TRACE、STAR、VISTA、MARS 也很多。`resolve` 的"证据多一倍就归它"规则（`:50-57`）在第三方说法里（如"X improves SAGE"）会把名字给证据最多的那篇，随着 T 变化归属还会翻转。③每次构造都全量读 self 和 other 表述（`:32,42`），17.5 万条时 1.35 s。④`aliases()` 遍历所有名字（`:59-60`），lineage 的回退路径里对同句每个对象都调一次（`lineage.py:52-55`），代价是平方级 | 改成提及级链接（mention-level linking）：规则做候选召回，LLM 只裁定有歧义的提及（§4.3），结果存进 `mention_link` 表并带日期；查询"某名字在 T 时指谁"时，在表上按日期聚合 | L |
| `families.py` | 需重构 | ①label propagation 是确定的，但结果取决于 id 的字典序：同一张图把节点改名后，族数从 6 变成 17（实测）；有社区结构的图（10×50，p_in=0.15，p_out=0.01）塌缩成 1 个族；两个团加一个枢纽并成 1 族；合成数据上 535 个对象的范围得到 1 个 483 成员的族。②每个句子调一次 `co_cited`（`:39`），field_map 的范围在合成数据上要 20,982 次，在真实库上不加索引约 52 分钟。③`support` 的计算对每个成员遍历全部边（`:98`），平方级。④分类短语只做小写归一（`:47`），"GNNs" 和 "graph neural networks" 被当成不同类别 | 共现边改为一条 SQL 自连接预聚合（`cites a JOIN cites b ON a.sentence_id=b.sentence_id`），离线算好存表；按度数做枢纽降权（`w/√(deg_a·deg_b)`）；算法换成固定种子的 Leiden（需新增依赖 leidenalg，或自己实现 Louvain），节点按稳定键排序；族按快照物化（§5）；族命名与边界成员判定交给 LLM（§4.4）；类别短语先经 §4.2 规范化 | L |
| `lineage.py` | 需重构 | 逻辑正确：时间不一致的边被丢弃（`lineage.py:36-38`），self 和 third 两类分开存；但每次调用全量扫描（17.5 万条时 2.74 s）；`paper_profile` 里算了两遍（`tools/api.py:224-225`）；回退匹配走 `aliases()`（平方级） | 物化为 `lineage_edge` 表（§5）；回退匹配改用 `mention_link` | M |
| `profiles.py` | 需优化 | 语义上可用：按句子去重（`profiles.py:44-47`），描述按时间分散抽取；按对象查询时走索引，hub 对象 0.02 s；但早期/晚期的切分点取决于 T（`:72-79`），换一个 T 结果就重算 | 份额改从 `reception_daily` 累计聚合（§5）；shift 部分统一由 shifts.py 负责 | S |
| `facts.py` | 替换 | ①Jaccard ≥ 0.5 聚类：实测 "slow on large graphs" 和 "does not scale to large graphs" 为 0.43，不合并；"requires large labeled datasets" 和 "needs a lot of annotated data" 为 0.00；而 "is slow" 和 "is not slow" 为 0.80，"without fine-tuning" 和 "with fine-tuning" 为 0.83，"outperforms" 和 "underperforms" 为 0.50，都被合并。②NEG 正则把 "no additional parameters are needed"、"not only faster but also more accurate"、"trained without labels" 都当成否定。③独立性只看姓氏（`corpus/papers.py:12` 只存姓氏）：两篇随机的 cs.LG 论文有 1.5% 被判为同组；100 篇互不相关的论文只算出 82.4 个独立来源；3 个不同的 "Wang" 团队被并成 1 组；作者缺失时每篇各算一个独立来源（`facts.py:49-58`，`set(a) or {s}`）；贪心合并不满足传递性（a–b 共享 Y、b–c 共享 Z，三篇被并成 1 组）。④聚类对每条表述扫描全部簇（`:68-75`），平方级 | 聚类改为"向量召回 + LLM 裁定"，同时输出立场（same/opposite/narrower），取代 NEG（§4.1）；独立性改用 registry 的作者 id（OpenAlex author id，或规范化的全名加机构），以共享作者 id 的连通分量计数（这是结构性判断，可以用规则）；作者缺失时标记为"独立性未知"，不计入 consensus | L |
| `shifts.py` | 需重构 | 固定阈值加二分切分，在零变化的模拟上（份额不随时间变，4 个同义类别）：`became_component` 在 n=8/12/20/40 时分别误报 30.4% / 16.1% / 14.8% / 7.8%；`became_baseline` 在 n=8 时 22.8%；`recategorized` 在 n=8 时 7.9%，n=40 时 76.1%，n=80 时 75.1%。也就是说 MIN_N=8（`shifts.py:22`）太小，recategorized 基本上是在报同义词噪声。另外，切分点取决于 T，事件日期不能单调累积 | 候选检测改为固定时间窗（年或半年）对比之前的累计，用 Fisher/二项检验并做多重比较校正（统计判断，规则可以做）；类别先规范化；通过检验的候选事件交给 LLM 对照证据句确认（§4.5）；事件只依赖日期 ≤ d 的证据，可以按事件时间物化（§5）。**这一条改变了现有行为定义**（二分 → 时间窗），需要用户确认 | M |
| `comparisons.py` | 需优化 | `baselines_in` 返回 `about` 时不检查可见性（`comparisons.py:34-41`），`tools/api.py:251-253` 因此可能返回 T 时不可见的论文 id；按 speaker 查询时没有 function 索引 | 加可见性过滤；在 `statements(about, function, date)` 上建索引或物化 `comparison_edge` 表 | S |

### 1e. `compile/state`、`compile/skeleton`、`kb_compiler/views/field_state.py`

| 模块/功能 | 判定 | 证据 | 具体改法 | 工作量 |
|---|---|---|---|---|
| `compile/state/field_state.py` | 需重构（迁移） | 这是 LLM 族归纳（PROPOSE/合并/FACTS/PROBLEMS），唯一的生产消费者是 `eval/field.py:22` 的领域层检验，另有 `experiments/pilots/p2_reception_vs_self.py`；`cognition/families.py:14` 写明"不用它决定成员" | 作为冻结的评测对照组整体迁到 `eval/arms/field_state_llm.py`，保住已预注册的领域层结果可复现；其中"族命名 + 定义"的提示词抽出来，复用到 §4.4 的族命名步骤 | S |
| `compile/state/coarse.py` | 退役 | 只被 field_state 使用；`coarse.py:21` 运行时从 kb_compiler import salvage；CLI 里用到的 argparse/json/sys 都没有 import（`coarse.py:106,114,116`，运行即 NameError，AST 检查确认）；`extract/self_pass.py:11` 写明它已被 T1 取代 | 并入上面的 eval 对照组（只给 field_state 用）；salvage 改从 `llm/jsonparse` import；删掉 CLI | S |
| `kb_compiler/views/field_state.py` | 退役 | 和 `compile/state/field_state.py` 只差 2 行 import（diff 实测） | 删除；`tests/fixtures/characterize/make_goldens_components.py:181` 里的 'old' 分支一并删 | S |
| `compile/skeleton/bib.py` | 直接可用（换位置） | 被 `citations/markdown.py:21`、`citations/html.py:14` 使用；`tests/test_skeleton_bib.py` 覆盖 | 改名为 `citations/bib.py`，功能不变；顺带补进 documents 阶段的代码哈希（现在漏了） | S |
| `compile/skeleton/resolve.py` | 拆分 | `entry_title_year` 被 `citations/resolve.py:16` 使用；`Resolver`（基于 KB 标题、面向综述）只有 experiments 和测试在用；文档串列了 OpenAlex/Crossref 两个阶段（`resolve.py:8-9`），代码里并没有实现（`:82-106`） | `entry_title_year` 移进 `citations/resolve.py`；`Resolver` 退役；DOI/OpenAlex 解析按 INTEGRATED §3 在 citations 里实现 | S |
| `compile/skeleton/proposes.py` | 拆分 | `validate` 和 `GENERIC` 被 `extract/self_pass.py:18`、`cognition/identity.py:15` 使用；`extract()` 只有 experiments 在用；`PROPOSAL`、`RELIANCE` 两个正则（`proposes.py:40-48`）在判断"这句话是不是提出新方法"，属于语义判断 | `validate` 拆成两部分：结构检查（原句子串、名字出现在句中）留作规则；"是否在提出新方法"交给抽取提示词，并在 §4.3 的 LLM 裁定里复核，正则改成仅作召回提示，不再一票否决。`GENERIC` 移进 identity 的候选过滤，并改为在归一化之前、对原词形匹配 | S |

### 1f. 测试

| 对象 | 判定 | 证据 | 具体改法 | 工作量 |
|---|---|---|---|---|
| 现有测试整体 | 需优化 | 251 个全部通过，用时 30.3 s（需加 `-p no:logfire`，环境里缺 `executing` 模块）；44 个测试文件中 20 个测的是旧包（kb_compiler/kb_infra）；`test_llm_ledger_fields.py`、`test_purity_min_calls.py` 测的是旧的 `kb_infra.llm`，新客户端没有对应测试 | 随能力迁移把这些测试改测新位置；新客户端补测试（见下） | M |
| 数据规模 | 缺失 | 最大的合成数据：`test_dfc_endtoend.py` 3 篇，`test_cognition_compile.py` 6 篇、9 条表述；没有任何超过 1k 行的测试 | 加一个规模冒烟测试：合成 10 万条表述，按耗时预算断言（物化后单个工具调用 < 200 ms），并用 `EXPLAIN QUERY PLAN` 断言关键查询走索引（确定、便宜、可在 CI 跑） | M |
| 并发与契约 | 缺失 | 本次 8 个探针（见 1c）都没有对应的仓库测试 | 把探针改写成正式测试，断言"修复后的行为"：代码改动后 rebuild 会重算；数据变化会传播过期；失败不标 done；manifest 原子；两进程构建时第二个报错 | S |
| 失败与重试 | 缺失 | — | 用本地假 HTTP 服务（标准库 http.server）测客户端：429 带 Retry-After、5xx、慢速滴流、完全挂住、截断 JSON、400 不重试；断言车道计数、台账字段、熔断 | M |
| as_of 泄漏 | 缺失 | e2e 测试声称"每个 id 在 as_of 时可见"，但数据里没有"参考文献晚于 v1"的情况 | fixture 里加一篇施引论文，其参考文献 v1 晚于施引论文 v1；断言 14 个工具都不返回它（或只返回带标记的 stub） | S |
| 跨领域 | 缺失 | 所有 cognition 测试都是 arXiv id、按天精度的日期 | 加：只有 DOI、日期只到年的论文；作者缺失的 stub；非 arXiv 的 paper_id 格式 | S |
| 干净 clone | 失败 | `tests/test_config.py:11` 读 `.research_tmp/.../arm_vnext/config_test100_r1.json`，该文件未被跟踪（`git ls-files` 为空） | 随 INTEGRATED §5 迁到 `results/cs2-test100-v9b-20261004/`，纳入跟踪 | S |
| golden 生成器失效 | 需处理 | `make_goldens*.py` 的 'old' 分支依赖 `.research_tmp/.../_shared/tools` 和 kb_compiler（`_characterize_impl.py:22,39-40`，`make_goldens_components.py:22,180-181`，`make_goldens_scoring.py:17`）；'new' 分支的 field_state 用例会经过 `coarse.py:21` import kb_compiler，因此**kb_compiler 一删，表征测试就会挂** | ①先把 salvage 迁到 `llm/jsonparse`，再删 kb_compiler；②goldens 视为冻结数据：`tests/fixtures/characterize/GOLDENS.sha256` 记录每个 golden 的哈希和生成时的 tag（cs2-test-v9b-final），并用一个测试断言哈希不变，防止有人误重生成；③删掉生成器里的 'old' 分支（tag 里有历史可查），生成器只保留 'new' 加 `intended_changes.py` 这条"有意变更"的路径；④每一次有意变更都在 `intended_changes.py` 的 INTENDED 表里登记 | S |

## 2. 合并后的唯一 LLM 客户端（`compilescholar/llm/`）

### 2a. 实测（本地 27B，共 18 次调用）

| 项 | 结果 |
|---|---|
| 并发 15（8 次 T1 自述遍 + 7 次他述遍，每次 24 对，均为真实论文和引用句） | HTTP 失败 0/15；JSON 解析失败 0/15；他述遍 168 对解析出 167 对（1 对词表越界）；自述遍 46 条中 2 条原句校验不通过 |
| 吞吐 | 每路 23–32 tok/s；15 路聚合 228 tok/s（108 s 内生成 24,583 个 completion token）；他述遍单次调用 87–108 s，约 2,738 个 prompt token、2,896 个 completion token；T1 单次 18–29 s，约 533 / 539 token |
| 超时 | `LLM_WALL_TIMEOUT=3` 时 3.0 s 返回 None；5 s 后后台读线程仍然存活，说明服务端还在生成，而车道已经释放 |
| 可复现 | 同一提示、温度 0、不设 seed，连续调用两次，输出不同（2,820 / 2,825 字符） |
| 台账 | 18 行（17 成功、1 失败）；字段有 ts、run_id、caller、provider、model、ok、latency_ms、prompt/completion_tokens、attempt、prompt_hash、fallback_for；**没有**失败原因、http 状态、finish_reason、温度、模板 id |

### 2b. 三套实现对比

| 能力 | `compilescholar/llm` | `kb_infra` | `retrieval/_see_llm` |
|---|---|---|---|
| 并发控制 | 两条车道（普通 / 大输出），首次调用时才创建（`client.py:148-157`） | 模块导入时就创建信号量，每个 provider 一个（`llm.py:544,648,651`） | 无 |
| 重试 / 超时 | 本地：4 次，固定退避，有墙；Paratera：2 次，无退避 | CST/Intern：5 次，指数退避加抖动加 Retry-After（`llm.py:548-559`）；本地同新版 | openai SDK 默认 |
| 限流 | 无 | 无（只有并发上限） | 无 |
| 台账 | 有，但默认不写盘 | 同新版，另有 `call_log_summary` | 无 |
| 视觉 | 无 | `call_paratera_vision`（`llm.py:385`） | 无 |
| arm purity | 只有 allowlist 闸 | allowlist 加 `check_arm_purity`（含 min_calls） | 无 |
| embedding 标签 | 无 | `embed_texts_robust` 返回 provider 标签，但会跨模型回落 | — |
| JSON 修复 | 首个平衡块加控制字符修复 | 同新版，另有 `common.salvage_json_records` 截断抢救 | `response_format=json_object` |
| 密钥 | `core.secrets`（进程环境优先） | .env 优先（违反要求） | `load_dotenv` 改写全局环境 |
| TLS | 开启 | 全局关闭 | SDK 默认 |

### 2c. 唯一客户端应包含的能力与出处

| 能力 | 设计 | 取自 |
|---|---|---|
| Provider 注册 | 在 `configs/base.yaml` 的 `llm.providers` 下声明 `local`、`paratera`、`cst`、`vision`：base_url 的键名、模型列表、思考关闭写法（qwen 用 `chat_template_kwargs`，DeepSeek 族用 `thinking: disabled`，CST 不传 `thinking` 以免 422）、车道数、限速 | 思考关闭写法：`client.py:175,222-226` 与 `kb_infra/llm.py:562` 文档串 |
| 传输 | httpx 客户端（已安装 0.28.1），连接池大小等于车道数；本地 provider 设 `trust_env=False`（本机系统代理指向 127.*，urllib 对本地地址能绕过，httpx 未实测；历史上 GPUStack 出过 httpx 被注册表代理劫持的问题）；分别设 connect/read/pool 超时，另设整体墙；**使用 stream=true，墙到点就关闭连接**，让服务端中止生成（vLLM 在客户端断开时会中止请求，需在 GPUStack 上实测确认） | 墙的概念来自 `client.py:97-140`；改为可取消的实现 |
| 并发 | 每个 provider 两条车道（普通 / 大输出），车道持有到连接真正关闭为止；车道宽度从配置读；`build` 和 `answer` 共用同一份配置 | `client.py:148-157` |
| 限流 | 每个 provider 一个令牌桶（每分钟请求数，Paratera 再加每分钟 token 数）；跨进程时用文件锁共享 | `sources/sciverse.py:86-135` |
| 重试 | 可重试的错误（连接错误、超时、429、5xx）：指数退避加抖动，遵守 Retry-After；不可重试的（400/401/403/404/422）：直接返回并记下状态码和响应开头；内容层重试（JSON 解析或 schema 校验失败）单独计数，次数可配；熔断：连续 N 次失败就抛 `ChannelDead`，阶段因此报错停下，而不是把条目标成完成 | `kb_infra/llm.py:548-559,620-645` |
| 台账 | 始终开启，写到 `runs/<run_id>/llm_calls.jsonl`。每次调用一行，字段：ts、run_id、stage、task、item_key、provider、model、served_model（响应里的 model）、template_id、template_sha、payload_sha、temperature、max_tokens、seed、thinking、ok、http_status、error_class、attempt、latency_ms、prompt_tokens、completion_tokens、reasoning_tokens、finish_reason、cached。内存里只保留汇总 | 字段骨架来自 `client.py:58-75`；汇总来自 `kb_infra/llm.py:125` |
| 可复现 | 温度 0 不可复现（实测），所以把**响应缓存**当作复现手段：键为 `sha(provider, model, messages, temperature, max_tokens, seed, schema)`，原始文本存进 `cache/llm/responses.sqlite`；重建时命中缓存，结果逐字节一致；阶段 digest 包含模板 sha、模型和采样参数；默认带 seed | 新增 |
| arm purity | 每次运行在配置里声明允许的 `(provider, model)`；调用前先过 allowlist 闸；运行结束时用 `check_arm_purity(min_calls=…)` 扫一遍台账，结果写进 manifest | allowlist：`client.py:78-90`；purity：`kb_infra/llm.py:837-883` |
| 视觉 | `vision(prompt, images, model)`：OpenAI 兼容的多模态 content 数组，走同一套传输、台账和限流；台账里记图片数和字节数 | `kb_infra/llm.py:385-446` |
| JSON | `call_json(prompt, schema/validate, salvage=(item_key, wrapper), retries)`：依次尝试每个平衡块，返回第一个通过校验的；截断时抢救完整子对象；修尾逗号；可选服务端 guided JSON | `llm/jsonparse.py` 加 `kb_compiler/records/common.py:45-92,113-137` |
| Embedding | `embed(texts, model_spec) -> (vectors, model_label, dim)`；只用本地模型，不跨模型回落；维度不一致就报错；同样有台账和墙 | `llm/embedding.py`（读响应体改走统一传输） |
| 配置 | 客户端从 `RunConfig.llm` 构造，不读环境变量；环境变量只用于密钥（经 `core.secrets`） | `core/config.py` 扩展 |

预计 1 个模块、约 600 行，替换现有 3 套共约 1,400 行。工作量 L（含测试）。

## 3. 配置与路径：硬编码清单与统一方案

### 3a. 新包 `compilescholar`（运行时生效）

| 位置 | 内容 | 改为 |
|---|---|---|
| `core/paths.py:28-30` | `legacy_bench()` → `.research_tmp/experiments/benchmarks` | 迁移完成后删除；调用方改用 `paths.benchmarks(name)` |
| `sources/arxiv_snapshot.py:20` | 默认值是 NAS 的 UNC 路径 `\\192.168.199.138\Share400T\…\arxiv-metadata-oai-snapshot.json` | `paths.resource("arxiv_snapshot")`，从 `configs/local.yaml` 读，不设默认值 |
| `sources/refgraph.py:30` | `legacy_bench()/_shared/refgraph_cache` | `paths.cache("refgraph")` |
| `sources/refgraph.py:56` | `~/.pace_<src>.json`（用户主目录下的状态文件） | `paths.cache("pace")/<src>.json` |
| `eval/cs2/judge.py:60-61` | `.research_tmp/scratch/ai2_baseline_2026-09-08/asta/asta-bench-main` | `paths.third_party("asta-bench")` |
| `eval/cs2/scoring.py:45` | `legacy_bench()/scholarqa_multi/<rubric>` | `paths.benchmarks("cs2")/rubrics/` |
| `eval/dsb.py:28` | `legacy_bench()/deepscholar/dsb` | `paths.benchmarks("dsb")` |
| `eval/dsb.py:105` | `.research_tmp/review_1002/p6/oracle_inputs.json` | `paths.benchmarks("dsb")/oracle_inputs.json` |
| `eval/field.py:27` | `legacy_bench()/cs2/base_kb_v2` | `paths.benchmarks("field")` |
| `baselines/memorized.py:24` | `legacy_bench()/cs2` | `paths.benchmarks("cs2")/arms` |
| `baselines/harness/runner.py:97` | `C:\cs2_harness_cwd` | `paths.resource("harness_cwd")` |
| `baselines/harness/runner.py:130` | `legacy_bench()/cs2/arm_harness` | `paths.benchmarks("cs2")/arms/harness` |
| `configs/base.yaml:13` | `kb_dir: .research_tmp/…/base_kb_v2` | `data/benchmarks/cs2/kb_v2` |
| `documents/build.py:32-33` | `data/external/ideaforecast`（已经过 paths） | 改成 `paths.external("ideaforecast")`，统一写法 |

只在文档串里提到、不影响运行的：`compile/skeleton/__init__.py:2`、`documents/tables.py:3`、`__init__.py:3-4`，随迁移一并更新。

### 3b. 旧包与测试

- `build.py:44,74,85-86,99,109,122,132,142,153,165`，`conf/base.yaml:31-33`，`kb_compiler/records/registry_growth.py:50`（`_FAIL_DUMP` 写到 `.research_tmp`），`kb_compiler/config.py:27-28`，`kb_infra/llm.py:36-47`（.env 路径由 `__file__` 推出）：随旧包退役。
- `tests/test_config.py:11-12,24`，`tests/_characterize_impl.py:22`，`tests/fixtures/characterize/make_goldens_components.py:22`，`make_goldens_scoring.py:17`：前者随 INTEGRATED §5 迁移表改路径，后三者按 §1f 处理。
- `retrieval/acquisition_chain.py:306-355`、`processing.py:62`、`_see_upstream.py:84-133`：用 `SCIEVO_*` 环境变量加 `load_dotenv`。迁入 `acquire/`、`sources/` 时改为读配置（MinerU URL、档案根目录）和 `core.secrets`（MinIO/Postgres 密钥）。

### 3c. 环境变量当配置用（新包直接读的 38 个名字）

- **行为参数，应移入 configs**：LOCAL_MAX_CONCURRENT、LOCAL_LARGE_MAX_CONCURRENT、LOCAL_SOCK_TIMEOUT、LOCAL_MAX_ATTEMPTS、LLM_WALL_TIMEOUT、LLM_SOCK_TIMEOUT、LLM_SEED、LLM_PROVIDER_ALLOWLIST、LLM_CALL_LOG、LLM_RUN_ID、LLM_CALLER、ANSWER_MODEL、FIELD_JUDGE、SCIVERSE_RATE_PER_MIN、SCIVERSE_MAX_WAIT_S、SCIVERSE_SHARED_BUCKET、KNOWLEDGE_CUTOFF、DFC_AS_OF、RETRIEVAL_MCP_MODE/MANIFEST/TEXTS、HARNESS_CWD、HARNESS_ANTHROPIC_BASE_URL、CS_TLS_INSECURE、CC_PROXY_DUMP。
- **位置覆盖，可以保留**：CS_ROOT、CS_DATA、CS_CACHE、CS_RUNS；其余的 CS_ARXIV_SNAPSHOT、CS_ASTABENCH、CS_DSB、CS_FIELD_KB、CS_REFGRAPH_CACHE 改用 `paths.resource`。
- **向第三方注入密钥**：`eval/cs2/judge.py:66-73`、`eval/dsb.py:41-42` 把 Paratera 密钥写进 `os.environ`，inspect/lotus 只能这样读，无法避免。改法是限定在子进程或上下文管理器内注入，用完恢复。

### 3d. 统一方案

1. `configs/base.yaml`（入库）< `configs/local.yaml`（本机，gitignore，放外部资源路径和车道宽度）< 实验配置 < `--set`。解析后的配置写进每个运行目录和阶段 manifest。
2. `core/paths.py` 是唯一的路径入口；`paths.resource(key)` 没配置就报错，绝不回落到硬编码默认值。
3. `core/secrets.py` 是唯一的密钥入口。
4. 在仓库里加一个闸门测试，扫描 `src/compilescholar`：禁止出现 `.research_tmp`、盘符绝对路径、UNC 路径；`os.environ` 只允许出现在 `core/` 和明确登记过的第三方注入点。

## 4. LLM 化设计（复杂语义改由 LLM 裁定）

共同框架：
- **规则或统计做候选召回，LLM 做裁定**。裁定结果写进带日期的表（§5），记录 `llm_call_sha`，可以审计。
- 每类裁定一个模板，模板 sha 进入阶段 digest。
- 批处理：每次调用裁定一组（一个锚点加若干候选）。
- 缓存键为 `sha(template_sha, model, sampling, 排好序的输入 id, 每个输入的文本 sha)`；增量构建时只裁定新增的输入。
- 验证：每类各取 ≥ 200 条样本，用 Paratera 做双模型标注，一致率必须达到阈值才能启用（对应 INTEGRATED §6 阶段 D 的闸门）。
- 成本基准：本地 27B 在 48 路并发时聚合约 720 tok/s（用户给的数字）；实测单路 23–32 tok/s。下文的"小时数"只按 completion token 估算，各类裁定的比例都是假设，需要用试点实测替换。

### 4.1 族级事实：同一说法 / 相反说法 / 条件不同（取代 Jaccard 聚类和 NEG 正则）

- **召回**：同一个 `about` 对象（或同一个族）、同一个 facet 内，用表述向量（与 index 用同一个 embedding 模型）取 kNN top-15、余弦 ≥ 0.70 作为候选；没有候选的表述单独成簇，不调 LLM。
- **裁定**：每次调用给 1 个锚点（簇的代表表述）和 ≤ 15 个候选，每条都附原句。对每个候选输出 `same | opposite | narrower(条件) | broader | unrelated`，以及一句"条件差异"说明。
- **用法**：`same` 和 `narrower` 归入同一簇；`opposite` 用来判"contested"，取代 NEG；条件差异写进事实的 `conditions` 字段。
- **成本**：单次约 1,200 prompt / 300 completion token。
  - 现有 cs.LG 范围（30.2 万对 → 约 36 万条他述）：假设 40% 属于事实 facet、其中一半有候选，约 4.8k 次调用、1.4M completion token，约 0.6 h；
  - 千万级：约 13 万次调用，约 15 h（离线一次性；之后增量只算新增）。

### 4.2 类别短语规范化（recategorized 和族命名的前提）

- **召回**：对全部类别短语去重、做 embedding，再层次聚类（阈值 0.80）。
- **裁定**：每次调用给 ≤ 50 个短语，LLM 输出若干规范类别及其成员，并标出"伞形词"（例如 "deep learning methods"，不携带族信息，取代 `families.py:56` 的"超过 50 个成员即视为伞形词"规则）。
- **输出**：`category_canon(phrase_norm, canon_id)`。
- **成本**：每次约 600 completion token。千万级时假设有 100 万个不同短语，约 2 万次调用，约 4.6 h。

### 4.3 方法身份：提及到论文的链接（取代归一化名字加计数权重）

- **召回（规则，结构性）**：先在归一化之前，对原词形做 GENERIC 和伞形词过滤；再按精确名、别名、缩写展开、编辑距离 ≤ 1，以及名字加上下文的向量 top-5 收集候选论文。候选论文必须在**提及日期**时已经可见。
- **不调 LLM 的情况**：引用句里的名字直接挂在它引用的条目上（`about` 已经由引用解析确定），属于结构信息；只有一个候选、而且是自述"提出"的名字。
- **调 LLM 的情况**：≥ 2 个候选（实测有 18.3% 的命名论文重名），第三方 `builds_on` 的名字不在同一句的引用里，或者名字接近伞形词。
- **裁定**：每次调用 10 个提及，每个附句子、施引日期、候选论文的标题、日期和摘要首句。输出候选编号，或者 `generic`、`none`。
- **成本**：每次约 800 / 150 token。歧义提及按他述的 5–10% 估算，现有范围约 2–4k 次调用（< 0.3 h），千万级约 5 万次（约 3 h）。
- **物化**：每个提及的决定与时间无关（候选集已经限定为提及日期时可见的论文），所以 `AsOf(T)` 只需按日期过滤 `mention_link`。

### 4.4 族：命名与边界成员

- **召回**：族的成员仍由共现图聚类决定（结构性），按 §1d 的改法做。
- **裁定**：每个族快照中，如果成员变化超过 20%，或者是新出现的族，调用一次：给成员标题、自述贡献、规范类别，LLM 输出族名、一句话定义，以及"不像本族"的成员（只标记，不删除；被标记的成员在工具输出里降权）。提示词沿用 `compile/state/field_state.py` 的 PROPOSE 段。
- **成本**：每个族快照约 2k / 400 token；月度快照下每月新增或变化的族约几百个，总计不到 1 h。

### 4.5 认识变迁：事件是否成立

- **候选（统计）**：按 §1d 的时间窗检验通过的 `became_component`、`became_baseline`、`recategorized`（基于规范类别），以及有 `replaces` 边的 `superseded`。
- **裁定**：每个候选给前后各 ≤ 10 条证据句，LLM 输出 `holds | not_supported`、一句描述和支撑句 id。
- **成本**：候选数量取决于检验阈值；按对象数的 1–3% 估算，现有范围在 1k 次以内，千万级约 3 万次（约 2 h）。

### 4.6 不交给 LLM 的部分

以下仍用规则：引用标记到条目的对应；DOI/arXiv 识别；日期过滤与可见性；原句子串检查；共现计数与图聚类；统计检验；按共享作者 id 判断独立性（前提是作者身份已经解析好）。

## 5. 物化设计（编译阶段离线算好，在线只做过滤和聚合）

**原则**：编译层的每个对象都是一组带日期证据的函数。凡是"证据只增不减，结论单调"的对象，按**事件时间**存成追加式事件，`as_of T` 精确等于 `WHERE valid_from <= T`，不需要网格。只有"全局重算才会变"的对象（族），才按 **as_of 网格**存快照。

| 对象 | 存法 | 表（`data/derived/cognition.sqlite`） | as_of 查询 | 增量更新 |
|---|---|---|---|---|
| 方法身份 | 事件 | `mention_link(mention_id PK, sentence_id, citing, date, surface, name_norm, paper_id, decided_by rule\|llm, llm_call_sha)`，索引 `(name_norm, date)`、`(paper_id, date)` | 名字在 T 时的归属 = 对 `date <= T` 的链接按 paper 求和取最大；一篇论文的别名 = 按 `paper_id` 查 | 新句子只追加 |
| 谱系边 | 事件 | `lineage_edge(child, parent, relation, kind self\|third, asserted_by, date, sentence_id, quote, valid_from)`，其中 `valid_from = max(date, child_v1, parent_v1)`；索引 `(child, valid_from)`、`(parent, valid_from)`；n 元 combines 存 `lineage_hyper(sentence_id, child, parents_json, valid_from)` | `WHERE valid_from <= T` | 追加 |
| 接受情况（画像） | 日累计 | `reception_daily(object, date, function, role, facet, n_statements, n_sentences, n_new_citing)`，主键 `(object, date, function, role, facet)` | 份额 = `SUM(...) WHERE date <= T`；描述样本从 `statements(about, date)` 现取（已有索引） | 追加或累加 |
| 类别 | 映射 + 日累计 | `category_canon(phrase_norm PK, canon_id, decided_by, llm_call_sha)`；`category_daily(object, canon_id, date, n)` | 按 T 求和 | 新短语先映射到已有规范类别（kNN 加 LLM），无法映射的另建 |
| 族级事实 | 成员事件 + 状态事件 | `fact_cluster(cluster_id PK, scope, facet, canonical_text, conditions)`；`fact_member(cluster_id, statement_id, speaker, about, date, stance)`；`fact_status_event(cluster_id, date, status, n_independent, n_members)` | 成员 = `fact_member WHERE date <= T`；状态 = `date <= T` 的最后一个事件。consensus/established/contested 都单调，扫一遍日期即可得到精确的时间线 | 新表述先在 ANN 里找最近的簇代表，交 LLM 裁定，归入已有簇或新建；只为受影响的簇从最早受影响日期起重算状态事件 |
| 作者独立性 | 维度表 | `author_link(paper_id, author_key)`（来自 registry 或 OpenAlex） | 现算（作者重叠的连通分量） | 追加 |
| 认识变迁 | 事件 | `shift_event(object, type, date, stats_json, evidence_ids, decided_by, llm_call_sha)`，索引 `(object, date)`；每个事件只依赖日期 ≤ `date` 的证据（时间窗定义） | `WHERE date <= T` | 对象有新证据时，只重算日期 ≥ 新证据最早日期的窗口 |
| 比较图 | 事件 | `comparison_edge(citing, compared, date, function, outcome, sentence_id)` | `WHERE date <= T`，并要求被比较对象在 T 时可见 | 追加 |
| 共现边 | 事件 | `cocite(a, b, citing, date)`，`a < b`；由一条 SQL 自连接离线生成 | 族快照构建时读取 | 追加 |
| 族 | as_of 网格快照 | `family_snapshot(snapshot_date, family_id, stable_id, member, support)`，索引 `(snapshot_date, member)`；`family_name(snapshot_date, family_id, name, definition, llm_call_sha)` | 取 `snapshot_date <= T` 中最新的一份；再在线并入 `(snapshot_date, T]` 区间内新出现的、与族内成员有强边的对象（一次查询）；输出注明 `snapshot_as_of`。这样做不会泄漏未来信息，最多滞后一个网格周期 | 网格：最近 3 年按月，更早按季，另外加上各基准的截止日（如 CS2 的 2025-04-30），总计约 80 份快照；新边只让受影响的连通分量变脏，只重算日期 ≥ 最早新边的快照；`stable_id` 按与上一快照成员的最大重叠来延续 |

**在线算的部分**：检索与召回（index）、工具结果的排序和截断、族快照之后的增量并入、作者独立性计数。这些都不调 LLM，按索引查询，每次调用的目标是 < 200 ms。

**双时间（bitemporal）**：补进来一篇旧论文时，会追加日期在过去的事件，过去某个 T 的答案因此改变。语义上这是对的（"现在所知的、关于 T 时的情况"），但会让早先的运行无法复现。所以每行带 `build_id`，运行 manifest 记录构建 digest；需要复现旧运行时，加一个 `ingested_build <= B` 的过滤条件，或者保留那次构建的库快照。

**存储与规模**：
- 千万级表述约 10 GB，SQLite 单机可以承受，前提是只有一个写进程、阶段加锁、批量插入、所有查询都有索引。
- 向量不能整块载入内存：`index/build.py:119-126` 现在是整表读入，1,000 万 × 4,096 维 × 4 B ≈ 164 GB。需要一个支持日期过滤的 ANN 索引（hnswlib 或 faiss，两者都还没安装，属于新依赖，需确认）。也可以先用 MRL 把向量截到 1,024 维，再配合分片。

## 6. 规模实测与外推

合成库：1 万篇、174,913 条表述（其中他述 139,913 条）、100,000 个句子、233,369 条引用行。`cites(sentence_id)` 已补索引，否则 families 测不完。

| 读路径 | 17.5 万条表述时 | 千万级线性外推 | 说明 |
|---|---|---|---|
| `AsOf.statements(kind="other")` | 0.99 s，峰值 282 MiB（约 2 KB/行） | 约 57 s，约 20 GB | 全表读入并做 JSON 解码 |
| `Identity(view)` | 1.35 s | 约 77 s | 每次工具调用都重新构造 |
| `lineage.edges` | 2.74 s | 约 157 s | 全量扫描 |
| `tools.paper_profile` 等价操作 | 7.01 s | 约 400 s | Identity 一次，edges 两次 |
| `profile` / `shifts.events`（hub，1,468 条） | 0.02 s | 与该对象的被引量成正比 | 走 `(about, date)` 索引，不是瓶颈 |
| `families`（535 个对象，20,982 个句子） | 1.68 s（有索引） | 与范围内被引量成正比，`support` 部分是平方级 | 真实库没有索引时每次查询 0.15 s，合计约 52 分钟 |
| `facts`（200 个成员，limitation） | 0.39 s | 平方级 | 贪心聚类 |
| 抽取准备（真实库，cs.LG 2018 年起） | 38,365 个被引对象，301,917 对 | — | 每对一次无索引查询（`extract/build.py:139-141`），在任何 LLM 调用之前就要约 12.6 h |
| 他述遍 LLM（同上范围） | 12,580 次调用 × 2,896 completion token | — | 48 路、720 tok/s 时约 14 h；build 默认 4 路时约 87 h |

## 7. 之前被默认可用、实际不满足要求的点

1. **"重建能清除过期"**：各阶段都按已完成集合跳过。代码改动后执行 rebuild，重算 0 条，manifest 却被盖成新鲜（探针 2）。真实库的 citations 现在就是过期状态，`build all` 会把它重新盖成新鲜。
2. **"上游一变，下游自动过期"**：只有上游的参数或代码变才会传播。数据变了而参数没变时，digest 不变（探针 4）。
3. **"阶段代码哈希覆盖了阶段代码"**：漏了 `llm/client.py`、`llm/jsonparse.py`、`compile/skeleton/proposes.py`、`llm/embedding.py`、`sources/refgraph.py`、`compile/skeleton/bib.py`（探针 1）。
4. **"改参数续跑"**：manifest 记的是新参数，表里留着旧结果（探针 3）。
5. **失败条目**：LLM 全部失败时仍然标 done，阶段判新鲜（探针 5）。INTEGRATED §3 只提到了 done_self，done_pairs 也有同样问题。
6. **manifest 与并发**：写入不是原子的（读失败 318/816）；没有阶段锁，两进程并跑会产生重复数据（探针 6、7）。
7. **"每个工具返回的 id 在 as_of 时可见"**：测试之所以通过，是因为合成数据里没有反例。真实库 6.6% 的施引论文含有 v1 之后发表的参考文献，正文版本和日期对不上；`baselines_for`、`find_evidence`、`paper_card` 的引用计数，以及 stub 都会泄漏。
8. **"并发 48"**：只有 `answer` 生效；`build` 实际按 4 路跑。
9. **"墙挡住一切挂死"**：只对调用方成立。服务端仍在生成，车道已释放，重试还会叠加。embedding 的读取路径没有墙，仍可能挂死。
10. **"温度 0 可复现"**：实测同一提示两次输出不同。
11. **"有调用台账"**：默认不写盘，缺失败原因和 finish_reason，内存列表无限增长。
12. **arm purity 与视觉调用**：只存在于旧的 kb_infra，新客户端没有；相应的测试也测的是旧代码。
13. **Paratera 调用**：没有退避、没有限流、不处理 429。
14. **JSON 解析**：会静默返回错误的对象（`[1]`）。
15. **identity 的 GENERIC 过滤**：在归一化之后匹配，复数和类别名都漏过；18.3% 的命名论文存在重名。
16. **独立性**：只看姓氏，不满足传递性，缺作者时按独立计。
17. **shifts 阈值**：零变化数据上误报 8–76%。
18. **label propagation**：在有社区结构的图上塌缩成 1 族；结果取决于 id 的字典序。
19. **`cites(sentence_id)` 没有索引**：真实库每次 0.15 s；一次 field_map 约 52 分钟；抽取准备约 12.6 h。
20. **向量检索**：`Index._dense` 整表载入内存，千万级不可行。
21. **两套配置系统取值矛盾**（96/48/4，900/240），而 build 命令两套都不读。
22. **表征测试依赖 kb_compiler**（`coarse.py:21`），旧包一删就挂；干净 clone 下 `test_config.py` 失败。
23. **`coarse.py` 的 CLI**：一运行就 NameError。
24. **`compile/skeleton/resolve.py`**：文档串写了 OpenAlex/Crossref 两个阶段，代码里没有实现。
25. **cutoff 与 as_of 两套时间语义**：前者"严格早于某月"，后者"含当天、按天"。日期只到年或月的论文（DOI 优先身份以后会大量出现）目前没有可见性规则。

## 8. 建议的实施顺序

1. **阶段契约**（§1c，M）：闭包哈希、digest 含输入指纹、done 绑定 digest、失败不标 done、原子 manifest、阶段锁。先做这一项，否则后面的每次重建都不可信。
2. **统一 LLM 客户端与配置**（§2、§1b，L+M）：build 读配置、车道生效、台账开启、响应缓存、salvage 迁入。完成后才能删 kb_infra、kb_compiler 和 `_see_llm`。
3. **as_of 修复**（§1d asof，M）：引用句日期按文本版本记、被引对象可见性过滤、`cites(sentence_id)` 索引。补上对应的泄漏测试。
4. **物化**（§5，L）：先做事件表（mention_link、lineage_edge、reception_daily、comparison_edge、cocite），再做族快照，最后做事实和变迁。
5. **LLM 化**（§4，每项 M–L）：先在 200 条样本上做双模型核对试点，用实测比例替换本文的成本假设，再全量运行。
6. 需要用户确认的设计变更：shifts 从二分切分改为时间窗（行为定义改变）；新增依赖 leidenalg、hnswlib/faiss；`compile/state/field_state.py` 降为评测对照组。
