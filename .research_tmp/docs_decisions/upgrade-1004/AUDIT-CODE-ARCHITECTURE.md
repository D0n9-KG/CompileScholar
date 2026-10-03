# 代码架构审计（AUDIT-CODE-ARCHITECTURE，10-04）

范围：整个仓库（src/、tests/、build.py、conf/、.research_tmp/experiments/benchmarks/、.research_tmp/review_1002/）。
目标：判断离"能随论文发布的研究代码库"还差什么，给出目标架构和不碰在跑任务的迁移顺序。
本档只读审计，除本文件外没有改动任何文件，没有发起 LLM / 外部 API 调用，没有碰进程。

## 0. 结论先行

1. 生产答题代码不在包里。三个跑批入口（cs2/run_vnext.py、cs2/run_vnext_dsb.py、scholarqa_multi/run_vnext_multi.py）
   都靠 sys.path.insert 导入 .research_tmp/experiments/benchmarks/_shared/tools/answer_pipeline.py；
   answer_pipeline 再用 "../../../../../src" 把 src 塞进 sys.path。pyproject 只打包 kb_compiler、kb_infra，
   src/retrieval 和 sci_evo_extract 垫片没打包，pip install 后答题路径直接 ImportError。
2. 真正在干活的核心很小（answer_pipeline 584 + refgraph 439 + cutoff 72 + hybrid 132 + llm/embedding/sources 中实际用到的
   函数，约 1.7k 行），但一次导入会拉进约 20 个仓库内文件 / 9.9k 行（registry.py 3,806 行只为一个 normalize_doi，
   8 个 _see_* 垫片模块顺带拉进 openai、dotenv）。
3. src/ 80 个文件 27.9k 行，30 个文件约 7.0k 行从任何现有入口都不可达；另有 8 个文件 2.9k 行只服务 Multi-108 旧建库链。
4. 硬编码遍地：sys.path 注入 742 处 / 558 个文件，"C:/Users/D0n9" 绝对路径 254 处 / 138 个文件，局域网 IP 32 处 / 26 个文件。
   活路径上的 direct_judge.py、retrieval_mcp.py、mcp_config_open.json、judge_nuggets.py、run_dsb_judges.sh 都写死绝对路径，换机即坏。
5. 测试 146 个全过（2.8 s），但一个都没碰活路径：answer_pipeline / refgraph / cutoff / hybrid / field_state / cs2_scoring /
   direct_judge 零覆盖；被测的是旧建库栈和已被取代的 2,784 行 ReAct harness。
6. 冻结清单 FREEZE_CS2_TEST_1003_v9b.json 的 14 个哈希与当前工作区一致（在跑的 test 用的确实是冻结代码），
   但哈希算的是 Windows CRLF 工作区字节，不是 git blob；且没覆盖 llm.py / embedding.py / retrieval 链 / astabench 判分源码 /
   rubric / refgraph 缓存。换一台 Linux 机器无法核验。
7. KB v2 不能从 git 复现：build_report 记录的输入哈希只匹配工作区的 base_kb，与 HEAD 已提交版本内容不同；
   records_merged.json 被至少 5 个脚本就地改写，没有记录顺序。
8. 发布阻断：本地 main 领先 origin 215 个提交，其中 17 个 blob 超过 100 MB（最大 2.44 GB），GitHub 会拒推；
   .research_tmp 下跟踪了 6.1 GB；promote_to_deep.py 里有明文 MINERU_TOKEN（仅本地未推送）。
9. 两个官方判分器都在未跟踪目录里且带本地补丁：astabench 是指向 .research_tmp/scratch 的 editable 安装，
   direct_judge 在运行时打了 4 处 monkeypatch；DSB 的 nuggetizer/llm.py 被改过、外层仓库完全不跟踪。
10. 对照臂公平性有架构级风险（未证实）：GPTR 臂经 ExternalTools→SearchService（15 s 档期 + 600 s 熔断）访问 Sciverse，
    answer_pipeline 自己的注释写明这条路在并发下 45 次查询 15 次为空；我们的臂走直连 + 令牌桶排队。
11. 潜伏崩溃：run_vnext_multi.MultiKB 不调父类 __init__，而 answer() 在 Multi 跑完之后加了默认 use_state=True，
    现在重跑 Multi 每题都会 AttributeError（已用不联网的方式实测复现）。

建议：收成单一包 src/compilescholar/{core,llm,sources,kb,state,retrieve,answer,eval,baselines}，实验只留 YAML + 薄 runner，
运行产物进 runs/<run_id>/（带自动生成的 manifest），缓存与数据出 git、按哈希清单发布。重构在 sparse worktree 里做，
CS2 test 跑完、打 tag 之后再合回 main；对外发布用新的干净仓库，不推现有历史。

## 1. 方法与范围

做了什么：
- 静态导入闭包：用 ast 解析 import，按各脚本实际注入的 sys.path 目录解析，处理 sci_evo_extract 垫片映射和 build.py 里
  "python -m kb_compiler.records.X" 字符串。注意：静态闭包对 "from . import X" 有漏判（table_channel 就是这样被 table_extract 引用的）。
- 运行时导入追踪：只 import answer_pipeline、SciverseClient、refgraph（不构造 KB、不发请求），列 sys.modules 里的仓库文件。
- 重算冻结清单与 build_report 的哈希；比对 git blob、工作区原始字节、LF 归一化字节三者。
- pytest：先 --co 收集，再 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 LLM_PROVIDER_ALLOWLIST=none 全量跑（5 分钟上限，实际 2.8 s）。
- 读 arm_vnext/answers_test100_r1.json 的 trace 统计静默降级。

没做什么（以下结论未核实）：
- 没跑任何 runner / judge，没有逐题统计 GPTR 臂的空检索率，没有核对 SearchService 熔断在 GPTR 跑批中是否真的触发过。
- 没有逐行比对 scratch 里的 asta-bench 与上游；只确认它是独立 git 克隆（9bf087a）且无已跟踪文件改动。
- "死代码"判定基于当前入口的可达性；手工跑过、产出过现有制品的生产者脚本（如 survey_extract）单独标为"溯源活"。

## 2. 活代码地图

### 2.1 入口与调用链（当前真正在用的）

答题：
- cs2/run_vnext.py:19-22、cs2/run_vnext_dsb.py:18-22、scholarqa_multi/run_vnext_multi.py:16-20
  → sys.path.insert(_shared/tools) → import answer_pipeline as AP
- answer_pipeline.py:27-31 → sys.path.insert("../../../../../src", _shared/tools)
  → kb_infra.llm.call_local / parse_json_response、kb_compiler.retrieve.hybrid.HybridIndex、cutoff（顶层模块名）
  → 惰性：kb_infra.embedding.embed_local（KB 有向量时）、sci_evo_extract.library.sources.SciverseClient（:154）、refgraph（:262/:377）
- run_vnext_multi.MultiKB(AP.KB)：自己复制了一份加载逻辑，不调 super().__init__（见 §8 H9）。

评测：
- CS2：cs2/direct_judge.py（官方 scorer + 补丁）→ cs2/cs2_scoring.py（四 facet 平均，缺答记 0）
- DSB：review_1002/p6/judge_nuggets.py（官方 Nuggetizer）→ review_1002/p6/analyze.py；另有 cs2/run_dsb_judges.sh 走 dsb/eval.main
- 对照臂：cs2/harness_arm_run.py + harness_to_judge.py（Claude Code + _shared/mcp/retrieval_mcp.py + cc_compat_proxy.py）、
  cs2/dsb_harness.py、cs2/cs2_gptr.py（检索走 _shared/tools/external_tools.py）、cs2/memorized_adapter.py

建库（CS2 KB v2，论文主实验用的 KB）：
- cs2/base_kb_build/build_kb_v2.py → embed_kb_v2.py → build_state_v2.py → merge_state_v2.py，产物 cs2/base_kb_v2/
- 上游 base_kb/ 由 compile_base_kb.py（硬编码 C:\Users 路径，subprocess 调 kb_compiler.records.registry_growth / views.compiler）、
  手工跑的 survey_extract / coarse_extract / deep_extract，以及 cs2/ 根目录下的就地改写脚本（normalize_entities、resolve_citations、
  translate_absences_v2、classify_paraphrases、recover_deep_records、backfill_*）共同产出，没有驱动脚本。
- build.py 的 STAGES 和 conf/base.yaml 指向的是 Multi-108 的 KB（scholarqa_multi/kb），不是 CS2 KB v2。

领域层检验：src/kb_compiler/views/field_state.py ← cs2/base_kb_build/heldout_field_eval.py（另有 fetch_heldout_refs、
build_survey_gold、clean_survey_gold、summarize_field_eval）。答题端消费的状态层是旧的 state_merged.json，不是 field_state。

### 2.2 答题路径运行时实际加载的仓库文件

| 文件 | 行数 | 实际用到的部分 |
|---|---|---|
| _shared/tools/answer_pipeline.py | 584 | 除 _ext / _s2* / array 外全部 |
| _shared/tools/refgraph.py | 439 | 全部 |
| _shared/tools/cutoff.py | 72 | 全部 |
| src/kb_compiler/retrieve/hybrid.py | 132 | 全部 |
| src/kb_infra/llm.py | 993 | load_env、call_local、parse_json_response、_walled_*、_log_call |
| src/kb_infra/embedding.py | 195 | embed_local |
| src/retrieval/sources.py | 1,646 | SciverseClient.semantic_search、令牌桶、_knowledge_cutoff，约 300 行 |
| src/retrieval/registry.py | 3,806 | 只有 normalize_doi（retrieval/__init__.py 与 sources.py:18 都导入它） |
| src/retrieval/_see_*.py（8 个） | 1,971 | 无；由 src/sci_evo_extract.py 无条件导入，带进 openai、dotenv |
| src/sci_evo_extract.py + 若干 __init__ | 约 50 | 模块名重映射 |

运行时实测：import 这三样后 sys.modules 里有 29 条仓库模块（_see_* 因垫片别名重复计数），第三方顺带加载 dotenv / openai / requests / yaml。
静态闭包更大（45 个文件），因为死函数 _ext()（answer_pipeline.py:136-141）静态引用 external_tools → evidence_gate2r_harness →
kb_compiler.views.tools / records.deep_extract 等；运行时从不执行。

### 2.3 src/ 可达性

| 类别 | 文件 | 行数 | 说明 |
|---|---|---|---|
| 答题 + 评测 + 建库都可达（静态） | 37 | 16,248 | 大半只经死路径 _ext / 基线臂可达 |
| 只被建库可达 | 8 | 2,864 | config、embed_block、notation_harvest、registry*、table_extract、runlog（Multi 链） |
| 其他组合 | 3 | 410 | |
| 无入口（不含 table_channel/table_semantic，二者经 table_extract 的 "from . import" 可达） | 30 | 约 6,990 | 见下 |

无入口清单：kb_compiler 的 cli、maintenance、arbitration_list、canary、canary_granular、chunk_retry、figure_channel、manifest、
manifest_paperscope、registry_merge、resolve_refs、run_granular_pilot、run_smoke、survey_extract、usage_audit、verification/*、
views/render、views/warmup_emb、records/test_*（2 个错放的测试）；retrieval 的 acquisition_chain、evidence_units、export、
extraction_runs、pdf_resolver、process_workflow、processing、sciverse_ingest、search_resolve。
其中 cli.py 是 pyproject [project.scripts] 声明的 kb 命令（无 import 方但非死）；survey_extract 是 CS2 综述层的手工生产者
（溯源活，需保留可运行）；acquisition_chain 被 scholarqa_multi 的取 PDF 脚本用。其余可归档。

.research_tmp 侧：benchmarks 下（不含第三方 dsb）129 个 .py / 16.7k 行。_shared/tools 30 个文件 11.6k 行中活的只有
answer_pipeline、refgraph、cutoff（1,095 行），external_tools（1,483）仅作 GPTR/STORM/PaperQA 臂的检索适配器，
multi_* 系列支撑已发表的 Multi-108 数字（溯源活），其余（evidence_gate2r_harness 2,784、report_adapter 736、report、
judge_answer、judge_kb、ps53r_run、behavior_probe 等）是 PaperScope / ReAct 时代遗留。cs2/ 根目录 55 个 .py 中当前在用
约 10 个，其余是 probe / audit / patch / growth / 旧 judge 脚本。

### 2.4 导入方式与硬编码（计数口径：.py/.sh，排除 venv、site-packages、第三方 dsb）

| 范围 | sys.path 注入 | C:/Users 绝对路径 | 局域网 IP |
|---|---|---|---|
| src/ | 18 处 / 15 文件 | 2 / 1（registry_growth.py:99,268 fail_dump） | 0 |
| tests/ | 25 / 21 | 8 / 7 | 0 |
| _shared/ | 44 / 23 | 21 / 11 | 4 |
| cs2/ | 59 / 41 | 59 / 42 | 11 |
| scholarqa_multi/ | 22 / 19 | 41 / 25 | 2 |
| review_1002/ | 35 / 31 | 75 / 50 | 7 |
| 整个 .research_tmp + src + tests | 742 / 558 | 254 / 138 | 32 / 26 |

局域网 IP 只有两个：192.168.199.73（GPUStack，含 :9600/:9601）与 192.168.199.138（NAS 共享 \\...\Share400T，arXiv 快照）。

活路径上的硬编码位置：
- answer_pipeline.py:28（"..×5/src"）、:32（NO_PROXY 追加 .73）、:36（~/.sciverse_bucket.json）、:156（"src/../.env"）、:199-200（import 时 makedirs）
- refgraph.py:33-34（import 时 makedirs）、:58-59（~/.pace_*.json）、:93（"..×5/.env"）
- direct_judge.py:28（scratch/asta-bench 绝对路径）、:34（.env 绝对路径）
- _shared/mcp/retrieval_mcp.py:37、mcp_config_open.json（retrieval_mcp.py 绝对路径 + ~/.sciverse_bucket.json）、cc_compat_proxy.py:26（UP=.73）
- harness_arm_run.py:22-24（%APPDATA%\npm\...\claude.exe）、:76（NO_PROXY）、:115（CS2.parents[3]/.env）
- run_vnext_dsb.py:24（产物写进 review_1002/p6）、run_vnext.py:46 与 cs2_scoring.py:42（CS2 题目从 scholarqa_multi/ 读）
- build_kb_v2.py:30（KB 输入依赖 review_1002/main/hub_oai_meta.json，缺失时静默当 {}）
- embed_kb_v2.py:15、merge_state_v2.py:24、heldout_field_eval.py:28（NO_PROXY）；survey_arxiv.py:28、snapshot_fill.py:36（NAS UNC 路径）
- compile_base_kb.py:21-24、patch_cards_view.py:23（C:\Users 绝对路径）
- review_1002/p6/judge_nuggets.py:13-15、cs2/run_dsb_judges.sh（DSB 路径、venv311 python、.env 全部绝对路径）

## 3. 分层问题

### 3.1 生产逻辑放在实验目录

- 答题管线、引文图、知识截止三件生产核心都在 .research_tmp/experiments/benchmarks/_shared/tools/；被 3 个 runner、
  FREEZE 清单、论文结果直接依赖。.gitignore:102-105 用 "!" 规则把 benchmarks 重新纳入跟踪，所以它们在 git 里，
  但不在包里、不可安装、没有测试。
- src/retrieval/sources.py:34-43 的 _knowledge_cutoff() 用 sys.modules.get("cutoff") 读实验目录模块的线程局部值：
  src 反向依赖实验目录，且依赖"调用方先以顶层名 cutoff 导入过"这一隐式约定。DSB 每题截止的正确性就靠它。
- CS2 KB v2 的建库链 15+ 个脚本散在 cs2/base_kb_build/ 和 cs2/ 根目录；build_kb_v2 的输入之一在 review_1002/main/，
  DSB 的输入（oracle_inputs.json）和输出（gen/<sys>/）在 review_1002/p6/。审查草稿目录事实上是生产数据目录。
- 输出与代码混放：cs2/ 根目录 55 个 .py、185 个 .json、144 个 .log、25 个子目录；run_vnext 把 judge_input_*.json 写在代码目录。

### 3.2 重复实现

| 功能 | 实现位置 | 现状 |
|---|---|---|
| LLM 调用 | kb_infra/llm.py（call_llm/paratera/cst/intern/local，各自一套重试）；retrieval/_see_llm.py（OpenAI SDK，环境变量拼成 PERATERA_*）；retrieval/query_understanding.py:77 与 rerank.py:61（裸 urllib）；cs2_gptr、cs2_storm、cc_compat_proxy、review_1002/p1,p3,p4,p5,p6 各一份 | 活：call_local、call_paratera |
| 嵌入 | embedding.embed_local / embed_cst、llm.embed_batch、rerank._default_embed、merge_state_v2.emb | 活：embed_local |
| Sciverse 访问 | ① answer_pipeline._sciverse 直连 + 令牌桶等 600 s + 3 次重试；② retrieval_mcp._sciverse 直连（harness 臂）；③ external_tools→SearchService 分层扇出 + 熔断（GPTR/STORM/PaperQA 臂） | 三条都活，见 §8 H8 |
| S2/OpenAlex/Crossref/arXiv | sources.py 的四个 Client；refgraph.py 再实现一遍（自带节流、缓存）；answer_pipeline._s2（死）；retrieval/citation_graph.py（同目的，未用）；base_kb_build 的 fetch_heldout_refs、survey_pool、arxiv_fill、backbone_pool、hub_consensus、fetch_hub_texts 各自直连 | 活：refgraph |
| 跨进程节流 | sources._FileTokenBucket（~/.sciverse_bucket.json）、refgraph._pace（~/.pace_<src>.json + ~/.pace_s2_cooldown.json）、answer_pipeline._s2_pace（~/.s2_pace.json，死） | 状态都在 $HOME，仓库外不可见 |
| 知识截止 | cutoff.py；sources._knowledge_cutoff；Sciverse 服务端 year ≤ 截止年−1（sources.py:853-858）；build_kb_v2.CUTOFF_YEAR=2025；retrieval_mcp before_year；external_tools filter_rows | 语义大体一致，但 5 处各写各的 |
| KB 加载 | answer_pipeline.KB；run_vnext_multi.MultiKB（复制而非继承初始化）；evidence_gate2r_harness.TextKB；views/tools.KBTools；probe_l1*._KB | 活：KB、MultiKB |
| .env 读取 | llm.load_env（:36-58，含一段重复的空循环 :44-49）；answer_pipeline:156；refgraph:89-102；direct_judge:34-38（全部灌进 os.environ）；harness_arm_run:115；retrieval_mcp:51-60；judge_nuggets:18-23；run_dsb_judges.sh；_see_llm / _see_upstream 用 python-dotenv | 活路径 7 种实现；29 个文件碰 .env |
| 判分 / 汇总 | CS2：direct_judge + cs2_scoring（活）、judge_run、judge_dev20、judge_batch_task（两份）、gen_judge_task、judge_smoke_task、judge_runs_ours{,2,3,4}/task_sqa_*、score_compare（仍用旧口径 (IR+AP+F1)/3，score_compare.py:4）；DSB：judge_nuggets + analyze（活）与 run_dsb_judges.sh→dsb/eval.main 两条；Multi：multi_judge_incremental、multi_rubric_judge、multi_judge_freeze、score_multi_f1、score_multi_strict；领域层：heldout_field_eval.judge_match | 活的只有少数几支 |

### 3.3 配置分散

两套配置并存：conf/base.yaml + kb_compiler/config.py（分层设计不错，但只被 build.py 消费，路径指向 Multi 语料）；
答题 / 评测 / CS2 建库全部走 环境变量 + argparse + 脚本内常量。answer_pipeline 一行都不读 conf/。

活路径读取的环境变量（约 45 个）：
- 答题：ANSWER_MODEL、KNOWLEDGE_CUTOFF、LOCAL_MAX_CONCURRENT、LOCAL_LARGE_MAX_CONCURRENT、LOCAL_BASE_URL、LOCAL_API_KEY、
  LOCAL_MAX_ATTEMPTS、LOCAL_SOCK_TIMEOUT、LLM_WALL_TIMEOUT、LLM_SEED、LLM_RUN_ID、LLM_CALL_LOG、LLM_CALLER、
  LLM_PROVIDER_ALLOWLIST、EMBEDDING_MODEL、SCIVERSE_API_TOKEN、SCIVERSE_MAX_WAIT_S、SCIVERSE_RATE_PER_MIN、
  SCIVERSE_SHARED_BUCKET、OPENALEX_API_KEY、NO_PROXY（被改写）、S2_SHARED_PACE（死）
- 判分：JUDGE_MODEL、CS2_SPLIT、PARATERA_API_KEY/BASE_URL → 重映射成 GLM_* / DEEPSEEK_*、OPENAI_API_KEY="sk-placeholder"、FIELD_JUDGE
- 对照臂：CS2_OFFSET、CS2_LIMIT、HARNESS_FANOUT、HARNESS_TIMEOUT_S、HARNESS_OUT、HARNESS_CWD、HARNESS_ANTHROPIC_BASE_URL、APPDATA、
  GPTR_OUT、GPTR_FANOUT、GPTR_LIMIT、LLM_KWARGS、OPENAI_BASE_URL、RETRIEVAL_MCP_MODE/MANIFEST/TEXTS
- 建库：T_MERGE、T_ATTACH（merge_state_v2 会把值写进输出 params，这点是对的）

具体问题：
- LOCAL_MAX_CONCURRENT 有四个默认值：conf 96、run_vnext 48、cs2_gptr 16、llm.py 4。llm.py:648 在 import 时就建信号量，
  所以 runner 必须在 import answer_pipeline 之前 setdefault；run_vnext_multi 没设，实际只有 4 路。
- llm.py:764-766 是 ENV.get(...) or os.environ.get(...)：.env 覆盖进程环境，与通常约定相反，命令行无法临时覆盖。
- run_vnext 的 config_<tag>.json 只记 12 个开关，不记 git sha、代码哈希、KB 哈希、并发、seed、端点；同 tag 重跑会覆盖它，
  而断点续跑按 qid 合并，换了配置也会把新旧答案混在一个文件里。
- .env.example 过时：列的是 NEO4J / OPENROUTER / SILICONFLOW，活路径要的 LOCAL_BASE_URL、PARATERA_*、SCIVERSE_API_TOKEN、
  OPENALEX_API_KEY 一个没有。
- 模型名写死：answer_pipeline 用 ANSWER_MODEL，field_state.py:30 写死 MODEL="Qwen3.8-27B"，direct_judge 默认判分模型写在函数签名里。

### 3.4 Prompt 内联

answer_pipeline 内联 PLAN / PROBE_BLOCK / WRITE / SCREEN 4 个，field_state 4 个，heldout_field_eval 3 个，harness_arm_run 抄官方
PROMPT_TMPL。词数上限靠 answer_pipeline.py:508-509 对 WRITE 里的一句英文做 str.replace：哪天改了那句措辞，长度约束会静默失效。
trace 不记 prompt 版本或哈希（只能间接靠 FREEZE 里整文件哈希）。

## 4. 质量

### 4.1 测试

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 LLM_PROVIDER_ALLOWLIST=none python -m pytest tests -q`：收集 146，通过 146，用时 2.82 s，1 个 pydantic 插件警告。
全部 mock LLM，不联网。覆盖面：

| 活模块 | 有测试吗 |
|---|---|
| answer_pipeline（plan/gather/screen/write/assemble/_interleave） | 无 |
| refgraph（路由、缓存策略、_bib_title 解析） | 无 |
| cutoff.allowed（截止判定，泄漏防线） | 无 |
| retrieve/hybrid（BM25 + RRF + 每篇上限） | 无 |
| views/field_state | 无 |
| cs2_scoring / direct_judge / judge_nuggets / analyze | 无 |
| run_vnext*（断点续跑、产物格式） | 无 |
| records 层（coarse/deep extract、postcheck、table、registry、embed_block） | 有，大部分测试在这里 |
| evidence_gate2r_harness（已被取代） | 有（test_backref_and_cite_repair.py） |

其他：7 个测试文件写死 sys.path.insert(r"C:\Users\D0n9\Desktop\CompileScholar\src")；test_external_tools_backflow.py:11-15
还插入兄弟仓库 C:\Users\D0n9\Desktop\sci-evo-extract\src；src/kb_compiler/records/test_*.py 两个文件不在 tests/ 下、不被收集；
pyproject 没有 [tool.pytest] testpaths；没有 CI；import-gate 测试只管 kb_compiler，不管答题路径。

### 4.2 错误处理：静默降级链

- llm.call_local 重试耗尽返回 None（:820-835）→ answer_pipeline.chat 变 ""（:45-48）→
  plan 两次解析失败退化成单节 "Overview"（:355，trace 无标记）；screen 解析失败不剔除（:466-470，合理但无计数）；
  write 返回 "" 时 assemble 直接跳过该节（:523-524，无计数）。死掉的 LLM 通道会产出"短一些但看起来正常"的答案。
- refgraph._http 吞掉所有异常、重试 3 次后返回 None（:144-146），调用方分不清"无参考文献"和"请求失败"；
  references() 在 refs 为空且 S2 不在冷却时照样写路由缓存（:421-422），注释说"所有来源正常应答时才写"，实际条件没检查这一点：
  超时 / OpenAlex 额度跳过 / Crossref 网络错 都会被永久缓存成空。实测 refgraph_cache 1,554 条路由缓存中 7 条 source=none 空结果。
- refgraph 统计 STATS 从不写进 trace；cite 通道失败在 trace 里不可见。
- run_vnext 每题异常记为错误行、判分记 0（与 cs2_scoring 缺答记 0 一致，这是对的）；但错误行被当作"已完成"，续跑不会重试。
- run_vnext_dsb.py:69-74 异常只 print "ERR"，不落盘；judge_nuggets.py:95-97 判分异常 continue；analyze.py:24 对各系统取交集：
  DSB 缺答被剔除而不是记 0，与 CS2 口径（E7 修过的幸存者偏差）相反。
- external_tools.py 39 处宽 except，其中 22 处紧跟 pass / 返回空（GPTR 臂检索走它）。

test100_r1 实测（100 题）：runner 错误 0，plan 回退 0，计划节数 = 输出节数，外检错误 0 / 3,550 次调用；
11 题 cite 通道证据为 0，原因从 trace 无法区分（无共引 / 超时 / 缓存的空结果）。也就是说这次没出事，但出事了也看不出来。
另：同一篇论文经 KB 通道（paper_key=kb:<pid>）和外部通道（ext:<md5(title)>）进来会拿到两个引用编号，r1 中 355 节里 11 节出现同题双编号。

### 4.3 决定性与缓存

- LLM：写作 temperature 0.2，plan 0.0→0.3 重试，screen 0.0；test 脚本没设 LLM_SEED，seed 不下发（llm._env_seed 的注释自己也说 provider 是否遵守 seed 未验证）。
  r1/r2 两次运行是唯一的方差控制，论文里需写明。
- Sciverse 结果完全不缓存（sources.py 中无 cache）：重跑检索会随服务端索引漂移，只有 trace.evidence 留存了当时的证据。
- refgraph_cache（3,728 个 HTTP 响应 + 1,554 条路由）与 s2_cache 被 gitignore、无 TTL、无版本，FREEZE 不记录其状态；
  run_test_final.sh 明说 r2 复用 r1 的 OpenAlex 缓存，所以 r2 在引文通道上不是独立样本。
- 引文扩展受墙钟预算控制（cite_expand budget_s=90、prefetch 120 s、Sciverse 等待 600 s），并行臂共享 $HOME 下的节流状态：
  一个臂慢了会让另一个臂的引文证据变少。结果依赖负载。
- 缓存写入非原子（refgraph :129、:374、:422 直接 open(...,"w")）；多线程同题并发时可能读到半截 JSON，路由缓存读取（:394）不带异常保护。
- KB 向量 record_vecs.f32（475 MB）被 gitignore；KB 只校验"条数相等"（answer_pipeline.py:66-67），不校验 meta 里已有的 records_sha16。
- import 时副作用：answer_pipeline 改 NO_PROXY、setdefault 两个 SCIVERSE 变量、建 s2_cache 目录；refgraph 建 refgraph_cache；
  llm.py 按环境变量建信号量（大车道 :802-810 惰性创建且不加锁）。导入顺序会改变并发行为。

### 4.4 可复现性

冻结清单 cs2/FREEZE_CS2_TEST_1003_v9b.json：
- 14/14 哈希与当前工作区一致；记录的 git_head=26fbd63d 与冻结提交 68f1a67e 上这些文件的 blob 相同。在跑的 test 用的是冻结代码。
- 哈希对象是工作区原始字节。core.autocrlf=true 且没有 .gitattributes，工作区是 CRLF、blob 是 LF：
  例如 run_vnext.py 工作区 99dc588b…（与清单一致），blob 与 LF 归一化都是 3233c6e4…。干净的 Linux 克隆无法核验这份清单。
- 漏项：kb_infra/llm.py（重试 / 超时 / 信号量）、kb_infra/embedding.py、retrieval/registry.py 与 _see_*、sci_evo_extract.py、
  cutoff 之外的截止实现、astabench 判分源码与 commit、rubric 文件（scholarqa_multi/sqa2_rubrics_v2_recomputed.json）、
  refgraph_cache 状态、pip 包版本、GPUStack 服务端模型版本、并发等环境变量。

KB v2 溯源：
- base_kb_v2/build_report.json 的 inputs_sha16 与当前工作区 base_kb/ 的 5 个输入一致；但 HEAD 里的 records_merged.json
  （81d03418… vs 0824d108…）和 deep_read_records.json（5deef333… vs 43ee6403…）内容不同（LF 归一化后仍不同，不是换行问题）。
  从 git 检出无法重建 KB v2。
- records_merged.json 被 normalize_entities.py:223-227、resolve_citations.py:335-336、translate_absences_v2.py:54/82-85、
  classify_paraphrases.py:151-155 就地改写（各自留 .bak_pre_* 备份），deep_read_records.json 被 recover_deep_records.py:94-95 改写；
  执行顺序没有记录在任何清单里。
- 产物 papers.json / records.json / state_merged.json 已提交且与清单一致（papers.json 只差 CRLF）；record_vecs.f32 不在 git。

第三方判分器：
- astabench 0.5.4 是 editable 安装，映射到 .research_tmp/scratch/ai2_baseline_2026-09-08/asta/asta-bench-main（gitignore、
  独立 git 克隆 9bf087a）；direct_judge.py:28 还用绝对路径把它插到 sys.path 最前。
- direct_judge 的补丁：httpx 同步 / 异步都强加 Connection: close（:17-26）；generate_with_retry max_retries 20→4、base_delay 1.0（:79-92）；
  criteria_idx 0 起改 1 起（:61-77）；判分模型 DeepSeek-V4.1-Flash（官方是 gemini）。这些都改变判分器行为，必须作为判分适配层披露。
- DSB：deepscholar/dsb 是嵌套 git 克隆（上游 c95413b），外层仓库完全不跟踪；其索引处于异常状态（414 个文件全部"已暂存删除"同时又"未跟踪"）；
  与上游 HEAD 比 2 个文件改过（eval/nuggetizer/src/nuggetizer/core/llm.py 加了 base_url 支持；results/…/aggregated_results.csv），缺 43 个文件。
  判分依赖的这个补丁不在任何版本控制里。venv311 也放在 benchmarks 目录里。

依赖与打包：pyproject.toml:30 只打包 src/kb_compiler、src/kb_infra；依赖只有 numpy、pyyaml。活路径另需 filelock、openai、python-dotenv
（经垫片导入）、inspect_ai、astabench、httpx、mcp、gpt_researcher；无 lock 文件；requires-python>=3.13 而 DSB 用 3.11 venv。
cli.py 用 parents[2]/build.py 定位仓库根，装成 wheel 后失效。

git 卫生：
- 跟踪 4,288 个文件，其中 4,161 个在 .research_tmp 下：3,123 个在 experiments/benchmarks（.gitignore:103-105 有意重新纳入），
  另 1,038 个命中忽略规则仍被跟踪（review_1002 553、runs/kernel_v2 392、docs_decisions、literature 等，是规则生效前加入或强制添加的）。
  后果之一：review_1002/p6 里新增的 DSB 判分脚本默认不会进 git。
- 工作区里被跟踪的 .research_tmp 数据合计 6.11 GB，12 个文件 >50 MB（最大 cs2/base_kb/embed_cache/embed_r2_props.json 1.75 GB，
  scholarqa_multi/baselines/ours/emb_cache_records.bin 854 MB）。
- 历史里最大 blob 2.44 GB（lightrag vdb_relationships.json，已取消跟踪但仍在历史中）；origin/main..main 范围内 17 个 blob >100 MB。
  main 领先 origin 215 个提交（origin 停在 09-20），按 GitHub 100 MB 单文件上限，现状 git push 必然被拒。
- .git 松散对象 10.19 GiB，另有 2.94 GiB 残留 tmp_pack 垃圾。
- 明文密钥：scholarqa_multi/promote_to_deep.py:47 MINERU_TOKEN 字面量（提交 9872253b，仅本地）。.env 本身未跟踪。
- 安全：kb_infra/llm.py:31-33 全局关闭 TLS 校验（check_hostname=False、CERT_NONE），call_paratera 等带 API key 的公网调用也走它。
  cc_compat_proxy 绑定 127.0.0.1（无鉴权但只本机，可接受）。

## 5. 目标架构

### 5.1 包与目录

```
pyproject.toml            # 单一包 compilescholar；extras: eval / baselines / dev；附 uv.lock（或 requirements.lock）
src/compilescholar/
  core/     config.py（分层 Settings）  paths.py（仓库根 / data / runs / cache，全可配）  secrets.py（唯一 .env 读取点）
            manifest.py（每次运行自动生成清单）  cutoff.py（显式 Cutoff 对象，取代线程局部 + sys.modules）
  llm/      client.py（provider 注册表：local / paratera / ...，统一重试、车道并发、调用账本、可选录制回放）
            embedding.py  jsonparse.py
  sources/  http.py（统一节流 + 内容寻址缓存 + 录制回放；状态在 cache_dir 而非 $HOME）
            sciverse.py  openalex.py  s2.py  crossref.py  arxiv.py  refgraph.py（多来源参考文献路由）
            search.py（唯一的检索门面，我们和所有对照臂共用）
  kb/       schema.py  store.py（KB 目录 + MANIFEST 校验哈希，取代 AP.KB / MultiKB）  index.py（原 hybrid.py）
  compile/  records/（coarse / deep / survey / table extract、postcheck）  registry/  views/
            state/（build_state、merge_state、field_state）  pipelines/（KB v2 与 Multi KB 两条 DAG，复用 build.py 的 Stage 设计）
  answer/   pipeline.py（answer()，只接收 AnswerConfig + LLM + Searcher + KB，无全局状态）
            stages/{probe,plan,gather,screen,write,assemble}.py  prompts/*.txt（带版本与哈希）  formats/{cs2,dsb,multi}.py
  eval/     cs2/（rubric 加载、judge 适配层：把 direct_judge 的补丁集中成一个带开关的文件、scoring）
            dsb/（nugget judge、汇总：缺答记 0）  multi108/  field/（heldout_field_eval）  stats.py（配对 bootstrap）
  baselines/ harness/（claude code runner、retrieval MCP server、compat proxy）  gptr.py  storm.py  paperqa.py  memorized.py
  cli.py    compilescholar {kb build|answer|judge|score|freeze|verify|field-eval}
configs/    base.yaml  local.example.yaml（端点、LAN、NAS 路径，本机副本 gitignore）
            kb/cs2_v2.yaml  kb/multi108.yaml  bench/cs2_test_v9b.yaml  bench/dsb.yaml  bench/multi108.yaml  ablations/*.yaml
experiments/<bench>/README.md + run.sh（一两行 CLI 调用，仅此而已）
third_party/  asta-bench@<commit>  deepscholar-bench@c95413b（git submodule）  patches/*.patch（nuggetizer base_url 等）
tests/      unit/  golden/（录制回放的端到端）  fixtures/（小 KB、录好的 HTTP / LLM 响应）
docs/       ARCHITECTURE.md  REPRODUCE.md  RESULTS.md  DECISIONS.md
legacy/     归档代码 + INDEX.md（不打包、不进 CI）
data/       （gitignore）kb/cs2_v2/、kb/multi108/、bench/{cs2,dsb,multi108}/、MANIFEST.sha256，外部发布
runs/       （gitignore）<bench>/<run_id>/{config.resolved.yaml, manifest.json, answers.jsonl, trace/, judge/, scores.json, logs/}
cache/      （gitignore）http/<source>/…、llm/…；可打成快照随运行发布
```

规模控制：单人研究项目，不引入 Hydra / DVC / 插件框架。配置用 dataclass + pyyaml（沿用 kb_compiler/config.py 的分层合并规则），
数据发布用"目录 + sha256 清单 + 下载脚本"。

### 5.2 几条设计约束

1. 检索门面唯一：sources/search.py 一个 Searcher 接口，我们的管线、harness MCP、GPTR / STORM / PaperQA 臂全部经它访问 Sciverse，
   超时、重试、截止、熔断策略相同，并逐次记录"空 / 错 / 熔断"。
2. 依赖注入：answer() 接收 llm、searcher、kb、cfg 参数，不在模块里建全局客户端；测试用假实现。
3. 录制回放：所有 LLM 与 HTTP 调用经同一网关，可选按 (端点, 请求哈希) 落盘；--replay 时只读缓存，离线复现一次运行。
4. 失败可见：每题 trace 带 degradation 计数（plan_fallback、screen_parse_fail、write_empty、cite_timeout、cite_http_fail、ext_errors），
   运行级 health.json 汇总，直接对应 UPGRADE-PLAN §3 第 5 条的作废规则。
5. 缓存三态：正结果、确定的负结果（404 + 原因 + 时间戳）、暂时失败（不缓存）；超时、额度跳过、网络错永远不写成空。
6. 截止显式化：Cutoff 对象从 cfg 传到每个通道和 KB（KB 检索也按论文年份过滤），不再依赖线程局部和 sys.modules。
7. 清单自动生成：每次运行写 manifest.json，包含 git sha + dirty 标记、加载到的仓库模块及其 blob 哈希（LF 归一化）、
   prompt 哈希、KB MANIFEST 哈希、第三方 commit + patch 哈希、pip freeze 哈希、服务端模型名、非密钥环境快照、缓存快照 id。
   compilescholar verify <run_id> 在任意机器上核验。取代手写 FREEZE_*.json。

### 5.3 文件去向

| 现在 | 去向 |
|---|---|
| _shared/tools/answer_pipeline.py | answer/pipeline.py + stages/* + prompts/*；KB 类 → kb/store.py；ext_search → sources/search.py；删 _ext、_s2*、array |
| _shared/tools/refgraph.py | sources/{openalex,s2,crossref,arxiv,refgraph}.py，HTTP 走 sources/http.py |
| _shared/tools/cutoff.py、sources._knowledge_cutoff、build_kb_v2.CUTOFF_YEAR | core/cutoff.py |
| src/kb_compiler/retrieve/hybrid.py | kb/index.py |
| src/kb_infra/llm.py、embedding.py | llm/client.py、llm/embedding.py、llm/jsonparse.py；去掉 CERT_NONE 与重复循环 |
| src/retrieval/sources.py（SciverseClient、令牌桶） | sources/sciverse.py、sources/http.py |
| src/retrieval/search_service、circuit、rerank、query_understanding、gap_search、lineage_walk、coarse_store、citation_graph | 选一份并入 sources/search.py 与 refgraph；其余 legacy/ |
| src/retrieval/registry.py、acquisition*、pdf_resolver、processing、process_workflow、evidence_units、export、extraction_runs、sciverse_ingest、search_resolve | 若 Multi 语料获取进发布范围 → compile/acquire/；否则 legacy/ |
| src/sci_evo_extract.py、retrieval/_see_* | 删除（改为直接导入） |
| src/kb_compiler/records、registry*、views/compiler、views/search_text | compile/records、compile/registry、compile/views |
| src/kb_compiler/views/field_state.py；cs2/base_kb_build/{build_state_v2,merge_state_v2}.py | compile/state/ |
| cs2/base_kb_build/{build_kb_v2,embed_kb_v2}.py | compile/pipelines/kb_v2.py 的阶段，由 compilescholar kb build --config configs/kb/cs2_v2.yaml 驱动 |
| cs2/base_kb_build 其余上游脚本（survey_pool、fetch_*、snapshot_fill、arxiv_fill、backbone_pool、hub_consensus、compile_base_kb 等）与 cs2/ 下就地改写脚本 | 二选一（§8 决策 3）：显式成为 DAG 阶段（每步写新文件），或冻结 base_kb 为带哈希的数据制品、脚本进 legacy/ 并写清顺序 |
| cs2/direct_judge.py、cs2_scoring.py、harness_to_judge.py | eval/cs2/judge.py、eval/cs2/scoring.py、baselines/harness/adapter.py |
| review_1002/p6/{judge_nuggets,analyze,build_inputs,citeconv}.py、cs2/run_dsb_judges.sh | eval/dsb/ |
| cs2/{harness_arm_run,dsb_harness}.py、_shared/mcp/* | baselines/harness/ |
| cs2/{cs2_gptr,cs2_storm,dsb_storm,cs2_paperqa,memorized_adapter}.py | baselines/*.py（检索改走 sources/search.py，见 §8 决策 2） |
| cs2/run_vnext*.py、scholarqa_multi/run_vnext_multi.py | 删除，换成 compilescholar answer --bench <x> --config … |
| _shared/tools/multi_*、scholarqa_multi/score_multi_* | eval/multi108/、eval/stats.py |
| base_kb_build/{heldout_field_eval,fetch_heldout_refs,build_survey_gold,clean_survey_gold,summarize_field_eval}.py | eval/field/ |
| build.py、src/kb_compiler/cli.py | compilescholar kb build（Multi 链） |

归档到 legacy/（不删，留 INDEX.md 说明它支撑过哪些已报结果）：_shared/tools/evidence_gate2r_harness.py、report_adapter.py、report.py、
render_records.py、judge_answer.py、judge_kb.py、ps53r_run.py、behavior_probe.py、bench_local_concurrency.py、decompose_points.py、
citation_correctness_eval.py；cs2/ 下 probe_l1*、audit_s5_*、growth_*、patch_*、test_*（3 个非 pytest 脚本）、cmp15、batch_compare、
judge_run、judge_dev20、judge_batch_task（两份）、gen_judge_task、judge_smoke_task、judge_runs_*/、cs2_runner、queue_run、watchdog、
deps_check、min_task；scholarqa_multi/kb/patch_*、gate3_*、probe_*；src 中 §2.3 列出的无入口模块。
external_tools.py 等 GPTR/STORM/PaperQA 改走统一检索之后再归档。
score_compare.py 直接删（口径错误，留着会被误用）。

### 5.4 配置与 CLI

- 合并顺序：代码默认 < configs/base.yaml < configs/local.yaml（本机端点、LAN、NAS，gitignore） < --config 实验 YAML < 环境变量（只放密钥与端点）< CLI。
- 库代码不读环境变量；cli.py 调 core/secrets.load() 一次，把 Settings 往下传。进程环境优先于 .env。
- 截止、并发、top-k、预算（cite budget_s、prefetch、max_ev、screen batch、min_sim 0.55 等现在散在函数默认值里的数）全部进 AnswerConfig。
- 每次运行把解析后的配置写成 runs/<id>/config.resolved.yaml；同一 run_id 下配置变了就拒绝续跑。

```
compilescholar kb build   --config configs/kb/cs2_v2.yaml
compilescholar answer     --bench cs2 --split test --config configs/bench/cs2_test_v9b.yaml --run-id cs2_test_v9b_r1 [--replay]
compilescholar judge      --bench cs2 --run-id cs2_test_v9b_r1
compilescholar score      --bench cs2 --runs cs2_test_v9b_r1 cs2_test_v9b_r2 --paired harness_test100
compilescholar baseline   gptr --bench cs2 --config configs/bench/cs2_test_v9b.yaml
compilescholar field-eval --config configs/field/heldout18.yaml
compilescholar verify     --run-id cs2_test_v9b_r1
```

experiments/cs2/run.sh 只剩上面几行；基准适配（题目加载、输出格式）在 eval/<bench>/data.py 与 answer/formats/。

### 5.5 发布形态

研究仓库保留为私有实验记录；对外发布用新的干净仓库（或 orphan 分支）：只含 src/、configs/、experiments/、tests/、third_party 子模块 + patches、docs/；
KB v2、题集派生物、test 运行的 answers / trace / judge 输出 / HTTP 缓存快照作为数据制品单独发布（Zenodo / HF dataset），附 sha256 清单与下载脚本。
发布前跑一次密钥扫描。

## 6. 迁移顺序（保护在跑任务）

冻结闭包（CS2 test 跑完、用户确认之前一律不动）：_shared/tools/{answer_pipeline,refgraph,cutoff,external_tools}.py、_shared/mcp/*、
cs2/ 下全部 runner / judge / scoring / harness 文件与 run_test_final.sh、cs2/base_kb_v2/、整个 src/、.env、
scratch/asta-bench（editable 安装）、$HOME/.sciverse_bucket.json 与 .pace_*.json、refgraph_cache/、s2_cache/。
原因：r2 与各 judge 是 r1 跑完后新起的进程，会从磁盘重新 import；harness 每题新起一个 retrieval_mcp 子进程；refgraph 等是惰性导入。

主工作区禁止：checkout / switch / stash / reset / pull / rebase / gc / filter-repo / 改 autocrlf / 改 .gitattributes 后 renormalize。

阶段 0（现在，只读）：本审计；与 AUDIT-WORKSPACE 对齐。

阶段 1（现在即可，不碰主工作区）：建 sparse worktree。全量检出会再落 6 GB 被跟踪数据，所以用：
```
git worktree add --no-checkout ../CompileScholar-refactor -b refactor/package
git -C ../CompileScholar-refactor sparse-checkout set --no-cone /src/ /tests/ /conf/ /build.py /pyproject.toml \
    "/.research_tmp/experiments/benchmarks/_shared/tools/*.py" "/.research_tmp/experiments/benchmarks/_shared/mcp/*.py" \
    "/.research_tmp/experiments/benchmarks/cs2/*.py" "/.research_tmp/experiments/benchmarks/cs2/base_kb_build/*.py"
git -C ../CompileScholar-refactor checkout refactor/package
```
worktree 内约束：不联网（$HOME 下的节流 / 冷却文件与在跑任务共享，会抢配额）；不运行任何带 C:\Users 绝对路径的旧脚本
（它们会读写主工作区）；KB 数据只读引用主工作区（通过 configs/local.yaml 指过去）。

阶段 2（worktree 内，行为零变化）：
1. 建 src/compilescholar 骨架，把活代码复制进去（主分支上暂不 git mv），修导入，删 sys.path / 绝对路径 / import 时副作用。
2. 先写表征测试再改动：cutoff.allowed 判定表；hybrid 在小 KB 上的排名；_interleave、assemble 用 answers_test100_r1 的 trace
   重放（从 trace 里的 evidence 与文本重建，断言 sections 逐字节一致）；refgraph 用 refgraph_cache 里的真实响应做离线夹具；
   cs2_scoring.summarize 对现有 direct_scores 文件复算一致。
3. 新旧并排等价：同一输入下，新包渲染出的每个 prompt 与旧 answer_pipeline 逐字节相同（不调 LLM 即可验证）。
4. 再做行为修复（§7 中会改变输出的条目），每条单独提交，dev 上记录差异。

阶段 3（用户确认 CS2 test 已跑完、已判、已存档之后）：
1. 在 main 提交当前状态，打 tag cs2-test-v9b-final；把 refgraph_cache / s2_cache 打包存档并记哈希；补一份覆盖 §4.4 漏项的清单。
2. 合并 refactor/package；旧文件 git mv 进 legacy/（复现 v9b 用 tag，不留兼容垫片）。
3. 加 .gitattributes（* text=auto eol=lf），一次性 renormalize；此后哈希按 LF 计算。
4. git rm --cached 重型数据（embed cache、*.bin、answers 大文件、review_1002 等），数据目录改为 data/ + MANIFEST.sha256。

阶段 4（需用户明确批准，破坏性）：处理历史。推荐不改写研究仓库历史，而是新建干净发布仓库；若坚持推送研究仓库，
在镜像克隆上用 git filter-repo 去掉 >100 MB blob 与 promote_to_deep.py 的 token，先打 bundle 备份。无论哪种，MINERU_TOKEN 都要轮换。

阶段 5：对照臂迁到 sources/search.py；DSB 汇总改缺答记 0；判分适配层成文；dev 上小规模冒烟（只看机制指标），确认后进入 UPGRADE-PLAN 的 W2。

## 7. 问题清单

| # | 级别 | 问题 | 证据 | 修法 |
|---|---|---|---|---|
| H1 | 高 | 生产答题管线不在包里，靠 sys.path 注入；retrieval 未打包，依赖未声明 | run_vnext.py:20、answer_pipeline.py:27-31、pyproject.toml:30 | §5 包结构；pyproject 补依赖 + lock |
| H2 | 高 | 活路径零测试 | §4.1 表 | 阶段 2 的表征测试 + golden 回放，进 CI |
| H3 | 高 | 现有历史无法推送 GitHub | origin/main..main 17 个 blob >100 MB，最大 2.44 GB；main 领先 215 | 新发布仓库；或镜像上 filter-repo（需批准） |
| H4 | 高 | 明文密钥进了提交 | scholarqa_multi/promote_to_deep.py:47（9872253b，本地） | 轮换；改读环境变量；从历史中去除 |
| H5 | 高 | KB v2 输入不在 git 中，建库链无顺序记录 | build_report inputs_sha16 只匹配工作区；HEAD blob 不同；5 个就地改写脚本 | 冻结 base_kb 为哈希制品，或 DAG 化每步写新文件 |
| H6 | 高 | 冻结清单换机不可核验且漏项 | 工作区 CRLF 哈希 vs LF blob；漏 llm.py 等（§4.4） | manifest 自动生成，LF 归一化 + 全闭包 + 环境快照 |
| H7 | 高 | 官方判分器在未跟踪目录且带运行时补丁 | astabench editable → scratch；direct_judge.py:17-26,61-92；dsb nuggetizer/llm.py 被改 | third_party 子模块 + patches/；补丁集中成判分适配层并在论文披露 |
| H8 | 高（未证实） | 三个臂经三条不同代码访问 Sciverse，GPTR 走的是自家注释说并发下会空的路径 | cs2_gptr.py:53-58 → external_tools.py:310-314,415 → search_service.py:184（15 s）、circuit.py:38（600 s）；answer_pipeline.py:147-151 注释 | 统一 Searcher；先从 GPTR trace 统计每次检索空 / 熔断率，再决定是否重跑 |
| H9 | 高 | Multi 跑批重跑必崩 | run_vnext_multi.py:27 不调 super().__init__，无 _fam_vecs；answer() 默认 use_state=True（eefea3a0，晚于 Multi 跑批）；不联网实测 AttributeError | KB 统一加载；无状态层时 state_search 返回 []；加冒烟测试 |
| H10 | 高 | TLS 校验全局关闭 | kb_infra/llm.py:31-33，call_paratera 带 key 走它 | 默认校验；只对配置白名单内的自签端点放行 |
| M1 | 中 | 静默降级不计数 | answer_pipeline.py:45-48,355,466-470,523-524；refgraph.py:144-146；STATS 不入 trace | §5.2 第 4 条 degradation 计数 + health.json |
| M2 | 中 | 暂时失败被永久缓存成空 | refgraph.py:421-422；1,554 条路由缓存中 7 条 source=none | 缓存三态；只在确定的负结果时写 |
| M3 | 中 | 外部检索不可重放，缓存不进清单，r2 复用 r1 缓存 | sources.py 无缓存；run_test_final.sh 注释；.gitignore:167-168 | 录制回放层；缓存快照随运行发布；论文说明 r1/r2 引文通道非独立 |
| M4 | 中 | 结果依赖墙钟与并行负载 | cite_expand budget_s=90（:255）、prefetch 120 s（:380）、$HOME 共享节流 | 预算进配置并记录触发次数；节流状态放 cache_dir |
| M5 | 中 | DSB 缺答被剔除，与 CS2 缺答记 0 相反 | run_vnext_dsb.py:69-74；judge_nuggets.py:95-97；analyze.py:24 | 统一缺答记 0，失败写错误行 |
| M6 | 中 | 截止逻辑反向依赖实验目录 | sources.py:34-43 sys.modules.get("cutoff") | core/cutoff.py，显式传参 |
| M7 | 中 | 一次导入拉进 9.9k 行与 openai / dotenv | sci_evo_extract.py 无条件导入 8 个 _see_*；retrieval/__init__.py 导入 registry.py 3,806 行 | 删垫片；normalize_doi 挪到小模块 |
| M8 | 中 | 重复实现（LLM、嵌入、学术 API、节流、KB 加载、.env、判分） | §3.2 表 | 各收敛成一份；其余归档 |
| M9 | 中 | 配置散、默认值打架、.env 覆盖进程环境 | LOCAL_MAX_CONCURRENT 96/48/16/4；llm.py:648,764-766；.env.example 过时 | §5.4 分层配置；进程环境优先 |
| M10 | 中 | 硬编码绝对路径 / LAN IP | §2.4 表（254 处 C:/Users、32 处 LAN IP） | paths.py + configs/local.yaml；CI 加 lint 拒绝绝对路径 |
| M11 | 中 | 运行配置不全、续跑会混配置、错误行不重试 | run_vnext.py:53-55,71 | resolved config + manifest；配置哈希不同拒绝续跑；--retry-errors |
| M12 | 中 | 断点文件非原子整写 | run_vnext.py:75（每题重写 21 MB）；refgraph.py:129,374,422 | answers.jsonl 追加；缓存 tmp + os.replace |
| M13 | 中 | 词数上限靠 str.replace，prompt 无版本 | answer_pipeline.py:508-509 | prompt 模板变量；prompt 哈希入 manifest |
| M14 | 中 | 产物与代码混放、审查目录当数据目录 | cs2/ 185 json + 144 log；run_vnext_dsb.py:24；build_kb_v2.py:30 | runs/ 与 data/ 分区 |

| M15 | 中 | 1,038 个命中忽略规则的文件仍被跟踪；6.1 GB 数据进 git | git ls-files -ci --exclude-standard；§4.4 | 阶段 3 git rm --cached；规则改成白名单式 |
| M16 | 中 | 同一论文经 KB / 外部两通道得到两个引用编号 | assemble 按 paper_key（kb:<pid> vs ext:<md5>）；r1 355 节中 11 节同题双编号 | 证据表按 DOI / arXiv / 规范化标题做论文身份合并 |
| M17 | 中 | KB 向量只按条数对齐 | answer_pipeline.py:66-67；meta 已有 records_sha16 未用 | 加载时校验哈希 |
| M18 | 中 | 判分 / 汇总脚本泛滥，旧口径未删 | CS2 9 个 judge 相关脚本 + judge_runs_* 4 份；score_compare.py:4 | 每个基准一个 judge + 一个 score；score_compare 删除 |
| L1 | 低 | 死代码与过时说明 | answer_pipeline.py:25（array）、:136-141（_ext）、:198-252（_s2*，含 import 时 makedirs）；run_vnext.py:35-36 与 run_vnext_dsb.py:33-34 帮助文本仍说"S2 引文扩展"，实际是 refgraph | 删除；改帮助文本 |
| L2 | 低 | import 时副作用 | answer_pipeline.py:32-36,199-200；refgraph.py:33-34；embed_kb_v2.py:15 等改 NO_PROXY | 移到 CLI 启动或客户端构造 |
| L3 | 低 | llm.load_env 重复的空循环 | kb_infra/llm.py:44-49 | 删 |
| L4 | 低 | 大车道信号量惰性创建不加锁 | kb_infra/llm.py:802-810 | 构造时创建 |
| L5 | 低 | 环境变量名拼错 | retrieval/_see_llm.py:14 PERATERA_* | 随垫片删除 |
| L6 | 低 | 测试写死路径、引用兄弟仓库、2 个测试不被收集 | tests/ 7 个文件；test_external_tools_backflow.py:15；src/kb_compiler/records/test_*.py | conftest + 安装包；移入 tests/；pyproject 设 testpaths |
| L7 | 低 | OpenAlex / Crossref polite pool 用占位邮箱 | refgraph.py:36,317 example.org | 从配置读真实 mailto |
| L8 | 低 | 模型名写死 | field_state.py:30 | 进 Settings |
| L9 | 低 | src 内 15 个文件自行 sys.path.insert | 例 records/common.py:18-20、views/tools.py:980 | 包安装后删除 |
| L10 | 低 | registry_growth 失败转储写死绝对路径 | registry_growth.py:99,268 | 改成 runs/<id>/forensics |
| L11 | 低 | kb CLI 装成 wheel 后找不到 build.py | kb_compiler/cli.py:20 parents[2] | 并入 compilescholar CLI |
| L12 | 低 | 第三方仓库与 venv 嵌在 benchmarks 下 | deepscholar/dsb（嵌套 .git，索引异常）、deepscholar/venv311 | third_party/ 子模块；venv 出仓库 |
| L13 | 低 | .git 垃圾 2.94 GiB、松散对象 10 GiB | git count-objects -vH | test 跑完后 git gc（耗盘，避开在跑任务） |
| L14 | 低 | 误写的目录 | _shared/cs2/arm_ours/（旧 runner 相对路径写错，09-26） | 归档 |

## 8. 需要用户拍板

1. 发布形态：新建干净发布仓库（推荐），还是在镜像上 filter-repo 改写研究仓库历史后推送。两者都要先轮换 MINERU_TOKEN。
2. 对照臂检索统一：把 GPTR / STORM / PaperQA 改走统一 Searcher 会改变这些臂的行为，已有数字不能直接沿用。
   选项 a：先统计 GPTR trace 的空检索 / 熔断率，影响可忽略则冻结旧结果并在论文披露路径差异；
   选项 b：统一后重跑对照臂（花额度与时间，但公平性无可挑剔）。建议先做 a 的统计再定。
3. 旧建库链（compile_base_kb + 就地改写脚本）：做成可重跑的一等 DAG，还是把 base_kb 当冻结数据制品发布、脚本归档并写明顺序。
   前者要重跑带 LLM 的步骤且依赖 NAS 快照；建议后者，论文写"KB 构建过程 + 制品哈希"。
4. 重构是否允许顺带改行为（缓存三态、截止进 KB 检索、同论文引用合并、DSB 缺答记 0）。建议：搬迁阶段行为零变化、有等价测试守着；
   行为修复逐条单独提交，只在 dev 上评估，按 UPGRADE-PLAN §3 预注册规则处理。
5. 判分补丁（max_retries 4、base_delay 1.0、idx 平移、Connection: close、DeepSeek 判分模型）是否在论文里作为判分适配层逐条披露，
   以及 UPGRADE-PLAN 决策 5 的官方 judge 全量重判是否用 patch 关闭的原样 scorer。
6. 数据制品托管位置（Zenodo / HF dataset / 校内存储）与许可：KB 里含综述与论文的逐字片段，发布前需要确认可再分发范围。
