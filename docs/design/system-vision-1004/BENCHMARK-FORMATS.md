# 下游基准的数据格式与接入方式（2026-10-05）

用途：给"历时领域认知"系统（带 `as_of` 参数的工具）定接口。本文回答每个基准的题目形态、论文 ID、时间截止、候选语料、官方评测入口、外部工具接入点和外部依赖。

核查方式与可信度：
- **ScholarCatalyst、IdeaForecastBench**：本人一手核查。读了 HF datasets-server 的 rows/statistics 接口、HF 文件树、GitHub 源码（raw）。IdeaForecastBench 另抽了 40 篇 markdown 做统计。
- **PreScience、TaxoBench、Old Ideas/NovGauge、MasterSet、LimitGen**：由 3 个子代理一手核查（HF API、GitHub 源码；TaxoBench data.jsonl 和 MasterSet eval.parquet 实际下载解析过），本人没有逐行复核。
- 各节"未核实"条目注明了卡在哪一步。全程没有下载超过 50 MB 的文件。

## 0. 结论先行

| 问题 | 答案 |
|---|---|
| 论文 ID 能否对齐 arXiv id | **能直接对齐**：ScholarCatalyst（语料 99.8% 是 `arxiv_` 前缀）、IdeaForecastBench（全是 arXiv id）、PreScience（每行都有 `arxiv_id` 列）。**部分能**：NovGauge（926 条记录中 431 条有 arxiv_id）。**只能按标题对齐**：TaxoBench、MasterSet、Old Ideas。**按 OpenReview id，需经 OpenReview 转一道**：LimitGen-Human |
| 截止粒度 | **按天**：PreScience（`date < cutoff`，字符串比较）、Old Ideas（全局固定 2025-03-01）。**按月**：ScholarCatalyst（源论文月份及以前全部可见，同月放行）、IdeaForecastBench（截止月 1 号，论文日期按 arXiv id 推成月末）。**按年**：MasterSet（候选池 2018–2024，题目 2025）。**无截止**：TaxoBench、LimitGen（RAG 不限日期）、NovGauge |
| IdeaForecastBench 的 markdown 能否切出引用句 | **能，质量不错**。抽样 40 篇，39 篇有可识别的参考文献块，正文引用标记保留完好：21 篇是数字式 `[n]`，19 篇是作者-年份式。粗算标记能解析到参考条目的比例：数字式 89%（1352/1518），作者-年份式 99%（1233/1245）。主要问题有两个：(a) 参考条目映射到 arXiv id 要靠标题匹配，条目里自带 arXiv 号的只有 29/39 篇，且只占其中一部分条目；(b) MinerU 转换有噪声，例如 alpha 标签 `[AZLS19]`、丢失的方括号、LaTeX 里的 `w_{2}[0]` 会误匹配为数字标记。详见 §6.4 |

---

## 1. ScholarCatalyst（arXiv 2610.02202）

数据：HF `ScholarCatalyst/ScholarCatalyst`，未设 gating。代码：GitHub `stanford-iris-lab/ScholarCatalyst`。许可 CC BY-NC 4.0（数据和代码同一许可）。

### 1.1 文件与样本
HF 文件树里只有 `corpus.jsonl`（277,604,140 B）、`queries.jsonl`、`rels/core_query.jsonl`、`rels/subfield_query.jsonl`，外加 README 和插图。**没有 `citations.jsonl`，也没有 `corpus_full_text.jsonl`**，代码里的 `expand` 和 `read` 工具都依赖这两个文件（见 §1.5）。

`corpus.jsonl`：190,896 行（datasets-server 统计）。字段 `id, title, text, primary_category, categories, published`。`text` 是"标题+摘要"，**没有全文**。样本（截短）：
```json
{"id": "arxiv_2001.00116", "title": "Exploiting the Sensitivity of $L_2$ Adversarial Examples to Erase-and-Restore",
 "text": "By adding carefully crafted perturbations to input images, adversarial examples (AEs) ...",
 "primary_category": "cs.CV", "categories": ["cs.CV","cs.CR","cs.LG","eess.IV"], "published": "2020-01-01"}
```

`queries.jsonl`：894 行（207 条 `core_query`，687 条 `subfield_query`）。字段 `id, paper_id, question, type, paper_published, paper_domain`：
```json
{"id": "q_2507.06542_core_query_0", "paper_id": "arxiv_2507.06542",
 "question": "Our question is how a limited communication budget should be allocated over the course of fully decentralized training ...",
 "type": "core_query", "paper_published": "2025-07-09", "paper_domain": "cs.LG"}
```

`rels/core_query.jsonl`：`positive_docs` 是 `[{id, author_rationale}]`（每题平均 3.19 个），`hard_negatives` 是 id 列表（每题平均 10.0 个）。`rels/subfield_query.jsonl` 的 `positive_docs` 是纯 id 列表，`author_rationale` 放在顶层字符串里。
```json
{"query_id": "q_2507.06542_core_query_0",
 "positive_docs": [{"id": "arxiv_2310.14423", "author_rationale": "Gu et al. introduced the Quadratic Synchronization Rule ... directly motivated us to treat communication timing as a schedulable resource ..."}, ...],
 "hard_negatives": ["arxiv_...", ...]}
```

### 1.2 ID 体系
- 语料 id 有三种前缀。用 datasets-server 按 offset 二分定位，三类在文件里连续排列：
  - `arxiv_YYMM.NNNNN`：190,543 条（offset 0–190542）
  - `oa_W…`（OpenAlex work id）：182 条（190543–190724）
  - `s2_<40 位 hex>`（S2 paperId）：171 条（190725–190895）
  - oa/s2 两类的 `primary_category` 填的是 OpenAlex/S2 的概念名（如 "Computer vision"、"Computer Science"），不是 arXiv 类目。
- 源论文 `paper_id` 894 条全部是 `arxiv_` 前缀（长度恒为 16）。
- 金标里 arXiv 占绝大多数：core 正例 604 arxiv / 32 s2 / 25 oa（91%）；subfield 正例 3904 / 154 / 126（93%）。
- **结论**：去掉 `arxiv_` 前缀就是 arXiv id。剩下约 7–9% 的金标是 S2/OpenAlex 记录，需要我们的系统也能产出这两类 id（可以按标题对齐到语料里的那一条）。

### 1.3 日期与截止
- `paper_published`：按天，范围 2025-01-02 到 2026-08-14。语料 `published` 也按天，另有 21 条只有年份（datasets-server 字符串长度统计显示长度为 4–6）。
- **语料本身不按截止裁剪**：里面有 2026-01 的论文（offset 190500 是 `arxiv_2601.07060`）。截止在运行时做。
- 实现：`src/evaluation/retrieve.py::temporal_filter(ranking, query_date, corpus_dates)`，**按月比较**：
  - 文档年月 ≤ 源论文年月就保留，**同月一律放行**；
  - 日期无法解析的放行；
  - 只有年份（month=0）且与源论文同年的放行；
  - 日期缺失时，`parse_ym` 从 `arxiv_YYMM` 推年月。
- 源论文自身由 `agentic/ranking.py::withhold()` 从搜索结果剔除，`assemble()` 也会从最终排名里剔除。按 `utils.source_ids()`，同标题的其他 id 也应一并剔除，但 queries 里没有 `paper_title` 字段，实际只剔除 `paper_id` 本身。
- 对我们的含义：工具的 `as_of` 可以取 `paper_published` 所在月的月末，与官方口径一致；也可以更严，取 `paper_published` 前一天。两种都要在报告里写明。

### 1.4 官方评测入口
```bash
python agentic_search.py --agent toolcall --model <model> --backfill bm25 --bench-dir $BENCH_DIR
python evaluate.py --model agentic/toolcall-bm25 --bench-dir $BENCH_DIR
```
- 输入：`evaluate.py` 会自动发现 `runs/agentic/**` 下的叶子目录，读其中的 `{query_type}.jsonl`，每行 `{"query_id": "...", "ranking": [{"doc_id": "..."}, ...]}`。
- 指标：`precision@k / recall@k / ndcg@k`，k = 5, 10, 15, 20, 25, 50, 100；外加 `trajectory_recall`（正例在 agent 整个轨迹里出现过的比例）。论文主报 Recall@20。

### 1.5 agent 与工具定义（`src/evaluation/agentic/agents/toolcall.py`）
- 常量：`MAX_STEPS=5`（工具轮数）、`SEARCH_K=10`、`POOL_CAP=60`。默认 backbone 是 `gpt-4.1`（`agentic_search.py --model`）。
- 工具 schema（OpenAI function 格式）：
  - `search(query: str)`：返回至多 10 篇，每篇格式为 `"[{id}] {title} ({year})\n{abstract[:400]}"`，篇与篇之间用空行隔开。底层是 `withhold(build_retrieve_fn(...), withheld)(query, 10, qdate)`，已做日期过滤。
  - `expand(doc_id: str)`：列出该文在语料内引用的论文，同样经 `temporal_filter`。**只有 `<bench>/citations.jsonl`（每行 `{"id","cites":[...]}`）存在时才注册**，HF 不提供这个文件。
  - `read(doc_id, section?)` / `sections(doc_id)`：需要 `--full-text corpus_full_text.jsonl`（代码注释说约 7.7 GB），**HF 也不提供**。
- 流程：
  1. 先用原问题做一次种子搜索；
  2. 至多 5 轮工具调用；
  3. 池中每篇用同一模型按 0–10 打"有用性"分（`JUDGE` prompt），按分排序；
  4. 输出 JSON id 列表；
  5. `ranking.assemble()` 依次拼接 agent 列表、轨迹里见过的 id（上限 `TRAJECTORY_CAP=25`）、backfill 检索器结果，补到深度 100，再做一次 `temporal_filter`。
- 另外两个 agent：`grep` 用 bash/ripgrep 在每题的月份桶视图里检索（`agentic/views.py::build_view`）；`deepresearch` 至多 20 次 search、10 次 read。

### 1.6 工具接入点
- **推荐做法**：新建 `agentic/agents/<ours>.py`，实现 `run(query, view_dir, traj, tools) -> str`（返回 JSON id 列表），注册到 `agentic_search.py` 的 `AGENTS` 字典。
  - 以 `toolcall.py` 为模板，在 `schemas` / `handlers` 里加一个工具，例如 `field_state(query)`，内部传 `as_of=query["paper_published"]`。
  - 返回内容里的论文 id 必须是语料 id（`arxiv_*` / `s2_*` / `oa_*`），才能进池和计分。
  - 做 A/B 时，有/无两臂用同一 backbone、同一 `--backfill`，只差这一个工具。
- **轻量做法**：不走 runner，直接写 `runs/agentic/<pipeline>/<query_type>.jsonl`，`evaluate.py` 照样能评。
- 我们的"他述"层需要自带全文：官方语料只有摘要，引文图和全文都没发布。

### 1.7 外部依赖与坑
- LLM 按模型名前缀路由（`llm_rerank.py::make_client`）：裸名走 OpenAI（`OPENAI_API_KEY`），`google/…` 走 Gemini，`anthropic/…` 走 Anthropic。
- **OpenAI 兼容端点要改一行代码**：`make_client` 显式传了 `base_url=OPENAI_BASE`（常量 `"https://api.openai.com/v1"`），所以环境变量 `OPENAI_BASE_URL` 不生效，需要改这个常量或 monkeypatch。
- BM25（`bm25s`）在本地跑。dense 检索器（qwen3-8b 等）从 HF 下载后本地跑；`gemini-2`、`text-embedding-3-large` 走 API。
- 依赖固定在 `requirements.txt`：openai 2.21.0、anthropic 0.102.0、bm25s 0.3.9、faiss-cpu、torch 等。

---

## 2. PreScience（arXiv 2602.20459）

数据：HF `allenai/prescience`，许可 ODC-BY。代码：GitHub `allenai/prescience`，Apache-2.0。核查来源：子代理（HF `/info` `/rows` `/statistics` + 源码）。

### 2.1 样本与字段
`train.parquet` 和 `test.parquet` 是同一套 schema：

| 字段 | 类型 | 说明 |
|---|---|---|
| `corpus_id` | str | S2AG corpusId |
| `arxiv_id` | str | test 全表无缺失，形如 `2509.25791` 或 `math/9201203`，不带版本号 |
| `date` | str | `YYYY-MM-DD` |
| `title`, `abstract` | str | |
| `categories`, `roles` | list[str] | |
| `key_references` | list[{`corpus_id`, `num_citations`}] | S2 "highly influential" 引用；伴随论文多为 null |
| `authors` | list[{`author_id`, `name`, `publication_history`: list[corpus_id], `h_index`, `num_papers`, `num_citations`}] | |
| `citation_trajectory` | list[int] | 逐月累计引用数，只有 target 有 |

```json
{"corpus_id":"281681978","arxiv_id":"2509.25791","date":"2025-09-30","roles":["target"],
 "key_references":[{"corpus_id":"268532501","num_citations":5}],
 "authors":[{"author_id":"y gao_75","publication_history":["52185833", "..."],"h_index":37}],"citation_trajectory":[0,0]}
{"corpus_id":"55712766","arxiv_id":"math/9201203","date":"1989-10-26","roles":["target.author.publication_history.key_reference"],"key_references":null,"authors":null}
```
- 目标论文和伴随论文在同一张表，用 `roles` 区分。四种取值：`target`、`target.key_reference`、`target.author.publication_history`、`target.author.publication_history.key_reference`。一行可以有多个 role。
- 规模：test 464,942 行，其中 target 52,836 篇（2024-10 到 2025-09）；train 373,716 行。每个 split 自带完整的伴随集，最早到 1989 年。
- 另有两个作者文件：`author_publications.jsonl`（`{"key": author_id, "value": [corpus_id...]}`）和 `author_disambiguation.jsonl`。

### 2.2 ID 与日期
- **ID**：S2 corpusId 是主键，每行同时有 `arxiv_id`，可以直接互转。构建时 `parse_s2_paper_records` 丢掉了没有 `externalIds.ArXiv` 的论文，所以语料只含 arXiv 论文。
- **日期按天**：
  - 目标论文取 arXiv 第一版的创建日期（`download_target_papers.py::parse_arxiv_created_date` 读 `versions[0].created`）；
  - 伴随论文取 S2 的 `publicationDate`（`utils.parse_s2_paper_records`）。
  - 两种来源的口径不同，S2 日期可能是会议日期。

### 2.3 prior work selection 任务（`task_priorwork_prediction/`）
- **输入**：`dataset.py::create_evaluation_instances(all_papers, sd2publications, all_papers_dict)` 生成 `(date, {"corpus_id","author_ids","gt_reference_ids"})`。**模型只拿到作者 ID 和日期，拿不到目标论文的标题和摘要**。只纳入"至少一位作者此前有带 key_references 的论文"的 target。
- **候选池**：
  - frequency 基线：作者最近 10 篇（`--num_recent_papers 10`）论文的 key_references，由 `get_cited_references_for_author` 取，不足时随机补早于目标日期的论文；
  - embedding 基线：全语料中 `date < cutoff` 的论文，默认 `--k 1000`。
- **H<t 的实现**：
  - 统一用 `paper["date"] < cutoff_date` 做字符串比较，按天；
  - 作者历史由 `get_preexisting_publications_for_author` 对日期列表做 `bisect_left`（前提是列表已按日期排序，**未核实**：HF filter 接口一直返回 "index is loading"）；
  - embedding 基线在 `create_paper_index(cutoff_date, …)` 建好初始索引后，按日期逐篇"先预测、再 `add_vector_to_index`"，同一天排在前面的论文会进入候选池，比严格的 `<` 略宽。
- **预测文件**：`{"metadata":[…], "data":[{"corpus_id","gt_reference_ids","predicted_reference_ids","predicted_reference_scores"}]}`。
- **评分**：`python3 -m task_priorwork_prediction.evaluate --predictions_path <file>`，调 `evaluate_predictions()`：
  - `utils.calculate_ndcg`：二值相关；预测条数少于 gt 条数时返回 0；
  - `utils.calculate_precision_recall_f1`：k = len(gt)，即 R-Precision；
  - 输出 `avg_ndcg / avg_precision / avg_recall / avg_f1`；
  - gt 直接读预测文件里自带的那份，不回查数据集。

### 2.4 工具接入点
- 在 `task_priorwork_prediction/` 下新写一个 baseline，实现 `predict_references(author_ids, cutoff_date, ...) -> (ids, scores)`，复用 `create_evaluation_instances`，`as_of = cutoff_date`。
  - 我们的工具要从"作者历史论文 → 领域认知 → 推荐 refs"这条链路出结果，最终必须映射回表内的 `corpus_id`，表外论文永远不得分。
- 多轮模拟版本：`multiturn/simulate.py::predict_prior_work` 里按 `args.priorwork_baseline` 分派，加一个分支即可。
- contribution generation 任务（给作者和 key refs，生成标题摘要）可以把"后人如何描述这些 refs"作为输入表示，但评分走 LACERScore（LLM）。

### 2.5 外部依赖
- prior work 任务不调 LLM，只用 embedding（gtr / specter2 / grit，依赖 faiss-gpu-cu12）。
- `OPENAI_API_KEY`：GPT 基线、LACER（默认 `gpt-5-2025-08-07`）。`ANTHROPIC_API_KEY`：Claude 基线。`S2_API_KEY`：只在重建数据集时需要。
- 代码里是 `openai.OpenAI()`，没传参数，按 SDK 惯例 `OPENAI_BASE_URL` 应该生效（**未实跑验证**）。
- train 和 test parquet 分别是 376 MB 和 480 MB，本次没下载。

---

## 3. MasterSet（arXiv 2604.17680，SDM 2026）

**与 AgentExpt 不是同一篇**。AgentExpt（arXiv 2511.04921，清华 FIB lab）全文没出现 MasterSet；它的仓库 `tsinghua-fib-lab/AgentExpt` 只有 README 和 LICENSE，**数据未发布**（论文写的是 108,825 篇、按日期 8:1:1 切分、Recall@20 / HitRate@5,10）。MasterSet 全文也不提 AgentExpt。核查来源：子代理（eval.parquet 已下载解析，train 只按 range 读了三列）。

### 3.1 位置与格式
- 代码分四个仓库：`NIU-Graph-Intelligence/MasterSet`（只有 README）、`masterset-benchmark`（基线与评测）、`masterset-toolkit`（抽取与标注）、`OpenPapers`（爬虫）。
- 数据：HF `trratul/MasterSet` 的 `data/train_eval_set/v1.0/{train.parquet 246MB, eval.parquet 29MB, all_papers_with_refs_and_labels.parquet 448MB}`。datasets-server 报 scan size 超限，用不了。
- 列：`paper_id`（内部 UUID）、`path`、`title`、`abstract`、`year`、`venue`、`authors`、`references`（JSON 字符串）。
  - 每条 reference 含 `target, matched_paper_id, title, title_from_ref, venue, year, type_1_output, type_2_output, type_3_output`，标签值是字符串 "1.0" / "0.0" / "nan"。
```json
{"matched_paper_id":"61e92a4f-…","title":"LoRA: Low-Rank Adaptation…","venue":"iclr","year":2022,"type_1_output":"1.0","type_2_output":"4.0"}
```
（这条 reference 来自查询论文 DataEnvGym，ICLR 2025。）

### 3.2 Type I 子任务
- **输入**：查询论文的 title + abstract。
- **金标**：`type_1_output == "1.0"` 且 `matched_paper_id` 非空。6,105 题至少有 1 条金标，平均 5.18 条。
  - Type I 的定义比"实验基线"宽：对比基线、所用基准或数据集的出处、直接在其上构建的方法都算。
  - 标签由 Gemini 2.5 Flash 标注，人与人之间的一致性 Krippendorff α = 0.461。
- **候选池**：train.parquet，67,761 篇，15 个会议，2018–2024。
- **查询**：eval 7,028 篇，全部是 2025 年（ICLR 3,704 + ICML 3,324）。
- **时间切分按年**，没有逐题"候选必须早于目标"的约束。
- 36,399 条金标里只有 31,391 条在池内，其余是 2018 年前或 2025 年的论文。评测脚本不剔除池外金标，**Recall 上限约 0.842**。
- **ID**：没有 arXiv id，也没有 S2 id，只能用标题（加 venue/year）对齐到我们的语料。

### 3.3 评测与接入
- 没有统一的评测入口，也没有约定的预测文件格式，每个基线的 `3-*-eval.py` 自己算。可参照 `src/sparse/bm25/2-bm25-eval.py`：
  - `extract_relevant_sets(references)` 抽金标集合；
  - `compute_all_metrics(relevant, retrieved)` 算指标，`retrieved` 是有序 paper_id 列表，剔除查询自身；
  - `EVAL_K_VALUES={"recall":[10,50,100,500],"ndcg":[10,20,30,50],"hr":[10,20]}`，另有 MAP / MRR。
- 论文 Table 7.1 的 Type I R@100：BM25 0.301，SciBERT-NTX 0.383。
- 没有 agent 或工具基线。接入方式：对每道 eval 题输出池内 paper_id 的排序，复用上面两个函数；`as_of` 统一取 2024-12-31。
- 外部 API：跑评测不需要。重标签需要 Gemini，作者用的 `run_on_prompts` 包没放出。

---

## 4. Old Ideas novelty-judge-bench（arXiv 2610.02022）与 NovGauge（arXiv 2609.11234）

**两篇独立的论文**。2610.02022 全文 grep "NovGauge" 为 0 次，互不引用。核查来源：子代理（源码与 HF）。

### 4.1 novelty-judge-bench
- 论文：*Old Ideas, Novel Problems: The Instability of LLM-Based Novelty Evaluation*，v1 2026-10-01。
- 代码：`noy-sternlicht/novelty-eval`，MIT。数据：HF `noystl/novelty-judge-bench`，CC-BY-4.0。
- **格式**：成对和逐点两种都有。共 14 个 config，只有 test split，另有 `_plan` 和 `backbone-*` 两类变体。
  - Human-Only：成对 154，逐点 299；Human+Generated：成对 154，逐点 308。
  - 逐点字段：`id, iclr_area, idea, label(POSITIVE/NEGATIVE), idea_source, title, rating, contribution, positive_signals, negative_signals`。
  - 成对字段：`idea_a/idea_b, expected_winner(0/1)`，其余字段带 `_a/_b` 后缀。
```json
{"id":0,"iclr_area":"foundation or frontier models, including LLMs","idea":"Aligning LLMs with human preferences…","label":"POSITIVE","title":"Token-Importance Guided Direct Preference Optimization","rating":6.5,"contribution":3.0}
```
- **金标**：用 claude-opus-4-6 从 ICLR 2026 审稿意见里抽取"新颖性信号"，再结合接收结果、评分前 10%、contribution ≥ 3 等条件筛选；负类条件镜像。人写的 idea 是去掉了实验数字的摘要。
- **ID 与日期**：只有 `title`，没有 arXiv id、OpenReview id，也没有日期。全部样本是 ICLR 2026 投稿。
- **"5 篇截止前摘要"的构造**（`src/novelty_eval/retrieval/retrieve_candidates.py::retrieve_candidates_for_idea()`，配置 `retrieve_candidates.yaml`）：
  1. `extract_search_queries()`：用 claude-opus-4-6 抽至多 3 条贡献点，再生成 `n_queries: 5` 条查询；
  2. 每条查询默认送 Ai2 Paper Finder（`src/paper_finder_api.py::search_papers()`，需要 secret `mabool_actor_id`）；设 `use_semantic_scholar: true` 可改走 S2 `/graph/v1/paper/search`；
  3. `retrieval_common._filter_and_select_candidates()`：
     - `_filter_by_date`：**全局固定 `cutoff_date: "2025-03-01"`，按天比较 `publication_date < cutoff`**；缺这个字段时退回 `year < 2025`。
     - 去掉同标题论文和 `relevance_score < 0.8` 的结果，按分数降序排列，按 corpus_id 去重。
     - 每条查询配额 floor(5/查询数)，不足时轮转补齐到 `top_k_candidates: 5`。
  4. `format_papers_for_prompt()` 拼成 `[metadata-i] … [abstract-i] …` 的文本，结果缓存为 `retrieval_cache.json`。
- **评委**：gpt-5.1/5.2/5.4、claude-sonnet-4-5、claude-opus-4-5/4-6；强 RAG 评委是 gpt-5.6-sol 加 arxiv.org web_search。
  - 路由：`src/utils.py::prompt_openai_client()`，模型名含 "claude" 的走 Anthropic，其余走 OpenAI。
  - 坑：调用时带 `reasoning`，走的是 `responses.create`（/v1/responses），只有遇到 AttributeError 才回退 chat.completions。**兼容端点如果不支持 Responses API，要改这一处**。
  - prompt 模板：`templates/pairwise_novelty.jinja2`、`pointwise_novelty.jinja2`；判据：`judge.py::EVALUATION_CRITERIA(_RETRIEVAL)`。
- **评测入口**：`python src/novelty_eval/run_benchmark.py --config src/novelty_eval/config/example_eval.yaml [--set llm_engine=...]`。
  - 主要配置：`test_mode: pairwise|pointwise`、`mec_k: 3`、`retrieval_cache_file`。
  - 指标（`metrics.py`）：成对主报 `accuracy_strict`（平局算错），逐点报 `f1_macro`；显著性用 paired bootstrap。
- **接入点**：
  - 最省事的是**自己生成 `retrieval_cache.json`**，不改代码：条目里只写 `{text, related_work}`，不写 `candidates`；`run_pairwise_experiment` / `run_pointwise_experiment` 会把 `related_work` 字符串原样注入模板的 `[Start Related Work]` 块。我们 `as_of=2025-03-01` 的领域认知文本放这里。
  - 如果写 `candidates` 列表，会被按 `relevance_score` 重排、截断，缺 `corpus_id` 的候选会被静默丢弃。
  - 注意：只要设了缓存，判据就切到 `EVALUATION_CRITERIA_RETRIEVAL`；论文 C.3 指出这个措辞本身会让评委过度依赖给定材料。对照臂要用同一判据。
- **外部 API**：OpenAI、Anthropic（评委）；Ai2 Paper Finder（`mabool_actor_id`，是否对外开放**未核实**）或 S2（检索）。

### 4.2 NovGauge
- 数据：HF `ZitaGo/NovGauge`，CC-BY-4.0。代码：`Zita-Go/NovGauge`，MIT。
- 任务**不是给新颖性打分**，而是判断两篇论文在 task / problem / method 三个维度上是否相似，属于成对二分类。规模：positives 463、negatives 156，另有 50 个多论文分组。
  - 字段：`pair_id, source(iclr/survey), paper_a, paper_b, labels{task,problem,method: 1/0/null}, source_survey`。
- **论文记录**：`paper_id(ngp-000001), title, authors, year, venue, arxiv_id, doi, forum_id, s2_paper_id, source_url`。
  - 926 条记录中 431 条有 arxiv_id，236 条有 forum_id；日期只到年份，303 条年份为空。
  - 不分发正文和摘要，评测前需要自己补 `markdown_path` 或 `abstract`。
- 评测：`bash bench/run_eval.sh`、`bash bench/run_grouping_eval.sh`。`CONTENT_MODE` 可选 abstract / abstract+intro / full；指标是 Acc / P / R / F1，以及 Verified F1。
- 原生支持 OpenAI 兼容端点（`.env` 里设 `NOVGAUGE_BASE_URL / NOVGAUGE_API_KEY / EVAL_MODEL / JUDGE_MODEL`）。
- **没有检索槽**。接入方式是把我们的"每篇工作被如何描述 / 方法族"作为论文内容表示，替换 `run_bench.py` 里组装论文内容的那一步（具体函数**未细读**）。

---

## 5. LimitGen（arXiv 2507.02694，ACL 2025）

代码：GitHub `yale-nlp/LimitGen`。数据：HF `yale-nlp/LimitGen`。HF 上的 `LimitGen_data.py` 缺 `import datasets`，建议直接下载文件。核查来源：子代理。

### 5.1 数据
- **LimitGen-Syn**：500 篇 arXiv cs.CL（2024-03 到 05），1,000 例，11 个子类型。文件在 `syn/annotated/<type>/*.json`，金标在 `syn/sections/<type>.json` 的 `ground_truth` 字段。
- **LimitGen-Human**：1,000 篇 ICLR 2025 投稿（从 9,844 篇里抽样）。
  - 全文：`human/paper/<id>.jsonl`，共 1,000 个文件。每行 `{"page","text","type"}`，type 有 Title / Section / Paragraph / Header 等；PDF 解析结果，带行号噪声和 "Under review…" 页眉。
  - 金标：`human/classified_limitations.json`，结构 `{"<id>": {"title","abstract","limitations":{"methodology":[…],"experimental design":[…],"result analysis":[…],"literature review":[…]}}}`。共 6,046 条，每篇平均 6.05 条。
  - ID 形如 `x1SfON9HvT`，按格式推断是 OpenReview forum id，**未去 OpenReview 核对**。没有 arXiv id。是否只含被接收的论文**未核实**。

### 5.2 原 RAG 设置（`retrieval/`）
1. `query_gen.py`：GPT-4o 从摘要生成 5 词 TLDR。
2. `search.py::get_paper(month, query)`：S2 `/graph/v1/paper/search`，limit=3。
3. `recommendation.py::get_paper(s2_id, limit)`：S2 Recommendations API，每个种子取 5 篇 open access。
4. `rerank.py`：用 gpt-4o 选 top 5（论文写的是 gpt-4o-mini，与代码不一致）。
5. 手动下载 PDF，可选 MMDA 预处理。
6. `section_locate.py`（gpt-4o-mini）和 `rewrite.py`（gpt-4o）产出 `<paperId>/<aspect>_final.txt`。

**没有时间截止**：Human 路径里 `search.py` 把 `month` 写死为 -1，不带 `publicationDateOrYear`；`is_paper_earlier()` 被注释掉了。

### 5.3 评测
- 生成：`identification/main_human.py`。输出 `limitations/{retrieval|non-retrieval}/{model}/generated_limitation.json`，结构 `{doc_id: {"limitation": {"experiment":[3条],"methodology":…,"result":…,"literature":…}}}`。
- 评分（`evaluation/human/`）：
  1. `measure_overlap.py`：同一方面内，生成项与金标两两配对，送 GPT-4o 判断（`prompts/overlap.txt`），得到 relatedness（none / weak / medium / high）和 specificity；
  2. `match_calculate.py::measure_overlap_for_all_papers`：medium / high 算匹配，逐方面计算 recall / precision / pseudo-Jaccard；
  3. `rating.py`：GPT-4o 打 1–5 分，未匹配记 0。
- **代码坑**：
  - `main_human.py` 读 `paper_dir/<doc_id>/sections.jsonl`，与 HF 的文件布局对不上；
  - `elif "result":` 恒为真，导致 literature 方面用的是 result 的 prompt；
  - RAG 模式下 `paper_content` 会跨方面累加；
  - `rating.py` 写死 `paper_cnt = 10`；
  - `--retrieval` 参数用了 `type=bool`，传任何字符串都为真。

### 5.4 接入与依赖
- **接入点**：`identification/main_human.py::paper_retrieval(doc_id, retrieval_path, retrieval_dir, aspect)`。它返回一段字符串，拼在论文正文前面，换成我们的 `as_of` 工具输出最直接。我们的系统特别适合"literature review"和"公认局限"这两个方面。
- `as_of` 建议取 ICLR 2025 投稿截止日（约 2024-10 初，**具体日期未核实**）。原设置不限日期，所以加截止后是更严的条件。
- **外部 API**：OpenAI（生成与 judge 都是 GPT-4o）；S2 search 与 Recommendations（不用 key，但遇到非 200 会无限重试）。

---

## 6. IdeaForecastBench（arXiv 2609.00747，EMNLP 2026）

数据：HF `4R5T/idea-forecast-bench`。代码：GitHub `social-world-model/idea-forecast-bench`，MIT。数据许可是 `license: other`（每篇论文沿用其 arXiv 许可，仅供研究使用）。

### 6.1 文件与字段
- 34 个月度 parquet（`2023-01.parquet` … `2025-10.parquet`，单文件 38–97 MB），外加 `manifest.json`（每月篇数、字节数、sha256，合计 108,768 篇）。
- **只有 4 列**：`arxiv_id`（如 `2301.00004`，长度恒为 10，即新式 arXiv id，不带版本号）、`month`（`YYYY-MM`）、`title`、`text`（MinerU 转换的全文 markdown）。
  - `text` 长度中位数约 6.5 万字符，最长 194 万字符（datasets-server 统计的是前 69,032 行的部分样本）。
  - **没有日期、类目、作者、引文字段**。
  - 抽样 40 篇，`month` 都等于 arXiv id 的 YYMM 前缀。
- 样本（截短）：
```
arxiv_id: 2301.00004   month: 2023-01
title: SESNet: sequence-structure feature-integrated deep learning method for data-efficient protein engineering
text: "# SESNet: sequence-structure feature-integrated ... \n\nMingchen $\mathrm{Li^{1,4^{\dag}}}$ , Liqi Kang1,2†, ..."
```
- 加载方式：`idea-forecast-bench fetch --from-hf 4R5T/idea-forecast-bench` 把每篇写成 `<YYYY-MM>/<arxiv_id>.md`（`examples/benchmark/fetch.py::download_from_hf`）。之后 `papers.py::parse_markdown_paper` 解析成 `PaperRecord`：
  - `paper_id`：文件名，即 arXiv id；
  - `title`：第一个 `# ` 标题；
  - `summary`：Abstract 一节或前言，最多 1500 字符；
  - `keywords`：从标题抽取；
  - `references`：`_extract_bibliography` 只认 `^#{1,6} References$` 这一种标题，按编号或空行切条目，结果是 `[{"text": ...}]`；
  - `citations`：恒为 `[]`。

### 6.2 日期、episode 与 topic
- **日期按月**。markdown 里没有 `date:` 元数据（40/40 篇都没有），于是 `_extract_published_date` 退回文件名前 4 位 YYMM，再取 `month_end_date`。**每篇论文的日期等于其 arXiv id 所在月的月末**。
- **截止**：`backtest.py::split_train_future_by_cutoff`。
  - cutoff_month 解析为该月 1 号；`train = date ≤ cutoff`，等价于截止月之前的所有月份；
  - `future = cutoff < date ≤ (cutoff_month + horizon) 的月末`；
  - docstring 写明：horizon=3 时 future 实际覆盖 4 个日历月，包含截止月本身。
- **论文 sweep 参数**（`scripts/benchmark/benchmark.sh`）：`START_MONTH=2024-04, END_MONTH=2025-09, MIN_CUTOFF_MONTH=2024-07, HORIZON_MONTHS=3, TOP_K=5, MIN_TRAIN_PAPERS=2`，得到 12 个截止（2024-07 到 2025-06）。
- **topic**：`config/topics_v2.yaml` 定义 52 个 topic，每个有 `id / name / aliases / keywords`，例如 `llm_alignment_rlhf` 的关键词有 rlhf、dpo、reward model 等。
  - 归类方法：`topics.py::classify_paper_topics` 对 title + summary + keywords 做**关键词子串匹配**，一篇可以属于多个 topic。
  - 一个 episode = (topic, cutoff)，共 52 × 12 = 624 个。
- **策略能看到的历史**只有"本 topic 内、日期 ≥ START_MONTH(2024-04)、≤ cutoff"的论文。因为语料只按 2024-04 到 2025-09 加载，第一个截止只有 3 个月的历史，最后一个约 14 个月。各基线还会再截断：summary 取最近 60 篇的 `summary[:300]`，predictor 取最近 20–40 篇，retrieval 取 top 20。

### 6.3 "历史压缩策略"接口（第 6 种策略的接入点）
```python
# idea_forecast_bench/strategy/base.py
class IdeaStrategy(ABC):
    name = "base"
    @abstractmethod
    def generate(self, train_papers: list[PaperRecord], cutoff_month: str, top_k: int) -> list[IdeaPrediction]: ...

# idea_forecast_bench/models.py
@dataclass
class IdeaPrediction:
    rank: int; title: str; rationale: str; approach: str = ""; score: float = 0.0
    confidence: float | None = None; key_terms: list[str] = field(default_factory=list); metadata: dict = ...
```
接入步骤：
1. 新建 `strategy/field_state.py`，写一个 `IdeaStrategy` 子类。在 `generate()` 里调我们的工具，`as_of` 取 `cutoff_month` 前一个月的月末，与 `train` 口径一致。再仿照 `summary_prompting.py` 的 `_build_forecast_prompt` 和 `_parse_predictions`，让模型输出 `{"ideas":[{"title","rationale","approach","confidence","key_terms"}]}`。
2. 在 `strategy/registry.py::create_strategy()` 里加一个分支。
3. 把策略名加进 `examples/benchmark/baselines.py` 的策略列表，以及 `scripts/run_benchmark.sh` 的 `STRATEGIES`。
4. **注意**：`generate()` 拿不到 topic id，只拿到已按 topic 筛好的 `train_papers`。如果我们的工具需要 topic，有两个办法：要么改 `examples/benchmark/benchmark.py::_process_topic`，在调用 `backtest(...)` 之前把 `topic` 挂到 strategy 上；要么从 `train_papers` 反推 topic。
5. **公平性**：基线只看本 topic 内 2024-04 以后的论文。我们的工具如果使用更早或 topic 外的历史，就是信息条件不同的一臂，必须在报告里单列，或另做一臂"只用 `train_papers` 构建的地图"。

### 6.4 markdown 能否切出引用句（40 篇抽样实测）
抽样方法：用 datasets-server 在 13 个 offset 各取 3 行，再加 row 0，覆盖 2023-01 到 2025-09。

- **参考文献列表基本保留**：用宽松正则识别（`References/REFERENCES/Reference/Bibliography`，允许没有 `#` 或带编号，如 `7. REFERENCES`），39/40 篇能找到。
  - 只有 32/40 能被官方 `_extract_bibliography` 的严格标题正则命中。
  - 参考块的位置在全文 27%–96% 之间，有附录排在后面的情况，不能简单取"文末"。
- **正文引用标记保留**：数字式 21 篇，作者-年份式 19 篇，没见到上标式。
  - 数字式：1518 个标记中 1352 个（89%）的编号能在参考块里找到对应的 `[n]` 或 `n.` 条目。
  - 作者-年份式：1245 个标记中 1233 个（99%）的"姓 + 年"能在参考块里找到。这只是粗检查，没有做到条目级的精确解析。
- **原文片段**（数字式，2504.12045）：
  > Artificial intelligence and reinforcement learning (RL) have proven to excel in complex games, such as Chess [30], Go [29], Starcraft [37], and Minecraft [21]. Apart from board and video games, computer vision models have recently started playing an important role in sports with several applications in generating sports analytics [13] and analyzing game strategies and tactics [36,24].

  对应参考块：`References\n\n1. Alaniz, S.: Deep reinforcement learning with model learning and monte carlo tree search in minecraft (2018)\n2. ...\n3. Brockman, G., Cheung, V., ... Zaremba, W.: Openai gym (2016)`

  作者-年份式（2508.08279）：
  > Some methods include auxiliary features like weather or location (Han et al. 2021; Kim et al. 2024), but typically treat spatial context static...

  （2506.04788）：
  > Prior studies: Existing surveys provide valuable insights into multimodal LLMs. Shukang Yin et al. (2023) conducted a comprehensive review covering various aspects of multimodal LLMs ...
- **失败模式**（切句器要处理）：
  1. alpha 标签式引用 `[AZLS19]`，见 2401.16613，数字正则全部漏掉；
  2. MinerU 丢失或损坏方括号，例如 `ar23] Scott Aaronson`、`$[\mathrm{ASR^{+}}24]$`；
  3. LaTeX 下标被误识别为数字标记，例如 `w_{2}[0]`；
  4. 参考块标题不规范，例如 `References Bi, J.; Wang, Z.; ...` 与条目连在同一行；
  5. 条目切分依赖编号或空行，作者-年份式的条目之间有时没有空行。
- **条目 → arXiv id**：39 篇里有 29 篇的参考块中至少出现一个显式 arXiv 号（`arXiv:XXXX.XXXXX` 或 arxiv.org/abs），12 篇出现 DOI，但都只覆盖部分条目。**主要还得靠标题匹配**，可以匹配到本语料的 108,768 篇，或我们自己的 arXiv 元数据库。
- 判断：作为"他述"来源可用，标记和条目都在。切句器需要支持两种引用风格，还要处理上面列出的噪声；条目解析建议用 LLM 或 GROBID 级的工具，不建议沿用官方 `_extract_bibliography`。样本只有 40 篇，比例是粗估。

### 6.5 评测与外部依赖
- 生成：`idea-forecast-bench benchmark --strategy <name> --model-name <m> --skip-matching --output output/backtest/<name>.json`。
- 判分：`idea-forecast-bench judge-eval --input-json ... --papers-dir ... --output ...judged.json`，然后 `main-table` 汇总。
- judge 流程（`judge/topics.py::process_topic` 加 `judge/windows.py::process_window`）：
  1. 用 voyage-3-large 嵌入 future 论文（`paper_text[:4000]`）；
  2. 每条预测召回 `top_r=10` 篇；
  3. `gpt-4.1-mini`（`DEFAULT_JUDGE`）按 problem / method / specificity 打分，judge 看到的是论文摘要的前 800 字符；
  4. 指标：`hit_at_k`、`mrr`、`precision_at_k`、`soft_score`、`cluster_coverage`、`avg_novelty`。
- 外部 API：
  - `VOYAGE_API_KEY`：embedding，用于匹配和 judge 召回；
  - `OPENAI_API_KEY`：生成，以及默认 judge。
- 换成本地端点：生成用 `OPENAI_BASE_URL`（注意：只有模型名以 `gpt-4o` / `gpt-4.1` / `gpt-5` 开头时才走这个地址，本地模型要起这类别名），embedding 用 `VOYAGE_BASE_URL` / `EMBED_BASE_URL`，judge 用 `--judge-base-url` 或 `JUDGE_BASE_URL`。换了 judge 后的数字和论文不可比。
- 全量语料约 2.1 GB。

---

## 7. TaxoBench（arXiv 2601.12369）

数据：HF `konglongge/TaxoBench`，标注 CC BY-NC 4.0。评分代码：GitHub `KongLongGeFDU/TaxoBench`，Apache-2.0。核查来源：子代理（data.jsonl 已下载解析，并跑了 toy 冒烟）。

### 7.1 数据
- `dataset/data.jsonl`：72 行。字段 `id(int), survey(str), survey_topic(str，与 survey 相同), gt_paper_count(int), pdfs(list[{title, abs}]), gt(dict)`。
- `gt` 是树：内部节点 `{name, subtopics}`，叶子 `{name, papers: [标题字符串]}`，深度 3–7。3,815 个叶子标题都与 `pdfs` 里的 title 精确一致。
- **整个文件没有任何 ID、年份或 arXiv 号**。综述也只用标题标识，而且冒号被替换成了 "-"。
```json
{"name":"<综述标题>","subtopics":[{"name":"Large Models for Time Series…","subtopics":[…{"name":"Action Recognition","papers":["Language Knowledge-Assisted Representation Learning for Skeleton-Based Action Recognition"]}]}]}
```

### 7.2 Bottom-Up 模式
- 输入：`dataset/prompts_title_abstract.jsonl`，字段 `id, gt_paper_count, survey_topic, input_content`。`input_content` 是一整段字符串（id 0 约 8.7 万字符），格式为：
  `"SYSTEM PROMPT:\n…USER PROMPT:\nPerform a bottom-up hierarchical clustering of the following 57 papers…Survey Topic: …\nPaper List:\nPaper 1:\n  Title: …\n  Abstract: …"`
- 要求的输出：包在 ```json 代码块里的单根树；只有叶子带 `"papers"`；每篇论文恰好出现一次。
- 预测文件（`dataset/SCHEMA.md`）：每行 `{"id": 0, "hierarchy_tree": {"name":…, "subtopics":[…]}, "retrieved_papers": [标题…]}`。`retrieved_papers` 可选；键名也接受 `tree`。

### 7.3 评分 CLI
```bash
taxobench-score --data dataset/data.jsonl --predictions your_predictions.jsonl --output scores.jsonl [--threshold 0.92]
```
- 入口：`taxobench.score:main`（定义在 `pyproject.toml`）。
- 输出：`n_scored, leaf_ari, leaf_v_measure, leaf_homogeneity, leaf_completeness, sem_path, retrieval_recall, retrieval_precision, retrieval_f1`。预测里缺失的 id 直接跳过，不按 0 分计。
- 标题对齐：`metrics/alignment.py::align_titles(reference_titles, prediction_titles, threshold=0.92)`，贪心一对一匹配。`title_similarity` 先归一化，相等或互为子串记 1.0，否则用 `difflib.SequenceMatcher` 的 ratio。
- 聚类指标：未对齐的参考论文统一归入一个"缺失"簇。
- Sem-Path：`sem_path_score(..., similarity=None)`，默认用字符串相似度，不调 embedding 或 API。论文里用的是 embedding，但 CLI 没暴露这个参数。
- 依赖只有 numpy 和 scikit-learn。

### 7.4 接入
- 只评地图：把每题的 `pdfs` 标题对齐到我们的库，用我们的方法族划分生成树，写成预测文件。
- 也可以做输入表示：把我们对这批论文的"他述"与方法族作为 prompt 补充，看 LLM 建树是否变好。
- `as_of` 没有现成定义，**综述的发表年份未核实**（数据里没有）。要做时间切片，得先按标题查到综述日期，并把截止设在综述发表之前。
- 风险：综述高被引，taxonomy 很可能已被模型记住。

---

## 8. 汇总表

| 基准 | 论文 ID 体系 | 截止粒度 | 候选语料规模 | 是否需要全文 | 接入方式 | 需要的外部 API |
|---|---|---|---|---|---|---|
| ScholarCatalyst | 语料 id：`arxiv_YYMM.NNNNN` 190,543 条，`oa_W…` 182 条，`s2_<sha>` 171 条；源论文全是 arxiv。**arXiv 直接对齐**，金标 91–93% 是 arxiv | **按月**（`temporal_filter`，同月放行）；源论文日期按天 | 190,896 篇（标题+摘要），894 题 | 官方不需要，也没提供（`citations.jsonl`、`corpus_full_text.jsonl` 未发布）；我们的"他述"要自带全文 | 检索工具：新 agent 模块注册进 `agentic_search.py::AGENTS`，或在 `toolcall.py` 的 `schemas/handlers` 加工具；也可直接写 `runs/agentic/<p>/<qt>.jsonl` | LLM（OpenAI / Anthropic / Gemini 三选一；兼容端点要改 `llm_rerank.OPENAI_BASE`）；BM25 和 dense 本地跑 |
| PreScience（prior work） | S2 corpusId 是主键，**每行带 `arxiv_id`，可直接对齐** | **按天**（`date < cutoff` 字符串比较） | test 464,942 行（target 52,836），train 373,716 行 | 不需要（只有标题+摘要） | 新 baseline：`predict_references(author_ids, cutoff_date, …)`，复用 `create_evaluation_instances`；输出必须是表内 corpus_id | prior work 任务不需要；LLM 基线和 LACER 需要 OpenAI / Anthropic |
| MasterSet（Type I） | 内部 UUID，只有 title / venue / year，**无 arXiv / S2 id，需标题对齐** | **按年**（池 2018–2024，题目 2025），无逐题约束 | 池 67,761 篇，题目 7,028 篇（6,105 题有 Type I 金标） | 不需要（输入是标题+摘要） | 输出池内 paper_id 排序，复用 `extract_relevant_sets` / `compute_all_metrics` | 不需要 |
| Old Ideas novelty-judge-bench | 只有 title（ICLR 2026 投稿），**需标题对齐** | **按天**，全局固定 2025-03-01 | 154 对 / 约 300 条逐点；检索走外部 Paper Finder / S2 | 不需要（idea 是摘要级文本） | 输入表示：自写 `retrieval_cache.json` 的 `related_work` 字段 | OpenAI / Anthropic 评委（走 Responses API）；原检索用 Ai2 Paper Finder 或 S2 |
| NovGauge | `arxiv_id`（431/926）、`forum_id`、`doi`、`s2_paper_id`，**部分可对齐** | **按年**（只有 year），任务本身无截止 | 619 对 + 50 组，926 篇论文 | 可选（abstract / abstract+intro / full，需自己补） | 输入表示：替换 `run_bench.py` 组装论文内容的那一步 | 任意 OpenAI 兼容端点 |
| LimitGen-Human | 疑似 OpenReview forum id（未核实），**无 arXiv id** | 原设置**无截止**；建议 `as_of` 取 ICLR 2025 截稿前 | 1,000 篇，6,046 条金标局限 | **需要**（PDF 解析全文已提供） | 输入表示：替换 `main_human.py::paper_retrieval()` 的返回字符串 | OpenAI（生成和 GPT-4o judge）；原 RAG 用 S2 search 与 Recommendations |
| IdeaForecastBench | **全是 arXiv id**（`arxiv_id` 列，新式 10 位） | **按月**（日期从 arXiv id 推成月末，截止为月初） | 108,768 篇全文 markdown；benchmark 用 2024-04 到 2025-09；52 topic × 12 截止 = 624 episodes | 语料自带全文，引用标记和参考块都在；我们可以从中切"他述" | 第 6 种策略：`IdeaStrategy.generate(train_papers, cutoff_month, top_k)` 子类，注册进 `strategy/registry.py::create_strategy` | Voyage embedding 和 OpenAI（生成 + gpt-4.1-mini judge），都可换本地端点但不可比 |
| TaxoBench（Bottom-Up） | 只有标题，**无 ID、无年份，需标题对齐** | **无截止**（综述年份未核实） | 72 篇综述，3,815 篇论文 | 不需要（输入是标题+摘要） | 只评地图：写 `{"id","hierarchy_tree"}` 预测文件，跑 `taxobench-score` | 不需要 |

## 9. 未核实清单
- ScholarCatalyst：`citations.jsonl` 和全文是否会另行发布（HF 文件树里没有）。
- PreScience：作者论文列表是否按日期排序，这关系到 `bisect_left` 的正确性（HF filter 接口一直在 loading）；`refine_key_references_and_publication_histories_to_ensure_membership` 是否在发布管线中调用过。
- AgentExpt：数据以后是否会发布。
- Old Ideas：Ai2 Paper Finder 是否对外开放（需要 `mabool_actor_id`）。
- NovGauge：组装论文内容的具体函数。
- LimitGen：id 是否就是 OpenReview forum id；Human 子集是否只含被接收论文；ICLR 2025 截稿的具体日期。
- IdeaForecastBench：引用解析比例来自 40 篇抽样和启发式检查，不是条目级的精确解析；topic 归属关键词子串匹配的准确率。
- TaxoBench：72 篇综述的发表日期。
