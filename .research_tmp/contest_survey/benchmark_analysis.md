# 华为赛题三：论文搜索评测数据集形态调研

调研日期：2026-08-26。对象：RealScholarQuery / AutoScholarQuery(SPAR) / PaperFindingBench / LitSearch。
核心目的：搞清"gold 集大小 + F1 口径"，判定召回策略是精找几篇还是广撒网。

---

## 1. RealScholarQuery（PaSa 论文评测集）

来源：PaSa: An LLM Agent for Comprehensive Academic Paper Search (ByteDance Seed, ACL 2025, arXiv:2501.10120)。数据在 HuggingFace `CarlanLark/pasa-dataset`（gated），代码 github.com/bytedance/pasa。

### a) 查询形态
真实 AI 研究者提出的细粒度查询，全部 query date = 2024-10-01。论文附录 Table 12 唯一 verbatim 例子：

> "Give me papers about how to rank search results by the use of LLM"（65 字符，该查询 gold 有 39 篇）

被过滤掉的过宽查询（不在数据集内）："multi-modal large language models"、"video generation"——说明查询普遍带方法论/场景约束，不是宽泛主题词。

### b) gold 集大小与构造
- **50 个查询**；**平均 15.82 篇 gold/查询**（单例最高 39 篇）。
- 构造：先人工收集 + 汇总 PaSa/Google/Google Scholar/ChatGPT/G+GPT-4o 各系统检索结果作为候选池，再由专业标注者（中国某顶尖大学 CS 教授）逐篇审核。**平均每个查询需审核 76 篇候选**。成本极高，所以只做了 50 条。

### c) 评分口径（读了 repo 里 metrics.py 源码）
- gold 按论文标题归一化匹配（`keep_letters`：去非字母+小写）。
- **precision / recall 都是集合级**：对 agent 最终选中输出集合（select_score>0.5）算 `TP=|pred∩gold|, FP=|pred−gold|`，**gold 之外的论文一律算 FP，不容许"合理但不在 gold 里"** → precision 天花板完全取决于 gold 完备性。好在 RealScholarQuery 标注是"as comprehensive as possible"，gold 相对全。
- recall@20/50/100 只用于 Google 类 baseline（对 crawler 抓到的论文按 selector 分数排序取 top-k）。
- **论文不报 F1**，主指标是 recall + precision 两个独立数。宏平均（逐查询求比率再平均）。
- PaSa-7B vs 最强 baseline：recall 超 PaSa-GPT-4o 30.36%、precision 超 4.25%；vs Google+GPT-4o recall@20 超 37.78%。

### d) 领域
纯 AI/ML（论文自述局限："primarily focused on the field of machine learning"）。

### e) 评测时系统可访问什么
**开放式 live 检索**：Google Search API（serper.dev）+ arXiv/ar5iv API。repo 里的 paper_database（cs_paper_2nd.zip）只用于 PPO 训练，评测不限定语料库——原则上可以预先爬建自己的库（gold 只按标题匹配，不查来源）。

---

## 2. AutoScholarQuery（PaSa 构造，SPAR 用其 test split 评测）

来源：同 PaSa；SPAR (BAAI, arXiv:2507.15245) 在其上做 set-F1 对比。SPAR repo（github.com/xiaofengShi/SPAR）的 `benchmark/AutoScholarQuery_test.jsonl` 就是 PaSa 的 test split，已下载到本地分析（`.research_tmp/contest_survey/autoscholar_test.jsonl`）。

### a) 查询形态
GPT-4o 从论文 Related Work 段落反向生成的学术查询（源论文：ICLR/ICML/NeurIPS 2023、ACL 2024、CVPR 2024）。Verbatim 例子（论文 Table 1 + test 文件实测）：

> "Could you provide me some studies that proposed hierarchical neural models to capture spatiotemporal features in sign videos?"
> "Which studies have been conducted in long-form text generation, specifically in story generation?"
> "Are there any studies that analysed the use of target networks for Deep Q-learning?"（test.jsonl 第 2 条，gold=1 篇）
> "Any resources providing information about attempts to detect or calibrate biases automatically in peer reviews?"（test.jsonl 第 3 条，gold=4 篇）

### b) gold 集大小与构造
- 33.5k train / 1000 dev / **1000 test**。
- gold = 该 Related Work 段落实际引用的文献（限定 arXiv 可得的，arxiv_id 标识），查询日期=源论文发表日，只考虑之前的论文。
- **test 集 gold 实测：均值 2.40 篇/查询，中位数 2，最大 16，最小 1**（论文按会议报均值 2.16-2.94，p50=2，p90=4-6）。

### c) 评分口径
- PaSa 原口径：同上，集合级 P/R + recall@k，不算 F1。
- **SPAR 口径（关键）**：文档级集合 F1（TP/FP/FN 对 gold 集合），且"每个检索系统用自己原生搜索源、无共享语料库"的端到端对比。**gold 外论文一律 FP**。
- 后果实证（SPAR 论文 Table 1，AutoScholar）：
  - PaSa：recall 0.7931 / precision 0.1448 / **F1 0.2449**
  - SPAR：recall 0.4105 / precision 0.3612 / **F1 0.3843**（+56.92%）
  - **高召回广撒网在 set-F1 下被 precision 拖死**。gold 只有 ~2.4 篇且=引用列表，多返回任何一篇"合理相关"论文都是 FP。

### d) 领域
AI/ML 五大顶会（含 CVPR→CV 方向）。

### e) 评测时系统可访问什么
开放式：SPAR 源集合 = {Google, ArXiv, OpenAlex, Semantic Scholar, PubMed}，各 baseline 用自己的源。可预爬（匹配只看标题）。

### 附：SPARBench（SPAR 自己的新 benchmark，值得参考）
50 查询（35 CS + 15 生物医学），~560 篇 gold（**均值 ~12 篇/查询**）。查询=真实场景种子+GPT-4o 扩写（故意带 multi-intent、语法不全、拼写错误），gold 三阶段构造：多源检索 ~198K 候选 → Qwen2.5-7B 相关性过滤到 ~3K → Qwen2.5-72B 过滤到 ~2K → 专家人工审定 ~560。仍是集合 F1。Verbatim 例子：
> "Provide me with some top-tier journal papers to expand my ideas on using synthetic data to augment supervised fine-tuning (SFT) while ensuring data quality and diversity, maintaining a balance between the two."（gold 10 篇）
> "How can deep learning enhance the perception and decision-making accuracy of autonomous driving systems? Please provide a comprehensive analysis with supporting research papers."（gold 13 篇）

---

## 3. PaperFindingBench（Ai2 asta-bench）

来源：allenai/asta-bench（AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite, ICLR 2026）。数据在 HF `allenai/asta-bench`（gated，需 token），评分代码 `astabench/evals/paper_finder/`（已读源码）。

### a) 查询形态
test 集 333 查询，三类混合、不标注类型（要求 agent 自己路由）：
- **semantic 242 条**：按内容描述找一批论文。Verbatim（README 实例，query_id 29）：
  > "visual question answering papers using Earth Mover's Distance (EMD) as an evaluation metric"（known_to_be_good 3 篇）
- **navigational 48 条**：找一篇用户已知的论文，如 "the alpha-geometry paper"。
- **metadata 43 条**：按元数据条件定义集合，如 "acl 2024 papers that cite the transformers paper"。

### b) gold 集大小与构造
- navigational：1 篇。
- metadata：**内部关联 Python 代码调 API 精确算出 gold 集合**（完备）。
- semantic：**完备 gold 不存在**。每查询带 (i) 小规模 known_to_be_good（例中 3 篇）+ (ii) LLM 自动生成、1/5 人工校验过的加权相关性判据（relevance_prompt，如 "must include VQA or question answering in the visual domain. also must include usage of EMD metric."）；分母用"估计集合大小"= 宽松阈值跑多轮内部 PaperFinder 取并集 × 2-10 倍系数。

### c) 评分口径（最精细的一家）
- 输出：**最多 250 篇、按相关性排序**，每篇给 S2 CorpusID + 从原文 verbatim 摘的 markdown_evidence（评的是 evidence 不是全文，防长上下文）。
- navigational/metadata：**标准集合 F1** vs 已知 gold。
- semantic：LLM judge 逐篇判相关性（对 evidence）→ **estimated-recall@estimated-k + nDCG，二者调和平均 = adjusted F1**。**这里 gold 之外的合理论文可以得满分（LLM 判的是"与查询相关"而非"在 gold 里"）**——是四个数据集中唯一明确容许 gold 外合理论文进 precision 的。
- 总分 = 全部查询平均（adjusted_f1_micro_avg），另按 semantic/metadata/specific 分组报。

### d) 领域
AI/CS 为主（Ai2 生态，查询涉及 VQA/ACL/transformers 等）。

### e) 评测时系统可访问什么
**限定 Asta MCP 工具**（Semantic Scholar 检索/读论文），带**日期截止**（insertion date = 数据集发布月前一个月，防新论文污染）；可选原生 web search（OpenAI/Anthropic/Google/Perplexity）；自定义工具允许但归"Custom Interface"类别、可能得分偏低。**不允许预先自建任意库**（与 PaSa/SPAR 的开放 web 相反）。

---

## 4. LitSearch（Princeton NLP, EMNLP 2024, arXiv:2407.18940）

### a) 查询形态
597 条。两源：inline-citation（GPT-4 把含引用的段落改写成搜索问题，gold=被引论文）+ author-written（ACL 2023/ICLR 2024 作者为自己的论文写问题）。Verbatim 例子（摘要 + datasets-server 首行）：
> "Where can I find research on the evaluation of consistency in generated summaries?"
> 询问是否有论文用 task-agnostic knowledge distillation 压缩大模型（gold 1 篇）
> 询问 post-hoc techniques for hallucination detection at token and sentence level（gold 2 篇）
分 broad/specific 两档（specificity 字段），带 quality 标注。

### b) gold 集大小与构造
**约 1 篇/查询**：inline 1.21(broad)/1.07(specific)，author 1.03/1.00。corpusids 字段通常 1-2 个。gold=被引论文或作者本人论文，全部专家人工校验过问题质量。

### c) 评分口径
**recall@5（specific）/ recall@20（broad）**，无 precision、无 F1。标注 rubric 承认 broad 问题可能有 ~20 篇合相关、specific ~5 篇，但实测 gold 只有 1-2 篇 → 返回额外相关论文不扣分也不加分。是**固定语料检索 benchmark**（BM25/dense/rerank），不是 agent benchmark。最佳 dense 检索 + LLM 重排也只到 recall@5 ~40-50% 区间；Google 比最佳 dense 检索低最多 32 个 recall 点。

### d) 领域
NLP（ACL Anthology）+ ML（ICLR）。

### e) 评测时系统可访问什么
**封闭固定语料**：S2ORC 抽取的 64,183 篇（59,383 ACL + 4,807 ICLR），提供 title+abstract 和全文两种格式，自己建索引（本地完全自由）。系统不接触外部 API。

---

## 横向对比与对赛题设计的含义

| 数据集 | 查询数 | gold/查询 | gold 构造 | 口径 | gold 外论文 | 访问方式 |
|---|---|---|---|---|---|---|
| RealScholarQuery | 50 | **均值 15.8**（最高 39） | 多系统候选池+教授级标注（均审 76 篇/查询） | 集合 P/R + recall@k，无 F1 | 一律 FP | 开放 web（Google+arXiv） |
| AutoScholarQuery test | 1000 | **均值 2.4**（p50=2） | Related Work 引用列表 | 集合 P/R（PaSa）/ **集合 F1（SPAR）** | 一律 FP | 开放 web/API |
| SPARBench | 50 | 均值 ~12 | 多源检索 198K→Qwen 两级过滤→专家审 | 集合 F1 | 一律 FP | 开放 web/API |
| PaperFindingBench | 333（test） | nav 1 / meta 精确集 / semantic 估计集 | nav 已知 / meta API 算 / semantic LLM 判据 | **nav+meta 集合 F1；semantic estimated-recall+nDCG 调和平均** | **semantic 容许（LLM 判相关性）** | 限定 Asta MCP（S2）+日期截止 |
| LitSearch | 597 | **约 1** | 被引论文/作者自指 | recall@5/20，无 F1 | 不奖不罚 | 封闭 64K 语料自建索引 |

**对系统设计的判定**：
1. **口径分裂成两个世界**：小 gold（1-2.4 篇，LitSearch/AutoScholar）+ 集合 F1 ⇒ 精找几篇、宁缺毋滥，PaSa 的 recall 0.79/F1 0.24 是反面教材；大 gold（12-16 篇，RealScholarQuery/SPARBench）⇒ 必须广撒网、多轮扩展引文。**华为赛题 F1 70% 的 gold 口径未公布，但赛题文本强调"综合排序/区分高度与部分相关"，更像大 gold 世界**——需按"查询分解→广召回→严格精排"两段式设计，且输出数量应自适应查询类型（navigational 类返回 1 篇，semantic 类返回一批）。
2. **排序普遍重要**：PFB 用 nDCG+顺序敏感、SPAR/PaSa 集合无序但 top-k recall 隐含排序——重排层（我们的超图/schema 关联推理+evolution 边）是明确得分位。
3. **可溯源证据是趋势**：PFB 强制 markdown_evidence verbatim 片段，直接契合我们超图边带 evidence/cited_from 的设计（赛题"结构化 10%"同理）。
4. **访问模式**：PaSa/SPAR 系开放 web（可预爬），PFB 限定 API+日期截止，LitSearch 封闭语料。赛题明说对接 Semantic Scholar/OpenAlex/PubMed API ⇒ 开放 API 世界，可预先建自己的增强索引（PaSa 先例合法）。
5. **AutoScholarQuery 是现成的训练/评测资产**（35k 训练对，gold=Related Work 引用），与我们的颗粒流语料构造思路同构，可直接用于训练查询分解器。

证据文件（本地）：`.research_tmp/contest_survey/autoscholar_test.jsonl`（SPAR 用的 AutoScholar test 全量）、`spar_readme.md`、`spar_*.py`（SPAR 源码）。
