# CDP 发现层补充：跨论文聚合层(2026-10-04)

## 0. 方法与覆盖范围(先读)

- 发现层：用户真实 Chrome(CDP Proxy)，搜索入口为 DuckDuckGo(google.com.hk 的结果链接被包装，Bing 对长尾学术查询返回垃圾，按 site-patterns 经验直接走 DDG)。约 25 轮查询，覆盖：编译式科学知识层、证据独立性/依赖证据、共识/争议、方法归属(MPR/PwC)、产品页、中文社区。
- 验证层：arXiv abs 页 citation_* meta、Crossref、OpenAlex 摘要、GitHub 原始文件、产品官方帮助页。表中"存在"均已落到一手页面。
- 覆盖的局限(如实)：
  1. 本轮是快速重跑，查询数有限，不是穷尽检索。结论"未发现撞车"的强度是"25 轮 CDP 查询加对命中项核验"，不是"不存在"。
  2. 中文社区：知乎未登录是登录墙(site-patterns 已记)，只做了一次 DDG 查询，命中 arxivdaily 对 LKM 的转述，未取得有价值的新线索。该项基本未完成。
  3. Elicit / SciSpace / Undermind / Asta：Asta FAQ 返回 403，Elicit 的搜索无一手命中，SciSpace 与 Undermind 未逐页核。这几家只能标"未核实"。
  4. ACM DL 被 Cloudflare 挡(CDP 下仍如此)，MPR 改走 Crossref + OpenAlex + GitHub 核实，见 §2。

## 1. 判决表

列：工作 | 一手链接 | 发现渠道 | 跨论文对象 | 独立性/条件/时间处理 | 直接评测 | 和我们的关系

| 工作 | 一手链接 | 发现渠道 | 跨论文对象 | 独立性/条件/时间 | 直接评测 | 关系(威胁) |
|---|---|---|---|---|---|---|
| **Record Grouping Controls Evidence Weight in Language Models** | [2609.08698](https://arxiv.org/abs/2609.08698)，2026-09-08 | DDG 查 "counting copies as evidence" | 不是科学文献对象。给定一个"记录分组(partition)"，分组内去除拷贝、保留互补内容，限定每组对证据的贡献 | **独立性的核心**：错误拆分(false split)使证据权重增加 10.27-32.66 个百分点，错误合并减少 9.13-31.79；推导了 partition 误差界。身份(分组)错误传播到证据权重，恰好是我们"身份错误如何传播到聚合"的通用版 | 有：104,402 次试验、6 个 checkpoint、48 项受控 campaign 面板 | **半真，中等**。不是科学文献，也不做族/条件/时间。但它是"支持集去重 + 分组错误敏感性"的最近一手先例，我们必须引用并说明差异(科学来源独立性的信号是作者/数据/代码重叠，而非文本拷贝)。与 RESEARCH-AGGREGATION §5 第 5 条"身份稳健聚合"直接相关 |
| **Counting Copies as Evidence: Confidence Inflation from Dependent Evidence in RAG** | 仅见 ResearchGate PDF 链接(DDG 命中)，**未取得 arXiv/DOI**，未核实 | DDG | 标题指向 RAG 中依赖证据导致置信度膨胀 | 同上主题 | 未知 | **未核实**，列入 §4。若存在则同属"依赖证据"线 |
| **How retriever redundancy and diversity impact RAG effectiveness** | [2608.13956](https://arxiv.org/abs/2608.13956)，2026-08-14 | DDG | 无(检索集的冗余/多样性) | 控制了混杂因素，看冗余对生成正确性的影响 | 有，RAG 实验 | 假威胁。相关背景，不涉及科学主张或来源独立 |
| **Ideas Have Genomes (IdeaGene-Bench / IG-Bench)** | [2607.08758](https://arxiv.org/abs/2607.08758)，2026-07-09 | DDG | 每篇论文表示为一组 typed、带证据的 Idea Genome 对象；GenomeDiff 记录 inheritance / mutation / loss / external import / novel insertion，六种演化动力 | 时间：谱系 trace；无独立性、无共识 | 有：1,961 条 golden lineage trace、1,085 Idea Genome 对象、920 条 pairwise GenomeDiff、10 个学科；IG-Exam 42 类任务，最强系统 27.3% exact accuracy | **半真，中高**。与"方法演化/继承"重合，且带 gold。不与族级事实/共识重叠，但是对"演化关系"一块的新对手(Intern-Atlas 之外)。记忆库里 IG-Bench 曾作为"gold 扩容机会"，需重看其对象定义能否复用为演化边 gold |
| **Single-Round Vector RAG vs an LLM-Compiled Wiki (preregistered)** | [2605.18490](https://arxiv.org/abs/2605.18490)，2026-05-18 | DDG "LLM wiki" | 24 篇研究语料上的 LLM 编译 markdown wiki | 无独立性/条件/时间 | 预注册，13 题、2 名盲评 LLM judge；wiki 在"连接发现"上大幅更好，但组织优势未过预注册阈值 | 假威胁，但有用：是"编译式知识层 vs RAG"的小规模预注册对照，可当方法学参照 |
| **EviGraph: Evidence-Guided Autonomous Research Agents** | [2608.04738](https://arxiv.org/abs/2608.04738)，2026-08-05 | DDG | typed evidence graph(Problem/Gap/Hypothesis/Experiment/Finding/Claim)，是 agent 自身研究过程的状态，不是文献聚合 | 检查证据链缺失 | 见 abs | 假。对象是 agent 研究过程，不是跨论文 |
| **An Autonomous Scientific Knowledge Generation Framework (AI-Ready Scientific Knowledge Base)** | [2607.09806](https://arxiv.org/abs/2607.09806)，2026-07-09 | DDG | 本体引导的文献获取、抽取、语义协调、知识融合、验证，面向材料类数据库 | 有 "knowledge fusion"，abs 层未见独立性/条件对齐细节 | abs 未给聚合层评测 | 假到半真，低。材料数据库构建，不是主张/方法族层。全文未读，细节未核 |
| **Traxia** | [2606.08256](https://arxiv.org/abs/2606.08256)，2026-06-06 | DDG | agent 原生发表框架，含"带矛盾检测的知识图" | 每条主张带置信区间；信誉/质押 | 形式化框架，abs 未见直接评测 | 假。是发表基础设施设计，不是文献聚合 |
| **MetaLead** | [2601.22420](https://arxiv.org/abs/2601.22420)，2026-01-30 | DDG | 人工标注的 ML 排行榜，含实验类型(baseline / proposed method / variation)和训练/测试分离 | 元数据里区分"proposed method"，可作方法归属的旁证 | 数据集本身 | 半真，低。1-2 月，略早于本轮窗口，但其"proposed method vs baseline"标签在方法提出者归属上可借。全库规模、是否含 introduced-in 字段未核 |
| **Eligibility-Aware Evidence Synthesis (EligMeta)** | [2604.02678](https://arxiv.org/abs/2604.02678)，2026-04-03 | DDG | 临床试验元分析，把"资格标准对齐"纳入研究权重，给出 cohort-specific pooled estimate | **条件对齐进入聚合权重**：这是"条件对齐后才聚合"的一个具体实现，但限临床试验 | 见 abs | 半真，低到中。条件对齐聚合的先例，领域是临床，不是 CS 方法族。可当"条件对齐"的旁证 |
| **DeepEvidence**(Nature Machine Intelligence) | [s42256-026-01266-0](https://www.nature.com/articles/s42256-026-01266-0)，2026-07-02 | DDG | 生物医学深度研究 agent，增量构建 key entities 与 observations 的证据图 | 证据图追踪与归因 | 4 个公开基准 + 自建 7 个任务 | 假。agent 探索图，非领域级聚合对象 |
| **MPR**(He, Gu, Chen, Yan；JCDL 2024 / ACM DOI) | DOI [10.1145/3677389.3702579](https://doi.org/10.1145/3677389.3702579)；数据 [github.com/ChenCbnu/methodanno](https://github.com/ChenCbnu/methodanno) | DDG；ACM 被 Cloudflare 挡，改 Crossref + OpenAlex 摘要 + GitHub 原文 | 无(单句层) | 无 | 有数据集与 SciBERT 基线 | **已占位，但占的是句级标签，不是跨论文归属**。见 §2 |
| **Consensus Meter(现行版)** | [help.consensus.app 文章 10069920](https://help.consensus.app/en/articles/10069920-the-consensus-meter)，页面日期 2026-04-22 | DDG 命中后用 CDP 读 | 对 top-20 论文的立场计数：Yes / No / Possibly / Mixed；至少 5 篇才显示 | 无独立性。Snapshot 给四个质量指标：Recency(平均发表时间)、Methods(元分析/系统综述/RCT 的篇数)、Journals(平均 SJR)、Citations(引用总和)。即证据等级只按研究设计与期刊计，不按来源独立 | 官方未给评测协议 | 真，但无撞车。相对前一轮已核的旧版：**标签集已更新为含 Mixed**，并新增 Snapshot。仍是立场计数，无条件对齐、无独立性、无 as-of |
| **scite(现行文档)** | [scite.ai/llms.txt](https://scite.ai/llms.txt) | CDP 读 llms.txt | Smart Citations：supporting / contrasting / mentioning；Collections 可"track supporting or contrasting evidence"并提醒撤稿/编辑通知 | 撤稿与编辑通知是 scite 已有的一种时间/状态信号；仍无独立性与条件 | 与前一轮一致，无新评测 | 真，无撞车。撤稿提醒是"状态修订"的最小产品形态，前一轮没记 |
| **Consensus 文档站**(Research Agent / Deep Search / comparison table / gap analysis) | [docs.consensus.app/llms.txt](https://docs.consensus.app/llms.txt) | CDP 读 | 从多篇论文抽取对比表(样本量、效应量等)；Library 的 gap analysis；"supporting and challenging studies" 查找 | 无独立性计数、无条件对齐共识判定 | 无 | 真，无撞车。"族级综合"不在产品功能里，最接近的是 comparison table 与 gap analysis |

## 2. MPR 补查结果(前一轮未核实项)

- 一手核实链：Crossref 返回 DOI 10.1145/3677389.3702579、会议 JCDL'24 论文集、作者 He / Gu / Chen / Yan；OpenAlex 与 Semantic Scholar 返回完整摘要；GitHub 仓库 ChenCbnu/methodanno 含 `data_github.jsonl` 与 "annotation guidline.pdf"。
- 摘要原文要点：把"方法实体与宿主论文的关系"做成端到端 NER 序列标注，而不是整句分类；数据集公开的是一部分(Part of the MPR dataset)；SciBERT 基线最好。
- 公开样本实测(我直接读了 raw 文件)：1000 行句子，实体标签分布 propose 541、use 413、mention 273、compare 112、improve 58。即 MPR 的标签集就是 **propose / use / mention / compare / improve**，粒度是"句内方法实体 vs 本文"。
- 结论：
  1. **MPR 已占"方法实体与宿主论文关系(提出/使用)"这一句级标注位**，不是空白。前一轮"需补查是否已占位"的答案是：已占，且是公开数据。
  2. 但它不覆盖本项目 E1 要测的东西：把一个引用标记/方法名解析到**提出它的那篇论文**(跨论文归属)。MPR 的关系对象是"宿主论文"，不含"被解析的目标论文"。因此可以作为我们"提出者识别"单篇层的外部标注参照(propose 标签对标我们的提出句识别)，但不是跨论文归属的 gold。
  3. 未核：完整数据规模(摘要只说 part public)、标注一致性(在 PDF 里，页面被挡)、论文正文。公开仓库样本只有 1000 行，不能当全量结论。
- Papers with Code introduced-in 数据：本轮 DDG 命中的都是镜像站(paperswithcode.co)与无关结果，**没有一手页面证明 PwC 存档含 introduced-in 字段的可用公开数据集**，列未核实。

## 3. 前一轮漏掉的最重要条目(3-5 条)

1. **2609.08698 Record Grouping Controls Evidence Weight**：前一轮"独立性"一节只写了 GraphEcho 与元分析，漏了这条"分组错误导致证据权重 +10~33 / -9~32 个点，且给出误差界"的直接量化。它为我们"身份/独立性错误传播到聚合"的敏感度实验提供了方法学先例，论文里必须引用并区分。
2. **MPR 公开数据与标签集(propose/use/mention/compare/improve)**：前一轮因页面被挡列未核实；现在已核实占位，且是句级。这改变 E1 的叙事：不能再写"没有数据集评测过 propose vs use"，应写"句级有 MPR，跨论文归属无"。
3. **2607.08758 IG-Bench(Ideas Have Genomes)**：带 1,961 条 golden lineage trace 的演化/继承 gold，前一轮演化边一节只有 Intern-Atlas。它是演化对象上的第二个有 gold 的对手，也是潜在可复用的 gold。
4. **Consensus Meter 现行形态**：标签含 Mixed，Snapshot 用方法/期刊/引用/时间四项质量指标，证明"证据强度"已被产品化为研究设计加期刊质量，而不是来源独立。前一轮引的是旧版博客。
5. **EligMeta(2604.02678)**：条件(资格标准)对齐进入聚合权重的具体实现，限临床。可作为"条件对齐聚合"的先例引用。

## 4. 未核实清单(不进主表)

- "Counting Copies as Evidence: Confidence Inflation from Dependent Evidence in RAG"：仅 ResearchGate PDF 链接，无 arXiv/DOI。
- Papers with Code introduced-in 存档数据：无一手页面。
- Ai2 Asta：faq 页 403；是否有共识度/证据计数功能未核。
- Elicit、SciSpace、Undermind：未逐页核功能。site-patterns 显示这些站点可读(Elicit support 站、Undermind /mcp)，本轮没有时间深入。
- 中文社区(知乎/公众号)：登录墙，仅得 arxivdaily 对 LKM 的转述，无新工作线索。
- 2607.09806 全文细节(knowledge fusion 是否含来源独立/冲突处理)。
- MetaLead 是否含 introduced-in 类字段及规模。
- 2604.02678 EligMeta 的评测细节；2608.04738 EviGraph 的评测细节(只读了 abs 开头)。

## 5. 对"有没有直接撞车"的判断

- 没有发现新的"方法族 + 族级事实 + 独立性计数 + 条件对齐共识 + as-of"同时具备的工作。LKM / ScholarStack / Agents-K1 之外，本轮 2026-06 至 2026-10 的新命中都是单点：证据独立性(非科学)、临床条件加权、演化 gold、句级提出标签、产品计数。
- 相对前一轮，威胁等级上调的只有两处：独立性有了 2609.08698 这个一手量化先例(不撞车，但必须引用)；方法归属的句级标注位被 MPR 占了(要改措辞)。
- 本结论的强度受上面覆盖局限约束。
