# 查新：自述 / 他述与"一篇论文的描述随时间变化"（2026-10-04）

## 0. 结论先行

1. 叙事的开场论点"他述不同于自述，他述反映社区认识"已经有 18 年历史，不能当贡献。Mei & Zhai（ACL 2008）的引言几乎是原句："the abstract of a paper mostly reflects the expected impact of the paper as perceived by the author(s), which could significantly deviate from the actual impact of the paper in the research community. Moreover, the impact of a paper changes over time"。Elkiss et al. 2008、Qazvinian & Radev 2008、Divoli et al. 2012、CL-SciSumm（"community created summary"）都提出过同样的观点，Chen et al. 2025（Scientometrics）用 153 万条引用语境量化了自述与他述的差距（称为 "ideal bias"）。
2. "对同一篇论文的描述随施引年份变化"作为实证发现也已被占：Divoli 2012（第一年施引者与作者基本一致，之后逐渐分化）、He & Chen 2018（按时期训练引用语境 embedding，量化单篇论文的角色变化）、Lin et al. 2025（高被引论文随年龄增长转向背景性、符号性引用）、KCT 2026（单篇论文贡献类型分布的早晚期 JSD 轨迹）。
3. 最近的直接邻居是 **Arnaout, Sternlicht, Hope, Gurevych，"In-depth Research Impact Summarization through Fine-Grained Temporal Citation Analysis"（arXiv 2505.14838，v2 2026-04）**。它用 LLM 从带年份的细粒度引用意图生成 time-aware 的单篇论文影响摘要，覆盖 confirmation 和 correction 两类，并有专家评测。它与我们最核心的那句话高度重合，但输出是自由文本，粒度是单篇论文，没有方法族、领域级聚合、按截止时间作答、自述与他述的显式对照，也不给 agent 用。
4. 没有找到直接撞车的系统，也就是同时做到以下四点的工作：自述加他述双抽取、按时间汇总成结构化的领域认知、识别类型化的认识变迁事件、作为可按截止时间查询的工具给 agent 用。空位在组合与结构化这一层，不在任何单个观点上。
5. 风险：UKP（Gurevych 组）已经分别做了 Nature of NLP（自述贡献随时间的分布）、ClaimFlow（主张关系随时间变化）和 Arnaout（时间感知的影响摘要）。把这三件拼起来就覆盖了我们叙事的大半，这是最可能出现的撞车来源，需要持续监控。

## 1. 方法与覆盖

- 发现层：CDP 真实浏览器上的 DuckDuckGo（Google 上一轮触发过验证码，本轮未用）；arXiv 站内检索页（curl，后段被 429 限流）；Semantic Scholar API 的 cited-by 追踪（Arnaout、Elkiss、Mei & Zhai、Divoli、He & Chen、Lin et al. 六篇）；scite 与 Ai2 的官方文档页。
- 验证层：主表每条都打开了一手页面，包括 arXiv abs/html/PDF 正文、ACL Anthology PDF、PMC 全文、Frontiers 全文、Springer 摘要（CDP）、Deep Blue 机构库、OpenAlex/Crossref DOI 元数据、scite 的 llms-full.txt 和 Asta MCP 页面。凡是标"读正文"的条目，结论都来自 grep 正文后读上下文。
- 未覆盖：Google Scholar、中文社区（知乎、CNKI）没有查。ScienceDirect 触发验证码，Small et al. 2017 的摘要没有拿到。
- 威胁等级的口径：**真**＝直接占掉我们打算主张的一项贡献；**半真**＝占掉其中一部分，或做了同一件事的弱形态；**假**＝只构成背景或动机，必须引用但不构成威胁。

## 2. 判决表

### 2.1 引用摘要（citation-based summarization）一脉

| 工作 | 一手链接 | 发现渠道 | 做了什么 | 是否涉及时间变化 | 是否聚合成结构化认知 | 是否给 agent/下游用 | 与我们的关系 |
|---|---|---|---|---|---|---|---|
| Elkiss et al. 2008, JASIST 59(1)："Blind men and elephants" | [Deep Blue hdl 2027.42/57540](https://hdl.handle.net/2027.42/57540)，DOI 10.1002/asi.20707 | DDG(CDP) | 生物医学论文的 citation summary（引用句集合）与摘要部分重叠，但侧重不同的方面；不同施引者各自只描述被引论文的一部分。用 cohesion 词汇相似度论证 | 否 | 否，只做描述性对比 | 否 | 假。"他述≠自述"的源头之一，必须引用 |
| Qazvinian & Radev 2008, COLING："Citation Summary Networks" | [ACL C08-1087](https://aclanthology.org/C08-1087/)（读 PDF） | 一手猜 ID 直访 | 用 "others' viewpoint of the target article's contributions" 做单篇摘要，动机是 "Quickly moving to a new area of research is painful" | 否 | 否，抽取式摘要 | 人读 | 假。我们"帮初学者建立认知"这个次要用途就是这一脉的原始动机 |
| Mei & Zhai 2008, ACL："Impact-Based Summaries" | [ACL P08-1093](https://aclanthology.org/P08-1093/)（读 PDF） | 一手直访 | 用引用语境估计 impact language model，抽取论文中有影响的句子；SIGIR 1978–2005 共 1,303 篇 | **引言里明确提出**"影响随时间变化，十年前的算法不再是 SOTA，但问题定义仍被接受"，但方法不建模时间（正文 grep 无时间维） | 否 | 人读 | 假，但开场论点几乎被它原句说过，必须正面引用 |
| Mohammad et al. 2009, NAACL："Using Citations to Generate Surveys of Scientific Paradigms" | [ACL N09-1066](https://aclanthology.org/N09-1066/)（读 PDF） | 一手直访 | 用引用文本生成技术综述 | 否 | 否 | 人读 | 假。"由他述生成领域综述"的早期先例 |
| Abu-Jbara & Radev 2011, ACL："Coherent Citation-Based Summarization" | [ACL P11-1051](https://aclanthology.org/P11-1051/)（读 PDF） | 一手直访 | 把引用句分为 Background / Problem Statement / Method / Results / Limitations 五类，聚类后生成连贯摘要 | 否 | 半：按功能分槽，但输出是文本 | 人读 | 假。"他述指认局限"作为一个槽位在 2011 年就有 |
| Divoli, Nakov, Hearst 2012, Adv. Bioinformatics："Do Peers See More in a Paper Than Its Authors?" | [PMC3514807](https://pmc.ncbi.nlm.nih.gov/articles/PMC3514807/)（读全文），PMID 23227044 | DDG(CDP) | 对照摘要（作者视角）与 citances（同行视角）：citances 多 20% 概念，同行更关注实验方法 | **是**：按施引年份分组（0 到 4+ 年），"in the first year peers largely agreed with the authors, while differentiation was observed later" | 否，概念计数 | 否 | 假。"他述随时间偏离自述"的直接实证先例 |
| CL-SciSumm 2016–2018 共享任务（Jaidka et al.） | [ACL W16-1511](https://aclanthology.org/W16-1511/)（读 PDF）；[arXiv 1909.00764](https://arxiv.org/abs/1909.00764)；[repo](https://github.com/WING-NUS/scisumm-corpus) | DDG(CDP) | Task 1A 为每条 citance 找被引原文片段；1B 给片段分 facet；Task 2 生成结构化摘要。把 citances 称为 "(community created) summary" | 否 | 半：有 facet，但单篇 | 人读 | 假。"他述↔原文片段对齐"的数据与评测可复用 |
| Cohan & Goharian 2015, EMNLP | [ACL D15-1045](https://aclanthology.org/D15-1045/)（读 PDF） | 一手直访 | 给每条引用补上被引原文语境，解决 "inconsistency between the citation summary and the article's content" | 否 | 否 | 人读 | 假。它把他述可能走样当作需要修补的噪声，这一点与我们相关 |
| ScisummNet（Yasunaga et al., AAAI 2019） | [arXiv 1909.01716](https://arxiv.org/abs/1909.01716) | 一手直访 | 1,000 篇 CL 论文的人工摘要；hybrid 摘要把作者 highlights（摘要）和社区 impact（引用）合在一起 | 否 | 否 | 人读 | 假。"自述加他述双视角"在单篇摘要层面早已有人做 |
| Syed et al. 2023, Findings EMNLP："Citance-Contextualized Summarization" | [arXiv 2311.02408](https://arxiv.org/abs/2311.02408) | S2 cited-by(Elkiss) | 给定 citance，生成被引论文中与该处相关的摘要；54 万篇、460 万 citances | 否 | 否 | 写作/阅读辅助 | 假 |
| CiteRead（Rachatasumrit et al., IUI 2022） | DOI [10.1145/3490099.3511162](https://doi.org/10.1145/3490099.3511162)（OpenAlex 摘要） | DDG(CDP) | 把后续论文对本文的评论定位到本文对应位置，放在页边展示 | 否 | 否 | 人读，12 人用户研究 | 假。"阅读时看后来人怎么说"的交互先例 |
| **Arnaout, Sternlicht, Hope, Gurevych 2025/2026："In-depth Research Impact Summarization through Fine-Grained Temporal Citation Analysis"** | [arXiv 2505.14838](https://arxiv.org/abs/2505.14838)（读 html 正文；v1 2025-05-20，v2 2026-04-16） | arXiv 站内检索 "impact summarization" | 新任务：用 LLM 为每条引用生成细粒度意图，判定是否 impact-revealing（confirmation 或 critique/correction），再把带年份的引用语境与意图喂给 GPT-4o，生成 time-aware 影响摘要。评测：105 篇（心理、医学、CS）；指标为 faithfulness、coverage（0.34）、citation year compliance（0.59）、trend awareness；9 位教授评自己的论文。应用：10 个主题的对比，以及作者级聚合 | **是**：摘要要求追踪 "the evolution of its impact over time" | 否：自由文本；只有单篇和作者级；无方法族、无领域状态、无变迁事件类型 | 否：给人读，不是工具 | **半真，最近邻**。占掉"按时间汇总他述"这一句。我们的差异必须落在结构化、领域级、变迁事件分型、自述与他述显式对照、按截止时间作答和 agent 评测上。它自报的年份归属错误（0.59）提示：按截止时间作答首先要解决引用年份与施引年份的混淆 |
| SECite（Pyreddy et al., IEEE CCWC 2026） | [arXiv 2601.07939](https://arxiv.org/abs/2601.07939) | arXiv 站内检索 | 9 篇软件工程论文：引用情感分正负，再由 LLM 分别总结优点与局限，并对照 "the authors' own presentation" | 否 | 否 | 人读 | 假，规模很小。不过"他述与自述的一致和分歧"是它摘要里的原话 |

### 2.2 科学计量与科学社会学

| 工作 | 一手链接 | 发现渠道 | 做了什么 | 是否涉及时间变化 | 是否聚合成结构化认知 | 是否给 agent/下游用 | 与我们的关系 |
|---|---|---|---|---|---|---|---|
| Small 1978, Social Studies of Science："Cited Documents as Concept Symbols" | DOI [10.1177/030631277800800305](https://doi.org/10.1177/030631277800800305)（OpenAlex 摘要） | DDG(CDP) | 从引文周边文字确定施引者把哪个概念与被引文献关联；高被引化学文献呈现高度一致的 "standard symbols"；称之为 "a dialogue among citing authors on the 'meaning' of earlier texts" | 否 | 否 | 否 | 假。"一篇论文的意义由后来人的表述构成"这一理论表述的源头 |
| Cozzens 1982, JASIS："Split Citation Identity" | DOI [10.1002/asi.4630330407](https://doi.org/10.1002/asi.4630330407)（OpenAlex 摘要） | DDG(CDP) | Klein & Rubin 1948 在 1975–1977 年被两个研究网络分别关联到两个不同概念 | 是，限定在某个时段 | 否 | 否 | 假。"同一篇被不同社群重新归类"的经典个案 |
| Hargens 2000, ASR："Using the Literature" | DOI [10.1177/000312240006500603](https://doi.org/10.1177/000312240006500603)（OpenAlex 摘要） | DDG(CDP) | 基础性文献被不成比例地引用，是因为它被当作"某种视角或一般路径的例子"而非具体论据 | 间接 | 否 | 否 | 假。对应我们说的"变成默认组件、只作代表被提及" |
| McCain 2011 / 2012 / 2015（JASIST）：obliteration by incorporation（OBI） | DOI [10.1002/asi.21536](https://doi.org/10.1002/asi.21536)、[10.1002/asi.22719](https://doi.org/10.1002/asi.22719)、[10.1002/asi.23335](https://doi.org/10.1002/asi.23335)（OpenAlex 摘要） | DDG(CDP) | Nash Equilibrium、ESS、bounded rationality：概念短语出现但不引用原作的比例随时间上升，并随学科变化；全文检索与记录级检索测得的 OBI 不同 | **是**：OBI 的时间趋势 | 否 | 否 | 假。"变成默认组件"在计量学里的名字就是 OBI；它的后果是这类变迁在引用语境里不可见，必须靠无引用提及来检测 |
| Meng, Varol, Barabási："Hidden Citations Obscure True Impact in Science"（arXiv 2023，PNAS Nexus 2024 待核） | [arXiv 2310.16181](https://arxiv.org/abs/2310.16181) | arXiv 站内检索 "obliteration by incorporation" | 全文无监督识别 hidden citation（明确提到某发现但不引用原文）；对有影响的发现，hidden citations 多于正式引用 | 是，随话语增多而增加 | 否 | 否 | 假。如果我们要检测"被吸收/变成默认组件"，这是必须对比的方法 |
| Greenberg 2009, BMJ："How citation distortions create unfounded authority" | DOI [10.1136/bmj.b2680](https://doi.org/10.1136/bmj.b2680)（OpenAlex 摘要） | DDG(CDP) | 一个医学主张的完整引文网络（242 篇、675 条引用）：citation bias、amplification、invention（"conversion of hypothesis into fact through citation alone"） | 是，主张随引用链被放大 | 否，个案网络分析 | 否 | 假，但对叙事是**反向约束**：他述可以系统性地走样，"领域里的位置由后来者重新表述"不等于"后来者的表述是对的" |
| Jergas & Baethge 2015, PeerJ：引文准确性荟萃分析 | DOI [10.7717/peerj.1364](https://doi.org/10.7717/peerj.1364)（OpenAlex 摘要） | 一手 DOI 直查 | 28 项研究，医学论文引用错误率合计 25.4%（主要错误 11.9%） | 否 | 否 | 否 | 假。他述噪声率的基线数字 |
| Simkin & Roychowdhury 2002："Read before you cite!" | [arXiv cond-mat/0212043](https://arxiv.org/abs/cond-mat/0212043) | DDG(CDP) | 从引文印刷错误的传播推断：只有约 20% 的施引者读过原文 | 否 | 否 | 否 | 假。他述多为转抄的证据，提示需要区分独立他述与转抄他述 |
| Hsiao & Schneider 2021, QSS：撤稿论文的持续引用 | DOI [10.1162/qss_a_00155](https://doi.org/10.1162/qss_a_00155)（OpenAlex 摘要） | DDG(CDP) | 7,813 篇撤稿论文、48,134 条引用语境：撤稿后的引用方式不变，只有 5.4% 提到撤稿 | **是**：60 年纵向分析，撤稿前后对比 | 否 | 否 | 假。说明他述对状态变化的反应很慢，是"截至时间 T 的认知"滞后于事实的实证 |
| Catalini, Lacetera, Oettl 2015, PNAS：负面引用 | DOI [10.1073/pnas.1502280112](https://doi.org/10.1073/pnas.1502280112)（OpenAlex 摘要） | 一手 DOI 直查 | 负面引用更多指向高质量论文，集中于 findings 而非理论或方法；收到负面引用后长期被引下降略快 | 是，后续引用轨迹 | 否 | 否 | 假 |
| Li 2020："(Re-)instrumentalization of the DSM" | [arXiv 2010.11101](https://arxiv.org/abs/2010.11101) | arXiv 站内检索 | DSM 新版发布后，引用语境中的 hedges 和动词画像逐渐变化，即越来越被当作有效工具 | **是**：跨版本的引用语境轨迹 | 否 | 否 | 假。"变成默认组件/被确立"可以从引用语境的语言标记中测出来，这是方法先例 |
| Ke et al. 2015, PNAS：Sleeping Beauties | DOI [10.1073/pnas.1424329112](https://doi.org/10.1073/pnas.1424329112)（OpenAlex 摘要） | DDG(CDP) | 2,200 万篇论文中的延迟认可现象，无参数的测度 | 是，只看引用计数 | 否 | 否 | 假 |
| Chen, Ding, Song, Qu 2025, Scientometrics 130(5)："Exploring scientific contributions through citation context and division of labor" | DOI [10.1007/s11192-025-05318-x](https://doi.org/10.1007/s11192-025-05318-x)（Springer 摘要，CDP） | KCT 参考文献 → Crossref → Springer | Nature/Science 论文，153 万条引用语境，用 LLM 识别"实际贡献"类型：实验类贡献占主导，而作者自述以理论和方法为主，作者称之为 "ideal bias" | 否 | 半：论文级贡献类型分布 | 否 | **半真**。"自述与他述的贡献类型系统性不同"已被大规模量化。如果我们把自述/他述差异作为发现来报，必须对比它 |
| He & Chen 2018, Frontiers RMA："Temporal Representations of Citations for Understanding the Changing Roles of Scientific Publications"（workshop 版 [arXiv 1711.05822](https://arxiv.org/abs/1711.05822)） | DOI [10.3389/frma.2018.00027](https://doi.org/10.3389/frma.2018.00027)（读全文） | DDG(CDP) | PMC OAS 2007–2016，136 万篇全文；按时期训练引用语境的 temporal embedding，量化每篇论文"角色变化"的程度，并用邻近词解释怎么变的 | **是**：单篇论文的角色变化分数 | 否：只有向量和变化分数，没有结构化类型 | 否 | **半真**。"单篇论文被描述的方式随时间变化"的计算方法先例；我们的增量必须是可解释的类型化变迁，而不只是一个距离 |
| Lin, van Eck, Hou, Hu 2025："The changing role of cited papers over time" | [arXiv 2509.04190](https://arxiv.org/abs/2509.04190) | DDG(CDP) | 约 900 篇高被引论文，22 万篇施引全文：随论文变老，引用位置前移、提及次数减少、与其他文献合并引用增多、文本相似度下降，即从直接方法性接触转向背景性、符号性引用 | **是**：群体层面 | 否 | 否 | 假。"变成默认组件/被符号化"的群体层面实证 |
| **KCT**（Quan et al., arXiv 2026-08）："From citation intent to knowledge contribution" | [arXiv 2608.20697](https://arxiv.org/abs/2608.20697)（读 PDF 正文） | 用户线索，一手核实 | 按被引论文实际贡献的知识类型分类（Method / Resource Tool / Empirical Finding / Background × core/non-core），ACC 85.5%；ACL 80 万条引用中 core 只占 39.09%。§4.2 Fig.6：选 2005–2014 年的论文，计算引用年龄 1–5 与 6–10 两期贡献分布的 JSD，给出四种轨迹 | **是**：单篇论文贡献类型的早晚期轨迹（4 个案例） | 否：单条引用打单标签，再做分布统计 | 否：用于科研评价和传播预测 | **半真**。"被描述的贡献类型随施引年份变化"已被它在个案层面做过。差异：不汇总成领域认知，不处理局限和比较对象，只做描述 |

### 2.3 科学术语/方法名的语义漂移

| 工作 | 一手链接 | 发现渠道 | 做了什么 | 是否涉及时间变化 | 是否聚合成结构化认知 | 是否给 agent/下游用 | 与我们的关系 |
|---|---|---|---|---|---|---|---|
| Soni, Lerman, Eisenstein 2019/2021, JASIST："Follow the Leader" | [arXiv 1909.04189](https://arxiv.org/abs/1909.04189) | arXiv 站内检索 | 把 diachronic word embedding 的语义变化落到文档上，度量文档的 semantic progressiveness；处于语义变化前沿的科学论文更多被引 | 是 | 否 | 否 | 假 |
| Kazi et al. 2022, SDP workshop | [ACL 2022.sdp-1.10](https://aclanthology.org/2022.sdp-1.10/)（读 PDF 首页） | DDG(CDP) | PubMed 六十年的术语语义漂移计算与可视化 | 是 | 否 | 人看 | 假 |
| Simons 2024："Meaning at the Planck scale?" | [arXiv 2411.14073](https://arxiv.org/abs/2411.14073) | arXiv 站内检索 | 用领域预训练 BERT 做 "Planck" 的义项消歧，追踪三十年的义项变化 | 是 | 否 | HPSS 研究 | 假 |
| Liu, Gerdes, Deltorn 2026："Beyond frequency measures" | [arXiv 2609.18804](https://arxiv.org/abs/2609.18804) | arXiv 站内检索 | 天体物理与 NLP 语料 2010–2024，对比频率与上下文 embedding 两种方式检测术语意义变化，并用专家标注作 gold | 是 | 否 | 否 | 假 |
| Zichert & Simons 2026：计算概念史综述（书章） | [arXiv 2606.04118](https://arxiv.org/abs/2606.04118) | DDG(CDP) | 梳理从早期数字方法到 LLM 的科学概念史计算方法 | 是 | 否 | 否 | 假。可作该方向的综述入口 |
| Pramanick et al. 2025, ACL："The Nature of NLP" | [arXiv 2409.19505](https://arxiv.org/abs/2409.19505)（comments：accepted at ACL 2025） | DDG(CDP) + KCT 参考文献 | 从摘要抽取作者自述的贡献并分类，覆盖约 2.9 万篇论文、五十年 NLP，给出贡献类型趋势 | 是：领域层面的自述趋势 | 半：贡献类型分布 | 否 | 假。它只有自述，正好是我们自述层的对照；与 ClaimFlow、Arnaout 同属 UKP |

这一方向的现状：术语漂移检测在词或概念层面很成熟，但对象是词的义项，不是"某篇论文或某个方法被怎样定位"。没有找到把方法名漂移与施引描述的变化联系起来的工作。

### 2.4 LLM 时代：把后续论文的描述汇总成结构化认知，或给 agent 用

| 工作 | 一手链接 | 发现渠道 | 做了什么 | 是否涉及时间变化 | 是否聚合成结构化认知 | 是否给 agent/下游用 | 与我们的关系 |
|---|---|---|---|---|---|---|---|
| Arnaout et al. 2025/2026 | 见 §2.1 | | | 是 | 否 | 否 | 半真，最近邻 |
| **AgentExpt**（Li et al., arXiv 2025-11） | [arXiv 2511.04921](https://arxiv.org/abs/2511.04921)（读 html 正文） | arXiv 站内检索 "citation contexts" LLM agent | 为 baseline/数据集推荐建 "collective perception"：抽取下游论文的引用语境，汇总成 aggregate usage profile，与 artifact 自述拼成 dual-view 表示，微调 embedding 检索，再用交互链 reranker；按时间顺序切分以防泄漏。消融显示 collective perception 贡献最大 | 半：只有时间切分评测，没有随时间的描述变化 | 半：每个 artifact 一段汇总文本 | **是**：自动实验设计 agent | **半真**。"自述 + 他述汇总 → 供 agent 使用"已经出现在 baseline/数据集这一窄对象上，而且有消融收益。差异：对象不是论文的领域位置，没有时间维和变迁，没有方法族 |
| **scite MCP / Smart Citations** | [docs.scite.ai/agents](https://docs.scite.ai/agents)、[llms-full.txt](https://docs.scite.ai/llms-full.txt)（读一手文档） | DDG(CDP) | 25 个 MCP 工具，ChatGPT/Claude 连接器；Smart Citation 把每条引用分为 supporting / contrasting / mentioning，给出 tally 和按章节的 tally；检索支持 yearFrom/yearTo；Collections 可长期监控新的支持或反驳、撤稿提醒。文档示例："Check whether later studies support or contradict this DOI." | 半：有日期过滤和监控，没有变迁类型 | 否：三分类计数，没有定位、族、局限 | **是**：已是面向 agent 的产品 | **半真**。"agent 查后来人怎么说这篇论文"已是产品形态，只是语义很粗。论文中必须写清对比：三分类计数 vs 类型化的定位与变迁 |
| Asta Scientific Corpus Tool（Ai2 MCP） | [allenai.org/asta/resources/mcp](https://allenai.org/asta/resources/mcp)（读一手页面） | DDG(CDP) | get_citations(paper_id, publication_date_range)、snippet_search(..., inserted_before) 等 | 有时间过滤参数 | 否：返回原始引用和片段 | 是 | 假。它是原料层：agent 已能按日期拿到施引片段，但没有编译好的认知。这说明"按截止时间取他述"本身不是贡献，编译和汇总才是 |
| Ai2 Paper Finder | [allenai.org/blog/paper-finder](https://allenai.org/blog/paper-finder)（读一手） | DDG(CDP) | 用含引用的句子来识别"某名称指的是哪篇论文"（如 alphageometry），再用前向和后向引用扩展检索 | 否 | 否 | 是，检索 agent | 假。施引句只用作指认，不做汇总 |
| ClaimFlow（Pramanick et al., arXiv 2026，v2 2026-06） | [arXiv 2603.16073](https://arxiv.org/abs/2603.16073)（读 html 正文） | 用户线索 + arXiv 检索 | 主张级 support / extend / qualify / refute / background；计算首次被挑战的时间；qualify（8.3%）多于 refute（2.8%）；不确定性在生命周期早期最高。个案：BERT 的主张"被逐步限定而非推翻"，BLEU "尽管持续被批评，仍深嵌在评测实践中" | **是**：主张的生命周期 | 半：主张关系图 | 否 | **半真**。我们说的"适用范围被收窄"和"暴露新局限"在主张层面已有标注 gold 和时间分析；BLEU 个案就是"被批评但成了默认组件"。差异：没有论文/方法的定位画像，没有按时间截止的查询接口，变迁类型只有关系类，没有事件类 |
| Crystal（Collison, Van Durme, Khashabi, arXiv 2026-03） | [arXiv 2603.26791](https://arxiv.org/abs/2603.26791) | S2 cited-by(Arnaout) | 在一篇施引论文内联合排序所有被引文献的相对影响；ACL Test-of-Time 个案 | 否 | 否 | 否 | 假 |
| MUSES / CiteRoots（Pandey, Kwon, Yu, arXiv 2026-08） | [arXiv 2609.00313](https://arxiv.org/abs/2609.00313) | S2 cited-by(Arnaout) | intellectual roots 检索基准；修辞角色层（LLM judge 与人工 κ=0.896）对照作者自认的启发来源层：二者几乎无关（κ=0.037） | 前瞻切分 | 否 | 检索基准 | 假，但对叙事是警示：施引句**怎么写**与施引者**实际受了谁的影响**是两回事。他述作为"社区认识"成立，作为"影响事实"不成立 |
| Nguyen, Pruski, Da Silveira 2026, Scientometrics："Deepening citation understanding … LLM-powered context extraction" | DOI [10.1007/s11192-026-05637-7](https://doi.org/10.1007/s11192-026-05637-7)（Springer 摘要，CDP） | DDG(CDP) | 改进引用语境抽取，并用 LLM 做结构化提示的语义解读，面向数字图书馆和 KG | 否 | 半：引用级语义表示 | 下游 KG | 假 |
| Liu et al. 2026, EMNLP 2026："Citing Less Critically" | [arXiv 2609.01432](https://arxiv.org/abs/2609.01432) | DDG(CDP) | LLM 生成的引用句比人类更少批评，更偏向老的热门论文 | 否 | 否 | 否 | 假。如果他述越来越多由 LLM 代写，"他述反映社区认识"这个前提会被稀释，2025 年之后的语料要注意 |

## 3. 判决：已被占 / 空位

### 已被占（不能当卖点，只能当动机或引用）

- **"他述不同于自述，他述代表社区认识"**：Mei & Zhai 2008、Elkiss 2008、Qazvinian & Radev 2008、Divoli 2012、CL-SciSumm、ScisummNet、Chen 2025、SECite 2026。
- **"对一篇论文的描述随时间变化"作为现象和测量**：Divoli 2012（年份分组）、He & Chen 2018（单篇论文的角色变化分数）、Lin et al. 2025（群体层面转向符号性引用）、KCT 2026（单篇论文贡献类型轨迹）、Li 2020（DSM 被确立的语言标记）、McCain（OBI 时间趋势）、Hsiao & Schneider（撤稿前后）。
- **"用 LLM 把带年份的他述汇总成单篇论文的时间感知描述"**：Arnaout et al. 2025/2026。
- **"主张被限定/被推翻的时间分析"**：ClaimFlow。
- **"自述加他述双视角表示，供 agent 使用"**：AgentExpt（窄对象：baseline/数据集）；scite MCP（粗语义：三分类）。
- **"帮初学者快速进入新领域"**：Qazvinian & Radev 2008 和 Mohammad et al. 2009 的原始动机。
- **认识变迁事件的各个类型都有概念先例**：被重新归类（Cozzens split citation identity）、范围被收窄（ClaimFlow qualify）、暴露新局限（Abu-Jbara 2011 的 Limitations 槽、ClaimFlow 的 BLEU 个案）、被替代（Intern-Atlas replaces，见前几轮报告）、被吸收或变成默认组件（Merton/McCain 的 OBI、Meng et al. hidden citations、Hargens、Lin et al.、Li DSM）。

### 空位（在已核实范围内没有找到）

1. **领域级、结构化的他述汇总**。现有汇总要么是单篇论文的自由文本（Arnaout、SECite、引用摘要一脉），要么是计数或分布（scite、KCT、Chen），要么只针对窄对象（AgentExpt 的 artifact，ClaimFlow 的主张）。没有人把他述汇总成"方法族与成员、每个工作的定位、共同局限、比较对象"这样的领域状态。
2. **类型化的认识变迁事件，作为可检测、有日期、有证据的对象**。各类型都有先例，但没有一项工作定义一套变迁事件类型学，在单篇论文或方法层面自动检测并评测。ClaimFlow 只有关系类，没有事件类；He & Chen 只有距离；KCT 只给出 4 个案例的分布变化。
3. **自述与他述的显式逐项对照**，作为每条论文记录的字段（作者说自己提出了 X，后来者把它归为 Y 类、指认局限 Z）。Chen 2025 只在类型分布层面对照，ScisummNet 只是把两者混合进一段摘要。
4. **按截止时间 T 作答的编译认知**，给 agent 当工具用，并评测下游收益。Asta 和 scite 已经能按日期过滤原始引用，但没有截至 T 的编译状态；Arnaout 的摘要不支持截止查询，且自报年份归属准确率只有 0.59。
5. **他述的可信度处理**。Greenberg、Jergas & Baethge（25.4%）、Simkin（约 20% 读过原文）、MUSES（κ=0.037）都说明他述会走样、会转抄、与实际影响无关。现有汇总系统都直接把他述当事实。区分"被广泛这样描述"与"这样描述有原文支撑"可以成为差异点，但引文核验本身是活跃方向（如 [arXiv 2608.30145](https://arxiv.org/abs/2608.30145) CNCV），只能作为组件，不能作为主卖点。

### 对叙事的直接含义（推断，供讨论）

- 叙事第一句话要改写。"一篇论文的位置由后来的论文不断重新表述"可以作为立场，但必须紧接着引用 Small 1978、Mei & Zhai 2008、Divoli 2012，然后说明"已有工作停在单篇摘要、计数或个案轨迹；我们把它编译成可查询的领域状态"。
- 最强的可辩护贡献组合是"空位 1 + 2 + 4"，并配上对 Arnaout（单篇时间感知摘要）、scite MCP（agent 三分类）、AgentExpt（双视角 artifact profile）三者的正面对比实验。
- 评测可以借：变迁事件用 ClaimFlow 的 qualify/refute 标注当部分 gold；单篇层面与 Arnaout 的 105 篇比较（数据和代码已开放，需核许可）；"变成默认组件"用 McCain/Meng 的无引用提及做对照；按截止时间作答的泄漏控制参考 AgentExpt 的时间切分和前几轮报告中的 ForeSci。
- "变成默认组件 / 被吸收"这一类在引用语境里天然不可见（OBI），只靠他述抽取会系统性漏掉。要么承认它是盲区，要么加入无引用提及检测。

## 4. 最接近的 5 篇及差别

| 排名 | 工作 | 它做到的 | 它没做到的 |
|---|---|---|---|
| 1 | Arnaout et al.（2505.14838） | 带年份的细粒度他述 → LLM 生成单篇论文时间感知影响摘要；专家评测 | 结构化；领域级与方法族；自述对照；变迁事件分型；按截止时间作答；agent 使用 |
| 2 | ClaimFlow（2603.16073） | 主张级 qualify/refute 的时间分析与人工 gold；BERT/BLEU 的轨迹个案 | 论文或方法的定位画像；领域状态；查询接口；事件类型学 |
| 3 | KCT（2608.20697）与 Chen et al. 2025 | 他述的贡献类型分类；自述与他述类型分布的差距（ideal bias）；单篇论文早晚期贡献分布的 JSD | 汇总成认知；局限与比较对象；变迁检测；下游 agent 使用 |
| 4 | AgentExpt（2511.04921） | 自述 + 他述汇总 profile → agent 检索，消融收益显著，时间切分 | 对象仅限 baseline/数据集；无时间维和变迁；无领域结构 |
| 5 | scite MCP | 面向 agent 的"后来者支持还是反驳"，日期过滤，长期监控 | 只有三分类计数；无定位、族、局限；无变迁事件 |

次近邻：He & Chen 2018（单篇论文角色变化的连续度量）、Divoli 2012（随年份偏离自述的实证）、Mei & Zhai 2008（论点原句）。

## 5. 未核实清单（不进主表）

- **Merton 原始的 obliteration by incorporation 出处**（*Social Theory and Social Structure* 1968 版，以及 *On the Shoulders of Giants* 1965）：未打开一手，只通过 McCain 与 Meng et al. 的转述确认概念。引用时需要回原书核页码。
- **Small 2011, Scientometrics："Interpreting maps of science using citation context sentiments"**（DOI 10.1007/s11192-011-0349-2）：书目元数据已核，摘要未拿到，内容未核。
- **Small, Tseng, Patek 2017, J. Informetrics："Discovering discoveries"**（DOI 10.1016/j.joi.2016.11.001）：元数据已核，ScienceDirect 触发验证码，内容未核。
- **Gou et al. 2022, Scientometrics："Encoding the citation life-cycle"**（DOI 10.1007/s11192-022-04437-z）：元数据已核，无摘要，内容未核。
- **Meng et al. "Hidden Citations" 的正式发表处**：arXiv 已核；PNAS Nexus 2024 只是记忆线索，未核。
- **Arnaout et al. 的正式发表会议**：arXiv v2 无 comments 字段，未核实是否已被会议接收。
- **"Teufel et al. 2006 认为引用文本不适合做摘要"**：转述自 Mohammad et al. 2009 摘要，Teufel 原文中的这一论断未核。
- **Chengzhi Zhang 组的方法实体演化工作**（情报学报 2023，DOI 10.3772/j.issn.1000-0135.2023.08.007）：只读到摘要开头，未核是否涉及引用语境中的描述变化。
- **2506.12242（LLM for HPSS）、Cohan & Goharian 2017（1705.08063）、"Insights from CL-SciSumm 2016"（Springer IJDL）**：只见标题，未打开。
- **Nakov, Schwartz, Hearst 2004 的 citances 原始论文**：未查。
