# RESEARCH: step3 把引用关系作结构信号注入 lift_corpus

Date: 2026-08-14
Goal step: 阶段顺序 step 3 — lift_corpus 用引用关系作结构信号升层，验证 vs 纯文本

## 现状（已摸清）
lift_corpus(meta, edges_by_paper):
  1. cluster_methods_by_llm(edges_by_paper) → {method_label: [pooled edges]}
     每条 edge 带 `paper` 字段（论文名）。一篇论文的 edges 可能被分到多个方法族
     （granular 论文常含多个方法，如 Midi_2004 同时讲 μ(I) 和 segregation）。
  2. lift_into_schema → 对每方法族 induce_method_node；对每对 judge_relation。
     judge_relation 当前**只用文本互提**（_cross_mention：A 的 edges 的 evidence 里
     出现 B 的核心量关键词）。建模形式/适用范围判断已加（commit 99c83a40）。
  3. 写进 meta schema（frozen 不调=不长高阶）。

引用关系底座已就位：corpus_driver.load_references(paper_id) → sci-evo API。
ARFM2024 的 21 篇 lift eval 论文有文件名=规范化标题，但 paper JSON 无 DOI/year，
可借 OpenAlex title 查引用关系（侧A fetch_work_references(doi) 或 search-resolve）。

## 核心设计难点：引用是论文级，judge 是方法级
A 方法族含多篇论文 {Pa1, Pa2,...}，B 方法族含 {Pb1, Pb2,...}。
若 Pa1 引用 Pb2 → 这是 A extends/improves/background B 的**强结构信号**，
但不直接等于方法关系（一篇论文含多方法；引用可能是背景而非方法继承）。

## 三个候选方案（不静选，待定）

### 方案1：引用作 judge 的先验证据（最小侵入）
judge_relation 增加 `citation_evidence` 参数：A 族论文引用 B 族论文时，
prompt 注入"A 所在论文 X 引用了 B 所在论文 Y"作 extends/background 倾向先验。
LLM 仍综合建模形式/适用范围判断，引用是加分项。
- 优点：最小改动，引用信号不override建模判断。
- 缺点：论文级→方法级映射靠"族内任一论文引用即算"，可能噪声。

### 方案2：引用作 candidate 生成 + 双信号融合
引用关系先生成候选关系对（A 引 B → 候选 extends/improves/background），
文本互提生成 compares 候选。两路候选合并去重，judge 各自判 + 置信度加权。
- 优点：引用信号显式可控，能做消融（关引用看掉多少 extends）。
- 缺点：改动大，候选生成逻辑新写。

### 方案3：引用关系建方法级 ground truth，不作抽取信号而是作评测标尺
引用是论文级真值（A 引 B 客观存在），用它验证抽取出的 extends/improves 是否
覆盖了真实引用链。即引用不进 judge，进 evaluator。
- 优点：评测独立不循环（引用是外部客观事实）。
- 缺点：引用≠方法继承（A 引 B 未必 extends B），作标尺有噪声。

## A/B 对照实验设计（任何方案都需）
- arm A = 纯文本 lift（当前 lift_corpus）
- arm B = 文本 + 引用结构信号
- 评判：gold 关系覆盖（ARFM2024 的 GOLD_RELS，6 条 extends/improves/compares）
  + 引用链召回（A 引 B 的论文对里，B 臂是否更多升出 extends/improves/background）
- 同 LLM 受控（纪律：被测臂 deepseek / judge GLM-5 / gold GLM-5.2）

## 待用户拍板
1. 方案1/2/3 选哪个？（我倾向方案1最小侵入+方案3作标尺，引用既进judge又进eval可能循环）
2. 引用数据取法：用 ARFM2024 论文标题走 OpenAlex 查引用（侧A），还是只用手头
   已有 references 的 sci-evo 库论文（PPR_6AB55969EFBC 那30条，但和ARFM2024不重叠）？
3. ARFM2024 论文命名是 Midi_2004 等，要接 paper_registry 需先 register（title→DOI→references）。
   是否现在就给这21篇建引用数据？
