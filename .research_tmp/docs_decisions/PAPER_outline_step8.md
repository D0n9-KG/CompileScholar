# 论文大纲：自演化Schema与n-ary超图的低阶升层长高阶系统

step8 成文大纲（基于 step1-7 已验证机制+评测）

## 标题（候选）
Self-Evolving Schema with N-ary Hypergraph: Lifting Low-Order Relations to
Higher-Order Method Evolution via Structural Signals

## 摘要要点
- 问题：科学论文抽取的 n-ary 超图低阶关系如何系统化长出高阶方法演化关系（extends/improves/compares/background），不靠大规模标注。
- 方法：(1) 自演化 schema 五操作+升层第六（frozen 不调=不长高阶）；(2) n-ary 超图（多节点+role+qualifiers+evidence）；(3) 低阶升层用结构信号全利用（引用关系作 judge 先验 + edge quals 证据构成），不止文本归纳；(4) 论文注册底座闭环（DOI/title→原文+超图+引用）。
- 实验：消融5臂证明升层必需（frozen 0/3 gold）；结构信号更精准（improves +67%~167%，extends 11→5 抑制过度）；综述复杂问题 frozen 0/4 vs full 2/4；SciFact 诚实域边界。
- 贡献：自演化升层长高阶（6邻近工作不做）+ 结构信号全利用 + 可溯源（每关系有 evidence）。

## 1 Introduction
- 科学方法演化建模需求（方法继承/改进/对比是科学进步的核心结构）
- 现有局限：IncSchema 直接 probe 不从低阶升；AutoSchemaKG 文本驱动无自进化；Hyper-KGGen 测抽取质量不测 schema 路由
- 贡献声明：自演化升层第六操作 + n-ary 超图结构信号全利用 + 论文注册底座 + 颗粒流综述 gold 评测

## 2 Related Work
- Schema induction: AutoSchemaKG, IncSchema, ASEE
- n-ary KG: Hyper-KGGen, DIAL-KG
- Method evolution: Intern-Atlas (综述 gold 代理)
- 差异化：6工作不做自演化升层（表1对比）

## 3 Method
### 3.1 自演化 Schema + 五操作 + 升层第六
- add/split/merge/retire/rename + lift_into_schema（写进 meta，frozen 不调）
### 3.2 n-ary 超图抽取
- extract_hypergraph: map_structure→DAG→pattern matching→n-ary edge (nodes+roles+quals+evidence)
- IncSchema 防幻觉四措施：retrieval-augmented + decomposed verification + log_probability + strict admission
### 3.3 低阶升层长高阶（核心1）
- cluster_methods_by_llm（LLM 分族，embedding 失败因方法≠主题）
- induce_method_node + judge_relation
- **结构信号全利用**（核心创新）：
  - 引用关系作 judge 先验（A引B→extends/improves/background 倾向，先验非硬规则）
  - edge quals 证据构成（evidence_strength derived/measured + cited_from prior_art + method theory/sim/exp）
  - （后续）节点共现/拓扑/cited_from 方法名
### 3.4 论文注册底座闭环（核心2）
- paper_registry HTTP client + corpus_driver 双轨 + sci-evo external_artifacts 存储
- DOI/title→论文→原文MinerU→超图LogicKG→引用，paper_id 统一键

## 4 Evaluation
### 4.1 消融（模块4，核心证据）
- 5臂：F frozen / T text / C +citation / Q +quals / B +full
- F 0/3 gold 证明升层必需；citation improves +67%；quals improves +167%；B full 最精准保守
### 4.2 综述复杂学科问题（模块1，创新评测）
- 需高阶 schema 才答的方法关系问题；frozen 0/4 vs full 2/4
### 4.3 全量结构 vs 文本 A/B（step5）
- improves 5→9（识别真继承），extends 11→5（抑制过度，区分并行/背景）
### 4.4 公开数据集 SciFact（模块2，诚实域边界）
- hypergraph 0.250 vs naive-RAG 0.333——诚实报告域不匹配
### 4.4b 公开数据集 SciREX（模块2b，强结果）
- SciREX n_ary_relations {Method/Material/Metric/Task} 对应我们 n-ary 超图
- Method 召回 1.000 (14/14)——n-ary 超图抽取质量高
### 4.5 SOTA baseline 公平 PK（模块3）
- IncSchema-style direct-probe (同 deepseek 同 ARFM2024 受控) vs ours-lift
- direct 2/3 gold (缺 improves) vs ours 3/3；improves 0 vs 8——低阶升层比直接 probe 高阶更准
- NMR 评测器（Intern-Atlas 式，NMR=0.700）

## 5 Discussion
- 诚实域边界：方法关系推理（颗粒流）价值大；事实核查（SciFact）无优势
- 结构信号互补：引用先验给继承倾向，quals 校正过度 extends
- LLM 非确定性：重复跑波动，需多次取均值（future work）
- judge 并行化工程贡献

## 6 Conclusion
- 自演化升层长高阶 + 结构信号全利用 + 可溯源，在颗粒流域实证有效

## 实验证据汇总表
| 实验 | 指标 | 结果 | 结论 |
|---|---|---|---|
| 消融5臂 | gold覆盖 | F 0/3, B 3/3 | 升层必需 |
| 消融 | improves | T3→C5→Q8 | 结构信号提升继承识别 |
| 消融 | extends | 11→5(B) | 抑制过度 extends |
| 复杂问题 | 准确率 | frozen 0/4, full 2/4 | 升层提升方法关系推理 |
| 全量A/B | improves/extends | 5→9 / 11→5 | 结构信号更精准 |
| SciFact | SUPPORT acc | 0.250 vs 0.333 | 诚实域不匹配 |
| SciREX | Method 召回 | 1.000 (14/14) | n-ary 抽取质量高 |
| baseline PK | gold/improves | direct 2/3(0) vs ours 3/3(8) | 低阶升层比直接 probe 更准 |
| NMR | 节点匹配 | 0.700 | 语义匹配可行 |

## 诚实声明
- SciFact 结果不利但诚实报告
- 小样本（N=4/12）需扩规模
- LLM 非确定性需多次均值
- baseline PK 部分用消融模拟（同LLM受控）+文献对比
