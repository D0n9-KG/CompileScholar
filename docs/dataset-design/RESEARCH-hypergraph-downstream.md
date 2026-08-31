# 超图下游应用策略调研报告（2024-2026）

来源：调研 agent（14 篇 arxiv ID 逐字核验）

## 关键工作的下游使用策略

### 1. HyperRAG (2602.14470) — n-ary facts reasoning，双检索变体
- **HyperRetriever（学习型）**：query→抽 topic entity→取关联超边→展开成伪二元三元组→DDE 结构邻近编码（双向传播）→MLP 似然打分→自适应阈值搜索；context 按 50%超边/30%实体/20%源文本分配
- **HyperMemory（LLM 引导）**：beam search（宽3深3），LLM 直接对超边打分→对 tail entity 打分→组合保留 top-w
- 优势：+2.95% MRR；复杂度论证 n-ary 超图检索比二元 KG 更高效；证明任何二元 reduction 必违反 recoverability/role/multiplicity 三者之一

### 2. HyCE-RAG (2607.22597) — 置信度在 incidence 矩阵上传播
- 离线：LLM 抽 (V,E,A)，每条超边带抽取置信度 γ(e)+有向关系集 R(e)
- 在线：双路入口实体（精确匹配+语义检索）→超边候选集→query-aware 超图组装（结构重叠比+关系相似度）→**incidence 矩阵置信度传播**（HGNN 式 vertex↔hyperedge message passing）→六维打分（覆盖/传播/抽取置信/入口/关系可靠度）
- **唯一显式处理证据冲突**：低置信路径显式喂 LLM 暴露不确定性
- 优势：accuracy+relevance+faithfulness 三维一致超基线

### 3. PRoH (2510.12434) — 动态规划 DAG + Entity-Weighted Overlap
- 同义超边增强（实体名 embedding cos≥τ→同义超边）提升连通性
- **Plan Context Graph**：受控探索 sketch 超图结构喂 LLM 生成 plan，避免臆造子问题
- reasoning=DAG 状态搜索；**每个子问题重新锚定**（比原始更局部）
- **EWO（Entity-Weighted Overlap）**：超边在多实体上重叠时，聚合每个重叠实体对子问题的 question-specific 相关度（普通图边只共享1节点，超边可共享多节点）
- 优势：+19.73% F1 over HyperGraphRAG；长程多跳（3-6 hop）鲁棒

### 4. OG-RAG (2412.15235) — Schema/Ontology-grounded（★ 最直接对应 schema 层）
- schema=ontology 三元组 (s,a,v)；LLM 把文档映射到 ontology→递归展平嵌套→每个展平块=一条超边，hypernode 是 key-value 对
- **schema 直接决定超边形状**
- **检索零 LLM 调用**：相关节点 top-k→**最小超边覆盖**（贪心解 matroid 约束→最优）→字典格式 context
- 优势：+55% 召回、+40% 正确性、+30% 归因速度、+27% reasoning；查询时零 LLM

### 5. Higher-Order Knowledge (2601.04878) — 科学领域，schema 即 Pydantic
- schema=类型系统：Event{source:List, target:List, relation:str} 强制类型一致
- **超图最短路径**：倒排索引（节点→含它的超边）；BFS 在超边上走，相邻超边须共享 ≥S 节点（IS 参数）；Yen k-shortest
- 超图拓扑作为**可验证护栏**（verifiable guardrail）；teacherless agentic reasoning

### 6. HyperAgent (2608.02650) — Tool-Schema Hypergraph（schema 层超图最纯粹案例）
- 工具=从 input-schema 节点到 output-schema 节点的有向超边；schema 层超图，节点是 schema 字段
- **deficit-oriented expansion**：识别未满足要求→按当前 state 检索"生产者工具"

### 7. Hyper-Align (2605.21858) — 超图如何喂 LLM 的理论基础
- **HIDT-O**：vertex-hyperedge 严格交替的固定形状 incidence 树；incidence bipartite 表示无损
- **HIP**：语义-结构解耦（semantic core + role 映射）；Hyper-Incidence Block 做 vertex↔hyperedge 双向 message passing

### 其他
- OKH-RAG (2604.12185)：顺序作为超图一阶结构属性，序列推断
- HyperGraphPro (2601.17755)：RL + structure-aware 超图检索
- HGRAG (2508.11247)：细粒度实体=节点、粗粒度 passage=超边；diffusion 整合
- NS-HART (2503.20676)：n-ary semantic hypergraph 子图提取辅助 inductive link prediction
- DocTrace (2606.10921)：query-triggered 按需构造超图工作内存 + 经验复用
- EHRAG (2604.17458)：结构超边+语义超边混合；线性索引复杂度

## 超图 vs 二元 KG 已验证优势
| 优势 | 工作 | 数值 |
|---|---|---|
| 保留 n-ary 不可分解性 | HyperRAG | +2.95% MRR +复杂度论证 |
| 更短推理路径 | HyperRAG | per-output O(1) 当 arity 有界 |
| 结构重叠建模 | PRoH | +19.73% F1 |
| incidence 矩阵置信度传播 | HyCE-RAG | 三维一致超基线 |
| schema 决定超边形状 | OG-RAG | +55% recall / +40% correctness |
| schema 字段做超边 | HyperAgent | 减冗余 API/token |
| 无损序列化喂 LLM | Hyper-Align | in-domain + zero-shot 双超 |
| 低度-高度概念桥接 | Higher-Order | 成功连通案例 |

**无任何论文在结构检查（SHACL 式确定性规则）上比较超图 vs 二元 KG** —— 优势全在表达力/检索效率/LLM 喂入无损性。

## SOTA pipeline 典型形状（5 步）
1. 离线超图构造：LLM 抽实体+超边+incidence；或 schema 驱动展平（OG-RAG）；每条超边带 provenance/置信度/有向 relation
2. 问题→入口锚点：LLM 抽 topic entity + 语义检索 top-K（双路）
3. 候选超边选择：入口关联超边 + 多跳邻域扩展；结构项（重叠比/EWO）+ 语义项混合打分
4. 超图推理（四范式）：学习型路径检索（HyperRAG）/ incidence 置信度传播（HyCE-RAG）/ DAG 规划搜索（PRoH）/ 最短超路径（Higher-Order）
5. 结构化 context + LLM 生成：超边→实体→源块按预算填充；HyCE-RAG 分离支持/低置信路径暴露冲突

## 可迁移到两层结构的策略

### A. Schema 层指导 instance 检索（已验证）
1. **OG-RAG ontology-grounded 构造**：schema 三元组递归展平成超边，hypernode 是 key-value 对；schema 直接定义 instance 超边字段结构
2. **OG-RAG matroid 最小覆盖检索**：确定性问题→相关 schema 字段→贪心最小超边覆盖，零 LLM、可证明最优
3. **HyperAgent deficit-oriented expansion**：schema 超图做"未满足要求→检索生产者"逆向路由
4. **Higher-Order Pydantic schema 强制类型**：schema 即类型系统，抽取阶段强制一致

### B. Instance 层 verbatim evidence 作超边 provenance
- HyCE-RAG/PRoH/HyperRAG 都把源 chunk 作超边 provenance；verbatim evidence span 直接套这个槽位

### C. Schema 层 dep/con/comp 边做推理路径约束
1. **Higher-Order IS 约束**：相邻超边须共享 ≥S 节点才允许 BFS 转移；dep 边定义"哪两个 pattern 必须共享哪些 anchor 字段"
2. **PRoH EWO**：超边重叠按 question-specific 相关度聚合；dep 边权重可加权聚合
3. **HyCE-RAG incidence 置信度传播**：schema 和 instance 的 incidence 矩阵拼接，置信度跨层传播，schema dep/con/comp 边作先验注入 instance 初始置信度

### D. 两层联合 QA pipeline（综合最可迁移组合）
1. 离线：schema 层（OG-RAG ontology 三元组/Higher-Order Pydantic）+ instance 层（n-ary 超边带 verbatim evidence）
2. 问题锚定：HyCE-RAG 双路（LLM 抽实体+语义检索）→入口实体集
3. Schema 层路由：OG-RAG matroid 最小覆盖选相关 schema pattern→HyperAgent deficit-oriented 沿 dep 边回溯缺什么→schema pattern 作约束传 instance 检索
4. Instance 层推理：schema pattern 约束下，PRoH EWO + 子问题重新锚定 + DAG 规划；或 HyperRAG 学习型似然打分
5. 生成：HyCE-RAG 结构化 context（支持+低置信分离），超边带 verbatim evidence span 喂 LLM

## 关键观察与风险
1. **无任何工作显式做两层（instance+schema）超图** —— OG-RAG 最接近但 schema 只是抽取约束不在检索时跨层路由；HyperAgent 是 schema 层单一超图。**真实方法论空位**，但需自证"两层比单层有收益"
2. **schema 在检索阶段参与度普遍低** —— OG-RAG/HyperAgent 把 schema 用在构造阶段；HyCE-RAG/PRoH/HyperRAG 检索阶段几乎不用 schema。**schema-guided retrieval 是更空的位**
3. **超图 vs 二元 KG 优势全部依赖"超边一次连多实体"** —— 无"超图在结构检查/确定性约束上更强"的证据。若下游收益来自确定性 schema 约束，超图表达力优势不一定能迁过去
4. **arxiv ID 必须直查** —— WebSearch 虚构 ID（2502.06959/2505.09785 实为量子计算/SUSY 暗物质）
