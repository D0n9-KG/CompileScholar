# Schema/Ontology-Guided 检索与 QA 调研报告（2024–2026）

来源：调研 agent（2024-2026 arxiv 逐字核实）

## 1. schema 使用机制聚类

### 1.1 Schema 约束提取 + 双通道/层次化检索（GraphRAG 后继谱系）
- **OMD-GraphRAG** (2603.25152)：预定义 schema 注入 extraction prompt；双通道图检索（graph + community）
- **Youtu-GraphRAG** (2508.19855)：seed graph schema 约束 extraction agent + 持续扩展适配未见域；**同一 schema 被两处复用**——extraction + agentic retriever 分解查询为 sub-queries；token 省 90.71%，acc +16.62%
- **HCG-RAG** (2607.22592)：schema-constrained causal graphs；消融 causal graph 隔离为 structured retrieval filter 相对 embedding-only +6pp
- **FAIR GraphRAG** (2607.11464)：通过 ontology links 把 metadata 和语义关系作为可检索结构

### 1.2 Schema 作为 query-time routing contract
- **Executable Schema Contracts** (2606.05415)：schema 作为 KG 构建 + 查询检索的共享契约；schema-conditioned multi-tool agent routing（structured lookup / graph traversal / vector search）
- **SCAIR** (2607.22571)：注入 schema-conditioned structural priors + 强制 schema-aware traversal；无需 retraining

### 1.3 Ontology-guided query expansion / IR
- **BMQExpander** (2508.11784)：UMLS 定义+关系 + LLM 做 query expansion；perturbation 下比监督 baseline 鲁棒 +15.7%
- **Ontology slice injection** (2607.28662)：ontology slice 按嵌入相似度 live retrieved 注入 extraction prompt

### 1.4 Schema-guided LLM 推理骨架
- **STEM** (2604.22282, ACL 2026 Oral)：多跳推理表述为 schema-guided graph search；Semantic-to-Structural Projection 把 query 分解为 atomic relational assertions 构造 adaptive query schema graph；Triple-Dependent GNN 生成 Global Guidance Subgraph
- **SGR** (2606.04454)：抽取 entities/relations/constraints → 构造 structured schema → schema-guided querying 从 KG 检索紧凑 subgraph → 引导 LLM 分步推理；消融：schema guidance 和检索都不可或缺
- **Neuro-Symbolic** (2602.17826)：★ 双刃——ontology-guided context 在检索质量高时提升、无关时主动伤害
- **ORT** (2502.11491)：基于 ontology 构造 label reasoning paths 指导检索
- **OntoSCPrompt** (2502.03992)：ontology-guided hybrid prompt learning

### 1.5 Type-constrained / schema-constrained reasoning
- **OPI** (2606.28076)：relation-centric ontology graph 捕获 head-tail type constraints；bidirectional retrieval（answer type → compatible final-hop relations）；iterative refinement 过滤 type-compatible 但 question-irrelevant 证据
- **CypherBench** (2412.18702)：RDF schema 过大 + resource identifiers + 重叠 relation → LLM 难用；property graph views

### 1.6 Meta-level / hypergraph / self-evolving schema
- **HyperSU** (2606.28351)：semantic-unit hyperedges via MDL 优化；clue-guided bidirectional expansion over semantic-unit hypergraph
- **OMAGR** (2606.11910)：query 分解为 ontology-aligned anchors；每维并行 graph retrieval；解决 multi-dimensional retrieval bottleneck
- **LLM-Wiki** (2605.25480)：retrieval-as-reasoning；self-evolving structured Wiki + Error Book 持久自校正；超 HippoRAG 2/LightRAG 2.0-8.1 F1
- **Ψ-RAG** (2605.00529, ICML 2026)：iterative merging/collapse 层次抽象树索引；比 RAPTOR +25.9%、HippoRAG 2 +7.4%
- **ExBI** (2603.10625)：hypergraph data model + Source/Join/View 算子支持 dynamic schema evolution
- **DIAL-KG** (2603.20059)：schema-free incremental KG + dynamic schema induction + evolution-intent assessment

### 1.7 Schema 通过后训练内化
- **CRAFT** (2607.22642)：反对 schema-stuffed prompt；schema-stripped PLAN SFT + execution-shaped RL；token 减 9×，schema-discovery loops 减 5×

## 2. schema 验证优势点（相对无 schema）
| 论文 | 证据 |
|---|---|
| HCG-RAG | structured retrieval filter +6pp vs embedding-only |
| OPI | type constraint 抑制 mixed-type expansion，Hit@1 +4.6/+8.9 |
| STEM | schema-guided graph search 提升准确率 + 证据完整性 |
| SGR | 消融：schema guidance 和检索都不可或缺 |
| Youtu-GraphRAG | 同 schema 复用 extraction+retrieval：token −90.71%, acc +16.62% |
| Executable Schema Contracts | 消融：routing/structural/guidance 各自贡献 |
| BMQExpander | NDCG@10 +22.1% vs sparse；perturbation 鲁棒 +15.7% |
| Neuro-Symbolic | ★ 双刃：无关 context 主动伤害 |
| OMAGR | multi-anchor 并行检索提升 Context Precision/Faithfulness |

## 3. SOTA pipeline 结构（schema 多触点复用）
```
① 构建期：schema 约束提取（slice 注入 / predefined types / 蒸馏词汇表）
② 组织期：schema 驱动 community/hierarchy（schema-aligned community / 两层图）
③ 查询期：schema-conditioned query decomposition + routing（anchors → 并行检索 / sub-queries / multi-tool routing / bidirectional）
④ 推理期：schema 作为推理骨架（atomic assertions → adaptive query schema graph / label paths / subgraph evidence）
⑤ 过滤期：type constraint 过滤（抑制 mixed-type / 过滤 irrelevant）
⑥ 演化期：schema 自进化（seed 扩展 / evolution-intent / Error Book）
⑦ 内化期：后训练进权重（非 prompt 注入）
```
关键趋势：**同一 schema 被多处复用**（extraction + retrieval + decomposition + routing + filtering），而非只在一处。

## 4. 可迁移到我们的策略（高度可迁移）
1. **Schema-aligned query decomposition → 并行 per-pattern retrieval（OMAGR）** — 我们的 meta-hypergraph 有 pattern 间边，可把 query 分解为 pattern-aligned anchors 沿 schema 边并行检索 instance 超边
2. **Bidirectional retrieval via answer-type → compatible relations（OPI）** — 我们的 n-ary 角色约束更丰富，可做 role-level type constraint
3. **Atomic relational assertions + adaptive query schema graph（STEM）** — 构造查询时的小型 schema subgraph 去 anchoring instance 超边；depends_on 即 guidance 边
4. **Schema 作为多工具路由 contract（Executable Schema Contracts）** — schema condition agent 在 structured lookup / graph traversal / vector search 间路由
5. **Schema slice live-retrieved 注入 prompt（2607.28662）** — 不塞全 meta-hypergraph，按 query 相似度检索相关 schema slice
6. **Schema-conditioned traversal enforcement（SCAIR）** — 多跳 traversal 只走 schema 允许的 pattern 间边
7. **Semantic-unit hyperedges via MDL（HyperSU）** — MDL 优化可作 hyperedge 质量指标；clue-guided bidirectional expansion 适配超边
8. **Reverse thinking via ontology label paths（ORT）** — pattern 间 depends_on 边构造 reasoning paths

## 差异化定位（我们的 SOTA 缺口）
无任何工作同时具备：(a) schema 层是 meta-hypergraph（pattern 间多类语义边 dep/con/comp）+ (b) instance 层是 n-ary 超边带 verbatim evidence + (c) schema 自进化且轨迹可追溯。

卖点：**meta-hypergraph 的 pattern 间语义边作为多 anchor 并行检索 + traversal 硬约束 + 演化顺序仲裁**——三者已有工作各占其一，无人合一。
