# 创新性重新审视（基于代码深读 + 广泛调研交叉）

## 我们的 9 个技术特征 vs SOTA（交叉核实）

| # | 我们的技术特征 | 最接近 SOTA | 被占? |
|---|---|---|---|
| 1 | pattern 间非树富拓扑(dep/con/comp) | HyperAgent 工具域有 schema-level dependency；KG 域**全空**（方向4确认） | **KG 域空位** |
| 2 | pattern-level split（父变 abstract + fallback） | DIAL-KG 有 schema 演化但不列操作；SCOPE 有 merging/filtering；**无 split + abstract fallback** | **空位** |
| 3 | 约束违反检测（detect_constraint_violations） | 无工作做 KG schema 自身结构错误检测 | **空位** |
| 4 | hyper-relational instance（n-ary + qualifier + verbatim） | HyperRAG/HyCE-RAG 有 n-ary；LineageRAG 有 verbatim；**组合无** | **组合空位** |
| 5 | intra-DAG forward propagation | DIAL-KG 有反馈循环但跨轮非 intra-pass；EDC post-hoc；**无 intra-DAG** | **空位** |
| 6 | abstract pattern 作 validate fallback | ontology 有 subClassOf 但无 fallback 校验 | **空位** |
| 7 | 多标签节点 | 普遍 | 被占 |
| 8 | bounded op + conservative gate | DIAL-KG 有 governance 但不形式化操作集 | **部分空** |
| 9 | LLM 仅命名不判断 | 无工作明确做此分离 | **空位** |

## 关键发现：5 个真正空着的组合

广泛调研确认的 5 个空位（方向1-5）：

1. **★ schema 层自进化 + instance 层是 n-ary 超边** — DIAL-KG 的 schema 演化是二元三元组的；EvoGraph-R1 超图演化但 schema 不演化；Text2NKG 有超图 schema 但不演化。**这个交叉无人占。**

2. **★ 超边 pattern 间的 dep/con/comp 关系** — KG schema 层无 pattern 间关系建模（方向4 全空）。HyperAgent 在工具域做了但不是 KG relation。

3. **schema-guided 抽取 + schema-guided 检索统一框架，schema 在两边角色不同** — FlexStructRAG 最接近但 schema 非诱导、两边角色无区分。Youtu-GraphRAG 做了统一但 schema 是普通 entity/relation/attribute 三元组。

4. **meta-level schema reasoning（pattern 层推理）** — 方向5 基本全空。

5. **schema 演化的形式化操作集（add/split/merge/retire/rename + 触发条件）** — DIAL-KG/SCOPE 都未形式化操作集。

## 诚实定位

### 真正的创新点（代码实际有 + SOTA 确认空位）

**最强候选：自进化 n-ary 超图 schema（特征 #1 + #4 + #8 + #9 的组合）**
- schema 层有 pattern 间 dep/con/comp 语义边（KG 域全空）
- instance 层是 n-ary 超边带 verbatim evidence（组合无）
- schema 自进化且有形式化操作集 add/split/merge/retire/rename（DIAL-KG 不形式化）
- LLM 仅命名不判断，决策是确定性聚类（无工作做此分离）
- intra-DAG forward propagation（无工作做 intra-pass 演化）

**但这是"组合型"创新**——每个单点可能有人部分做了，组合无人占。风险是审稿人认为 incremental。

### 要让创新成立，必须证明的

1. **dep/con/comp 多类语义边比单一 ontology 边有可量化优势**（在检索或约束检查上）
2. **n-ary 超边 + verbatim 比 proposition + paraphrase 在 citation grounding 上更准**
3. **自进化 schema + 形式化操作集比 DIAL-KG 式隐式演化收敛更好**（可控性）
4. **intra-DAG forward propagation 比 post-hoc 演化抽取质量更高**

### 诚实风险

- Youtu-GraphRAG 已占"schema 自进化 + schema-guided 抽取 + schema-guided 检索"的大框架
- DIAL-KG 已占"schema↔instance 反馈循环"
- HyperRAG/HyCE-RAG 已占"n-ary 超图检索"
- 我们的差异在"n-ary 超边 pattern 间 dep/con/comp 语义边 + 形式化操作集 + intra-DAG propagation + LLM 仅命名"——这些是细节差异，需证明产生能力差异

### 最诚实的结论

**我们的工作不是"新范式"创新，是"已有范式的精细化"创新**：
- 自进化 schema KG 已有（DIAL-KG）→ 我们加了形式化操作集 + n-ary 超边 pattern
- schema-guided 检索已有（Youtu-GraphRAG）→ 我们加了 dep/con/comp 多类语义边导航
- n-ary 超图检索已有（HyperRAG）→ 我们加了 verbatim evidence + schema 层约束

**真正的 contribution 必须落在"精细化带来了可量化的能力提升"**——不是"我们也做了"而是"我们这样做比已有方式更好"。这需要实验证据，不是调研能定的。
