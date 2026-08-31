# 多跳图检索 / GraphRAG 前沿调研报告（2024-2026）

来源：调研 agent（arxiv 逐字核实，40 篇展开 18 篇）

## 核心 SOTA 形态（2026 共识）
「demand decomposition → schema-conditioned 起点选择 → query-aware gated traversal（实体作 join key + path 作显式对象 + 不分解 triple）→ verbatim span grounding lineage → policy-based context assembly」

**实体中心 + 每步 semantic gate 是性能与延迟双收益的最小必备件**（2606.30133 实测：gate off F1 掉 3.6-7.4，latency 慢 1.5-4.9×）

## 关键工作

### 与我们设定 1:1 对应
- **SAG** (2608.12129)：不建全局 KG；每个 chunk = event + entities 形成latent hyperedge（保留 n-ary 不分解）；共享实体作 join key 动态连接；MuSiQue Recall@5 80.36%（+11.52）
- **LineageRAG** (2608.16004)：每个 evidence demand 维护 lineage，completion 时用 verbatim source span grounding；HotpotQA/2Wiki/MuSiQue +3.51/+5.96/+5.22
- **Query-Aware Spreading Activation** (2606.30133)：spreading activation + 每步 semantic gate（cos(entity desc, question)）；一条 Cypher 实现；beat HippoRAG +5.3 EM；**gate 是性能+延迟双收益来源**
- **SCAIR** (2607.22571)：schema-conditioned structural priors + schema-aware traversal enforcement；training-free；企业 KG 必须 schema-aligned

### schema/有向导航
- **SemFlowRAG** (2606.28447)：诊断"概率黑洞"（flat undirected 拓扑下 flow 卡在高 degree 抽象节点）；用 entity 抽象度把无向边变有向语义约束；abstractness-guided directed PageRank 强制高抽象→低抽象梯度
- **PAGE-RAG** (2607.19301)：自动构建的图是 source 的不完整投影，不能当独立知识源；图当 semantic skeleton 组织/导航；task-adaptive routing + knowledge boundary abstain
- **DocNavRAG** (2608.01565)：沿文档结构导航；locate/navigate/expand/fetch 四操作；evolving evidence state；跨 section 多 hop +17.7% context sufficiency

### path/proposition 为中心
- **PropRAG** (2504.18070)：context-rich propositions（不分解成 triple 避免 context collapse）；LLM-free beam search over proposition paths；zero-shot SOTA Recall@5
- **PathRAG** (2502.14902)：key relational paths + flow-based pruning（max-flow 去冗余）；论点：问题是 redundancy 不是 insufficiency

### agentic / 自进化
- **EvoGraph-R1** (2607.12764)：retrieval 当 MDP；action={GraphRetrieve, WebSearch, GraphEdit, Answer}；图在检索过程中被修改
- **ACE-GraphRAG** (2608.01269)：context construction 当 policy；gap-aware refinement + depth/breadth branches；provenance + abstraction level 保留
- **Noesis** (2608.15919)：bidirectional traversal（模拟渐退记忆）+ AIMD 并发控制 + 跨 KB 语义路由；HotpotQA 59.5 EM

### 经典基础
- GraphRAG (2404.16134)：社区 Leiden + local/global search
- HippoRAG (2405.14831)：Personalized PageRank；单步 ≈ IRCoT 迭代效果但 10-30× 便宜
- ToG (2307.07697)：LLM iterative beam search on KG
- RoG (2310.01061)：先 plan relation paths 再 retrieve
- GraphReader (2406.14550)：agent-on-graph，4k 窗口 beat GPT-4-128k
- LightRAG (2410.05779)：dual-level（low+high）+ 增量更新

## 实体中心 vs 边相似度检索对比
| 维度 | 实体中心（HippoRAG/ToG/SAG） | 边相似度（PathRAG/QAFD-RAG） |
|---|---|---|
| 起点 | 实体匹配→seed 节点 | 关系/路径语义匹配 |
| 多跳表达 | 隐式（PPR/beam 路径） | 显式（path 作 first-class） |
| 优势 | 实体消歧天然；n-ary 友好（SAG） | 路径可解释；query-aware 容易 |
| 劣势 | query-blind 风险；高 degree 黑洞 | 依赖 embedding；triple 化丢 context |

**SOTA 共识**：纯实体中心不够，纯边相似度不够。**实体作 join key + path 作显式对象 + query-aware gate 的混合**

## SOTA pipeline
```
[Query]
 ↓ ① Demand decomposition（拆成 N 个 evidence demands）
 ↓ ② Seed entity linking（实体匹配到图节点；多 seed）
 ↓ ③ Query-aware graph traversal
    - 结构：hyperedge / proposition path（不分解 triple）
    - 跳转：spreading activation / beam / PPR
    - gate：每步 cos(候选, query/demand)（必备）
    - 方向：directed（抽象→具体梯度）/ bidirectional（回溯）
    - schema-conditioned：traversal 须 schema-aligned
 ↓ ④ Path/neighborhood 构建
    - provenance tracking（每个 evidence 标所属 demand）
    - flow-based pruning 去冗余
    - verbatim span grounding
 ↓ ⑤ Stopping
    - demand 全 supported / agent 自反思 sufficient / PPR 收敛+top-K
 ↓ ⑥ Context assembly（policy-based，保留 abstraction level 分层）
 ↓ ⑦ LLM generation + knowledge boundary abstain
```

## 2026 共识 design choices
1. **不分解成 triple**（SAG/PropRAG 一致）—— 用 hyperedge/proposition 保留 n-ary context
2. **query-aware gate 必备**（2606.30133 ablation）
3. **provenance tracking**（LineageRAG/ACE-GraphRAG）—— 每个 evidence 知答哪个 sub-demand
4. **verbatim grounding**（LineageRAG）—— 不用 paraphrase
5. **schema-conditioned traversal**（SCAIR）—— schema-bound 图必须 schema-aligned
6. **planning > offline indexing**（2606.29399）—— +38pp gap 来自 planning，offline edge inference 不提升 acc

## 可迁移到我们的策略
| 我们的组件 | 对应 SOTA 机制 | 来源 |
|---|---|---|
| n-ary 超边 = latent hyperedge | SAG chunk-as-event + shared entity as join key | 2608.12129 |
| verbatim evidence 字段 | LineageRAG verbatim source span grounding | 2608.16004 |
| schema dep/con/comp 拓扑 | SemFlowRAG directed semantic gradient（有向 PPR） | 2606.28447 |
| schema 作结构约束 | SCAIR schema-conditioned traversal enforcement | 2607.22571 |
| 跨 section 多 hop | DocNavRAG document-structured navigation | 2608.01565 |
| schema 是 instance 投影 | PAGE-RAG projection-aware skeleton | 2607.19301 |

## 推荐迁移设计：双层图检索
```
Schema 层（pattern graph, dep/con/comp 边）
  - 导航骨架（projection-aware skeleton）
  - dep 边作有向依赖（directed PPR，抽象→具体梯度）
  - traversal 须 schema-aligned
  - 起点：query → schema pattern 匹配

Instance 层（n-ary hyperedge, 带 verbatim evidence）
  - 每个 hyperedge = 抽取出的 frame（SAG event）
  - 共享实体 = join key（SAG）
  - verbatim_evidence → LineageRAG grounding
  - query-aware gate: cos(hyperedge 实体描述, query)
```

检索流程：
1. Query → 分解 evidence demands
2. Demand → 匹配 schema pattern（起点选择，schema-conditioned）
3. schema 层沿 dep/con/comp 边做 directed PPR（抽象→具体）
4. 每到一 pattern 节点，下钻到其 instance hyperedges
5. instance 层用 shared entity join 扩展到相邻 hyperedge
6. 每个 candidate 带 demand provenance
7. verbatim_evidence 字段完成 lineage grounding
8. demand 全 supported → 停止
9. lineages（含 verbatim span）喂 LLM 生成答案

## 陷阱警告
1. **概率黑洞**（SemFlowRAG）：高 degree 通用 pattern 吞 flow → 必须有向梯度
2. **query-blind traversal**（2606.30133）：纯结构 PPR 慢且差 → 每步 semantic gate 必备
3. **planning > offline edge inference**（2606.29399）：不要离线建全部 instance 间边
4. **triple 分解丢 context**（PropRAG context collapse）：超边不要拆成 triple
5. **license/cost**（2608.16096）：报 SOTA 前需 disclose embedder license
