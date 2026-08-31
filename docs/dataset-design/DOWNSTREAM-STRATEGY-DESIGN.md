# 下游应用策略设计：Schema-Guided Multi-Hop Hypergraph Retrieval (SG-MHR)

**设计日期**：2026-08-14
**调研基础**：RESEARCH-schema-guided-retrieval.md + RESEARCH-multihop-graph-retrieval.md + RESEARCH-hypergraph-downstream.md
**目标**：设计一套成熟的下游应用策略，让 schema 层真正参与检索路由，区分"schema/抽取器问题"vs"下游策略太粗糙"

---

## 一、设计原则（来自三份调研共识）

1. **schema 层在检索阶段参与路由**——不是事后扩展，是路由决策（三份调研确认是空位）
2. **实体节点作 join key**——不只是边整体相似度（SAG 2608.12129 验证）
3. **不分解成 triple**——保留 n-ary 超边整体（SAG/PropRAG 一致）
4. **query-aware gate 每步语义门**——2606.30133 实证：gate off F1 掉 3.6-7.4，latency 慢 1.5-4.9×
5. **verbatim evidence 作 lineage grounding**——不用 paraphrase（LineageRAG 2608.16004）
6. **demand decomposition**——query 拆成 N 个 evidence demands（LineageRAG）
7. **planning > offline indexing**——2606.29399 实证 +38pp gap 来自 planning

## 二、我们的独特性（设计锚点）

1. **schema 层有 role_slots 模板**——每个 pattern 定义了 n-ary 超边的角色槽位（output/input/parameter...），普通 ontology 没有
2. **schema 层有多类语义边**——dep（依赖追溯）/con（约束检查）/comp（组合展开），可做不同导航语义
3. **schema 层自进化**——和 instance 共生演化，dep/con/comp 边随抽取动态生成
4. **instance 层带 verbatim evidence_span**——直接可作 citation grounding
5. **instance 层是 n-ary 超边**——一条边连接多实体多角色，共享实体作 join key

## 三、Pipeline 设计

### 总览

```
[Multi-hop Query]
  ↓
Phase 1: Demand Decomposition（拆成 N 个 evidence demands）
  ↓
Phase 2: Schema-Guided Routing（schema 层 dep/con/comp 边导航到相关 pattern）
  ↓
Phase 3: Entity-Centric Instance Retrieval（实体作 join key + query-aware gate）
  ↓
Phase 4: Lineage Grounding & Context Assembly（verbatim evidence grounding）
  ↓
Phase 5: LLM Generation（用 verbatim span 作 citation）
```

### Phase 1: Demand Decomposition

**输入**：multi_hop question（如"Informer 的 ProbSparse 机制如何影响它相对 DeepAR 的性能优势？"）

**机制**（来自 LineageRAG 2608.16004）：
1. LLM 把 query 分解为 N 个 evidence demands，每个 demand = "需要知道什么"
2. 每个 demand 关联期望的 pattern_type（哪个 pattern 最可能含答案结构）
3. 每个 demand 标注期望的角色（如 demand 需要"定义"→ role=definition；demand 需要"对比"→ role=from/to）

**示例**：
```
Query: "Informer 的 ProbSparse 机制如何影响它相对 DeepAR 的性能优势？"
Demand 1: "ProbSparse 是什么" → pattern=defines, role=subject/definition
Demand 2: "Informer 的组成" → pattern=composed_of, role=whole/component
Demand 3: "Informer vs DeepAR 性能对比" → pattern=claim_relation, role=from/to/parameter
```

**输出**：N 个 demands，每个带 {pattern_hint, role_hint, entity_hints}

### Phase 2: Schema-Guided Routing

**输入**：N 个 demands + schema 层（meta-hypergraph，pattern 间 dep/con/comp 边）

**机制**（综合 OMAGR 2606.11910 + SCAIR 2607.22571 + SemFlowRAG 2606.28447）：
1. **Demand → pattern 匹配**：每个 demand 的 pattern_hint 匹配到 schema 层的 pattern（embedding 相似度）
2. **沿 schema 边导航**：从匹配到的 pattern 出发，沿 dep/con/comp 边扩展到关联 pattern
   - **dep 边（depends_on）**：如果 demand 需要"X 的定义"，当前 pattern 沿 depends_on 找定义 X 的 pattern（依赖追溯）
   - **comp 边（composes）**：如果 demand 需要"X 的组成"，沿 composes 找组合 pattern（组合展开）
   - **con 边（constrains）**：如果 demand 需要"X 的约束条件"，沿 constrains 找约束 pattern（约束检查）
3. **Schema-conditioned traversal**：traversal 只走 schema 允许的 pattern 间边（SCAIR 机制）
4. **role_slots 作锚**：每个 pattern 的 role_slots 定义了"该检索什么角色的实体"——demand 的 role_hint 匹配 pattern 的 role_slots

**输出**：每个 demand 对应一组相关 pattern（schema-guided pattern set），每个 pattern 带 role_slots 约束

**关键创新**：schema 层 dep/con/comp 边做不同导航语义（不是单一 ontology 边），且 role_slots 定义了角色级检索约束

### Phase 3: Entity-Centric Instance Retrieval

**输入**：每个 demand 的 schema-guided pattern set + query 里的关键实体

**机制**（综合 SAG 2608.12129 + 2606.30133 query-aware gate + PRoH EWO 2510.12434）：
1. **Entity linking**：从 query 抽取关键实体（Informer, ProbSparse, DeepAR），在 instance 层找含这些实体的节点
2. **Pattern-filtered retrieval**：只在 demand 的 schema-guided pattern set 内检索含关键实体的超边
   - 不是平铺所有边，而是先用 schema pattern 过滤（schema 当过滤器）
3. **Entity as join key**（SAG 机制）：沿共享实体扩展到相邻超边
   - 如果实体 A 在 pattern P1 的超边里，也在 pattern P2 的超边里，则 P1 和 P2 通过 A 连接
   - 这是跨 section 的关键——同一实体在不同 section 的不同 pattern 超边里出现
4. **Query-aware gate**（2606.30133 机制）：每步扩展时，cos(候选超边的实体描述, demand) 作语义门
   - gate 阈值过滤掉语义不相关的超边
   - 这是性能+延迟双收益的必备件
5. **不分解 triple**：保留 n-ary 超边整体（SAG/PropRAG 共识）

**输出**：每个 demand 对应一组 instance 超边（带 verbatim evidence_span），每条超边标所属 demand（provenance）

**关键创新**：schema pattern 过滤 + 实体 join key + query-aware gate 三者联合，不是事后扩展

### Phase 4: Lineage Grounding & Context Assembly

**输入**：每个 demand 的 instance 超边集（带 verbatim evidence_span）

**机制**（综合 LineageRAG 2608.16004 + HyCE-RAG 2607.22597 + ACE-GraphRAG 2608.01269）：
1. **Verbatim lineage grounding**：每个 demand 的超边用 verbatim evidence_span 完成 lineage
   - 不用 paraphrase，直接用 evidence_span 作 citation material
2. **Provenance tracking**：每个 evidence 标所属 demand（哪个 demand 支持哪个证据）
3. **Demand completion check**：检查每个 demand 是否被 supported（有超边含相关 evidence）
   - 未满足的 demand 标记为 deficit → 可回到 Phase 2 沿 schema 边扩展更多 pattern
4. **Context assembly**（ACE-GraphRAG policy-based）：
   - 按 demand 组织 context（不是平铺所有超边）
   - 保留 abstraction level 分层（schema pattern → instance 超边 → evidence span）
   - 低置信路径分离暴露冲突（HyCE-RAG 机制）

**输出**：按 demand 组织的 lineages（含 verbatim span），每个 lineage 标 provenance

### Phase 5: LLM Generation

**输入**：按 demand 组织的 lineages + 原始 query

**机制**：
1. 把 lineages（含 verbatim evidence_span）按 demand 顺序喂 LLM
2. LLM 用 verbatim span 作 citation（citation_accuracy 要求原文子串）
3. LLM 跨 demand 综合答案（multi-hop 推理）

**输出**：answer + citations（verbatim spans from evidence_span）

## 四、与 C1/C2 的对比（为什么之前太弱）

| 维度 | C1/C2（Phase 1b） | SG-MHR（新设计） |
|------|-------------------|------------------|
| 检索单元 | 边整体相似度 | 实体节点 + schema pattern 过滤 |
| schema 参与度 | C2 事后扩展（选完种子才扩展） | Phase 2 路由决策（先 schema 导航再检索） |
| 多跳 | 无（平铺 top-k） | demand 分解 + schema 边导航 + 实体 join |
| 实体匹配 | 被拼进边文本（信号稀释） | 独立 entity linking + join key |
| query-aware | 无 gate | 每步 semantic gate |
| verbatim evidence | 喂 LLM 但不组织 | 按 demand lineage grounding |
| context assembly | 平铺所有边 | 按 demand 组织 + provenance |

## 五、诊断价值（区分问题来源）

用 SG-MHR 后：
- **如果 section_coverage 显著提升** → 之前是策略太粗糙（schema/抽取器没问题）
- **如果仍不提升** → 可能是 schema/抽取器问题：
  - schema dep/con/comp 边不够准（导航到错误 pattern）
  - 抽取的 instance 超边质量不够（实体识别不准/evidence 不 verbatim）
  - schema 自进化没产生有用的 dep/con/comp 边
- **如果部分 demand 满足部分不满足** → 可定位是哪个 pattern 的检索有问题

## 六、实现计划

### 优先级排序（先实现核心，再补完整）

**P0（必须，先实现验证）**：
1. Phase 3: Entity-Centric Instance Retrieval（实体 join key + pattern 过滤 + query-aware gate）
   - 这是最核心的改进——从边相似度升级到实体中心+schema过滤
2. Phase 1: Demand Decomposition（LLM 拆 query）
   - 简单实现：LLM prompt 拆 query 为 demands + pattern hints

**P1（重要，第二步）**：
3. Phase 2: Schema-Guided Routing（dep/con/comp 边导航）
   - 沿 schema 边从 demand pattern 扩展到关联 pattern
4. Phase 4: Lineage Grounding（按 demand 组织 + verbatim grounding）

**P2（优化，后续）**：
5. Phase 5: Context Assembly 策略（policy-based, 低置信分离）
6. Deficit-oriented re-routing（未满足 demand 回 Phase 2 扩展）

### 测试验证

1. **在已有 3 篇缓存论文上测试 SG-MHR**（不需要重新 extract）
2. **对比 C1/C2/SG-MHR**（同一批论文同一批题）
3. **看 section_coverage / cite_accuracy / cite_precision**
4. **按 demand 分析**（哪个 demand 满足/未满足）
5. **审视实际检索到的超边**（是否含 expected evidence）

### 成功标准

- SG-MHR section_coverage > C1/C2（证明策略升级有效）
- 或 SG-MHR 仍低但能定位是哪个环节（schema 导航/实体匹配/gate）
- 在 ResearchQA multi_hop 题上对比 naive RAG / bare LLM
