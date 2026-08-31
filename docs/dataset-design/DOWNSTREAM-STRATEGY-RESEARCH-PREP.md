# 下游应用策略调研准备笔记

## 我们的独特性（调研结果回来后要对接的锚点）

### 两层结构
1. **Schema 层 (meta-hypergraph)**：pattern_type 之间有 depends_on/constrains/composes 边
   - pattern: influences/defines/composed_of/constitutive_law/measures/claim_relation
   - schema 边是"类型间关系"：influences --depends_on--> defines
   - schema 会自进化（add/split/merge/retire/rename）
2. **Instance 层**：具体论文的 n-ary 超边
   - 每条边连接 N 个具体实体（N>=2，多角色）
   - 每条边带 verbatim evidence_span（论文原文片段）
   - 边带 qualifiers（method/evidence_strength/cited_from/applies_in_regime 等）

### n-ary 特性
- 一条边可连接 >2 个节点（如 claim_relation: Informer→DeepAR→49.3%，三个节点）
- 节点有 role（from/to/parameter/whole/component 等）
- 不是二元 KG，是真正的超图

### evidence_span verbatim
- 每条 instance 边带论文原文片段
- 可直接作为 citation（citation_accuracy 要求 verbatim）

## 当前下游应用策略的问题（C1/C2 失败诊断）

### Phase 1b 实际做的（弱）
```
Q → bge(Q vs instance 边文本拼接) → top-k 边 → LLM 答
C2: + 沿 schema 拓扑边事后扩展相邻 pattern 的边
```

### 三个根本缺陷
1. **只算边整体相似度，不算实体节点匹配**
   - Q 里的关键实体（Informer/ProbSparse）只是被拼进边文本，没有单独做节点匹配
   - 实体匹配信号被边文本稀释
2. **schema 富拓扑只做事后扩展，不参与检索决策**
   - schema 边（dep/con/comp）只在选完种子边后扩展，不在路由阶段起作用
3. **没有多跳导航**
   - multi_hop 题需要跨 section 追踪同一实体，沿 schema 拓扑边跳到关联 pattern 的边
   - 当前是平铺 top-k，没有"沿图跳转"的概念

### 失败数据（8 题 C1 vs C2）
- C2 只在 1 题 >C1（W4385245566-Q1: 0→0.50），其余持平
- 净效应微弱，富拓扑价值没体现

## 目标下游任务

ResearchQA multi_hop 题：
- 跨 section 推理（答案证据散在 Methods/Results/Discussion 不同 section）
- 需引用 verbatim evidence（citation_accuracy 要求原文子串）
- expected_references 带 section_label + alternatives（verbatim 片段）
- 评分：section_coverage（命中所有 required section）、cite_accuracy（verbatim）、cite_precision

## 待调研结果回来后综合设计

需要回答的关键问题：
1. 实体节点如何参与检索路由（entity-centric retrieval）
2. schema 拓扑边如何做 pattern 间导航（schema-guided multi-hop）
3. n-ary 超边如何被查询（超图查询 vs 平铺相似度）
4. verbatim evidence 如何作为 citation 被精准召回
5. 自进化 schema 的动态性如何影响检索（schema 变了检索路径也变）
