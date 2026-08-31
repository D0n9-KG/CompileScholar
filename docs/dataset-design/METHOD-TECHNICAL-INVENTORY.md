# LogicKG 方法技术特征清单（代码实际实现）

来源：深读代码 agent（不靠设计意图，只看代码）

## 1. Schema 层数据结构
- MetaHypergraph: meta_nodes + patterns + meta_edges + version + family_roots
- MetaNode: type_id + description + is_abstract
- MetaHyperedgePattern: pattern_id + description + role_slots + allowed_qualifiers + deprecated + split_from + is_abstract + family
- MetaEdge.relation 实际取值: subclass_of / depends_on / constrains / composes / merged_into / renamed_to / retired
  - 注：设计文档的 type_relation / pattern_dependency 在代码中未实现为独立 relation

## 2. Instance 层数据结构
- InstanceHypergraph: paper_id + nodes + hyperedges + metadata
- HGNode: nid + labels(多标签) + surface + properties(value/_source_papers) + evidence_span + source_paper
- Hyperedge: eid + pattern_type + node_ids(N≥2) + node_roles + qualifiers + evidence_span(verbatim)
- InstanceCorpus: 跨论文累积 + merged_nodes(surface 合并) + align_embeddings + cross_paper_nodes

## 3. 自进化操作集

### 5 个 bounded operations（LLM probe 提议 + gate）
| op | gate |
|---|---|
| add_meta_node | evidence + near_dup Jaccard≥0.5 拒 + conservative gate |
| add_pattern | evidence + role-structure 相同且 id token-near-dup 拒 + 语义 cosine≥0.85 拒 + conservative gate |
| add_subclass | evidence + sup 必须已在 schema + 不可成环 |
| split_meta_node | evidence + type_id 必须在 schema |
| merge_meta_nodes | evidence + a/b 都在 schema + a≠b |

### 3 个 repair operations（确定性触发，非 probe）
| op | 触发 | gate |
|---|---|---|
| split_pattern | instance 边数≥2×MIN_CLUSTER(6) 且聚类得≥2 簇每簇≥3 | 父不可已是 abstract；子 role_sig 必须等于父；LLM 仅命名 |
| merge_patterns | 两 active pattern embedding cosine≥0.85(同 sig)/0.90(异 sig) | 拒 abstract+concrete 混合；子类 re-parent |
| retire_pattern | (1) concrete orphan 有 split_from 且 live 边数=0 (2) abstract 所有 descendants 都 deprecated | 永不 retire seeded + 永不 retire 有 active descendant 的 abstract |

### rename_pattern（第 5 bounded op）
- 触发：evolved pattern id 长≥45 字符且≥2 下划线，或全 UPPER_SNAKE 且 len>4
- LLM 仅提议新名；重写所有 meta-edge 引用

### split 聚类四 tier
1. discrete qualifier（dependency_type 等 distinct/total≤0.6）
2. dependency_context（边节点的其他 family 共现集合）
3. embedding（cosine≥0.55）
4. token Jaccard（≥0.35）
5. LLM semantic grouping（fallback）

### Conservative gate
- CONSERVATIVE_CROSS_NODE=2、CONSERVATIVE_CUMULATIVE=3
- 只 gate growth op；repair op 不受 gate
- mismatch_signature 不含 pattern_type（防碎片化）

## 4. Schema ↔ Instance 关系

### schema 约束 instance（validate）
1. 两遍匹配：concrete patterns 先 → abstract fallback → deprecated 永不匹配
2. role 匹配：variadic 用 set+count；非 variadic 用顺序严格
3. type 兼容：node.labels 任一 is_subtype slot.type
4. qualifier 检查：key 必须在 allowed_qualifiers；enum key 的 value 必须在枚举内
5. 失败返回 (False, "no-matching-meta-pattern") 触发 evolution

### instance 反馈 schema（forward propagation）
- 每个 DAG node 顶部 meta.to_prompt() 重新拉取
- 失败边累积 → run_evolution_loop 原地 mutate meta
- evolution 后重新 validate 暂存的失败边
- 后续 DAG node 看到已演化的 schema（intra-DAG forward propagation）

### instance 归纳 schema 富拓扑（确定性，无 LLM）
- infer_pattern_dependencies: definition-family B + 非 definition A 引用同 surface → A depends_on B
- infer_pattern_constraints: constitutive_law A + dependency/measure/claim B 共享节点 → A constrains B
- infer_pattern_compositions: composition-family A + 任何其他 B 共享节点 → A composes B
- split 时子 pattern 继承父的 dep/con/comp 边

## 5. 富拓扑边精确生成逻辑
| 边 | 触发 | 连接 | 方向 |
|---|---|---|---|
| depends_on | 非 definition A + definition B 引用同 surface | A→B | consumer→producer |
| constrains | constitutive_law A + dependency/measure/claim B 共享节点 | A→B | authority→consumer |
| composes | composition A + 任何其他 B 共享节点 | A→B | whole→part |

## 6. validate 完整逻辑
1. pattern 遍历：concrete 先 → abstract fallback → deprecated 永不试
2. role 匹配：variadic set+count / 非 variadic 顺序严格
3. type 兼容：node.labels 任一 is_subtype
4. qualifier：key 在 allowed_qualifiers + enum value 在枚举内
5. 空 schema：直接返回 False

## 7. 与普通 ontology 的结构性差异（代码实际有的）
1. **pattern 间非树富拓扑**：depends_on/constrains/composes — ontology 通常只有 subclass_of
2. **pattern-level split**：split 后父变 abstract 保留为 IS-A 根 + validate fallback；子继承拓扑边
3. **约束违反检测**：detect_constraint_violations — schema 自身能发现结构错误（NUMERIC 引用但未定义）
4. **hyper-relational instance layer**：Hyperedge 连 N≥2 节点 + 角色 + qualifier + verbatim 证据
5. **intra-DAG forward propagation**：schema 在 extraction 中途演化，下游 DAG node 同 pass 内看到演化后 schema
6. **abstract pattern 作为 validate fallback**：split 后父是泛化层 + 兜底匹配
7. **多标签节点**：HGNode.labels 是 list
8. **bounded op 集保守 gate**：cross_node≥2 或 cumulative≥3 + merge/retire 防发散
9. **LLM 仅命名不判断**：split/merge/rename 决策是确定性聚类/embedding，LLM 只做 labeling

## 设计文档与代码不一致处
- QUALIFIER_REGISTRY 多 5 个 load-bearing key（applies_in_regime/dependency_type/relation_type/function_form/parameters）
- discourse-role weighting 未实现
- 没有专门的 add_role op
- type_relation 字符串未使用
