# Plan: LangGraph 智能体编排方案（完整项目重设计，不糊弄）

## 目标（覆盖整个项目，不只 re-type）
自演化 schema + n-ary 超图 + 跨论文 A-box 对齐 + 富拓扑 + 可溯源 + 下游 QA agent。
用 LangGraph 把整个系统建模成**分层 orchestrator-worker workflow**（确定性图 + LLM 节点 + 条件分支，
非 autonomous loop——这是我们"确定性流 + 事件触发演化"本质的工程化）。

## 调研确认的 LangGraph 原语（已读官方 concepts）
- StateGraph(typed state schema) + Reducers（状态合并语义：edges 追加 / meta 覆盖）
- Nodes = 函数（LLM 或纯代码），收 state 返 partial update
- Conditional edges（validate 失败 → 演化探针）
- **Send** = fan-out 并行 map-reduce（L2 跨 DAG 节点）
- **Command** = 动态 goto + state update 一次返回（演化闭环重路由）
- **Checkpointer** = 每超步持久化（= 暂存 + 崩溃 resume + time-travel 回放）
- **Store** = 跨调用长期记忆（= 共享 meta_hg schema + concept_graph A-box 跨论文持久）
- **Subgraphs** = 图作节点（分层：L0 corpus → L1 paper → L2 section，stateful/per-invocation）
- **Interrupts** = HITL（审 gate 接受的新 pattern / 手动 re-type 抽查）
- Recursion limit（演化循环有界）+ Streaming + LangSmith tracing（可观测）

## 总体范式：三层 orchestrator-worker
```
L0 Corpus Orchestrator (top graph, 长生命周期, Store 存 meta+concept_graph)
 ├─ 对每篇论文 sequential（schema 跨论文演化，必须保序；用 Store 持久 meta）
 ├─ fan-out 不跨论文（保 schema 演化语义），L1 内部才并行
 └─ 全量后: align_concepts (A-box LLM 对齐) → 下游 QA agent dispatch

L1 Paper Subgraph (per-paper, 作为 L0 节点)
 load_blocks → structure_map → Send(L2 每个 DAG 节点) → repair(split/merge/retire/rename)
 → infer_pattern_deps/cons/comp → detect_violations → consolidate → rich_topology
 → ingest_to_A-box → checkpoint
 边: structure_map 失败 → fallback(老 extractor) ; 演化失败累积 → HITL interrupt

L2 Section/DAG-node Subgraph (per section, 作为 L1 节点, Send 并行)
 extract_entities → label_entities → extract_edges → validate(结构)
 → re_type_edges → existence_filter → (新结构/新类型 → evolution_probe subgraph)
 边: LLM 空产出 → retry(更小chunk) ≤N → skip+log
     re_type 标 "new" → evolution_probe ; re_type 标 no_relation → existence_filter 删

Evolution Probe Subgraph (从 L2/L1 调用, 闭环)
 probe(LLM 提议) → validate_proposal(gate 判 variant/new-kind) → apply(add/split/merge/retire/rename)
 → mutate meta(in Store) → Command goto re_validate 失败边
```

## 共享状态（Store + 各层 state schema）
**Store（跨论文长期，对应现有 self.meta_hg + self.concept_graph）**:
- `meta`: MetaHypergraph（T-box，演化中变异；reducers=overwrite by version）
- `concept_graph`: ConceptGraph（A-box，跨论文累积；reducers=merge by surface+LLM-align）
- `trigger`: EvolutionTrigger（跨论文 recurrence 计数）

**L0 corpus state**: paper_queue, processed_papers, A-box ref, meta ref, qa_dispatch
**L1 paper state**: paper_id, blocks, structure_map, per_section_results(reducer=merge), 
  instance(InstanceHypergraph), evolutions(reducer=append), failed_edges(reducer=append), 
  rich_topo, citations, metadata
**L2 section state**: section_id, chunk_text, nodes(entities), labels, edges(Hyperedge[], reducer=append), 
  predecessor_summary, re_type_verdicts, schema_prompt_snapshot(随 meta 演化 re-fetch)

## 节点清单（LLM 节点单职责，确定性节点纯代码）
| 层 | 节点 | 类型 | 职责 | 现有映射 |
|---|---|---|---|---|
| L1 | load_blocks | 代码 | MinerU/md → blocks | load_paper_blocks |
| L1 | structure_map | LLM | 全文 → sections+DAG | map_structure |
| L1 | run_repair | 代码 | split/merge/retire/rename | run_split/merge/retire/rename |
| L1 | infer_schema_topology | 代码 | deps/cons/comp + violations | infer_pattern_* |
| L1 | consolidate | 代码 | 域门控清理 | consolidate_instance |
| L1 | rich_topology | 代码 | 7类富边直读 | infer_rich_topology_direct |
| L1 | ingest_Abox | 代码 | instance → concept_graph | concept_graph.ingest_instance |
| L1 | checkpoint_paper | 代码 | 持久化 bundle | dump_paper_bundle(扩展 per-node) |
| L2 | extract_entities | LLM | step1 surface+evidence | _STEP1 |
| L2 | label_entities | LLM | step2 label | _STEP2 |
| L2 | extract_edges | LLM | step3 超边 | _STEP3 |
| L2 | validate | 代码 | 结构校验 role/type/qualifier | meta.validate |
| L2 | **re_type_edges** | LLM(≠抽取模型) | 重判 pattern_type→canonical/new/no_relation | 新增(已有 judge 原型) |
| L2 | **existence_filter** | 代码 | 删 no_relation 边 | 新增 |
| Evo | probe | LLM | 提议 add/split/merge/retire | evolution_probe |
| Evo | gate | LLM+代码 | variant/new-kind 判定(已健康) | validate_proposal |
| Evo | apply | 代码 | 改 meta | apply_proposal |
| L0 | align_Abox | LLM | 跨篇 METHOD/PHENOMENON 对齐 | concept_graph.align_concepts |
| L0 | **downstream_qa_agent** | LLM agent | 基于超图+schema 推理问答(目标,现 stub) | 未建,接入点预留 |

## 边 + 条件路由
- L1: structure_map ok → Send(L2 per DAG node) ; fail → fallback_old_extractor → END
- L2: extract_edges → validate → {pass→re_type | no-matching-pattern→Evo.probe}
- L2: re_type → {canonical→existence_filter ; "new"→Evo.probe ; no_relation→drop}
- L2: Evo.apply 后 → Command(goto re_validate 这批边) [闭环]
- L2: LLM 空产出(长chunk→0) → retry(chunk减半) ≤3 → skip+log(错误恢复)
- L1: run_repair 后 → infer_schema_topology → consolidate → rich → ingest → checkpoint → END
- L0: 全量 paper 处理完 → align_Abox → (可选)downstream_qa_agent → END
- **Interrupt**: Evo.gate accept 新 top-level family / re_type 不确定 → 暂停等人审(HITL)

## 并行策略（关键：保 schema 演化语义）
- **L2 跨 DAG 节点**: Send 并行（一篇论文内 section 间无 schema 写冲突？有——meta 在 L2 演化会跨 section）。
  → 诚实约束：**L2 并行只读 schema_prompt_snapshot，写 meta 串行化**（用 Command 串到 Evo 子图，
  reducer 合并 evolution proposals）。或保守：L2 串行保 intra-DAG propagation（现有行为，schema 随
  节点演化前传）。**先保守串行，验证后再试并行**——不赌。
- **L0 跨论文**: 串行（meta 跨论文演化必须保序，现有 trigger 累积语义）。

## 持久化（= 用户要的暂存 + 崩溃 resume）
- Checkpointer(L1/L2): 每超步存 state → 任意节点崩溃可 resume 从最近 checkpoint
  + LangSmith tracing 看每步中间结果（= 暂存审查需求，自动化版）
- Store: meta + concept_graph 跨论文（现有 self.meta_hg/concept_graph 的工程化）
- 扩展 bundle.py: per-node checkpoint（不只 per-paper）

## 稳定性工程（融入图，不是外加）
- **结构化输出**: 每个 LLM 节点声明 JSON schema（entities/labels/edges/re_type_verdict/proposal），
  Paratera payload 加 response_format json_schema（测支持；fallback parse+retry）
- **有界重试**: LLM 节点空/垃圾 → retry(chunk减半) ≤3 → skip+log（治长chunk→0静默丢）
- **递归上限**: Evo loop 已 MAX_ACTIVE_PATTERNS；加 max_retype_iterations 防 re_type↔evo 死循环
- **Command 闭环**: 演化接受新 pattern 后 Command 回 re_validate，不靠手写 loop

## 下游 QA agent 接入点（目标终点）
- L0 全量对齐后 → downstream_qa_agent 节点（基于 concept_graph A-box + meta T-box 推理）
- 这是"利用超图+演化schema做问答/推理"的实现位（memory: scope-mismatch 战略转向后的下游开发利用）
- 本方案留接入点 + state（A-box ref + meta ref + query），agent 实现后续单独设计

## 迁移路径（渐进，不推倒重来——现有代码大多可复用为节点函数）
1. **状态建模**: 定义 L0/L1/L2 state schema + Store(meta/concept_graph) — 把现有 self.* 迁成 Store
2. **节点封装**: 现有函数(load_blocks/map_structure/extract_hypergraph 多步/run_evolution_loop/
   consolidate/rich_topo/ingest_instance/align_concepts)包成 LangGraph node（加 @task 或纯函数）
   — 逻辑不动，只加 schema 声明 + partial-state return
3. **图装配**: L2 subgraph → L1 subgraph(嵌 L2 + repair + consolidate + rich + ingest) → L0 graph
4. **条件边 + Command**: 接 validate/re_type/evo 的路由 + 闭环（替换现有手写 if/loop）
5. **re_type + existence_filter 新节点**: 接 L2（已有 judge 原型，正做）
6. **Checkpointer + Store + tracing**: 接 SqliteSaver/InMemoryStore + LangSmith（= 暂存自动化）
7. **Interrupt**: 接 HITL（gate 审 / re_type 抽查）
8. **下游 QA agent 接入点**: 留 state + stub

## 验证（不用下游，用 instance/schema 级）
- before/after: compound-name 率(28%→?)、sink 率、re_type judge(≠抽取模型) 边正确性、
  pattern 数/redundancy(schema 紧凑度)、checkpoint resume 可重放
- 4 跨域 + 5 颗粒流 before/after
- LangSmith tracing 看每节点中间态（暂存审查自动化）

## 不做
- 不改 extract prompt（用户要求）
- 不加严 gate（实测健康）
- 不冻 pattern_type enum（丢演化）
- 不上 AutoGen/CrewAI（多智能体对话，错配确定性 pipeline）
- 跨论文不并行（保 schema 演化语义；L2 保守串行先）

## commit 计划（渐进，每步可验可回退）
1. state schema + Store(meta/concept_graph) 迁移
2. L2 subgraph（含现有 step1/2/3 + 新 re_type + existence_filter + 结构化输出 + retry）
3. Evo probe subgraph + Command 闭环
4. L1 subgraph（嵌 L2 + repair + consolidate + rich + ingest + checkpoint）
5. L0 graph（串行 papers + align_Abox + 下游接入点 stub）
6. Checkpointer + LangSmith tracing + Interrupt HITL

## 诚实风险
- 现有代码逻辑多复用为 node 函数，但 extract_hypergraph 内部 DAG+blackboard+evolution 手写循环
  要拆成图节点——这是主要重构点，要小心保 intra-DAG schema 前传语义
- Paratera response_format json_schema 支持待测；不支持则 parse+retry（已 fallback）
- L2 并行会引入 schema 写冲突，先串行保守
- LangGraph 依赖 + LangSmith（tracing 要 key）——研究项目可接受，但要记一笔
