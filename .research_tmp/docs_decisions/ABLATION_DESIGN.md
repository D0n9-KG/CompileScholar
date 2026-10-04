# 消融开关设计 DECISION (2026-08-14)

## 现状核实（关键，影响实验有效性）
- 完整管线在 `src/granular_agent/agent.py:process_paper_hypergraph`（290-360）：
  共享 meta_hg + hg_trigger（跨论文演化）→ extract_hypergraph（add_pattern via
  failure loop）→ run_split/merge/retire/rename（pattern 级维护）→ infer dep/con/comp。
- 简化路径 `.research_tmp/researchqa_phase1.py:build_hypergraph_from_sections`：
  每篇新建 meta+trigger（无跨论文积累），且**不调** run_split/merge/retire/rename
  → 只 add_pattern，且单篇几乎不触发（gate cross_node>=2）。
- **结论**：实验必须用 agent.py 式编排（共享 meta+trigger+五操作），不能用简化路径，
  否则自演化根本不发生，消融臂全部相同=实验废。

## 消融臂定义（4 臂 + Naive RAG baseline）
| 臂 | evolve | propagate_intra_dag | pattern 维护(split/merge/retire/rename) |
|----|--------|---------------------|----------------------------------------|
| full          | True  | True  | 跑 |
| add_only      | True  | True  | **跳过**（仿 AgentCAT 只 ADD） |
| frozen        | False | -     | 跳过（meta 恒为 seed） |
| no_intra_dag  | True  | False | 跑 |
| naive_rag     | -     | -     | 不抽超图，bge-m3 检索 chunk |

## 最小核心改动（extract_hypergraph 加两个可选参数，默认保持原行为）
- `evolve: bool = True`：False 时跳过 run_evolution_loop（frozen）。
- `propagate_intra_dag: bool = True`：False 时 schema_prompt 只在循环前算一次，
  不按节点 re-fetch（关 intra-DAG 传播；evolution 仍 mutate meta 供跨论文+maintenance）。
- add_only 的 split/merge/retire/rename 跳过在**驱动器层**处理（不改 extract_hypergraph）。

## 语料驱动器（.research_tmp/corpus_driver.py，镜像 agent.py 编排）
1. 一篇综述 → 共享一个 meta(seed)+trigger，按引用论文顺序逐篇处理。
2. 每篇：mineru_md_to_sections → build_smap_blocks → extract_hypergraph(ablation) →
   （非 add_only/frozen）run_split/merge/retire/rename → infer dep/con/comp → 收 instance edges。
3. instance edges 聚合成大图（matching.py 用）。
4. 每篇结果持久化 .research_tmp/runs/<survey>/<arm>/<paper>.json（防 deepseek 卡死丢进度）。
5. 抽完一篇即 append，可断点续跑。

## 顺序敏感性（诚实记录）
- 跨论文演化依赖处理顺序。固定每篇综述的 refs 按 safe_name 排序（可复现）。
- 不同臂用相同顺序，保证唯一变量是消融配置。

## Naive RAG baseline
- 把每篇 ref markdown 切 chunk（~500 token 滑窗）→ bge-m3 嵌入 → 检索。
- NMR: gold method 名 cos vs 所有 chunk。ERR: 无边结构→预期≈0（证明超图边价值）。
