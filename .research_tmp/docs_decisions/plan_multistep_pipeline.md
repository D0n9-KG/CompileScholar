# Plan: 多步抽取 pipeline + re-type 节点 + 稳定性工程

## 目标
把抽取拆成单职责多步（每步一次 LLM、轻负担），加 re-type 节点治 54% 命名噪声/兜底病，
加稳定性工程（结构化输出、有界重试、每步 checkpoint）让流程稳。不碰 prompt（用户要求）。

## 调研结论（已做）
- **LangGraph Functional API** 最贴合：`@entrypoint`+`@task` 装饰**现有** if/for/function 代码，
  加持久化/流式/追踪，**最小改动不强制重构成显式 DAG**。正是我们要的"现有工作代码+加稳定层"。
- Graph API（显式 StateGraph）控制更细但要重写，overkill now。
- 多智能体框架（AutoGen/CrewAI）= agent 间对话协作，我们是确定性多步 pipeline，**错配**，不用。
- 文献：两阶段 extract-then-type 在 RE 领域"基本缺席"（agent B 调研），所以这是我们的增量，不是抄。
- 结构化输出：Paratera 裸 urllib 无 response_format，需加 payload 参数测 DeepSeek json_schema 支持；fallback=parse+retry。

## 架构：单职责节点 + 共享状态 + 条件边 + 错误恢复

### 节点（LLM 节点单任务，确定性节点纯代码）
| 节点 | 类型 | 职责 | 现状 |
|---|---|---|---|
| extract_entities | LLM | step1 抽实体 surface+evidence | 已有 |
| label_entities | LLM | step2 标 label | 已有 |
| extract_edges | LLM | step3 抽超边（找关系+给role+qualifier+evidence）| 已有 |
| **re_type_edges** | **LLM 新增** | **单任务**：拿每条边(evidence+节点)+seed schema，重判 pattern_type→canonical 名 / "new"+理由 / "no_relation" | **新** |
| **existence_filter** | **确定性新增** | 删 re_type 标 no_relation 的边 | **新** |
| validate | 确定性 | 结构校验 role/type/qualifier（已有，保留）| 已有 |
| evolution_probe | LLM | re_type 标 "new" 时触发，gate 判 variant/new-kind（已健康，实测 0-2/篇）| 已有 |
| consolidate / rich_topo | 确定性 | 已有 | 已有 |
| checkpoint | 确定性 | 每节点后存状态 | 扩展现有 bundle |

### 控制流（条件边）
```
extract_edges → validate(结构)
  ├ pass → re_type_edges → existence_filter → consolidate → checkpoint
  └ fail(真·新结构) → evolution_probe → {accept_new → re_validate; reject → drop}
re_type "new" → evolution_probe（同 gate）
错误恢复: LLM 节点产出 empty/garbage(长chunk→0) → retry(更小chunk/温度调整) ≤N → skip+log
```

### 共享状态 schema（typed）
nodes, labels, edges(带 re-typed pattern_type), schema(meta), failed_edges, evolutions,
source_text, per_node_summaries, checkpoints.

## 稳定性工程（用户核心诉求）
1. **结构化输出强制**：每个 LLM 节点声明 JSON schema（entities/labels/edges/re_type_verdict）。
   加 `response_format: json_schema` 到 Paratera payload（测 DeepSeek 支持；不支持则 parse+retry fallback）。
   → 杀"LLM 乱写 pattern_type"（validate 不查名，但 re_type 节点输出 schema 强制 canonical 名或 new 哨兵）。
2. **有界重试**：parse 失败/空产出 → retry（chunk 减半 / temperature 微调）≤3 → skip+log。
   → 治"长chunk→0 边静默丢失"。
3. **每步 checkpoint**：每节点后 dump 状态（扩展 bundle.py 到 per-node）。
   → 既满足"跑完暂存审查"，又支持崩溃后 resume（长语料跑全量时关键）。
4. **递归上限**：evolution loop 已有 MAX_ACTIVE_PATTERNS；加 max_retype_iterations 防 re-type↔evolve 循环。

## re_type 节点设计（核心新增）
- **输入**：一批边（每条：evidence_span + 节点 surface/label + 当前 pattern_type + qualifiers）+ seed schema 全表（含每个 pattern 的语义边界描述）。
- **输出**（JSON schema 强制）：每条边 → {canonical_pattern_type ∈ seed 名 ∪ {"new"}, is_real_relation: bool, drop_reason?}
  - canonical 名：re-type 到 seed（治 28% 复合名 + 26% 兜底选错）
  - is_real_relation=false → existence_filter 删（治 ~20% borderline/无关系尾巴）
  - "new"+理由 → evolution_probe（真新关系照样演化，schema 不冻）
- **单任务单批**：一次只判类型+存在性，不抽不找，LLM 负担轻、判断准（任务分解逻辑，用户认可）。
- **judge 模型 ≠ 抽取模型**：re_type 用 deepseek-chat 或 GLM-5（非 V4-Flash），避免自我认同偏差（上次实测 caveat）。

## 框架取舍（research 交付）
| 选项 | 采纳成本 | 得到 | 判决 |
|---|---|---|---|
| **LangGraph Functional API** | 低（装饰现有函数，不重写）| checkpoint/resume/streaming/tracing 免费拿 | **推荐**：稳定层免自研，最小改动 |
| 纯手写（加 retry+structured-output+per-node persist）| 中（自研稳定层）| 无依赖、全控 | **fallback**：若 Functional API 加摩擦则退此 |
| Graph API（显式 StateGraph）| 高（重写）| 更细控制 | overkill，不取 |
| AutoGen/CrewAI（多智能体）| — | agent 协作 | 错配，不取 |

**采纳判据**：先按 Functional API 设计；若实现时装饰器/状态 schema 与现有 DAG+blackboard 冲突摩擦大，退到手写（retry+structured-output+扩展 bundle.py per-node）。两条路架构同构（节点+状态+条件边），切换成本低。

## 实施顺序（成本递增，每步可验）
1. **re_type 节点 + existence_filter**（核心治 54% 病）→ 验：compound-name 率 28%→?、sink 率、边正确性（不同模型 judge）修前修后
2. **结构化输出**（re_type 节点先上，验证 schema 强制有效）→ 验：parse 失败率降
3. **有界重试**（所有 LLM 节点）→ 验：长chunk 0-边丢失率降
4. **per-node checkpoint**（扩展 bundle.py）→ 验：崩溃 resume + 暂存审查齐
5. **（可选）LangGraph Functional API 包装**：1-4 手写稳定层若维护变重，再迁

## 验证基线（守 SciERC 负结果红线）
- **不用下游 F1**（没下游）。用：instance 级边正确性审计（不同模型 judge）+ schema 紧凑度（pattern 数/redundancy/compound 占比）。
- 修前修后对比，re_type judge 必须用 ≠ 抽取器的模型（避免 self-agreement 偏乐观）。
- 4 篇跨域 + 5 篇颗粒流（已有 bundle）做 before/after。

## 不做
- 不改 extract prompt（用户要求）
- 不加严 gate（实测健康，0-2/篇，不是膨胀主因）
- 不冻 pattern_type enum（丢演化，SCION 陷阱）
- 不上多智能体框架（错配）
- 不预先迁 LangGraph（先手写稳定层，摩擦大再迁 Functional API）

## commit 计划
- 1 个 commit：re_type 节点 + existence_filter + 结构化输出 + 有界重试（核心+稳定层一起，架构同构）
- per-node checkpoint 扩展随同 commit
- LangGraph Functional API 迁移（若做）单独 commit
