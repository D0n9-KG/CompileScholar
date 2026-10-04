# Plan: 智能体编排系统设计（第 1 块：整体骨架 + 自建底座内核）

## 总体架构骨架（先确认整体）

**自建 KnowledgeBase 内核（schema-grounded mutable hypergraph memory）+ builder/consumer 二分 agent 群 + LangGraph 作编排脚手架。**

调研定论：主流框架（LangGraph/AutoGen/CrewAI）都不原生把 KG/超图作 mutable working memory 一等公民（PatchBoard/HyperAgent/AI-Supervisor/DocTrace 都是自建 schema-grounded shared state）。所以——
- **底座内核自建**（schema-grounded mutation + write contract + transactional commit + 并发 + 审计 + 跨域分层 + 准入 + 5-outcome + utility 剪枝），对标 PatchBoard/AI-Supervisor/SEDM/MELD/HASTE/SCION。
- **LangGraph 只作 builder/consumer agent 的编排脚手架**（StateGraph + Command + Store 指向 KnowledgeBase 内核），核心理念不依赖框架。

```
┌─────────────────────────────────────────────────────────┐
│  LangGraph 编排脚手架 (StateGraph + Command + Store→KB) │
│  ┌──────────────────┐         ┌──────────────────────┐  │
│  │ Builder agent 群  │ write→  │ Consumer agent 群     │  │
│  │ extractor/evolver│ (contract│ paper-search/QA/    │  │
│  │ /aligner/maintain│  +commit)│ survey/conflict/    │  │
│  │                  │ ←reader │ evolution-analysis   │  │
│  └──────────────────┘ feedback └──────────────────────┘  │
│         │                            │                   │
│         ▼ read/write (role-gated)    ▼ read + propose-mutation
│  ┌─────────────────────────────────────────────────────┐ │
│  │ 自建 KnowledgeBase 内核 (本次详设, 第1块)            │ │
│  │  T-box(MetaHypergraph) + A-box(ConceptGraph n-ary)   │ │
│  │  + provenance + mutation ledger + write contract    │ │
│  │  + transactional commit + 并发 + 跨域分层 + 准入      │ │
│  │  + 5-outcome claim 裁决 + utility 剪枝              │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
演化闭环 (SAGE writer-reader + DIAL-KG 三阶段): consumer 读→反馈→builder 改底座→后续看到新底座
```

---

## 底座内核详设（第 1 块，本次交付）

### 三层数据（复用现有数据结构，不动核心）
- **T-box**：`MetaHypergraph`（现有）— pattern + role_slots + allowed_qualifiers + family + IS-A subclass + deprecated/is_abstract + version + family_roots。复用。
- **A-box**：`ConceptGraph`（现有）— Concept(multi-surface+symbol) + ConceptHyperedge(n-ary, node_ids/node_roles/kind/provenance)。复用。
- **provenance**：每条边 evidence_span(verbatim) + cited_from + method + evidence_strength + source_paper + version。现有部分，补 version。

### KnowledgeBase 内核层（新增，包现有之上）— 对标前沿

**1. 显式 Mutation 对象**（PatchBoard schema-grounded mutation）
所有写操作是 `Mutation` dataclass，非裸调用：
```
Mutation(op, target, payload, proposer_role, evidence, rationale, 
        domain, base_version, timestamp)
op ∈ {add_pattern, add_node, add_edge, split, merge, retire, rename, 
      add_subclass, relabel, align_merge, conflict_mark}
```
现有 add_pattern/split/merge/retire/rename 等方法改成"构造 Mutation → 提交内核"，逻辑复用。

**2. Write Contract**（PatchBoard role-specific write contract）
每个 agent role 声明能写什么：
- `builder.extractor`: add_edge（边带 evidence+cited_from），不能改 schema
- `builder.evolver`: add_pattern/split/merge/retire/rename/add_subclass（改 T-box）
- `builder.aligner`: align_merge（合并 A-box concept），不能改 T-box
- `builder.maintainer`: retire/relabel（utility 剪枝），不能加 pattern
- `consumer.*`: 只读 + 提 mutation 提案（propose，不直接写，走 builder 审）
内核 commit 前校验 proposer_role 对 op/target 的权限，越权拒。

**3. Transactional Commit + 确性 Kernel**（PatchBoard 确性 kernel + SCION candidate-evidence 绑定）
Mutation 批量 commit，commit 前确性 kernel 校验（全过才 commit，任一失败回滚）：
- schema 约束（pattern_type ∈ 注册名、role ∈ allowed、qualifier key/value ok）— 复用 validate 逻辑
- role write contract 权限
- runtime invariant（无孤儿 pattern、IS-A 树完整、family 归属）
- **candidate-evidence 绑定**（SCION）：每个 add_pattern/split Mutation 必须带 verbatim evidence + rationale，无证据拒
- **recurring crystallize 准入**（Metis）：新 pattern 提案需 recurrence 计数达阈值（EvolutionTrigger 现有累积，复用）才 commit，单次出现不进 schema
- **reproducible replay 准入**（SEDM）：Mutation 可重放（确定性输入+版本），非确定性 LLM 判定结果记 ledger 供审计

**4. Mutation Ledger**（AI-Supervisor audit + SEDM replay + time-travel）
每个 commit 的 Mutation 落 ledger：who/when/why/evidence/version_diff/domain。→ 可审计、可 replay 重放到任意版本（time-travel）、可复现。落盘 = 用户要的暂存（自动化版）。

**5. 并发控制**（CoAgent/Position 立论：shared-state MAS 失败多源于并发缺失）
- version stamp + 乐观锁（CAS）：Mutation 带 base_version，commit 时若底座 version 已变（其他 agent 写过）→ 冲突，重试/合并
- **builder 写串行化**（保 schema 演化语义，现有 trigger 累积行为）；consumer 读并发
- consumer 提的 mutation 提案进 queue，builder 串行审+commit

**6. 跨域分层 namespace**（HASTE global/domain/competition + DecentMem 去中心化）
- 底座 key 带 domain namespace：`meta:granular` / `meta:ml` / `meta:molecular` ... + `meta:global`（通用 seed 12 pattern + THING 根 + 通用 node 类型，全域共享）
- domain-specific pattern + concept 挂 domain 标签，跨域不污染
- 通用 seed 全域复用（constitutive_law 谁都用），域专属隔离
- 对齐（aligner）只在同 domain 内做，跨域不对齐（DecentMem：centralized 塌缩多样性）

**7. 5-outcome Claim 裁决**（MELD：consumer/aligner 写 A-box 时）
A-box 写入（add_edge / align_merge）走 5-outcome：
`insert`（新概念新边）/ `merge`（合并到已有 concept）/ `relate`（连到已有但独立）/ `conflict`（标矛盾，进冲突检测）/ `reject`（丢）
由 claim-key identity + embedding + NLI(LLM) 判定。consumer 写回底座（如综述发现新关系）也走这协议。

**8. Utility 排序剪枝**（SEDM utility controller + MemoryBank 遗忘）
retire 按 utility（使用频次/被引用/演化贡献）排序，低 utility 退。builder.maintainer 定期跑。防膨胀。

---

## 后续部分清单（逐个再详设，本次不展开，守"每部分精心设计不降级"）

| 部分 | 对标前沿 | 重新设计核心 |
|---|---|---|
| **抽取 agent** | Hyper-KGGen(skill-evolving+stability feedback+Global Skill Library+n-ary) + Reflexion(self-verify) | plan-execute + schema-in-context + self-verify/critic(内化re-type) + 存在性gate + 结构化输出 + verbatim后检 + 核心实体复用；skill 演化 |
| **schema 自演化 agent** | DIAL-KG 三阶段闭环 + SAGE writer-reader + 2608.18104 schema-constrained rewrites | Dual-Track Extraction→Governance Adjudication→Schema Evolution；writer-reader 闭环；schema-constrained rewrites；有控收敛(Metis recurring/SEDM utility/MemoryBank 遗忘/SCION 保守融合) |
| **跨论文对齐 agent** | MELD 5-outcome + EDC Extract→Define→Canonicalize | embedding+LLM judge 标准归一；5-outcome；增量对齐；跨域不混 |
| **富拓扑推断** | DocTrace hypergraph working memory | 7类直读(现有win)+LLM辅助ambiguous；超图作working memory |
| **维护 agent** | SEDM utility + SCION conservative fusion | retire/split/merge/rename principled判据；utility剪枝；保守融合 |
| **consumer: 论文检索(华为赛)** | PaSa/SPAR + HippoRAG PPR + GraphLoom reliability routing + grounding三处 | API检索(粗)+超图/schema(关联推理/重排/归纳/可溯源,覆盖内精)+LLM agent(查询分解/迭代)；evolution边+富拓扑做细粒度关联；grounding三处 |
| **consumer: QA** | DocTrace + PaperQA provenance | 精确检索+多跳推理+grounded回原文 |
| **consumer: 综述** | AutoSurvey/Agentic AutoSurvey/SciSage/DAS | 广覆盖+跨篇综合+结构化+可溯源；shared revisable state |
| **consumer: 冲突检测** | MELD conflict outcome | 交叉对比矛盾claim；5-outcome的conflict入口 |
| **consumer: 演化分析** | evolution边链推理 | extends/improves/compares链 |
| **协调/HITL** | MemGPT interrupts + AI-Supervisor consensus commit + 并发控制 | builder串行/consumer并发；consensus commit；HITL审新family/不确定；tracing+checkpoint+replay |
| **框架脚手架** | LangGraph StateGraph+Command+Store | 编排builder/consumer；Store指向KB内核；Command演化闭环；不靠框架做核 |

---

## 不做 + 风险
- 不冻 schema / 不过拟合 / 不降级(n-ary 不退 binary) / 复杂语义LLM+结构确定性规则 / 可溯源grounding
- 不指望框架原生支持（调研证伪）——核自建
- 风险：底座内核自建工作量大（mutation/contract/commit/并发/ledger/分层/准入/5-outcome/utility 全要），分阶段实现，先 contract+commit+ledger+分层（最小可用内核），再加并发/准入/5-outcome/utility
- 风险：EvoMemBench 警告 memory 收益非必然——底座建完必须诚实基线对比（不用下游，用 instance/schema 级边正确性+紧凑度，judge≠抽取模型）
- 风险：Zombie Agents——写入准入（recurring+replay+candidate-evidence）必须配套，防一次性注入持久劫持

## 验证（底座层）
- Mutation ledger 可 replay 重放到任意版本（time-travel）
- write contract 越权被拒
- transactional commit 失败回滚（无部分写）
- 跨域 namespace 隔离（granular 写不污染 ml）
- 5-outcome 裁决正确路由 insert/merge/relate/conflict/reject
- 现有 ARFM/跨域 bundle 可在底座内核上重放（兼容现有数据）

## 交付节奏
- 本块（整体骨架+底座内核）确认后 → 逐个 builder 部分（抽取→演化→对齐→富拓扑→维护）详设 → consumer 部分（论文检索优先，比赛紧）详设 → 协调/HITL → 实现
- 每部分 ExitPlanMode 过目，不直接动手
