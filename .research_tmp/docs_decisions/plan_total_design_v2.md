# 总设计 v2（重锚后自洽版，取代分散 plan）

## 主卖点（重锚，三证据确认方向成立）
**n-ary 超边语义保真 + schema 并进演化 + agent mutable working memory 三点同时成立的系统**，正面引 Intern-Atlas(2604.28158) 作 binary 大规模先例（103 万篇 + 6/6 边类型重合 + verbatim 闸门 + lineage 评测，最硬竞品）head-to-head。差异三点：
1. **n-ary 富拓扑**表达 binary 无法表达的多体 method-param-phenomenon 关系；富拓扑直读（从超边读 7 类 vs binary co-occurrence 猜噪声）是 n-ary 红利，memory 已实测 win（735→595 噪声降）——**最稳兜底卖点**
2. **schema 并进自演化** vs Intern-Atlas fixed-at-release——长尾新方法族命中（机制层指标，不赌 aggregate 下游，P0 风险对冲）
3. **agent mutable working memory** vs Intern-Atlas read-only data layer——多 consumer 共享写回

schema 演化降为支撑机制（锚机制层指标 + 下游不输，不赌下游显著增益）。MINDSET related work 一句话澄清（名同实异，不占真空）。

---

## 底座内核（8 机制 + 断点修正）

三层数据（**Skill Library 提为第四层**，断点 5 修）：
- T-box：MetaHypergraph（pattern+role_slots+allowed_qualifiers+family+IS-A+version）+ **semantic_boundary 字段**（schema-in-context 治类型判断弱）
- A-box：ConceptGraph（Concept+definition 字段+central 标记 / ConceptHyperedge n-ary+**provenance 子结构一等公民**：evidence_span/cited_from/method/evidence_strength/source_paper；qualifiers 只放业务键）
- provenance：随 Hyperedge 一等公民 + mutation ledger
- **Skill Library（第四层，新）**：Skill(domain/pattern/extraction_hint/stability_score/provenance_papers)，受 evolver write contract

**Mutation 对象**（断点 1/5/7 修）：op enum = {add_node, add_edge, add_pattern, add_subclass, split, merge, retire, rename, relabel, **distill_skill**, align_merge, add_concept_relation, conflict_mark}。aligner op 进 enum（断点 1 修）。add_node 归 evolver（node 类型）/ extractor add_edge 附带 concept 实例（断点 7 修）。

**Write Contract**（role→op）：
- extractor: add_edge（**必带 domain**，断点 4 修；走 candidate-evidence+schema 约束，**不进 5-outcome**，断点 2 修）
- evolver: add_pattern/split/merge/retire/rename/add_subclass/distill_skill/add_node（改 T-box + Skill）
- aligner: align_merge/add_concept_relation/conflict_mark（**只 aligner 进 5-outcome**，断点 2/8 修）
- maintainer: retire/relabel（utility 剪枝）
- consumer.*: 只读 + 提 mutation 提案进 queue

**Transactional Commit 分两阶段**（断点 3 修，关键修正）：
- **validate 阶段**（pass/fail 原子）：schema 约束 + role 权限 + runtime invariant + **A-box referential integrity**（dangling node_ids 拒，断点 11 修）+ candidate-evidence 绑定 + recurring crystallize（Metis）。任一失败回滚。
- **route 阶段**（5-outcome，非原子逐边）：只对 aligner 的 A-box 写。insert/merge/relate/conflict/reject，reject=合法丢弃（写 conflict_mark 或不写），不回滚整批。**5-outcome 只 aligner 进**（断点 2/8 修，降成本）；规则+embedding 先过滤，只 ambiguous 才上 LLM。

**Mutation Ledger**：who/when/why/evidence/version_diff/domain，可 replay/time-travel/落盘暂存。

**并发控制**（断点 10 修，收窄）：builder 写串行（保 schema 演化语义）；**去掉 builder-builder CAS**（串行无并发）；只保留 consumer 读带 version stamp 做 snapshot read。接口预留 CAS（Mutation 带 base_version）但 builder-builder 不触发。

**跨域分层 namespace**（HASTE+DecentMem）：meta:global（通用 seed 12 共享）+ meta:granular/ml/molecular（域专属）；domain 标签**注入点 = extractor Mutation 必带**（断点 4 修，从 paper metadata/planner 指定）；对齐只同域。

**schema-constrained rewrite**（断点 9 修，诚实收窄）：演化校验**只 IS-A 树完整/family 归属/runtime invariant**，**不声称对标 2608.18104 全部**（2608.18104 要建模 consumer contract，重，不做）。诚实写"IS-A/family invariant preserved"。

**5-outcome claim 裁决**（MELD）：aligner A-box 写入走 insert/merge/relate/conflict/reject（claim-key identity+embedding+NLI/LLM）。

**Utility 剪枝**（SEDM）：retire 按 utility（频次/引用/演化贡献）。

---

## Builder 群

**抽取 agent**（plan-execute-verify+skill 演化，re-type 内化 verifier）：
- planner（LLM）：section+schema-in-context(pattern+semantic_boundary+例子标举例非规则)→核心实体清单(central+次要)+预期关系大纲+paper-level 锚。**planner 指定 domain**（断点 4）
- executor（LLM 单任务，结构化输出 pattern_type∈seed∪{new}/role∈allowed）：按 plan 逐条精抽，**Mutation 必带 domain**
- verifier/critic（LLM ≠抽取模型）：每边审 verbatim 在原文+类型对（re-type 内化）+关系真（存在性 gate 内化）→标 fix。**不进 5-outcome**（extractor 边走 candidate-evidence+schema，断点 2）
- fixer：按 critic 标 fix
- skill distiller（LLM 偶发，阶段 2）：stability feedback 蒸馏进 Skill Library（**走 distill_skill op，受 evolver contract**，断点 5）
- 跟底座：add_edge Mutation→validate 阶段（不 route）

**schema 自演化 agent**（DIAL-KG 三阶段+SAGE writer-reader）：
- 触发：validate 失败 + recurring mismatch + **consumer 反馈（writer-reader，SAGE）** + split/merge/retire 触发。**所有触发源（含 evolver 自身 validate/recurring）统一进同一 queue，builder 单消费者串行出队**（断点 6 修）
- 治理裁决：variant/new-kind（role+family+语义 embedding 多视角）+ candidate-evidence + recurring crystallize + consensus + HITL
- 演化：add/split/merge/retire/rename/add_subclass，schema-constrained rewrite（IS-A/family invariant，不声称 2608.18104 全部）
- 跟底座：evolver Mutation→validate 阶段

**跨论文对齐 agent**（EDC+MELD 5-outcome）：
- Define（LLM）：concept 生成 definition
- Canonicalize（embedding+LLM judge ≠抽取模型）：同域内找候选→**5-outcome**（只 aligner 进，断点 2）
- 增量对齐（每篇只对齐新 concept）；跨域不混
- 跟底座：aligner Mutation→validate（pass/fail）+ route（5-outcome，断点 3 分阶段）

**富拓扑推断**（直读+ambiguous 规则）：
- 7 类直读保留（确定性 win）；**ambiguous 判定规则明确**（断点 12 修：>1 类 role 重叠度>0.5 且证据同句才上 LLM，不总问）
- 作 consumer working memory（DocTrace 式 on-demand 取子图，非全量预计算）
- 带域 namespace

**维护 agent**（utility+保守融合）：
- retire 按 utility（SEDM）；split/merge 触发加 utility+语义判据；consolidate 域门控保留；保守融合（SCION）
- **衰减 reaper**（断点 13 修）：要么协调层加 background reaper 周期降权，要么诚实降级"retire=软删+ledger 可恢复，不主动衰减"（去 MemoryBank 对标）。**先诚实降级**，reaper 后加。

---

## Consumer 群（全 grounded）

**论文检索（华为赛，优先）**：查询分解（schema 概念 map 方法论约束）+ API 检索（粗，Semantic Scholar/OpenAlex）+ 超图/schema 关联推理重排（**evolution 边+富拓扑做细粒度关联**，差异化核心）+ LLM agent 迭代策略（PaSa/SPAR 式）+ 综合排序+结构化归纳+grounding 三处。**grounding"查询理解处"明确**（断点 14：=查询词映射 schema concept + 回源 concept 定义）。诚实张力：超图是增强重排层非检索源，覆盖度偏颗粒流。

**QA**：精确检索（query→超图子图 on-demand）+ 多跳推理（PPR over 超图）+ grounded 回原文（PaperQA provenance）。

**综述**：广覆盖+跨篇综合（超图跨论文 concept+evolution 链）+ 结构化（shared revisable state, DAS）+ 反思（SciSage Reflector）+ 可溯源。

**冲突检测**：交叉对比矛盾 claim + 5-outcome conflict 入口（aligner 标的 conflict 走这）+ embedding+NLI 判矛盾 + 报告+可溯源。

**演化分析**：方法族+evolution 链遍历（extends/improves/compares）+ 演化路径+图谱+可溯源。**正面对照 Intern-Atlas lineage reconstruction**（我们的 n-ary + schema 演化 vs 它 binary+static 在 evolution chain 重建的增益）。

---

## 协调/HITL/可观测
- **queue 统一串行**（断点 6）：所有 mutation 提案（consumer + evolver 自触发）进同一 queue，builder 单消费者串行出队
- HITL（MemGPT interrupts）：新 top-level family/不确定/re-type 抽查暂停人审
- 可观测：mutation ledger tracing + checkpoint + replay（暂存自动化+崩溃 resume）
- 框架：LangGraph StateGraph+Command+Store 指向 KB 内核（编排脚手架，核自建）

---

## ablation 接入（预注册 plan_ablation_preregistration.md）
- **A1 n-ary vs binary**（主对照 Intern-Atlas 式 binary baseline）——最稳兜底，富拓扑直读质量+语义保真+co-occurrence 噪声率
- **A2 schema 并进 vs static**（Intern-Atlas static）——schema 紧凑度+新 pattern 复用率+演化链追溯+长尾命中，**不赌 aggregate 下游 ERR**，备负面 fallback
- **A3 富拓扑直读 vs co-occurrence**（Higher-Order 2601.04878 baseline）——7 类边质量+稳定性，已 win
- ≥4 seed paired bootstrap 95%CI（复用 P0 脚本）；judge 交叉（GLM+qwen，不一致进 HITL）；qualitative+长尾对冲 aggregate 稀释；预注册承诺不事后挑指标

---

## 实现顺序（守全部设计好再动手）
1. 底座内核（Mutation+Contract+Ledger+分层+distill_skill op+A-box referential integrity；5-outcome 分两阶段；CAS 收窄；schema-constrained rewrite 收窄 IS-A/family）
2. 抽取 agent（阶段 1 planner+executor+verifier+fixer；semantic_boundary/provenance/central 字段改）
3. schema 自演化 agent（queue 统一；DIAL-KG 三阶段；SAGE writer-reader）
4. 跨论文对齐 agent（EDC+5-outcome 只 aligner；Concept 加 definition）
5. 富拓扑+维护（ambiguous 规则；utility 剪枝；衰减诚实降级先）
6. consumer（论文检索优先，比赛紧；QA/综述/冲突/演化分析）
7. 协调/HITL/可观测
8. ablation（A1/A2/A3 + 主对照 Intern-Atlas + 机制层指标）

## 不做（守铁律）
不冻 schema/不过拟合/不降级(n-ary 不退 binary)/复杂语义 LLM+结构确定性规则/可溯源 grounding；不声称对标 2608.18104 全部（诚实收窄 IS-A/family）；不假装发明方法演化关系（正面引 Intern-Atlas）；不赌 schema 演化下游显著增益（P0 风险，锚机制层）。
