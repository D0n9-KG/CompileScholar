# Plan: 抽取 agent 详设（builder 群第 1 个）

## 形态：plan-execute-verify 闭环 + skill 演化（非 step1/2/3 线性）
对标 Hyper-KGGen(skill-evolving+stability feedback+Global Skill Library+n-ary) + Reflexion(self-verify) + plan-execute。
re-type/存在性gate/verbatim 全内化到 verifier，不独立。

## 子流程
1. **planner**(LLM): section+schema-in-context → 核心实体清单(central+次要)+预期关系大纲+paper-level 锚(central method/contribution)
2. **executor**(LLM 单任务, 结构化输出 pattern_type∈seed∪{new}/role∈allowed): 按 plan 逐条精抽(找关系+role+qualifier+抄evidence)
3. **verifier/critic**(LLM ≠抽取模型): 每边审 ①verbatim 在原文 ②类型对(re-type 内化) ③关系真(存在性 gate 内化) → 标 fix(重抽/改/删)
4. **fixer**(LLM): 按 critic 标的 fix
5. **skill distiller**(LLM 偶发, 阶段2): stability feedback 蒸馏稳定模式进 Global Skill Library; 不稳标 hard case 演化 skill

## 跟底座接
role=builder.extractor, write contract=只 add_edge(带 provenance)。Mutation→底座 transactional commit(schema约束+role权限+candidate-evidence绑定)。schema-in-context 来自 T-box(meta.to_prompt+semantic_boundary)。

## 现有结构要改(增量,不破坏n-ary/schema演化/provenance)
1. MetaHyperedgePattern 加 semantic_boundary 字段(schema-in-context 要清晰语义边界,治类型判断弱)
2. Hyperedge 加 provenance 子结构(evidence_span/cited_from/method/evidence_strength/source_paper 提一等公民,qualifiers只放业务键)
3. ConceptGraph 加 central entity 标记(paper-level锚显式,跨section复用不只靠surface)
4. Global Skill Library 新概念(Skill:domain/pattern/extraction_hint/stability_score/provenance_papers)

## 成本取舍
planner(1)+executor(逐边N)+verifier(逐边N≠模型)+fixer(部分)+skill distiller(偶发)。质量优先。
分阶段: 阶段1 核心(planner+executor+verifier+fixer) 治54%mistyped+14%改写+20%borderline; 阶段2 skill演化。

## 验证(不用下游)
边正确性审计(verifier≠抽取模型judge,before/after); compound 28%→?/sink 26%→?/verbatim改写14%→?/borderline20%→?; 4跨域+5颗粒流; skill复用stability提升。

## 不做
不冻pattern_type enum(new走演化); re-type内化verifier不独立; 不强行迁就现有结构(该加就加)。

## commit 节奏
1. 结构改动(semantic_boundary+provenance+central+skill library 增量加,不破坏现有)
2. 阶段1 抽取 agent(planner+executor+verifier+fixer)
3. 阶段2 skill distiller+Global Skill Library
每步 before/after 验证。
