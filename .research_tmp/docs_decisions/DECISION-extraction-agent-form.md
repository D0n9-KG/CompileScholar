# DECISION: 抽取 agent 重设计 = 重组形态(plan-execute-verify) + 加缺失件, 非推翻已验证内核

## 背景
设计补充 (plan_detailed_design_supplement.md section 4) + GOAL_MODE_BOOTSTRAP 偷懒前科都写"不默认沿用 step1/2/3 重设计抽取 agent"。但现有 `_run_hg_node_multistep` (entities→types→relations 三步) 已泛化验证 (memory: 4 跨域论文 GRANULAR LEAKAGE=0, ML/bio/fluid/molecular 全跑通, DQN 抽到 Bellman 方程/FCT 抽到 donor cell, commit 38f1ff24)。

## 张力
- "重设计抽取 agent" 若 = 推翻 step1/2/3 重写 → 破坏已验证内核 + 违反 surgical + 风险大 (重写易引入新 bug)
- "重设计抽取 agent" 若 = 搬现有流程进新框架 → 偷懒降级 (用户前科: "搬现有流程进框架当设计")

## 决策
**重设计 = 重组为 plan-execute-verify 形态 + 加缺失件(planner/verifier/skill/底座对接), 保留 _run_hg_node_multistep 作为 executor 内部实现 (surgical 复用已验证内核)。**

## 理由
1. 现有 step1/2/3 的**结构** (entities→types→relations) 是合理多步抽取, 避免一次 LLM 调用既要抽又要分类的过载。问题不在结构而在**缺 plan 前置 + verify 后检 + skill 注入 + 跟底座**。
2. "re-type 内化 verifier" = verifier 审 type 正确性 (retype:fix 修正), 而非删 step2。step2 独立标 type 合理 (先抽实体再分类)。
3. 不破坏已验证内核 (surgical + 不降级)。新形态 (plan-execute-verify) 是设计要求, 满足之; 内核复用是 surgical, 满足之。
4. 用户铁律"不搬现有流程进框架" = 不把 agent.py 整个搬进 LangGraph 当设计; 这里是把已验证的**抽取内核**作为 executor 内部实现, 外面包 plan-execute-verify 形态, 不是搬流程当设计。

## 实现
`src/granular_agent/extraction_agent.py`:
- planner: LLM → Plan{domain, central_entities, secondary_entities, expected_patterns, relation_outline, paper_anchor}. domain 注入点 (断点4). schema-in-context 含 semantic_boundary.
- executor: 委托 `_run_hg_node_multistep` (已验证内核), skill 注入 (SkillLibrary lookup by domain+pattern), plan.relation_outline 作 schema-in-context 提示.
- verifier: LLM ≠抽取模型, 每边审 verbatim_in_source + type_correct(re-type内化) + relation_exists → fix{keep/retype/reextract/drop}. 不进 5-outcome (断点2).
- fixer: 应用 fix (keep/retype 改 pattern_type/drop). reextract 诚实降级为 drop-with-note (一次性重抽未实现, 避免无限循环; keep/retype/drop 路径完整).
- commit_edges: add_edge Mutation (validate only 不 route, 断点2), domain 必带 (断点4), concept 实例内联 (断点7).
- skill distiller: 偶发, stability>0.6 蒸馏 (Step 3 evolver 走 distill_skill op commit).

## 诚实范围
- reextract fix 降级为 drop-with-note (一次性重抽未实现, keep/retype/drop 路径完整非降级). 标注诚实.
- skill distiller 在本模块是数据收集+提议, 真正 distill_skill op commit 由 Step 3 evolver transaction 做 (evolver write contract).
