# DECISION: schema 自演化 agent = 委托现有 evolution 内核 + 走底座 transaction + 加 queue/SAGE/三阶段组织

## 背景
现有 `hypergraph_evolution.py` 内核很完整: EvolutionTrigger(P1 cross-node recurrence) / evolution_probe(P2 LLM提案) / validate_proposal(P2/P3 evidence+near-dup+distinctness gate) / apply_proposal(P4 mutate meta) / run_evolution_loop / run_split/run_merge/run_retire/run_rename / detect_*_triggers / infer_pattern_dependencies/constraints/compositions / consolidate_instance / infer_rich_topology_direct.

设计补充要求 Step 3 schema自演化agent: DIAL-KG三阶段(治理裁决) + SAGE writer-reader + 统一queue(builder串行) + schema-constrained rewrite收窄 + distill_skill真commit + 走底座KnowledgeBase.commit.

## 张力
- 现有 run_evolution_loop 直接 mutate meta (meta.add_pattern/split_pattern...), **没走底座 transaction** (不进 ledger, 不过 validate, 不 CAS, 不可 replay/time-travel)
- 现有 apply_proposal 的 add_pattern 没传 semantic_boundary (Step2 加的字段)
- 现有没统一 queue (evolver 自触发 + consumer 反馈各走各的)
- 现有没 SAGE writer-reader (consumer 反馈触发)

## 决策
**EvolutionAgent = 委托现有 evolution 内核(trigger/probe/validate_proposal/detect_triggers)做"裁决", 但 apply 阶段走底座 KnowledgeBase.commit(evolver Mutation→validate→apply) 而非直接改 meta. 加统一 queue + SAGE writer-reader 收集触发源 + distill_skill 真commit. surgical 复用裁决内核, 不重写.**

## 理由
1. 现有 evolution_probe/validate_proposal/detect_*_triggers 是验证过的裁决逻辑 (smoke test 覆盖, P0 多seed用过). 重写风险大 + 违反 surgical.
2. "走底座 transaction" 是 Step1 kernel 存在的意义——evolver 改 T-box 必须进 ledger/过 schema-constrained validate/可 replay. 现有 run_evolution_loop 绕过 kernel 是历史遗留 (kernel 是 Step1 才加的). Step3 让 evolver 走 kernel = 闭环设计.
3. apply_proposal 直接改 meta → 改为: 把 proposal 转成 evolver Mutation, kb.commit([Mutation]) 走 validate+apply. kernel _apply 已委托 tbox.add_pattern/split/merge/retire/rename (Step1 实现). 但 kernel add_pattern 要传 semantic_boundary——需要 proposal 生成 boundary (LLM) 并放进 Mutation.payload.
4. 统一 queue: EvolutionAgent 维护一个 _queue, 触发源 (validate 失败/recurring/consumer SAGE 反馈/自触发) 都 propose EvolutionTrigger 项进 queue, drain() 串行出队走 commit.
5. SAGE writer-reader: consumer 反馈 (e.g. "这个 pattern 在我查询里总是 miss") 作为 writer-reader 信号触发 recurring mismatch 检测. Step3 实现 collect_consumer_feedback 接口 (consumer 调), feedback 累积触发演化提案. (consumer 本身 Step6 才做, 但接口预留)
6. distill_skill 真commit: skill distiller 收集同 domain+pattern 抽取共性, stability>0.6 时 propose distill_skill Mutation 走 evolver contract commit (kernel Step1 已有 op+阈值校验).
7. schema-constrained rewrite 收窄: kernel _schema_constraint 已对 evolver op 做 IS-A/family/invariant 校验 (Step1). Step3 不重复, 委托 kernel.

## 诚实范围
- DIAL-KG 三阶段在代码里是 probe→governance→apply 三步函数, 不是独立服务. 三阶段逻辑委托现有 probe+validate+新加 governance(consensus/HITL).
- HITL: 预留接口 (新 top-level family / 不确定 re-type 抽查), 真正 HITL 由 Step7 协调层接. Step3 只标记 needs_hitl 不阻塞.
- recurring crystallize: 现有 EvolutionTrigger 的 cross_node_count/cumulative_count 已是 recurring 检测. crystallize = recurring>=阈值 才接受 growth (现有 CONSERVATIVE_CROSS_NODE gate). 委托.
- consumer SAGE 反馈: 接口预留 collect_consumer_feedback, 真 consumer Step6 做后接入. Step3 不能端到端验证 SAGE (无 consumer), 但 queue 统一机制可测.
