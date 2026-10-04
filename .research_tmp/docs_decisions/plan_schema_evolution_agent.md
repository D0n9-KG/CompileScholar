# Plan: schema 自演化 agent 详设（builder 群第 2 个）

## 形态：DIAL-KG 三阶段闭环 + SAGE writer-reader + schema-constrained rewrites
对标 DIAL-KG(Meta-KB 编排三阶段) + SAGE(writer-reader反馈) + 2608.18104(schema-constrained rewrites) + Metis/SEDM/MemoryBank/SCION(有控防膨胀)。

## 三阶段
1. **触发(Dual-Track)**: validate失败(no-matching-pattern真新结构) + recurring mismatch达阈值(EvolutionTrigger现有) + **consumer反馈(writer-reader,SAGE:发现schema gap提mutation提案进queue)** + split/merge/retire触发(现有detect)
2. **治理裁决**: gate多视角审——variant vs new-kind(role重叠+family+语义embedding) + candidate-evidence绑定(SCION,verbatim+rationale) + recurring crystallize准入(Metis,单次不进,recurrence达阈值才commit) + consensus(AI-Supervisor,corroboration或多视角投票) + HITL(新top-level family/不确定→interrupt)
3. **演化(apply)**: add/split/merge/retire/rename/add_subclass/relabel, **schema-constrained rewrites**(2608.18104,不破坏IS-A树/family归属/invariant,底座kernel校验), version bump + mutation ledger

## writer-reader闭环(SAGE)
consumer读底座→发现schema gap→提mutation提案queue→builder.evolver审+裁决+commit→后续consumer见新schema。演化不只builder触发,consumer也触发。

## 有控防膨胀
Metis recurring crystallize + SEDM utility剪枝 + MemoryBank遗忘 + SCION保守融合 + 现有conservative gate/MAX_ACTIVE_PATTERNS。

## 跟底座接
role=builder.evolver, write contract=add_pattern/split/merge/retire/rename/add_subclass(改T-box)。Mutation→transactional commit(schema约束+role权限+candidate-evidence+recurring crystallize+schema-constrained rewrite校验)。

## 创新点作用+ablation(守顶会)
schema自演化agent是"schema自演化并进"创新点核心。ablation必证:
- frozen vs 演化(质量/下游,⚠️P0实测结构信号臂无显著增益,演化收益非显然须证)
- 并进 vs 离线批量演化
- writer-reader闭环 vs 单向builder触发
演化触发通道(validate失败/recurring/consumer反馈/split-merge)做可开关方便ablation。

## 现有结构改
- validate_proposal加多视角consensus(role/family/语义embedding现有加强)
- 演化触发加consumer反馈通道(mutation提案queue+consumer提案接口,新)
- schema-constrained rewrite校验在底座kernel(contract 3),这里调
- recurring crystallize阈值gate(EvolutionTrigger现有累积+阈值)

## 验证(不用下游)
- ablation: frozen/演化/并进/离线/writer-reader/单向, 抽取质量+schema紧凑度+下游(论文检索)
- pattern数/redundancy/compound占比 before/after
- 演化质量: 新pattern被后续抽取复用率(utility)
- ⚠️诚实: P0多seed结构信号臂无显著增益——演化收益要真证明,否则创新点站不住

## 不做
不冻schema(丢演化=丢创新点); 演化触发通道可开关(ablation需要); 不强行迁就现有gate(多视角consensus该加加)
