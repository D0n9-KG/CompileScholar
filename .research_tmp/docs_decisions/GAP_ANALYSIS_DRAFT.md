# 缺口分析草稿 (2026-08-14, 待 agent 结果综合)

## 从各工作 limitations 提取的缺口

### Intern-Atlas 自承 limitation
- "does not capture structured relationships that explain how/why methods emerge, adapt, build upon"
  → 只做 citation-causal binary 边, 不抓方法涌现的结构化关系
- 方法演进是 DAG, 但它的因果边是 binary (非 n-ary)
- 需要 citation graph (我们颗粒流有综述引用, 可用)

### Hyper-KGGen 自承 limitation
- "context window limitations" → chunking 解决, 但没自演化 schema
- "fail across domains due to scenario gaps" → skill static per scenario, 不演化
- "balance structural skeletons with fine-grained details" → 结构骨架 vs 细节平衡难

## 初步缺口 (待 agent 确认)

### Gap 1: 收敛的自演化 schema 控制
- DIAL-KG: 有 merge/retire 但仍膨胀 (我们测 30篇 237 pattern)
- AgentCAT: ADD-only 避膨胀但丢 split 价值
- SCION: single-shot 避膨胀但不演化
- **无人有 principled + 收敛保证的 schema 演化控制**
- 我们已开始 (coherence gate), 这是真方向

### Gap 2: schema 演化质量评测
- Intern-Atlas: 评 coverage (NMR/ERR), 不评演化质量
- Hyper-KGGen: 评抽取 F1, 不评 schema
- **无人严格评测"自演化 schema 是否优于固定 schema" via 下游任务**
- 我们撞的就是这堵墙 (所有 schema 指标不可靠)

### Gap 3: 篇内 n-ary ↔ 跨论文方法演化的桥接
- Intern-Atlas: 跨论文 via citation-anchored (binary, 需引用图)
- Hyper-KGGen: 篇内 n-ary (无跨论文)
- **无人连接篇内 n-ary 抽取到跨论文方法演化**
- 这是我们层级错配的根因

### Gap 4: 强语义的 pattern 级拓扑
- 我们 dep/con/comp 弱 (共现推断)
- Intern-Atlas 有 citation-causal (强但 binary)
- **无人有 n-ary schema + 文本语义的 pattern 级拓扑**
- 但要大改 (拓扑从共现改成文本抽)

## 候选创新点 (诚实评估)

### A. 收敛的自演化 schema (攻 Gap 1)
- "让 schema 演化收敛" 而非 "做 schema 演化" (很多工作做)
- coherence-gated split + 下游验证 add + 激进 merge/retire
- 我们已部分做 (coherence gate)
- 风险: 收敛难证明

### B. 下游任务驱动的 schema 评测 (攻 Gap 2)
- 用下游任务 lift 评 schema 演化价值 (非内部指标)
- 我们有 survey-gold + 可建 QA 下游
- 方法论创新, 可独立成贡献

### C. 篇内 n-ary ↔ 跨论文方法演化桥接 (攻 Gap 3)
- n-ary 篇内抽取 + 跨论文方法链接 (共享实体)
- 我们 n-ary + 综述语料可做
- 但 Intern-Atlas 接近 (citation-anchored)

### D. 强语义拓扑 (攻 Gap 4)
- 文本抽的 pattern 级拓扑 (非共现)
- 大改但让拓扑真有用

## 初步推荐: A+B 组合
"收敛的自演化 schema + 下游任务驱动评测"
- 攻两个我们直接撞的缺口
- 已有部分工作 (coherence gate, survey-gold)
- 可辩护: 人人做 schema 演化, 无人让它收敛+正确评测

待 agent 结果确认缺口真实 + 补充 2025-2026 新工作。
