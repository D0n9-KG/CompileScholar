# Gold 抽取设计 (DECISION, 2026-08-14)

## 原则
gold 由 GLM-5.2 独立从综述 markdown 抽取，**绝不**用 extract_hypergraph（循环论证=废）。
gold 结构必须能匹配被测大图（extract_hypergraph 抽引用论文聚合）。

## 被测大图节点/边词汇（匹配目标，gold 要对齐）
大图 edge: {pat, nodes:[(surface,role)], ev, quals}
- pat ∈ {composed_of, influences, measures, defines, constitutive_law, claim_relation} (6)
- schema_topo ∈ {depends_on, constrains, composes}
- 节点 surface = 方法/概念名（自由文本，如 "μ(I) model"/"nonlocal granular fluidity"）

## gold schema (JSON)
```json
{
  "methods": [
    {"id":"M1","name":"μ(I) inertial rheology",
     "aliases":["mu(I) model","inertial rheology"],
     "refs":["MiDi 2004","Jop et al. 2006"]}
  ],
  "evolution_edges":[
    {"src":"M2","tgt":"M1","type":"extends",
     "evidence":"NGF...extends the μ(I) model to account for nonlocal phenomena"}
  ],
  "composition_edges":[
    {"whole":"M2","component":"M3","evidence":"..."}
  ],
  "law_constraints":[
    {"law_text":"μ(I) = μ_s + b I","constrains":"M1","consumer":"M2",
     "evidence":"the basic premise... μ ≡ τ/P ..."}
  ]
}
```

## 多任务 → 对齐三类 schema 拓扑
| 任务 | gold 边类型 | 映射大图 schema_topo | 创新点 |
|------|------------|---------------------|--------|
| A 演化链 | extends/improves/replaces/adapts (+compares/background 弱) | depends_on | 形式化操作集+intra-DAG |
| B 方法组成 | uses_component | composes | nary 超图 |
| C 定律约束 | law_constraints (我们独有) | constrains | 富拓扑 con |

> 演化链(A)是主卖点验证场；组成(B)借 Intern-Atlas uses_component；约束(C)是我们独有、Intern-Atlas 没有。

## 7 种 causal edge 语义 (Intern-Atlas 借)
- extends: B 在 A 基础上推广/扩展（A 是 B 的特例或前提）
- improves: B 改进 A 的精度/适用范围
- replaces: B 取代 A（A 被弃用）
- adapts: B 把 A 适配到新场景
- uses_component: A 使用 B 作为组件
- compares: A 与 B 比较
- background: A 以 B 为背景/动机（弱因果）

## gold prompt 要点（给 GLM-5.2）
1. 角色：领域专家读综述，抽取方法演化图谱（非逐句抽取，是综述级共识）
2. 节点：方法/模型/理论/定律（具名实体，给 aliases 方便匹配）
3. 边：只抽综述明确陈述的因果/组成/约束关系，每条带 evidence 原文片段
4. 定律约束：抽 constitutive_law（公式/定律文本）+ 它约束哪个方法 + 被哪个方法消费
5. 约束：不发明综述没说的关系；不抽纯背景介绍；refs 填综述引用标记（如 "Jop et al. 2006"）

## 匹配方式 (pilot 简单版)
- 节点匹配 NMR：gold method.name+aliases → bge-m3 cos vs 大图所有 node surface，阈值 τ（先 0.55）
- 边可达 ERR：gold edge 两端 method 在大图都匹配到节点 + 大图两节点间有 path（BFS，任一 pat 边作路径）
- PSC：LLM-judge 路径语义正确性（辅助，pilot 后期）

## 自审清单（pilot 阶段我做）
- [ ] 删幻觉边（综述没明确说的）
- [ ] 纠方向（extends 的 src/tgt 是否反了）
- [ ] 合并重名（同一方法不同写法）
- [ ] 核原文（evidence 是否 verbatim）
- [ ] 补 aliases（提升 NMR 匹配率，避免因表面写法不同漏匹配）
