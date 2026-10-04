# 评测重新设计: 独有能力断点全面扫描清单 (诚实)

Date: 2026-08-14
前提: 用户判断当前评测只评方法演化边(ERR/PSC)太简单, 要评独有能力(violation/自演化/富拓扑)凸显优势. 扫描发现这些独有能力在ARFM评测里数据流全断.

## 断点全清单 (按数据流)

### 断点1: labels丢失 (最底层, 影响violation+富拓扑)
- ARFM19篇用简化导出: nodes=[surface,role]二元组, 5142节点0个dict(无labels)
- 完整InstanceHypergraph(hg_out/PPR那篇)有labels: PROPERTY100/MATERIAL1/NUMERIC1/REGIME2
- 但ARFM19篇无完整instance存储(hg_out里0个ARFM)
- 影响: detect_constraint_violations要NUMERIC labels → 拿不到 → violation检测跑不出
- 影响: 富拓扑节点类型评测也拿不到labels

### 断点2: ARFM评测走lift_corpus不走agent.py主流程
- agent.py主流程调 split/merge/retire/rename + infer_pattern_compositions/constraints + detect_constraint_violations (独有能力全在agent.py)
- 但ARFM评测用lift_corpus(hypergraph_lifter.py): 只cluster+induce+judge, 不调这些
- 影响: 自演化操作(split等)+富拓扑(comp/cons)+violation 在ARFM评测全没产出
- 这是为什么评测只评了方法演化边——因为lift流程只产这个

### 断点3: NUMERIC抽取不积极 (抽取器问题)
- 完整存储PPR那篇103节点只1个NUMERIC
- prompt要NUMERIC(REQUIRED)但deepseek打得少
- 影响: 即使接完整instance, NUMERIC稀少 → violation检测样本不够

### 断点4: ARFM19篇完整InstanceHypergraph缺失
- hg_out只有1个instance(PPR, 别的实验), ARFM19篇0个
- 要评独有能力必须重抽19篇产出完整InstanceHypergraph(走extract_hypergraph+agent.py主流程)

## 好的部分(没断的)
- ARFM19篇edges有quals(relation_type/method/evidence_strength/cited_from) — 结构信号完整
- defines边有230条(19篇) — 定义边有, 只是节点没标NUMERIC
- gold有composition_edges(4)+law_constraints(41) — 可评富拓扑(但41条很多constrains=None要修)

## 根因诚实判断
之前评测"简单"不是因为设计偷懒, 是因为**ARFM评测走了lift_corpus简化路径, 绕过了agent.py主流程(独有能力全在主流程)**. lift流程只产方法演化边, 所以只能评方法演化边. 要评独有能力, 必须让ARFM走完整agent.py主流程(extract_hypergraph→evolution→split/merge/retire→infer comp/cons→detect violation), 产完整InstanceHypergraph.

## 修复优先级 (建议)
1. **重抽1篇走完整agent.py流程** (最小验证): 看extract_hypergraph能否产完整InstanceHypergraph(NUMERIC够不够)+violation/comp/cons能否跑通. 不盲目重抽19篇.
2. 修NUMERIC抽取(prompt强化/deepseek遵守)
3. 重抽19篇走agent.py主流程, 产完整instance
4. 设计violation评测(注入扰动)+富拓扑评测(gold comp/law对比)+自演化评测(frozen对照+生长曲线)
5. 扩输入样本

## 待用户定
这个断点扫描全了吗? 还有没有别的独有能力/数据流要查? 先做1(重抽1篇验证整链路)对吗?
