# DECISION: step4 — edge qualifiers as structural signal in lift

Date: 2026-08-14
Goal step: 阶段顺序 step 4 — 结构信号扩展（节点共现/qualifiers/拓扑/cited_from）

## 现状（已摸清）
edge 结构: {pat, nodes:[[surface, role]], ev, quals:{relation_type, method, evidence_strength, cited_from}}
当前 lift 完全不用 quals:
  - _fmt_edges (method induction) 只用 pat/nodes/ev/paper
  - _fmt_paper_edges (cluster) 同
  - judge_relation 只用 cross_mention (文本) + citation_evidence (step3 论文级引用)
quals 是 n-ary 超图的核心结构信息之一，浪费了。

## ARFM2024 quals 分布（已测，1701 edges）
  cited_from: prior_art 394 / this_work 1274 / definition 33
  evidence_strength: derived 891 / measured 431 / hypothesized 174 / assumed 205
  method: theory 1108 / simulation 255 / experiment 219 / review 119
  relation_type: 已含 extends 8 / outperforms 6 / contrasts 35 / supports 60 / derivation 37 ...

## 本 step 范围（纪律7 逐个加+验证，不一次全堆）
step4 只做 **cited_from + evidence_strength 聚合注入**（最贴合 step3 引用先验、最直接关系关系性质）：
  - 对每个方法族，聚合 quals 统计：evidence_strength 分布、cited_from 分布。
  - 注入 induce_method_node 的 METHOD_PROMPT（让 LLM 知道这方法的证据强度构成）。
  - 注入 judge_relation 的 REL_PROMPT（A/B 族证据构成对比，辅助判关系）。
节点共现/拓扑/cited_from方法名提取留 step4 后续子步。

## 设计
1. _quals_profile(edges) -> {evidence_strength: {derived:X%,...}, cited_from:{this_work:X%,prior_art:Y%}}
2. _fmt_quals_profile(profile) -> 简洁文本块
3. METHOD_PROMPT 加 {quals_profile} 段
4. REL_PROMPT 加 {a_quals}/{b_quals} 段（A/B 证据构成对比）
5. induce_method_node / judge_relation 透传 quals_profile

## 验证（纪律4）
聚焦对照复用 step3 的 eval_lift_citation_focused.py 框架：
  - arm A = step3 现状（citation 先验，无 quals）
  - arm B = +quals 聚合
  - 看是否提升 confidence 或细化 relation（尤其 evidence_strength=derived 居多的族→extends 倾向；
    measured 居多→compares 倾向；cited_from=prior_art 居多→extends/background 倾向）

## 不做
  - 不逐条 quals 注入（噪声大，用聚合统计）
  - 不动 cluster（step3 已说不动 cluster，只动 induction+judge）
  - 节点共现/拓扑/cited_from方法名 = step4 后续子步
