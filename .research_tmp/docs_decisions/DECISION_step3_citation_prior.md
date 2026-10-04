# DECISION: step3 — citation relation as judge prior (方案1)

Date: 2026-08-14
Goal step: 阶段顺序 step 3 — lift_corpus 用引用关系作结构信号升层，验证 vs 纯文本
用户拍板: 方案1（judge 先验）+ ARFM2024 建引用数据

## 核心设计（方案1: judge 先验，最小侵入）
引用是论文级、judge 是方法级。处理：lift_corpus 从 edges_by_method 反推
paper→method 映射（一个 paper 可属多族，任一匹配即算），对每对方法族 (A,B) 算：
  a_cites_b = A 族任一论文引用 B 族任一论文 (A 站在 B 肩上 → extends/improves/background 先验)
  b_cites_a = 反向
注入 judge_relation 的 REL_PROMPT 作【引用关系证据】段，LLM 综合建模形式/适用范围
判断不 override（引用是先验加分项，非硬规则）。诚实：引用≠方法继承（A引B未必extends B），
prompt 明确"引用是方法继承/背景的先验线索，仍以建模形式判断为准"。

## 代码改动（最小、可消融）
1. REL_PROMPT 增加 {citation_evidence} 段（默认"无引用关系"）。
2. judge_relation 加可选参数 citation_evidence: dict|None = None，
   含 a_cites_b / b_cites_a 论文对列表，格式化进 prompt。
3. lift_into_schema 加可选参数 paper_citations: dict[str,list[str]]|None = None，
   对每对方法族算引用方向传给 judge_relation。
4. lift_corpus 加可选 paper_citations 透传。
5. 评测 A/B 对照：
   - arm A = 纯文本（paper_citations=None，当前行为）
   - arm B = 文本+引用（paper_citations=citations.json）
   评判：gold 6 关系覆盖 + 引用链召回（A引B对里 B臂是否更多升出 extends/improves/background）
   同 LLM 受控（被测臂 deepseek，judge GLM-5，gold 已有）

## 数据
citations.json: 19 篇 ARFM2024 论文，OpenAlex resolve→register→fetch references，
匹配 in-corpus 引用边（A cites B）。build_arfm2024_citations.py 产出。

## 验证标准（纪律4）
A/B 对照跑同一 19 篇 edges_by_paper：
1. arm B 引用先验确实被注入（log 检查 prompt 含引用段）。
2. arm B 比 arm A 多/准升出 extends/improves（尤其 gold 里有 extends 的对：
   M17→M1, M11→M10, M21→M10 — 看这些方法族所在论文是否互引）。
3. 不虚报：若引用先验未提升 gold 覆盖，诚实承认引用信号在该语料不强。

## 不做
- 不改 cluster_methods_by_llm（引用只进 judge，不进聚类）。
- 不做候选生成+融合（方案2，后续若方案1不够再考虑）。
- 不把引用硬编码成关系规则（保持 LLM 判断）。
