# P0 完成检查表 (goal 8阶段进度)

## goal阶段进度
- [x] 阶段3 ERR边级评测 — eval_err_edges.py, 弃3/3类型覆盖
- [x] 阶段4 PSC路径语义 — eval_psc.py, 比ERR更严
- [x] 阶段5 多seed统计 — 4seed×4臂, aggregate_multiseed paired bootstrap
- [x] 阶段6 failure case分析 — failure_case_aggregate.py, 37/41漏边逐条诊断
- [x] 阶段2 NMR完整gold — nmr_embedding.py, 跨语言已解决, 两版报(embedding+LLM)
- [ ] 阶段1 gold扩建(3-5篇) — Thornton2026第2篇抽取中(小window重跑); 标注一致性kappa待做
- [~] 阶段7 quals拆维度消融 — 降级: 多seed已证quals整体无显著增益, 拆维度从略(论文说明)
- [ ] 阶段8 论文重写 — 基于真数字, 待gold扩建后

## 已完成诚实结果
1. ERR/PSC边级(替代3/3自欺): ERR_uncond~0.09(低,输入只覆盖12/53 gold方法分母不公平), fair_recall 0.83, ERR_cond/PSC~0.6/0.75(机制在coverable准)
2. 多seed显著性: C/Q/B vs T 全sig=no(结构信号臂无统计显著增益); Q方差0更稳定(英文命名cluster稳定)
3. failure case: 37/41漏边=33输入不涉及(公平)+3 type判错(improves/compares弱)+1配对遗漏; 4命中边全在覆盖范围
4. NMR: 跨语言已解(I-gradient vs M1 cosine差0.23), fair_NMR embedding 0.50/LLM 0.83, 两版报标注循环
5. 英文命名修复(commit f50c5daf): fair_recall 0.42→0.83, PSC 0.75→0.89

## 阶段1 gold扩建 (进行中)
- Thornton2026(segregation专题)第2篇gold抽取: 小window(3500)重跑, 避reasoning爆token
- 完成后: eval_cross_survey.py测跨综述一致性(ARFM vs Thornton同方法族)
- 标注一致性kappa: 诚实做LLM辅助+人工核验报协议(单LLM无法真两人, 或两不同prompt独立标算agreement)
- 第3篇候选: 2008 Forterre&Pouliquen或2018 suspension

## 阶段7 quals消融 (降级诚实)
多seed已证quals整体vs纯文本无显著增益(sig=no), 拆evidence_strength/cited_from/method单独消融预期也不显著。
诚实策略: 论文里说明"quals整体已证无统计显著增益(P0-2 paired bootstrap), 拆维度消融从略", 避免无价值实验。
若审稿要求, 再做: _fmt_quals_profile加quals_dims参数透传, 跑3臂(evidence_strength only/cited_from only/method only)多seed。

## 阶段8 论文重写 (待gold扩建后)
基于P0真数字重写PAPER_draft_step8相关章节:
- 主结果表用ERR/PSC/fair_recall(非3/3类型覆盖)
- 主卖点改: 诚实评测框架+机制精度+英文命名修复召回+quals稳定性(非结构信号增益)
- threat to validity: LLM非确定性(多seed控制)/gold自建规模/gold循环/embedding循环
- failure case: 37/41漏边诊断入discussion
