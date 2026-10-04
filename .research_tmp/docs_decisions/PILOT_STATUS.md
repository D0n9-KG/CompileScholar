# Pilot 进度 (2026-08-14)

## 已验证
- [x] mineru_md_adapter: markdown→sections (ARFM 12 段干净, 公式保留)
- [x] 本地化 251 篇 ref markdown (ARFM147+GRA2008 104)
- [x] extract_hypergraph 加 evolve/propagate_intra_dag 开关 (commit b0e2cca8)
- [x] corpus_driver: 镜像 agent.py 编排 (共享meta+trigger+五操作+dep/con/comp)
- [x] smoke 单篇 Pouliquen1999 full: 74边/200节点/pat 6→13/三类拓扑全触发/split×2 ★
- [x] ★四臂试水5篇完成 核心创新指标全显现:
  | arm | he | acc | uniq_pat | split | merge |
  |-----|----|----|---------|-------|-------|
  | full | 959 | 17 | 38 | 12 | 0 |
  | add_only | 710 | 15 | 8 | 0 | 0 |
  | frozen | 764 | 0 | 6 | 0 | 0 |
  | no_intra_dag | 715 | 30 | 31 | 9 | 2 |
  - 形式化操作集(split): full 38pat vs add_only 8pat (split贡献30子类) ✓
  - 演化价值: frozen恒6pat/acc0 vs 其他都演化 ✓
  - intra-DAG传播: full 38 vs no_intra_dag 31 (有信号但不单调, acc反低) ⚠️需细析
- [x] 内容审视: split产物语义真实(influences→timing_shift带cause角色, constitutive_law_analytic_formula独立解析公式)
- [x] frozen Agarwal constitutive_law边含真实μ(I)本构定律verbatim

## gold 问题 (诚实)
- GLM-5.2 (reasoning) 对综述后段w17-w21 timeout/empty严重 (w7/w9/w17/w18/w19 skip, w20空产出)
- 前段w0-w16成功, known methods 53
- 补抽方案: gold_backfill.py 用 GLM-5(非reasoning)重抽全部窗口 (稳定)

## ★ NMR 虚高问题 (诚实发现, 用户直觉正确)
- full 臂原 NMR=0.868 (5篇含科普文+locomotion综述)
- 排除科普文/综述, 真颗粒流2篇 NMR=0.642 (低22点)
- 虚高来源: (1)科普文2005贡献982泛化节点(322短节点) (2)cos阈值0.55偏松
- 审视实际匹配: 真实对应(M45 RFT→RFT 1.0, M4 0.98)混杂松匹配噪声
  (M2→continuum model 0.79泛化撞, M22→dynamic RFT 0.50不相关)
- 结论: 流程跑通+四臂差异信号真实, 但绝对NMR值含20-30%松匹配噪声
- 修正: 需加BM25+rerank(Intern-Atlas三级检索)或提高阈值+类型约束
- pilot可行性结论不变, 匹配质量是下一步优化点

## 进行中
- [ ] gold w21 + MERGED (GLM-5.2, 后段残缺)
- [ ] gold_backfill (GLM-5 重抽, 补全后段)

## 待做
- [ ] gold 自审 (gold_audit.py)
- [ ] naive_rag 臂
- [ ] 匹配算 NMR/ERR (metrics.py)
- [ ] schema 层指标 (演化增益 vs frozen, 收敛稳定性)
- [ ] intra-DAG 信号细析 (为何 full acc<no_intra_dag acc)
- [ ] pilot 过则: 引专家闸门 + 扩全 5 综述 + 复刻 AgentCAT + 写论文

## 关键修正记录
1. 用 agent.py 式共享 meta 驱动 (非 build_hypergraph_from_sections 每篇重置)
2. GLM-5.2 是 reasoning 模型: 分段抽 + max_tokens=16384 + finish=length 翻倍 + 240s timeout
3. mineru_md_adapter cap max_sections=20 (防 91 节点科普文拖死)
4. gold 跑时独占 API (并发 timeout 严重)
5. 噪声论文 2005_So_much_more_to_know 在 ARFM refs 首篇 (Science 125问科普), 影响 5 篇试水观察, 全量时占比小

## 命令备忘
- gold: `python -u .research_tmp/gold_extract.py --survey ARFM2024`
- gold 自审: `python .research_tmp/gold_audit.py --survey ARFM2024`
- 臂试水: `python -u .research_tmp/corpus_driver.py --survey ARFM2024 --arm <full|add_only|frozen|no_intra_dag> --limit 5`
- naive: `python .research_tmp/naive_rag.py --survey ARFM2024`
- 指标: `python .research_tmp/metrics.py --survey ARFM2024`
