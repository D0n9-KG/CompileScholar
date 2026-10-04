# 主系统优化分析 (抽取+schema自演进)

Date: 2026-08-14
前提: gold已定版(125边覆盖全维度), 回到主贡献系统优化. 之前发现5个问题.

## 主系统完整链路 (agent.py process_paper_hypergraph)
extract_hypergraph(结构map→n-ary超图) → evolution(split/merge/retire/rename)
→ infer_pattern_dependencies/constraints/compositions → detect_violations
→ 存InstanceHypergraph

## 5个已发现问题的根因深入分析

### 问题1: 结构信号臂无显著增益(多seed sig=no)
**现象**: C/Q/B vs T 全sig=no, quals/citation没提升ERR/PSC
**根因(诚实)**: quals注入是prompt hint, judge可完全忽略
- REL_PROMPT line98-101: "证据构成是结构线索...但最终仍以建模形式/适用范围判断为准"
- judge被允许忽略quals, 所以quals没真起作用
- citation同理: "引用先验仅作倾向提示, 最终以建模形式判断为准"
**这是设计问题**: 信号是"建议"非"约束", judge按文本判断就够, 信号冗余
**深入优化方向**:
- 方向A: 让judge必须引用quals作依据(强制: 输出rationale必须含qual证据)
- 方向B: 改信号用途 — 不用于judge type, 用于recall(过滤: prior_art高的方法才参与extends判断)
- 方向C: 诚实接受 — 信号臂对ERR/PSC无增益, 转卖稳定性(Q方差0)和其他维度
**诚实判断**: A/B都是强行让信号有用, 可能反而降准. C最诚实. 但要再看: 信号是否在n-ary/参数/现象这些新维度有用(之前只评了方法演化边)

### 问题2: induce命名归并(I-gradient+NGF→同名)
**现象**: 英文命名后改善, 但M17仍部分漏(cluster把I-gradient和NGF归同名)
**根因**: induce单次归纳, 两族独立induce可能产出同名(都是"nonlocal granular fluidity")
**深入优化**: induce时传入"已归纳的方法名列表", 要求新方法名与已有区分(类似gold_extract的known methods)
**难度**: 低, 改METHOD_PROMPT传known方法名

### 问题3: improves vs compares type判断弱(M26全判错compares)
**现象**: M26(Sarkar-Khakhar) improves M23(Gray-Thornton) 被判compares×16
**根因**: REL_PROMPT的决策路径对"第一性原理推导系数"算improves还是compares不清晰
- M26从first principles推导系数, 是对M16的改进(更准系数), 但综述说"derived coefficients"非明说improves
- judge按字面"derived"判compares(推导vs对比), 没判出"更准系数=improves"
**深入优化**: REL_PROMPT决策路径补"第一性原理推导更准参数/系数 → improves"
**难度**: 低, 改prompt决策路径

### 问题4: cluster同族过度合并
**现象**: segregation各家(Gray-Thornton/Savage-Lun/Sarkar-Khakhar)被并入M3
**根因(英文命名后已部分修)**: 英文命名让cluster区分了, 但仍可能合并近义
**深入优化**: cluster prompt加"按提出者区分同族变体"(已加英文命名, 确认效果)
**难度**: 已修, 验证

### 问题5: NUMERIC抽取不积极
**现象**: Midi完整流程271节点只1个NUMERIC(violation检测样本不够)
**根因**: 抽取器prompt要NUMERIC但deepseek打得少, 或存格式问题
**深入优化**: 
- 强化prompt(明确: μ_s/b/I_0/A/ξ等系数必须NUMERIC)
- 抽取后post-process: 正则识别公式里的系数符号补NUMERIC标签
**难度**: 中

## 新维度问题(gold v4后浮现)
gold现在有参数/现象/n-ary, 但主系统之前只评方法演化边. 这些新维度的抽取质量未知:
- 参数关系(uses_parameter): extract_hypergraph抽得出吗?
- capture(方法→现象): 抽得出吗?
- n-ary: 超图能产n-ary吗? agent流程的infer_pattern_*产n-ary吗?
**深入优化**: 先跑完整流程, 看这些维度产出质量, 再针对性修

## 优化优先级建议(诚实)
1. **先跑完整流程看新维度产出** — 不盲目改, 先看参数/现象/n-ary在主系统抽得怎样
2. **问题3(improves/compares弱)** — 改prompt决策路径, 低成本高收益(M26这类type错影响ERR)
3. **问题2(induce命名归并)** — 传known方法名, 低成本
4. **问题1(信号臂)** — 看新维度是否有用, 若仍无用诚实接受降为稳定性卖点
5. **问题5(NUMERIC)** — 强化prompt+post-process, violation检测才可用
6. **问题4(cluster)** — 已修, 验证

## 待用户定
这个分析全吗? 优先级对吗? 先跑完整流程看新维度, 还是直接改问题2/3(prompt低成本)?
