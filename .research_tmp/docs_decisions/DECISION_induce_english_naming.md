# DECISION: lift 方法命名改英文 + 抑制 induce 归并

Date: 2026-08-14
Branch: research/era-reconstruction

## 决策
改 `cluster_methods_by_llm` 的 CLUSTER_PROMPT 和 `induce_method_node` 的 METHOD_PROMPT:
1. 方法命名必须用**英文学术术语**（如 μ(I) rheology / I-gradient model / nonlocal granular fluidity (NGF) / Savage-Lun kinetic sieving），不用中文。
2. 明确要求**近义但不同的方法必须用不同名区分**（I-gradient model ≠ NGF model；各 segregation 模型按提出者命名区分）。

## 触发证据（P0-1 诚实 ERR, T臂19篇单seed）
召回不足 fair_recall=7/12=0.583。3 层归因（逐层确认）：

1. **cluster 层（次因）**: 14篇合并, 19篇已分得够细——cluster 给了 "非局部梯度扩展本构律" 和 "非局部流体性本构律" 两族（I-gradient vs NGF 分开了）。cluster 不是主因。

2. **induce 层（主因）**:
   - **归并**: 两族 induce 都输出 "nonlocal granular fluidity model"（同名!）→ M17 漏抽根因。
   - **命名非确定性**: T/C 共享 cluster + 同 induce 配置(use_quals=False), 但 2 个方法名不同（"动理学理论的密堆扩展" vs "动理学理论驱动的颗粒流本构关系与能量平衡模型"）→ 对齐覆盖波动 T 7/12 vs C 5/12。
   - **中英文混乱**: segregation 5 族 induce 出 "统计力学分选/密度驱动分选/颗粒温度分选/经验分选" 等中文名, 对齐都错对 M3。

3. **对齐层**: induce 中文名 vs gold 英文名近义错配（segregation 4族→M3）。

## 为什么修 induce 命名比多 seed 优先
单 seed 已暴露 induce 命名是召回瓶颈。修前多 seed 只确认"低召回稳定", 无价值; 修后召回若显著提升, 再多 seed 才有意义。先修瓶颈再控方差。

## 验证标准
- fair_recall 0.583 → ? （目标: segregation 各家对上 M23/M24/M26, I-gradient 对上 M17, fair_recall 显著升）
- induce 非确定性波动降低（多 seed 同族产出同名率提升）
- embedding NMR 跨语言问题同步解决（[[embedding-nmr-crosslingual-fails]]）

## 改动范围（待实施）
- `src/granular_agent/hypergraph_lifter.py`:
  - CLUSTER_PROMPT (line 556-590): 例子改英文 + 加"不同方法不同名, 按提出者/建模形式区分"指令
  - METHOD_PROMPT (line 46-64): method_name 强制英文 + 区分指令
- 不改 induce 逻辑（仍重新归纳）, 只改 prompt 约束命名

## 风险
deepseek 英文归纳可能不如中文准。mitigation: rationale 仍可中文, 只 method_name 英文; 改后 T臂单seed先验证 fair_recall 不降反升再全面。
