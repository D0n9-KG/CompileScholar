# 夜间推进汇报 (2026-08-14, 你睡着时)

## 你睡前的问题：split 拉不开差距，分析抽取信息差距

### 我做了什么
1. 实现了 4 个 schema 层指标(轮廓系数/演化增益/收敛性/一致性)
2. 用轮廓系数(silhouette)测四臂 split 纯度 → 发现反直觉: **演化越多 silhouette 越低**
3. 深挖根因: 排除了 5 种可能解释(命名不收敛已修commit 3ce0f502/小簇偏差已排除/特征错配已对齐/科普文噪声已排除/指标bug已重写)
4. 修正了"split有害"的初步结论 → **silhouette指标与自演化价值方向相反**

### 关键诚实发现
- split 在 5 篇试水上没通过 silhouette 纯度验证(任何特征空间都不利 full)
- **但 silhouette 不适合测自演化**: 它系统性惩罚小簇(=新发现), 奖励保守不拆(frozen)
- full 实际 influences 299条未拆(89%), 只拆出37条小簇碎片 → 触发不足+碎片化
- 这不是"split拆错", 是"split触发不足 + 指标方向对立"

### 更根本的张力(需你决策)
Intern-Atlas 测的是**方法间演化关系**(extends/improves, 引用因果边)。
我们 extract_hypergraph 抽的是**篇内 nary 实体超边**(continuum model→intrusion)。
两者层级不同 → NMR/ERR 测不出 split 价值, silhouette 也不对路。

## commit 记录
- 3ce0f502: split 命名跨篇收敛(传meta列已有子pattern, LLM复用)
- b0e2cca8: extract_hypergraph 消融开关(evolve/propagate_intra_dag)
- a5fd47bc: embed_batch 容错(per-text 400零向量回退)

## 完整数据(schema silhouette, aligned+公平)
| arm | role+qualifier | entity-text |
|-----|---------------|-------------|
| full | 0.1057 | -0.0328 |
| add_only | 0.1630 | 0.0124 |
| frozen | 0.1585 | 0.0026 |
| no_intra_dag | 0.0342 | -0.0266 |

## 需你决策的方向(三选一, 我没独断)
A. 修 split 触发+判据: 提高influences拆分率, 按关系性质(因果/条件/依赖)拆非物理领域
   → 但即使拆好了, silhouette仍可能不利(指标方向问题)
B. 换评测: 放弃silhouette, 用下游任务(方法角色消歧/综述重建)验证split对下游有用
   → 最对路但要重新设计评测+可能要加引用因果边抽取
C. 重新审视创新定位: split可能在5篇+颗粒流上不是有效卖点
   → 主卖点转向intra-DAG传播(但acc信号也复杂)或nary超图

## 我的判断
B 最该做: silhouette已证明不适合, 继续调它无意义。
split真价值在"对下游有用"(角色消歧让检索更准), 不在统计纯度。
但B需要先想清: 评测什么下游任务能触到split价值, 且gold可得。

等你醒了定方向。当前所有脚本/数据在.research_tmp, 可复现。
