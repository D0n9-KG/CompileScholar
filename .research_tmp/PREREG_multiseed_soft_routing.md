# 预注册：多 seed × SOFT_ROUTING 两态判读门柱（2026-08-27，结果落地前定死）

背景：SOFTB 单 run 对照判决主门柱 FAIL（-5.4pt）但在单 run 方差带内（前科 37% 边数
波动），无法区分真降与方差。本实验 = 预注册"正式数字多 seed"的兑现，同时裁决改动①
（gate 放行 novel 边）降级为 SOFT_ROUTING 开关后的悬决。

## 实验设置（已启动）

- 6 runs：seeds {1,2,3} × SOFT_ROUTING {0,1}，同 5 篇（DQN/理论/实验/建模/短文），
  arm=full，修后管线（含批量修 5 类 + 对齐修复 faeae6a7+后续）
- 标签：MSEED_S{seed}R{0,1}
- 判分：API 双 judge（GLM-5-Turbo + qwen3.5，judge_cross.py，逐边三检 both-pass/either）

## 门柱（不许事后挪）

| 门柱 | 判 SOFT(R1) 胜的条件 |
|---|---|
| G1 主门柱：3-seed 均 both-pass | mean(R1) ≥ mean(R0)（配对，seed 内配对比较 R1 vs R0，3 对中 ≥2 对 R1≥R0 且均值不降） |
| G2 召回（边总量） | mean edges(R1) ≥ mean edges(R0) × 0.95 |
| G3 novel 通道活性 | R1 各 seed novel 升格 ≥1 且升格 pattern 被复用 |
| G4 divergence | R1 novel 类型数/篇 ≤ 8 |

**判读分支**：
- G1 过 + G2-G4 过 → 改动①转正（SOFT_ROUTING 默认 1 定稿），论文报两态消融
- G1 过但 G2-G4 有挂 → 保留开关，默认改 0，novel 通道仅作 schema 侦察
- G1 挂（R1 均值显著低于 R0，3 对中 ≥2 对 R1<R0 且均值差 >3pt）→ 改动①回滚（默认 0），
  ②③归纳通道保留吃 dropped 边
- G1 平（均值差 ≤3pt 且不一致）→ 两态等价，默认 0（保守），论文报"约束旋钮等价带"——
  这本身是有价值的负面/中性结果（对应调研"约束强度是旋钮"的实证）

## 附带产出

- 3-seed R1 数字 = FIX1 75.4%（单 run 内层口径）的方差带标定 + API 口径校准
  （内层 Claude judge vs API 双 judge 的 39pt 差在 6 bundle 上取仲裁样本）
- 多 seed 边数方差 → 案例库考卷的噪声带参考
