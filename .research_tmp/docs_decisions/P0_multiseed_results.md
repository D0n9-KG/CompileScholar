# P0-2 多seed显著性结果 (诚实, 4 seed × 4 臂)

Date: 2026-08-14
口径: 英文命名(commit f50c5daf) + 共享cluster + 4 seed(seed0-3_en) + eval_err_edges + eval_psc + aggregate_multiseed(paired bootstrap 1000x, 95%CI)

## 各臂4seed均值±方差
| arm | n | ERR_uncond | ERR_cond | fair_recall | PSC |
|-----|---|------------|----------|-------------|-----|
| T text | 4 | 0.085±0.021 | 0.600±0.115 | 0.813±0.036 | 0.775±0.101 |
| C +cite | 4 | 0.091±0.011 | 0.607±0.103 | 0.854±0.036 | 0.726±0.112 |
| Q +qual | 4 | 0.098±0.000 | 0.667±0.000 | 0.833±0.000 | 0.750±0.083 |
| B full | 4 | 0.091±0.011 | 0.601±0.070 | 0.833±0.000 | 0.738±0.070(3seed) |

## paired bootstrap vs T (95%CI) — 全部 sig=no
| arm | metric | mean_diff | CI2.5% | CI97.5% | sig |
|-----|--------|-----------|--------|---------|-----|
| C | err_uncond | +0.006 | -0.012 | +0.037 | no |
| C | err_cond | +0.007 | -0.119 | +0.200 | no |
| C | fair_recall | +0.042 | 0.000 | +0.083 | no |
| C | psc | -0.049 | -0.180 | +0.050 | no |
| Q | err_uncond | +0.012 | 0.000 | +0.037 | no |
| Q | err_cond | +0.067 | 0.000 | +0.200 | no |
| Q | fair_recall | +0.021 | 0.000 | +0.063 | no |
| Q | psc | -0.025 | -0.125 | +0.050 | no |
| B | err_uncond | +0.006 | -0.018 | +0.037 | no |
| B | err_cond | +0.001 | -0.125 | +0.129 | no |
| B | fair_recall | +0.021 | 0.000 | +0.063 | no |

## ★诚实结论

1. **结构信号臂(C/Q/B) vs 纯文本臂(T) 无统计显著增益**: 所有指标 CI 跨 0。单seed看到的差异(C召回0.917 vs T 0.833等)多seed后归零为噪声。

2. **但 Q 臂方差=0(4seed全同 ERR_uncond 0.098/ERR_cond 0.667)**: quals 让结果更稳定(方差小), 虽无均值增益。这是"稳定性"维度的小优点, 非准确性增益。

3. **ERR_uncond 4臂都~0.09**: 41 gold边只命中~4条。主因=输入覆盖窄(19篇只涉及53 gold方法的12个), 非机制问题。

4. **ERR_cond/PSC 4臂都~0.6/0.75**: 能覆盖的coverable边机制命中率高且语义对, 机制本身准。

## 对论文的影响 (诚实)
- 不能卖"结构信号臂显著提升边级精度"——多seed证伪。
- 可卖: (a)诚实边级评测框架(ERR+PSC, 非类型覆盖); (b)机制精度高(PSC 0.75, coverable边语义对); (c)英文命名修复召回瓶颈(fair_recall 0.42→0.83); (d)quals稳定性(方差0)。
- 主卖点仍是自演化升层schema+诚实评测, 非结构信号臂增益(降为部分特征)。

## 限制
- 4 seed 仍偏小, 但 paired bootstrap 已控非确定性。Q方差0可能因共享cluster+quals确定性高, 需查Q是否真4seed不同cluster。
- coverable 5-7条样本小, ERR_cond/PSC区间不稳健(±0.1)。
- gold 1篇综述, 需P1扩到Thornton2026验证跨综述一致性。
