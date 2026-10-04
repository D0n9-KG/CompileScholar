# P0-1 诚实 ERR 边级评测 — 汇总 (英文命名修复后, 共享cluster)

Date: 2026-08-14
口径: eval_err_edges.py (方法LLM对齐→41边逐条判) + eval_psc.py (语义对齐PSC)
替代旧"3/3类型覆盖"自欺指标。cluster各臂共享(同seed), induce/judge各臂独立。

## ★英文版4臂单seed (19篇, 共享cluster, commit f50c5daf后)

| arm | 公平召回 | ERR_uncond | ERR_cond | PSC | coverable | edge_hit |
|-----|----------|------------|----------|-----|-----------|----------|
| T text | 0.833 | 0.098 | 0.667 | 0.833 | 6 | 4 |
| C +cite | 0.917 | 0.073 | 0.429 | 0.571 | 7 | 3 |
| Q +qual | 0.833 | 0.098 | 0.667 | 0.833 | 6 | 4 |
| B full | 0.833 | 0.098 | 0.667 | 0.667 | 6 | 4 |

## 对比: 中文版4臂单seed (修复前, 同19篇)

| arm | 公平召回 | ERR_uncond | ERR_cond | PSC |
|-----|----------|------------|----------|-----|
| T text | 0.583 | 0.073 | 0.750 | 0.750 |
| C +cite | 0.417 | 0.098 | 1.000 | 1.000 |
| Q +qual | 0.667 | 0.098 | 0.500 | 0.375 |
| B full | 0.500 | 0.073 | 0.600 | 0.600 |

## 关键发现 (诚实)

1. **英文命名修复有效**: 公平召回 中文0.42-0.67 → 英文0.83-0.92 (显著提升, 召回瓶颈=induce命名归并已解决)。M17 I-gradient 现在对上 gold。

2. **ERR_uncond 仍低 (0.07-0.10)**: 41 gold边只命中3-4条。分母不公平——19篇只涉及53 gold方法的12个。诚实暴露输入论文覆盖窄。

3. **ERR_cond/PSC 中高 (0.43-0.83)**: 能覆盖的coverable边里机制命中率高且语义对(M11 extends M10, M15 improves M1, M8 compares M6 等)。机制本身准, 问题在覆盖不在判断。

4. **臂间差异不显著(单seed)**: T/Q/B 都是 ERR_cond 0.667 PSC 0.67-0.83, C 略低(0.429/0.571)。但单seed非确定性大(T/C同配置中文版都7vs5波动), **必须多seed+显著性检验才能定论**。

5. **C臂(citation)召回最高0.917但PSC最低0.571**: citation帮induce对齐更多方法(召回↑)但判断更粗(PSC↓)? 需多seed确认是否稳定。

## PSC vs ERR (诚实性)
PSC比ERR更严: 接受type措辞差异但判语义。Q臂曾 PSC 0.375 < ERR_cond 0.500 (id匹配但语义不对)。报PSC更诚实, id匹配可能高估。现统一报PSC。

## 限制 (诚实)
- 单seed: 各臂非确定性大, 上述差异不可靠 → P0-2多seed(3新seed进行中)
- coverable 6-7条样本小, ERR_cond/PSC区间不稳健
- gold 1篇综述(ARFM2024), 需扩3-5篇(P1, Thornton2026第2篇计划已写)

## 下一步
- 多seed(seed1-3_en)完成后: aggregate_multiseed --suffix _en --baseline T → 均值±方差 + paired bootstrap 95%CI
- 若C/Q/B vs T的CI跨0 → 诚实报"结构信号臂无统计显著增益"
- 若CI>0 → 该臂显著有效
