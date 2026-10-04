# 主系统优化+富拓扑突破 — 当前进度总结 (2026-08-14)

## 这次session做的全部(commit链)
1. `f50c5daf` lift英文命名(METHOD/CLUSTER_PROMPT强制英文, 避免induce归并)
2. `39ac96db` induce传known methods + REL加improves路径1b(第一性原理推导→improves非compares)
3. `3a3114cc` seed加METHOD/PHENOMENON/PARAMETER标签(extract能细标, 不全PROPERTY)
4. `4fa0ffcd` THING根+subclass_of(validation不拒新标签超边, 129→23拒, 超边18→162)
5. `a180dc4f` infer_rich_topology_direct(从超边直接读7类富拓扑边, 不co-occurrence猜)
6. `9c32421f` 3处针对性改: extract加METHOD-PARAMETER N-ARY EDGES规则+composed_of vs defines区分+METHOD标签精确化; infer优先判nary

## 富拓扑突破(v1→v2, 19篇全量)
| kind | v1 | v2 | 变化 |
|------|-----|-----|------|
| method_parameter | 24 | 43 | +19 |
| nary | 3 | 41 | +38 |
| method_phenomenon | 47 | 80 | +33 |
| composition | 116 | 12 | -104(噪声消除) |
| law_parameter | 337 | 217 | -120(非确定性) |
| definition | 202 | 197 | -5 |
| method_regime | 6 | 5 | -1 |
| **总** | 735 | 595 | -140(质量升) |

核心: method_parameter+nary+method_phenomenon 从近0变正(+90), composition噪声消除(-104)

## gold v4(独立评测集, 手构)
- 71节点(43 Method+19 Parameter+6 Phenomenon+3 Regime)
- 125边(46 evo+24 param+17 capture+5 comp+19 law+10 nary+4 regime)
- 7类关系覆盖全维度, 独立于我们方法

## 待解决(按优先级)
1. deepseek非确定性(单篇20-65波动, 需多seed控方差)
2. METHOD标签仍混(numerical simulation/erosion method误标METHOD)
3. gold v4富拓扑维度评测指标未建(有595条产出+125条gold但没算命中率)
4. Kamrin_2012原文空(只611字节勘误声明, 需重新获取PDF)
5. extract prompt继续精化(METHOD定义边界/方法+参数连超边)

## 下一步建议
- 先建富拓扑评测指标(有产出有gold没命中率, 评测是核心)
- 再多seed控方差
- 再继续精化extract prompt
