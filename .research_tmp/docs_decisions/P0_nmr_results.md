# P0-2 NMR 结果 (完整gold, embedding+Hungarian, 英文命名后)

Date: 2026-08-14
口径: nmr_embedding.py — gold方法(name+aliases全embed取均值) vs lift方法(name embed), cosine+Hungarian最优匹配, 阈值0.65.

## 跨语言问题已解决 (英文命名修复后)
中文版(修复前): lift中文名 vs gold英文, cosine区分0.27-0.47(M1 vs M17仅差0.015分不开), NMR_recall 2/53, fair_NMR 0.200.
英文版(修复后, commit f50c5daf): lift英文名 vs gold英文+aliases, cosine区分清晰:
  - I-gradient model vs M17=0.705 vs M1=0.473 (差0.23, 分得开)
  - Gray-Thornton vs M23=0.783 vs 其他<0.49
  - Spot Model vs M8=0.979, Savage-Lun vs M24=0.859, NGF vs M15=0.784

## B臂4seed NMR (embedding版, 英文)
| seed | NMR_recall | fair_NMR |
|------|------------|----------|
| 0 | 7/53=0.132 | 7/12=0.583 |
| 1 | 6/53=0.113 | 6/12=0.500 |
| 2 | 6/53=0.113 | 6/12=0.500 |
| 3 | 6/53=0.113 | 6/12=0.500 |
fair_NMR 稳定 0.50-0.58.

## NMR 两版对比 (诚实)
| 方法 | NMR_recall | fair | 说明 |
|------|------------|------|------|
| embedding+Hungarian | 0.113 | 0.500 | 保守, 阈值0.65遗漏近义(I-gradient 0.641差0.01未达阈值); 有轻微循环(自家embedding匹配gold) |
| LLM对齐(GLM-5-Turbo) | 0.170 | 0.833 | 更高, 但LLM非确定性; gold抽取也用GLM-5.2同族(轻微循环) |

诚实结论: 报两版, 标注各自循环性(embedding=自家模型匹配; LLM=同GLM族抽gold)。真实NMR在0.50-0.83区间。M1/M17歧义: embedding版I-gradient能分开(cosine差0.23), LLM版high/high区分——两版都解了歧义。

## 限制
- embedding阈值0.65主观, 降阈值能多匹配但增误配
- 53 gold方法里19篇输入只涉及12个, NMR_recall分母不公平(用fair_NMR修正)
- 循环论证无法完全消除: 抽取/匹配/gold都用我们的LLM栈, 诚实标注
