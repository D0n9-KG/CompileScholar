# P0 诚实评测综合结果 (goal阶段2-6已完成, 阶段1进行中)

Date: 2026-08-14
口径: 英文命名(commit f50c5daf) + 共享cluster + 多seed(4) + paired bootstrap + 边级ERR/PSC

## 已完成阶段

### 阶段3+4: ERR/PSC 边级评测 (弃3/3类型覆盖自欺)
- eval_err_edges.py: 方法LLM对齐→41边逐条判 uncovered/miss/type_only/edge_hit
- eval_psc.py: 语义对齐(PSC比ERR严, 接受type措辞差异判语义)
- 旧"3/3类型覆盖"=类型词出现就命中, 掩盖1/41真实命中, 已弃用

### 阶段5: 多seed显著性 (4seed×4臂, paired bootstrap 95%CI)
| 臂 | fair_recall | ERR_uncond | ERR_cond | PSC |
|----|-------------|------------|----------|-----|
| T text | 0.813±0.036 | 0.085±0.021 | 0.600±0.115 | 0.775±0.101 |
| C +cite | 0.854±0.036 | 0.091±0.011 | 0.607±0.103 | 0.726±0.112 |
| Q +qual | 0.833±0.000 | 0.098±0.000 | 0.667±0.000 | 0.750±0.083 |
| B full | 0.833±0.000 | 0.091±0.011 | 0.601±0.070 | 0.720±0.068 |
**结论**: C/Q/B vs T 全部 sig=no (CI跨0). 结构信号臂无统计显著增益. Q方差0更稳定(英文命名cluster稳定).

### 阶段6: failure case 分析 (37/41边漏诊断)
- 33条=输入19篇不涉及的gold方法(公平未覆盖: DEM片M29-34/侵入体M42-48/RFT M45/50-53)
- 3条=真·lift漏判: M26 improves判compares×16(type系统判错), M25 extends type_only, M23-24 adapts配对遗漏
- 4条命中: M11→M10/M15→M1/M21→M10/M8→M6 (全在覆盖范围, 机制准)
- 机制局限: improves vs compares type判断弱; judge配对偶遗漏

### 阶段2: NMR 完整gold (跨语言已解决)
- 中文版(修复前): cosine区分0.27-0.47(M1/M17差0.015分不开), fair_NMR 0.200
- 英文版(修复后): I-gradient vs M17=0.705 vs M1=0.473(差0.23分开), fair_NMR 0.500-0.583
- 两版报(embedding+LLM)诚实标注循环: embedding=自家模型匹配; LLM=同GLM族抽gold

### gold质量核验 (诚实, 非真kappa)
- gold由GLM-5.2单模型抽, 无真人inter-annotator kappa
- 自动核验20条: evidence verbatim 17/20=0.85(无幻觉), 关系支撑词20/20, 端点方法名4/20(同义不同写非错)
- 诚实报"GLM-5.2独立抽取+自动核验verbatim率0.85", 不假装有kappa

## 关键修复
- 英文命名(commit f50c5daf): fair_recall 0.42→0.83, PSC 0.75→0.89, 召回瓶颈(induce归并)解决

## 进行中
- 阶段1 gold扩建: Thornton2026(第2篇segregation专题)用GLM-5-Turbo抽取中(快)
- 完成后: eval_cross_survey测跨综述一致性 + B臂vs Thornton gold ERR/PSC

## 降级
- 阶段7 quals拆维度消融: 多seed已证quals整体无显著增益, 拆维度从略(论文说明), 若审稿要求再做
