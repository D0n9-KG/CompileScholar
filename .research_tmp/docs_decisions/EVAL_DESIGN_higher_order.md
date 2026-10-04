# 高阶升层评测设计 (EVAL_DESIGN)

独立评测"低阶升层长高阶"的价值, 不循环论证.

## 1. 独立性 (核心)
- gold 高阶 = 综述 ARFM2024 的人工/LLM 归纳 (methods + evolution_edges),
  独立于被测大图抽取 (综述作者读多篇后写的跨方法判断)
- 升层器从被引论文低阶 evidence 升高阶, 对齐 gold → 非循环
- LLM-as-judge 用 GLM-5 (与被测臂 deepseek 不同家族) 独判升出关系正确性

## 2. 已验证结论 (5 条 gold 边, 见 DECISION step10)
- extends/improves 可升: NGF improves μ(I) ✓ conf=high,
  I_gradient extends μ(I) ✓ conf=medium (2/2 对齐)
- compares 系统性升不出: 0/2 (I_grad-NGF + drainage spot-kinematic)
  根本局限: 被引论文不互提对方方法名, compares 需综述视角

## 3. metrics (诚实, 分类型)
- extends/improves 对齐率 = 升出且类型对齐 gold 的边 / gold extends+improves 边
  (主可升类型, 目标覆盖率)
- compares 覆盖低 (诚实标注, 不宣称全类型覆盖)
- 方法节点归纳准确率 = 升出方法对齐 gold methods 名 / gold 方法数

## 4. multi-arm 对照 (测自演化对升层贡献 — 真正独立价值评测)
同一组方法论文, 各 arm 抽低阶后升层, 比 gold 对齐率:
- full (自演化五操作) vs frozen (固定 seed schema) vs no_intra_dag
- 若 full 升层对齐率 > frozen → 自演化 schema 对升层有贡献 (独立价值)
- 这是替代 NMR/QA/silhouette 的真正价值评测 (那些都验证过不合适)

★ 待做: 当前新论文 (Bouzid/Kamrin/Bazant/Tüzün/Gray/Tripathi) 只抽了 full arm.
要做 multi-arm 对照, 需补抽 frozen/no_intra_dag arm (成本翻倍, 后续).

## 5. 诚实约束
- compares 根本局限 → 评测不苛求全 6 类型覆盖, extends/improves 为主
- gold 是综述视角, 升层是被引论文视角, 不可能 100% 对齐 gold
  (被引论文不写综述那种跨方法对比叙述)
- 方法学诚实: 升层产出"方向对且可解释"的关系即算成功, 不苛求类型完全对齐
  (如 I_gradient extends vs improves 边界主观, gold 标 extends LLM 标 improves
  两者都对, 算方向对)

## 6. 下一步
1. 扩 extends/improves 到 ~6-8 条 gold 边确认覆盖率稳定 (当前 2 条)
2. multi-arm 对照 (补抽 frozen/no_intra_dag) 测自演化对升层贡献
3. LLM-as-judge (GLM-5) 独判升出关系正确性, 非仅对齐 gold
