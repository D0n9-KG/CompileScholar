# Pilot 结果与诚实结论 (2026-08-14)

## 流程可行性: 通过 ✓
- 251篇ref本地化、gold抽取(GLM-5.2,53方法/41演化边/41定律)、4臂抽取(共享meta跨论文演化)、匹配metrics全跑通
- 自演化发生: frozen恒6pattern, full/add_only/no_intra_dag都演化(pattern增长)
- 五操作生效: full split×12, no_intra_dag split×9+merge×2

## ★ 核心创新指标验证 (5篇同题对比)

### schema层 (pattern粒度) — 形式化操作集价值显现 ✓
| arm | uniq_pat | split | merge |
|-----|---------|-------|-------|
| full | 38 | 12 | 0 |
| add_only | 8 | 0 | 0 |
| frozen | 6 | 0 | 0 |
| no_intra_dag | 31 | 9 | 2 |
- full 38pat vs add_only 8pat: split贡献30个语义子类(influences→timing_shift等)
- add_only(仿AgentCAT只ADD)完全无法split → 形式化操作集(五操作)价值证实
- frozen恒6pat(不演化) → 演化本身价值证实

### node层 (NMR/ERR) — split价值不显现 ⚠️诚实发现
| arm | NMR | ERR_evo | ERR_comp | law_cov |
|-----|-----|---------|----------|---------|
| full | 0.868 | 0.268(11/41) | 0.500(2/4) | 32/41 |
| add_only | 0.849 | 0.268(11/41) | 0.250(1/4) | 29/41 |
- full与add_only的NMR/ERR_evo几乎相同(0.868 vs 0.849, ERR_evo都0.268)
- 原因: NMR/ERR测"节点/边在不在",split测"pattern分得细不细",不同层面
- split细化pattern不改变node覆盖 → 创新卖点须用schema层指标,非NMR/ERR

## ★★ NMR虚高问题 (诚实, 用户直觉正确)
- 原5篇NMR=0.868(含科普文2005贡献982泛化节点+locomotion综述)
- 排除科普文/综述,真颗粒流2篇NMR=0.642(低22点)
- 虚高: 泛化surface(cos松撞) + 阈值0.55偏松
- 实际匹配: 真实(RFT→RFT 1.0, M4 0.98)混杂噪声(M2→continuum model 0.79泛化撞)
- 修正: 加BM25+rerank / 提阈值 / 排噪声论文

## intra-DAG传播: 信号复杂不单调 ⚠️
- full acc=17 < no_intra_dag acc=30 (传播反而演化接受少?)
- 可能: 传播让schema更早稳定减少重复演化,或导致pattern趋同
- 需扩样本细析,不能简单说"传播更好"

## 诚实结论
1. pilot流程可行 ✓ — 自演化+五操作+dep/con/comp全跑通,内容真实
2. 形式化操作集(split)价值证实 — 但只在schema层(pattern粒度)显现,不在node层(NMR/ERR)
3. ★ 创新卖点定位修正: 主卖点须配schema层指标(uniq_pat/split数/收敛性),NMR/ERR只证"抽取覆盖"不证"操作集价值"
4. NMR虚高须修正匹配后才能做绝对值比较
5. gold后段(w17-21)缺失需GLM-5补抽
6. intra-DAG传播效果待扩样本细析

## 下一步
- pilot已过流程验证关 → 修正匹配(BM25+rerank) + 扩样本(全147篇) + GLM-5补gold后段
- 再引专家闸门核验gold + 复刻AgentCAT baseline + 写论文
- schema层指标需补"收敛稳定性"(跨论文pattern名趋同度,当前5篇跨论文pattern名差异大,收敛性可能不好)
