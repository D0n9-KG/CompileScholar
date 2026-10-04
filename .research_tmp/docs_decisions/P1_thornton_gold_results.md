# P1 gold扩建: Thornton2026第2篇gold结果 (诚实)

Date: 2026-08-14
抽取: deepseek-chat (非reasoning, 避GLM reasoning超时), 16/29 window成功, merge得36方法27边4组成27定律

## Thornton gold 内容
- 36方法(带aliases): continuum/CA/DEM/micromechanics/kinetic sieving/percolation/squeeze expulsion/buoyancy/trajectory/elutriation/MD/advection-diffusion/Bridgwater flux/Dolgunin flux/mixture theory...
- 27演化边: compares 12/background 7/extends 4/improves 2/adapts 2
- evidence verbatim (deepseek抽取, 未单独audit但抽样质量好)

## 跨综述一致性 (ARFM vs Thornton)
- 方法名重叠仅4(通用: DEM/discrete element/kinetic sieving)
- 演化边重叠0
- 结论: 两综述方法体系基本不重合(ARFM偏continuum rheology, Thornton偏segregation modeling)
- 无法用"同方法跨综述一致性"测; Thornton是不同方法族的gold

## B臂4seed vs Thornton gold ERR (诚实负面)
- ERR_uncond=0.000, ERR_cond=0.000, coverable=0/27, fair_recall=0/0
- lift覆盖4个Thornton gold方法(M5/M19/M31/M34,通用DEM/kinetic sieving)
- 但coverable=0: 没有任何Thornton gold边两端都被覆盖
- 根因: 19篇输入论文(ARFM continuum/kinetic)与Thornton方法族(segregation flux)不重合

## ★诚实重要发现
**评测高度依赖输入论文集与gold方法族的匹配**:
- ARFM gold: 19篇输入(continuum/kinetic)→覆盖12/53方法→ERR/PSC可测
- Thornton gold: 同19篇输入→不覆盖Thornton方法族→ERR=0
- 跨综述泛化失败不是机制问题,是输入-gold方法族不匹配
- Thornton gold需抽取Thornton综述引用的segregation论文作输入(非ARFM19篇)

## 结论 (诚实)
1. 第2篇gold已建(Thornton, 36方法27边, deepseek抽取标注质量差异)
2. 但同输入论文集无法跨gold评测(方法族不重合)
3. 真跨综述一致性需: 各gold用各自综述的源论文做输入, 而非共享19篇
4. gold规模: 2篇(ARFM 53/41 + Thornton 36/27), 仍小于Intern-Atlas 30篇, 诚实标注

## 限制
- Thornton gold用deepseek非GLM-5.2, 抽取模型不同(诚实标注)
- 16/29 window成功(部分丢失, 后半段deepseek密集hang)
- 无真跨综述一致性数字(方法族不重合)
- 需补: 抽Thornton源论文(segregation)做输入才能真正评Thornton gold
