# P0 多seed计划 (英文版, 完成英文C/Q/B后执行)

## 当前状态
- 英文版T臂单seed: fair_recall 1.000, ERR_uncond 0.122, ERR_cond 0.556 (PSC待)
- 英文版C/Q/B: 运行中 (~30min, bnrt842p7)
- 单seed已证英文命名修召回瓶颈

## 多seed执行 (英文C/Q/B完成后)
1. 跑3 seed (seed1,2,3) 全4臂英文版:
   `python run_arms_shared_cluster.py --arms T,C,Q,B --suffix _en --seeds 3`
   (seed0已有, 跑1-3共3个新seed; 或重跑seed0-2取3个)
2. batch_err --suffix _en 批量ERR
3. 各臂PSC (eval_psc.py)
4. aggregate_multiseed --suffix _en --baseline T (均值±方差+paired bootstrap)

## 关注问题
- 各臂cluster共享(seed内), 跨seed cluster不同 → fair_recall应稳定在~1.0(英文命名修了归并)
- ERR_cond/PSC 跨seed方差: 若大说明judge非确定性主导, 需更多seed
- 配对显著性: C vs T (citation效果), Q vs T (quals效果), B vs T (full), 用paired bootstrap 95%CI
- 若CI跨0 = 该信号臂不显著(诚实报)

## 诚实底线
- 若某臂多seed后ERR_cond/PSC仍<0.3, 诚实承认机制在边级评测下不强
- 若各臂CI全跨0, 诚实报"结构信号臂无统计显著增益"(可能单seed的T7/C5/Q8差异是噪声)
