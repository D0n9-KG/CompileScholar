# 多 seed × SOFT_ROUTING 两态终判（预注册判读，2026-08-27）

数据：6 runs（seeds 1-3 × R0/R1）× 5 篇 × API 双 judge（GLM-5-Turbo + gpt-oss-120b，
qwen3.5 因延迟劣化 11-80s/调用被替换——判分组合变更，30 bundles 全量统一重判）。
预注册：PREREG_multiseed_soft_routing.md（结果落地前定死，未挪）。

## 判决表

| 门柱 | 判据 | 数据 | 判 |
|---|---|---|---|
| G1 主门柱 | 3 对中 ≥2 对 R1≥R0 且均值不降 | **3/3 对 R1≥R0**（+5.8/+2.0/+1.1pt），均值 38.9% vs 36.0% | **PASS** |
| G2 召回 | R1 边量 ≥ 0.95×R0 | 801 vs 849（-5.7%） | 边缘 FAIL |
| G3 novel 活性 | 升格 ≥1 且被复用 | novel 边 6 runs 全 0 | **FAIL** |
| G4 divergence | ≤8/篇 | 无 novel → 无 divergence | PASS（空洞） |

## 落入预注册分支："G1 过但 G2-G4 有挂"

处置（按预案）：
- SOFT_ROUTING 保留 env 开关，**默认改 0**（novel 通道无生产量，放行开关无对象）
- 改动②③（kernel 豁免+归纳通道）保留吃 dropped 边
- 软路由实验线**关闭**：novel=0 的根因是批量修的"优先 schema pattern"判据压没了
  LLM 自由命名——要恢复 novel 生产需放松该判据，但 FIX1 的 +8.2pt 正来自收紧，
  不回退

## 附产判决

1. **SOFTB 单 run -5.4pt 正式判为方差**：3 seed 反号（+5.8/+2.0/+1.1）。"单 run
   数字不可引用"纪律的实证回报——若当时按单 run 回滚改动①，就是被噪声骗了。
2. **FIX1 数字稳固**：both-pass 36.0-39.8%（seed 带宽 ~4pt），either 60.6-64.0%。
   对比 SOFTB 时代（28.4%/51.9%）：both +8pt 量级上移且 3 seed 复现。
3. **API 双 judge vs 内层 Claude judge 口径差**：内层 75.4% vs API both ~38%——
   39pt 差持续存在，仲裁（抽 20 条终裁）仍是待办。
4. 边数方差带：766-926（±13%）——再坐实。

## 效率注记

judge 提速 ~25 倍（40min+超时/全灭 → 1.5min/bundle）：qwen3.5 弃用换 gpt-oss-120b
（2-3s/调用，模型独立性合规）。
