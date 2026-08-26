# 抽取失败案例库（回归考卷）

建于 2026-08-26（新打法：诊断驱动+小步快跑，治"微调→1.5h 全量→再微调"慢循环）。

## 是什么

从 SOFTB 5 篇（DQN/ml + 4 篇 granular：实验/短文经典/建模/理论）的内层 judge
（Claude 子代理，2026-08-26）75 条失败 + 24 条 pass 边构建的句子级回归考卷。
每条案例 = 原文 chunk（±1400 字符）+ watch 句 + 期望行为。runner 重跑真实
plan→execute→gate→verify→fix 链（冻结 schema `schema_frozen.json`=5 个 SOFTB
snapshot 的并集，27 patterns），10 分钟级验证一轮修复。

## 文件

- `cases_main.jsonl` — 77 条（dev）：5 类 scored 25 + misc 监控池 40 + pass 回归 16（soft）
- `cases_holdout.jsonl` — 19 条 scored（class×domain 分层 30%，**修复时禁止看**）
- `schema_frozen.json` — 冻结考卷 schema（union，27 patterns）
- `build_case_library.py` — 构建脚本（含 75 条失败的手工分类映射）
- `runs/` — 每次 runner 输出

## 5 个修复类（scored，32 条 + holdout 8 条）

| class | n_main | 修什么 | bad 判据 |
|---|---|---|---|
| negation | 8 | 否定/独立性句子（"would not influence"/"not affected by"/"no impact on"）被抽成正向 influences | watch 句上出现 influences 即错 |
| routing | 2 | schema 里已有正确 pattern（agrees_with/equivalent_formulation/adapts/validates_against）但抽成 compares/extends/background | 错 pattern 出现即错，preferred pattern 出现=FIXED_CORRECT |
| polarity | 5 | 因果/方向反转（"A depends on B"抽成 A→B）| 特征角色绑定重现即错 |
| lawinput | 1 | 定律/方程本身绑进 constitutive_law 的 input/parameter 槽 | law 名出现在 input/parameter 槽即错 |
| setup | 4 | 装置/实现/画图描述当科学主张抽取 | watch 句出现该 pattern 即错 |
| pass_regression | 16 | 过修护栏：judge 判 pass 的边修复后不能消失 | 同 pattern 边仍在（soft：单次缺失可能是方差） |

misc 40 条只监控不计分（bad spec 从 judge 绑定自动派生，粗）。

## 用法

```
python src/run_cases.py --cases .research_tmp/case_library/cases_main.jsonl \
    --out .research_tmp/case_library/runs/<name>.jsonl
```

结果三态：STILL_WRONG（坏绑定重现，硬失败）/ FIXED_CORRECT（好绑定或替代
pattern 出现）/ FIXED_DROP（watch 句无该边，可接受修复）。pass 案例两态：
RETAINED / REGRESSED。

## 纪律

- 修复在 main 上验证；holdout 只在修复完成后跑一次
- judge 分层：本库判分是确定性的（不调 judge），里程碑才上 API 双 judge
- 防过拟合：分域（ml/granular）已内置；修复优先 prompt 层，规则层标注过拟合风险

## 已知盲区（2026-08-27 标注）

1. **compact 检索路径未覆盖**：冻结 schema 27 patterns ≤ RETRIEVAL_K=30 → 考卷走
   全量 `to_prompt()` 路径。真实 A2 长跑 schema>30 后进入 `_render_patterns_compact`
   检索路径——"pattern 建 12 只用 4"的接线断点在那条路上，考卷绿≠该路径修好。
   该路径的判据只能靠 A2 级长跑（多 seed 轮）。
2. **基线判读**（修前×1）：75 条失败中只有 negation 类系统性复现（5/8
   STILL_WRONG）；setup/polarity/lawinput 基线重跑即被 verifier 杀（SOFTB 那次
   是方差性漏放）。案例库的作用之一就是把系统性和随机性失败分开。
3. pass 案例绝对 RETAINED 率低（6/16）是无上下文 chunk+冻结 schema 的固有噪声，
   有意义信号是修前后 REGRESSED **差值**与逐案对比。
4. **watcher 邻句误配**：`_watches` 用 token 重叠（≥50%）判边是否在 watch 句上，
   邻接句谈同一批量词时会误配（holdout C012 即此：边全来自邻句，watch 句本身
   已无边）。复核 STILL_WRONG 时先看 watched 边的 evidence 是不是 watch 句本身。
