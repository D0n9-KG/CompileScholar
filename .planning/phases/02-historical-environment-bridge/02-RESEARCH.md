# Phase 2: Historical Environment Bridge - Research

**Researched:** 2026-04-02
**Confidence:** HIGH

## Summary

真实 jamming pilot 的 replay 结果说明，`L1` 的主要工程价值不是“背景知识补充”，而是给 `RouteState` 提供历史时点上真实可用的 benchmark、resource、toolchain、protocol 约束。只要这些字段长期为空，后面的 `WhyNow`、`Prior`、`DecisionEpisode` 就会持续偏薄。

本阶段已经验证了一个关键判断：

- 即使不改 `L2` schema，只要 `L1-lite` 更准，`L3/L4` 下游产物就会立刻变得更可用

已经得到的实证结果：

- `active_benchmarks` 从空变为 `maximally random jammed state` 与 `packing fraction phi c`
- `toolchains_and_infrastructure` 新增 `lubachevsky-stillinger compression protocol` 与 `radial distribution function`
- `DecisionEpisode.minimal_attack_path.required_resources` 从空列表变成了具体历史资源集合
- replay 剩余 flag 仍集中在 `support / alternative / held_out` route states 缺失，而不是 `L1` 字段缺失

## Standard Pattern

推荐模式：

1. 从真实 `PaperLogicTrace` 中提取保守的 `L1-lite`
2. 把 snapshot 序列化为稳定 JSON 资产
3. 让 `RouteStateSynthesizer` 直接消费该 snapshot
4. 用 replay delta 而不是主观感受评估 `L1` 是否有效

不推荐模式：

- 先引入大量外部知识源，再试图事后做 anti-hindsight 清洗
- 把 `L1` 做成自由文本摘要，无法被 route-state/replay 稳定消费
- 在没有 replay delta 的情况下，继续抽象扩展 `L1` 字段

## What Phase 3 Should Inherit

Phase 2 给 Phase 3 的最重要交付不是“一个更漂亮的 snapshot”，而是：

- 一个可复用的 `HistoricalEnvironmentSnapshot` contract
- 一条真实可运行的 `L1 -> RouteState -> Replay` 链路
- 一个明确的结论：接下来最该补的是 route-state packaging，而不是继续盲目扩 `L1`

---

*Phase: 02-historical-environment-bridge*
*Research completed: 2026-04-02*
