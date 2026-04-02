# Phase 3: Grounded Route Replay Compilation - Research

**Researched:** 2026-04-02
**Confidence:** HIGH

## Summary

Phase 3 当前最值得先做的不是更复杂的聚类算法，而是把 replay 需要的多 route-state 输入做成明确 contract。因为真实 pilot 已经证明：

- primary route state 可以生成
- `L1-lite` 可以实质性改善 downstream route compilation
- 但 `L3/L4` 仍然因为缺少 support / alternative / held_out inputs 而保持结构性不完整

因此，最小可用方案应当是：

1. 定义 route-state package manifest
2. 允许每个 entry 指向一个 packet + traces + optional L1 snapshot
3. 输出按角色分组的 route-state bundle
4. 让 replay 可以一次性加载这些 grouped artifacts

## Why This Is The Right Next Step

如果没有 packaging 层，后续即使临时编出几个 route states，也会继续停留在：

- 机器本地手工拼命令
- 难以复现 support cluster 构成
- alternative / held-out 的来源与边界不清
- replay delta 无法稳定对比

先把 packaging contract 做出来，后面无论是扩 jamming 还是换其他 topic，都能沿同一条 artifact 边界推进。

---

*Phase: 03-grounded-route-replay-compilation*
*Research completed: 2026-04-02*
