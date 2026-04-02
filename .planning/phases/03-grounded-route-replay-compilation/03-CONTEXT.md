# Phase 3: Grounded Route Replay Compilation - Context

**Gathered:** 2026-04-02
**Status:** Ready for verify

## Phase Boundary

Phase 3 的目标不是再证明“一个 primary route state 能被编译出来”，而是把 replay 运行时真正需要的多 route-state 输入变成标准化、可复用、可审计的工件。这个目标现在已经在真实 jamming slice 上跑通到 `green`。

当前最重要的已验证信号是：

- route-state package validation: `quality_tier = green`
- replay summary: `quality_flags = []`
- replay inspection:
  - `route_comparison.selected_quality_tier = green`
  - `decision_prior_card.quality_tier = green`
  - `decision_episode.quality_tier = green`
  - `decision_episode.ready_for_training = true`

这说明当前的主要缺口已经不再是 route-state packaging 本身，而是后续 Phase 4 里如何把 replay 失败继续压缩为明确的 `L2` 诊断与修补闭环。

## Immediate Objective

当前不再继续补 packaging contract，本阶段的直接下一步是：

1. 对 Phase 3 做 formal verify / closeout
2. 决定当前 jamming runtime subset packets 是否要升级为 committed canonical assets
3. 把下一阶段焦点切换到 replay failure taxonomy 和 `L2` surgical loop

## Locked Decisions

- 不改变 `L2` 边界
- 不把 `L3/L4` 退化成单篇论文摘要
- 先做可重复的 packaging contract，再做更复杂的 route clustering / induction
- 优先保证审计性和复现性，而不是一次性把所有 route family 都做完
- 后续 Phase 4 优先处理 `L2` comparator density、expected-slot completeness、relation stitching，而不是重新解决 support / alternative / held_out 缺件问题

## Expected Deliverables

- 已交付：
  - route-state package manifest / bundle contract
  - 批量编译 CLI
  - replay 对 package 的直接消费入口
  - package validation 与 replay inspection 输出
  - 一轮真实 jamming packaged replay 的全绿结果
- 待衔接：
  - Phase 3 verify artifacts
  - Phase 4 replay failure taxonomy context and plan

---

*Phase: 03-grounded-route-replay-compilation*
*Context gathered: 2026-04-02*
