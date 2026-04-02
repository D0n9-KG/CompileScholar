# Phase 6: Decision Episode Audit Export - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

## Phase Boundary

Phase 6 的目标不是宣称 `DecisionEpisode` 已经达到大规模训练数据成熟度，而是先把已经过审的 `L1/L2/L3/L4` 决策链路导出成可审计、可评估、可回放的样本资产。
这一阶段要解决的是 audited export policy、leakage-safe assembly 和 sample packaging，而不是跳过审计边界直接进入“训练就绪”叙事。

## Current Inputs

- Phase 5 已完成多路由 prior / anti-pattern candidate induction，并产出 review bundle。
- 当前真实试点里 accepted prior ids 仍为空，accepted anti-pattern ids 已被显式记录。
- replay bundle 仍保持 `green`，但 package-level prior review 维持保守，不会因为 held-out 通过就自动提升为 accepted prior。

## Planning Focus

- 明确 audited `DecisionEpisode` export 的 schema 和 bundle layout。
- 规定哪些 reviewed prior / anti-pattern outputs 可以进入 export，哪些只能保留在 review 侧。
- 继续保持 hindsight 仅用于 label / eval，不进入训练输入。
- 把 export readiness 表述为“audit-grade pilot”，而不是“training-ready dataset”。

## Canonical References

- `.planning/PROJECT.md`
- `.planning/REQUIREMENTS.md`
- `.planning/ROADMAP.md`
- `.planning/STATE.md`
- `.planning/phases/05-multi-route-prior-induction/05-VERIFICATION.md`
- `docs/replay/reports/phase5-prior-review-pilot.md`
- `backend/app/research_logic/decision_episode_builder.py`
- `backend/app/research_logic/historical_replay_compiler.py`
- `backend/app/research_logic/replay_io.py`

---

*Phase: 06-decision-episode-audit-export*
*Context gathered: 2026-04-02*
