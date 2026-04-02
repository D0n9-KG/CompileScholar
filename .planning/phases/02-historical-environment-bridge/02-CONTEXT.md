# Phase 2: Historical Environment Bridge - Context

**Gathered:** 2026-04-02
**Status:** Backfilled from executed work; ready to hand off into Phase 3

## Phase Boundary

Phase 2 的目标不是构建完整的外部科学史数据库，而是先为一个真实 replay pilot 补出最小可用的 `L1 Historical Environment` 资产，并把它接到现有 `RoutePacket -> RouteState -> WhyNow -> Prior -> Episode` 链路里。

当前已经锁定的约束：

- `L1` 现在只做 `paper_grounded_l1_lite`，允许 `missing / unknown / contested`
- 不能为了“看起来完整”而编造外部事实
- `L1` 的价值在于让 replay 拥有真实的 benchmark / resource / toolchain / protocol 约束，而不是替代 `L2`
- `L2` 仍然是单篇论文证据层，`L3/L4` 仍然是跨论文历史编译层

## Why This Phase Existed

Phase 1 的真实 replay 已经证明当前编译链条能跑通，但也暴露出关键缺口：

- `DecisionEpisode.minimal_attack_path.required_resources` 为空
- `RouteState.active_benchmarks` 与 `toolchains_and_infrastructure` 过薄
- `why_now_features` 缺少明确的历史环境解锁因子

因此，Phase 2 的工程目标是把这些空位从“逻辑占位符”提升为“有来源、可追踪、能被下游消费的历史环境信号”。

## Executed Decisions

- 保持 `L1` 轻量且保守，不引入超出论文证据范围的外部知识
- 先做真实 jamming pilot 的 `L1-lite`，而不是抽象设计一个大而全的环境层
- 先让 `RouteStateSynthesizer` 和 replay 真正消费 `L1`，再讨论更大规模的环境资产建设
- 当 `benchmark_candidates` 为空时，允许从 trace 的 protocol/resource/toolchain hints 做保守 fallback 恢复

## Canonical References

- `docs/replay/reports/phase1-pilot-failure-inventory.md`
- `docs/replay/reports/l1-lite-snapshot-pilot-2026-04-02.md`
- `docs/replay/reports/l1-benchmark-fallback-replay-2026-04-02.md`
- `backend/app/research_logic/historical_environment.py`
- `backend/app/research_logic/route_state_synthesizer.py`
- `backend/app/research_logic/historical_replay_compiler.py`
- `backend/scripts/run_l1_snapshot_pilot.py`
- `backend/scripts/run_route_state_pilot.py`
- `backend/scripts/run_replay_pilot.py`

## Exit Condition

Phase 2 可以视为完成，当且仅当：

- 至少一个真实 pilot 拥有可落盘、可复用的 `HistoricalEnvironmentSnapshot`
- `RouteState` 能消费该 snapshot，并在 benchmarks / infrastructure / why-now features 上体现出可解释增益
- replay 的剩余主要瓶颈不再是 `L1` 空字段，而是多 route-state 打包与比较结构

---

*Phase: 02-historical-environment-bridge*
*Backfilled: 2026-04-02*
