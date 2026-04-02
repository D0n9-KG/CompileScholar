# Testing Patterns

**Analysis Date:** 2026-04-01

## Test Framework

**Runner:**
- `pytest` - backend unit / integration style tests
- `Vitest` - frontend unit tests
- `Pester` - PowerShell script tests

**Assertion Library:**
- backend: pytest 自带断言
- frontend: Vitest + Testing Library matcher

**Run Commands:**
```bash
cd backend; .\.venv\Scripts\python.exe -m pytest -q
cd frontend; npm run test
cd frontend; npm run lint
cd frontend; npm run build
Invoke-Pester tests
```

## Test File Organization

**Location:**
- backend tests 在 `backend/tests/`
- frontend tests 在 `frontend/tests/`
- PowerShell tests 在根目录 `tests/`

**Naming:**
- backend: `test_*.py`
- frontend: `*.test.ts` / `*.test.tsx`
- PowerShell: 跟随 Pester 约定

**Structure:**
```text
backend/
  tests/
    test_route_state_synthesizer.py
    test_decision_prior_builder.py
frontend/
  tests/
    *.test.tsx
tests/
  *.Tests.ps1
```

## Test Structure

**Suite Organization:**
- backend 倾向以模块 / builder 为粒度写 pytest suites
- research-logic tests 常先构造最小 typed fixture，再断言 schema 字段、quality flags 和 gating behavior

**Patterns:**
- 优先使用小型 fixture helper 构建 `PaperLogicTrace`、`RouteState`、`RoutePacket`
- 重点验证 typed output、quality tier、error path 和 invariants
- 对 PowerShell / frontend，优先覆盖启动脚本和关键 UI / state regression

## Mocking

**Framework:**
- frontend 使用 Vitest 的 mock 能力
- backend 目前更多依赖纯数据 fixture，而不是大规模 mocking

**What to Mock:**
- 外部 API / provider
- 网络或真实持久化依赖
- 时间 / 环境变量（必要时）

**What NOT to Mock:**
- schema-heavy pure builders
- typed transformation 逻辑

## Fixtures And Factories

**Test Data:**
- research-logic tests 已形成最小 factory helper 模式：构造 `MentionValue`, `ResearchMove`, `PaperLogicTrace`, `RoutePacket`
- 当前 fixture 更偏手工构造，以保证字段级断言精确

**Location:**
- 多数 helper 直接放在对应 test 文件顶部

## Coverage

**Requirements:**
- 仓库没有强制覆盖率阈值
- 实际要求是：修改 API behavior、graph transformation、loaders、state model、script behavior 时必须加定向测试

**Current Reality:**
- `backend/app/research_logic/` 当前已有较好的单元测试覆盖
- 真实语料 replay、跨层 pipeline、前端 E2E 仍是明显空白

## Test Types

**Unit Tests:**
- 目前最成熟的类型
- 特别适合 `paper_logic_trace`、`research_logic`、small utilities

**Integration Tests:**
- 存在但不系统
- 真实 Neo4j / ingest / replay 闭环测试不足

**E2E Tests:**
- 当前没有成熟的端到端浏览器自动化体系

## Common Patterns

**Async Testing:**
- backend 主要是同步 builder / compile tests
- frontend 用 `async` + Testing Library 等待交互结果

**Error Testing:**
- backend 常验证 invalid contract 时抛出 `ValueError` 或 model validation failure
- replay / compiler 还会验证 `ready_for_*` 与 quality flags 是否降级

**Snapshot Testing:**
- 当前不是主流模式

## Current Testing Gaps

**Replay On Real Corpus:**
- What's not tested: 用真实文献 packet 编译 `RouteState -> DecisionEpisode`
- Risk: 现在的 research-logic 原型可能只在 toy fixtures 上稳定
- Priority: High

**Cross-Layer Regression:**
- What's not tested: `L2` 改动对 `L3/L4` 编译成功率的影响
- Risk: 抽取优化可能提升单篇观感却伤害 replay
- Priority: High

**Frontend Replay Operations:**
- What's not tested: 未来 packet / replay 运维入口的 UI 行为
- Risk: 后续加功能时容易出现无人覆盖的回归
- Priority: Medium

---
*Testing analysis: 2026-04-01*
*Update when replay integration tests or UI workflows are introduced*
