# Coding Conventions

**Analysis Date:** 2026-04-01

## Naming Patterns

**Files:**
- Python 模块使用 `snake_case`
- React 组件与页面使用 `PascalCase.tsx`
- frontend 其他模块常用 `camelCase.ts`
- 后端测试命名为 `test_*.py`
- 前端测试命名为 `*.test.ts` / `*.test.tsx`

**Functions:**
- Python / TypeScript 函数均以可读的 `snake_case` 或 `camelCase` 为主，跟随所在语言习惯
- builder / helper 命名通常直接表达输出对象，例如 `build_route_comparison_case`, `compile_historical_replay`
- task handlers 常使用 `handle_*` 前缀

**Variables:**
- Python 变量使用 `snake_case`
- TypeScript 变量使用 `camelCase`
- 常量使用 `UPPER_SNAKE_CASE`

**Types:**
- Pydantic models、TS types/interfaces 使用 `PascalCase`
- schema / contract 对象名称直接沿用 scientific layer 命名，如 `PaperLogicTrace`, `RouteState`, `DecisionEpisode`

## Code Style

**Formatting:**
- Python 使用 4 空格缩进，保持 PEP 8 友好
- TypeScript / TSX 使用 2 空格缩进
- frontend 使用单引号，不主动写分号
- 代码倾向于显式类型和显式 schema，而不是隐式字典约定

**Linting:**
- frontend 使用 ESLint，命令为 `cd frontend; npm run lint`
- backend 没有统一 formatter；改动时按现有风格保持整洁

## Import Organization

**Order:**
1. 外部依赖
2. 项目内模块
3. 相对路径导入

**Grouping:**
- 现有代码通常按语义分组，不追求过度机械排序
- research-logic 与 Pydantic 模型文件通常把 typed models 放在靠前位置

**Path Aliases:**
- frontend 主要使用相对路径，暂未形成复杂 alias 体系

## Error Handling

**Patterns:**
- 边界处用 schema 校验和显式 `ValueError` / API error 报错
- 后端核心 contract 倾向于“先 validate，再进入 builder / compiler”
- 质量不足时优先标记 `quality_tier` / `quality_flags`，而不是静默继续下游

**Error Types:**
- 输入 contract 错误：抛出 validation error / `ValueError`
- 下游 readiness 不满足：保留对象，但显式给出 `ready_for_* = false` 或 quality flags
- 运行期服务问题：由 API / task boundary 处理

## Logging

**Framework:**
- 当前以脚本日志、服务启动日志与 task / artifact 输出为主

**Patterns:**
- 更偏向“产物可检查”而不是重日志依赖
- 研究链路里质量字段和测试比运行期 log 更重要

## Comments

**When to Comment:**
- 注释优先解释“为什么这样做”或“这里的约束是什么”
- schema / builder 中如存在非显然规则，应该写简短注释
- 避免重复代码字面含义的注释

**TODO Comments:**
- 临时 TODO 不应替代正式 spec / `.planning/` 记录
- 中长期事项优先进入 `docs/superpowers/specs/` 或 `.planning/`

## Function Design

**Size:**
- 倾向用小型 pure helper + builder class 组合，而不是超长流程函数

**Parameters:**
- 参数较多时优先使用 typed model / keyword arguments，而不是长位置参数列表
- 对 schema-heavy 逻辑，显式传 `built_at`, ids, `reviewer_ids` 等上下文更清晰

**Return Values:**
- 返回 typed objects 优于 loosely shaped dict
- replay / research logic 倾向返回“对象 + 质量信息”，而不是只返回字符串摘要

## Module Design

**Exports:**
- backend 域模块中常用显式 named export
- `__init__.py` 用于整理领域模块公共 API

**Barrel Files:**
- `backend/app/research_logic/__init__.py` 之类的入口适合对外暴露稳定 API
- 避免为了 convenience 形成不清晰的跨模块循环依赖

---
*Convention analysis: 2026-04-01*
*Update when repo-wide style or schema patterns change*
