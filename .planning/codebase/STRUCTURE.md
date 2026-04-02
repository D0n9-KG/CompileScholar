# Codebase Structure

**Analysis Date:** 2026-04-01

## Directory Layout

```text
LogicKG/
|- backend/                 # FastAPI backend, extraction, graph, research logic, tests
|- frontend/                # React + Vite workbench
|- docs/                    # Product, architecture, and scientific-reasoning specs
|- tests/                   # PowerShell / startup-script tests
|- .planning/               # GSD project memory, roadmap, codebase map, phase artifacts
|- docker-compose.yml       # Local Neo4j bootstrap
|- run.ps1                  # Full-stack dev bootstrap
|- README.md                # Project overview and usage
|- TECHNICAL_OVERVIEW.zh-CN.md
\- AGENTS.md                # Repo-specific collaboration rules
```

## Directory Purposes

**backend/**
- Purpose: 后端应用与大部分核心研究逻辑
- Contains: `app/` 域模块、`tests/` pytest、requirements、scripts
- Key files: `backend/app/main.py`, `backend/requirements.txt`
- Subdirectories: `app/api`, `app/extraction`, `app/paper_logic_trace`, `app/research_logic`, `app/tasks`, `app/rag` 等

**frontend/**
- Purpose: React 工作台
- Contains: `src/` 源码、`tests/` Vitest、package.json
- Key files: `frontend/src/App.tsx`, `frontend/package.json`
- Subdirectories: `src/components`, `src/pages`, `src/loaders`, `src/state`, `src/styles`

**docs/**
- Purpose: 设计文档、中文技术总览、research-logic specs
- Contains: 科学家思维项目文档、`docs/superpowers/specs/` 下的 schema / design docs
- Key files: `docs/科学家思维AI项目技术文档.md`, `docs/superpowers/specs/*.md`

**.planning/**
- Purpose: GSD brownfield 项目记忆、roadmap、phase planning artifacts
- Contains: `PROJECT.md`, `REQUIREMENTS.md`, `ROADMAP.md`, `STATE.md`, `codebase/`, `phases/`
- Key files: `.planning/config.json`, `.planning/phases/01-replay-pilot-packetization/01-CONTEXT.md`

## Key File Locations

**Entry Points:**
- `backend/app/main.py`: FastAPI backend 入口
- `frontend/src/App.tsx`: 前端应用入口
- `run.ps1`: 本地联启脚本

**Configuration:**
- `.env.example`: 根目录环境变量基线
- `backend/requirements.txt`: 后端依赖
- `frontend/package.json`: 前端依赖与脚本
- `docker-compose.yml`: Neo4j 本地运行配置

**Core Logic:**
- `backend/app/paper_logic_trace/`: `L2` canonical export 与 derived views
- `backend/app/research_logic/`: `L3/L4` builder 原型与 replay compiler
- `backend/app/extraction/`: 论文抽取主链
- `backend/app/rag/`: Ask 检索 / 回答

**Testing:**
- `backend/tests/`: pytest
- `frontend/tests/`: Vitest
- `tests/`: Pester / PowerShell tests

**Documentation:**
- `README.md`: 用户与开发者概览
- `TECHNICAL_OVERVIEW.zh-CN.md`: 当前技术架构总览
- `docs/superpowers/specs/`: 面向 research-logic 的细化规格

## Naming Conventions

**Files:**
- Python 模块用 `snake_case.py`
- React 页面与组件多用 `PascalCase.tsx`
- frontend 其他模块与 state helpers 多用 `camelCase.ts`
- backend tests 用 `test_*.py`
- frontend tests 用 `*.test.ts` / `*.test.tsx`

**Directories:**
- 领域模块目录多为简短 noun / plural，例如 `api/`, `rag/`, `tasks/`, `similarity/`
- phase 目录采用 `NN-phase-slug` 形式，例如 `01-replay-pilot-packetization/`

**Special Patterns:**
- specs 集中放在 `docs/superpowers/specs/`
- `.planning/` 是 GSD 的唯一项目记忆入口，后续 phase / plan 总结都应落在这里

## Where to Add New Code

**New backend domain logic:**
- Primary code: `backend/app/<domain>/`
- Tests: `backend/tests/`
- API exposure if needed: `backend/app/api/routers/`

**New replay / research-logic code:**
- Implementation: `backend/app/research_logic/`
- Upstream data contract changes: `backend/app/paper_logic_trace/`
- Specs / design docs: `docs/superpowers/specs/`
- GSD planning context: `.planning/phases/<phase>/`

**New frontend capability:**
- Routes / pages: `frontend/src/pages/`
- Reusable UI: `frontend/src/components/`
- Data loaders: `frontend/src/loaders/`
- Tests: `frontend/tests/`

## Special Directories

**backend/storage/**:
- Purpose: 运行时产物 / 本地数据
- Source: ingestion / rebuild / task outputs
- Committed: No

**backend/runs/**:
- Purpose: 运行时 / 评测产物
- Source: local execution artifacts
- Committed: No

**.codex_tmp/** and **tmp/**:
- Purpose: 临时调试或代理工作目录
- Source: local tooling
- Committed: No

---
*Structure analysis: 2026-04-01*
*Update when directory structure or phase layout changes*
