# Architecture

**Analysis Date:** 2026-04-01

## Pattern Overview

**Overall:** Full-stack research knowledge workspace with an emerging scientific-reasoning compiler inside the backend.

**Key Characteristics:**
- 一个可运行的产品底座已经存在：导入、抽取、图谱、Ask、运维配置、前端工作台都在同一仓库内。
- 后端以 FastAPI 为边界，但大量核心逻辑位于领域模块与 task handlers 中，而不是路由函数本身。
- `PaperLogicTrace -> research_logic -> replay` 是正在形成中的第二条主线，目前更多是库层原型，还没有完全接到产品闭环。

## Layers

**Ingest / Extraction Layer:**
- Purpose: 解析论文 / 教材输入，生成结构化抽取结果与图谱写入物。
- Contains: `backend/app/ingest/`, `backend/app/extraction/`, `backend/app/citations/`, `backend/app/paper_logic_trace/`
- Depends on: LLM provider、文本解析、Neo4j client、local storage
- Used by: task handlers、重建流程、后续 `research_logic` 编译

**Graph And Retrieval Layer:**
- Purpose: 管理 Neo4j 主图、community / similarity / FAISS 以及 Ask 检索消费链路。
- Contains: `backend/app/graph/`, `backend/app/community/`, `backend/app/similarity/`, `backend/app/vector/`, `backend/app/rag/`, `backend/app/fusion/`
- Depends on: Neo4j、FAISS、embedding provider、task system
- Used by: API routes、frontend workbench

**Research Logic Compiler Layer:**
- Purpose: 把 `L2 PaperLogicTrace` 进一步编译成 `RoutePacket / RouteState / WhyNow / Prior / Episode` 等更高层对象。
- Contains: `backend/app/research_logic/`
- Depends on: `paper_logic_trace` typed models、spec-driven schemas、future `L1` assets
- Used by: 当前主要被单元测试与离线实验调用，未来应进入 replay workflow

**API And Task Orchestration Layer:**
- Purpose: 提供 HTTP 边界、异步任务入口与运行期配置控制。
- Contains: `backend/app/api/routers/`, `backend/app/tasks/`, `backend/app/main.py`, `run.ps1`
- Depends on: 各领域模块
- Used by: frontend、本地脚本、手工 API 调用

**Frontend Workbench Layer:**
- Purpose: 提供论文/图谱浏览、Ask、导入中心、Ops 等界面。
- Contains: `frontend/src/pages/`, `frontend/src/components/`, `frontend/src/loaders/`, `frontend/src/state/`
- Depends on: backend API、React Router、graph visualization libs
- Used by: 研究者 / 开发者进行人工检查与系统操作

## Data Flow

**Paper / Textbook Ingestion:**

1. 用户提交本地路径、上传文件或任务请求。
2. `ingest` / `tasks` 模块把输入转成 ingestion job。
3. `extraction`、`citations`、`paper_logic_trace` 生成结构化结果与质量信号。
4. 图谱、artifact 文件、FAISS / community / similarity 相关结果写入存储层。
5. API 与 frontend 消费这些资产进行浏览、问答、重建和运维。

**Scientific Replay Compilation (Emerging Path):**

1. 从真实语料中选出一个 bounded `RoutePacket`。
2. 读取 packet 内的 `PaperLogicTrace`，并接入未来的 `L1` snapshot。
3. 通过 `research_logic` builder 生成 `RouteState / WhyNow / Comparison / Prior / Episode`。
4. 对 replay bundle 做质量分级、失败分析和人工审查。
5. 用失败样本反向驱动 `L1/L2/L3/L4` 修补。

**State Management:**
- 运行时主状态分散在 Neo4j、artifact 文件、FAISS 索引与 task queue 持久化中。
- 当前 `research_logic` 更偏“纯函数 + typed model + unit test”风格，尚未形成长期持久化存储规范。

## Key Abstractions

**PaperLogicTrace:**
- Purpose: 当前 `L2` 的 canonical 单篇论文逻辑资产。
- Examples: `PaperLogicTrace`, `ResearchMove`, derived views
- Pattern: Pydantic data contract + derived compiler-friendly views

**Research Logic Objects:**
- Purpose: 表达 `L3/L4` 编译链中的中高层对象。
- Examples: `RoutePacket`, `RouteState`, `WhyNowCase`, `RouteComparisonCase`, `DecisionPriorCard`, `DecisionEpisode`
- Pattern: Spec-first typed schemas + builder classes

**Task Pipeline:**
- Purpose: 把耗时操作放入异步任务系统，并驱动 rebuild / ingest / cleanup。
- Examples: `TaskManager`, task handlers, rebuild jobs
- Pattern: file-backed queue + explicit handler registry

## Entry Points

**Backend API:**
- Location: `backend/app/main.py`
- Triggers: FastAPI startup、HTTP requests
- Responsibilities: 注册 routers、应用 settings profile、启动 task manager

**Frontend App:**
- Location: `frontend/src/App.tsx`
- Triggers: 浏览器访问
- Responsibilities: 路由导航、页面装配、调用 loaders / API

**Local Dev Bootstrap:**
- Location: `run.ps1`, `run.lib.ps1`
- Triggers: `npm run dev`
- Responsibilities: 安装缺失依赖、选择端口、启动 backend + frontend

## Error Handling

**Strategy:** typed validation at boundaries, quality gates in extraction / compilation, and boundary-level HTTP/task failures.

**Patterns:**
- Pydantic models 用于 schema 校验与 builder 输入输出约束。
- `quality_tier` / `ready_for_*` / `quality_flags` 被广泛用作“可不可以进入下一层”的 gate。
- 复杂失败目前更多体现在 artifact / test / manual inspection 中，系统级 replay failure taxonomy 仍未成型。

## Cross-Cutting Concerns

**Logging:**
- 以脚本 / 服务日志和 task 产物为主，没有统一的 observability 平台。

**Validation:**
- 数据模型校验依赖 Pydantic。
- 单篇抽取与 research-logic 均有质量标记，但跨层评估还不完整。

**Configuration:**
- 以 `.env` / `backend/.env` / `frontend/.env.local` 为主，运行时 profile 由 `ops_config_store` 注入。

---
*Architecture analysis: 2026-04-01*
*Update when replay pipeline or storage boundaries materially change*
