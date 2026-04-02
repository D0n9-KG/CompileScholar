# Technology Stack

**Analysis Date:** 2026-04-01

## Languages

**Primary:**
- Python 3.11 - 后端应用、抽取流程、research-logic 编译、测试
- TypeScript 5.9 - 前端应用、状态管理、loaders、组件测试

**Secondary:**
- PowerShell - 本地启动脚本、部分测试与运维入口
- Markdown - 设计文档、spec、`.planning/` 项目记忆

## Runtime

**Environment:**
- Python virtualenv (`backend/.venv`) - 后端运行与 pytest
- Node.js 18+ - frontend、Vite、Vitest、root dev bootstrap
- Browser runtime - React workbench
- Neo4j 5.x - 图谱主存储

**Package Manager:**
- npm - root bootstrap 与 frontend 依赖管理
- pip / requirements.txt - backend 依赖
- Lockfile: frontend 依赖由 `package-lock.json` 管理；backend 使用固定版本 requirements

## Frameworks

**Core:**
- FastAPI 0.115.6 - 后端 HTTP API
- Pydantic 2.10.4 - schema / config / typed contracts
- React 19.2 + React Router 7 - 前端工作台
- Vite 7.2 - 前端开发与构建

**Testing:**
- pytest - 后端测试
- Vitest - 前端单元测试
- ESLint - TypeScript lint
- Pester - PowerShell 脚本测试

**Build / Dev:**
- uvicorn - FastAPI 开发服务器
- TypeScript compiler - frontend build
- Docker Compose - 本地 Neo4j

## Key Dependencies

**Critical:**
- `neo4j` - 后端图数据库连接
- `langchain`, `langchain-community`, `langchain-openai` - LLM orchestration pieces
- `faiss-cpu` - 向量索引
- `sentence-transformers` - embedding / semantic similarity
- `react-router-dom` - frontend 路由

**Infrastructure:**
- `3d-force-graph`, `cytoscape`, `three` - 图谱可视化
- `scikit-learn`, `scipy`, `numpy` - similarity / clustering / numeric utilities
- `python-dotenv` / `pydantic-settings` - env config

## Configuration

**Environment:**
- 根目录 `.env.example` 作为基线
- 后端优先读取 `backend/.env`，找不到时回退到根目录 `.env`
- 前端通过 `frontend/.env.local` 注入实际 backend URL
- 关键变量覆盖 Neo4j、LLM provider、embedding provider、task / extraction profile

**Build:**
- `run.ps1`, `run.lib.ps1` - 本地联启脚本
- `frontend/tsconfig*.json`, `vite.config.ts` - 前端构建配置
- `docker-compose.yml` - Neo4j 本地运行配置

## Platform Requirements

**Development:**
- Windows PowerShell 是当前最顺手的开发路径，但核心应用本身并不强绑定 Windows
- 本地最好可用 Docker / Neo4j / LLM API key / embedding API key
- 教材特定链路可能需要额外工具如 `autoyoutu`

**Production / Long-Running Use:**
- 至少需要稳定的 Neo4j、LLM provider、embedding provider
- 当前更像研究型本地工作台，而不是完全云原生部署方案

---
*Stack analysis: 2026-04-01*
*Update after major dependency or runtime changes*
