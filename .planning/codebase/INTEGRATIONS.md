# External Integrations

**Analysis Date:** 2026-04-01

## APIs And External Services

**LLM Providers:**
- DeepSeek / OpenAI / OpenRouter-compatible APIs - 用于抽取、Ask、以及部分 research-logic 上游信号
  - SDK/Client: 通过 backend provider wrappers / LangChain integration
  - Auth: `LLM_API_KEY`, `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`
  - Notes: provider 可切换，属于核心运行依赖

**Embedding Providers:**
- SiliconFlow-compatible embedding API - 用于 similarity / vector retrieval
  - SDK/Client: embedding provider wrappers
  - Auth: `EMBEDDING_API_KEY`, `SILICONFLOW_API_KEY`
  - Notes: 与 FAISS 紧密耦合

## Data Storage

**Databases:**
- Neo4j 5.x - 主图数据库
  - Connection: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
  - Client: Python `neo4j` driver
  - Notes: 论文 / 教材 / claim / logic / community 等主资产都在这里

**Vector Storage:**
- FAISS - 相似度与检索索引
  - Connection: local filesystem artifacts
  - Client: `faiss-cpu`
  - Notes: 由 rebuild / similarity pipeline 维护

**Local File Storage:**
- 本地 artifact 目录 - 存放 ingestion、trace、run、evaluation 产物
  - Source: backend tasks / scripts / local corpus processing
  - Notes: `backend/storage/`, `backend/runs/` 等目录不应提交

## Optional Tooling

**Docker Compose:**
- Local Neo4j bootstrap
  - Entry: `docker-compose.yml`
  - Use: 本地无 Neo4j 时快速启动

**AutoYoutu / Related Graph Tools:**
- 教材链路与 community / chapter graph 相关的外部工具
  - Auth: local install / subprocess invocation
  - Notes: 并非所有链路都必需，但属于 repo 中的实际外部依赖点

## Environment Configuration

**Development:**
- Required env vars: Neo4j、LLM、embedding 至少各有一套可用配置
- Secrets location: `.env` / `backend/.env` / `frontend/.env.local`
- Mock / stub services: 当前没有系统化 mock service 方案，更多依赖本地真实服务

**Production / Long-Running Use:**
- 仍以本地研究环境或私有部署为主
- secret 管理依赖 `.env`，尚未形成集中式 secret management

## Webhooks And Callbacks

**Incoming:**
- None currently

**Outgoing:**
- None currently beyond API calls to LLM / embedding providers

## Integration Risks

**Provider Variability:**
- 不同 LLM / embedding provider 的行为差异会直接影响抽取和 Ask 质量

**Local Corpus Dependence:**
- route packet / replay 很可能依赖本地大语料目录，后续需要 manifest 化而不是把路径知识留在会话里

**Graph Storage Coupling:**
- Neo4j 是当前平台底座，但 research-logic replay 还没有完全定义长期存储方式

---
*Integration audit: 2026-04-01*
*Update when replay pipeline gains new external services or storage backends*
