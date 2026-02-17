# LogicKG

面向力学等科研论文的“结构化理解 + 可追溯证据”的知识图谱系统：将 MinerU 输出的论文 Markdown 解析为 **Paper / Chunk / ReferenceEntry / CITES** 等图结构，并结合向量检索（FAISS）与大模型生成，提供可交互的前端工作台（导入、浏览、图谱、问答、编辑与重建）。

本仓库已做过一次清理：不包含运行产物（`backend/runs/`、`backend/storage/`）、依赖目录（`node_modules/`、`backend/.venv/`）与个人计划文档；适合直接上传到 GitHub 供他人部署。

---

## 1. 你将获得什么

- **图谱入库**：Paper、段落 Chunk、参考文献、引用关系（CITES）等写入 Neo4j
- **引用网络**：不仅记录“引用了谁”，也支持“引用目的”（多标签 + 置信度）
- **证据可追溯**：图谱边与回答可回指到 MinerU `md` 的文本片段（chunk + 行号/跨度）
- **双轨抽取（Phase1）**：`Raw Pool` 保存全候选，`Validated KG` 仅写入通过质量门禁的结果
- **质量门禁 v2（Phase2）**：新增 `critical_slot_coverage` 与 `conflict_rate`，支持缺口检测与冲突约束
- **GraphRAG**：向量召回（FAISS）+ 图信息（Neo4j）+ LLM 生成
- **前端工作台**：导入、论文列表、论文详情、未解析引用、图谱视图、问答、任务等
- **Schema 全可配**：前端 `Schema` 页面支持规则/提示词细粒度配置 + `Rules JSON`/`Prompts JSON` 任意键扩展

---

## 2. 系统架构（概览）

```
MinerU 输出(md+images)
        |
        v
backend(FastAPI)  ----->  backend/storage/   (规范化存储：论文源文件、派生物等)
   |  |  |               backend/runs/      (每次 ingest 的运行产物：IR、crossref、Raw Pool、索引等)
   |  |  +----->  FAISS 向量索引
   |  +--------->  Neo4j (Paper/Chunk/CITES/...)
   +------------>  LLM/Embeddings (DeepSeek/SiliconFlow/OpenAI/OpenRouter)

frontend(React/Vite) <---- HTTP API ---- backend
```

---

## 3. 目录结构（重要）

> 下面带 “（生成/不提交）” 的目录会在运行时自动产生，默认在 `.gitignore` 中忽略。

```
.
├─ backend/                 # FastAPI 后端
│  ├─ app/                  # API、入库、RAG、任务等
│  ├─ scripts/              # 辅助脚本（可选）
│  ├─ runs/                 # （生成/不提交）每次 ingest 的产物
│  ├─ storage/              # （生成/不提交）规范化存储与派生物
│  └─ requirements.txt
├─ frontend/                # React 前端工作台（Vite）
├─ docs/
│  └─ neo4j-setup.md         # Neo4j 本地启动说明
├─ docker-compose.yml        # 仅用于启动 Neo4j（可选）
├─ .env.example              # 根目录示例配置（复制为 .env）
└─ run.ps1                   # Windows 一键启动（开发用）
```

---

## 4. 上传 GitHub 前：敏感信息检查（已替你检查 + 你也可复查）

结论（基于当前仓库内容）：
- 未发现私钥块（`BEGIN ... PRIVATE KEY`）、常见 token 前缀（`ghp_`、`github_pat_`、`sk-` 等）
- 仓库中出现的 `..._API_KEY` / `..._PASSWORD` 主要是 **示例占位符** 或配置字段名
- **根目录 `.env` 已在 `.gitignore` 中忽略**，但请确保你本地 `.env` 没有被强制 add

建议你推送前在仓库根目录执行：

```bash
git status -sb
git ls-files | rg '\.env'          # 确认未提交 .env
rg -n "ghp_|github_pat_|sk-|BEGIN .*PRIVATE KEY" -S .
```

如果你怀疑历史里曾经误提交过密钥：请立即在对应平台 **吊销并换新**（rotate），再考虑用 `git filter-repo` 清理历史。

---

## 5. 快速开始（本地开发）

### 5.1 前置依赖

- Python 3.10+（推荐 3.11）
- Node.js 18+（推荐 20）
- Neo4j 5.x（Neo4j Desktop 或 Docker 均可）
- 需要的 API Key（至少一个 LLM）：
  - DeepSeek：用于 LLM 生成/抽取（默认）
  - SiliconFlow：用于 embedding（可选但推荐，默认模型 `BAAI/bge-m3`）

### 5.2 配置 `.env`

在项目根目录复制示例配置：

```bash
cp .env.example .env
```

然后编辑 `.env`（不要提交），至少保证 Neo4j 与 LLM 可用：

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=请改成你自己的

LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=你的_key
```

#### 5.2.1 配置项说明（常用）

- `NEO4J_URI`：Neo4j Bolt 地址（默认 `bolt://localhost:7687`）
- `NEO4J_USER` / `NEO4J_USERNAME`：Neo4j 用户名（后端同时兼容两个变量名）
- `NEO4J_PASSWORD`：Neo4j 密码
- `LLM_PROVIDER`：`deepseek | openrouter | openai`（默认 `deepseek`）
- `LLM_MODEL`：LLM 模型名（例如 `deepseek-chat`）
- `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `OPENROUTER_API_KEY`：对应平台 key
- `LLM_API_KEY`：可选；统一 key（优先级高于 provider-specific key）
- `LLM_BASE_URL`：可选；自定义 OpenAI-compatible base url
- `EMBEDDING_PROVIDER`：`siliconflow | openai | openrouter | deepseek`（必需；用于 RAG/相似度/聚类）
- `EMBEDDING_MODEL`：向量模型（例如 `BAAI/bge-m3`）
- `SILICONFLOW_API_KEY`：SiliconFlow key（当 `EMBEDDING_PROVIDER=siliconflow`）
- `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL`：可选；统一 embedding key / base url
- `PHASE1_GATE_ALLOW_WEAK`：是否允许 `weak` 支持的 claim 进入 validated 轨道（默认 `false`）

> Phase2 门禁阈值（如 `phase2_gate_critical_slot_coverage_min`、`phase2_gate_conflict_rate_max`）位于 schema 的 `rules` 中，而非 `.env`。

> 说明：后端会优先读取 `backend/.env`，其次读取根目录 `.env`；两者任选其一即可。
>
> 抽取策略的细节参数（如 `phase1_*` / `phase2_*` 规则、各阶段 prompt）请在前端 `Schema` 页面配置并保存版本。

前端 `Schema` 页面可直接配置（含可视化表单 + `Rules JSON` / `Prompts JSON`）：
- 抽取流程阈值：`phase1_logic_chunks_max`、`phase1_logic_chunk_chars_max`、`phase1_doc_chars_max`、`phase1_claim_*`
- 质量门禁 v2：`phase2_gate_*`、`phase2_critical_*`、`phase2_conflict_*`（含正负极性词与 stop 词）
- grounding 评分：`phase1_grounding_*` 全部阈值与分数
- 引用目的抽取：`citation_purpose_max_*`、`citation_purpose_fallback_score`
- 提示词：`logic_claims_*`、`evidence_pick_*`、`phase1_logic_bind_*`、`phase1_chunk_claim_extract_*`、`citation_purpose_batch_*`

#### 5.2.2 内置三套抽取配置（重启/换机器仍可用）

项目已内置三套策略模板（写在后端代码中，不依赖本地存储文件），因此：
- 每次重启服务都可直接使用；
- 其他人 clone 到新目录后也天然具备同样模板。

三套模板：
- `high_precision`（高精度）：更严格证据约束与门禁，优先低幻觉、低冲突；
- `balanced`（均衡）：覆盖率与准确性折中，作为日常默认；
- `high_recall`（高召回）：扩大候选与抽取范围，便于探索性知识发现。

后端接口：
- `GET /schema/presets`：列出所有内置模板及中文说明；
- `POST /schema/presets/apply`：将指定模板应用到当前 schema 草稿（返回新 schema，不自动落库）。

前端用法：
- 在 `Schema` 页面顶部“内置抽取配置（策略模板）”面板中点击“应用到当前草稿”；
- 在“版本名称”输入框中填写你希望展示在“切换版本”下拉里的自定义名称；
- 再点击“保存为新版本并激活”；
- 对目标论文执行“重建”后，才会按该模板重新抽取。
- 若某个历史版本不再需要，可在“切换版本”处选中后点击“删除版本”（系统至少保留一个版本）。

> “切换版本”下拉会优先显示你填写的版本名称（并附带 `vN`），便于多人协作时识别配置用途。

### 5.3 启动 Neo4j（两种方式）

方式 A：Neo4j Desktop（推荐新手）
- 参考 `docs/neo4j-setup.md`

方式 B：Docker Compose（可选）

> 需要你已安装 Docker；本仓库的 `docker-compose.yml` 只负责启动 Neo4j。

```bash
docker compose up -d
```

### 5.4 启动后端与前端

Windows（PowerShell，推荐一键）：

```powershell
.\run.ps1
```

启动后：
- 前端：`http://127.0.0.1:5173/`
- 后端 Swagger：`http://127.0.0.1:8000/docs`

Linux / macOS（手动两终端）：

终端 1（后端）：

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

终端 2（前端）：

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev -- --host 127.0.0.1 --port 5173
```

---

## 6. 使用方式

### 6.1 导入 MinerU Markdown（Ingest）

MinerU 通常是一篇论文一个文件夹，里面有：
- `xxx.md`
- `images/`（或其他图片目录）

后端提供 `POST /ingest/path`：递归查找指定路径下的 `*.md` 并导入（会跳过 `backend/`、`frontend/`、`docs/`、`node_modules/`、`.venv/` 等常见目录）。

PowerShell 示例：

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/ingest/path `
  -ContentType application/json `
  -Body (@{ path = "D:\\papers\\mineru_output" } | ConvertTo-Json)
```

curl 示例：

```bash
curl -X POST 'http://127.0.0.1:8000/ingest/path' \
  -H 'Content-Type: application/json' \
  -d '{"path":"/data/mineru_output"}'
```

导入后：
- Neo4j 中会出现 Paper/Chunk/ReferenceEntry 等节点与关系
- `backend/runs/` / `backend/storage/`（本地）会产生运行产物与规范化存储（均不提交）

### 6.2 前端工作台

打开前端首页后，你通常会按以下路径使用：
1. Ingest：导入论文
2. Papers / Paper Detail：浏览与定位证据
3. Graph：查看引用网络/内容图
4. Ask：基于 GraphRAG 的问答
5. Unresolved：处理未能解析的引用

### 6.3 后端 API 快速入口

- Swagger（交互式文档）：`http://127.0.0.1:8000/docs`
- 健康检查：`GET /health`
- 导入：`POST /ingest/path`

> 其余路由可直接在 Swagger 中查看（Graph、Papers、Tasks、Schema 等）。

---

## 7. 部署到服务器（推荐做法：Neo4j Docker + 后端 systemd + 前端 Nginx）

> 下面以 Linux 服务器为例；目标是：外网只暴露 80/443（Nginx），Neo4j 端口不对公网开放。

### 7.0 部署前准备（建议）

- 目录规划：例如将仓库放到 `/opt/LogicKG`（或任意你习惯的位置）
- 运行用户：建议创建专用用户（例如 `logickg`），避免用 root 直接跑后端
- 数据持久化：
  - `backend/storage/`：论文规范化存储与派生物（重要，建议定期备份）
  - `backend/runs/`：每次 ingest 的运行产物（可按需清理，但用于复现与排查很有价值）
- 安全建议：
  - 不要把 Neo4j 的 `7474/7687` 直接暴露到公网
  - `.env` 只放服务器本地，权限建议 `600`

### 7.1 Neo4j（Docker）

1) 在服务器上放置 `.env`（不要提交）：

```env
NEO4J_USER=neo4j
NEO4J_PASSWORD=请改成强密码
```

2) 启动：

```bash
docker compose up -d
```

确保 Neo4j 只对内网或本机开放（可通过防火墙或反向代理策略控制）。

### 7.2 后端（systemd）

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

建议以 `127.0.0.1:8000` 方式运行（由 Nginx 反代）：

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 2 --proxy-headers
```

你可以创建一个 `systemd` 服务（示例，按需修改路径与用户）：

```ini
# /etc/systemd/system/logickg-backend.service
[Unit]
Description=LogicKG Backend
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/LogicKG/backend
EnvironmentFile=/opt/LogicKG/.env
ExecStart=/opt/LogicKG/backend/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

启用并启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now logickg-backend
sudo systemctl status logickg-backend
```

### 7.3 前端（构建 + Nginx 静态托管）

1) 构建：

```bash
cd frontend
echo 'VITE_API_URL=/api' > .env.production
npm ci
npm run build
```

2) 将 `frontend/dist/` 部署到 Nginx 的静态目录，例如 `/var/www/logickg/`。

3) Nginx 示例配置（同时反代后端 `/api`）：

```nginx
server {
  listen 80;
  server_name your.domain.com;

  root /var/www/logickg;
  index index.html;

  location / {
    try_files $uri $uri/ /index.html;
  }

  location /api/ {
    proxy_pass http://127.0.0.1:8000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
  }
}
```

启用 HTTPS：推荐使用 certbot / acme 自动签证书（略）。

---

## 8. 常见问题（Troubleshooting）

- Neo4j 连不上：确认 `NEO4J_URI`（bolt 7687）与账号密码；检查防火墙/端口占用
- Neo4j Console 启动报 `store_lock`：通常是数据库已被另一个 Neo4j 进程占用（重复启动导致）。先确认 `http://localhost:7474/browser/` 是否已可访问；若已运行不要再开第二个 `neo4j console`
- 端口冲突：Windows 开发建议直接用 `run.ps1`（会自动避开被系统保留/占用的端口）
- Embedding 不可用：必须配置 `EMBEDDING_PROVIDER` 和对应 API key，否则 RAG/相似度/聚类等核心功能会报错。可选择 SiliconFlow、OpenAI、OpenRouter 等平台
- 导入慢/失败：先在后端 Swagger（`/docs`）观察任务与报错；检查 MinerU 输出路径是否包含大量无关 md

---

## 9. License

如需开源发布，请补充许可证（LICENSE）与贡献指南（CONTRIBUTING）。
