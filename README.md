# LogicKG

力学领域科技论文知识图谱系统：
- **单篇论文逻辑链条**（IMRaD 主线 + Claims）抽取并入库
- **跨论文引用网络**：不仅存“引用了谁”，还存“引用目的”（多标签 + 置信度）
- **证据可追溯**：回答与边属性都能回指到 MinerU `md` 的文本片段（chunk + 行号）
- **GraphRAG**：向量召回（FAISS + `BAAI/bge-m3`）+ 图信息（Neo4j）+ DeepSeek 生成

---

## 0. 目录结构（你需要关心的）

- `backend/`：FastAPI + 解析/抽取/写库/GraphRAG
- `frontend/`：React 工作台（Ingest / Papers / Paper Detail / Unresolved / Ask）
- `backend/runs/<run_id>/`：每次 ingest 的产物（解析 IR、Crossref、LLM 输出、FAISS 索引等）

---

## 1. 前置条件（Windows / PowerShell）

- Python 3.10+
- Node 18+
- Neo4j 5.x（推荐 Neo4j Desktop 本地启动）
- API Key：
  - DeepSeek：用于所有 LLM 生成/抽取
  - SiliconFlow：用于 embedding（推荐 `BAAI/bge-m3`）

说明：项目会读取 **项目根目录的 `.env`**（也支持 `backend/.env`）。

---

## 2. 配置 `.env`（放在项目根目录）

根目录 `.env` 至少需要：

```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...

# LLM (DeepSeek, OpenAI-compatible)
DEEPSEEK_API_KEY=...
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat

# Embeddings (SiliconFlow, OpenAI-compatible)
SILICONFLOW_API_KEY=...
# 可选：显式指定（不写也能自动推断）
EMBEDDING_PROVIDER=siliconflow
EMBEDDING_MODEL=BAAI/bge-m3
```

默认 embedding base_url 为 `https://api.siliconflow.cn/v1`；如你账号只支持 `.com`，可在 `.env` 里加：

```env
EMBEDDING_BASE_URL=https://api.siliconflow.com/v1
```

---

## 3. 启动 Neo4j（无 Docker 推荐）

如果你没有 Docker：参考 `docs/neo4j-setup.md`。

启动后，你应该能连上：
- Bolt：`bolt://localhost:7687`
- Browser：`http://localhost:7474`

---

## 4. 一键启动（推荐）

在项目根目录一条命令启动前后端（Neo4j 仍需你先单独启动）：

```powershell
.\run.ps1
```

等价命令（喜欢用 npm 的话）：

```powershell
npm run dev
```

启动后：
- 前端：`http://127.0.0.1:5173/`
- 后端：`http://127.0.0.1:8000/docs`

---

## 5. 后端（FastAPI）启动与使用

### 5.1 安装依赖 & 启动

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Swagger 文档：`http://127.0.0.1:8000/docs`

### 5.2 导入 MinerU Markdown（Ingest）

MinerU 通常是一篇论文一个文件夹，里面有：
- `xxx.md`
- `images/`

你可以对**项目根目录**执行导入（会递归寻找 MinerU 输出的 `*.md`，并自动跳过 `backend/`、`frontend/`、`docs/`、`node_modules/`、`.venv/`、根目录 `README.md` 等）：

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ingest/path -ContentType application/json -Body (@{ path = "C:\\Users\\jd\\Desktop\\LogicKG" } | ConvertTo-Json)
```

如果你传入的是**单篇论文文件夹**（里面就是 `xxx.md` + `images/`），也可以直接导入该文件夹。

> Windows PowerShell 有时会把中文路径编码成 `?` 导致后端找不到目录；建议用 UTF-8 bytes 方式提交 JSON：
>
> ```powershell
> $json = @{ path = "C:\\Users\\jd\\Desktop\\LogicKG" } | ConvertTo-Json -Compress
> $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
> Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ingest/path -ContentType 'application/json; charset=utf-8' -Body $bytes
> ```

导入会做这些事（按顺序）：
1) 解析 Markdown → `Chunk`（带行号范围）
2) 抽取参考文献与 in-text citation → Crossref 解析 DOI → 写入 Neo4j
3) DeepSeek 抽取：IMRaD + Claims；并为每条引用边生成“引用目的”多标签 → 写入 Neo4j
4) SiliconFlow(bge-m3) 计算 chunk embedding → 建 FAISS → 用于 GraphRAG

产物都在 `backend/runs/<run_id>/`：
- `*.document_ir.json`（结构/引用事件）
- `*.citations.json`（Crossref + cites）
- `*.llm_imrad.json`（IMRaD + Claims）
- `*.llm_citation_purposes.json`（引用目的）
- `faiss/`（索引）

### 5.3 GraphRAG 问答（Ask）

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/rag/ask -ContentType application/json -Body (@{ question = "What is the new dimensionless number introduced?"; k = 6 } | ConvertTo-Json)
```

返回包含：
- `answer`（DeepSeek 生成）
- `evidence[]`（FAISS 检索到的 chunk，含 md 路径与行号）

---

## 6. 前端（React 工作台）启动与使用

```powershell
cd frontend
npm install
Copy-Item .env.example .env.local
npm run dev
```

打开：`http://127.0.0.1:5173/`

页面说明：
- `Ingest`：点 Run 触发导入（用你填写的 path）
- `Papers`：论文列表（点击进详情）
- `Paper` 详情：
  - Logic Chain (IMRaD)：DeepSeek 摘要链
  - Claims：论文关键主张
  - Outgoing Cites：引用边 + 引用目的标签（可点击标签进行手工修正）
- `Unresolved`：未解析引用条目（Crossref candidates 展开 + 手填 DOI 修复）
- `Ask`：GraphRAG 问答 + Evidence 卡片

---

## 7. 常见问题（Troubleshooting）

### 7.1 `faiss_built=false`

- 检查 `.env` 是否配置：`SILICONFLOW_API_KEY`
- 检查 `EMBEDDING_BASE_URL` 是否和账号匹配（`.cn` / `.com`）

### 7.2 `neo4j_written=false`

- 确认 Neo4j 已启动并且 `NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD` 正确

### 7.3 Paper 详情页 404

`paper_id` 采用 `doi:...`（包含 `/`），后端已支持该路径参数；如仍出现旧数据干扰，建议清库后重新 ingest。

---

## 8. （可选）清空 Neo4j 重新开始

在 Neo4j Browser 里执行（慎用，会删光）：

```cypher
MATCH (n) DETACH DELETE n;
```
