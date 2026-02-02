# LogicKG v1.1 设计稿：可续传上传导入 + DOI 冲突仲裁 + Figures 入库与可视化 + 人工编辑闭环

日期：2026-02-02  
范围：在现有 v1（本地 path ingest + Neo4j + GraphRAG + 基础工作台）基础上做可用性与“可编辑/可进化”增强。  

---

## 1. 目标与非目标

### 1.1 目标
- **上传导入（浏览器端）**：支持两种入口
  - **文件夹拖拽/选择**（Chrome/Edge 优先）：上传 `*.md` + `images/`（以及必要的目录结构）到后端。
  - **Zip 上传**：上传一个 `dataset.zip`（更通用）。
- **大文件可用**：支持单次 0.5–5GB 级别上传；提供**分片上传、断点续传、失败重试、进度**。
- **全库 DOI 唯一**：同 DOI 的论文在全库只保留一套 canonical 数据。
  - 导入中遇到同 DOI → **用户二选一：保留旧 / 用新替换旧（并删除旧的派生物与原始文件）**。
  - DOI 缺失 → **不入库**，进入 “Need DOI” 队列，待用户补齐。
- **Figures 入库与展示**：仅导入 Markdown 中被引用到的图片，提取图注并提供前端可视化。
- **人工编辑闭环**：提供可编辑入口（先做最小闭环），并记录审计信息，为后续“自进化”提供样本。

### 1.2 非目标（YAGNI）
- 不引入对象存储（S3/MinIO）或外部消息队列；以单机后端落盘为主。
- 不追求跨浏览器完全一致的“目录拖拽”体验（Firefox/Safari 允许退化到 zip）。
- 不在 v1.1 内做复杂权限系统；默认单用户本地使用。

---

## 2. 关键原则（确定性与可回滚）

1) **上传与入库解耦**：上传先进入 staging（临时区），扫描后再“分批入库”。  
2) **canonical 替换不破坏引用网络**：替换同 DOI 时保持 `(:Paper {paper_id:'doi:...'})` 节点稳定，删除/重建其子图（Chunks/LogicSteps/Claims/Figures/References），避免其他论文的 `CITES -> (Paper)` 断裂。  
3) **证据可追溯**：Figures/Chunks/人工编辑都要保留 `md_path + span(line)` 或可定位信息。  
4) **派生物可重建**：缩略图、向量索引、LLM 输出等放入 derived 区；允许在替换/重建时清空再生成。  

---

## 3. 存储布局（backend/storage）

> 仅示例，具体目录名可调整；核心是“uploads staging”和“papers canonical”分离。

- `backend/storage/uploads/<upload_id>/`
  - `manifest.json`：会话信息（文件清单、总大小、chunk_size、完成情况、hash）
  - `payload.zip.part/` + `payload.zip`：zip 分片与合并结果（zip 模式）
  - `files/<relpath>`：按相对路径落盘（folder 模式，可不生成 zip）
  - `extracted/`：zip 解包目录（只读 staging）
  - `scan.json`：扫描结果（ready/conflicts/need_doi/errors）
- `backend/storage/papers/doi/<doi_sanitized>/`
  - `source.md`
  - `images/`：仅保存 md 引用到的图片（原文件名）
  - `meta.json`：title/year/authors、来源 upload_id、导入时间等
- `backend/storage/derived/doi/<doi_sanitized>/`
  - `document_ir.json`、`llm_*.json`、`thumbs/`、（可选）`faiss/` 等

清理策略：
- `uploads/` 提供 `/ingest/upload/<id>/cancel` 删除；另提供后台定期清理（例如 24h 未完成的会话）。

---

## 4. 上传导入 API（分片/可续传）

### 4.1 上传会话

**POST `/ingest/upload/start`**  
请求：`{ mode: "zip"|"folder", filename?, total_bytes, chunk_bytes, files?: [{path, size}] }`  
返回：`{ upload_id, chunk_bytes, accepted: true }`

**POST `/ingest/upload/chunk`**（multipart）  
字段：`upload_id`, `index`, `sha256?`, `bytes`  
语义：后端按 `index` 写入 `payload.zip.part/<index>`（zip 模式）；folder 模式下可先把客户端打包成 zip 再走同一套，或把文件按相对路径落盘（实现上建议统一为 zip，减少后端复杂度）。

**GET `/ingest/upload/status?upload_id=...`**  
返回：已完成分片集合、总分片数、服务端已接收字节数、是否可 finish。

**POST `/ingest/upload/finish`**  
行为：
- 合并分片 → 校验大小/（可选）哈希。
- 解包到 `extracted/`：
  - 必须防 zip-slip（拒绝包含 `..`、绝对路径、盘符前缀等）。
  - 限制解包总大小与文件数，防 zip bomb。
- 触发扫描：生成 `scan.json` 并返回 summary（ready/conflicts/need_doi/errors 数量 + 示例）。

### 4.2 目录拖拽兼容性
- folder 选择：`<input type="file" webkitdirectory>` 读取 `webkitRelativePath` 作为 zip 内路径。
- folder 拖拽：优先 `DataTransferItem.webkitGetAsEntry()`（Chrome/Edge）；不支持时提示改用 zip。

---

## 5. 扫描与队列（Ready / Conflicts / Need DOI）

### 5.1 论文单元识别规则（用户选择的 B）
在 staging 根目录递归查找“论文文件夹”，满足：
- 目录内存在 **1 个主 `*.md`**
- 同级存在 `images/` 目录（允许为空）
若不满足（例如多个 md / 没有 images）→ 进入 `errors[]`，不进入后续队列。

### 5.2 DOI 缺失（用户选择 A）
- 若 md 中无法抽取 DOI：进入 `need_doi[]` 队列，**不入库**。
- 前端提供 “Need DOI” 页面：展示候选（文件夹名、标题猜测、md 路径），用户输入 DOI 后：
  - 校验格式（`^10\\.\\d{4,9}/\\S+$`）
  - 将该 DOI 写入 `uploads/<upload_id>/overrides.json`（不修改原 md），并再次扫描/推进队列。

### 5.3 DOI 冲突（全库只保留一套）
扫描到 DOI 后，查 Neo4j 是否已存在 `Paper.paper_id = doi:<doi>`：
- 不存在 → 进入 `ready[]`
- 已存在 → 进入 `conflicts[]`（包含 existing/new 的基本 meta 预览）

### 5.4 分批入库（用户选择 B）
导入流程：
1) 先对 `ready[]` 直接执行入库（写 Neo4j、写 canonical 文件、写 derived）。
2) `conflicts[]` 与 `need_doi[]` 留在队列中，等待用户操作后再分别推进。

---

## 6. DOI 冲突仲裁（保留旧 / 替换为新）

新增页面：`Conflicts`（按 upload_id 展示当前会话的冲突列表）

用户对每条冲突选择：
- **Keep existing**：丢弃 staging 中该论文单元（只删除该单元文件；保留 upload 会话其他内容）。
- **Replace with new**：执行“canonical 替换”：
  1) 文件层：删除 `backend/storage/papers/doi/<doi>/` 与 `backend/storage/derived/doi/<doi>/`，用 staging 重建。
  2) 图谱层：保留 `(:Paper {paper_id})` 节点，删除其子图并重建：
     - 删除关系：`HAS_CHUNK/HAS_LOGIC_STEP/HAS_CLAIM/HAS_REFERENCE/HAS_FIGURE`
     - 删除节点：`Chunk/LogicStep/Claim/ReferenceEntry/Figure`（仅限该论文域内）
     - 重新 upsert：paper 属性、chunks、references、logic_steps、claims、figures、以及本论文的 outgoing cites/unresolved
  3) 向量层：v1.1 可以先按“每次入库后重建全量索引”（小规模阶段可接受）；或在 v2 做增量。

注意：
- 替换时不删除其他论文对该 DOI 的 `CITES -> (Paper)` 入边。
- 若已有用户人工编辑（canonical 级），替换会覆盖；需在 UI 上明确提示。

---

## 7. Figures：只导入 md 引用到的图片

### 7.1 抽取与复制
在解析 md 时，提取图片引用（支持 `![](...)` / `![alt](...)`）：
- 仅接受相对路径，拒绝 `..`、绝对路径、盘符等。
- 只允许落在同目录的 `images/` 下（或显式白名单）。
- 若引用图片不存在：记录为 warning（不阻塞入库）。

文件落盘：
- 将被引用到的图片复制到 canonical `papers/doi/<doi>/images/<filename>`（保持文件名稳定）。
- 可选生成缩略图到 `derived/doi/<doi>/thumbs/<filename>`（PIL，限制最大边长 512）。

### 7.2 图注提取（规则优先）
对每个图片引用点，提取相邻图注：
- 优先识别 “Fig./Figure/图 + 编号 + :” 开头的行/段。
- 否则取图片行的下一段非空文本块（限制长度，例如 400 chars）。
保存 `caption_text` 与 `caption_span`（行号范围），便于回溯。

### 7.3 Neo4j Schema（新增）
- `(:Figure {figure_id, paper_id, filename, rel_path, caption_text, caption_start_line, caption_end_line})`
- 关系：
  - `(Paper)-[:HAS_FIGURE]->(Figure)`
  - （可选）`(Figure)-[:MENTIONED_IN]->(Chunk)`（当图片引用落在某个 chunk 内）

### 7.4 图片访问 API
提供受控静态文件服务：
- `GET /papers/{paper_id}/images/{filename}`
- `GET /papers/{paper_id}/thumbs/{filename}`
必须校验 `paper_id/filename`，确保路径在 canonical/derived 目录内。

---

## 8. 前端：Ingest / Conflicts / Need DOI / Figures

### 8.1 Ingest（上传工作台）
- 两个入口：拖拽/选择文件夹、选择 zip。
- 显示上传进度（已传/总量、速度、剩余时间估算）。
- finish 后展示扫描 summary：ready/conflicts/need_doi/errors。

### 8.2 Need DOI
- 列表展示待补 DOI 的论文单元，支持输入 DOI、校验、保存 override，并触发“重新扫描/推进”。

### 8.3 Conflicts
- 对每条冲突展示 existing vs new：
  - DOI、title/year、source_md_path（或 staging 路径）
  - 以及 “Replace 会覆盖人工编辑” 提示
- 两个动作：Keep existing / Replace with new。

### 8.4 Paper Detail：Figures 面板
在 `PaperDetailPage` 新增 `Figures` 区域：
- 缩略图列表/网格 + caption
- 点击打开 lightbox（放大、左右切换、下载）

---

## 9. 人工编辑闭环（v1.1 最小可用）

> 目标是形成可审计的人类修正数据，为后续迭代 prompt/抽取器提供训练样本。

建议先落地 4 类编辑（逐步上线）：
1) `LogicStep.summary` 编辑（保持 step_type 不变）
2) `Claim` 增删改
3) `SUPPORTED_BY` 证据绑定调整（LogicStep/Claim ↔ Chunk）
4) `Figure.caption` 编辑（手动补图注/修正）

审计字段（Neo4j 属性即可）：
- `edited_at`, `edited_by`（默认 `"local"`）, `edit_source="human"`
- 可选保留 `previous_value` 到旁路日志文件（避免 Neo4j 节点膨胀）

---

## 10. 错误处理与安全

- 上传限制：单会话总大小上限（例如 6GB），分片大小 8–32MB。
- 并发限制：同一时刻最多 N 个 active upload sessions（防止磁盘打满）。
- 解包安全：
  - zip-slip 防护（归一化路径 + 前缀检查）
  - zip bomb 防护（限制解包文件数与总字节数）
- 文件类型白名单：`*.md`, `*.png`, `*.jpg`, `*.jpeg`, `*.webp`（必要时再扩）
- 失败可恢复：chunk 写入采用原子落盘（先 `.tmp` 再 rename），status 基于已存在分片计算。

---

## 11. 测试建议（最小集合）

后端（pytest）：
- zip-slip 用例（`../`, 绝对路径，盘符路径）必须被拒绝
- 扫描规则 B：正确识别 “1 md + images” 的论文单元；多 md/缺 images 进入 errors
- DOI 缺失进入 need_doi，补 DOI 后进入 ready 或 conflicts
- replace 行为：Paper 节点保留，子图重建，入边 CITES 不丢

前端：
- 上传进度条与断点续传（模拟中断后继续）
- Conflicts/Need DOI 流程走通

