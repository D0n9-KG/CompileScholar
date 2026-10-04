# DECISION: LogicKG paper_registry.py HTTP client (Side B1)

Date: 2026-08-14
Branch: research/era-reconstruction
Goal step: 阶段顺序 step 1 — LogicKG 接 sci-evo-extract 侧A API

## 背景
旧 goal 已完成"升层机制/自动分族/防幻觉/feedback闭环/GLM-5 judge/建模形式判断/引用关系底座侧A"。
新 goal 两核心：(1) 超图结构信号全利用升层；(2) 论文注册底座完整闭环。
侧A 已在 sci-evo-extract 加好引用关系表 + sources 提取 + API（commit 0da9d14）。
侧B = LogicKG 侧消费这些 API。本 step = 建 HTTP client 把 LogicKG 接到 sci-evo-extract。

## 决策（最小、可验证）
1. 新建 `src/granular_agent/paper_registry.py`，类 `PaperRegistryClient`。
   - 仅用 stdlib `urllib`（纪律2：不污染 LogicKG venv，不加 requests/httpx 依赖）。
   - base_url 走环境变量 `SCIEVO_API_BASE`（默认 http://127.0.0.1:8765/api）。
2. 暴露的接口（对齐消费方 lift_corpus 的 `edges_by_paper` 构建 pipeline）：
   - `health()` → GET /health
   - `resolve(doi=None, title=None, register=False, process=False)` → 先 search-resolve(dry-run)
     拿 candidates + 已注册 paper_id；若 register 且未注册 → POST /library/acquisitions（process=True 触发 MinerU，重，默认 False）。
   - `get_paper(paper_id)` → GET /library/papers/{id}
   - `get_references(paper_id, source="openalex", fetch=False)` → GET /library/papers/{id}/references
     fetch=True 触发 OpenAlex 拉取并 upsert（侧A 已验证）。这是 **step3 引用关系结构信号** 的数据源。
   - `get_citations(paper_id, source="openalex")` → GET /library/papers/{id}/citations
   - `get_fulltext(paper_id, kind="mineru_markdown")` → GET /library/papers/{id}/artifacts
     过滤 kind=mineru_markdown 且 downloadable，按 content_url 取正文文本。
   - `get_hypergraph(paper_id)` → 同 artifacts 接口找 kind='logickg_hypergraph'；
     **当前 sci-evo 侧无此 artifact（step6 才完善超图存储）**，故返回 None 不报错，接口先就位。
3. content_url 是 `/api/library/artifacts/{id}` 相对路径；client 从 base_url 拆 origin 拼 full URL。
4. 错误：PaperNotFound(404) / PaperRegistryError(其它非2xx)。

## 验证标准（纪律4 goal-driven）
- 起 sci-evo-extract 服务（uvicorn --port 8765，用其 .venv，纪律2）。
- `.research_tmp/smoke_paper_registry.py`：
  1. health() 返回 alive=True。
  2. get_paper('PPR_6AB55969EFBC') 拿到 paper + readiness。
  3. get_references('PPR_6AB55969EFBC', fetch=False) 拿到 ≥1 条引用（库内已有 30 条，不联网）。
  4. get_fulltext 对一个带 mineru_markdown 的论文拿到非空 markdown 文本（找库内已 processed 的论文）。
  5. resolve(title=...) dry-run 返回 candidates（不写库）。
  6. get_hypergraph 对任意论文返回 None（接口就位，不崩）。
- smoke 全绿 = step1 完成。

## 不做（防过度）
- 不在 client 里做超图抽取（那是 LogicKG extract_hypergraph 的事）。
- 不改 lift_corpus（step3 才动）。
- 不写 corpus_driver（step2 才建，把 client 接进 pipeline）。
- 不引入新依赖。
