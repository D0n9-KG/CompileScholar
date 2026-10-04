# 集成 sci-evo-extract 引用关系（HTTP API，不合并）

## 状态
- [x] 侧 A1: registry 引用表 (paper_references/paper_citations) — commit 0da9d14
- [x] 侧 A2: sources 提取 (extract_openalex/crossref_references + fetch_work_references) — commit 0da9d14
- [x] 侧 A3: API /papers/{id}/references?fetch=true + /citations — commit 0da9d14
- [x] 侧 A 端到端验证: Kamrin_2012 → 注册 → fetch references 30条(doi/title/year) ✓
- [ ] 侧 B1: LogicKG paper_registry.py HTTP client
- [ ] 侧 B2: corpus_driver 接 paper_registry (双轨)
- [ ] 侧 B3: lift_corpus 用引用关系作结构信号升层

## 侧 A 验证记录
- 服务: uvicorn sci_evo_extract.api.app:app --port 8765 (sci-evo .venv)
- POST /api/library/acquisitions {"doi":...} → paper_id (metadata_only)
- GET /api/library/papers/{pid}/references?fetch=true&source=openalex → 30条含doi/title/year, 存registry
- registry: register/list_references, register/list_citations 全工作

## 下一步: 侧 B1
LogicKG: src/granular_agent/paper_registry.py — HTTP client 调 sci-evo-extract:
  resolve_paper(doi|title) → paper_id
  get_references(paper_id, fetch=True) → 引用列表
  get_fulltext(paper_id) → MinerU markdown (替代 pilot_refs)
