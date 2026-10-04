# DECISION: step6 — LogicKG hypergraph storage in sci-evo-extract (core2 closed loop)

Date: 2026-08-14
Goal step: 阶段顺序 step 6 — 论文注册底座完整闭环

## 现状
- step1 paper_registry.get_hypergraph(paper_id) 返回 None（接口就位，sci-evo 侧无超图 artifact）。
- sci-evo artifact 存储强绑 mineru_artifacts + processing_jobs（需 mineru job_id）。
- LogicKG 抽的超图不是 mineru 产物，无法存进现有表。
- 核心2 闭环缺最后一环：DOI/title→论文→原文→超图→引用，超图这一段没落地。

## 决策（最小闭环）
sci-evo-extract 加**通用 external_artifacts 表**（不绑 mineru job），存 LogicKG（及未来其它非 mineru 产物）：
1. 新表 external_artifacts:
   artifact_id PK, paper_id FK, kind (如 'logickg_hypergraph'), source ('logickg'),
   local_path, checksum, status, manifest_json, created_at。
2. registry 加方法：store_external_artifact(paper_id, kind, path, source, status='ready')
   -> 用 sha256 + path_manifest，relative_local_path 校验，stable_id。
   list_external_artifacts(paper_id, kind=None)。
   paper_artifact_manifest / list_paper_artifacts 合并 external_artifacts（让 artifacts 端点能看到）。
3. API 端点：
   POST /api/library/papers/{paper_id}/external-artifacts (multipart: kind, source, file upload)
   -> 存到 library_root/external/{paper_id}/{kind}/ 下，register。
   （GET 已有 /papers/{id}/artifacts 含 external，client.get_hypergraph 走 KIND_LOGICKG_HYPERGRAPH。）
4. LogicKG 侧：corpus_driver.save_instance 时可选 push 到 sci-evo（store_external_artifact via API）。
   先不强制，保留本地 .research_tmp/instances 作主存，sci-evo 作可选镜像。

## 验证（纪律4）
smoke_step6.py:
1. 起 sci-evo 服务。
2. POST 一个 logickg_hypergraph artifact (小 JSON) 给 PPR_6AB55969EFBC。
3. paper_registry.get_hypergraph(PPR_6AB55969EFBC) 返回非 None（闭环通）。
4. /papers/{id}/artifacts 列出该 artifact。

## 不做
- 不改 mineru_artifacts（独立表，互不干扰）。
- 不强制 LogicKG 每次抽取都 push（保留本地优先，API 镜像可选）。
- 不做 extraction_runs 统一接口（step6 子步，先做 artifact 存储）。
