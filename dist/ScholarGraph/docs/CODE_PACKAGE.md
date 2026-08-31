# 代码包清单（提交用，2026-08-30 打包）

提交物 = 项目名称 + 简介 + 项目报告 + 代码包（材料评审制）。
打包前逐项核对，敏感凭据（.env 的 API key）必须剔除，只留 .env.example。

## 包内结构

```
ScholarGraph/
├── README.md                  # src/contest/README.md 提到根（运行/复现/评测）
├── .env.example               # 配置项说明（无真实 key）
├── requirements.txt           # fastapi/uvicorn/vis 前端走 CDN 无需构建
├── src/
│   ├── contest/               # 检索系统（唯一召回实现+①③+②+成本打点）
│   │   ├── pipeline.py
│   │   ├── intent_recall.py
│   │   ├── explore_recall.py
│   │   ├── memory_recall.py   # 历史挖掘标题语义回池（检索沉淀为能力）
│   │   ├── google_recall.py / disambig_recall.py
│   │   ├── cost_ledger.py
│   │   ├── run.py
│   │   └── snowball.py
│   └── granular_agent/        # 超图抽取内核（建图/引用意图/图存储）
│       ├── agent.py, knowledge_base.py, concept_graph.py, ...
│       ├── citation_stage.py, citation_intent.py
│       ├── graph_store.py, query_expander.py
├── demo/
│   ├── app.py                 # FastAPI 演示后端（/api/search /api/graph*）
│   └── index.html             # 单页演示前端
├── eval/
│   ├── spar_runner.py         # SPARBench 评测循环（薄封装）
│   └── spar_bench.jsonl       # 评测查询与标准答案
└── docs/
    ├── REPORT.md              # 项目报告 v3（最终数字版）
    └── screenshots/           # 演示截图
```

## 打包前检查表

- [x] 50q 最终数字口径已填入 REPORT/简介/项目文档（08-31，终跑@10题止损口径）
- [x] 域图统计与 cites 分布写进报告
- [x] .env 剔除，.env.example 就位
- [ ] RSQ（官方族大 gold 基准）10 题数字写入（跑完 ~11:35 后）
- [ ] `python -m contest.run "测试查询"` 冷启动冒烟（RSQ 跑完后做，防 Paratera 争用污染）
- [ ] `python -m contest.run "测试查询"` 冷启动可跑（无本地缓存依赖声明）
- [ ] spar_runner 50q 复现命令写入 README
- [ ] 域图统计与 cites 分布写进报告 §4.2
- [ ] 演示视频（可选，若制作）

## 依赖说明（写进包内 README）

- Python 3.13；本地 sci-evo-extract 服务（:8000）负责论文获取+MinerU 解析
- LLM 网关：Paratera（.env 配置 base/key）；embedding：CST 或 Paratera GLM-Embedding
- 检索源：Semantic Scholar 匿名 bulk（唯一 arXiv 覆盖）+ Crossref（经 sci-evo 代理）
