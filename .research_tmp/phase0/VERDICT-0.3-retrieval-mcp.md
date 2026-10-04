# 阶段 0.3 判决：检索/语料 MCP 服务器建成（2026-09-27）

## 交付物

`experiments/benchmarks/_shared/mcp/retrieval_mcp.py`（FastMCP stdio，
槽 3/4/5 harness 臂共用；我方 broker 不经此层——走内部工具）。

| 工具 | 说明 |
|---|---|
| `search_papers(query, k=8, before_year=None)` | 论文检索，返回 title/year/venue/verbatim 摘录/文档 id |
| `fetch_chunk(doc_id, chunk_id?, offset?, limit?)` | 全文读取（open=Sciverse /content；corpus=本地分块） |
| `list_corpus()` | corpus 模式=语料清单；open 模式=开放语料说明 |

双模式（env `RETRIEVAL_MCP_MODE`）：
- **open**（槽 3/4）：Sciverse 语义检索（agentic-search，词汇鸿沟主杠杆）
  + `/content` 全文；**before_year 服务器侧过滤**承载 CS2 2025-05-01
  截止纪律（防 harness 臂侧遗忘）
- **corpus**（槽 5 固定语料断网）：manifest+texts 本地词法索引
  （tf-idf，标题×3 加权，同篇多块去重取最高分块），零外部依赖零 GPU

## 验证（全部真实调用）

| 测试 | 结果 |
|---|---|
| corpus 检索（Multi 430 篇/25,780 块，"protein corona lipid nanoparticle"） | 前 5 全对口（corona 论文排第 1，score 26.1） |
| corpus fetch_chunk | 按块/按 offset 均正常 |
| Sciverse 语义检索（"sparse mixture of experts routing"） | 5 hits 全对口；before_year=2025 正确滤掉 2 条 2026 |
| Sciverse fetch（chunk_id 路径） | 整篇全文返回（64KB/41KB/53KB 三例） |
| **Claude Code×27B×MCP 集成（corpus）** | **6 轮循环全通**：检索→读全文→三段带引答案；四篇引用全部真实库内论文（与我方臂同源答案同论文集——公平装置直接可复现） |
| **Claude Code×27B×MCP 集成（open）** | **3 轮全通**：before_year=2024 纪律被模型主动传入参数、返回 5 篇真实论文全部 ≤2024、输出合法 JSON |

## 工程坑（记录在案）

1. **`sciverse_request_json` 只读环境变量**：`SciverseClient(token=...)`
   构造参数被鉴权层忽略——必须先设 `SCIVERSE_API_TOKEN` env 再调用
   （服务器已在 `_sciverse()` 内处理）
2. Sciverse offset 路径个别文档返回空（text="#"）——chunk_id 路径
   （search hit 自带）稳定返回全文，fetch_chunk 文档已引导优先用 chunk_id
3. 全文单块可达 64KB → fetch_chunk open 模式截断到 12k chars 并带
   `total_chars/truncated` 标记（防撑爆 harness 上下文）

## 接线配方（复用）

```
claude -p "<任务>" --model Qwen3.8-27B \
  --mcp-config <config.json> \
  --allowedTools "mcp__retrieval__search_papers,mcp__retrieval__fetch_chunk,mcp__retrieval__list_corpus"
```
config 见 `phase0/harness_smoke/mcp_config.json`（corpus）与
`mcp_config_open.json`（open）。

## 对设计档的回写

- §八-1"检索/语料 MCP"完成：槽 5 固定语料断网臂与槽 3/4 开网臂的
  基础设施均就位且经 Claude Code 实跑验证
- 槽 3 CS2 harness 臂的完整形态已可拼装：Claude Code + 27B + open 模式
  MCP（before_year=2025）——阶段 1 dev20 直接开跑
