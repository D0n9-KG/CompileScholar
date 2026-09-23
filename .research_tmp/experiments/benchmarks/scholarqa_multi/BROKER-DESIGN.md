# SQA2 Broker 设计骨架（2026-09-24 深夜起草，裁点等用户）

> A6 判决的直接推论：单发检索 Recall@10=0.062、词汇鸿沟结构性存在
> （gold 靠特定发现连题，题面不含 gold 专属词汇）→ broker 必须把外部
> 检索放进 ReAct 循环迭代（搜→读→精炼→再搜），而非一次性预取。
> 我们的 harness 本来就是多步循环——检索作为**新工具**接入最自然。

## 一、形态：外部检索 = 答题栈的第 13/14 个 typed tool

```
13. search_papers(query, k?)      → SearchService.search(mode=auto)
    返回：论文列表（title/year/doi/oa链接 + rerank 分 + 延迟档）
    观测形态对齐现有工具（record_id 风格锚点 = doi/arxiv_id）
14. citation_graph(doi|title, direction) → CitationGraphService.graph
    返回：edges + in_corpus 标注（命中库内直接指回 paper_id）
```

要点：
- 工具观测是**自包含文献卡**（title+abstract 摘要+年份+DOI），可直接
  引用为证据（SQA2 开放语料的引用格式=文献列表，不是 [ctx_idx]）
- **KB 优先短路**：search_papers 先查库内（BM25+嵌入已有 text_index），
  命中即回 KB 记录（引用走已有通道）；库内空/薄才外扩——这与 G1 的
  "typed tools 主力、search_text 兜底"纪律同构，方向反过来：
  库内主力、外网兜底
- 每题外扩调用预算闸（如 ≤6 次），防延迟失控+成本失控

## 二、SQA2 提交形态

- Custom 类别（自带工具合规）；108+100 题
- 判分：官方 Citation F1（引用=返回文献列表）+ leaderboard 57 agents
  对照（open-weights 类别）
- ASTA_TOOL_KEY 若批 → Standard 类别加跑一轮（锦上添花）

## 三、裁点（需要用户定）

1. **外扩证据的引用格式**：SQA2 官方判分怎么认外部文献？(a) 引用
   DOI/URL 列表 (b) 引用文献 title——需要看 SQA2 官方评测器源码定
2. **每题外扩预算**：默认 ≤6 次搜索 + ≤2 次 citation_graph？
3. **延迟目标**：A5 实测 fast 1.7s/缓存 0ms/full 14.6s（GPU 满载）——
   答题循环里 full path 会拖慢单步，是否限制只用 fast 档+缓存？
4. **S2 API key**：A6 词汇鸿沟的主杠杆（语义检索）。用户能否申请一个
   （免费档就够 SQA2 规模）？
5. **语料边界披露**：SQA2 提交要不要声明"闭卷 430 篇 + 开放外扩"混合
   模式（对照 leaderboard 的纯开放 agents 公平性口径）

## 四、接线工作量估计

- 工具包装（两个 tool 函数+观测渲染）：~半天
- KB 短路逻辑：~2h
- 引用格式适配官方判分器：待裁点 1 定了才知道（半天~1 天）
- SQA2 题面格式适配+提交脚本：~1 天
- 全流程冒烟（Multi 20 题上跑通外扩臂）：~半天
