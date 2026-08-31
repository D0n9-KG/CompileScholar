# 可交付 consumer 设计（比赛提交物，2026-08-28 起草）

## 定位

报告驱动策略下的"落地性 15% + 效率 20% + 结构化 10%"三位一体证据载体：
一个一键入口脚本，输入复杂学术查询，输出可溯源的结构化结果（论文列表+关联推理+
evidence），全程成本自动打点。

## 形态（三个入口，一个包）

```
contest/
  run.py              # 一键入口: python run.py "复杂查询" [--mode full|fast]
  pipeline.py         # L1召回(多源) → L2精排(权威+图结构) → L3归纳(结构化输出)
  cost_ledger.py      # 每层 API 调用/延迟/费用累计打点, 结束输出成本表
  output_schema.py    # 结构化输出: papers[] + relations[](带evidence) + answer_graph
```

## 管线（用今天实测校准过的事实）

- L1 召回：查询改写（DeepSeek-V4-Flash，~$0.0001/query）→ S2 bulk（主力，全 arXiv
  覆盖）+ crossref（权威被引补充）+ OpenAlex（解封后并入）。磁盘缓存全源。
- L2 精排：被引数预排序 → LLM 分档判分（H/S/N，带 citation_count 显示）→
  权威信号融合重排（H 档 log-cited 排序）。
- L3 归纳：对 top-k 论文（超图已覆盖者）走 evolution 边+方法族，输出
  "为什么相关"的结构化关联；图外论文按需增量建图（只做 top-50，单篇 ~$0.01）。

## 成本模型（效率 20% 证据）

| 层 | 调用 | 单查询成本（实测后填） |
|---|---|---|
| 查询改写 | 1×LLM | ~$0.0001 |
| 召回 | 5-8×检索API | $0（免费池）+延迟 |
| 精排判分 | 1×LLM（100候选） | ~$0.0005 |
| 增量建图（可选） | top-50×$0.01 | ~$0.5 上限 |

目标：fast 模式（不建图）<$0.001/query、<30s；full 模式按需。

## 报告对应关系

- 创新 15% → L2 的"结构化关联推理重排"vs 三系统纯相关分（对照实验在 LitSearch）
- 落地 15% → 本 consumer 一键可跑+全链自持（本地 sci-evo 服务）
- 泛化 10% → 查询域无关（4 域已验证的抽取栈）+SPARBench 30% bio 查询对冲说明
- 效率 20% → cost_ledger 自动产出成本表
- 结构化 10% → output_schema 的 relations+evidence 示例

## 状态

- [ ] pipeline.py 骨架（从 spar_runner.py 提炼，今天可完成大半）
- [ ] cost_ledger（薄封装 call_paratera + 检索调用计数）
- [ ] run.py CLI
- [ ] 结构化输出 schema（复用 hyperedge evidence 字段）
