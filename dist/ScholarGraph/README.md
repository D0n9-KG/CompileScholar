# ScholarGraph 检索系统（华为赛题三：智能论文搜索）

复杂学术查询 → 查询理解与分解 → 多源多策略召回 → 超图 schema 意图扩展 →
（可选）引用意图探索 → 分块 LLM 判分 → 语义-权威混合排序 → 结构化输出。

## 目录

| 文件 | 职责 |
|---|---|
| `pipeline.py` | 召回-判分-排序唯一实现（单副本纪律：评测与演示都走这里） |
| `intent_recall.py` | 机制①③：冻结 schema 意图解析 + 槽位/代表性收敛查询扩展（盲生成，不碰答案） |
| `explore_recall.py` | 机制②：检索时轻量建图（cites-only，跳演化/对齐）+ 引用挖掘回池 |
| `cost_ledger.py` | 每次 LLM/检索调用打点（次数/时延/成本），全流程可审计 |
| `run.py` | 单查询一键入口 |
| `snowball.py` | 滚雪球基线（诊断用，已实测一跳增益低，不在主链） |

评测循环（SPARBench 50q）在 `.research_tmp/spar_runner.py`——`pipeline` 的薄封装，
只加基准读取与计分；`CONTEST_ORDER` 环境变量切换输出排序臂
（`hybrid`/`semantic`/`citation`），输出文件名带臂名防覆盖。

## 运行

```bash
# 单查询（快模式）
python -m contest.run "How can machine learning improve climate prediction?"

# SPARBench 50q 评测（默认 hybrid 排序；①③ 已内置）
python .research_tmp/spar_runner.py --limit 50 --offset 0

# 开启机制②（检索时增量建图+引用挖掘）
CONTEST_EXPLORE=on CONTEST_EXPLORE_K=2 python .research_tmp/spar_runner.py --limit 50

# 排序臂消融（串行跑防限流）
CONTEST_ORDER=semantic python .research_tmp/spar_runner.py --limit 50
CONTEST_ORDER=citation python .research_tmp/spar_runner.py --limit 50
```

依赖：本地 `sci-evo-extract` 服务（:8000，论文获取+MinerU 解析）、
Paratera LLM 网关（.env 配置）、可选 CST embedding（.env）。
演示前端：`demo/app.py`（:8899）+ `demo/index.html`。

## 确定性纪律（复现要求）

- 查询改写按查询哈希磁盘缓存（提供方 temp=0 仍漂移——缓存后逐次运行同分）
- S2 bulk / crossref 召回结果磁盘缓存（v3，带 abstract 字段）
- 对比实验必须同判分配置（`grade_window=150, grade_chunk=50, semantic_trim`）；
  不同配置的结果禁止互比

## 已验证数字（截至 2026-08-29）

| 项 | 数值 |
|---|---|
| 50q 基线（hybrid） | micro-F1 0.035（17/50 命中） |
| ①③ 召回池覆盖增益 | 开发集 +30.4pt、留出集 +16.7pt（判据预注册，留出复核） |
| 单查询成本（快模式） | ~$0.0013、13-36s |
| S2 约束 | 匿名 bulk 端点（AND 匹配、无排序、唯一 arXiv 覆盖源）；key 端点不可用 |

负结果与诊断链（覆盖缺口分解、排序臂查询类型依赖性、LLM 中间层稀释图信号）
见项目报告与 `.research_tmp/goal_log_0829.md`。
