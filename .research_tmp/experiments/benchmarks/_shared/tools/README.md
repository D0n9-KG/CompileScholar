# 共享工具（benchmark 无关）

从 PaperScope 战役（2026-09 归档）提取的可复用仪器与基建。

## goldcov 仪表（gold 要点覆盖率测量）

用途：把 gold answer 拆成原子要点，量化"库里有没有/答案写没写"，
2×2 分解定位损失层（抽取 vs 检索/笔记 vs 草稿）。ScholarQA-Multi 与
QASA 的 gold 覆盖审计直接复用。

| 文件 | 作用 | 备注 |
|---|---|---|
| render_records.py | 记录层 → BM25+向量混合索引（复用 src/kb_compiler/views/search_text.py 的 TextSearchIndex 格式） | 嵌入 provider=cst-qwen3（4096 维） |
| decompose_points.py | gold answer → 原子要点（LLM，GLM-5.3 thinking-off） | 30 题≈1409 点（PaperScope 实测） |
| judge_kb.py | 要点 → KB 覆盖两级判定（t1 k=20 / t2 k=50） | 含 numpy 矩阵化补丁（0.28s/查询，纯 Python 版 30-60s） |
| judge_answer.py | 要点 → 答案覆盖判定 | 环境变量 GOLDCOV_ANSWERS/GOLDCOV_ANS_OUT 可配 |
| report.py | 2×2 汇总+分题型+动刀清单 | |

原版档案：archive/paperscope_2026-09/paperscope_r2/ps53/d2/goldcov/
（含 SPEC.md、GOLDCOV-VERDICT.md——判决方法论与仪器自审协议的原始记录）

## 答案循环 harness（修复栈，本目录已含副本）

- `evidence_gate2r_harness.py`：答题循环引擎（ReAct 循环/A1 多 action 步/
  A2 动态步帽/F28e/note_gate/card dict 行修复/pre-answer 穷尽性审计/
  输出帽 10000）。**含 GOLD-COV 轮全部修复**（grep 验证：card dict 行
  修复 + _RE_PREANSWER_AUDIT 均在）。
- `ps53r_run.py`：runner（F21b 投影 + F18 resolver + fetch_chunk +
  staging 目录参数化 PS53_ARM_DIR）。新基准沿用其 staging 约定，改造
  点=数据加载层（换 qfile/记录/视图）。
- `citation_correctness_eval.py`：ScholarQABench 官方判分脚本（确定性
  Citation F1 + AutoAIS），Multi/QASA 判分直接用或对齐改写。
- `PS53-SCALE-PREREG.md`：修复预注册的历史范例（新基准预注册照此规格）。

原件位置：见 ../../ARCHIVE-INDEX.md（PaperScope 战役归档）。
适配注意：harness 的 QUESTION/GOLD 路径、judged 输出路径均为 PaperScope
硬编码，新基准使用时需参数化（改动只做在 benchmarks/ 侧的薄壳里，
本副本保持修复终态只读）。

## 判分方法论模板

judge_official.py 的"官方 prompt 逐字对齐 + 判定器 JSON 自报解析 +
单引号兼容"模式，做新基准判分器时照此模式写（原文件随 PaperScope 归档）。
