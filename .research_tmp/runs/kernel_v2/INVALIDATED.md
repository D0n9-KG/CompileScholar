# ⚠️ kernel_v2 现存 bundle 全部作废 — 审计判决 2026-08-25

**本目录下所有 run（ML_DQN_2015 / ML_DQN_s1 / ML_DQN_s2 / ML_DQN_rolefix / FLUID_FCT_s1 / MOL_micromorphic_s1 / BIO_cockroach_s1 / test_paper_2020）的数字不可用于论文，仅供诊断。**

依据：`AUDIT_2026-08-25_glm53_full_review.md`（六路全面审查）+ `DECISION-2026-08-25-b-plus-rebuild.md`（commit 27d25ba5）。

## 三重污染

1. **单块 mock 输入**：运行脚本把全文 mock 成单个 block（`load_paper_blocks = lambda ...: [{"index":0,"text":txt}]`，见 tmp_scripts_2026-08-archive/_tmp_clean_run.py:11），`_locate_chapter_text` strict 正则在零换行文本上永不命中 → 5 个互相重叠的近全文切片 → **同一篇论文抽了 5 遍**（kept 387 = 5 遍并集，含 114 重复；实际 ~314）。真实 DQN MinerU 输入是 143 块（PPR_24493BE6E8C2）。
2. **修复前代码状态**：FLUID/MOL 跑在 evolution seed 修复（commit 6a6de676）之前；BIO 是修复前产物且 post-fix 重跑被 GBK print 崩掉未存 bundle——**BIO 在 HEAD = 未验证**。
3. **已知 bug 未修**：`_STEP3_PROMPT` 斜杠列表 → 复合 pattern_type "extends/improves/compares" → role gate 整边丢弃（81+ 条演化边被杀，≥62 条 roles 完美可救）。

## 仍然有效的部分（诊断价值）

- ledger 事务完整性、概念合并行为、verbatim 无捏造（93.5%）等**机制层**观察
- 逐条语义审的结论（语义正确率 ~43% 可辩护 / ~30-37% 明确错误）——这是 B+ 重写验收基线（43%→≥70%）的出处
- rich_topology 计数可从 concept_graph 重建

## 重挣数字的路径

DECISION 执行序：采纳探针 → 确定性修复（真实 MinerU 块+确定性切块+单 writer）→ 抽取栈重写（judge 交叉 ≥70% 验收）→ 全部数字在新管线上重跑。
