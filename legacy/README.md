# legacy/ — 旧管线封存（2026-08-25）

依据：`.research_tmp/DECISION-2026-08-25-b-plus-rebuild.md`（commit 27d25ba5）+ 六路全面审查判决。

## 这里是什么

B+ 重建前的**旧管线代码**，物理封存于此。单管线原则：**主代码（src/）不 import legacy/**，此目录下的代码**不可直接运行**（依赖已被手术移除，如 agent.py 的 legacy 入口）。

| 文件 | 原位置 | 说明 |
|---|---|---|
| `granular_agent/extractor.py` | src/granular_agent/ | 旧 atom 管线抽取器（截断式） |
| `granular_agent/chained_extractor.py` | src/granular_agent/ | 旧三阶段链式抽取 |
| `granular_agent/schema_manager.py` | src/granular_agent/ | 旧 schema 演化（绕过 KB） |
| `granular_agent/gap_discovery.py` | src/granular_agent/ | 旧 gap 扫描 |
| `granular_agent/qa_generator.py` | src/granular_agent/ | 旧 QA 生成（consumer 已砍） |
| `granular_agent/corpus_driver.py` | src/granular_agent/ | 死代码（逻辑已 mirror 进 process_paper_hypergraph） |
| `granular_agent/hg_qa_generator.py` | src/granular_agent/ | 死代码（零引用） |
| `granular_agent/hg_retrieval.py` | src/granular_agent/ | 死代码（零引用） |
| `run_test.py` / `run_stage2.py` / `run_stage2_fast.py` / `run_cross_domain.py` | src/ | legacy runner |

已直接删除（不留 legacy）：`capabilities/`、`hooks/` 两个 0 字节空壳目录（Pydantic AI 架构残留，零引用零内容）。

agent.py 中的旧入口方法（`process_paper` / `process_batch` / `_extract_adaptive` / `process_paper_hypergraph` / `process_batch_hypergraph` / `save_hypergraph_results` / `get_schema_evolution_summary` / `save_results`）已随手术删除——见封存 commit 的 diff。

## 如需受控复跑（新旧对比）

git 历史保全了一切：

```bash
# 回到封存前状态（只读查看）
git log --oneline -- src/granular_agent/extractor.py   # 定位最后版本
git show <commit>:src/granular_agent/extractor.py      # 查看任意旧文件

# 复跑旧管线：在一个旧 commit 上开 worktree，不污染主线
git worktree add ../LogicKG-legacy <pre-archive-commit>
```

## 仍然留在 src/ 的共享模块（勿再移）

审计确认以下模块虽被 legacy 引用过，但 **kernel 新路径/评测资产仍在用**，整体保留原地（B+ 重写时再拆）：
- `hypergraph_extractor.py`（eval scifact/scirex 用 `extract_hypergraph`；kernel 借 `_extract_metadata`/`HGBlackboard` 等内部件）
- `hypergraph_evolution.py`（kernel 深度复用 `detect_*_triggers`/`infer_*`/`consolidate_instance`）
- `grounding.py`（`_tokens` 被 hypergraph_evolution 模块级依赖）
- `structure_mapper.py`（kernel 直用；切块部分待确定性重写）
- `hypergraph_lifter.py`（SCION PK/消融/complex_q 三组活 eval 资产唯一依赖）
- `paper_registry.py`（build_arfm2024_citations.py 依赖）
