# FROZEN (2026-09-05, Stage B kickoff)

本目录是比赛期/验证期的旧栈（超图 20 模块），**自 Stage B 起冻结：不改一行代码**。

- 保留原因：16 处存量引用 + 消融对照价值 + 历史基线可复现。
- 新栈在 `src/kb_compiler/`（记录层+编译层），基础设施在 `src/kb_infra/`
  （llm.py = 本目录 llm_client.py 的忠实移植；embedding.py 整合了
  hypergraph_evolution._embed_texts_robust 的两级 fallback）。
- 新栈 **禁止 import 本目录任何模块**——由
  `tests/test_kb_compiler_import_gate.py` 机器闸强制，越界即红。
- 冻结决议与隔离五条见 `.research_tmp/docs_decisions/STAGEB-EXTRACTOR-SPEC-v1.md` §6。
