# 新 goal 方向转换 (2026-08-14)

旧 goal (升层机制) 核心已验证完成, 转新 goal:
- 核心1: 超图结构信号全利用升层 (不止文本归纳)
  当前 lift_corpus 本质"超图拍扁成文本喂LLM", 浪费 n-ary 超图结构价值.
  要用: 节点共现/qualifiers/pattern拓扑/n-ary多节点/引用关系/cited_from 提取被引方法
  验证: 结构信号+文本结合 vs 纯文本归纳, 谁准 (重新评测对比)
- 核心2: 论文注册底座完整闭环 (DOI/title→原文+超图+引用统一存取, 用户应用基础)
- 顶会要求: 综述复杂问题评测 + 公开数据集(SciREX/SciFact) + SOTA baseline公平PK + 消融

## 起点资产 (不重复)
- 升层机制 lift_corpus (commit 1e678d42/eb7831c3/013f215d/225bc013/99c83a40)
- 引用关系底座侧A (sci-evo-extract, commit 0da9d14): registry paper_references/paper_citations + sources提取 + API /papers/{id}/references?fetch=true, 端到端验证30条
- judge_relation 用 GLM-5 (合规+绕deepseek400)
- 建模形式judge (commit 99c83a40): segregation 8篇20关系GLM-5判100%正确

## 阶段顺序
1. LogicKG paper_registry.py HTTP client (接侧A API)
2. corpus_driver接paper_registry (双轨, 超图按paper_id存)
3. lift_corpus用引用关系作结构信号升层 (首个结构信号, 验证vs纯文本)
4. 结构信号扩展 (节点共现/qualifiers/拓扑/cited_from)
5. 重新评测结构vs文本
6. 论文注册底座完整闭环
7. 评测设计 (综述问题+公开集+baseline+消融)
8. 成文 (academic skills)

## sci-evo-extract 服务起法
cd Desktop/sci-evo-extract && .venv/Scripts/python.exe -m uvicorn sci_evo_extract.api.app:app --port 8765
