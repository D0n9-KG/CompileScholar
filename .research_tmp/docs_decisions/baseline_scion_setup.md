# Baseline SCION 真版适配流程 (从空中楼阁到真版对比)

Date: 2026-08-14

## 之前问题(诚实)
baseline PK 用自己复刻的"IncSchema适配版direct-probe", 非开源原版。审稿必质疑"把对手实现弱了", 空中楼阁。

## 真版SCION适配 (github.com/wandugu/paper_scion, 已clone .research_tmp/paper_scion)
1. **独立venv装核心依赖** (.venv, 不污染backend): openai/requests/networkx/jsonschema/yaml/numpy/bs4/lark/knowledge-graph-maker
2. **DeepSeekClient跑通**: SCION原生支持deepseek (config llm.provider:deepseek), smoke返回OK. LLM backend零代码改动(诚实)
3. **输入适配**: 19篇论文md (.research_tmp/pilot_refs/ARFM2024/*.md) 聚合成background_methods_en.txt (46961字符)
4. **ontology schema改方法演化** (config_methods_en.yaml, 诚实只换schema定义不改SCION方法逻辑): entity=ModelingMethod, 6关系(extends/improves/compares/replaces/adapts/background=我们gold边类型)
5. **去socks proxy** (config llm.proxy=None, 本地无socks)
6. **开graph_extraction**: GraphMaker.from_documents抽KG实例边
7. 跑 `OT_CONFIG_PATH=config/config_methods_en.yaml python -m src.ontology_generate`

## SCION产出
- graph_edges_en.json: 59条边 (node_1.name/node_2.name/relationship描述句)
- 81方法 (抽得多但碎, 含噪声如"3D model"/论文标题碎片)
- relationship是自然语言描述句, 非type → scion_to_lift_format.py用关键词映射+LLM兜底转type
- 转后: 81方法58边, type分布 extends10/improves8/compares12/adapts5/replaces1/background22

## 评测bug修复(诚实教训)
首轮SCION评测ERR=0是eval_err_edges的LLM对齐bug: 81方法一次对齐超max_tokens截断→parse失败→空alignment→0命中。修复:align()分批每20方法。SCION实际抽了"inertial rheology"/"local rheology"/"gradient expansion of μ(I)"等明确对得上gold M1/M1/M17的方法。

## PK (重评中)
- 我们的B臂: fair_recall 0.833, ERR_uncond 0.098, ERR_cond 0.667, PSC 0.667 (单seed)
- SCION baseline: 重评中(分批对齐后真实数字)
- 同一ARFM gold + 同一19篇输入, 公平PK
