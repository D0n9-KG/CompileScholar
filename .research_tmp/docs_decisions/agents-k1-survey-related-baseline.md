# Agents-K1 调研 (相关工作+潜在baseline, 用户指定)

Date: 2026-08-14
来源: arXiv 2606.13669 (curl直接下载PDF到.research_tmp/refs/agents_k1_2606.13669.pdf, 63页)

## Agents-K1 是什么
Agents-K1: Towards Agent-native Knowledge Orchestration
- 作者: Zongsheng Cao, Bihao Zhan 等; Bo Zhang, Lei Bai (♠Shanghai AI Lab + ♣ECNU + ♦Fudan)
- end-to-end pipeline: 论文→agent-native科学KG
- multimodal parser 五模块schema: entities/multimodal evidence/citations/typed inter-entity relations/**method lineages**(方法谱系!和我们方法演化直接重叠)
- 4B IE backbone + GRPO(rule-based reward, 监督NER/RE/long-form extraction)
- graphanything CLI: web search + multimodal graph retrieval + cross-document
- 评 SciERC/SciREX 等 benchmark 的 task-conditioned F1
- 发布百万篇 Scholar-KG 子集

## 全部开源资源(PDF超链接提取)
- GitHub: https://github.com/InternScience/GraphAnything (已clone .research_tmp/GraphAnything)
- HF Model: InternScience/Agents-K1-LLM (4B模型)
- HF Dataset: InternScience/Scholar-kg (百万篇)
- SCP: https://scphub.intern-ai.org.cn/detail/42

## 和我们的关系(初步判断,需读全确认)
- **method lineages重叠**: 它抽方法谱系(BUILDS_ON), 我们方法演化边(extends/improves/compares/replaces/adapts/background) — 方向重叠, 必须作相关工作discuss
- 口径差异: Agent-K1 BUILDS_ON粗粒度单一扩展关系; 我们6种细粒度演化类型 — 对比有适配问题
- 定位差异: Agent-K1=agent-native extraction pipeline(4B+GRPO+multimodal+agent); 我们=schema-induction+自演化+富拓扑 — 定位不同但method lineage部分重叠

## 两条baseline路径
1. **GraphAnything CLI (papers schema)**: OpenAI-compatible HTTP client, env配GA_API_BASE/GA_MODEL/GA_API_KEY接Paratera/deepseek无需改代码; papers.yaml有Method+BUILDS_ON(method lineage); 跑19篇→抽→转格式评ERR/PSC. 适配: BUILDS_ON粗→我们6种映射(诚实标注). 能立即跑.
2. **Agents-K1 4B模型在SciERC/SciREX**: 它原生benchmark+指标,最公平(用户提议在他benchmark测). 但4B模型需GPU,且benchmark是NER/RE非方法演化(任务差异).

## papers.yaml schema (Agent-K1方法相关)
entities: Paper/Author/Affiliation/Method/Dataset/Metric/Baseline/Domain_Term/Problem/Contribution/Finding/Limitation/FutureWork
relations: authored_by/affiliated_with/proposes_method/uses_method/evaluates_on/reports_metric/compares_to/has_contribution/cites/**BUILDS_ON(Method lineage/extension)**

## 下一步
- 路径1 smoke: 装GraphAnything, env配Paratera, 跑1篇论文看Method+BUILDS_ON产出质量
- 路径2: 确认本机GPU能否跑4B(V100/4090可行,无GPU不行) — memory记硬件V100/4090
- 在它benchmark(SciERC/SciREX,本地有scirex_repo)测我们方法: 用户提议,最公平避免适配

## SCION适配削弱问题(用户质疑,诚实确认)
SCION PK数字有水分: 1.任务错配(schema-induction用做instance extraction) 2.relationship→type映射损失 3.没报SCION原生Graph F1. Agent-K1路径1同样有BUILDS_ON→6种适配问题, 须诚实标注. 路径2(它原生benchmark)最公平.
