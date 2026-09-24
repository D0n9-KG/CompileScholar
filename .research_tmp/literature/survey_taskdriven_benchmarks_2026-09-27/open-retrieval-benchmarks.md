# 开放语料检索基准与方法判决（子代理报告，2026-09-27，一手核验 215 次工具调用）

## SAGE 核实
- Yale 团队（Tiansheng Hu/Yilun Zhao/Arman Cohan）借 allenai org 发布，非"AI2 官方"；预印本无 venue 无 leaderboard，HF 下载仅 132
- 采用度极低：7.5 个月 ~16 引用里仅 1 篇真评测（ScholarStack 89 题子集）
- **语料 20 万篇未公开发布且不可复现**（非确定性抽样）；HF allenai/sage-retrieval 只有 query+gold（7MB）
- **检索轨 API 形态可行**：S2 paperId 100% 覆盖=唯一对齐键（DOI 37%/arXiv 11%）；gold 出现在输出/引用中即得分=官方 web-search 协议同构；自建评分器有口径差
- 官方锚点：corpus 轨 BM25@10 EM 81.2 > LLM 检索器（agent 关键词式子查询所致）；web 轨 GPT-5 EM 71.7/WR 26.3；SearchR1-32B EM 35 失效

## 基准候选判决（开放检索）
1. **AutoResearchBench**（2604.25256，RUC）：1000 查询/300 万 arXiv+开放 web，Wide 轨 IoU 奖励召回完整性（与我们召回主张对口），SOTA 仅 9.31% IoU=展示空间大，HF+管线全公开
2. **PaperFindingBench**（asta-bench，ICLR 2026）：333 查询真实用户日志，AI2 生态+leaderboard，S2 开放语料；弱点规模小
3. PaSaMaster-Bench（2605.14306）：244 任务/38 学科，专家 checklist+幻觉率指标
4. **SciNet（2601.03260，清华 FIB）=战略级发现**：8940 任务/2.69 亿论文元数据，任务=关系感知检索（谱系/佐证/冲突/演化路径重建）——与我们知识模型卖点**逐字重合**；实测所有现有 agent（o3-DR 62.9%/PaSa 19.5%）在关系任务上崩坏
5. 不合适：LitSearch（固定小池）/ScholarQA 系/MDSAQA 等形态错配

## 外部方法判决（开放检索）
- **PaSa-7B（ACL 2025 Main，bytedance/pasa）**：代码+权重全开源，专属基准 recall@20=0.5301（vs Google+GPT-4o 0.1921）——最干净开源对照
- **DR Tulu-8B（2511.19399，AI2）**：开源，SAGE 官方基线 EM 42.0/WR 17.4
- GPT-5/Gemini 商用系统：直接引用
- Search-o1/DeepResearcher/WebThinker：**全部无学术检索基准数字**（只有 GPQA/GAIA/HotpotQA），只能重跑不能引用
