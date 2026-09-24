# 单篇QA benchmark + RAG基线选型调研（子代理报告，2026-09-27，一手核验）

## A. 单篇 QA 判决
- **勘误**：QASA venue=ICML 2023 非 ACL 2024；题量 1554+244 非 1375（本地 qasa_test.jsonl 1375 为子集）；PeerQA=NAACL 2025 确认。
- **推荐：QASPER + PeerQA 双件套**（=ScholarStack 同款，可直接对表其 Full-text/RAG/ScholarStack 三臂数字）：
  - QASPER：官方 Answer-F1+Evidence-F1 程序化判分，5049 题可子采样（ScholarStack 用 1428 题/416 篇）
  - PeerQA：2025 新鲜度，579 题/208 篇，真实审稿人问题+原作者作答，官方三轨判分（nDCG/answerability F1/ROUGE-L）
  - 协议坑：PeerQA 检索必须全库检索，oracle 逐篇索引会虚高 R@10 0.011→1.000（Agents4Science 审计论文实证）
  - QASA 不作主选（无官方判分器+原论文自认噪声），可作第三轨走 OpenScholar 的 ROUGE-L 协议
- **单篇 QA 外部方法**：PaperQA2 + OpenScholar 唯二被 2025-26 反复实际运行的权威系统

## B. RAG 基线（5 篇 2025-26 论文基线表实读统计）
- 出现频率：HippoRAG 2 3/5、RAPTOR 3/5、Naive RAG 3/5、GraphRAG 2/5、LightRAG 2/5、PaperQA2 2/5、OpenScholar 2/5
- **结构性发现**：图/树 RAG 家族统治固定语料学术 QA 基线表；agentic 科学 RAG（PaperQA2/OpenScholar）统治开放检索
- **固定语料跨论文 QA 首选基线：HippoRAG 2**（ICML 2025，3/5 最高频+QASPER 最强基线 60.32 F1 被SF-RAG击败的公开数字链+PGR/AGR 双模式开源成熟+语料级多跳 QA 设计目标精确对位）
- 次选 GraphRAG（建库成本高）；RAPTOR 旧代际参照

## 附加发现
- ⚠️ arXiv 2608.07370 外部团队已发布同名 LitTraceQA 基准（Liu Xuye 等 8 人，4978 题）——我们若沿用此名有撞名风险（历史计划中的 LitTraceQA 战役名）
