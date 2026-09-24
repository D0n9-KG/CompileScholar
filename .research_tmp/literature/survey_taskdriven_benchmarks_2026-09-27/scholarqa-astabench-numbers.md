# ScholarQA/AstaBench 公开数字与 judge 口径（子代理报告，2026-09-27，一手核验）

## 关键判决
1. **ScholarQA-Multi（108 题多域）不在 AstaBench 里**——AstaBench 只有 CS 单域（CS2）。Multi 上的公开可比数字只有 Nature 论文那一套（OpenScholar/PaperQA2/Perplexity/GPT-4o/Human，judge=Prometheus v2）。
2. **两代 judge 口径已分叉不可混对**：Nature（GPT-4o Turbo rubric + Prometheus v2 + 引文 F1）vs AstaBench CS2（gemini-2.5-flash→2026-03 起或为 gemini-3-flash，四指标等权）。leaderboard 同配置分数比论文高 1-2.4 点（rubric 重算），引用必须注明来源。
3. **Valyu"总榜第一"实锤 gaming**：厂商自报（74.5% CS），其自述"按 rubric criterion 生成 JSON schema 强制逐条覆盖"=把测试集 rubric 泄漏进生成端，协议无效。豆包清单此条正式处决。
4. CS2 rubric 有 held-in 偏置：提材系统若被 holdout 平均掉 2.5 分（论文自证 p≤0.01）——外部新系统进场会被压分。
5. Leaderboard 现行页面只有公开提交条目；论文表里的 Elicit/OpenAI DR/STORM/Crow/Falcon 数字在论文 Table 6/7，不在榜上。

## Nature Table 1 数字（Multi LLM judge=Prometheus v2 1-5 分制 / Cite F1）
- OpenScholar-GPT-4o: 4.51 / 37.5（Cite）——Multi 榜主
- PaperQA2: 3.82 / 47.2
- GPT-4o 裸: 4.16 / 0.7；GPT-4o+RAG: 4.03 / 31.5
- OpenScholar-8B: 4.12 / 42.8；OS-70B: 4.03 / 54.7
- 人类专家 Multi 引文 P44.4/R41.5；16 PhD 盲评 OpenScholar-GPT4o 胜人类 70%

## AstaBench CS2 test（论文表，0-100）：Asta Scholar QA w/Tables 87.9 > Elicit 85.5 > Crow 81.1 > OpenAI DR 79.4 > STORM 78.3 > OpenSciLM(=OpenScholar-8B) 58.0
## Leaderboard test（0-1 制）：榜首 gpt-5.4 Asta 0.919（judge 或已换 gemini-3-flash，需开箱日志确证）

## 产物文件
.research_tmp/scratch/ 下 astabench_arxiv.pdf / nature_full_clean.txt / nature_t*.txt / leaderboard_lit*.txt
