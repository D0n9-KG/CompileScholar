# 跨论文QA基准判决（子代理报告，2026-09-27；存在性/venue 一手验证，采用数字格子部分未完成——其未完成项已由"ScholarQA数字全景"报告补齐）

## A. 固定语料跨论文 QA
- **MDAQA 唯一正身候选**：EMNLP 2025 Findings+797 题/每题 2-4 篇 gold+HF/GitHub 公开——唯一同时满足"正式 venue+固定 gold 多篇形态+数据公开"
- 候选池其余（均一手确认存在但作备选）：M3SciQA（arXiv 2411.04075，多模态）、PaperScope（ACL 2026 Findings，16 篇缩水前科）、LitBench（medRxiv 2026-09 预印本，医学域）、DocNavRAG（系统论文非基准）
- 附注：DocNavRAG（2608.01565）在 MDAQA 上有结果——MDAQA 采用面补证

## B. 开放语料跨论文 QA
- **ScholarQA-Multi 维持最佳**：Nature 2026 正刊+repo 公开+无同形态替代者
- AstaBench（ICLR 2026）是标准化入口信号，但 ScholarQA 数字全景报告已确认：AstaBench 只收 CS 域，Multi 不在其中
- 2604.25256（AutoResearchBench）由开放检索报告覆盖（形态是文献发现非 QA）

## C. 外部方法
- OpenScholar = ScholarQABench 配套系统，Multi 对照不可绕开（但 repo 停更 2025-08+全管线绑定 OSDS 45M Datastore，复刻成本高——已有结论）
- SciRAG 维持除名（EACL 2026 venue 确认但 repo 404）
- DRACULA（allenai/dracula）= 2026 新的 deep research agent 训练方法，潜在基线候选（观察）

## 未完成项的补齐状态（由其他四路报告覆盖）
- MDAQA 采用情况 → ScholarStack 用 797 题全集+5 维判分（已确认）
- AstaBench ScholarQA 角色 → 只有 CS 域，Multi 不在（已确认）
- OpenScholar Nature Table 1 数字 → 已拿到全表（Multi LLM 4.51/Cite 37.5 等）
