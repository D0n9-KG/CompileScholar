# 必核清单 #3：ReportBench HF 实拉核验（2026-09-27 深夜）

## 结论：✅ 可用，与设计档口径一致

- repo：`ByteDance-BandAI/ReportBench`（HF dataset，61 downloads）
- 数据：`data/test-00000-of-00001.parquet`，**100 任务**（已实拉到
  本地 HF cache）
- 每题字段：prompt / ground_truth（引用清单）/ arxiv_id / title /
  abstract / authors / categories / application_domain 等
- 源综述元数据齐全（arxiv_id+title+abstract+authors）

## 关键核验

| 项 | 实测 |
|---|---|
| 规模 | 100 题（设计档"ML 子集 25 任务"从这里选） |
| 时间约束 | **99/100 prompt 含时间措辞**（"before April 2..."）——as_of 原生对齐坐实 |
| gold 形态 | 逐题引用清单（author/bib_id/title/meta_info）；引文数 min=5 / **中位 137.5** / max=884 |
| 判分可行性 | micro citation P/R 对 gold 引用集（确定性）+RACE（GLM judge）——中位 137 篇 gold 意味着召回面大，检索质量承重 |
| 域分布 | 十 application_domain；"AI and Data Intelligence"9+"ICT"11=20 题≈ML 子集候选池（设计档 25 任务的池子够） |

## 槽 4 落地备注

- 选题：从 AI+ICT 20 题里选 25（或按题目时间约束多样性补
  Basic Research 里 ML 相关题）——正式执行时定并预注册
- 对手：STORM 本地化（必核 #6 未做）+ CC harness 臂（0.1/0.3 已就绪）
- 我方臂：热启动基库 as_of 过滤（基库综述层本身就是分年代的——
  149 篇综述的年代分层 37/45/67 直接支撑时间切片供给）

（清单 #4 MDAQA license/#5 PeerQA 实拉待后续并行窗口）
