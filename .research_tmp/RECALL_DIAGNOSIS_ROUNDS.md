# SPARBench 召回源诊断 + 权威信号实验记录（2026-08-28）

练习赛第四轮。前三轮结论见 memory/contest-downstream-sprint.md。本轮把"排序缺权威
信号"假设放回实验，撞上更根本的墙，然后找到了出路。

## 一、覆盖墙实测（本轮最重要数字）

| 召回源 | gold 覆盖（10 query 实测） | 结论 |
|---|---|---|
| crossref（keyword+semantic，207-232 篇/query 池） | **5/129 = 3.9%** | 大量 gold 是 arXiv-only（DataCite DOI），crossref 结构性不收录。F1 天花板≈4%，排序层怎么改都没用 |
| OpenAlex | 未测（日预算 $0.1 耗尽，UTC 午夜重置） | 预期覆盖好（收录 arXiv），当前不可用 |
| S2 普通 search | 0（匿名池硬 429，无 key） | 不可用 |
| **S2 bulk search**（`/paper/search/bulk`） | **5/10（q0，naive 查询）** | **可用且稳定（7/8，偶发 500 重试即过）**。AND-token 匹配 title+abstract，需 LLM 改写多查询补覆盖 |
| arXiv API | 3/10（q0，pool 281） | relevance 排序弱，可作补充源 |

**修正第三轮诊断**："gold 标题可搜到=排序问题非覆盖问题"只对了一半——标题直搜能命中，
但**查询召回**（问题→改写→检索）在 crossref 池里结构性缺失 gold。真实病灶是双层的：
召回源覆盖（crossref 4%）+ 排序口径（代表性论文）。

## 二、权威信号的两面发现

1. **SPARBench gold 的 citationCount 中位数=0，55% 零被引**——但其中 2024-25 年占
   156/303，且 2021 年的 "Domain Generalization: A Survey"（著名综述）也显示 0 →
   **benchmark 的 citationCount 字段大量缺失**，不能用它反推"gold 不偏好高被引"。
   老 gold（2015 前）非零被引中位数 940。
2. crossref 候选行的 `is-referenced-by-count` 可靠（sci-evo 已接线透出
   `citation_count` 字段，服务已重启生效）。
3. 权威重排（H/S 档位 × log 被引）在 crossref-only 池上 F1 0.000 vs baseline
   0.007——覆盖墙之下无意义（两者都盲）。**待 S2 bulk 池跑完见真值**。

## 三、工程事实（报告可用）

- OpenAlex 429 根因升级：不是 IP 封禁，是**日预算 $0.1/1000 次耗尽**（mailto 同池），
  UTC 午夜重置。 Retry-After 实测 12.6-15.3h。
- S2 `/paper/search/bulk` 不受普通 search 的匿名限流墙约束，~5-10s/query，
  返回 title/year/citationCount/venue/externalIds——**今天唯一可用的 arXiv 覆盖召回源**。
- sci-evo `discovery_candidates_to_rows` 新增 `citation_count` 透出
  （crossref is-referenced-by-count / openalex cited_by_count）。

## 四、结论与下一步

1. 召回主干切换：S2 bulk（多改写查询）+ crossref（权威被引数补充）+ OpenAlex（解封后
   并入，元数据最全）。arXiv API 备用。
2. 权威重排的真实 A/B 在 S2 bulk 池上跑（进行中）。
3. 长期：申请 S2 API key（免费表单，100 req/5min）。
