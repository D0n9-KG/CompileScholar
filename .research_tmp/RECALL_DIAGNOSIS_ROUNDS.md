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

## 四、轮次实验链（同 10 query，每轮只动一个变量）

| 轮 | 配置 | micro-F1 | 关键变化 |
|---|---|---|---|
| R1a | crossref-only, LLM 排序 | 0.007 | 覆盖墙（gold in pool 5/129） |
| R1b | crossref-only, 权威重排 | 0.000 | 覆盖墙之下两臂同盲 |
| R2 | +S2 bulk（4 改写/查询） | 0.007 | 池变大但改写太窄 |
| R3 | S2 bulk 8 改写（含 survey 变体） | 0.014 | q1/q2 首命中；gold in S2 pool 14→更大 |
| R4 | +per-query top-20 trim +缓存 v2 修被引埋没 | **0.086** | q0=0.40 / q9=0.308 / q1=0.091；tp 合计 12 |
| R5/6/7 | 判分窗口 150+rewrite 缓存 | 0.057（三连同分=管线确定性） | R4 的 0.086 判为 rewrite 漂移的好运抽签 |
| R7 | **chunked grading（150→3×50）+ rerank 键 bug 修复** | **0.093** | 6/10 查询命中；q2=0.222（注意力稀释修复）|
| R7b | 同确定性管线 rank=llm 配对臂 | 0.100 | 唯一差异 q8（-1tp）→ **弃事后重排，定调 authority-in-prompt** |

**配对终判（确定性管线）**：auth rerank 0.093 vs llm 档序 0.100。判分器 prompt 已
显示被引数，事后被引重排中性偏负（q8 cap 边界切 gold）。机制定调=被引数在判分
时融合，不做独立重排层。首轮"中性"A/B 曾被 rerank citationCount 键 bug 污染
（S2 行键名不同→全部 0 分），教训：**A/B 前先验证 ranker 真的在排序**。

## 召回天花板（当前池）

gold in (S2+crossref) 池 = **63/129 = 49%** → 完美判分下 recall-bound F1 ≈ 0.49。
当前 0.093 的差距构成：~51% 覆盖缺失（引文滚雪球欠账）+ 判分精度（窗口内 gold
仍漏判）+ 15 篇输出截断（q1 gold=29）。

R4 的两个修复：
1. **S2 缓存版本 bug**：早轮缓存条目无 citationCount 字段（后加），被引预排序把它们
   当 0 被引沉底——facet gold（被引 12-65）排到 128-254 名，切出 top-100 判分窗口。
2. **全局被引排序 vs facet 多样性**：1930 池全局排序必然挤掉低被引 facet gold。
   改为 per-query top-20（保每个改写查询的代表）→ 并集再全局被引排序。

## 五、结论与下一步

1. 召回主干切换：S2 bulk（多改写查询+survey 变体）+ crossref（权威被引数补充）+
   OpenAlex（解封后并入，元数据最全）。arXiv API 备用。
2. 权威重排的真实 A/B 在 S2 bulk 池上跑（R4 起池健康）。
3. 长期：申请 S2 API key（免费表单，100 req/5min）。
4. q5 类极窄查询（gold=1 篇 voice conversion GAN）单靠 keyword recall 结构性难命中，
   引文滚雪球（L1 的第二腿）才是正解——下个实现位。
