# 全链路信息流审计判决（FULLCHAIN-AUDIT-0928）

日期：2026-09-28。方法：6 个并行代理逐级深审（S1 入库/S2 编译视图/S3 工具检索/S4 agent 策略/S5 笔记→报告适配/S6 判分），统一框架（信息损失点/语义降级为规则/上游可容忍下游放大/失败不可见/无测量硬参数），全部结论要求真实批数据（批 1-14 轨迹/笔记/分数文件/KB 盘上数据）背书。审计产物：cs2/audit_s5_replay.py 等回放脚本与 JSON 在盘。

**触发背景**：用户批评"修问题太粗暴、缺乏整体思维"——P1-6 orphan 删句被质疑（后实测误杀率 81-94% 坐实）、deep_read 挤出广度效应（b12→b14 笔记 25→6 行）暴露"每次修复只看局部"。本审计=以整个系统为主体的一次全链路体检。

---

## 一、总裁决：零数据损坏，全部通路断裂

六级审计一致的正向核验：
- quote 逐字保真 29/30（S2 分层抽样，coarse/deep/survey/hub 四层）
- chunk 锚与原文逐字符相等（S1 逐字节验证 3/3）
- 判分输入格式与官方 scorer 逐字段对齐、rubric 键全命中、参数=官方默认（S6）
- 粗抽覆盖面好（抽 5 篇 manifest 论文 5/5 有记录）

**问题全部发生在契约接缝上**：每一层单测都过，失真方向一致地朝"模型被告知没有证据"坍缩——恰是 ingredient_recall 最敏感的方向。这就是为什么批间分数在 0.62-0.76 晃：好证据大量存在，但多级通路各断一截。

## 二、证据死亡地图（三层断裂带）

### 断裂带 A：入口（证据查不到）
| # | 问题 | 位置 | 数据 | 来源 |
|---|---|---|---|---|
| A1 | findings kind 白名单排掉 survey_claim+domain_snapshot | tools.py:354 | 12,712+1,580 条（45.7% 记录）不可见，findings 是最高频工具（992 次/三批） | S2-1/S3-5 |
| A2 | claim_type 过滤 × 热库 method/limitation 记录 claim_type=None | tools.py:356-357 | 159 次零命中中 98 次（62%）由此+复合 pid 造成；正确 paper_id 也全零 | S3-2 |
| A3 | 复合 pid（`pid(title;cite)`）精确匹配死 + 语义兜底 KeyError 标"bad args" | tools.py:347,443 | 28/842 次 paper_id 调用撞死；b14 GIS 题 12 次崩溃+16 次静默零，3 篇相关论文整篇不可达 | S3-3 |
| A4 | search_text 覆盖 174/2099 篇（8.3%）但文案称"全部语料全文" | harness:285 | 1,925 篇不在索引；模型 miss 时做"语料没有"假阴性判断 | S1-3/S3-6 |
| A5 | search_text 批 1-10 全程死亡（PS53_TEXT_INDEX 时序 bug）+饱和 stub 谎称"结果在笔记里" | harness:74-77 | 559/823 次调用失败；批1 磨 369 次；饱和计数把失败也计入 | S3-1 |
| A6 | search()（唯一全量 31,309 条语义通道）被提示压抑 | tools.py:9, harness:282 | 三批仅 15 次调用（对比 findings 992） | S3-5 |
| A7 | limitation 通道：粗抽 kind=limitation（1,086 条）无 claim_type，findings(claim_type=criticism) 全灭 | tools.py:356 | 通道上限 195 条；提示词恰把 weaknesses/limits 类问题导到这里 | S2-2 |
| A8 | in_corpus_paper_id 全库 None → 星标永不亮 → 提示词教"无星标=card 必空别花步" | registry_v2, harness:280,204-221 | 3,152 张真实 dossier 只有 20 个被广告；b13 DSL 题弃答根因链主环（模型笔记原话"entities NOT in corpus"） | S2-3/S3-4 |
| A9 | cards 补丁只修 1/7 消费点：stats/describe_kb 恒报 0 dossier、entities in_corpus 恒 False、genealogy year 恒 None | tools.py:763,638; compiler.py:105 | describe_kb 第一步工具就对模型说谎 | S2-3 |
| A10 | as_of/时间轴字段名错配（读 arxiv_year，manifest 只有 year） | tools.py:589, compiler.py:224 | 15,100 节点 year 全 None；as_of 任何年 visible_papers=0 但谱系全通过（精神分裂快照）；当前 0 调用=潜伏 | S2-7/S3-7 |
| A11 | survey/hub 157 篇全文论文不在答题循环可见 manifest | cs2_runner.py:112 | 全文库 95% chunk 的论文对模型不可见 | S1-8 |

### 断裂带 B：中游（agent 策略被机械件挤压）
| # | 问题 | 位置 | 数据 | 来源 |
|---|---|---|---|---|
| B1 | F28 死刑 regex `[0-9A-Za-z]{8,20}` 装不下 CS2 的 52-89 字符 pid → 救援档死代码 → 3 步无证据硬杀 | harness:1292-1340,1503-1534 | 24/25 题被杀（平均浪费 16.8/30 步）；b13 DSL 题第 5 步被杀 | S4-1 |
| B2 | 硬杀后答案由 fallback compile 代笔 + 引用静默剥离 | harness:2209-2330 | 24/25 题走此路径；五批 264 条 citation 被剥——"agent 作答"叙事名存实亡 | S4-2 |
| B3 | EXT_CATALOG deep_read 条目 1200 字符（3 倍篇幅）+全目录唯一强指令簇（"NEVER deep enough"/"do NOT abstain"） | harness:325-341 | git 时间线落在 b13/b14 之间；b14 首步 findings 消失；广度塌缩文本源头 | S4-3 |
| B4 | deep_read sample_records 用 "id" 键，收割器只认 record_id/paper_id/chunk_id | external_tools.py:1133 vs harness:1044 | agent 忠实抄深读 id→bad_backref→[unsourced] 降级；b14 实测 10 连拒、14/28 行降级 | S4-4/S1-7 |
| B5 | 饱和计数器/_DEEP_READ_DONE/_current_question 模块级共享，4 线程并发互污 | harness:110,1403; external_tools.py:1016 | 题间互相清空饱和状态/深读短路/题目串写 | S4-5,12 |
| B6 | fetch_chunk 全文目录硬编码 Multi 语料路径 | multi_ours_run.py:181-182 | CS2 臂 100% 失败（b13 一题 6 次全 error，151/157 字符逐字命中）；另有 chunk 锚当 record_id 传入的键名错配 | S1-1 |
| B7 | 四个教学装置（F28 救援/F22 覆盖审计/pre-answer 审计/min-evidence）CS2 上 fired 0/0/0/0 | harness 多处 | 策略控制层只剩杀开关 | S4-1 |
| B8 | L1 语义节匹配塌缩为全局 top-1 chunk（多节请求只得一节） | external_tools.py:788-807,840-846 | per-query 向量白算；sel2[:5] 硬截断 | S1-6 |
| B9 | L2 裸 daemon 随进程退出死，深记录无 completeness 标记 | external_tools.py:897-1004 | 3/19 篇深读是残缺子集（无 absence）；叠加 B5 短路，第二题拿到残缺 | S1-5 |
| B10 | 深记录 2,419 条与 records_merged 零交集，typed tools 跨批全盲；backflow 每次命中 KeyError 被吞 | cs2_runner.py:100, backflow.py:157 | "KB 随探索生长"在查询层未兑现；L2 完成时还会顶掉粗抽记录 | S2-6 |

### 断裂带 C：出口（证据到不了判分）
| # | 问题 | 位置 | 数据 | 来源 |
|---|---|---|---|---|
| C1 | EvidenceStore 只加载 2/5 id 空间：views_cs2 从未加载（33 个真 id 查无）、chunk 锚 texts_dir 三调用方都没传（批11 修复=死代码）、三文本库三套命名无注册表 | adapt_batches.py:20-37; report_adapter.py:151,209 | 五批 89 个失效 ref 中 73 个（82%）背后是真实证据 | S5-3/S1-2 |
| C2 | P1-6 orphan 整句删：误杀率 81-94%（b14 删 16 句中 13-15 句有真证据；ontology 题 12/12 全真） | report_adapter.py:387-418 | 用户质疑坐实；主犯是 C1 不是 LLM 幻觉 | S5-2 |
| C3 | 8 题整题归零：笔记无 N 形态行→适配器异常→题从 judge_input 消失（degenerate=False 无人知） | report_adapter.py:265; harness:601 | b5/b6/b8/b9/b13 各 1-2 题；历史纵向数字混入结构性丢分 | S5-1 |
| C4 | word_budget 方向反：prompt 要求 600-1200 词，官方 rubric ≤300 满分/≥600 零分（weight 5%） | report_adapter.py:249 vs rubric.py:108-152 | 24 份终稿 9 份落零分区 | S5-4 |
| C5 | 数值召回无校验：8/24 题数值召回<0.7（最低 0.00） | report_adapter.py:237,304-458 | 笔记门有值锚，装配层无闸 | S5-5 |
| C6 | 空 snippets 引用被 scorer 规则化降档（title-only 半信用） | task.py:236-262 | b13 7/27、b14 4/31 空（b12 为 0）——上游 C1 断链的直接后果；b13/b14 citation 跌分部分源于此 | S6-6 |
| C7 | citation scorer 无输出校验：空 claims→静默零；claims 条数=分母全由 LLM 裁量 | citation_eval.py:184-211 | citation 是噪声最大 facet（同题 mean|Δ|=0.173）；可能是臂相关偏置 | S6-2 |
| C8 | 垃圾括号捕获（[plan]/[45]/devel[oped]）混入 ref 并腐蚀 anchor 引文 | report_adapter.py:62,91-93 | 五批 16 个垃圾 ref | S5-6 |

### 横向：测量与效度（S6+S4）
- "GLM 判崩率 10%"证伪：default 全零路径结构性不可触发；真零=absence 型答案；真实失败走 error 键（至今 0 触发）
- "sent_tokenize 10-20 次/题"证伪：all_at_once 路径不走分句；真实 5-8 次大调用/题——**提速方案 B 撤销**
- 同题混合方差（系统重跑+判分）：global mean|Δ|≈0.10 max 0.245；**n=3-5/批的臂间差需 >0.15 才越噪声带**
- 批间纵向比较混杂四股力量：目录文案、门阈值、KB 漂移（deep_read 缓存跨批持久化）、LLM 非确定性——b12→b13 零提交仍分叉
- 重试封顶 21→5 的语义：失败题从均值剔除（幸存者偏差）而非拉低均值；跨批配置不连续
- 判分 metadata 全丢弃（per-criterion/per-claim 明细）——排雷与审稿举证不可审计
- perplexity 臂 citation=0 是格式合规性度量（无 snippets），论文须披露口径

## 三、修复方案（统一优化，按预期收益×成本排序）

### P0 通路修复（白捡分：全部是"通路接通"而非新能力）
1. **findings 语义解锁**（A1+A2+A7）：kind 白名单放行 survey_claim/domain_snapshot（映射 claim_type）；claim_type=None 的记录在带 claim_type 过滤时放行（或降权不排除）；contains 的 `|` 语法推广到 claim_type
2. **F28 pid regex 修复**（B1）：gap 正则长度上限 20→96；括号扫描 40→96；救援档复活
3. **EvidenceStore 全空间加载**（C1）：加载 views_cs2.json 内嵌记录；texts_dir 三调用方接线；paper→路径注册表统一三文本库命名
4. **fetch_chunk 解绑 Multi**（B6）：文本目录改为注册表查找
5. **复合 pid 解析**（A3）：findings 入口剥 `(` 后缀；KeyError 修复；错误标签语义修正（bad args vs infra）
6. **星标/in_corpus 修复**（A8+A9）：从 cards 视图回填 in_corpus 判定（不再依赖 in_corpus_paper_id 字段）；describe_kb/stats/entities 读补丁后的真实数；提示词删掉"无星标=card 必空"的谎话
7. **word_budget 反转**（C4）：600-1200 → 250-450（官方 ≤300 满分）
8. **orphan 删除改为降级保留**（C2）：C1 修好后 orphan 本应只剩幻觉 id；在此前提下 P1-6 改为"删引用标记保留句+标 [unsourced]"或保留删除但逐句记录被删内容

### P1 策略层
9. EXT_CATALOG 再平衡（B3）：deep_read 条目瘦身到与广度工具等幅；恢复"Dossier-first for breadth"引导
10. id 键名统一契约（B4+A3 后半）：三出口统一 record_id 键名；chunk 锚格式统一并让 fetch_chunk 认锚
11. 并发隔离（B5）：饱和/深读状态 per-question 化
12. 整题归零防护（C3）：适配器异常时降级产出（哪怕 title-only 引用）+ degenerate 标记 + 结构化 ledger
13. 时间轴字段名（A10）+ 47 垃圾键清理（S1-9/S2）+ L2 截断标记（B9）
14. 深记录进 records_merged 或运行时合并（B10）——"KB 生长"兑现

### P2 测量层
15. direct_judge 保存 scorer metadata（per-criterion/per-claim/重试数）
16. gather 加 return_exceptions；error 题 resume 重试一次
17. score_compare 修 pre-fix 双行；批间比较只在新配置内做
18. **验证设计**：修复后批 15 = 双样本 A/B（同 KB 快照、同题、n≥2 重跑取均值）；判分同答案双判一次定界纯判分噪声；判读时用"噪声带 ±0.10"卡尺

### 撤销项
- 提速方案 B（NLTK 分句替换）——前提错误，撤销
- "全零题重判"排雷规则——对象不存在（真零是真零），简化为 error 键监控

## 四、预期与纪律
- P0 全部完成后跑批 15 双样本；预期通路修复合计值 +0.05~+0.15 global（62% 零命中主因消除+24/25 被杀+73 ref 断链+9 份零分区长度），但**必须以数据验证而非预期汇报**
- 每项修复标注对应审计编号（可追溯）
- 修复不引入新语义规则——凡需语义判断处（orphan 真伪等）走"上游修好+降级保留"而非硬删
- 本档为权威档：后续修复逐项引用本档编号，防止再次"局部修复无整体账"
