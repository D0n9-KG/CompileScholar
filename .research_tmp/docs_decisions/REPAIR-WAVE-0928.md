# REPAIR-WAVE-0928 — 系统地毯审计后的集中修复波

> 依据：五路并行审计（KBTools/深抽管线/答题循环/编译链/检索层）+
> 判分层自查 + 批1-13 全量实证。本档=修复计划（预注册式：每项
> 冻结判据后再动手），执行后跑批 14 验证。

## 修复清单（按伤害排序，全部通用无题特调）

### P0-1 card 视图空壳修复【严重·已验证】
- 病：views_cs2.json 的 cards 内层 0 dossier（15100 实体），card()
  工具 13 批全域死亡（26 次调用全 error）；hub_cards.json 有 21 篇
  卡片数据在盘但未编译进视图。连带 F31 A-block 三来源之一死亡。
- 修：视图编译接线——hub_cards 的 21 个 dossier 编译进 views_cs2
  的 cards 键（编译侧补线，非重建）。
- 判据：批14 card() 调用出现非 error 返回；auto_rows>0。

### P0-2 F31 A-block 三来源修复【严重·已验证】
- 病：批12/13 auto_rows 全零。来源①card 死（P0-1 连带）；②findings
  entries 无 value 字段（天然零命中）；③compare 极少调用。
- 修：auto_transcribe 的 findings 路径改为解析 claim 文本中的数值
  （regex 提取数值+上下文入 A-block），与 compare/card 路径并存。
- 判据：批14 a_rows>0、a_chars>0（数值密集题明显）。

### P0-3 findings(contains) 语义兜底【高·批13 弃答根因】
- 病：裸子串匹配，同义词盲赌（'teach by example' 11 hits vs
  'programming by example' 1 hit→整题弃答）。模型最高频工具
  （63 次/批）零语义容错。
- 修：词法零命中时 embedding 兜底——查询词组 vs 候选记录 claim 的
  余弦（阈值 0.50 同 deep_read 定标），命中记录以"semantic match"
  标注返回。记录 claim 预 embed（复用 emb_cache 机制）。
- 判据：批13 弃答的 DSL 题在批14 不弃答（contains 同查询语义命中）。

### P0-4 笔记静默裁剪可见化【高·批13 回退第二根因】
- 病：>5500 触发机械裁剪，cap 3000，被剪内容对模型不可见（批13
  剪 9489 字符/批）。
- 修：裁剪发生时在下一步 obs 注入系统提示（"笔记超限已剪 N 字符，
  涉及：[被剪行的 paper_id/主题摘要]"），模型可见可控（主动压缩）。
- 判据：批14 notes_truncated 事件的模型响应（下一步笔记长度下降
  或主动整理）。

### P1-5 anchor 内容不再丢弃【中高·81 行实证】
- 病：parse_notes 的 split("|")[0] 把 '| anchor:"verbatim"' 整段
  剥掉——最高质量证据被无差别丢弃（批12 每题丢 0-6 个数值）。
- 修：anchor 内容进两个去处——①claim text 保留（数值在主张里）；
  ②EvidenceStore 解析失败时 anchor 作 1.0 档 snippet 兜底。
- 判据：批14 适配报告的数值密度上升（distinct numbers > 批12 的 5/题→8.6/题 基线再升）。

### P1-6 orphan C 编号修复【中高·19 句实证】
- 病：narrative LLM 编造 [Ck] 超界编号→装配时句子保留为无引用主张。
- 修：装配时编号超界→句子删除（宁可少句不可无证据句）+诊断计数。
- 判据：批14 orphan_markers=0。

### P1-7 snippet 聚合上限【中高】
- 病：同论文 snippets 无上限 append（中位 7 条/引用，最大 11），
  稀释 citation entailment 信号。
- 修：每引用 snippet 上限 3 条（按与所在节文本的词面相关度选）。
- 判据：批14 citations 的 snippets ≤3/条；citation precision 不降。

### P1-8 search_text 饱和拦截宽松化【中·误杀一半】
- 病：_SEARCH_FAMILY token 重叠判定，6 条不同措辞查询被误拦
  （批12 ontology 题 12/23 次调用返回 286ch 饱和提示）。
- 修：饱和阈值 3→5 次同主题；且查询含新实体词（不在前 3 次查询
  的 token 集里）时不计同主题。
- 判据：批14 search_text 饱和拦截率 <20%（批12 实测 52%）。

### P2-9 首步空返回护栏【高·批13 放大器】
- 病：首步检索空返回无专门干预，模型可能 3 步弃答（F28 只管
  打转不管速弃）。
- 修：前 3 步内零命中时注入系统提示（"早期零命中通常是词面不匹配
  （记录用语≠问题用语），建议：①entities 查同域实体名 ②search_text
  原文检索 ③换 2-3 个同义词"）——把审计里已知的"猜词"行为
  系统性纠偏。
- 判据：批14 无 3 步内弃答。

### P2-10 lineage 空返回恢复【中·核心卖点工具】
- 病：边端点整串精确匹配；空返回零 hint 零 note 纯死路。
- 修：空返回时附 nearest 实体名建议（复用 card 的 redirect 机制）。
- 判据：批14 lineage 空返回 obs 含 nearest_candidates。

## 执行顺序
1. P0-1→P0-2（card 接线+A-block，同一根因链，一起修一起验）
2. P0-3（findings 语义兜底，独立）
3. P0-4+P1-8+P2-9（答题循环行为三件，同文件一起改）
4. P1-5+P1-6+P1-7（编译链三件，同文件一起改）
5. P2-10（lineage 恢复）
每步完成即单测，全部完成→批14（同批12/13 五题）验证。

## 批14 验收判据（冻结）
- 主判据：global ≥ 0.78（基线 0.738/0.760 的置信上沿）
- 行为判据：card 非 error 返回≥1 次；auto_rows>0；DSL 题不弃答；
  orphan=0；饱和拦截率<20%
- 回归判据：弃答=0；批11-13 已修行为不回退（sample_records 引用、
  chunk 锚解析、unsourced 剥离）
