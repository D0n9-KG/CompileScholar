# DECISION: 抽取 prompt 域无关化 + step3 qualifier 对齐

## 背景
泛化是当前任务。审计发现 4 个 extract prompt + _ALIGN_PROMPT 有 20+ 颗粒流特有内容
(μ(I)/NGF/segregation/Bagnold/Janssen/GDR MiDi/I-gradient/Gray vs Tripathi + PIV/MRI/几何长排除清单)，
对非颗粒域是阻碍。schema seed 层已具备泛化(seed_meta_hypergraph_general 通用 seed 已存在)，
瓶颈在 prompt 的例子和判别清单。

## 决策
**Option A 多域例子集**(用户铁律"典型可作例子标举例非规则"+ domain-specific 要 domain-gate):
- 每个判别点给跨域例子(颗粒流+ML+生物+化学)，全标 "illustrative, not rules"
- 原则定义 crisp 保留(METHOD=数学/科学 MODEL of behavior；不限于"material behavior")
- evolution decision logic 保留(improves=解决局限/extends=推广范围/compares=并行不同机制)，
  只把颗粒流举例换抽象表述
- 单次 EXTRACT_HG_PROMPT 也泛化(文档化默认+fallback，留颗粒化=半成品埋雷)
- 不 domain-gate 例子(每新城要手工配库=反泛化)
- 不动: 物理 seed(通用 seed 已存在,物理 seed 对物理域是正确特化)、evolution 域 regex(已 is_granular 门控)、
  lifter.py/chained_extractor.py(待删影子,不浪费工)

## 顺带修: step3 prompt-schema qualifier 不一致(预存在 bug)
验证 5 篇时发现 36/40 边被 validate 拒(no-matching-meta-pattern)。根因:
1. step3 给 LLM 全局 qualifier key 集,LLM 把 relation_type 加到所有边,
   但 constitutive_law/influences 的 allowed_qualifiers 不含 relation_type → 拒(主拒因)
2. evidence_strength enum 写错(claimed 应为 hypothesized)
3. cited_from enum 写错(cited/external 应为 prior_art/definition)
4. role "Do NOT invent 'subject'" 但 subject 是 defines 合法 role(自相矛盾)

修: step3 改为**按 pattern 列 allowed qualifier keys + 正确 enum**(对齐 QUALIFIER_REGISTRY)。
这不是泛化引入的(我没碰 step3 qualifier 区,git diff 可证),但泛化验证暴露了它——
不修则泛化后的 prompt 产 ~0 边,无法验证泛化质量(违背铁律#5 地基看边)。修后 Jop_2006
val_fail 36→0, 边数 3→40。

## 验证(双向)
**颗粒流(无退化)**: 5 篇 ARFM2024 V4-Flash 多步, 665 concepts/257 hyperedges
(baseline 412/117, qualifier 修复解锁边产出), 标签健康(METHOD 充足, n-ary arity 2-17),
law_parameter 富拓扑主导, val_fail=0。
**ML 跨域 smoke**: Transformer/attention/Adam/Multi-head attention 全正确标 METHOD;
learning rate/dropout/h=8 标 PARAMETER; 边 composed_of/defines/constitutive_law/compares/measures
全合理; **GRANULAR LEAKAGE=[]**(零颗粒泄漏,多域例子未偏向颗粒)。

## 预存留(非泛化引入, 后续)
- LLM 自造 pattern_type 复合名("constitutive_law_parameter_dependency")——validate 按结构匹配不按名,名是噪声
- METHOD 命名仍泛化("rheology"/"theory" 而非 canonical)→ gold 对齐弱
- 长 chunk 响应截断(26K→0 he)——max_tokens 限制
- Hill viol=47 偏高

## commit
hypergraph_extractor.py + concept_graph.py: 4 个 multistep prompt + EXTRACT_HG_PROMPT +
_ALIGN_PROMPT 泛化 + step3 qualifier 对齐
