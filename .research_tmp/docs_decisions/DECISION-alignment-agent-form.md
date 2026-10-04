# DECISION: 跨论文对齐 agent = EDC三阶段(Define新增) + 5-outcome走底座align_merge + 委托现有align_concepts内核

## 背景
现有 `concept_graph.py.align_concepts` 已有 LLM 批量对齐 (按type分组surface问LLM同概念) + `merge_concepts` (redirect+dedup). Step2 已给 Concept 加 definition/central 字段. Step1 已实现 align_merge op + 5-outcome n-ary子集关系 (insert/merge/relate/conflict/reject).

设计要求 Step4: EDC(Extract→Define→Canonicalize) + 5-outcome只aligner + Concept加definition + 同域对齐 + judge≠抽取模型 + 增量(每篇只对齐新concept).

## 张力
- 现有 align_concepts 的 merge 直接调 merge_concepts (绕底座, 无5-outcome区分 merge/relate/conflict). 语义"同组"全merge, 不区分: 完全相同→merge / 子集→merge超集 / 部分重叠→relate(不合并) / 不同→insert / 矛盾→conflict.
- 现有无 Define 阶段 (concept 无 definition).
- 现有 align 不走 KnowledgeBase transaction (不进ledger/不过5-outcome route/不可replay).

## 决策
**AlignmentAgent = 委托现有 _llm_align_batch/align_concepts 的"找同组"逻辑做Canonicalize候选发现, 但裁决走KnowledgeBase.align_merge Mutation (5-outcome route, Step1已实现). 加Define阶段(LLM生成definition填Concept.definition). 同域过滤. judge≠抽取模型. surgical复用对齐内核不重写.**

## 理由
1. 现有 _llm_align_batch 是验证过的"按type批问LLM同概念"逻辑 (跨域实证ML/bio/fluid/molecular用过). 重写风险大. 委托它做"候选发现"(哪些surface可能是同概念).
2. 5-outcome 走底座是Step1 kernel存在的意义——aligner改A-box必须进ledger/过5-outcome route/可replay. 现有merge_concepts直接改绕kernel是历史遗留.
3. Define阶段新加: LLM给每个concept生成一句话definition (填Concept.definition字段, Step2已加). 这是对齐的语义依据 (canonicalize时LLM看definition判, 不只看surface). EDC的D=Define.
4. 同域: align只在同domain内 (跨域concept不合并, DecentMem防混域). 过滤 self.concepts by domain (Concept.source_papers带domain? — 实际Concept没domain字段, domain在Hyperedge.provenance/paper. 同域过滤改为: 只对齐来自同domain论文的concept, 用concept.source_papers→paper→domain映射. 或简化: align时传domain参数, 只对齐该domain的concept).
5. judge≠抽取模型: AlignmentAgent构造器收llm_define+llm_judge两个fn, 文档强调judge≠抽取模型. (现有align_concepts只收一个llm_fn, Step4分离)
6. 增量: 每篇ingest后只对新concept对齐 (align_concepts全量, Step4改增量: 传new_concept_ids只对齐这些).

## 诚实范围
- 5-outcome的relate/conflict在concept对齐场景较少 (concept同组通常merge或不同insert). 但保留5-outcome路径 (kernel已实现), 对齐agent把"同组"判为merge, "不同"判为insert, "部分重叠非子集"判relate. 真conflict少但接口在.
- Define阶段LLM调用成本: 每concept一次. 可批量 (一次LLM给多个concept生成definition). 增量只对新concept.
- 同域过滤简化: Concept没domain字段, 用source_papers→domain映射成本高. 先诚实: align传domain参数, 对齐时只看该domain论文引入的concept (通过source_papers过滤). 真跨域综述对齐留Step6 consumer.
- conflict_mark op: concept级conflict少 (concept是实体不是claim), 主要在aligner发现"同surface但不同type/不同domain"时标conflict_mark.
