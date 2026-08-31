# SG-MHR 实现与验证 goal 模式提示词

目标：实现 SG-MHR（Schema-Guided Multi-Hop Hypergraph Retrieval）下游应用策略，验证能否区分"schema/抽取器问题"vs"下游策略太粗糙"，在 ResearchQA multi_hop 上对比 C1/C2/naive-RAG/bare-LLM。创新定位为"已有范式的精细化创新"——证明精细化（dep/con/comp 多类语义边 + 形式化操作集 + intra-DAG propagation + 实体中心检索）带来可量化能力提升。

【背景】之前 Phase 1b 的 C1（纯边相似度检索）/C2（事后拓扑扩展）失败——8 题中 C2 只 1 题>C1，净效应微弱。诊断出三个根本缺陷：(1) 只算边整体相似度不算实体节点匹配 (2) schema 富拓扑只做事后扩展不参与检索决策 (3) 没有多跳导航。

三份调研（docs/dataset-design/RESEARCH-*.md）确认 SOTA 共识：实体中心+query-aware gate 是必备件（2606.30133 实证 gate off F1 掉 3.6-7.4）；不分解成 triple（SAG/PropRAG 共识）；verbatim grounding（LineageRAG）；demand decomposition（LineageRAG）。

创新性重新审视（docs/dataset-design/INNOVATION-REASSESSMENT.md）：大框架被占（DIAL-KG 自进化 schema、Youtu-GraphRAG schema-guided 抽取+检索、HyperRAG n-ary 超图检索）。我们的差异化在：(1) n-ary 超边 pattern 间 dep/con/comp 语义边（KG 域全空）(2) 形式化操作集+LLM 仅命名不判断 (3) intra-DAG forward propagation (4) 自进化 schema + n-ary 超边 instance 交叉。必须证明"精细化带来可量化能力提升"。

【必读】先读 docs/dataset-design/DOWNSTREAM-STRATEGY-DESIGN.md（完整 SG-MHR 设计）。再读 docs/dataset-design/METHOD-TECHNICAL-INVENTORY.md（我们方法的完整技术特征）。再读三份调研 docs/dataset-design/RESEARCH-schema-guided-retrieval.md / RESEARCH-multihop-graph-retrieval.md / RESEARCH-hypergraph-downstream.md（SOTA 机制来源）。不靠对话记忆。

【设计核心】SG-MHR = 5 个 phase：
1. Demand Decomposition：LLM 拆 query 为 N 个 evidence demands，每个带 pattern_hint/role_hint/entity_hints
2. Schema-Guided Routing：demand 匹配 schema pattern → 沿 dep/con/comp 边导航到关联 pattern（schema 参与路由决策，不是事后扩展）
3. Entity-Centric Instance Retrieval：实体作 join key + pattern 过滤 + query-aware gate（每步 cos(候选实体, demand)）
4. Lineage Grounding：verbatim evidence_span 按 demand 组织 lineage + provenance tracking
5. LLM Generation：用 verbatim span 作 citation

【已有做法是起点不是终点】C1/C2 的实现（边相似度+事后扩展）是对照 baseline。SG-MHR 是新设计，实现时可调整——如果某 phase 实测无效，勇于简化或重新设计。改核心设计前 git commit+写 DECISION。

【★★ 最高纪律1：测评必须用我们真实的方法】
凡涉及抽取，必须调用 extract_hypergraph（走 schema prompt + validate + n-ary + evolution loop + split/merge/retire/rename + infer_pattern_dependencies/constraints/compositions + detect_constraint_violations）。绝对不允许用裸 call_llm 冒充"我们的方法"。SG-MHR 的 retrieval 是在 extract_hypergraph 产出的超图上做检索，不是裸 LLM 抽三元组。对比实验时"我们方法"臂必须是 extract_hypergraph + SG-MHR retrieval，裸 LLM 只能作为对比 baseline 标清楚。之前反复犯过（SciER/SPHERE 都用过裸 LLM），浪费大量时间。每次写测试脚本前先自问：这个脚本调的是 extract_hypergraph 还是 call_llm？如果是 call_llm，立即改。

【★★ 最高纪律2：看实际内容，不只看指标】
每轮测试后必须做：
1. 实际审视检索到的超边——每条超边连了什么实体、角色对不对、evidence 是否 verbatim、是否含 expected reference 的内容
2. 按 demand 分析——哪个 demand 满足/未满足，未满足的 demand 是因为 schema 导航错（到错 pattern）还是实体匹配漏（关键实体没连上）还是 gate 误杀（语义相关但 gate 拒了）
3. 原文对照：在原文找对应句子，确认检索的 evidence_span 和 expected_references 是否匹配
4. schema 审视：dep/con/comp 边导航到的 pattern 有没有道理，role_slots 是否合理
不满足于 section_coverage=K 这种数字——要看每条检索到的超边是不是真有意义，未满足的 demand 卡在哪。

【测试harness复用】
- 已缓存 3 篇论文的超图：.research_tmp/researchqa/hg_cache/（W3086667591/W4251560328/W4385245566）
- build_hypergraph 在 .research_tmp/researchqa_phase1.py（含 schema_topo/pat_edges/edges/raw_text 持久化）
- bge-m3 在 .research_tmp/paper_scion/models/bge-m3（用 sentence-transformers 加载）
- ResearchQA 数据：.research_tmp/researchqa/data/eval_dataset.jsonl（6211 QA，multi_hop 992 题）
- C1/C2 旧结果：.research_tmp/hg_out/phase1b.json（8 题，作对照）
- ResearchQA SOTA baseline 表：docs/dataset-design/RESEARCHQA-DOWNSTREAM-PLAN.md（8 模型 Table 1，section_coverage 0.636-0.955 区分度最大）

【实现优先级】
P0 先实现验证（核心改进）：
1. Phase 3: Entity-Centric Instance Retrieval——实体 join key + pattern 过滤 + query-aware gate
   - 这是 C1→SG-MHR 最核心的升级：从边相似度到实体中心+schema过滤
   - SAG(2608.12129) 验证：n-ary 超边 + 共享实体作 join key 是 SOTA 必备件
   - 2606.30133 验证：query-aware gate 是性能+延迟双收益必备件（gate off F1 掉 3.6-7.4）
2. Phase 1: Demand Decomposition——LLM 拆 query 为 demands
3. 在已有 3 篇缓存上对比 C1/C2/SG-MHR（P0 只做 Phase 1+3 的简化版即可先验证）

P1 第二步：
4. Phase 2: Schema-Guided Routing——dep/con/comp 边导航
5. Phase 4: Lineage Grounding——按 demand 组织 + verbatim grounding
6. 更完整对比

P2 后续：
7. Phase 5: Context Assembly 策略优化
8. Deficit-oriented re-routing（未满足 demand 回 Phase 2 扩展）
9. 扩展到更多论文

【诊断价值（核心目的）】
用 SG-MHR 后能区分问题来源：
- section_coverage 显著提升 → 之前是策略太粗糙（schema/抽取器没问题）
- 仍不提升 → schema/抽取器问题（dep/con/comp 边不准/实体识别不准/evidence 不 verbatim）
- 部分 demand 满足部分不满足 → 定位是哪个 pattern 的检索有问题

【成功标准（不只看数字）】
- SG-MHR section_coverage > C1/C2（证明策略升级有效）
- 或 SG-MHR 仍低但能定位是哪个环节（schema 导航/实体匹配/gate）
- 抽样 5-10 条检索到的超边人工确认：含 expected evidence、实体匹配准、gate 没误杀
- 按 demand 分析：未满足的 demand 能定位卡在哪
- 在 ResearchQA multi_hop 上对比 naive RAG / bare LLM

【纪律】
1. 每轮先实现→测试→审视内容（三者都要）
2. 不虚报：代码没跑通不说实现，报告前跑核实
3. 每轮记 docs/dataset-design/SG-MHR-LOG.md（试了什么+结果+实际内容审视+下一步+为什么错）
4. 改大方向前 git commit+写 DECISION
5. LLM 只用 deepseek；embedding 用 bge-m3（sentence-transformers）；临时文件 .research_tmp/
6. 勇于换：某 phase 实测无效就简化或重新设计，别死磕
7. build_hypergraph 加超时容错（threading + 240s timeout），跳过 deepseek 卡死的论文
8. 不分解 triple——保留 n-ary 超边整体（SAG/PropRAG 共识）

【创新定位（诚实）】
大框架被占（DIAL-KG 自进化 schema、Youtu-GraphRAG schema-guided 抽取+检索、HyperRAG n-ary 超图检索）。我们的差异化在精细化：(1) dep/con/comp 多类语义边 (2) 形式化操作集+LLM 仅命名 (3) intra-DAG propagation (4) 自进化 schema + n-ary 超边交叉。必须证明"精细化带来可量化能力提升"。如果效果不佳，记录诚实结论，写下一篇论文时再考虑新范式（如 schema 层做成真 n-ary meta-hyperedge、或往 meta-level schema reasoning 方向走）。

【最终目标（SG-MHR 验证成功后）】
在 ResearchQA multi_hop 上完整评测：
1. 对比 SG-MHR vs C1/C2 vs naive RAG vs bare LLM
2. 看 section_coverage / citation_accuracy / citation_precision
3. 顶会级 baseline：ResearchQA 原 paper 8 模型作外部参照（不直接比，因为我们 deepseek-only）
4. 如果 SG-MHR 有优势，写论文；如果无优势但能诊断问题，继续优化 schema/抽取器
5. 两个 ResearchQA：2607.11074（citation-grounded 6211QA 10域，主）/2509.00496（survey-distilled 21K 75域）
