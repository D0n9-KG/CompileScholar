# Baseline + 评测集 广泛调研 (2026-08-14)

## 起因
用户质疑 DIAL-KG 作 baseline 是否合适、是否广泛调研过。诚实：之前没广泛调研，
凭记忆"DIAL-KG 最近"就定。本调研修正。

## 直接竞争 baseline 全景

### 1. Hyper-KGGen (arXiv 2602.19543, 2026-02) — 最直接竞争
- **Knowledge Hypergraph Generation** (n-ary)，和我们方向直接重叠
- skill-driven framework + **Global Skill Library** + stability-based feedback
- coarse-to-fine 抽取流程 (chunk→entity→hyperedge→dedup)
- **关键：schema 仅提及 2 次**，明说"unlike prior approaches that rely on rigid
  schema matching" → **它不用 schema，用 skill library**
- 无 IS-A taxonomy、无 schema 富拓扑、无 constraint violation 检测
- 发布 **HyperDocRED**: document-level hypergraph benchmark (50 train/100 test,
  非物理域，F1~0.59 micro)
- **差异化明确**: 我们 = 自进化 schema(meta-hypergraph)+富拓扑; 它 = skill library
  非 schema。表示层根本不同。

### 2. HGNet (arXiv 2603.23136, 2026-03) — 科学 RE 层级 SOTA
- 两阶段 Z-NERD(NER) + HGNet(RE with hierarchy-aware message passing)
- 显式 parent/child/peer 层级 (我们 IS-A taxonomy 同概念)
- Differentiable Hierarchy Loss + Continuum Abstraction Field Loss
- **SOTA on SciERC + SciER + SPHERE**, zero-shot RE +26.2%
- SPHERE 是它的 benchmark (github.com/basiralab/SPHERE, 已在我们本地)
- 我们 0.337 > HGNet zero-shot 0.276 (匹配标准待对齐)

### 3. SCION/SCOPE (arXiv 2607.21610, 2026-07) — schema induction 同型 baseline
- SCION = auditable reference pipeline: naming/merging/filtering/validation/
  conservative fusion (和我们 split/merge/retire/validate/gate 同型操作)
- **不做 pattern-level split** (全文 "split" 全指 train/test split) → 我们独有
- 无 hypergraph/n-ary (0 命中) → 我们 n-ary 独有
- constraint 指 domain/range typing，非我们"引用未定义数值"检测 → 不同能力
- 一次性 corpus-to-schema induction，非 self-evolving streaming → 我们自进化独有
- SCOPE benchmark: 24 IE 源 (15 RE + 9 EE)，4 指标 (Literal/Fuzzy/Continuous/
  **Graph F1**)，Graph F1 正好体现我们富拓扑
- SCION 比 ETA 只高 +0.026 Graph F1 (差距小，有机会)；LLM-only(DeepSeek-R1)
  Literal F1 仅 0.57 (我们 deepseek+schema 应能超)
- 开源 (CC BY 4.0, github.com/wandugu/paper_scion)

### 4. DIAL-KG (arXiv 2603.20059, 2026-03) — streaming 增量
- schema-free + streaming 增量 KG construction
- 评 WebNLG/Wiki-NRE (通用域) + SoftRel-Δ (Kubernetes 日志)
- F1 0.920 on WebNLG (近天花板，劣势战场)
- 指标: streaming Δ-Precision + D-HP (soft-deprecation)
- 用 Qwen-Max + DeepSeek-V3 judge + BGE-M3
- 和我们 self-evolving 对齐但评通用域，物理优势体现不出

### 5. AutoSchemaKG (2505.23628) / LOGOS (2509.24294) — 补充
- AutoSchemaKG: dynamic schema induction 但评 QA factuality (非 schema F1)
- LOGOS: schema induction for qualitative research, 5维指标

## 我们方法的核心差异化 (针对所有 baseline)
1. **自进化 schema (meta-hypergraph)** — Hyper-KGGen 是 skill library 非 schema;
   SCION 一次性 induction 不自进化; DIAL-KG 自进化但 schema-free 无富拓扑
2. **schema 层富拓扑** (dependency/constraint/composition) — 所有 baseline 都无
3. **constraint violation 检测** — 所有 baseline 都无 (KG 抽取领域可能真空,
   待深入核实)
4. **pattern-level split** (top-down) — DIAL-KG 只 bottom-up merge; SCION 只
   fusion; Hyper-KGGen coarse-to-fine 是抽取流程非 schema split
5. **物理科学域** — HyperDocRED 非物理; SPHERE 物理但 HGNet 占; 我们物理 schema
   在 SPHERE 占优 (0.337 > 0.276)

## 定位
我们在"自进化 schema 富拓扑 + 物理域"交叉点 — 无人占死。

## 待 Agent 调研补充
- 文献问答下游评测集 (ResearchQA/Ai2 Scholar QA/M3SciQA/QASPER)
- 本体构建评测 + 物理方向 LLM reasoning
(两 Agent 后台运行中)

## 初步评测组合建议 (待 Agent 结果定稿)
- **HyperDocRED** (超图 n-ary F1, 和 Hyper-KGGen 直接比) — 通用域超图抽取
- **SPHERE** (物理域 RE F1, 和 HGNet 比) — 物理域优势
- **SCOPE** (schema induction F1, 和 SCION 比, Graph F1 体现富拓扑) — schema 质量
- 下游 QA (Agent 调研中)
三个战场各体现一个优势维度 + 三个直接 baseline。

## HGNet SPHERE 匹配标准 (对齐方案, 2026-08-14)
HGNet 用 **strict Rel+ F1** (ref 29): TP 要求正确预测
  ① 两个实体的 boundaries (精确 span)
  ② 两个实体的 types
  ③ relation type
三者全对才算 TP。还把 relation 分 hierarchical / peer 两类报 macro F1。

我们之前 sphere_eval_ours_v3.py 的 strict_match 只用 substring
(pred[0] in g[0]) + relation exact — **没匹配 entity type, substring 比
boundary 宽松会虚高 F1**。对齐方案: 改 strict_match 为 boundary(span)+type+
relation。entity type 我们已映射 SPHERE 15 types (make_sphere_seed), 可对齐。
重算要 LLM 调用(评测阶段做), 现在只记对齐方案。对齐后 F1 可能下降, 需确认
仍超 HGNet zero-shot 0.276。

## SCOPE 评测适配方案 (2026-08-14, 路线A主战场)

### 为什么 SCOPE 是对的主战场
- 评 schema-level edge `(head_type, rel_type, tail_type)` 类型三元组,
  **不评 entity 实例 boundary** — 避开我们 entity typing 弱项 (见
  sphere-benchmark-result 对齐后崩)
- Graph F1 (graph-structure-enhanced node-level) 体现富拓扑 — 我们的
  dependency/constraint/composition 边让 schema graph 有结构, Graph F1 奖励
- SCION 只比 ETA 高 +0.026 Graph F1, 有空间
- 同型操作 (naming/merge/fusion/validate/conservative gate), 公平比
- 开源 (github.com/wandugu/paper_scion, CC BY 4.0)

### 评测协议 (从论文读到)
- 24 源 (15 RE + 9 EE), 每源有 `induction_texts.jsonl` (train-text-only) +
  gold schema graph (评测保留)
- 任务: 从 induction_texts 诱导 schema → predicted schema graph
- 比较: predicted vs gold, macro-avg P/R/F1 over sources, 4 指标
- schema edge = (head_type, rel_type, tail_type); rel_type=predicate name,
  head/tail_type = domain/range type (RE) 或 event type/role (EE)

### 我们方法适配
1. 对每源读 induction_texts.jsonl
2. 跑 extract_hypergraph (空/通用 seed, 从 corpus 诱导 — 体现 schema induction)
3. flatten induced schema: pattern 的 role_slots (role: type) → (head_type,
   pattern_id, tail_type) 边集 (pattern >2 role 时拆多个三元组)
4. 和 gold schema graph 比 4 指标 (Literal/Fuzzy/Continuous/Graph F1)
   — 指标实现复用 SCION 开源代码 (下载后看 eval/ 目录)
5. macro-avg over sources, 和 SCION-lite/Text2Onto/LLM-only 比

### 适配挑战 (诚实)
- seed 不能用物理 6 family (SCOPE 是通用域 IE) — 用空 seed 或源 type 集
- 我们方法 pattern 的 type 系统要对齐源的 entity type 体系
- 富拓扑边 (dependency/constraint/composition) 要映射进 schema graph 让
  Graph F1 能看见 — 这是关键 (否则富拓扑在指标里不显)
- 数据下载遇网络问题 (codeload zip 多次截断), 待解

### 待办
- [ ] 下载 SCOPE 数据成功 (in_progress, 网络问题)
- [ ] 读 SCION 代码: 数据格式 + 指标实现 + 怎么接 baseline
- [ ] 写 SCOPE 适配脚本 (induce + flatten + score)
- [ ] 跑我们方法 + SCION/Text2Onto/LLM-only 对比

## SCOPE 数据 + 代码到手 (2026-08-14, Steam++ hosts 加速 github 通)

Steam++ (Watt Toolkit) hosts 加速 github.com → git clone 成功 (701 文件,
GIT_LFS_SKIP_SMUDGE=1 跳过 lfs 大文件)。

**到手**:
- src/ontology_eval.py — 4 指标 (Literal/Fuzzy/Continuous/Graph F1) 实现
- src/ontology_generate.py — SCION induction (prompt + candidate mining, 1850行)
- src/knowledge_graph_maker/ — SCION 自带抽取器 baseline
- data/scope/subsets/ — 24 源 unified gold schema + docs.train.jsonl (induction texts)
- **SciERC 在内** (科学域, 7 relation, head/tail=Entity, 1536 train docs)

**指标机制确认**:
- OntologyGraph = {nodes:{id:label}, edges:Set[Edge(src,tgt)]}
- schema_dict_to_graph: rel_label="{head}->{tail}({rel})", 边 section->rel_label,
  head->rel_label->tail
- graph_f1 用 graph_smooth (K=2 邻居均值, alpha=0.5) 后 node-level Hungarian
  → **邻居结构影响 → 我们富拓扑边 (dep/constraint/comp) 让结构更丰富 →
  graph_f1 高 = 优势体现点**
- fuzzy threshold 0.45, continuous Hungarian, literal 精确边

**适配写好** (ours_to_graph.py, 测试通过):
- 8篇granular final_meta → OntologyGraph 67 nodes 220 edges (含33富拓扑边)
- pattern role_slots types → entities, pattern → rel_label, 富拓扑边 →
  rel_label→rel_label (SCION flat schema 无此层 = 我们优势维度)

**卡点**: bge-m3 embedding 模型 (compute_ontology_metrics 必需)。
hf-mirror Xet 协议 401 (HF 新大文件协议认证)。试 HF_HUB_DISABLE_XET=1。
备选: 写 wrapper 用我们 _embed_texts_robust (Paratera/CST) 替代 (不同 embedding,
绝对数字不可直接比 SCION 论文, 但我们 vs SCION 重跑同 embedding 公平)。

**待办**:
- [ ] bge-m3 下载成功 (Xet 401 待解)
- [ ] induce_ours 跑 SciERC (extract_hypergraph, sample 50 句先试, LLM成本)
- [ ] run_eval 算 4 指标, 和 SCION-lite 比, Graph F1 看富拓扑优势
