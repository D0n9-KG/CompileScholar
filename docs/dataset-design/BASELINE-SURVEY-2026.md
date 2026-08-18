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

## SCOPE 全链路跑通 (2026-08-14, 20句初步)

**bge-m3 下载成功** (HF_HUB_DISABLE_XET=1 + hf-mirror, pytorch_model.bin 2165MB).
Steam++ hosts 加速 github 通; hf-mirror 通但 Xet 大文件要禁用.

**修了 1 个 bug**: validate() UnboundLocalError on empty-seed schema (空 meta
时 concrete+abstract 为空, for 循环不执行, pid 未赋值). granular 有 seed
没触发; schema-free induction 触发. 修: for 前 pid=None. smoke 53/53 无回归.

**20句 SciERC induce 初步数字** (4 pattern, 0 topology edges):
| metric | P | R | F1 |
|--------|---|---|----|
| literal | 0.188 | 0.125 | 0.150 |
| fuzzy | 1.000 | 0.667 | 0.800 |
| continuous | 0.867 | 0.578 | 0.693 |
| graph | 0.967 | 0.791 | 0.871 |

- induce 出语义合理 pattern: comparison_outperforms/contrasts/comparable_to
  (对应 gold "compare", 我们细分了) + composition (对应 "part of"/"conjunction")
- literal 低 (命名/粒度不一致, 预期) → fuzzy/graph 容差异 (0.80/0.87)
- graph F1=0.87 高但 **0 topology edges** → 是 pred 小 (9 nodes) 易匹配, 非
  富拓扑优势. 需 scale induce (更多句→topology edges 出现) 看 graph F1 随
  富拓扑变化. 100句 induce 进行中.

**诚实**: 20句 sample 太小 (4 pattern vs gold 7), 单源, 不可直接比 SCION 论文
macro-avg. 但全链路跑通 (bge-m3+SCION metrics+我们 induce+ours_to_graph+
4指标) 是路线 A 主战场可执行的关键证明.

## schema 自进化工作评测范式 (2026-08-14, 用户关键追问)

用户问"schema 自进化工作怎么评自己 schema 质量"。诚实: 这是我之前调研盲区。

调研 3 个 schema-evolving 工作评测方法:

### SCION (schema induction benchmark)
1. gold schema graph + 4 F1 (Literal/Fuzzy/Continuous/Graph) — 需 gold schema
2. compactness penalty (RL reward 权重0.10, 防 proliferation) — 无 gold
3. consistency checks (domain/range + role signature) — 无 gold
4. **5.8 downstream: fixed extractor + vary schema, 看 instance-level F1**
   (released-schema 0.5633, ETA 0.6523, SCION-lite 0.6800) — 需 gold INSTANCE 非 schema
5. controllability/auditability 统计 (parse/fallback/retention logs) — 无 gold

### DIAL-KG (schema-free streaming, RQ3 Schema Quality)
1. 静态 P/R/F1 (WebNLG/Wiki-NRE) — 需 gold
2. streaming Δ-Precision / D-HP — 需 gold
3. **RQ3 Schema Quality = compactness + redundancy**:
   - compactness = #relation types (少=好, vs EDC 少15%)
   - redundancy = 近重复 relation 比例 (vs EDC 降1.6-2.8点, 用相似度算)
   — 无 gold!
4. consistency checks (fidelity/currency 防 hallucination) — 无 gold

### Hyper-KGGen (超图 n-ary)
1. n-ary P/R/F1 (HyperDocRED) — 需 gold
2. graph quality = retrieval efficiency (同budget覆盖更多key fact) — 无 gold
3. downstream RAG accuracy — 需 gold QA
4. stability-based feedback (抽取稳定性作reward) — 无 gold

## 关键: 三个无 gold-schema 路径
1. **compactness** = #relation types (无 gold)
2. **redundancy** = 近重复率 (embedding cos, 无 gold)
3. **downstream F1** = schema 反哺 fixed extractor 的 instance F1 (需 gold INSTANCE
   非 schema — SciERC/HyperDocRED test split 自带 relation 标注!)

绕开"schema 无 gold"困境: 不评 schema 对不对, 评 schema 好不好用 (反哺抽取).

## 我们实测 (100句 SciERC induce, schema-free, 无 gold 评测)
- compactness = 22 patterns (gold 7, 我们过细3倍)
- redundancy_rate = 0.186 (43/231对≥0.85cos, 18.6%近重复)
- pct_with_near_dup = 0.773 (77% pattern 有近重复邻居)
- avg_max_sim = 0.880
- 近重复例: equivalence_relation ~ resource_equivalence/superiority/improvement/
  comparison_method (语义确相近, merge/semantic-dedup 没合并)

## 诚实诊断: schema-free 下 schema 质量不好
- 22 vs gold 7 过细; 18.6% redundancy 远高于 DIAL-KG vs EDC 的 1.6-2.8点
- 说明 schema-free 下 merge/semantic-dedup gate 没起作用 (evolution probe
  提议新 pattern, gate 该拒近重复但没拒) — 真实弱点
- 但这是 schema-free+100句+单源; 8篇granular(物理seed+多论文)是33拓扑边+收敛
- 设定差异导致结论差异

## 路线重塑
三个无 gold 路径中, **downstream F1 最有说服力** (schema好不好用看反哺抽取).
SciERC test 214 docs 自带 relation gold → 可做, 无需专家.
但 redundancy 18.6% 是真问题, 要先修 (gate 为何没拒近重复) 再报.
