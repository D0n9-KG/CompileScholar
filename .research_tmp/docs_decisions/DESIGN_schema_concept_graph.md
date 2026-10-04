# 设计文档：schema 作为跨论文概念超图（重设计）

状态：设计阶段，未动手。待审。

## 1. 目标重述

顶会级自演化 schema + n-ary 超图 + 可溯源，颗粒流域验证。
核心卖点：schema 随论文累积自演化（不死板）+ 富拓扑 + 跨论文 + 可溯源。
演化哲学：**演化关系从跨论文结构（时序+引用+内容）自然涌现，不专门判**。

## 2. 核心架构：T-box（schema）+ A-box（跨论文概念超图）两层

**不分层是错的**——把具体方法实体塞 schema 就不叫 schema 了（混了 T-box/A-box）。
正确分两层：

### schema（T-box 本体层）= 抽象定义
- 类型（METHOD/PARAMETER/PHENOMENON/REGIME/MATERIAL/NUMERIC）
- 关系模式（constitutive_law/measures/claim_relation/extends/improves... 的 pattern 定义 + role_slots + qualifiers）
- 约束
- **自演化** = schema 本身长新关系类型/约束（新论文引入 schema 没有的关系 → schema 加）
- **不放具体方法实体**（I-gradient 这种是实例层）

### 跨论文概念超图（A-box 实例层）= 具体知识
- 具体概念实体（I-gradient model / nonlocal creep 这种具体方法/现象）
- 具体关系（I-gradient uses I / NGF extends μ(I) 这种具体边，含富拓扑+演化）
- 跨篇对齐（LLM 判同异，同方法跨篇合一）
- 演化关系从这里**涌现**（时序+引用+内容信号）
- 带 provenance（可溯源）

### 两层关系
- A-box 实体引用 T-box 类型，关系引用 T-box 关系模式
- T-box 演化（长新关系类型）→ A-box 能抽/涌现新关系
- 演化关系在 A-box 涌现，不在 schema；schema 只定义关系类型（extends/improves 这些类型是 T-box 定义的，具体哪两方法 extends 是 A-box 涌现的）

## 3. 数据结构

### T-box（schema/MetaHypergraph，保留+完善现有 meta_hg）
- meta_nodes：类型（THING/METHOD/PARAMETER/...）+ 子类边
- patterns：关系模式（pattern_id + role_slots + allowed_qualifiers + family）——已有
- **自演化操作**：split/merge/retire/rename 操作 pattern（已有，泛化完善）+ 长新关系类型（新机制，见第11节不确定点7）

### A-box（跨论文概念超图，新建）
概念实体（Concept）：
- concept_id（实例层稳定 id）
- type（引用 T-box 类型）
- surface_variants：[{surface, paper_id, evidence_span, year}]（跨篇表述 + provenance）
- source_papers：[paper_id]（哪些篇贡献）
- year_range：(min, max)（时序）

关系边（ConceptRelation）：
- src_concept, tgt_concept
- kind（引用 T-box 关系模式：富拓扑 method_parameter/captures/...；演化 extends/improves/compares/...）
- emergence_signals：{temporal, citation, content, qualifier}（涌现依据，可溯源）
- provenance：[{paper_id, evidence_span, year}]
- confidence（涌现信号强度）

## 4. 数据流（每篇处理）

```
论文(带年份+引用列表) → map_structure(分节)
  → extract_hypergraph(LLM抽n-ary超边:节点surface+labels+evidence)
  → consolidate_instance(确定性清标签+去重)
  → 概念对齐(累积时LLM:抽出的surface→schema已有concept or 新建,批量)
  → 累积进schema(实体+关系+provenance,带年份)
  → 演化涌现读取(确定性:时序+引用+内容信号→演化候选边)
  → 语义补/定类型(LLM轻:只对涌现候选定extends/improves/compares类型 + 补弱信号)
```

哪步 LLM / 哪步确定性 / 哪步结构读：
- 抽取：LLM（必须）
- 清标签：确定性（consolidate）
- 概念对齐：LLM（批量，~16次/10篇，轻）
- 累积：确定性
- 涌现读取：确定性（读结构信号）
- 语义补/定类型：LLM（轻，只对涌现候选）

## 5. 涌现规则（从 gold 46 条反推，不拍脑袋）

gold 信号频次：cite 24/46, extends_sig 16/46, improves 5/46。

涌现候选规则（方向性，刻画发展史）：
- 时序(A晚于B) + 引用(A引B) + extends_sig → extends 候选
- 时序 + 引用 + improves(however/fail/limit) → improves 候选
- 时序 + 引用 + compares(whereas/contrast) → compares 候选
- 时序定方向（A晚于B → A建立在B上）

**诚实边界**：涌现覆盖 ~75-80% gold evolution（强信号边）。~10 条 background 弱信号边纯结构涌现产不出，靠语义补。**不是纯涌现，是涌现+语义补混合。**

## 6. 语义补 / 定类型（轻 LLM，不是从零判）

- 只对涌现出的候选边调用：判 extends/improves/compares 哪个类型
- 补弱信号：对没有强涌现信号但 LLM 抽取时明确标了演化关系（claim_relation 含 extends/improves 词）的，语义补
- **和 lift 区别**：lift 从零判"A extends B 吗"；我们涌现先给有据候选，语义只定类型+补，不判有无

## 7. LLM 评测评判接口（取代子串死规则）

判"抽出关系 vs gold 一致"：
- gold 边 (M5, F1, captures) + evidence
- 抽出边 (concept, concept, kind) + provenance/evidence
- LLM 判：method概念≈gold M5？phenomenon≈F1？kind≈captures？极性一致？
- 输出 hit/miss/polarity_error
- 多 seed 控方差

**诚实**：LLM 评判吸收表述不一致（解决 segregation reversal→F5 假阳），但吸收不了 scope 错位（源论文没综述作者总结的边）。所以 fair 分母还是排除综述性边 + 源论文不在输入的边。

## 8. provenance / 可溯源

- 实体：source_papers + 每篇 surface + evidence_span + year → 任一概念可追到源论文原句
- 边：emergence_signals + provenance → 任一关系可追到涌现依据 + 源 evidence
- 跨论文溯源链：schema 边 → provenance 论文 → evidence 原句 → section

## 9. 和现有代码的关系

- **保留**：extract_hypergraph（抽取n-ary）、consolidate_instance（清标签）、infer_rich_topology_direct 哲学（结构直读）、节点标签、超边 evidence、map_structure
- **升级**：meta_hg 从注册表→概念超图（加 MetaConcept 实体层 + MetaRelation，meta_nodes 从"类型"扩展到"类型+具体实体"）
- **新建**：概念对齐（累积时 LLM 批量，替代 InstanceCorpus surface 合并 + lift cluster）、涌现读取（结构信号→候选）、LLM 评测接口
- **删除**：lift 模块（功能吸收）、InstanceCorpus surface 合并（被概念对齐替代）、corpus_driver 影子、旧原子路径（SchemaManager+extractor+chained_extractor+gap_discovery）、死代码（align_embeddings/hg_retrieval/旧lift()/SPHERE/silhouette）

## 10. 清理 + 重写顺序

- **P0：升级 meta_hg 为概念超图**（MetaConcept + MetaRelation 数据结构 + 累积逻辑）—— 基础设施
- **P1：概念对齐**（累积时 LLM 批量，替代 surface+lift cluster）
- **P2：涌现读取**（时序+引用+内容信号→演化候选）+ 语义补/定类型
- **P3：LLM 评测接口**（取代子串，7维都走 schema + LLM 评判）
- **P4：删旧**（lift/InstanceCorpus surface/corpus_driver 影子/旧原子路径/死代码）
- **P5：下游 agent**（在概念超图上推理，前提 P0-P3）

## 11. 不确定 / 需审点（诚实）

1. **实体身份/对齐可靠性**：抽取 surface 噪声大（"non-local rheology"/"I-gradient"/"gradient expansion" 同一方法）。累积时 LLM 判同异可靠但可能误判。多 seed？需验证。
2. **涌现覆盖率**：~75-80%，弱信号边靠语义补——语义补会不会变相"专门判"？要盯（语义补只对涌现候选+明确信号补，不从零判有无，但有滑回 lift 的风险）。
3. **canonical_name 何时定**：实体刚建时 surface 是噪声，规范名何时确定（累积到一定信号后？还是不强制规范名，靠 LLM 评测时对齐）？倾向后者（不强求规范名，评测时 LLM 判同异）。
4. **时序对历史方法**：gold refs 有 1909/1950s 早期方法，输入 1983-2015 覆盖不到——涌现做不出历史方法演化。接受（scope 限制）还是扩输入？
5. **PARAMETER/NUMERIC 对齐**：符号 μ/I/d 可 surface/符号判，不用 LLM。但要区分符号歧义（μ 摩擦系数 vs μ(I) 律）。
6. **schema 演化操作泛化**：split/merge/retire/rename 从 pattern 级泛化到 concept 级（两同义 concept merge、过宽 concept split）——接口要设计。

## 12. 待用户定

- 这个整体架构认不认？
- P0-P5 顺序认不认？
- 第11节不确定点，哪些你先定方向？
