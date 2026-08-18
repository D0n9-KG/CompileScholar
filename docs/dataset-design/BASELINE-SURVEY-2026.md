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
