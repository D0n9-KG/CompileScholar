# Plan: 抽取 prompt 域无关化（泛化）

## 目标
让 4 个抽取 prompt + 1 个对齐 prompt 不再绑定颗粒流域，使任意领域论文都能跑。
不破坏颗粒流质量（原则定义保留，只换例子载体）。

## 设计决策（已定）
1. **Option A 多域例子集**：每个判别点给跨域例子（颗粒流+ML+生物+化学），全标
   "illustrative, not rules"。原则定义 crisp 保留。LLM 学抽象规则不学域。
2. **evolution decision logic 保留**（improves=解决局限/extends=推广范围/compares=并行不同机制
   是领域通用逻辑）；只有举例从颗粒流换成抽象或多域。
3. **边界**：大量混领域输入 → schema 分层+按需检索是架构演进，memory 标"现在不做"。
   这次只做 prompt 域无关（不破坏混域即可）。
4. **单次 EXTRACT_HG_PROMPT 也泛化**：它是文档化默认+fallback，留它颗粒化=半成品埋雷。

## 不动（正确域门控或待删）
- hypergraph_schema.py 物理 seed（通用 seed `seed_meta_hypergraph_general()` 已存在，物理 seed
  对物理域是正确特化，非过拟合）
- hypergraph_evolution.py `_TOOL_RE/_GEO_RE/_EFFECT_RE`（已 is_granular 域门控）
- hypergraph_lifter.py lift prompt（P0后半段待删影子，不浪费工）
- chained_extractor.py（P4 legacy 待删）
- hg_qa_generator.py Bagnold 注释（cosmetic）

## 要改的文件 + 具体编辑

### A. `src/granular_agent/hypergraph_extractor.py`

#### A1. `_STEP1_PROMPT` (L512-537)
- 删 "GDR MiDi" research-group 例子
- 例子块改成跨域 + 标 "Examples (illustrative, not rules):"
  - METHOD: 颗粒 "μ(I) rheology" / ML "Transformer architecture" / 生物 "Maximum likelihood phylogenetics"
  - PARAMETER: 颗粒 "inertial number I" / ML "learning rate" / 化学 "rate constant k"
  - PHENOMENON: 颗粒 "nonlocal creep" / ML "overfitting" / 生物 "allosteric regulation"
- "flow regimes, materials" → 通用 "regimes/conditions, materials/substances, metrics, tasks"

#### A2. `_STEP2_PROMPT` (L540-566) — 最重
- METHOD 判别定义保留（"names a mathematical/scientific MODEL of behavior"），例子跨域
- "NOT experiment tools (MRI, PIV, simulations)" → 通用 "NOT measurement tools/procedures
  (microscopes, PIV, simulations, assays, benchmarks)" 跨域举例
- "NOT geometries (plane shear, heap flow, rotating drum)" → 通用 "NOT experimental
  setups/geometries/configurations (plane shear, rotating drum, petri dish, reactor)"
- μ(I) vs μ 判别：保留原则（law-name vs bare-symbol），例子加通用说明 +
  ML 类比（"BERT"=METHOD名 vs "B"=某参数）
- "glass beads, sand" → "granular material, chemical reagent, cell line"
- regimes "quasi-static/dense/inertial" → 通用 "regime/condition (e.g. dense/quasi-static
  for physics, training/inference for ML, aerobic/anaerobic for bio)"

#### A3. `_STEP3_PROMPT` (L569-620)
- 删 "2-m-long plane" 例子 → 换通用 "an experimental apparatus description is NOT composition"
- 其余 pattern_type/role 约束已通用，保留

#### A4. `EXTRACT_HG_PROMPT` 单次 (L49-278) — 最重
- decision paths (L96-111)：保留抽象逻辑，颗粒举例改抽象或多域
  - path1 improves: "A handles a regime where B fails" (保留原则)，去 μ(I)/nonlocal 具体名
  - path2 extends: "A adds a term/parameter or generalizes B to new regime" 去 I-gradient/μ(I)
  - path3 compares: "A,B parallel different-mechanism same-phenomenon" 去 Gray/Tripathi
- METHOD 段 (L180-211)：定义保留，长排除清单（PIV/MRI/X-ray/numerical simulation/discrete
  particle simulations + plane shear/annular shear/heap flow/rotating drum/silo/chute/hopper）
  改成通用类别 + 跨域举例，标 "illustrative"
- PARAMETER 例子 (L212) μ,I,d,P,τ,g,ρ_s → 加 ML/化学 (learning rate, k, K_d)
- PHENOMENON 例子 (L216) → 加跨域 (overfitting, allosteric regulation)
- REGIME/MATERIAL (L219-220) → 通用化
- METHOD NAMING 例子 (L206-207) → 跨域 (μ(I) rheology / Transformer / PCR)
- NGF/μ(I) method-parameter n-ary 例子 (L114-123) → 通用化 "a named model M uses params
  p1, p2 → ONE arity-3 edge [M(METHOD) — uses → p1, p2(PARAM)]"，跨域举例
- 删 "GDR MiDi → MATERIAL/PROPERTY" 具体名，留通用 "research group/consortium name →
  not a METHOD"

### B. `src/granular_agent/concept_graph.py` `_ALIGN_PROMPT` (L411-419)
- 颗粒例子 "non-local rheology" vs "I-gradient model" → 通用 "e.g. a method called
  by two names across papers = same concept"，跨域举例可选
- type_map Chinese labels (建模方法/物理现象/物理参数/数值量) → 通用
  (方法/现象/参数/数值) 去"物理"限定

## 验证（不做完不算数）
1. 跑 5 篇 ARFM2024 V4-Flash 多步 (HG_MULTISTEP=1)，对比 gold：
   - METHOD 标签质量不退化（之前 55 全真方法，PIV/DEM 无误标）
   - 富拓扑边数不塌（law_parameter/method_parameter/nary 仍正）
   - 若退化 → 说明例子换了伤判别，回退该点
2. 跑 2 篇 ML 论文（researchqa 语料有 ML）做域无关 smoke：
   - 能抽 METHOD（Transformer/attention）、PARAMETER（learning rate）、PHENOMENON
   - 无颗粒术语泄漏到 ML 结果
3. 内容审计（看边+对比原文，不看数字）

## 风险 + 回退
- 风险：换例子伤颗粒流判别 → 用 gold 对比兜住，哪点退化回退哪点
- 不改原则定义只换例子载体 → 退化概率低
- 单次 prompt 太长改全风险高 → 多步路径是主路径，单次改完跑一次 smoke 即可

## commit
- 一个 commit：`feat(extract): generalize prompts to domain-agnostic (multi-domain examples, marked illustrative)`
- 不动 schema seed / evolution regex / lifter / chained
