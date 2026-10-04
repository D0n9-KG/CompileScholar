# 完整设计文档（仔细设计 + 自检）

## 0. 目标 & 哲学
- 自演化 schema + n-ary 超图 + 跨论文 + 可溯源，颗粒流域验证
- **演化从 A-box 涌现（时序+引用+内容），不专门判**
- LLM 表述不一致是唯一不可控，评测用 LLM 评判吸收（取消死规则）

## 1. 架构分层（A）
- **T-box（schema/MetaHypergraph）**：类型 + 关系模式(pattern) + 约束。自演化=长新关系类型（高门控）。不放具体实体。
- **A-box（跨论文概念超图 ConceptGraph）**：具体概念实体 + 具体关系（富拓扑+演化）。跨篇对齐 + 涌现 + provenance。
- **两层交互**：A-box 实体引用 T-box 类型，关系引用 T-box pattern。A-box 高频出现 schema 没有的关系 → 触发 T-box add_pattern（复现门控，不乱加）。

## 2. T-box schema 自演化（B）
- **B1 add_pattern 门控（三层防爆炸）**：
  - 源头 prompt：律的数学形式（乘积/三乘积/线性/表格/二次）归 constitutive_law + qualifier 记数学形式，**不拆新 pattern_type**
  - 兜底 LLM：LLM 提议新 pattern_type 时，LLM 判"真新关系类型 vs 已有类型的实例变体"，实例变体归已有
  - 复现门控：新关系类型跨篇复现频次≥3 才固化进 schema，单篇只入候选池
- **B2 split 门控**：split 要 LLM 确认是真子类（不是实例变体）+ 跨篇复现，防过细分
- **B3 merge/retire**：保留（现有有效）。retire 软删（deprecated flag，可回滚）
- **B4 rename**：规范化命名

## 3. A-box 跨论文概念超图（C，主体）
- **C1 Concept 实体**：concept_id, type, surface_variants[{surface,paper_id,evidence,year,section}], source_papers, year_range, deprecated
- **C2 跨篇对齐**：累积时 LLM 批量判同异（抽取 surface → 已有 concept or 新建），~16次/10篇轻。同义实体 merge（实体级 merge，不只 pattern 级）
- **C3 PARAMETER/NUMERIC**：符号/surface 判（μ/I/d 直接符号匹配），符号歧义（μ摩擦 vs μ(I)律）用 LLM 判
- **C4 canonical_name**：不强求定，评测时 LLM 判同异。累积复现可 LLM 提规范名（可选）
- **C5 provenance**：实体带 surface_variants（每篇 surface+evidence+year+section），边带 provenance[{paper_id,evidence,year}] + emergence_signals。可溯源链：schema 边→provenance 论文→evidence 原句→section

## 4. 演化涌现（D）
- **D1 涌现读取算法（确定性，读结构信号）**：
  - 时序（A year_range 晚于 B）+ 引用（A 源论文引用 B 源论文，paper_citations）+ 内容（A 的 evidence 含 B 的 surface/alias + however/fail/limit/extend 词）
  - 组合 → 候选演化边（A→B, 方向, 类型候选, 信号强度 confidence）
  - 规则从 gold 反推：cite+extend→extends, cite+improve→improves, cite+compare→compares
- **D2 语义补/定类型（LLM 轻）**：涌现候选 → LLM 定 extends/improves/compares 类型 + 补弱信号（无强信号但抽取明确标演化的）
- **D3 覆盖率边界**：~75-80% 强信号涌现，弱信号 background 边语义补。诚实边界
- 证据强度 confidence：emergence_signals 各权重算

## 5. 评测（E）
- **E1 LLM 评判接口**：gold 边 vs 抽出边，LLM 判 method概念≈/phenomenon≈/kind≈/极性一致 → hit/miss/polarity_error。多 seed 聚合
- **E2 7维走 ConceptGraph + LLM评判**（取代子串）
- **E3 scope fair 分母**：排除综述性边 + 源论文不在输入的边
- **E4 law_constraints**：taxonomy 错位，LLM 评判判"律+方法"语义对齐（接受 law→Param vs gold law→Method 的语义映射）

## 6. 主方法管线（F）
- **F1 删 corpus_driver 影子**，agent.process_paper_hypergraph 成真管线
- **F2 arm 逻辑**（evolve/propagate_intra_dag/arm）接进 process_paper_hypergraph
- **F3 lift 功能吸收**：cluster_methods_by_llm → C2 跨篇对齐；judge_relation → D1/D2 涌现+语义补。lift 模块删

## 7. 清理（G）
- G1 删死代码（align_embeddings/hg_retrieval/旧lift()/SPHERE/silhouette）
- G2 删旧原子路径（SchemaManager+extractor+chained_extractor+gap_discovery，run_test/run_cross_domain 改走新路径后）
- G3 删重复脚本（.research_tmp/eval_step7_* 留 experiments/）

## 8. 下游 agent（H，前提 F-G）
- H1 在 ConceptGraph 上推理（方法演化问答/跨方法比较/n-ary推理）

## 9. 数据来源（补漏）
- 论文年份：DOI→年份（corpus 2097/2129 可定年，100%）
- paper_citations：综述引用图（lift step3 已用 citation prior）+ 源论文 references

## 10. schema 演化可逆性（补漏）
- retire 软删（deprecated flag），split/merge 记变更日志可回滚，version 单调增

## 11. 跨论文冲突（补漏）
- 两篇对同一关系说法矛盾 → 记两条边 + confidence + provenance，不强行合并，下游推理看冲突

## 12. 多 seed/非确定性（补漏）
- 抽取+对齐+评判多 seed，聚合（多数投票/均值+方差）。控 deepseek 非确定

## 自检（矛盾/漏点检查）
- [x] A-box 对齐成本轻（16次/10篇）✓
- [x] 涌现+语义补不滑回 lift 专门判：语义补只对涌现候选+明确信号补，不从零判有无 ✓（守住这条线）
- [x] T-box add_pattern 三层门控防爆炸 ✓
- [x] 两层交互有复现门控不乱加 ✓
- [x] 可溯源链完整（实体+边 provenance）✓
- [x] 多 seed 聚合 ✓
- [x] 概念实体 merge（C2 覆盖）✓
- [x] 演化方向（D1 时序定）✓
- [x] canonical_name 不强求（C4）✓
- [x] section provenance 跨篇（C5）✓
- [x] schema 可逆（retire 软删）✓
- [x] 跨论文冲突记两条边（11）✓
- 残留不确定：LLM 对齐/评判可靠性需多 seed 验证；涌现覆盖率 75-80% 是估算，跑起来实测；schema add_pattern 门控阈值（复现≥3）要跑调

## 实施顺序（P0-P5，每步验证再下一个）
- **P0**：F1+F2 删影子 + arm 接进 agent → smoke 验证 agent 真跑消融臂
- **P1**：C A-box ConceptGraph 数据结构 + 跨篇对齐 → smoke 验证概念累积
- **P2**：D 涌现读取 + 语义补 → 验证演化候选产出
- **P3**：B T-box add_pattern 门控 → 验证 schema 不爆炸
- **P4**：E LLM 评测接口 → 验证 7 维评测
- **P5**：G 清理删旧 + H 下游 agent

过程随机应变，发现问题灵活解决。
