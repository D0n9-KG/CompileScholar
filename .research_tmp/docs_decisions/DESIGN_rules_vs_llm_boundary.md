# 设计：规则 vs LLM 边界（复杂问题不用规则）

状态：设计阶段，待审。

## 原则
- 复杂语义判别（是不是真方法、同义对齐、变体判别）→ LLM/语义，不用规则
- 纯结构/确定性判别（超边分类、frozenset 去重、case 规范化）→ 规则 OK
- 规则只做 LLM 不能更准或 LLM 不值得调的确定性变换

## 规则点逐个去留

### A. 保留规则（确定性结构，规则可靠）

**A1. infer_rich_topology_direct kind 分类**（hypergraph_evolution.py:1722）
- 判：超边连哪些节点类型 → kind（method_parameter/captures/nary/...）
- 保留：纯结构读（has_method+has_param），确定性，规则可靠
- 演化 kind（extends/improves）已 LLM 抽（extract），infer_rich 只识别

**A2. ConceptGraph frozenset 去重 + 自环去除**（concept_graph add_hyperedge/merge）
- 判：同 node_ids+kind 去重、src==tgt 去自环
- 保留：纯集合运算，确定性

**A3. _EQREF_RE**（方程引用 "Eq. (1)"）
- 判：surface 是不是公式引用
- 保留：极简单正则，确定性高（"Eq.X" 模式跨域通用）
- 边界：若 LLM 抽取时不产这种（该是 PARAMETER？），源头解决更好。暂留。

**A4. _LONG_TITLE_RE**（长 surface 像句子/标题）
- 判：surface 是不是论文标题/整句
- 保留：极简单正则（"of the/using/shows"），但边界模糊
- **倾向移除**：判标题该是 LLM（读 surface 语义判是不是方法描述），规则易误杀（"gradient expansion of the yield parameter" 被误杀过）。改 LLM 或强 prompt 不产。

**A5. case 规范化**（pattern_id.lower()、_singularize_token）
- 保留：纯字符串规范

**A6. _norm_surface 后缀剥离**（dedup）
- 判：合并同方法不同 surface（剥 model/theory/approach 后缀）
- **倾向移除**：同义合并该走 align_concepts（LLM），不该规则剥后缀（剥错/漏剥）。align_concepts 已 LLM 批量判同异，dedup 规则冗余且不准。

### B. LLM 化（复杂语义，规则治标）

**B1. consolidate_instance cleanup regexes**（_TOOL_RE/_GEO_RE/_EFFECT_RE/_GENERIC_RE）
- 判：METHOD 节点是不是垃圾（工具/几何/效应/泛词）
- **改 LLM**：读 surface+evidence+labels 判是不是真方法
- 但**根本是源头**：extract prompt 该让 LLM 准确标 METHOD（不产垃圾），事后 cleanup 是补漏。方向：强 prompt（METHOD 定义严）+ LLM 验证（不确定的标候选），而非事后规则清
- 过渡：domain-gate 已做（非 granular 不跑），但 granular 域内仍规则——改 LLM 判或源头强

**B2. _SYMBOL_RE 参数符号表**
- 判：PARAMETER 跨篇对齐（μ/I/d 符号合并）
- **改语义对齐**：PARAMETER 走 align_concepts（LLM 批量判同异，扩展到 PARAMETER/NUMERIC）。或 schema 声明符号（数据驱动）
- domain-gate 是过渡，根本是语义对齐
- 风险最高（换域错合并）

**B3. UNIVERSAL_PHYSICS_SYMBOLS 白名单**（hypergraph_schema.py:851）
- 判：哪些符号算"通用"（不要求定义）
- **改 LLM/移除**：物理域白名单，非物理域误压违规。该移除或 domain-gate
- 违规检测该 schema 驱动（schema 声明哪些要定义），不该硬白名单

**B4. _is_instance_variant 前缀判**（hypergraph_evolution.py:253）
- 判：新 pattern 是变体吗
- **已 LLM 接管**（LLM gate + family guard），规则只是兜底
- 规则可弱化或删（LLM gate 为主）

**B5. emergence content scan 词表**（_EXTENDS_WORDS 等）
- 判：扫 evidence 找演化词
- **emergence 定位层次3**（下游推理信号），层次1 已 LLM 抽演化
- 词表是辅助信号，非主判。保留作辅助，或层次3 整体重写时处理

### C. 源头 prompt 化（治本，事后少补）

**C1. extract prompt 强 METHOD 定义**（不产垃圾）
- 严定义 METHOD（命名建模方法/律/理论，排除工具/几何/泛词/标题）
- 不确定的不标 METHOD（标候选）或 LLM 验证
- 治本：源头准，事后少 cleanup

**C2. extract prompt 强符号命名**
- PARAMETER 用规范符号（μ 不是 "friction coefficient"）
- 或抽取时 LLM 标 symbol 字段，对齐用 symbol

## 优先级（风险×治本程度）

1. **B2 _SYMBOL_RE → 语义对齐**（最高风险，换域错合并）
2. **B3 UNIVERSAL_PHYSICS_SYMBOLS 移除/domain-gate**（误压违规）
3. **B1+C1 consolidate cleanup → 强 prompt + LLM 验证**（清错影响大，治本在源头）
4. **A6 _norm_surface dedup 移除**（align_concepts LLM 已接管，规则冗余）
5. **A4 _LONG_TITLE_RE 移除**（误杀真方法，改 LLM 或源头）
6. **B4 _is_instance_variant 弱化**（LLM gate 为主，规则兜底可留可删）

## 实施方式
- 不一次全改（成本爆），按优先级
- 每个改前设计接口（LLM 替代怎么调、成本多少、源头 vs 事后）
- 改后跑 5 篇对比效果（规则版 vs LLM 版）
- 保留的规则（A1-A3,A5）不碰

## 不确定点
1. B2 参数对齐：align_concepts 扩到 PARAMETER（LLM 批量判参数同异）—— LLM 能判 "I" vs "inertial number" 同吗？还是 symbol 更可靠？要测
2. B1 源头强 prompt：能让 LLM 不产垃圾 METHOD 吗？还是总要事后清？要测
3. 成本：B1 LLM 验证每节点？B2 扩 align 到 PARAMETER 加调用？
