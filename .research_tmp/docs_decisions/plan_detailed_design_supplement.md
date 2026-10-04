# 详细精心设计补充（动手前补 1-5 项，落到底层）
# consumer(6) 等基础打牢再设计

## 1. 创新点组合技术挑战论证（顶会非平凡核心，n-ary+schema演化组合的真实难题，binary+static 如 Intern-Atlas 不需要解）

### 挑战 A：role split 后 A-box 老超边传播
场景: schema 演化 split pattern `influences`(role:source/target/cause/effect) → `influences_functional` + `influences_correlation`。已有 A-box 超边标 `influences` role=[source,target,cause,effect]。
解法:
- split 时父 pattern 标 `is_abstract=True`(不 deprecated, 作 fallback parent)
- 老超边保持 `influences`(validates against abstract parent, fallback 通过)
- 后续抽取新边用具体子 pattern
- 维护 agent 可选重分类老边(embedding+LLM 判每条属于哪个子 pattern),**不强制**(保 abstract fallback)
- 非平凡: split 不破坏老 A-box(abstract fallback); n-ary role 集合匹配比 binary 单 edge type 复杂
- ablation: abstract fallback vs 强制重分类 vs deprecated(老边失效)

### 挑战 B：n-ary 5-outcome merge 语义
场景: aligner 判新超边 [M1,P1,P2](method_parameter) 跟已有 [M1,P1,P3](method_parameter) merge。
解法(merge 判定按 param 集合子集关系):
- 完全相同 param 集 → merge(合并 provenance)
- 子集/超集 → merge 到超集(union param, 保各自 provenance)
- 部分重叠非子集 → relate(建 relation 边不合并超边, 语义可能不同)
- 完全不同 param → conflict 或独立 insert
- 非平凡: n-ary merge 判子集关系, binary 只判 u==v; merge 要保超边结构完整(union param 不破坏 arity)
- ablation: n-ary 子集 merge vs binary 简单合并的语义保真度

### 挑战 C：schema-constrained rewrite 对 n-ary 约束
演化约束规则(保证演化后 n-ary 超边仍 validate):
- add_pattern: 新 pattern role_slots ⊆ 现有 node 类型(THING root 兜底)
- split: 子 pattern role_slots = 父 role_slots 子集(不引入新 role)
- merge: 合并后 role_slots = 两者并集(保所有 role 可达)
- retire: 不能有活跃超边引用(先重分类或 retire 超边)
- 非平凡: n-ary role 集合约束比 binary 单 edge type 约束复杂
- 这些约束在底座 kernel validate 阶段校验(诚实只做这些, 不声称对标 2608.18104 全部)

## 2. 底座内核数据结构+接口

```python
class KnowledgeBase:
    tbox: MetaHypergraph       # T-box (pattern+semantic_boundary+IS-A+version)
    abox: ConceptGraph        # A-box (Concept+definition+central / n-ary ConceptHyperedge+provenance)
    skills: SkillLibrary       # 第四层
    ledger: list[Mutation]     # 审计(replay/time-travel/暂存)
    version: str
    domain_ns: dict            # global / granular / ml / molecular

    def propose(mutation) -> MutationId        # 入 queue(consumer+evolver 提案)
    def commit(batch: list[MutationId]) -> CommitResult  # builder 单消费者串行出队
        # 阶段1 validate(pass/fail 原子): schema约束+role权限+invariant+A-box referential integrity+candidate-evidence+recurring crystallize
        # 阶段2 route(5-outcome 非原子, 只 aligner): insert/merge/relate/conflict/reject
    def audit(version) -> list[Mutation]       # replay/time-travel
    def snapshot(domain) -> KBSnapshot         # consumer 读(带 version stamp)

@dataclass
class Mutation:
    op: str  # add_node/add_edge/add_pattern/add_subclass/split/merge/retire/rename/relabel/distill_skill/align_merge/add_concept_relation/conflict_mark
    target: str
    payload: dict
    proposer_role: str  # extractor/evolver/aligner/maintainer/consumer.*
    domain: str  # global/granular/ml/...
    base_version: str  # CAS(consumer 读 snapshot 用, builder-builder 不触发因串行)
    evidence: str  # verbatim span(candidate-evidence 绑定)
    rationale: str
    timestamp: str
```
四层关联: T-box pattern 引用 A-box node 类型; A-box ConceptHyperedge.pattern_type 引用 T-box pattern; Skill 引用 T-box pattern(extraction_hint 针对哪个 pattern); ledger 引用所有。
ledger 格式: JSONL, 每条 Mutation + commit 结果 + version_diff。

## 3. schema 演化"并进"传播机制(核心创新点机制具体化)
- T-box 有 version(现有 _bump)
- extractor 每次抽取前 snapshot T-box 当前 version + to_prompt(含 semantic_boundary)
- 演化 commit 后 version bump
- 后续 extractor(下个 section/paper) snapshot 新 version → to_prompt 含新 pattern
- "并进" = intra-DAG(section 间 schema 变, 现有 meta.to_prompt re-fetch)+ cross-paper(trigger 累积, 现有行为)
- vs 离线: 离线 = 抽完所有再批量演化, 抽取时 schema 锁定
- vs static(Intern-Atlas fixed-at-release): 不演化
- ablation A2 三臂: static / 并进 / 离线

## 4. 抽取 agent 输入输出格式 + 现有结构改填法

### planner 输出 plan schema
```
Plan: {domain, central_entities:[{surface,type,role_hint}], 
       secondary_entities:[...], expected_patterns:[pattern_id],
       relation_outline:[{pattern_type, expected_nodes, evidence_hint}]}
```
### executor: 按 relation_outline 逐条抽, 每条一次 LLM(结构化输出 pattern_type∈seed∪{new}/role∈allowed), 输出 Hyperedge+provenance。Mutation 必带 domain。
### verifier 输出
```
Verdict: {edge_id, verbatim_in_source:bool, type_correct:bool(re-type内化), 
          relation_exists:bool(存在性gate), fix:"keep"|"retype:X"|"reextract"|"drop"}
```

### 现有结构改填法
- **semantic_boundary**: seed 12 pattern 手工写边界描述(influences="X函数依赖Y,X随Y变;非共列/输入/测试"等); 演化新 pattern 由 evolver add_pattern 时 LLM 生成(基于 evidence+rationale)
- **provenance 迁移**: 现有 Hyperedge.qualifiers 里 cited_from/method/evidence_strength/source_paper 提到 Hyperedge.provenance 子结构; 迁移脚本遍历 instance 移这些 key; qualifiers 只留 relation_type/dependency_type/applies_in_regime/function_form/parameters/condition
- **central 标法**: planner 输出 central_entities, executor 抽到这些 surface 时 ConceptGraph.get_or_create 带 central=True
- **Skill 蒸馏算法**: skill distiller 跑同 domain+同 pattern 多条超边抽共性(surface/role/evidence 模式)→ Skill(extraction_hint); stability_score=同模式频次/总抽取数; **threshold stability>0.6 才 crystallize**; Skill 进 library, executor 下次同 domain+pattern 读 extraction_hint 注入 prompt

## 5. ablation baseline 搭法 + 指标算法(落到可执行)

### A1 n-ary vs binary(主对照 Intern-Atlas binary+static)
- binary baseline: 把我们 n-ary 超边拆成 pairwise binary(每条 n-ary → C(n,2) binary, Intern-Atlas 式); 或复现 Intern-Atlas 抽取(重, 后)
- 指标:
  - 富拓扑直读质量: n-ary 直读7类 vs binary co-occurrence猜7类的 precision/噪声率
  - 语义保真度: method+多param+phenomenon多元关系完整度(n-ary一条保 vs binary拆开丢"同law绑param")
  - co-occurrence噪声率: binary猜的边里假关联比例

### A2 schema 并进 vs static vs 离线(三臂)
- static: T-box锁seed12不演化, A-box正常写(Intern-Atlas fixed-at-release式)
- 并进: schema随抽演化(现有行为)
- 离线: 抽完批量演化, 抽取时schema锁定
- 指标(机制层, 非NMR/ERR):
  - schema紧凑度 = pattern数 + redundancy(pattern间embedding cos>0.85比例)
  - 新pattern复用率 = 被≥2篇后续复用的pattern比例
  - 演化链追溯质量 = 跟gold evolution chain对齐(Intern-Atlas式)
  - 长尾命中 = frozen漏抽/演化命中method族(P0 failure case复用)

### A3 富拓扑直读 vs co-occurrence
- co-occurrence baseline: Higher-Order 2601.04878式, 从binary边共现猜7类
- 指标: 7类precision/噪声率 + 抽取稳定性

### 统计
- ≥4 seed paired bootstrap 95%CI(复用 P0 脚本 .research_tmp/eval_*.py + P0_multiseed_results.md 的 paired bootstrap)
- judge交叉 GLM+qwen(不一致进HITL)
- qualitative+长尾对冲aggregate稀释
- 预注册承诺: 指标+判定标准+统计跑前定死, 跑完照判报, 不事后挑, 负面诚实报

## 补完判断
1-5 落到了数据结构/接口/算法/阈值/技术挑战解法级, 非 skeleton。创新点非平凡(挑战A/B/C是n-ary+schema演化组合的真实难题)有解法+ablation。底座接口签名定。并进机制具体。抽取输入输出格式+填法定。ablation baseline搭法+指标算法可执行。
**consumer(论文检索等)等基础打牢(A1-A3地基+builder群)再设计数据流。**
认可后开干, 第1步底座内核, 到checkpoint暂停。
