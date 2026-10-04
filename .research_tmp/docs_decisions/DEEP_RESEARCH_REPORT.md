# 深度调研报告：未解决缺口与真正创新点 (2026-08-14)

> 说明：web search 配额满 + WebFetch 域名被封，无法联网调研。本报告基于本地 4 个邻近工作的全文材料 (DIAL-KG/SCION/Hyper-KGGen/Intern-Atlas) + 对 ontology learning/open IE/科学 KG 领域的通识。**不编造未核验的引用**——只引用本地已读的 4 个工作 + 通识性领域知识。

## 一、领域全景（从本地材料 + 通识）

### 1. schema 演化谱系
- **DIAL-KG**: schema-free 增量，dual-track (静态属性+事件)，retrieve top-K=30 schema，{Merge/Hierarchy/Separate}，parsimony + soft deprecation + transactional。**有简约原则但无量化收敛保证**。
- **SCION**: schema induction，single-shot，contract-constrained (JSON contract)，deterministic checks (domain/range/role signature)。**不演化，避膨胀**。
- **AgentCAT**: 渐进 schema 演化，**ADD-only**（为避膨胀只 add 不 split）。**主动放弃 split 以避失控**。
- **Hyper-KGGen**: skill-driven，**static skill per scenario**（不演化，跨域换 skill）。

### 2. n-ary / 超图抽取
- **Hyper-KGGen**: n-ary + qualifier，skill 引导，HyperDocRED benchmark。**篇内，无跨论文**。
- 通识: WebNLG/DocRED 是固定 schema 的 n-ary benchmark。

### 3. 跨论文方法演化
- **Intern-Atlas**: citation-anchored extraction（每条引用判 7 类因果边），103万论文，NMR/ERR/PSC + lineage (NR/ER/CAS)。**binary 边，需引用图，自承"不抓方法涌现的结构化关系"**。
- **SciAtlas**: 43M 论文，只定性评测。

### 4. 评测
- Intern-Atlas: coverage (NMR/ERR) + lineage (NR/ER/CAS) + idea evaluation（下游）。
- Hyper-KGGen: HyperDocRED F1 (semantic+Hungarian)。
- 零标注: SegReFree/KGCQual/MINEA（通识，未读全文）。

## 二、未解决缺口（从各工作自承 limitation + 我们撞的墙）

### Gap 1: 收敛的自演化 schema 控制（无人解决）
- DIAL-KG 有 parsimony + soft deprecation 但**无量化收敛**——我们实测 30 篇 schema 膨胀到 237 pattern，后篇抽取崩塌。
- AgentCAT 用 ADD-only 避膨胀，**主动放弃 split 价值**。
- SCION single-shot 避膨胀但**不演化**。
- **缺口**: 没有 principled + 可量化收敛的 schema 演化控制。何时 split/add/merge/retire 无 principled 判据（DIAL-KG 用频率阈值，我们试 coherence 阈值，都不够）。

### Gap 2: schema 演化质量评测（无人解决）
- Intern-Atlas 评 coverage (NMR/ERR)，**不评演化质量**。
- Hyper-KGGen 评抽取 F1，**不评 schema**。
- 我们撞的墙: silhouette/AUC/LOO 全不可靠（silhouette 惩罚小簇，AUC 依赖主观启发表，LOO frozen>full）。
- **缺口**: 无可靠评测"自演化 schema 是否优于固定 schema"。内部指标都依赖主观假设，下游任务驱动评测缺失。

### Gap 3: 篇内 n-ary ↔ 跨论文方法演化的桥接（无人解决）
- Intern-Atlas 跨论文但 **binary + 需引用图**，自承"不抓方法涌现的结构化关系"。
- Hyper-KGGen 篇内 n-ary 但**无跨论文**。
- 我们撞的墙: gold 方法级跨论文演化 vs 大图篇内实体关系，**层级错配**。
- **缺口**: 无人连接篇内 n-ary 抽取到跨论文方法演化。

### Gap 4: 强语义 pattern 级拓扑（无人解决）
- 我们 dep/con/comp **弱语义**（共现推断，LOO 测出无用）。
- Intern-Atlas 有 citation-causal（强但 binary，pattern 级无拓扑）。
- **缺口**: 无 n-ary schema + 文本语义的 pattern 级拓扑。

## 三、候选创新点（诚实评估）

### A. 收敛的自演化 schema（攻 Gap 1）★★★
- **定位**: 不是"做 schema 演化"（DIAL-KG/AgentCAT 做），是"让 schema 演化可量化收敛"。
- **我们已有**: coherence-gated split（coherence 阈值），3-edge 命名（跨篇复用），检索式/全量注入对比。
- **需补**: 量化收敛指标（pattern 增长率趋零 + 跨论文命名稳定性）+ 激进 merge/retire + 收敛证明。
- **差异化**: DIAL-KG 的 parsimony 是定性原则，我们做量化收敛控制。
- **风险**: 收敛难证明（要数学/实验保证）。

### B. 下游任务驱动的 schema 评测（攻 Gap 2）★★★
- **定位**: 不是"评抽取质量"（Hyper-KGGen 做），是"评 schema 演化本身的价值"。
- **我们已有**: survey-gold (53方法/41演化边)，可建颗粒流问答下游。
- **需补**: 构造颗粒流问答数据集（从 gold 派生），用 full vs frozen 的下游任务 lift 证演化价值。
- **差异化**: Intern-Atlas 评 coverage，我们评演化质量 via 下游。
- **风险**: 下游任务设计要触到"演化"而非"抽取"。

### C. 篇内 n-ary ↔ 跨论文方法演化桥接（攻 Gap 3）★★
- **定位**: 连接两层（篇内 n-ary + 跨论文方法链接）。
- **我们已有**: n-ary 抽取 + 综述引用语料。
- **需补**: 跨论文方法链接（共享实体/方法 embedding 匹配）。
- **差异化**: Intern-Atlas 用 citation-anchored binary，我们用 n-ary + 共享实体。
- **风险**: Intern-Atlas 已接近，差异化要讲清 n-ary 的额外价值。

### D. 强语义 pattern 级拓扑（攻 Gap 4）★
- **定位**: 拓扑从共现改成文本抽（像 Intern-Atlas citation-causal 但 pattern 级）。
- **需补**: 大改 infer_pattern_dependencies（从共现→LLM 抽语义依赖）。
- **风险**: 大改 + 引入 LLM judge (A4 循环)。

## 四、推荐：A+B 组合

**"收敛的自演化 schema + 下游任务驱动评测"**

### 为什么是 A+B
1. **攻我们直接撞的两个缺口**（Gap 1 + 2），不是凭空找方向。
2. **已有部分工作**: coherence gate (A)、survey-gold (B)，不是从零开始。
3. **可辩护**: 人人做 schema 演化（DIAL-KG/AgentCAT），无人让它可量化收敛 + 用下游任务评演化质量。这是两个清晰的差异化。
4. **A 提供"方法"，B 提供"评测"**——方法+评测配套，论文完整。

### C/D 的定位
- C（桥接）可作为 A 的应用场景（n-ary 演化 schema 用于跨论文方法链接）。
- D（强拓扑）暂缓——当前拓扑弱语义，先不主卖；A 做好后可考虑强化。

### 诚实警示
- A 的"收敛"要可量化证明，否则只是定性声称（和 DIAL-KG parsimony 一样）。
- B 的下游任务要真触"演化价值"——如果 frozen 在下游也够好，A 就白做。**B 是 A 的存亡验证**。
- 所以 **B 优先于 A**——先验证"演化有价值"（B），再投入"让演化收敛"（A）。如果 B 显示 frozen 够好，整个方向要重新审视。

## 五、建议执行顺序
1. **先做 B 的下游评测**（构造颗粒流问答，测 full vs frozen 下游 lift）——验证演化是否有价值。
2. 若 B 显示演化有价值 → 做 A（收敛控制 + 量化收敛指标）。
3. 若 B 显示 frozen 够好 → 重新审视创新定位（可能转 C 或别的）。

这个顺序避免在 A 上盲目投入。
