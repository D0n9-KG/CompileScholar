# GLM 5.3 全面审查报告（2026-08-25）

五路审计子代理并行深审（内容逐条/稳定性+新旧对比/schema演化/顶会战略/代码），交叉印证后的最终判决。本文档是当前项目状态的唯一权威快照，取代 session-glm53-handoff-status.md 中的乐观叙事。

---

## 一、总判决（一句话）

**方向站得住（组合真空确认），但当前实现的所有数字不可进论文；核心卖点的证据被三个确定性 bug 亲手杀死；判决：抽取栈重写+内核保留简化（B+），先跑 2-3 天采纳探针决定 schema 演化卖点的最终形态。**

---

## 二、五路审计核心发现

### A. 旗舰 run 内容审计（ML_DQN_2015，逐条人工语义审）

- **verbatim 93.5% 无捏造（真），但边级语义正确率仅 ~43% 可辩护 / ~30-37% 明确错误**（极性反转×3 成体系、垃圾绑定、"0%" influences DQN、方向反了）
- **论文核心演化事实全军覆没**："DQN 是 Q-learning 变体+ER+target network"约 14 条真边全部在 dropped（复合 pattern_type 整批拒）；kept 的 12 条 evolution 边约 10 条错误（"end-to-end"形容词当 extends 对象、游戏名/数字当 compares 节点）
- **kept 387 含 16-19% 重复注水**（实际 314；387 是同文抽 5 遍的并集）
- rich_topology.json 只存计数没存边（内容丢失，但计数可从 concept_graph 100% 重建）
- evidence-not-locatable gate 误杀 14/14（全是 LaTeX 归一化/引文上标/图注打断的格式问题）
- 正面资产：ledger 事务完整性、28 个 concept merge、公式类 constitutive_law 边、无捏造 evidence、49% 真 n-ary 结构

### B. 稳定性 + 新旧对比审计

- **5 倍差距（61/53/314）不是 LLM 方差，是结构性病态**：测试脚本把全文 mock 成单 block → `_locate_chapter_text` 终点搜索 strict 正则在零换行文本上永不命中 → 5 个互相重叠的近全文切片 → **同一篇论文抽了 5 遍**
- **真 LLM 方差也不小**：同状态 s1 vs s2 边级 evidence Jaccard 仅 0.245（3/4 的边不同）→ 任何单 seed 数字不可信
- **BIO 0/0 是修复前产物**（11:29 跑，修复在 12:57）；修复自认"治崩不治漂移（4/5 chunk step3 解析 0）"；post-fix 重跑被 GBK print 崩掉未存 bundle → **BIO 在 HEAD = 未验证**；旧管线同篇 51 边证明内容可抽
- **新旧对比从未公平过**：旧用真实 MinerU 块（67-129 块/篇），新全用单块输入；FLUID/MOL 新侧跑在 evolution seed 修复（14:02）之前
- **分环节净判决**：单边质量净提升（类型规范 100% vs 29-78% 非法、offset 级溯源、Zalesak limiter/virial 定理 8 槽位等真 n-ary 数学结构）；**召回净退步**（-26%~-41%，演化边 FLUID 0 vs 旧 17）
- **复合 pattern_type bug 跨 5 run 杀掉 81+ 条演化边，≥62 条 roles 完美可救**

### C. schema 演化/divergence 出口审计

- **"并进"行为证据为零**：机制接线完好（每 section 重取 tbox，RETRIEVAL_K=30>21 全量可见），但 ML_DQN_2015 新 pattern 提交后 215 条后续边 0 次使用——纯 LLM 行为（planner 惯性 + 3 个新 pattern 本身是 divergence 垃圾，不用可能是理性行为）
- **prune 是死出口**：`process_batch_via_kernel`（prune 唯一 caller）全仓库零调用；kernel 路径 self_retire 未接线；6 bundle retire=0
- **跨篇复用率（A2 核心指标）当前不可测**：所有跑 fresh-seed 单篇；反向证据存在（不同 run 独立命名 compares_to vs compares_to_baseline）
- T-box 拓扑推断绕过 kernel 直改 tbox + bump version 不进 ledger（version 0.4 vs 0.16 自相矛盾）
- 4 域 bundle domain 全部硬编码 'ml'（跨域叙事证据被污染）
- distinctness 拦截不持久化；单测 mock LLM 对新 prompt 零覆盖

### D. 顶会战略审计（含组合创新性核查，竞品全部 curl 抓原文核实）

- **DIAL-KG（DASFAA 2026 已录用）全文明确声明 in-loop 闭环**——证伪 memory "intra-DAG propagation 全库无声明独有"断言；但其 ablation **无 "w/o schema evolution" 臂** → 剩余贡献位 = **首次严格验证 in-loop 演化收益（正或负）**
- **EvoGraph-R1（CVPR 2026 已录用）占 mutable evolving hypergraph memory 概念**（实例级、0 提 schema）→ mutable memory 卖点降级架构描述
- **AgentCAT 发表了饱和反证**：schema evolution bootstrapping-then-convergence 快速饱和
- **单篇内 in-loop 收益天花板天然低**（同篇 section 间 schema 需求不漂移）——三证齐全（AgentCAT 饱和 + 我们 0/215 + P0 负面）；**唯一活着的版本是跨篇 in-loop**（从未被跑过）
- 四元组合（n-ary+in-loop 演化+科学论文+下游验证）终检确认真空
- IG-Bench（1961 gold lineage traces）是 gold 扩容机会；占位速度每月 1-2 个邻接工作，**6 个月硬时间窗**
- Intern-Atlas 仍 preprint（head-to-head 打折但先中则叙事干净）

### E. 代码审计（+新授权下重建判决）

- **三个设计层缺陷导致 43% 语义正确率**：①三步分解丢失绑定上下文（step3 拿脱离句子的实体清单绑边，共现即绑定是默认行为）②方向/极性无人校验（规则 gate 只查 role∈pattern 声明集，从不查哪个节点拿哪个 role；verifier 截断默认 keep）③可定位性 gate 证明 evidence 存在、不证明边是 evidence 的忠实读法
- 斜杠 prompt bug 根因：`_STEP3_PROMPT` 用 "extends/improves/compares" 斜杠列表教 LLM 输出字面量
- 谜底：new_patterns 依赖从未设置的 `_initial_seed_pats`；version bump≠新 pattern（add_node 等也 bump）；fix_stats 缺失=内置 writer 显式剔除 section_reports（双 writer 并存）
- map_structure 静默路由硬编码 Kimi-K2.6（`llm=="deepseek"` 字面量判断）
- **kernel 路径 ablation arm 是假的**：add_only/no_intra_dag 与 full 行为完全相同
- 5-outcome route / propose queue / Skill 层全是死代码
- 多 seed 科学性勉强：无 seed 参数、call_llm 静默 fallback GLM-5-Turbo、无 model provenance
- **重建判决：B+（抽取栈重写+内核保留简化）**，最小架构比现在更简（见下）

---

## 三、交叉锁定的因果链（最重要的综合发现）

```
_STEP3_PROMPT 斜杠列表（代码层）
  → executor 输出复合 pattern_type "extends/improves/compares"（5 run 81+ 条）
  → role gate 整边丢弃（即使 roles 完美+evidence 齐全+verifier 判 keep）
  → 核心演化边全灭（内容审计实证）+ 新 pattern 无人可用（演化审计 0/215）
  → "方法演化一等公民"+"并进抽取"两个卖点同时零证据
```
**一个 prompt bug + 缺一个确定性 remap，同时打掉两个核心卖点的旗舰证据——修起来便宜，修完卖点可能直接复活。**

```
测试脚本单块 mock + _locate_chapter_text strict 正则失配（工程层）
  → 5 个重叠近全文切片 → 同篇抽 5 遍 → 387 = 5 遍并集（114 重复）
  → 所有 kernel_v2 数字不可进论文
```

---

## 四、现状 vs 顶会论文差距清单

### 论文必须（缺一投不出去）
1. **采纳探针**（2-3 天）：注入手工 gold pattern + [NEW] 强标记 + 数 usage；三分支判读定 schema 卖点形态（管道可破→修采纳机制 / pattern 质量问题→转 governance 故事 / LLM 能力不够→砍）
2. **稳定性+确定性修复**：真实 MinerU 块输入、确定性按标题切 section、退化分节守卫、同篇 3 run 方差 <20%
3. **抽取栈重写（B+）**：联合抽取替代三步分解、语义 verifier（逐边对 evidence span、截断守卫、方向/成员确定性检查）、边级语义正确率 43%→**≥70%** 后才谈演化闭环
4. **跨篇顺序 KB 判决性实验**：3 域×10 篇共享 KB（同域方法演替序列如 DQN→Double→Dueling→Prioritized→Rainbow）+ frozen 对照臂；测跨篇复用率/prune 出口/长尾命中
5. **A1/A2/A3 ablation** × ≥4 seed paired bootstrap（预注册已写好；A2 主指标加"新 pattern 跨篇复用率"）
6. **Baseline 3 个**：SCION（已赢补多 seed）+ Hyper-KGGen + binary 消融臂；DIAL-KG 有代码必须尽力接入（声明撞车不比会被问死）
7. **Gold 扩容**：≥2 综述 gold 或 IG-Bench spike（2 天时间盒）

### 强烈建议
- 长尾定性 case 包 / judge 交叉（GLM+qwen）/ 成本报告（7.4min/篇是真卖点）/ per-call model provenance

### 可砍（含无用功清单）
- QA/综述/冲突三个 consumer、HITL 闭环、衰减 reaper、LangGraph 完善度、速度优化、Skill Library 残留、ML_DQN 单篇深挖（debug 使命完成）
- 设计断点纯负资产：CAS、统一 queue、Skill 层、边级 5-outcome、replay/time-travel
- 旧管线全部 legacy 路径（保留一次受控对比后封存）
- 华为赛可并行参赛但不吃论文主线时间

---

## 五、方向判决

**不调大方向，重组证据重心：以 b 的证据结构讲 a 的故事。**
- 下限（已验证）：A1 n-ary vs binary + A3 富拓扑直读 + SCION/Hyper-KGGen PK
- 上限（唯一赌注）：A2 跨篇 in-loop 演化判决性实验
- schema 演化定位改为"首次严格验证 DIAL-KG 声明但未验证的收益"（正面引，正负结果都可写）
- mutable memory 降级架构描述（正面引 EvoGraph-R1/Co-E）
- 主投 WWW 2027（10 月截稿），保底 CIKM 2027，冲高 ACL 2027；4-6 个月全勤

---

## 六、工程判决：B+ 重建

**保留**：knowledge_base.py 内核（Mutation/两阶段 commit/ledger，6 套测试全过，论文机制贡献本身）、hypergraph_schema.py（T-box+seed）、concept_graph.py（A-box+概念合并）、llm_client.py（加 provenance）、全部评测资产（P0 脚本/gold/baseline PK/7 维评测/本次逐条语义审脚本升格正式指标）
**推倒**：structure_mapper 的 LLM 切块（改确定性按标题切）、三步分解抽取、双 writer、全部 legacy 路径
**重写**：联合抽取（按句/短段，绑定局部发生，方向/极性 prompt 显式约束）+ 语义 verifier
**砍掉设计断点**：CAS/queue/Skill 层/边级 5-outcome/replay
**执行序**：①确定性修复（1-2 天）→ ②抽取栈重写（语义正确率≥70% 验收）→ ③演化闭环验收（新 pattern 被后续 section 实际使用且产出语义合格边）

---

## 六之补：gold 构造质量审计（第六审，审查 session 末追加）

- **覆盖错位比 memory 记载更严重**：gold v4 只有 12/46 边双端在语料内（74% 结构性不可覆盖），根因=构造时无 coverable 过滤，gold 方法宇宙（6 家族）远大于语料宇宙（3 家族）
- **版本漂移 → 旧数字全部不可复现**：ERR_cond=0.667、fair_recall=0.833、SCION PK、P0 多 seed 表都是对着已不存在的中间版 gold 算的，eval 输出没记 gold 版本。**冻结版 gold 重算前，"已赢的 baseline PK"不能当论文下限引用（方向大概率仍成立，但数字要重算）**
- gold 自身 20-25% type/端点不可靠（接错线/方向反/21 条 evidence 不含端点名）；verbatim 锚定好
- citations.json 6/19 篇 DOI 错解析 → P0"C 臂无显著增益"的 citation 信号本身是坏的（三证里这条要打折，AgentCAT 饱和+0/215 采纳两条仍立）
- Thornton gold 是类目节点堆，跨综述实验需按新原则重建
- **重建设计原则**（详见 memory arfm-gold-audit-rebuild-principles）：coverable=双端源论文在语料内（DOI 对齐表）；语料反向扩充（先综述定方法族→按源论文补语料）；每边三件套（verbatim+source_papers+type_confidence 分层）；四道闸（LLM 候选池→机器闸→人工闸→冻结版本化）；3 综述×15-20 显式边+40-60 篇语料；ERR 三层报+type 容差

---

## 七、后续 session 必读

- 本文档 + memory：audit-2026-08-25-verdict、innovation-final-positioning（已更新）
- 采纳探针设计（D 审计）：手工 gold pattern（不能用现有 3 个 divergence 垃圾）、[NEW] 标记+定义+催生证据、usage 直方图、三分支判读
- kernel_v2 所有现存 bundle 数字不可用于论文（单块输入+修复前代码状态双重污染）
