# GOAL 模式启动文档（compact 后第一件事读这个）

## 当前位置
设计 v2 已定（.research_tmp/plan_total_design_v2.md），三证据齐（MINDSET 不占/Intern-Atlas 缺三元/ablation 预注册），方向成立。**进入实现阶段，按 8 步顺序做**。守"全部设计好再动手"——设计已好，现在动手。

## 主卖点（重锚，正面引 Intern-Atlas，不假装发明）
n-ary 超边语义保真 + schema 并进演化 + agent mutable working memory 三点同时。差异三点：①n-ary 富拓扑（binary 拆开丢结构，富拓扑直读红利 memory 已 win 最稳兜底）②schema 并进自演化 vs Intern-Atlas fixed-at-release ③agent mutable vs read-only。schema 演化降支撑机制（机制层指标不赌下游）。

## 实现顺序（goal 模式只做基础打牢，不做下游 consumer）
**基础阶段（goal 模式做这些）**：
1. 底座内核（Mutation+Contract+Ledger+分层+distill_skill op+A-box referential integrity+5-outcome 分两阶段+CAS 收窄+schema-constrained rewrite 收窄 IS-A/family）
2. 抽取 agent（阶段1 planner+executor+verifier+fixer；semantic_boundary/provenance/central 字段改；domain Mutation 必带）
3. schema 自演化（queue 统一串行；DIAL-KG 三阶段；SAGE writer-reader）
4. 跨论文对齐（EDC+5-outcome 只 aligner；Concept 加 definition）
5. 富拓扑+维护（ambiguous 规则；utility 剪枝；衰减先诚实降级）
8. ablation 地基（A1 n-ary vs binary / A2 schema 并进 vs static / A3 富拓扑直读 vs co-occurrence；主对照 Intern-Atlas binary+static；≥4seed paired bootstrap；机制层指标非 NMR/ERR；预注册 .research_tmp/plan_ablation_preregistration.md）
**基础打牢后暂停，等用户确认再做**：
6. consumer（论文检索优先比赛紧；QA/综述/冲突/演化分析）—— **goal 模式不做，等基础扎实**
7. 协调/HITL/可观测（部分随基础做，完整 HITL 后）
**核心：基础不牢不做下游。下游论文检索是比赛紧但用户定调基础优先。**

## 防偷懒硬 checkpoint（goal 模式自动调子代理 review，不硬停用户，保证流畅做完）
**到这些点自动调子代理 review，子代理代替用户审核，发现问题我自纠后继续，不阻塞用户**（用户强调过多次约束，不重复让用户看；但子代理 review 是硬的，不能跳）。
- [ ] 第1步底座内核做完 → 子代理 review（断点修法全落地没？5-outcome 分两阶段？CAS 收窄？schema-constrained rewrite 收窄不声称对标2608.18104全部？A-box referential integrity？domain 注入？Skill Library 第四层+distill_skill op？+实际看成果对比原文）
- [ ] 第2步抽取 agent 做完 → 子代理 review（compound 28%→?/sink 26%→?/verbatim 改写14%→?/borderline 20%→?；verifier≠抽取模型 judge；**实际看抽取成果逐条对比原文分析不足**）
- [ ] 第3步 schema 自演化做完 → 子代理 review（queue 统一？writer-reader 闭环？schema-constrained rewrite 收窄？**实际看演化 schema 全表+对比原文**）
- [ ] 第8步 ablation 跑前 → 子代理 review（主对照 Intern-Atlas binary+static？≥4seed paired bootstrap？机制层指标非NMR/ERR？judge 交叉 GLM+qwen？预注册不事后挑指标？）
- [ ] 任何"声称对标前沿 X 全部"冲动 → 子代理 review（只声称做到的部分）
- [ ] 任何"降级简化"冲动（n-ary 退 binary / 5-outcome 退二元 / 冻 schema / 硬删不 ledger）→ 子代理 review

### checkpoint 之间自检（防中间段偷懒）
- 前科都在 checkpoint 之间中间段（qualifier bug 漏看拒因三轮/泛化坐实看数字没核对边）——不只 checkpoint 点看
- 每个实现小步也跑"实际看成果对比原文"自检，发现问题自纠
- checkpoint 报告里诚实说"中间段自纠了 X"
- 子代理 review 时也查中间段有没有偷懒/降级/看数字不看内容

### 子代理 review 怎么调
用 Agent 工具派 general-purpose 子代理，prompt 含：
- 读对应 plan（plan_total_design_v2.md + plan_detailed_design_supplement.md 对应部分）
- 实际看成果（跑出的 instance/meta/failed_edges/edges_dump）+ 对比原文
- 审：设计落地没/断点修法全没/偷懒降级没/声称对标过头没/实际看成果对比原文没（不只数字）/铁律守没
- 返回结构化问题清单 + 严重度
- 我照清单自纠后继续，不阻塞用户

## 铁律（compact 后 memory 自动加载，但这里再强调防偷懒）
1. 不偷懒不降级：n-ary 不退 binary / 5-outcome 不退二元 / schema 不冻 / 不硬删（ledger 可恢复）/ 不声称对标过头
2. 复杂语义用 LLM，结构/确定性用规则（pattern_type 名∈集合=规则；关系语义判断=LLM）
3. 防过拟合：特例不当硬规则，典型可作 prompt 例子标"举例非规则"，domain-specific 要 domain-gate
4. 可溯源 grounding（PaperQA/AutoResearch/GraphLoom 三处同设计）
5. 正面引 Intern-Atlas（2604.28158）作 binary 大规模先例，不假装发明方法演化关系
6. 不赌 schema 演化下游显著增益（P0 实测结构信号臂无显著增益风险）——锚机制层指标+长尾+qualitative
7. 现有结构不锁定（设计中发现局限可改，如 Hyperedge 加 provenance/Concept 加 definition/MetaHyperedgePattern 加 semantic_boundary）
8. 改核心设计前 git commit + 写 DECISION
9. 用完整方法（agent.process_paper_hypergraph，不裸 call_llm）
10. 地基看实际内容（边+对比原文），不只看数字

## 资源
- 设计 v2: .research_tmp/plan_total_design_v2.md
- ablation 预注册: .research_tmp/plan_ablation_preregistration.md
- 现有 bundle（4跨域+5颗粒流）: .research_tmp/runs/crossdomain/bundle/* + .research_tmp/runs/ARFM2024/hypergraph/bundle/*
- 现有代码: src/granular_agent/（MetaHypergraph/ConceptGraph/hypergraph_extractor/agent 等可复用为 agent 能力，但不锁定可改）
- memory 铁律文件: no-downgrade-simplicity-is-not-downgrade / design-no-downgrade-every-part / paper-novelty-must-prove / existing-structure-not-locked / intern-atlas-competitor-novelty-reanchor / p0-multiseed-no-signal-gain / rules-only-for-structural-deterministic-tasks
- 华为赛题（下游论文检索）: memory huawei-contest-paper-search-downstream；赛题文件 C:/Users/D0n9/Desktop/【7.8】2026年中国研究生人工智能大赛--华为赛题 (3).docx

## LLM 配置
- 抽取: DeepSeek-V4-Flash(Paratera) + thinking.type=disabled
- verifier/judge: ≠抽取模型（deepseek-chat 或 GLM-5，避免自我认同）
- 关 thinking 正确参数: {"thinking":{"type":"disabled"}}
- call_llm timeout=120s

## 偷懒前科提醒（用户反复纠正过，千万不要再犯）
- 搬现有流程进框架当设计（要重锚设计非搬；现有代码是 agent 能力实现，要重组为"底座读写者"非"管线阶段"）
- 只拆 re-type 一个节点（要全系统设计，不只补一个 pass）
- 默认沿用 step1/2/3（实测 54% 边 mis-typed 效果一般，要重设计抽取 agent）
- 看表面数字下结论（412/117/55 全真方法就停没查拒因；"泛化坐实"没逐条核对边——要逐条结合原文核对边语义+查 failed_edges 拒因+查 split）
- 声称对标前沿全部（schema-constrained rewrite 实际只 IS-A/family，不声称对标 2608.18104 全部；诚实只声称做到的）
- 最小可用版降级（要完整设计分阶段实现非简化版；5-outcome/utility 是有控核心不砍后加）
- 不深挖需求本质就设计（要先想"这个 agent 该是什么形态"从需求反推，非搬现有）
- 不等调研回来就设计（用猜测代替证据=偷懒；调研是证据来源）
- 纸面设计认了当验证（没做看不出真问题，做的过程发现方向性问题回来改设计，不硬推）
- 组合创新只声明真空不证收益（顶会要证组合>各部分，ablation 必跑）
- 不诚实报负面（P0 结构信号臂无显著增益，schema 演化 ablation 可能重现——无增益就诚实报+转卖点，不藏不利结果）
- judge 用同模型（自我认同偏差——verifier/judge 必须 ≠抽取模型，且交叉 GLM+qwen）
- 现有结构当神圣（MetaHypergraph/ConceptGraph/Hyperedge 不锁定，发现局限可改，如加 semantic_boundary/provenance/definition/central）
每次冲动→对照铁律+暂停问用户。**这一条是最高优先级：每次想"先简单版/先跳过/先声称/先不查"时，停下对照。**

## 验证硬要求（不只跑通，强调实际看成果对比原文）
- **每步实现后必须实际看具体抽取/演化成果 + 对比原文分析不足**——不只看数字（边数/标签/通过率），要逐条结合原文核对边语义对不对、演化出的 schema 全表长啥样、failed_edges 拒因、有没有偷懒/降级。
- **前科警示（多次犯过）**：①qualifier bug 漏看拒因三轮（36/40 边被拒没查，归因"抽取没产结构"误诊）②"泛化坐实"没逐条核对边（看 665/257 数字就下结论，深审发现 54% mis-typed）③"颗粒流够进下游"没逐条看（看 412/117/55 全真方法就停）——**这些都是看数字不看内容的病，goal 模式每步都要实际看成果+对比原文，不只报数字**。
- 每步实现后对照 v2 设计 + 补充设计(plan_detailed_design_supplement.md) + 铁律自检（不只"跑通了"）
- 抽取/演化/对齐质量验证：逐条结合原文核对边语义 + failed_edges 拒因 + before/after 数字 + judge≠抽取模型交叉 + 演化 schema 全表看（不只 pattern 数）
- ablation：≥4 seed paired bootstrap 95%CI（复用 P0 脚本）+ 预注册指标不事后挑 + 主对照 Intern-Atlas binary+static + 机制层指标非 NMR/ERR + qualitative+长尾对冲 aggregate 稀释 + 诚实报负面
- 跑完必暂存全细节 bundle（meta+failed_edges+evolutions+source+edges_dump），方便深审
- **checkpoint 暂停给用户看的不是"跑通了+数字"，是"实际抽取/演化成果样本 + 对比原文分析 + 不足在哪"**
