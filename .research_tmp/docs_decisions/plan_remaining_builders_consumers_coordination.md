# Plan: 剩余部分详设（富拓扑/维护 + consumer 群 + 协调）

## 4. 富拓扑推断（builder 第 4 个）
**现状**: infer_rich_topology_direct 从 n-ary 超边直读 7 类(law_parameter/method_parameter/method_phenomenon/method_regime/composition/definition/nary/evolution), memory 标过是 win(直读非 co-occurrence)。
**对标**: DocTrace(agent-shared hypergraph working memory + query-triggered on-demand "Trace Only What You Need")。
**设计**:
- 7 类直读保留(确定性 win,不降级——结构能判的不上 LLM)
- 加 LLM 辅助判 ambiguous case(超边可多解读时 LLM 判主类型,克制用)
- 富拓扑作 consumer 的 working memory(DocTrace 式: query-triggered on-demand 取子图,非全量预计算存死)
**创新点作用**: 富拓扑直读(非 co-occurrence)是候选创新点之一。ablation: 直读 vs co-occurrence baseline,证质量好; working memory on-demand vs 全量预计算,证效率。
**现有改**: rich_topology 边带 domain namespace + 查询接口(on-demand 取子图,不预计算存死)。
**不做**: 不退 co-occurrence(直读是 win 不降级); 不全量预计算(on-demand 但设计完整)。

## 5. 维护 agent（builder 第 5 个）
**现状**: run_split/merge/retire/rename + consolidate_instance(域门控清理)。
**对标**: SEDM(utility 排序剪枝) + SCION(保守融合) + MemoryBank(遗忘)。
**设计**:
- retire 按 utility(使用频次/被引用/演化贡献)排序,低 utility 退(SEDM)
- split/merge 触发更 principled(不只 co-occurrence/embedding,加 utility+语义边界判据)
- consolidate 域门控保留(确定性规则做结构)
- 保守融合(SCION: merge 谨慎,需 evidence+consensus)
- 遗忘(MemoryBank: 低 utility/旧 pattern 衰减而非硬删,可恢复)
**创新点作用**: 维护是"有控防膨胀"执行,支撑"schema 并进演化不失控"(DIAL-KG 237 膨胀教训)。ablation: 有维护 vs 无,看 schema 膨胀/质量/下游。
**现有改**: retire 加 utility 排序接口; split/merge 触发加 utility+语义判据; 衰减而非硬删(带可恢复 ledger)。
**不做**: 不硬删(遗忘可恢复,保守); merge 不放松(保守融合是有控核心)。

## 6. consumer: 论文检索（华为赛题,优先级高）
**对标**: PaSa/SPAR(多 agent+查询分解+迭代) + HippoRAG(PPR over KG 多跳) + GraphLoom(reliability-aware subgraph routing 防多跳噪声) + grounding 三处(PaperQA)。
**设计**:
- 查询理解分解(schema 概念层级 map 方法论约束 + 子查询分解 + 改写扩展)
- API 检索(粗,Semantic Scholar/OpenLex,覆盖广) + 超图/schema(关联推理/重排/归纳/可溯源,覆盖内精)
- evolution 边(extends/improves/compares)+富拓扑做细粒度关联("找跟 X 方法 extends 的论文",超关键词/语义) ← **差异化核心**
- LLM agent 迭代策略(PaSa/SPAR 式: 自主规划搜索词/过滤/迭代调整)
- 综合排序(细粒度相关性: 超图关联+语义+citation)
- 结构化归纳(关系图,超图本身是图)
- grounding 三处(查询理解/检索/生成,evidence 溯源)
- 成本控制(赛题要求: API 次数/Token)
**诚实张力**: 超图覆盖不了海量 API 检索→超图是增强重排层非检索源; 覆盖度偏颗粒流(要扩通用域或只用于已建超图子集精排)。
**创新点作用**: 论文检索用 evolution 边+富拓扑做细粒度关联=应用差异化(赛题层面,也是顶会创新下游验证场景——证超图+schema 对真实下游有用)。ablation: 超图重排 vs 纯 semantic/citation baseline。
**不做**: 不替代 API 检索(超图是增强层); 不砍 evolution 边关联(差异化核心)。

## 7. consumer: QA
**对标**: DocTrace(hypergraph working memory+多跳) + PaperQA(provenance) + HippoRAG(PPR 多跳)。
**设计**: 精确检索(query→超图子图,富拓扑 on-demand) + 多跳推理(PPR/路径推理 over 超图) + grounded 回原文(每结论回 evidence span) + grounding 三处。
**创新点作用**: QA 验证超图 working memory+方法演化关系链推理。ablation: 超图多跳 vs flat retrieval。

## 8. consumer: 综述
**对标**: AutoSurvey/Agentic AutoSurvey(4 专化 agent) + SciSage(hierarchical Reflector) + DAS(shared revisable state+reusable vs topic-specific 分离)。
**设计**: 广覆盖(多查询分解+检索) + 跨篇综合(超图跨论文 concept+evolution 链综合) + 结构化(超图作骨架,shared revisable state) + 可溯源 + 反思(SciSage Reflector)。
**创新点作用**: 综述验证 A-box 跨论文成图+方法演化链综合价值。ablation: 超图结构化综合 vs flat LLM 综合。

## 9. consumer: 冲突检测
**对标**: MELD conflict outcome(5-outcome 的 conflict 入口)。
**设计**: 交叉对比矛盾 claim(同 concept 不同 cited_from/method 的冲突边) + 5-outcome conflict 入口(aligner 标的 conflict 走这) + embedding+NLI 判矛盾 + 输出冲突报告+可溯源。
**创新点作用**: 冲突检测是 5-outcome conflict 出口,验证有控 A-box 写入价值。ablation: 5-outcome conflict 检测 vs 二元。

## 10. consumer: 演化分析
**对标**: evolution 边链推理(extends/improves/compares 链)。
**设计**: 方法族+evolution 链(extends/improves/compares 图遍历) + 演化路径(A→extends→B→improves→C) + 输出方法演化图谱+可溯源。
**创新点作用**: 方法演化关系一等公民的验证场景(最核心创新点之一)。ablation: evolution 边作一等公民 vs 从 text 推。

## 11. 协调/HITL/可观测
**对标**: MemGPT(interrupts) + AI-Supervisor(consensus commit) + CoAgent(并发控制) + LangGraph(StateGraph+Command+Store 脚手架)。
**设计**:
- builder 写串行/consumer 读并发(version CAS,Mutation 带 base_version)
- consensus commit(AI-Supervisor: 新 top-level family/重要演化 corroboration 后才 commit)
- HITL interrupts(MemGPT: 新 family/不确定/re-type 抽查暂停人审)
- writer-reader 闭环调度(consumer 提案 queue→builder 审, SAGE 式)
- 可观测: mutation ledger tracing + checkpoint + replay(=暂存自动化+崩溃 resume)
- 框架: LangGraph StateGraph+Command+Store 指向 KB 内核(编排脚手架,核自建不靠框架)
**创新点作用**: 协调支撑多 agent 围绕自演化底座读写一致,是 builder/consumer 生态运行保障。
**不做**: 不靠框架做核(调研证伪); 并发控制接口预留但 builder 串行时无并发(非降级,是设计内)。

---

## 全系统创新点总盘点(守顶会约束)
**最强卖点候选**: n-ary 超图作 agent 工作记忆 + schema 自演化并进 + 方法演化关系(extends/improves/compares)作 A-box 一等公民 + 富拓扑从超边直读(非 co-occurrence) 的组合。
**须证明**: ① 真空(组合没人做,尤其方法演化关系+schema并进) ② 收益(比 baseline: binary triple演化/n-ary不演化/无方法演化关系/co-occurrence富拓扑,在质量+下游好)。
**ablation 矩阵**(每部分设计时预留开关):
- schema: frozen/演化 × 并进/离线 × writer-reader/单向
- 抽取: plan-execute-verify 各子流程可省
- 对齐: embedding+judge/surface/LLM-only × 5-outcome/二元 × 隔离/不隔离
- 富拓扑: 直读/co-occurrence × on-demand/全量
- consumer: 超图关联/flat baseline
**诚实**: P0 实测结构信号臂无显著增益——演化收益非显然,ablation 必须真证明,否则创新点站不住。

## 全部设计完,下一步
派子代理审核整体设计(一致性/可行性/降级点/创新性真伪) + 评判创新性(深查组合真空+收益证明路径合理性)。审核完再动手实现。
