# 提案 v3：SciAudit——科研 agent 结论的 presence-only 结构审计

（v1 SciEpisteme 被 5 席审稿团否决；v2 双层方案（CR 统计+审计器）被审稿团判定"A 降级+B 重设计"；本 v3 吸收两轮共 8 份审稿意见 + 3 个判定实验 + 4 轮调研）

## 1. 定位与主张

**主命题：不完整的抽取不阻断可信的审计——跨家族一致性认证的高精度骨干 + presence-only 检查（只从图中存在的证据触发指控），就足以审计自主科研 agent 的研究结论。**

"验证 agent 产出"已被钉为领域公认瓶颈（35 个 AI-Scientist 系统审计：仅 38% 有任何 novelty-verification 方法 [2608.05179]）。我们的差异化：不做逐条 entailment 验证（FactScore/SAFE 谱系已占），做**结构检查**——只有知识图能承载、逐条验证原则上做不到的检查。

设计原则（prosecutor-not-judge，来自审稿修正）：
- 审计器只在**图中有据**时提起指控；图里没有 = 显式弃权（"此维度无法审计"），绝不判"通过"
- 这把"图不全"从缺陷变成安全边界：缺失知识的唯一后果是沉默，不是冤枉

## 2. 三个检查（按新颖性强度排序）

### 检查①（主打）：基线过时 / supersession 审计
agent 声称 "improves X"，沿方法演化链（improves/extends/replaces 边）+ 年份检查 X 是否已被淘汰。
- **新颖性判决（4 轮调研）**：真新颖，三查中最强。现象有文献（撤稿论文持续被引 [2501.00473/2503.18215/2509.18403]、cited-reference 死亡率 [2203.08649]）；演化链数据资源已存在（Agents-K1 [2606.13669] 的 method lineages，246 万篇）但**无人用于审计**；最近邻 E3 [2605.27072] 审人类论文判"弱基线"，不做图上 supersession 推理；Papers with Code 已死（实测 302 重定向）——SOTA 跟踪无活跃维护方
- 必引划界：E3 / Half-Lives of GenAI Evidence [2607.24032] / MemStrata+Supersede 集群（agent 记忆新鲜度，非方法淘汰）/ ABE-Ralph [2608.26753]（实验保真≠新鲜度）/ Agents-K1
- 数据层现状：gold 演化边 66 条、跨篇方法节点 3 个——链密度低是主要工程风险，骨干需跨家族认证提升演化边质量

### 检查②：伪共识 / 引文独立性塌缩
agent 引用 N 篇文献声称"领域共识"，分析这 N 篇在引文图上的祖先——若共享同一源头（cascade citation），表面共识实为单一证据。
- **新颖性判决**：组合层面真新颖。经典奠基（70-90% 引用是抄的 [1109.2272]）+ LLM 引用行为 2026 爆发（Citation Monoculture [2608.19230] 证明引用集集中度——选择层，非 ancestry 层）+ 引用工具全做存在性/支持度不做独立性（sciwrite-lint/AI-Powered Citation Auditing）
- **必引划界：JuryProbe [2608.20607]**（同一认识论论点的不同介质：评委 panel 共享盲区 vs 引文图 ancestry——正面引用并把差异钉在介质上）
- 数据层现状：ARFM 语料 19 篇 raw_refs **100% OpenAlex 解析**（1492 条）——祖先分析不受语料边界限制，基础设施成立

### 检查③（配套，降级）：矛盾检查
agent 结论与骨干中高置信事实冲突时触发。
- **新颖性判决**：机制层被占严重（FactScore/SAFE 谱系 + 2026 密集 KG 矛盾检测：HalluGraph/CuraView/AutoVerifier/Traxia）——仅存的差异化 = 对象是科研 agent 结论 + presence-only 语义
- 理论锚：2606.30247 证明不完备图证据下"仅凭已观测 KG 状态的硬规则"不可能同时拒掉所有假 unsupported 并保留所有真但未观测的——这正是我们选择 prosecutor 语义（只指控不判过）的理论依据兼限制声明
- 实验支撑：V4-Flash 加显式负向抽取指令后正确抽出 does_not_improve/fails_to_converge 带 polarity 标注——负向边数据层可行（旧 NEGATION 规则反转：从"不许抽"到"标注极性抽"）

## 3. 骨干认证（原 Idea A 的降级形态）

高精度骨干的选择机制：**跨家族模型抽取一致性**（不同模型家族都抽到的边才进骨干）。
- 实验二实测：同家族（V4-Flash vs deepseek-chat）14/14 全重叠=假独立；**跨家族（gpt-oss-120b）与两者只重叠 3 条**，三方一致的边恰是最稳事实；跨家族分歧集中在**类型判断**（V4 抽 influences 处 gpt-oss 抽 claim_relation）——一致性筛选的正是"事实+类型都稳"的边
- 统计框架（审稿修正）：Dawid-Skene / 多重列表分析（multiple-systems analysis），**不用** Lincoln-Petersen（R2：K>2 用 LP 是术语错误）；与 Snorkel label model 正面对比（弱监督已占"多标注函数学权重"，我们的差异=闭语料+schema 固定+结构化关系）
- 覆盖率估计（原 Idea A 的残留）降级为诊断性小节，全程**下界语言**（"至少缺失 X%"），用 Chao 族只做骨干密度的健康监控
- 工程约束（实测）：跨家族模型走 CST 通道大输出超时——骨干认证按"小 chunk × 多模型"设计；成本 K 倍，作为一等指标报告

## 4. 评测设计（预注册要点，来自两轮审稿修正）

**双轴**：检出率（错误结论被指控）× 清洁通过率（正确结论不被冤枉）——PPV 一并报告；审计器输出区分 "verified-accused / verified-clean / not-audited（弃权）" 三态，auditability rate 显式输出。

**错误分类学**：从真实 agent 失败记录派生（AutoScholar 失效模式 + holdout 表是现成原料），held-out 类别不参与检查器设计；播种错误分 naive 与 adaptive（红队 LLM 看过审计器设计后制造不可检测错误）两档。

**必做基线臂（DA 指令，采纳为设计原则）**：三级便宜护栏——数值代码重算 / 数据 provenance 去重 / 答案时 entailment 校验（MiniCheck 类）。结构化审计必须打赢或定位出它们的失效盲区才有存在权。预期盲区：便宜护栏全在"逐条"层，supersession 和引文 ancestry 是"结构"层——恰好是逐条验证原则上做不到的。

**杀手级评测（科学学贡献通道）**：在真实 agent 输出上测发生率——"X% 的结论锚定在已被淘汰的基线上、Y% 的'共识'引用集存在独立性塌缩"——这是关于自动化科学如何失败的实证发现，把科学计量学经典命题（过时引用/伪共识）变成可预测可干预的结构。

**消融链**：全图 vs 一致性骨干 × presence-only vs 含"已知性"检查（预期后者 precision 崩塌——这本身是论文的一个发现）。

## 5. 风险与诚实声明

1. 检查②（伪共识）依赖引文图质量：OpenAlex 有封禁前科（polite 池+备援）；引用意图分类本身是活跃研究领域
2. 演化链密度低（gold 下界 66 边/5 篇）——检查①的存活率是最大工程风险，骨干认证要先把演化边质量提上去
3. 结论→可查询声明的分解是隐藏 NLP 难题（claim decomposition+实体对齐——我们有对齐 bug 前科），检出率必须因式分解为 P(分解正确)×P(事实在骨干)×P(匹配命中) 三因子分开报
4. 审计器自身是 LLM——审计器的 judge 也要交叉校验（评测工具先于被测系统审计的纪律同样适用）
5. 窗口收窄：Traxia（KG+矛盾检测的 agent 发布框架）是并发 spec 无实验，若其放出实验并扩展到结论审计，检查③的最后缝隙关闭——动作以月为单位
6. "组合>各部分"仍需受控证明：detection-vs-backbone-size 曲线（A 的置信度排序做骨干子采样 vs 随机等量）+ auditability rate 预测散点
