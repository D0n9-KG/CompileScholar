# EDITORIAL DECISION：PROPOSAL v6 第一轮（2026-09-03）

> 综合五席报告（R1 方法学 / R3 数据库跨域 / DA 魔鬼代言人 / EIC 领域主席 / R2 领域专家，全文见 PROPOSAL-2026-09-03-v6-review-round1.md）。

## 决定：Major Revision（大修后重审）

五席无一拒绝方向本身；无一认为当前形态可发表。共识的失败模式集中在两处：**novelty 辩护的结构性空洞**（DB 理论轴 + 循证医学 evidence synthesis 轴两条查新盲区，正好是作者自己的质疑所指的方向）和**评测设计的混淆变量**（层消融不封闭、C3/C4 零验收标准）。

## 对用户原始问题的最终回答

**"提前处理好后面要用的"——五席一致裁决：这个动作本身无新意，且论文的差异化声明不能打在预计算上。** 但五席也一致认为存在可防守的创新内核，重排后的承重结构是：

| 声明 | 五席裁决 | 修订方向 |
|---|---|---|
| C2 冲突感知编译 | 潜力最大但当前近邻错误 | 换轴：从"vs 物化视图"改为"**真值判定 vs 条件等价性下的冲突检测**"（truth discovery 假设冲突同构，文献域冲突大量是伪冲突——条件不同且未声明，条件本身带噪声） |
| C3 失效代数 | 概念干净、表述过度 | 换框架："**视图维护在编译器不可靠时的退化形式**"——DB 的 V=Q(D) 可靠可重放，我们没有 Q；失效检测召回率/重算收敛性成为可测量的新对象 |
| C4 delta 继承解析 | **五席全部独立认定为最干净的新机制** | "零先例"降级重写（引 citation intent 线+化学 same-procedure-as 先例）；权重上升；作为 C2 的支撑机制 |
| C6 表示保真度评测 | EIC 判"全文最坚实的贡献位" | 补 OAEI/GERBIL/KGBench 划界；gold 构造者间信度；冲突敏感性 ROC；失效注入测试 |
| C1 问题定义 | 被直接反例（GraphRAG community summaries 就是离线预计算） | 重述："物化的不是通用内容摘要，而是负载类型化、对齐后、带维护语义的答案结构"；纳入 DocETL/Palimpzest/Lotus 作为查询时对照极 |
| C5 负载驱动方法论 | 三席独立判凑数/不成立 | 降级为方法论节；负载目录本身作为 artifact 贡献（独立价值更硬） |
| C7 缺失性查询 | closed-world 的直接应用（Reiter 1978）+ evidence gap maps 近逐字重合 | 降格为覆盖地图的能力演示；引 incomplete-DB 与 3ie gap maps 划界 |

## DA 五条 CRITICAL 的逐条裁决（铁律）

| DA | 裁决 | 依据 |
|---|---|---|
| C-A 编译与缓存不可区分 | **成立，阻断性** | R1 F2（混淆变量）、EIC MAJOR-2/7（成本模型）交叉佐证。必须补能力配对对照+摊销成本模型 |
| C-B 噪声放大无量化 | **成立，阻断性** | EIC CRITICAL-2（C3/C4 零验收）、R1 F7 交叉佐证。必须补 per-field error budget + k 跳 delta 链误差传播下界 + 置信度一等字段 |
| C-C 负载循环性 | **成立，阻断性** | R1 F1、R3 F4 独立提出（三席撞车）。Asta 降为动机证据；须收真实 agent 查询日志作桥接 |
| C-D SciAtlas 窗口+组合归因 | **部分成立** | 窗口风险 EIC 认可但反对被焦虑驱动；组合归因由层消融部分覆盖但需单视图消融细化。须写 scoop 硬阈值 |
| C-E 对齐键单点失效 | **成立（风险级，非阻断）** | 自家 split 命名不收敛/NMR 跨语言失败史佐证其现实性。需误差传播分析+降级通道的行为评测 |

## 五席交叉共识（按票数）

1. **DB 理论轴查新缺口**（R3 F1/F2 + EIC CRITICAL-1，两条 CRITICAL 撞车）——truth discovery/data fusion/provenance semirings/c-tables(1984)/view adaptation/AGM。第一优先。
2. **Asta 人类日志当设计输入**（R1+R3+DA 三席独立）——降级为动机证据。
3. **C4 升、C5 降**（R3+DA+EIC+R2 四席独立）。
4. **C3/C4 零评测挂钩**（EIC CRITICAL-2 + R1 F7）。
5. **编译 vs 缓存区分**（DA C-A + R1 F2 + EIC MAJOR-2/7）。
6. **L0 消融臂污染**（R1 F2 + DA M-B + R2）——需 L0' 原始切块臂。
7. **"knowledge compilation" 术语撞 Darwiche & Marquis 2002**（EIC MAJOR-3 + R2 F5）——改名（候选：answer-structure compilation / demand-driven materialization）。
8. **evidence synthesis 线缺席**（R2 F2 CRITICAL）——evidence gap maps 与覆盖地图近逐字重合，必须引 3ie/Campbell 划界。
9. **E2 扩版从"加固"升为"资格线"**（R1 F3 + EIC MAJOR-4）。

## 修订路线图

### P0-1 补两条查新盲区（先于一切设计工作）
DB 理论轴（truth discovery/data fusion、provenance semirings、uncertain DB、c-tables、view adaptation、incremental ER、AGM）+ 循证医学轴（evidence gap maps、living systematic reviews、Trialstreamer/RobotReviewer、元分析自动化、GRADE）+ citation intent 线 + 化学 same-procedure-as 线。产出：**机制 × 最近先例 × 我们的 delta** 对照表。

### P0-2 两个最便宜的致命对照（决定方向生死，先于 Phase 1）
(a) **Scaling baseline**：给 RAG 臂 10 倍 token 预算+前沿模型重跑 E2——若差距闭合，编译只剩成本故事（DA 替代路径 5）；(b) **最小视图 smoke test**：用现有 L1 资产构造一个最小对比矩阵视图，在 E2 聚合类任务上试跑（R2 建议）——论文成败押在层消融上而它是唯一没做过先导版本的核心实验。

### P0-3 真实 agent 查询日志（桥接证据）
从开源 deep research 系统收集几百条真实查询（R1 建议），把"人类分布≈agent 分布"从假设降为可检验命题。

### P1 提案重写
- 改名；C1 重述+限定词收紧；声明结构 7→5（C1'+C2+C3+C6 承重，C4 支撑，C5/C7 降格）
- 回答 DA 两难（冲突簇消解做成视图原生操作+置信度一等字段+残余消解任务单独评测）
- 论证 eager vs lazy（或采用 hybrid；Asta 重尾分布是 lazy 的证据）
- 评测重设计：L0'/L0/L1/L2 四臂、对齐冻结、甜区/非甜区探针、能力配对对照、预算预注册、per-field error budget、失效注入、delta 保真度、单一主要对比+效应量
- 引用修复：Intern-Atlas/ORKG/Eigenius/AI-Supervisor/DocETL 线/OAEI-GERBIL/Lacuna 双口径/1.58 臂配置交代

### P2 定位与战略
- Venue：主投形态最稳落点 **NeurIPS D&B**（C6+负载目录+层消融协议打包）；主轨升档条件四项（DB 轴闭合/层消融分离成功/C3C4 评测/扩版+第二域）
- Scoop 应急硬阈值：SciAtlas v3 出现内容四轴任一 → 切换 benchmark-first 论文形态（C1+C6+负载目录，DA 替代路径 3 与 EIC venue 判断独立收敛于此）
- 组合归因补强：单视图消融（比层消融细一级，R3 实质化路线同时回应）

## 保留未决（下一轮讨论项）

1. DA 两难的选择：冲突簇留在格子（agent 消解）vs 视图消解+置信度——倾向后者但需设计细节
2. eager vs lazy 的裁决依据：P0-2(a) 结果+摊销模型
3. benchmark-first vs 系统论文的主次——**这是作者的战略决定**，取决于对 SciAtlas 竞速的风险偏好
