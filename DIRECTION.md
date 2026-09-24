# DIRECTION — 论文大方向（状态：v3，2026-09-20 基准三考场定案后更新；项目已更名 CompileScholar）

> **本页历史**：09-16 版（三线框架）经 2026-09-17 代码一手核验判定作废；09-17 v2 重写后，09-20 又发生两件定案级事件——①基准切换（PaperScope 因 41% 脏 gold 弃用，换三考场组合）②项目定名 CompileScholar（编译 vs 检索的范式对话）。本页 v3 反映最新定案；与 IDEA-PLAN v4 冲突处以本页 09-20 节为准（v4 早于基准切换）。推导档案：`ccfa-workfiles/idea/logickg-idea-plan-2026-09-17/IDEA-PLAN.md`（v4）+ memory `session-handoff-2026-09-18` §六quindecies–novemdecies（基准判决全记录）。

> **09-27 更新注**：本页 v3 的"基准三考场"（§2）已被任务驱动实验矩阵设计取代（四任务族：固定/开放检索+固定/开放跨论文 QA+单篇 QA+库生长分析节；ScholarStack 竞品定位+外部方法 PK 制；终版矩阵见 memory session-2026-09-21-scaling-and-corpus.md 09-27 节，待调研子代理返回后定稿）。方向主张（§0/§1）不变。

## 0. 大方向一句话

做**内容可验证的类型化文献知识层**系统（构建管线+编译视图+typed tools），用同语料同后端的三家族 PK 矩阵（vs 通用 RAG / KG-RAG / 科学文献系统）证明它在科学 Agent 负载上的价值。〔用户 09-17 定调：主要工作=能与前沿系统 PK 的一套科学 Agent 文献系统〕

## 1. 主张结构（详见 IDEA-PLAN v4 §3）

- **C1（主）系统**：验证优先的构建管线（逐字锚+确定性三明治+表格通道+金丝雀）——"内容可验证性是构建出来的，不是评测出来的"。
- **C2（主）PK 矩阵**：科学 agent 场景首个预编译知识层 vs 通用 RAG vs KG-RAG vs 科学文献系统受控对线。
- **C3（辅）消融**：门消融/轴消融/成本 Pareto。
- **不作主张**：审计/溯源（用户 09-16 否）、规模定律（用户 09-16 否）、Find 开放网检索（饱和层）、Grow 演化、谱系主叙事（数据撑不住）。

## 2. 基准三考场定案（2026-09-20，覆盖此前一切基准计划）

- **主轴 = ScholarQA-Multi 108**（闭卷多论文综合；Citation F1=确定性字符串匹配+Prometheus 本地 rubric，零闭源 judge 零隐藏数据；闭卷 440 论文编译库）
- **量盘 = QASA 1375**（闭卷单篇 102 篇；短答案+引用确定性——需短答案适配层）
- **检索轨 = LitSearch 597/64,183 篇**（纯 Recall@K 客观零 LLM；已发表对照 OpenScholar 0.424 / Lacuna 0.538 R@10）
- PaperScope 弃用（summary 轨 41% 脏 gold）；arXiv2Table 降级可选；AirQA 不跑
- **对照表格局**：引用行（OpenScholar Table 2 全家 Multi-108 已发表数字）+ 闭卷协议自跑行（OpenScholar-8B/PaperQA2）+ 消融行（知识层组件拆解：视图/tools/card/编译分层——**ReAct-Direct 已除名**，"效果好非模型强"降为分析节归因，引 OpenScholar 裸模型行佐证）。LightRAG/商业系除名（闭卷考场进不来）
- **模型代际披露**：Qwen3.8-27B（2026 新架构）vs 表内 2024 模型，代际差双向；用代际定位不用"参数量小"自贬句式
- **外部检索模块（语料生长 broker）**：已定设计（四类召回靶/束搜索自适应深度/三档浅抽/两段式冷启动/引用原因驱动边权；访问层+存储底座=sci-evo-extract 已配 S2+compile_level），LitSearch=检索组件受控考场，不阻塞 Multi 闭卷主轴
- **成本约束解除**：本地 Qwen3.8-27B（230k ctx/16 并发实测）建库全本地零成本

## 2bis. 执行队列（2026-09-20 定稿：正式跑 Multi 前修完所有已暴露问题）

Multi 建库前：批 5 结构感知抽取 → F35 canary+apply（+ round2 默认 ON 入预注册）。
Multi 答题前：批 4 自由表组装 / 答案密度+引用格式对齐 extract_citations / A7 时机。
并行：图公式影响面评估（先测数据再决定）→ QASA 短答案适配 → 基建三小件。
收官：Multi-108 预注册 → 全本地建库 → 首跑。任务清单在 session 任务列表 #1-#9。

> 旧队列（09-17 版，全部已完成或并入上文，存档防引）：round2+仲裁消化 ✓（链接率 46.3%→76.1%/6.3%→66.2%，v38/v39）/ F35 上线 → 移入 Multi 前队列 / matrix 表键 → 随 round2 缓解 ✓ / PS-53 四臂闸门轮 → D53 终判完成（GLM 口径 +0.547 CI[+0.400,+0.700]，六轮修复至 3.747 新高）/ 答题栈定位 → 评测消费端候选维持。

## 3. 监控阈值（沿承）

- SciAtlas 出 experimental layer/v3 触内容四轴 → 定位重审（硬阈值）
- ASKS 补门/下游评测 → delta 收窄评估
- 两周节奏复查 zjunlp/SciAtlas commits（沿承既有机制）

## 4. 已废弃框架（防误引——不得作为任何后续工作的前提）

三线框架｜审计三角（Trust×Cognize 主 claim）｜乘法链中心句｜五层认知底座（对内愿景措辞，非系统现状）｜规模条件性定律主张｜IDEA-PLAN v1-v3 的过渡形态（v1 内容层主张/v2 首版 PK/v3 勘误版，被 v4 整合取代）｜PaperScope 基准计划（09-20 弃用）｜ReAct-Direct 消融臂（09-20 除名）｜LightRAG/商业系对照（09-20 除名）。

## 5. 纪律沿承

预注册不软/gold-blind/帽+10%停线/启动后活体确认/仪器先审/零token验证先行/官方数字永不进对比表/台账即更新/复杂脚本 Write 工具/临时文件进 .research_tmp/联网调研走 web-access（CDP 发现+一手 ID 核验+判决表）/系统通知≠用户批准/Claude 分析≠用户决定（事实/用户原话/Claude 判断三身份分离）。
