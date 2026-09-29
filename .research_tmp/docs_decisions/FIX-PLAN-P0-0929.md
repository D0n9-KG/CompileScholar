# 修复方案总档（09-29 深夜定稿待执行——新 session 从这里开始）

> **新 session 指令：先全面审读本档+关联档，找出方案中需要与用户深入讨论/
> 确认细节/可能有问题的地方，与用户讨论清楚后再开工。不要直接开始修。**

## 一、方案全貌（叙事→宣称→修复→实验的依赖链）

**叙事 v6（已用户裁定）**："一个会生长、知道自己缺什么、按意图定向探索的
领域世界状态"——四件套 vs LKM（推理图）/ScholarStack（事实资产）。
权威档：NARRATIVE-GROWING-STATE-0929.md。

**架构裁定（09-29）**：答题模式与生长模式显式分离——
- 答题模式（benchmark）：生长开关关闭，检索仅服务当前问题
- 生长模式（能力主张）：系统持续外部检索→抽取→入库→backflow 自主扩库
- benchmark 分数与生长主张解耦，各自干净

**四件套 → P0 修复依赖**：
1. 会生长 ← backflow 字段适配 bug（P0-1：深读记录字段名≠粗抽
   ——scope_ref_ref 等 vs method/subject，54/66 篇深读没回流，
   "生长"机制 9/26 后静默断裂）
2. 知缺 ← gap_status 动态比对层（P0-4：信息槽位×证据满足度差集）
   + absence 语义失效（P0-2：improves/replaces 边配对旧缺口→resolved_by）
   + 中文缺口 275 条清洗（P0-5）+ 派生缺失通道（空词表）
3. 定向探索 ← gap_status 前端=问题信息需求类型学分解（首步 LLM 调用：
   题面→[方法/指标/局限/应用/对比]槽位清单）+ 后端=槽位满足度检测
   + 意图边定向遍历（查新确认"agent 意图过滤遍历"arXiv 0 命中可主张；
   "意图边类型"已被 Typed Claim Network 2605.30966 占——引用划界）
4. 可追溯 ← 谱系边年份回填（P0-3：2406 边全 None，从论文年份编译期回填）
   + resolved_by 缺口生灭链

**CS2 提分的信息缺口方案（09-29 定）**：
核心=问题信息需求分解层（类型学来自 100 题池统计的通用模式：
方法 90%/局限 67%/应用 54%/指标 53%/对比 44%——科研综述通用要求非
CS2 特有，过拟合安全）。三层：
- 前端：首步分解题面→信息槽位清单（27B 自己执行，分解比答题容易）
- 中端：槽位→检索策略映射（指标槽→compare+deep_read；局限槽→
  findings(claim_type=criticism)）
- 后端：gap_status 槽位满足度检测（不满足的槽位=精准检索目标）
现状缺口（数据实锤）：agent 首步零分解直接扎 card/findings（27B 无
planning；harness 的 Claude 底座自带 planning——这是 IR 差距来源之一）

**预算治理（P1，调研背书）**：
- BATS（2511.17006）每步注入剩余预算播报——agent 动态调整
- s1 budget forcing（2501.19393）：26/30 步注入"停止探索立即作答"、
  28/30 只留写作工具——零模型依赖
- 快收束下限保护（替代"简单题减预算"——批间数据：快收束组(≤12步)
  G=0.52 最低且全是缺口题提前放弃；中段 13-22 步 G=0.739 最优；
  耗尽组 26-30 G=0.671 不差——30 步是够的，快收束才是失败模式）
- deep_read 维持片段式（2-5 定向 chunk），补：chunk 自包含（section
  语境头）/回传≤2.5k token/Self-Route 兜底

**生长能力实验（自建协议，独立于 benchmark）**：
- 载体：DSB 子领域聚集（cs.IR 21 题/cs.CV 16 题）或自建同领域递进序列
- 协议：生长模式跑固定题序（KB 只增），累积 KB 回放（同题在初始 KB
  vs 生长后 KB 各答，判据=生长行为度量[新增记录/边/回流率]+生长效用
  [命中/得分差]两层）
- 反作弊：题序预注册、KB 单调、无人工干预
- CS2 主题零重叠（领域词≥2 重叠仅 1 对）——生长实验不在 CS2 做

## 二、执行清单（新 session 讨论后开工）

### P0（叙事兑现依赖，全工程性）
1. backflow 字段适配（外部_tools.py:_backflow_deep_records 的 mentions
   提取改用深读字段：scope_ref_ref/target_ref/condition/claim_text）
   + 存量 54 篇批量补回流（离线脚本）
2. absence 语义失效（入库/离线管线：improves/replaces 边→配对旧论文
   absence→resolved_by 标记，保留历史链）
3. 谱系边年份回填（编译期一次性：边两端论文年份）
4. gap_status 层（前端类型学分解+中端槽位映射+后端满足度检测）
   ——与 V4 现有触发的关系需讨论：替换还是共存
5. 中文缺口清洗（275 条）+ 派生缺失词表修复 + 矩阵覆盖（0.9%）决策

### P1（agent 工程）
6. 预算治理（BATS 播报+s1 强制收尾+快收束下限保护）
7. deep_read 强化（chunk 自包含/≤2.5k/Self-Route 兜底）
8. KB 查询语义主通道化（PPR 库内排序升级——谱系边+实体图为底，
   词法降为精排；治规模化）

### P2（工程债）
9. 流水线串行化脚本（adapt/judge 竞态踩了 4 次）
10. watchdog（判分/答题挂死自动 resume——inspect 路径挂过 2 次）
11. 依赖链三件套检查脚本（pip 装包后 openai/litellm/inspect 连锁断 3 次）

### 修完后
12. 批 16-25 的 50 题用最终系统重跑（方案 A）+ 批 29-30 补完
    （offset 80-99+0-9）→ 100 题全量单一版本
13. 生长能力实验（DSB 载体）
14. GPT Researcher CS2 臂接线收尾（配置坑修到第 8 个）

## 三、冻结点资产（09-29 深夜）

- ours 臂：65/100 题双样本（offset 10-79，批 15-28）
  - 同尺批间：批16(核心带,V1-V3)=0.736 / 批23-25(缺口带,V4未生效)=
    0.60-0.63 / 批26-27(V4生效)=0.654-0.659
  - 同 5 题严格对比（批16 段）：ours 0.736 vs harness 0.730（打平）
  - facet：IR 0.69 vs harness 0.83（主差距）；CR 0.70 vs 0.42（我方赢）
- 判分：DeepSeek-V4.1-Flash（Paratera）全面启用，提速 20-30 倍
- DSB STORM 63/63 答完，36/48 判完=0.223（官方 0.183）
- DSB harness 63/63 答完（62 有效），63 目录重判在跑
- CS2 harness 45/100（29 有效——prompt 超长崩溃 16 题是它的可披露弱点）
- GPT Researcher：v0.16 配置七坑修复史（Config 无 kwargs/构造器不收
  config/retrievers 复数优先/strategic_llm 独立字段/env 时序/openai 被
  降级/报告 API 变化），冒烟未通
- 方差：V1-V3 收窄 37%（0.11→0.069）；批 24 极差 0.044 历史最低

## 四、纪律红线（违反=返工）
- benchmark 过拟合纪律（memory: benchmark-overfit-discipline）：
  机制驱动修复+held-out 验证+禁碰题目/rubric/判分 prompt
- KB 永不按题补库；信息类型学的过拟合安全性论证（来自通用科研写作
  结构非 CS2 特有）
- 答题/生长模式分离——benchmark 关生长跑
- pip 装包后依赖三件套检查
- adapt/judge 串行发起

## 五、关键文档索引
- NARRATIVE-GROWING-STATE-0929.md（叙事 v6+四件套+实验设计）
- RESEARCH-KB-ASSISTED-RETRIEVAL-0929.md（结构辅助检索调研）
- RESEARCH-EVIDENCE-BUDGET-0929.md（分层证据+预算调研）
- AUDIT-FULLCHAIN-0928.md（六代理全链路审计）
- MORNING-REPORT-0929.md（夜班战报）
- EXPERIMENT-DESIGN-REV-0928.md（三基准终裁）
- 调研结论：类型学统计/gap_search 错位证实/citation_graph 0 使用/
  快收束组全是缺口题/首步零分解
