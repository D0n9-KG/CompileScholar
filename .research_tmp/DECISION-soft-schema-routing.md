# DECISION: schema 约束旋钮调整 — 拒杀改道归纳（soft schema routing）

日期：2026-08-26。依据：前沿调研报告（子代理全文级，7 篇原文落盘 `_arxiv_tmp/`）+ 用户路线讨论拍板。
铁律执行：本文档 commit 后再动代码。

## 0. 背景：这个决策是怎么收敛的

1. 用户质疑"抽取质量上不去是不是架构路线问题" → 讨论出嫌疑二（schema-first 强约束）
2. 前沿调研判决（全文级证据）：**前沿全是混合形态，没有纯自由系统**；正确粒度是"约束强度"旋钮不是"有无 schema"开关
3. 用户指出"schema 当提示词旧管线就是"——查实：**对，8-20 就实现了**（retrieval injection，commit 02ec8a1e，现为 joint prompt 的软提示段）
4. 真差距定位：不是缺软提示，是 **软提示下游有一道硬闸**——`gate:unknown-pattern` 把 LLM 自由命名的类型拒杀，"提示软、闸门硬"名不副实

## 1. 证据表（调研核心数字）

| 证据 | 数字 | 含义 |
|---|---|---|
| EDC+R schema hint A/B | +4~5pt（WebNLG 0.746→0.794 等） | schema 当检索提示 > 无提示（我们的软提示段保留有据） |
| Intern-Atlas 封闭 7 类词表 | 生产模型分类 70.4% / 审计模型 93% | 类型混淆是任务固有难度；判定后移+更强模型有 ~20pt 空间 |
| SCION 归纳 schema 下游 | 归纳 0.68 > 人工 schema 0.56 | 归纳式不是妥协 |
| Hyper-KGGen 自由关系 | F1 ~0.60，只能模糊匹配评测 | 纯自由的代价：可测性/跨篇连接性崩 |
| 我们 EDC 探针旧数据 | 246 条自由三元组 0 条对齐到 9 类 schema | 自由命名不会自动落到 schema，必须有归纳层 |
| 我们 A6 前科 | 38 自由 pattern 仅 11 跨篇复现 | 无治理的自由 = divergence（当时没治理层，现在有） |
| DIAL-KG 设计 | 未见 schema 的候选**跳过约束**直接进动态归纳 | 正是我们要抄的闸门语义 |

## 2. 设计：拒杀改道（三处改动）

### 改动 1：`gate:unknown-pattern` 拒杀 → 改道标记
- 现状：pattern_type 不在 tbox → gate 拒杀进 dropped（reason=gate:unknown-pattern）
- 改为：**不拒**。边保留，qualifiers 打 `_novel_type: <自由命名>` 标，继续过其余 gate（binding-locality/role-legality 跳过——新类型没有 role_slots 声明）+ verifier 三检
- role-legality 对 novel 类型无 schema 可查 → 改由 verifier 的 slot_binding 检全权负责（LLM 判语义，符合铁律）

### 改动 2：commit 层放行 novel 类型边
- kernel B3（unknown pattern 拒绝）对带 `_novel_type` 标的边豁免——边进 A-box，pattern_type 存自由命名原值
- **这些边不污染 T-box**（T-box 仍是治理过的 pattern 集合）；它们是归纳层的原料

### 改动 3：演化器加归纳通道（吃 novel 类型边）
- 演化器现有的 propose_validate_failures 之外加一类输入：`_novel_type` 边
- 归纳流程（复用现有治理闸门，全部已建成）：
  1. novel 类型边按 pattern_type 分组，累积计数（cross_node 闸门照用：≥2 篇/≥2 section 复现才够格）
  2. 够格的走现有 distinctness check（embedding 相似 vs 已有 pattern）+ merge 候选
  3. 通过的 add_pattern 进 T-box（带归纳出的 role_slots/boundary——LLM 从实例生成，人审可后续）
  4. **不通过的留在 A-box 当 novel 边**（不删除——它们仍是合法知识，只是没升格成 pattern）

### 不改的（明确列出防 scope creep）
- joint prompt 的 schema 软提示段原样（调研证明提示本身有 +4~5pt 价值）
- seed pattern 集合、TOP_LEVEL_FAMILIES、分层 namespace 设计不动
- verifier 三检、fail-closed、规则层其余检查不动
- **高阶结构约束保持小而稳**（Intern-Atlas 路线：n-ary/演化关系类型表不放开自由命名——`outperforms_on`/`ablates`/`extends` 等演化关系类型仍从 schema 选，因为跨篇连接/评测靠它们稳定）——novel 类型通道只对非演化关系开放

## 3. 预注册对照实验（判决新路线是否更好）

同 5 篇验收语料，同 judge，双臂：
- 臂 A（现状）：硬闸拒杀
- 臂 B（改道）：本 DECISION 的三处改动

预注册指标（判读门柱，不许事后挪）：
| 指标 | 判 B 胜的条件 |
|---|---|
| both-pass 语义正确率 | B ≥ A + 3pt（超出单 run 噪声带） |
| 边总量（召回代理） | B 不低于 A（自由命名不应丢边） |
| novel→归纳升格数 | ≥1 且升格 pattern 被后续篇复用（通道活） |
| divergence 检查 | novel 类型数 / 篇 ≤ 8（A6 前科是 38/paper，治理闸应压住） |

失败分支预案：B 的 both-pass 反降或 divergence 复发 → 回滚改动 1（保改动 3 的归纳通道吃 dropped 边）→ 调研结论记为"不适用于本规模"。

## 4. 与 A2 的关系

- A2 演化臂（旧管线）跑完 = 路线 A 基线，不重跑
- 本改动落地后 A2 重跑 = 路线 B 对照——**同一语料两管线 A2，直接成为论文的路线 ablation**（比单路线 A2 更强的实验）
- frozen 臂两个路线共用同一个 frozen schema（seed only），无需区分

## 5. 风险与诚实预判

- novel 边的质量下限：自由命名的边过了 verifier 三检才有资格进 A-box，但"三检全过+类型自由"的边比 schema 边少一层结构保证——对照实验的 both-pass 就是测这个
- 归纳出的 pattern 质量：LLM 从实例生成 role_slots/boundary 可能弱于人写 seed——治理闸（distinctness/cross_node）挡住垃圾，但挡不住平庸；升格 pattern 的复用是唯一硬判据
- 工作量：改动 1/2 各 ~10 行，改动 3 是演化器加一个输入类型+归纳函数（现有 hev.evolution_probe 的变体），预计半天
