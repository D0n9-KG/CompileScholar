# 抽取栈重写验收 — Run 3（fail-closed verifier）— 仍未过门柱，但瓶颈已完全移位

日期：2026-08-25。run：REWRITE_PPR_24493BE6E8C2（Run 3 覆盖）。预注册门柱 both-pass ≥70%。

## 数字

| 指标 | Run 1 | Run 2 | **Run 3** |
|---|---|---|---|
| 边数 | 37 | 66 | **52** |
| both-pass | 24.3% | 13.6% | **15.4%**（8/52） |
| 单 judge（GLM/qwen） | 37.8% | 21.2% | 23.1%/25% |
| 一致率 | 73% | 85% | **83%** |
| 效率 | 261s | 260s | **304s**（分批 verify +调用，仍 <5min） |

**未过门柱。**

## Run 3 的真实进展（verifier 修复被验证）

Run 2 的 29 条 judge-fail 边重放诊断：修复后 verifier 判 21 drop + 4 retype:defines + 4 keep——**verifier fail-closed 修复实际有效**（Run 2 里这批边全放行）。Run 3 边数 66→52 正是被拦掉的量。

但 both-pass 只从 13.6%→15.4%：**新一批"可辩护但不完美"的边从 METHODS 段持续产出**（failure reason 从"算法步骤当定义"变成了更深一层的"subject 绑到 MDP 上下文而非被定义项"/"game score 是输入不是指标"）。

## 逐条对照后的结构性认识（三轮收敛的判断）

### 失效模式的层级迁移
- 探针时代：**绑错槽位/跨句偷换**（结构性）→ 联合抽取治愈
- Run 1-2：**算法步骤当定义/属性当依赖**（pattern 语义选择粗错）→ 判据+fail-closed verifier 治了大半
- Run 3 剩余：**半合法边**——句子确实相关、pattern 选择可辩护、但某个槽位绑得"不完美"（loser=领域泛指而非具体方法、subject=上下文而非被定义项、metric=输入而非指标）。这类边人也会争论。

### 方法学诚实声明（重要）
1. **双 judge both-pass 是极严口径**：要求两个不同家族的 LLM 对一条边同时给满分三检。both-pass 15.4% + 一致率 83% 的组合说明两 judge 严格但稳定。**单 judge 23-25% 与审计时代 43% "可辩护"口径更可比**（当年是我人工裁"可辩护"，比 judge 三检全过宽得多）。
2. **预注册门柱 ≥70% 定在 both-pass 口径上过于乐观**——当时没有意识到 DQN 这种 Nature 论文的 METHODS 段（数学定义+伪代码）对任何抽取系统都是最难文本。**门柱本身需要重新校准，但这是用户的决定不是我的**——本轮如实报告不挪门柱。
3. 对照信息：抽取器只产出 52 条边里 8 条双满分——但**这 8 条的质量是真高**（3 条 ablates 结构完美、2 条 defines 是教科书级定义句、3 条 influences 是真依赖）。低通过率 ≠ 低价值：现在的 A-box 是"少量高保真边"，与审计时代"387 条 43% 可辩护"是相反的取舍方向。

## 建议的下一步（三选一，需用户拍板）

**A. 换验收语料重新校准（推荐）**：DQN 全文里 front_matter+intro 段的边质量明显高（Run 1 该段 both-pass 率远高于 METHODS 段）。用 3-5 篇论文的完整 run 看整体率，而不是单篇最难文本定生死。同时把 70% 门柱的口径问题（both-pass vs 单 judge vs 分段）摆到桌面。
**B. 继续攻 METHODS 段**：上"数学句分流"（公式/伪代码/定义句走专门 prompt 或直接跳过抽取——很多 KG 系统对伪代码段就是跳过的）。预计能把 52 边里的最难 15 条干掉或转正。
**C. 接受当前质量进入下一阶段**：抽取质量优化到边际收益递减，A2 跨篇实验和 gold 重建（按 coverable 原则从综述构造）可以先动——gold 本来就只关心方法演化边（outperforms_on/ablates/extends 类），而这正是当前管线质量最高的部分（注入 pattern 5/5、ablates 100%）。

## 附：Run 3 注入 pattern 边的细节

outperforms_on 本轮 0/2 both-pass，但失败原因是**槽位 perfectionism**（loser="best existing reinforcement learning methods"被两 judge 一致要求"具体方法名"；metric="game score"被一致判"是输入不是指标"）——注意这两条边在 Run 1/2 的人工判读里我都判过可辩护。**judge 口径比人严**这个事实本身就是 Run 3 最重要的发现之一。
