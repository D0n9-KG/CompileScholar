# 抽取栈重写验收 — Run 2（判据修复+放宽 gate 终验口径）— 未过门柱

日期：2026-08-25。run：REWRITE_PPR_24493BE6E8C2（Run 2 覆盖 Run 1 bundle）。预注册门柱 both-pass ≥70%。

## 数字

| 指标 | Run 1（37 边） | Run 2（66 边） |
|---|---|---|
| both-pass | 9/37 = 24.3% | **9/66 = 13.6%** |
| 单 judge | 37.8% | 21.2% |
| 一致率 | 73% | 85% |
| 注入 pattern | 4/4 好 | **5/5 采纳、0 丢弃、judge 判 ablates 3/3=100% pass** |
| 效率 | 261s | 260s |

**未过门柱，且整体率下降。** 下降机理已定位（非回归）：

## 归因（三条，相互独立）

### 1. 分母效应：Run 2 边数 37→66
放宽 gate 让 Run 1 被严格规则拦掉的 25 条误杀回流 + Run 2 本身多抽（判据 prompt 让 LLM 更敢抽 METHODS 段）。**新增的 29 条边质量差**：both-pass 增量为 0——回流的边几乎全军覆没。

### 2. 真病灶：METHODS 段（n3/n4）的数学定义/算法描述文本
失败 47 条 either-fail 的分布：
- influences 18/24 fail：judge 理由高度一致——"定义不是依赖"（"in which each sequence **is** a distinct state"、"p **is** a policy"、"c was **set to** 0.99"、"Q* **obeys** an identity"）。判据修复对 front_matter/intro 的属性陈述生效（Run 1 该段失败模式未复发），但 METHODS 段的**数学定义句**是新一类：LLM 把"定义性恒等式/参数赋值/策略映射"全打成 influences
- defines 11/11 全 fail：证据全是 Algorithm-1 伪代码行（"Initialize action-value function Q with random weights"）——**初始化步骤/操作步骤不是定义**，且 Run 1 时代 verifier 曾正确拒掉同族边（gate 审计确认），Run 2 里它们穿过了
- composed_of 8/13 fail：网络架构描述（"the input consists of 84×84×4"——consist of 结构组成判对了但角色错）+ "stored in D"（存储关系非组成）

### 3. Run 1→Run 2 的口径差异
Run 1 的 24.3% 建立在严格 gate 把 25 条（含大量 METHODS 段）拦掉的基础上——**Run 1 的数字偏乐观**。Run 2 是全量真实口径。

## 诚实的结构性判断

1. **联合抽取治好了它的靶子**：探针病灶（槽位绑定/跨句偷换/范围泛化/ablates 被抢）在两轮 run 里零复发；注入 pattern 5/5 全好、ablates judge pass 率 100%。**架构换对了。**
2. **剩余失败全部是 pattern 语义选择问题**（数学定义句 vs 依赖、算法步骤 vs 定义、架构参数 vs 组成），不是绑定/结构问题。这不是 prompt 再加判据能根治的——Run 2 已经验证了一次判据修复对新一类文本无效（打地鼠）。
3. **需要机制层方案**（Run 3 方向，需 DECISION）：
   - 候选 A：**LLM 语义型 pattern 判型器**——每条边单独问一次"这个 pattern 的 boundary 适用于这个句子吗"（成本 +1 调用/边，或批量）
   - 候选 B：**数学句分流**——METHODS 段的公式/定义/伪代码文本走专门 prompt（seed pattern 里 constitutive_law/defines 本来就为它们设计，但判据没教过"算法步骤不是定义"）
   - 候选 C：verifier 把三必检真正用起来——Run 2 里 verifier 对这批边判了 keep（它有三必检 prompt 但没拦住），查 verifier 判 keep 的原因（可能又是截断默认 keep 的老 bug）

## 处置

- 验收未过 → 抽取栈重写保持"进行中"，不标完成
- Run 3 前先做小诊断：抽 verifier 对 11 条 defines 边的实际 verdict（确认是 verifier 放水还是根本没问对）
- 修复成本预算：+1 调用/边 ≈ +60 调用 ≈ +2-3min/篇，仍在 <5min 内
- 多 seed 仍未做（方差前科），单 run 数字本来就要打折
