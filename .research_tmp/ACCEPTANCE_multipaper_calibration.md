# 多语料验收校准（路线 A）— 5 篇 × 双 judge 交叉

日期：2026-08-25 深夜。run：REWRITE_PPR_24493BE6E8C2（DQN Run 3）+ ACC_{4 篇}。全部真实 MinerU 块、arm=full、fail-closed verifier 版管线。

## 语料构成（体裁多样性）

| 论文 | 体裁 | 边数 | both-pass |
|---|---|---|---|
| DQN（Nature 2015） | ML/方法论文（含大量数学+伪代码） | 52 | 15.4% |
| μ(I)-rheology compressibility | 理论/连续介质力学 | 20 | 20.0% |
| local rearrangements 准二维 | 实验 | 36 | 27.8% |
| rotating drum 混合分离 | 建模+实验 | 38 | 23.7% |
| diffusional mixing | 短文/经典 | 35 | 34.3% |

## 总量数字（n=181 边）

| 口径 | 通过率 |
|---|---|
| both-pass（双 judge 三检全满分） | **43/181 = 23.8%** |
| either-pass | 69/181 = 38.1% |
| GLM 单 judge | 28.2% |
| qwen 单 judge | 33.7% |
| judge 一致率 | 85.6% |

## 校准结论

### 1. 与审计基线的公平对比（口径对齐后）
审计时代"语义正确率 43%"是**人工裁"可辩护"**口径。本轮可比口径：
- either-pass 38.1% ≈ 审计 43%（同量级，稍低）
- **但分母性质完全不同**：审计的 387 条含 16-19% 重复注水+verbatim 之外的编造边全数在内；本轮 181 条全部过了 fail-closed verifier+规则层，无重复无编造。

### 2. 方差与单篇效应
5 篇通过率带宽 15.4%-34.3%（2.2 倍），与文本体裁强相关：数学密集的 DQN 最难、经典短文最高。**任何单篇数字都不代表系统**——这证实了换语料校准的必要性，也意味着论文评测必须多篇+多 seed。

### 3. 与 70% 门柱的距离（诚实）
both-pass 23.8% 离 70% 很远。但三层证据说明这不是同一层的问题：
1. **探针病灶零复发+注入 pattern 全绿**（管线把"该做的"都做对了）
2. 剩余失败分布：逐条看 judge 理由，~60% 是"半合法边"（槽位 perfectionism：loser=泛指方法集/subject=上下文），~40% 是真错（方向反/neglect 被忽略/类比当比较）——后者集中在 influences/claim_relation 两个 pattern
3. judge 比人严的实证：Run1/2 人工判可辩护的注入边，双 judge 判 perfectionism fail

### 4. 每篇 321-511s（<10min），其中 DQN 511s 因 10 sections——效率项在 DQN 类长文上接近边界

## 给用户的判断材料（不替用户决定）

**管线当前状态**：架构正确、机制全通（抽取→gate→fail-closed verify→演化→采纳）、无编造无重复、高保真边子集质量高（43 条双满分边）。语义正确率的进一步提高有两条已知路径但都需要投入：a) influences/claim_relation 的判据精修（本轮数据里真错集中处）；b) 槽位 perfectionism 的口径问题（这条可能根本不是管线问题而是 judge 口径问题——需要人工抽 20 条 judge-fail 边终裁）。

**建议下一步**（与 Run 3 报告的 A/C 合并）：先做 20 条 judge-fail 边的人工终裁（我逐条对照原文判，30 分钟）——把"judge 口径问题"和"真管线问题"的比例定下来，再决定是继续修管线还是带着 23.8% 进 A2（gold 只关心方法演化边，5 篇里该类边质量最高）。
