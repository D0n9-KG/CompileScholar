# Plan: ablation 预注册（第 3 件证据，动手前定）

## 目标
证明重锚后的主卖点"n-ary 超边语义保真 + schema 并进演化 + agent mutable memory 三点同时"有可证收益。守 P0 教训(结构信号臂无显著增益)——不赌 aggregate 下游 ERR,锚机制层指标 + 长尾 + qualitative。

## 主 ablation 3 个（必跑 ≥4 seed paired bootstrap 95%CI）

### A1. n-ary 超边 vs binary 拆开（主卖点 1，最稳）
- **baseline**: binary pairwise(Intern-Atlas 式 / 我们把 n-ary 拆 pairwise)
- **指标(机制层)**:
  - 富拓扑直读质量(7类边 precision/噪声率)——memory 已实测 win(735→595 噪声降)
  - 语义保真度(method+多params+phenomenon 多元关系完整度, binary 拆开丢"同一law绑这些param"结构)
  - co-occurrence 噪声率
- **判定**: n-ary 富拓扑直读质量 > binary co-occurrence(已有 win 实测, head-to-head 巩固)
- **风险**: 最稳, memory 已证

### A2. schema 并进演化 vs static schema（主卖点 2）
- **baseline**: Intern-Atlas 式 static schema(seed 12 不演化, A-box 正常写)——**不是 frozen 完全不写**(那样无意义), 是 T-box 锁死 seed
- **指标(机制层 + 长尾, 非 NMR/ERR)**:
  - schema 紧凑度(pattern 数 / redundancy / compound 占比) before/after
  - 新 pattern 后续复用率(utility, 被 ≥2 篇后续抽取复用)
  - 演化链追溯质量(Intern-Atlas 式 evolution chain ground-truth 对齐)
  - 长尾新方法族命中(frozen 漏抽 / 演化命中的 case, P0 已有 failure case 可复用)
- **判定**: 演化在 schema 紧凑度 + 新 pattern 复用率 + 长尾命中 > static; aggregate 下游 ERR **可能无显著**(P0 风险, 不赌)
- **⚠️ 诚实负面 fallback**: 若 aggregate 无显著增益, 报"schema 并进演化在 aggregate 质量无显著, 但 schema 紧凑度/长尾召回/演化链可追溯有 X 改善"——卖点转"有控防膨胀 + 可追溯 + 长尾"非"质量更好"

### A3. 富拓扑直读 vs co-occurrence（n-ary 红利, 技术细节）
- **baseline**: co-occurrence 猜(Higher-Order 2601.04878 式)
- **指标**: 7 类富拓扑边 precision/噪声率 + 抽取稳定性
- **判定**: 直读 > co-occurrence(memory 已 win)

## 次要 ablation（探索性, 单 seed, 不耗功率）
- 抽取: plan-execute-verify 子流程可省(verifier 必要性 / plan 必要性)
- 对齐: embedding+judge vs surface-only vs LLM-only / 5-outcome vs 二元 / 跨域隔离 vs 不隔离
- agent mutable memory vs on-demand(DocTrace 式)

## 统计检验
- 主 3 ablation × ≥4 seed paired bootstrap 95%CI(复用 P0 脚本 .research_tmp/eval_*.py + P0_multiseed_results.md)
- **judge 模型交叉**: GLM-5 + qwen3.5 各判一遍, 不一致进 HITL(避免单 judge 偏置, P0 教训)
- 多 seed 非确定性: V4-Flash 单篇 20-65 波动(memory), 必须 ≥4 seed

## baseline 选法(评判子代理建议)
- **主对照 Intern-Atlas binary+static**: n-ary+schema并进演化 vs binary+static 在 method evolution chain 重建增益——比"frozen vs 演化"更能凸显组合价值
- 富拓扑对照 Higher-Order 2601.04878(co-occurrence, 干净)

## qualitative + 长尾(对冲 aggregate 稀释)
- P0 failure case 复用: frozen 在 segregation flux M17/M23-26 等长尾新方法族漏抽, 演化命中的具体 case
- 演化链追溯 case: 给出方法 A→extends→B→improves→C 的可追溯路径, static schema 重建不出的

## 诚实风险(守顶会)
- P0 "结构信号臂无显著增益"极可能在 A2 重现 → A2 锚机制层 + 长尾, 不赌 aggregate; 备负面 fallback
- A1 最稳(已 win), 是兜底卖点
- 若 A1/A2/A3 全无显著 → 创新点降级为"n-ary 保多元结构"概念论证 + qualitative, 不强赌机制

## 预注册承诺
- 主 3 ablation 指标 + 判定标准 + 统计检验在动手前定死, 跑完照判定报(不事后挑有利指标, P0 教训)
- 负面诚实报(不藏)

## 等三件证据齐 → 定方向 → 修设计断点 → 重写 plan → 实现
依赖: MINDSET 核查 + Intern-Atlas 详读(子代理在跑) + 本 ablation 预注册。三齐后:
1. 若 MINDSET 不占组合真空 + Intern-Atlas 确认 binary+static → 重锚方向定, 修设计断点, 重写 plan
2. 若 MINDSET 占组合真空 → 致命, 换方向(重新找真空)
