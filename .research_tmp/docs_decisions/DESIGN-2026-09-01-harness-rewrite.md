# DESIGN-2026-09-01：抽取 harness 重构（两遍抽取：轻装抽取 + 定向补漏）

状态：**设计稿，待用户过目后动工**。动工前不写代码。
前置：裸 prompt 实验（harness_ablation.py）+ 前沿调研（17 来源）+ gold v1.1 冻结（R9）。

---

## 1. 证据链（为什么重构）

**实验证据（本 session，金丝雀 S1）**：

| 配置 | 正向召回 | 陷阱率 | verbatim |
|---|---|---|---|
| 当前 harness（22.2k prompt） | 4/6 | 0/3 | 0.90 |
| 裸 prompt（13.7k，砍 8.5k 固定规则） | **6/6** | 0/3（持平） | 0.90（持平） |

注意：裸 prompt **保留了 schema**（15-pattern 全量渲染，含 [boundary: ...]）——实验隔离的是
**固定规则层**（SLOT-BINDING DISCIPLINE + PATTERN-SELECTION CRITERIA + 9 条 HARD RULES，
共 ~8.8k chars），不是 schema 层。schema 判据无罪，规则层是召回抑制器。

**文献证据（调研 17 来源交叉）**：
- Format Tax：格式指令在解码前吃掉精度
- Capacity Not Format：schema 复杂度惩罚 p<0.0001——prompt 越长，容量竞争越烈
- EDC 阈值：schema ≤7 类全放没事、45 类崩；38+ pattern 在崩区
- **四个直接竞品全部抛弃"全量判据常驻 prompt"**：DIAL-KG 检索 top-30、EDC schema-free+后置
  概念化、AutoSchemaKG 后置概念化；有效路径=检索式注入 + EDC+R 定向第二遍 + 验证后置

**四轮迭代的教训**：轮次 4（"宁多勿漏"指令）负结果——prompt 层修召回的措辞敏感度极高；
真正有效的杠杆是**结构**（第二遍定向补漏），不是指令措辞。

## 2. 当前架构解剖（改动前的精确地图）

每 chunk 流程（`extract_section`，extraction_agent.py:916）：

```
plan → execute(_run_hg_node_joint: 13.5k-22k prompt, k=3 投票)
     → _deterministic_gate（结构规则：verbatim/绑定局部性/role 声明）
     → verify（LLM 批量 8 条/批：CHECK 1 证据支持 + CHECK 2 槽位绑定 + CHECK 3 极性方向）
     → fix（rolefix/retype/drop）
     → commit_edges（kernel 两阶段提交）
```

prompt 尺寸分解（参考种子 15 pattern，中位 chunk ~8k）：

| 组成 | 尺寸 | 后置承载情况 |
|---|---|---|
| schema（taxonomy 树 + boundary） | 5.4k | **保留**（这是 schema 的本职） |
| SLOT-BINDING DISCIPLINE（11 行） | ~1.6k | verifier CHECK 2 已承载 → **删** |
| PATTERN-SELECTION CRITERIA（48 行） | ~4.5k | schema [boundary:] + verifier CHECK 1/3 已承载 → **删** |
| Entity types（8 行） | ~0.7k | 无后置承载，输出需要 node type → **保留** |
| HARD RULES ×9 | ~2.0k | R1 verbatim=gate；R2 同句绑定=gate；R3 断言=verifier 1(e)；R4 极性=verifier C3；R6 role 声明=gate；R7 qualifiers=gate；R9 setup 句=verifier 1(a) → 仅 R5 n-ary 合并、R8 新 pattern 命名无后置 → **留这 2 条** |

已有资产（不用重造）：
- `_retrieved_schema_prompt`（hypergraph_extractor.py）：top-K 检索 + 拓扑保持扩展 +
  compact 渲染带 boundary[:100]——检索式注入**已存在**，K=30，参考种子 15 pattern 时走全量分支
- verifier `_VERIFY_PROMPT`：判据后置**已存在**（CHECK 1/2/3 + per-edge pattern desc/boundary）
- 投票 `_vote_chunk`：方差压制已存在（k=3，可开关）

## 3. 新设计：两遍抽取（EDC+R 形态）

### Pass 1 —— 轻装抽取（召回优先）

```
[task 一句话] + [schema（检索式，现状不动）] + [chunk 文本]
+ [输出 JSON 格式] + [Entity types] + [3 条结构规则] + [3-5 个真实示例]
```

- 3 条结构规则（仅留无后置承载的）：①evidence_span 原文最小连续句 ②n-ary 并列参与者进一条边
  ③优先 schema pattern，新 pattern 用 clean snake_case
- **示例是新增的关键件**：形式教学（输入句 → 输出边 JSON）比规则教学有效（文献一致结论）
- 示例来源纪律：**只能取自非 gold 论文**（金丝雀/gold 五篇之外的语料），防评测污染
- 删除的判据不丢失——它们在 schema [boundary:]（仍在 prompt 里）和 verifier（后置）里

### Pass 2 —— 定向补漏（EDC+R 第二遍，schema 判据的正式位置）

```
[chunk 文本] + [第一遍已抽边清单（pattern_type + evidence 前 60 字，防重抽）]
+ [该 chunk 检索 top-10 pattern，带完整 boundary] 
+ [指令：逐 pattern 检查 chunk 中是否有第一遍漏掉的实例，只输出新边，同 JSON 格式]
```

- 每个 chunk 一次调用，产出走同一 gate + verify（后置判据统一把关）
- 去重：沿用投票 key（pattern_type + 参与者 surface 集 + evidence 前缀规范化）
- **这是 schema 价值的正式落位**：boundary/描述/拓扑在第二遍完整注入——演化出的 pattern
  越准，第二遍的检索池和判据越好，召回越高。H1（in-loop 演化收益）第一次有了
  **可测量的机制通路**：evolving 臂的 schema 长出好 pattern → pass 2 检索命中 → gold
  演化族 recall 提升。frozen 臂 schema 不长 → pass 2 检索池不变。这在旧架构里不存在
  （schema 只是 pass 1 的被动 prompt 块，判据被 8.8k 规则层淹没）。

### 投票去留（决策点，倾向关）

- 现状 k=3：金丝雀逐字节复现，但 R6 DQN 召回反降（投票管"抽到的稳定"，可能压"抽到的多样性"）
- 新架构下投票的职责被拆分：漏抽由 pass 2 管，错误由 verifier 管
- 成本账：当前 3(投票)+verify ≈ 5-6 调用/chunk；新 1+1+verify ≈ 4-5 调用/chunk——**更便宜**
- 方差回归由多 seed 纪律（≥4 seed）承担，单 run 方差不再是需要压制的目标
- 实现：保留投票代码和开关（VOTE_SAMPLES 环境变量化），默认关

### verifier / gate：不动

已经是判据后置的正确形态。唯一增量：CHECK 1(e) 的断言判据是陷阱率的最后防线，
重构后第一遍没了 ASSERTION MOOD，**陷阱率防线完全押在 verifier 上**——这是预注册的
验收观察点（见 §5）。

### 模式开关（论文消融三列直接用）

`HARNESS_MODE` 环境变量，三档：
- `heavy`：现状（22k prompt + 投票）——对照列
- `slim`：仅 pass 1 轻装——隔离"瘦身本身"的贡献
- `slim+refine`：完整新架构——主方法列

三列正好对应论文的 harness 债务消融表（记忆下一步序第 4 条）。

## 4. 改动清单（动工时的精确范围）

| 文件 | 改动 | 性质 |
|---|---|---|
| `hypergraph_extractor.py` | 新增 `_SLIM_PROMPT`（pass 1）+ `_REFINE_PROMPT`（pass 2）+ 示例块（硬编码 3-5 条，来自非 gold 论文）；`_run_hg_node_joint` 加 HARNESS_MODE 分支；旧 `_JOINT_PROMPT` 保留（heavy 档用）；投票开关环境变量化 | 主体 |
| `extraction_agent.py` | 不动（pass 2 在 execute 内部完成，gate/verify/fix 对两遍的并集统一把关） | 零改动 |
| `hypergraph_schema.py` | 不动（schema 渲染、boundary、检索全复用） | 零改动 |
| 评测侧 | R6 bundles 用新 gold（109 边）重算 L2/L3 → 0.137' 调整基线（纯重算无新抽取，分钟级） | 仪器复用 |

## 5. 验收预注册（动工后按此判停，标准不放松）

1. **金丝雀快回路**（~11min/轮）：S1/S2 正向召回 ≥ 现状、**陷阱率 ≤0.17（停止线原值）**、
   verbatim ≥0.85。陷阱率若破线 = 判据后置失败 → 最小回退（把 ASSERTION MOOD 单条加回
   pass 1，不是恢复整个规则层）
2. **新基线五篇**（带 register_corpus_papers 注册——R8 教训）：新 gold(109) L2/L3 折算
   演化族 recall，与 0.137'（R6 重算值）对比 = harness 债务定量
3. **停止线不变**：0.35。到达或判停 → 判决实验
4. 效率：单篇墙钟 ≤8min 带内（效率是一等设计目标）

## 6. 风险与回退

| 风险 | 概率 | 缓解 |
|---|---|---|
| 陷阱率回升（断言判据离开 pass 1） | 中 | verifier 1(e) 兜底；破线则单条回加（§5.1） |
| pass 2 重抽第一遍的边（假新边） | 低 | 输入带 pass 1 清单 + 后置 key 去重 |
| 示例污染（来自 gold 论文） | 低 | 硬约束：示例只取自非 gold 语料 |
| slim 档召回反降（规则层有真贡献） | 低（裸 prompt 已证伪） | 三档开关保留 heavy 对照 |
| 投票关闭后 run 间方差回归 | 高 | 接受——多 seed 纪律本来就要求 ≥4 seed；报告时如实呈现 |

## 7. 与论文叙事的接口

- 贡献位重申：schema 贡献在演化与下游（pass 2 检索池），不在单篇抽取质量
- harness 债务表（heavy/slim/slim+refine 三列）= 论文消融表一节
- "四个竞品全部抛弃重 prompt"的调研 = related work 的直接弹药
- Format Tax / 容量竞争 = 为什么轻装有效的理论解释
- 判据后置（verifier 统一把关）与"规则只用于结构/确定性任务"设计铁律（记忆）一致：
  复杂语义判据交给 LLM 后置判断，不做成前置硬规则
