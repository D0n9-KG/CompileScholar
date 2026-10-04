# Schema v1.2 仲裁提案（冒烟 v3 残差队列产出，2026-09-05）

来源：v3 冒烟 residual 0.231 红闸 → 按预注册冻结承诺停止 prompt 迭代，169 条 overflow 理由聚类 → 三个真 schema 信号交仲裁。这是**残差队列机制的第一次正式产出**（设计路径：残差高频项→仲裁→版本化升版）。

## S1【主信号，~17-20 条】协议/方法学事实无记录类型可承

聚类证据：
- "实验协议/复现细节（超参数、默认设置、重复次数、报告方式）不适合 config/finding/result 中的任何一种"（~8条，多篇反复出现）
- "Table of number of trials reported in related works does not fit config/result"（7条，repro 的他篇试验数汇总表）
- "数值报告方式（均值+标准误）不适合…"、"评测协议细节（last 100 trajectories after 2M steps）…"

性质：**论文级/方法级的方法学事实**（怎么评测、怎么聚合、怎么报告），不是单一 config 项（多件套描述），不是 result（无数值结论），不是 finding（非研究主张）。块2 的"可比性分带"义务（批次1回测已记录）的原料正是这类事实——现在它们只能活在 overflow 里。

候选修法：
- **(a) 新记录类型 `protocol`**：{scope_ref(方法或论文级), aspect ∈ eval_aggregation/trials/training_scale/env_setup/reporting, description, quote}。语义最干净；代价=第七类型（膨胀敏感点）。
- **(b) config 语义扩展（推荐）**：config.item 明确涵盖协议项（item 名如 "evaluation protocol"/"trials"/"reporting" 走 Stage 1.5 注册），value 允许多件套描述文本；config.role 枚举补 `protocol` 值（或允许 role 空）。依据=XBRL"概念×维度组合不造新概念"纪律：协议事实本质是论文/方法的配置面；且五元组的 aggregation/timepoint 已内嵌 result.measure，protocol 记录是它们的论文级泛化，同族概念不必立新类。
- (c) 编译层扫全文提取——违反"编译层只消费记录"分层原则，不列。
- (d) 挂 result 记录的 repeats/setup 维度——独立协议句不依附单条 result，覆盖不了汇总表场景，否。

## S2【轻量，2+1 条证据同向】config 类型语义被模型窄读为"超参数"

证据：overflow 明说 "Config record lacks field for network architecture shape specification"（2条）；canary F3（buffer size 2M 的 config）Qwen 双跑漏抽——同向信号：模型把 config 理解为"调参项"，架构形状/组件参数/缓冲区大小这类**结构性配置**被排除在外。

候选修法（推荐=文档语义澄清，非新概念）：schema 文档与切片模板中 config.item 定义明确为"超参数/**组件与架构参数**/协议项"；value 允许形状描述（"3-layer MLP, 512 units"）与公式（已有条款）。与 S1(b) 合并为同一条 config 语义扩展。

## S3【轻量，~5 条】规范性建议主张（recommendation）在 finding 里无显式身份

证据：repro 型批评论文的核心产出——"应该报告 X"/"建议采用 Y"类规范性主张——overflow 理由"Recommendation/normative claim about publication practices, not a research finding"（~5条）。finding.strength 枚举（stated/demonstrated）是证据强度轴，规范性是主张类型轴，语义错位；批次1回测已把"finding 子体裁细分（claim_type）"记录为块2义务——现在抽取层模型自己撞上了。

候选修法：
- **(a) finding 加可选 `claim_type`**（推荐）：枚举 mechanism/criticism/definition/recommendation/qualitative_ablation/observation（批次1清单+本次证据），可选字段、不参与必填闸；块2 编译层仍可重分类。给抽取模型一个显式归宿，治 overflow 犹豫。
- (b) 维持块2 义务、记录层不动：省一个字段，但残差信号会继续出现（模型已经用脚投票 5 次）。

## 纪律性残余（~10条，6%）：不修，观察

引用句背景/定性比较/caption 描述——现有规则明文覆盖（不抽不进 overflow）但模型未全遵守。按 v3 冻结承诺不动 prompt；v1.2 重跑后若此类占比上升再议。诚实标注：residual 闸 0.10 意味着 v1.2 后仍需要 S1/S2/S3 修复吃掉 ~13pp 才能过闸——聚类核算显示量级刚好够（~20+3+5≈28条/169），**闸维持 0.10 不动，不为自己留后路**。

## 判读后的执行单（批准即动）

1. schema.py：config 语义扩展（S1b+S2 合并条款）+ finding.claim_type 可选枚举（S3a）→ SCHEMA_VERSION 1.2
2. 切片模板同步（config item 定义行 + finding claim_type 行）——模板是 schema 的渲染面，随仲裁一并生效
3. 冒烟 v4 重预注册（门柱全组不变）→ 重跑 Qwen 单臂
4. 金丝雀 F3 漏抽列入 v4 观察项（config 语义扩展的对症验证点：buffer 2M 应被抽出）
