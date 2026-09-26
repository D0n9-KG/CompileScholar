# survey_extract 设计稿 v1（2026-09-27，修订案 B 核心新件）

> 状态：设计冻结待实现。预注册=PREREG-0.6-basekb.md 修订案 B。
> 用户总则："把综述论文的价值尽可能发掘出来为系统所用"——
> 综述从被排除者（A'' 规则）变一等公民，代价=专门抽取器。

## 1. 为什么现有管线不能直接用（失效点亲验）

- cards.py 的六标签（experiment/method/related_work/...）为"提出
  方法的论文"设计：综述无 experimental_matrix、method_identity 空、
  "本篇贡献实体"概念不成立
- deep_extract.py 的 kind 路由（section 标签→记录类型）在综述上
  退化：综述每一节都是 related_work
- 但底层机制全部复用：chunk_text 分节、quote 逐字锚定、record
  schema、postcheck 框架、views 管线入口——只换任务头

## 2. 语义分节（替代论文六标签）

综述的通用节模式（领域无关，实测稳定）：

| 语义标签 | 识别特征（cards 综述变体打标） | 路由到的记录类型 |
|---|---|---|
| taxonomy | 分类法/体系结构/"we categorize X into..." | S1 谱系边 |
| comparison | 对比表/性能比较/benchmark 汇总 | S2 领域快照 + S4 转述 |
| chronology | 时间线/发展史/"the evolution of..." | S1 谱系边 + S2 快照 |
| challenges | open problems/future directions/limitations | S3 缺口 |
| methodology | 综述自己的检索方法/纳入标准 | 不抽（PRISMA 流程非领域知识） |
| other | 其余（intro/结论） | 低密度 S4 |

打标复用 cards 的 sections 机制（LLM 一次通读打标），标签集换成
上述六语义标签。

## 3. 四类记录（S1-S4）

### S1 survey_lineage（谱系边）
```
{kind: "survey_lineage",
 subject: X, relation: extends|improves|replaces|uses|compares,
 object: Y,
 claim: "综述 A 陈述的关系（一句话）",
 quote: "<综述原文逐字>",
 epistemic: "survey-claimed",        # 新档
 evidence_basis: {survey_paper_id},
 scope_ref/target_ref: 实体解析（走同一 registry）}
```
- 与一手 lineage 边同构，进同一谱系视图；视图层分轨显示
  （survey-claimed 边=虚线/浅色，一手边=实线）
- lineage 查询默认含两种边，返回时带 epistemic 标注（用户判
  断是否信任转述链）

### S2 domain_snapshot（领域快照）
```
{kind: "domain_snapshot",
 subject: "方法族名/任务名",
 snapshot_type: comparison_table | timeline | taxonomy_node,
 claims: [{claim, quote, claims_about, conditions?}],
 as_of: <综述发表年>,                  # 时间切片免费对齐
 epistemic: "survey-claimed"}
```
- 综述对比表=分带比较矩阵的外部供血：值+条件+来源（转述）
- compare 查询命中时作 band 背景层呈现（"综述共识"轨 vs
  一手记录轨——状态分层主张的展示品）
- 与槽 5 四节报告的"比较节"同构（复用）

### S3 survey_gap（缺口，四档 absence 扩展）
```
{kind: "survey_gap",
 subject: 领域/方法族,
 gap_statement: "综述明说的未解问题（逐字锚定）",
 gap_type: open_problem | stated_future_work | noted_deficiency,
 epistemic: "survey-claimed-absence"}   # 第四档
```
- 三态 absence 现有档：实证缺席（论文自认没做）/推导缺席（语料
  内未见）/未探测。本类新增第四档：**综述判断缺席**（综述明说
  "没人做过 X"）——与推导缺席严格分开（综述判断是领域专家的
  负知识转述，语料推导是我们自己的计算；混档会污染 absence
  语义）
- find_gap 查询返回时四档分列

### S4 survey_claim（受控转述）
```
{kind: "survey_claim",
 claims_about: 被转述方实体,           # 关键新字段
 claim: "转述的具体主张/数值",
 quote: "<综述原文逐字>",
 epistemic: "survey-claimed",
 conditions?: 转述附带的条件}
```
- **不进 result 层**（用户裁定的核心规则）——A 综述转述 B 的
  结果 ≠ A 的结果；张冠李戴是 Multi-108 的实测教训
- 进受控共识层：compare 的 band 背景、card(entity) 的档案页
  （"综述共识"节）、叙事编译的素材
- 一手 result 记录缺失时它是有价值的降级证据，但 epistemic
  分轨保证永不冒充一手

## 4. 门规则（postcheck 综述变体）

跳过：result 记录的实验条件门（综述无此槽）
新增：
- S1 必须 relation ∈ 枚举 且双方实体名非空
- S4 必须 claims_about 非空（无被转述方的转述=垃圾）
- 全类：quote 锚定综述全文（复用现有锚定门）
- 重复抑制：同综述内同 (claims_about, claim-hash) 只留一条

## 5. 识别路由

- 粗抽层打标：摘要正则（survey|review|we survey|comprehensive
  overview|systematic review of）+ 引用数>80 强信号 →
  records 加 survey:true 标记
- 深抽分流：deep_extract 入口见 survey:true → 走 survey_extract
- 题内生长摸到综述同样自动分流（一次开发三处受益：热库/生长/槽5）

## 6. 实现落点

新文件 `src/kb_compiler/records/survey_extract.py`：
- 输入：texts+manifest（综述子集）+registry
- cards 综述变体（语义分节）→ chunk（复用 chunk_text）→
  按 S1-S4 分节路由抽取（每 chunk 一次 LLM 调用，同 deep_extract
  的并发结构）→ postcheck 综述门 → records_slot 兼容输出
- views 兼容：S1 进谱系视图（带 epistemic 分轨）、S2/S4 进
  新增"共识层"视图、S3 进 absence 视图（第四档列）

## 7. 冒烟验收（实现后）

- 1 篇已知综述（如 Attention 综述/XXX survey）全文跑通
- S1-S4 四类记录各抽到≥3 条、quote 锚定 100%
- views 编译不炸、typed tools 能查到 survey 边
- 人工抽读 10 条：无张冠李戴（S4 的 claims_about 全部正确）
