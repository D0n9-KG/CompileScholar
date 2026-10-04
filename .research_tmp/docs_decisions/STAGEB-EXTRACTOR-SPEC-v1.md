# Stage B 抽取器重写规格 v1.3（定稿，2026-09-06）

状态：**已拍板定稿**（v1.0：§7 全按推荐批+"按实际效果迭代优化"；v1.1：gold 回测三批合并仲裁 RC1-RC8+治理两条；v1.2：冒烟 v3 残差队列仲裁 S1-S3；v1.3：颗粒流第二域 pilot 仲裁——notation 类型+图形数据 multimodal 后置，拍板记录见 §7）。回测三批针对 v1.0 文本（SCHEMA-BACKTEST-MERGED.md）；v1.2 仲裁输入=冒烟 v3 overflow 聚类（SCHEMA-V12-ARBITRATION.md）；v1.3 仲裁输入=颗粒流 pilot 残差聚类（GRANULAR-PILOT-PREREG.md 判读节）。

**迭代哲学**（用户拍板原话的落位）：schema 不是 immutable——缺口由三个实测信号驱动迭代（gold 回测缺口 / 残差队列高频项 / 冒烟失效归因），但修订一律走仲裁通道+版本化升版，不允许抽取现场或代码里静默改语义。

**治理对称条款（v1.1 新增）**：加法通道（三信号→仲裁→加概念）必须配减法议程——每次版本仲裁前跑**用量审计**（统计脚本：逐字段抽取填充率+下游视图引用率），低使用字段的删除候选清单与缺口清单**同议程仲裁**（`cannot_tell` 零使用是第一个在案候选）。仲裁人相同、证据门槛相同、方向相反；治的是"报警驱动加法、沉默掩盖减法"的注意力不对称。配套：schema 体积与单次 prompt 体积解耦（§3 Stage 2 切片注入），冻结层增长不吃抽取调用的容量预算。

## 0. 定位与范围

- 蓝图五块之**块1（记录层）**：记录类型逐字段 schema、XBRL 维度词表、Cochrane 五元组映射、两遍抽取管线、PaperScope 三新工作项并入、确定性后检。
- **不覆盖**（后续块各自出规格）：四视图编译、typed tools、跨记录一致性验证（三层验证第三层）、失效代数。
- 从零重写纪律：`src/granular_agent/` 旧栈 frozen 零 import；只带走设计模式（§5）。
- 域无关纪律：维度**名**、关系**词表**、记录**类型**固定域无关；维度**值**语料生长；prompt 不写 RL 特例。

## 1. 记录类型与逐字段 schema

**四类核心**（config/result/lineage/finding）+ **三类辅助**（absence/shift 拍板保留；notation v1.3 颗粒流仲裁新增）+ overflow（残差，§3 Stage 2）。

### 1.0 公共字段

| 字段 | 说明 |
|---|---|
| `id` | hashlib 稳定哈希（paper_id+kind+内容指纹）；禁内置 hash() |
| `paper_id` | 指向 Stage 0 manifest |
| `kind` | result/config/lineage/finding/absence/shift/overflow |
| `quote` | 原文逐字引文，**≤40 词**（拍板 F）；数值类记录 quote 必须逐字含数值与指标名；表格行=行头+列头+单元格 |
| `loc` | {section, char_start, char_end}，锚 mineru 解析文本；后检回源验证 |
| `epistemic` | stated / demonstrated / cited |
| `dims` | 维度绑定（§2） |
| `quality_flag` | 可选（RC8）：suspect_binding / parse_warning / figure_only——Stage 2/3 打标，渲染层必须带出（治"表头错位警示不随记录走"） |

### 1.1 result——Cochrane 五元组落位

`method_ref{surface,canonical}`；`measure{metric, value, unit, direction(higher/lower_better), aggregation, timepoint}`（五元组=测量工具名+标度+方向+聚合方式+时点）；`role ∈ main_result/baseline_comparison/ablation`；`delta`（可选，文中明示的增减+基线 method_ref；跨记录 delta 计算归编译层）；`dims`；`quote`。

**消融并入（拍板 A）**：ablation 不独立成类 = `role:ablation` + `dims.variant:"−组件名"` + `delta`。

### 1.2 config（v1.2 语义扩展：S1+S2 仲裁）

`method_ref`；`item`（**超参数/组件与架构参数/协议项**——buffer size、network architecture、evaluation protocol、trials、reporting 等，协议项 item 名走 Stage 1.5 注册）；`value`（带单位；协议项可为多件套描述文本；公式逐字）；`applicability`（dims 子集）；`role ∈ best_reported/default/used_in_experiment/protocol`；`epistemic`（§1.0 公共字段在 config 的显式落位，v1.2.1 模板同步：**cited=转述他篇的配置**——综述对比表中的他法架构/参数，冒烟 v4 残差实锤该槽位缺失时模型只能 overflow）；`quote`。

依据：残差队列主信号（~20 条协议/方法学事实无家）+ config 窄读证据（架构形状 overflow 明说 + 金丝雀 F3 buffer 2M 双跑漏抽）。XBRL 纪律"概念×维度组合不造新概念"：协议事实是论文/方法的配置面，不立第七类型。

### 1.3 lineage

`from_method_ref` / `to_method_ref`；`relation ∈ {extends, improves, uses, component_of, replaces, compares_with, motivated_by, generalizes, concurrent_with}`（**v1.1 九词封闭集**：+generalizes 理论收编/统一（T05-m4"把C51/QR-DQN/IQN收进框架"），+concurrent_with 对称同期边（C07"TD3 与 SAC 互为 concurrent work"）；**否定词悬案正式关闭**——三批回测各专列 8 例否定表述独立零需求，全部被 result.delta 负值/finding/absence 三通道承载）；`scope`（dims 子集：关系成立条件，条件性一等公民）；`evidence_basis ∈ explicit_claim/citation_context`；`quote`。

canonical 解析前置到 Stage 1.5（混合臂 F12 教训：lineage 必须编译期直接可 join）。lineage 端点放开到任何 entity_type 的注册实体（component_of from=n-step 本就强制机制级实体可注册，RC2）。

### 1.4 finding

`claim`；`scope_ref`（**可空；可指向任何 entity_type 的注册实体**——领域级批评如"RL 惯例训练测试同环境"绑定 practice 实体，RC2）；`target_ref`（可选，RC1：被批评/反驳的注册实体——**主张级 contradicts 边是跨篇边，单篇抽取的记录层原理上造不出（抽 Fedus 时 Z&S 的 finding id 不存在），记录层只提供 join 键，冲突检测义务归块2三层验证第三层**）；`condition`（dims）；`strength ∈ stated/demonstrated`；`claim_type`（**可选，v1.2/S3**：mechanism/criticism/definition/recommendation/qualitative_ablation/observation——主张体裁轴，与 strength 证据轴正交；不参与必填闸，块2 可重分类）；`quote`。

### 1.5 absence——Cochrane 三态（拍板 B）

`subject`；`missing`；`absence_type ∈ {not_reported, explicitly_stated, cannot_tell}`；`evidence`（穷尽性依据；not_reported 型以此承担举证责任）；`quote`（explicitly_stated/cannot_tell 必填）。

边界：单篇 absence=抽取层；跨篇"矩阵格子为空"的覆盖核算=编译层推导型 absence；两者来源标记必须可区分，不混渲染。

### 1.6 shift（拍板 E：作抽取类型）

`from_state`；`to_state`；`driver`（because Z）；`scope`；`time_range`（原文措辞不硬解析）；`source_type ∈ intro_narrative/related_work/discussion`；`quote`。语义=单篇论文自己的领域轨迹叙事主张；跨篇叙事链=编译层职责。

### 1.7 notation（v1.3，颗粒流 pilot 仲裁）

`symbol`（符号逐字，如 μ_p）；`quantity`（物理量/对象名）；`definition`（定义内容，忠于原文）；`unit`（可空）；`scope_ref`（所属方法/模型，可空）；`quote`（逐字必填）。

语义：符号→量的定义，公式承重域（物理/化学/数学）的结构性内容。价值：config/result 里公式型取值（μ(I)=μ_s+Δμ/(I_0/I+1)）的解释基础；旧 LogicKG 实测过"公式符号定义资产"价值。证据：颗粒流 8 篇 pilot ~15 条同根因残差聚类（d: particle diameter / e: restitution coefficient…），模型正确拒绝将其塞进 finding（定义非研究主张）。路由：nomenclature/notation/symbol 专路 + method/theory 段 + 全量兜底切片。

## 2. XBRL 维度化条件：维度词表（拍板 D）

维度**名**固定封闭；**值** explicit（枚举域，语料生长，机器校验）或 typed（开放带单位）。

| 维度名 | 类型 | 语义 | Cochrane 对位 |
|---|---|---|---|
| `subject` | explicit | 被评测对象（family→member 两级：Atari→Breakout） | Participants |
| `setup` | explicit | 协议/环境配置条件（含定性硬件条件） | Setting |
| `budget` | typed | 规模数值+单位（frames/steps/hours/FLOPs，含算力预算） | — |
| `variant` | explicit | 方法变体/组件开关；**合法域按方法族 scoped**（closed hypercube） | Intervention 细分 |
| `repeats` | typed | 种子数+聚合（聚合单一真源在 measure.aggregation） | Results 精度 |
| `hyperparam` | typed | **数值级超参条件轴（RC3，v1.1）**：{item, value, unit}——item 名走 Stage 1.5 注册（explicit item + typed value），按方法族 scoped；buffer 1M→10M、atom 数 N、replay ratio 档位这类"取值作自变量"的条件不再退入 finding 自由文本；与 config 记录 (item,value) 同构，编译层可交叉验证"result 条件轴 vs config 推荐值" | —（ML 域特有轴） |

- measure 内嵌 result.measure，不在 dims 双写。
- 生长纪律：Stage 1.5 注册+仲裁升版；Stage 2 不得发明 explicit 新值（新值→warning→仲裁队列）。
- **registry 实体粒度条款（RC2，v1.1）**：注册实体带 `entity_type ∈ {method, mechanism, practice, out_of_corpus}`——target network/ε-greedy/sticky-actions=mechanism；"RL 评测惯例"=practice；Sutton 1988/DDPG/OTRainbow=out_of_corpus（经 citation_context 注册）。文外实体增可选 `origin_year_cited{value, source_paper_id, quote}`（RC4：值=引文逐字作者-年份如"(Kielak, 2020)"，非 LLM 猜测，不违反 year 纪律；as-of 时点判定的要害字段，批次2 P×4）。批次2实测：无此条款时序题 E 率 88.4%→约50%。
- namespace 分层（带走旧栈跨域定案）：维度名 global；成员值先进 domain namespace；**≥2 域复用才升 global**。
- 版本化冻结：词表 vN 每次运行声明用哪版，跑中不长。

## 3. 两遍抽取管线（拍板 C：语料级两阶段）

```
Stage 0  元数据 manifest（非LLM）
Stage 1  skeleton pass（每篇1调用，全文）→ paper_card
Stage 1.5 语料注册（少量调用）→ method registry + dimension vocabulary vN
Stage 2  slot pass（每篇×section chunk，注入 card+registry+词表）→ 记录 / overflow
Stage 3  确定性后检（非LLM）→ repair-or-drop + 日志 + 残差率统计
```

- **Stage 0**（新工作项1）：`paper_id → {title, arxiv_id, arxiv_year, venue, venue_year, openreview_id, authors, affiliations, doi, year_source, source_file, source_version, parser, flags}`（authors/affiliations=RC6：arXiv/S2 非 LLM 零成本，机构分组负载 C08"DeepMind系"；API 拿不到的诚实置 null+flag，不编造。doi/year_source=v1.2.1 patch：期刊域语料的时序锚与来源审计——非 arXiv 域 chronology_anchor=db_year，颗粒流 pilot 引入；元数据层扩展，记录层语义零改动）。**记录层无 LLM 猜的 year**；渲染双年份（venue_year 主）；文内自述与 manifest 冲突→flag 进审计队列。`source_version` 同时是评测协议"材料源=gold 源"审计的数据基础。
- **Stage 1** paper_card：{method_identity（canonical+别名+是否本篇贡献）, 自述贡献, 实验矩阵+图表清单扫描（表格 subject×measure×variant 格子清单；图/表存在性+caption 逐字——RC7 最低配，图-only 数据本体记 known limitation）, 维度成员候选, related_methods}。表格扫描放 skeleton：先全文视野建"哪些格子存在"地图，Stage 2 逐格填。
- **Stage 1.5**：合并全 paper_card → 方法注册表 + 域词表 vN。
- **Stage 2**：quote-first 指令（新工作项2）——**先逐字抄原句/表行，再从抄件填字段**，字段值必须出现在抄件内；注入 card+registry+词表做槽位绑定护栏。chunk 重叠去重按 (kind, method, subject, value指纹) 确定性合并。**切片注入约束（v1.1 治理条款）**：Stage 2 不全量注入六类 schema——paper_card 是路由表，按 section 分片：实验/表格区→result+config 切片；related work→lineage 切片；intro/discussion→finding+shift 切片。schema 体积与单次 prompt 体积解耦（治旧 harness 重 prompt 抑制召回病理）；单次调用 kind 切片上限在冒烟计划预注册。**装不进六类的内容 → overflow 记录**{kind:"overflow", quote, loc, reason（装不进的原因）, suggested_kind?} 进残差队列——不硬塞、不静默丢（治旧 harness"重规则压掉装不进内容→召回抑制器"的病理）。
- **Stage 3**：①数值逐字包含（归一化千分位/%/单位缩写/科学计数法）②explicit 维度值合法性 ③必填完备 ④枚举合法（relation/role/absence_type/epistemic）⑤loc 回源（模糊定位容忍 mineru 噪声）。失败→**一次 repair 调用**（带失败字段+原 quote 定向重抽，修而不杀）→仍失败 drop+日志。**残差率 = 冻结层充分性运行期指标**：overflow 占比+高频 reason 聚类进批次报告，持续高频=升版候选。
- **金丝雀**：每批混入已知答案哨兵段，陷阱率监控，防 prompt 回归。
- 三层验证归属：结构+忠实性=Stage 3；跨记录一致性（条件等价检测）=编译层规格。

## 4. 模型策略（2026-09-05 用户拍板）

- **判分**：Kimi-K2.6（call_paratera，偶发返回数组需归一）——与抽取模型分离保持独立性。
- **抽取**：**低成本档先行试用**（候选：Paratera `DeepSeek-V4-Flash` 关思考 `{"thinking":{"type":"disabled"}}` / CST `deepseek-v4-flash`），效果不好回退 `Qwen3.8-Max`（关思考 `chat_template_kwargs`）。
- **回退闸（冒烟阶段预注册，不许省）**：低成本档 vs Qwen3.8-Max 同 5 篇 A/B，比较 Stage 3 通过率、金丝雀陷阱率、抽检记录质量（quote 忠实/字段绑定正确）；任一显著劣 → 回退。前科警示：E2-R2 矩阵里"直读 2.83"是 deepseek 中端模型产物，中端档质量风险有案底，闸必须真跑。
- 允许分阶段混用（skeleton 便宜档 / slot 强档）作为 A/B 的一个臂。

## 5. 设计模式带走清单（代码零复用）

| 旧模式 | 新落位 |
|---|---|
| 两遍式 | Stage 1/2（升为语料级两阶段） |
| 修而不杀 | Stage 3 repair-or-drop（一次定向修复） |
| 校准经验 | quote-first + canonical 前置注册 + dims 合法性闸 |
| 模糊定位 | Stage 3 loc 回源容忍匹配 |
| 金丝雀 | Stage 2/3 批次哨兵 |

## 6. 工程隔离（执行位）

1. infra 抽层：`src/kb_infra/`（llm + embedding，含 CST qwen3-embedding:8b dim4096 真批量与两级 fallback；chunks/query 同模型同维度纪律）。旧栈 `llm_client.py` 保持 frozen 不动（新栈用 kb_infra 移植版，不共享 import）。
2. 旧栈 `src/granular_agent/` frozen：README 标注，Stage B 起不改一行代码。
3. 新包 `src/kb_compiler/`（拍板 G；内部 `records/` 子包先行，`views/` 编译层后建）+ **import 白名单机器闸**：测试扫新包全部 import（ast），只准 stdlib+kb_infra+kb_compiler 自身+白名单第三方，禁 granular_agent，越界即红。
4. 旧实验脚本（stageA/、ext_bench/）不回头改。

## 7. 拍板记录（2026-09-05）

**v1.0（2026-09-05）**：A 消融并入 result ✅ / B absence 独立类型+三态 ✅ / C 语料级两阶段 ✅ / D 五维词表+relation 七词+硬件双落位 ✅ / E shift 作抽取类型 ✅ / F quote ≤40 词 ✅ / G 包名 kb_compiler ✅ / gold 回测进 §8 第 0 步 ✅ / 残差队列进 Stage 2/3 ✅ / 模型策略=低成本档先行+回退闸 ✅ / 动工顺序=infra+闸先行、回测并行 ✅。

**v1.1（2026-09-05，gold 回测三批合并仲裁，用户批"按推荐+继续"）**：RC3 新增 hyperparam typed 维（最高优先，D类条件负载）✅ / RC2 registry 实体粒度条款+scope_ref 可空 ✅ / RC5 relation 九词（+generalizes +concurrent_with）+否定词悬案关闭 ✅ / RC4 origin_year_cited ✅ / RC6 manifest authors/affiliations ✅ / RC1 主张冲突边记录层不加、finding.target_ref join 键+块2义务 ✅ / RC7 图表清单最低配+图数据本体记已知限制 ✅ / RC8 quality_flag ✅ / 治理两条：用量审计减法议程+切片注入 ✅。观察级流转（块2义务10项/Stage 2 prompt 规格3项/冒烟关注3项）见 SCHEMA-BACKTEST-MERGED.md。

**v1.2（2026-09-05，冒烟 v3 残差队列仲裁——残差机制第一次正式产出）**：S1+S2 config 语义扩展（item 涵盖协议项/组件与架构参数，role+protocol，不立第七类型）✅ / S3 finding.claim_type 可选枚举 ✅ / 纪律性残余 ~10 条不修 prompt 只观察 ✅ / residual 闸维持 0.10 不放松 ✅。提案全文 SCHEMA-V12-ARBITRATION.md。

**v1.3（2026-09-06，颗粒流第二域 pilot 仲裁——第二域证伪测试的产出）**：冻结层结构级存活（relation 九词零 miss/六维名层存活/质量闸 4/4 过且 first_pass 0.924 高于主场/颗粒金丝雀 5/5+0陷阱）；唯一 ≥5 条同根因新概念聚类=记号定义 → **新类型 notation（第七记录类型）** ✅。图形数据残差（~30条，该域 ~1/4 数值活在图里）→ **文本模态先做扎实，多模态视觉通道后置立项**（用户指明 Paratera 有视觉模型；任务 #12）✅。setup 值层重建（dims_new 24/27 集中 setup）=批次3 预言命中，生长层按设计工作，无需动作 ✅。

## 8. 验收路径

- **第 0 步 gold 表达力回测 ✅ 已完成**：258 要点 E 240（93.0%）/P 10/G 0/U 8 → 8 根因仲裁全部并入 v1.1（明细 SCHEMA-BACKTEST-MERGED.md + backtest_part1/2/3.md）。诚实限定在案：同源偏置不可由回测自消（解毒剂=独立出题人+第二域回测）、三判定者无交叉复核、shift 类型 must 层零调用（价值证据仅在 PaperScope trend 失分）。
- **冒烟**：Stage A 40 篇 RL 语料抽 5 篇（含 2 篇表格密集）×（低成本档 vs Qwen3.8-Max）A/B → Stage 3 通过率+金丝雀陷阱率+回退闸判定；验收数字（后检通过率门柱/repair 率/drop 率上限/残差率上限）在冒烟计划里预注册。
- **第二域**：PaperScope 53 篇 AI 论文全管线 → 域无关性检验（setup 维度词表预期最先崩，重点看）。
- **全量**：RL 40 篇重抽 → 与 Stage A v2 记录对比（同语料同 gold，记录层升级的直接证据）。
