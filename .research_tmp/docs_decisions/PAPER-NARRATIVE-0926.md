# 论文叙事定稿 v3：领域世界状态（2026-09-26）

> **本文件=叙事唯一权威档**。取代 v1（verified compilation）/v2（Terms of
> Use/licensed claims）——两版均被用户否决，废弃理由见 §4 防走偏清单。
> 配套：实验设计=EXPERIMENT-DESIGN-WORLDSTATE-0926.md（v5 终版）；
> 查新判决底稿=NARRATIVE-ANALYSIS-0926-draft.md；证据语法=
> eval-anatomy-0926.md；基准审计=benchmark-audit-0926/。
> 标题命名/会场等未决项见 §10。写作时以本档+v5 为准，不再重新论证方向。

---

## 1. 一句话定位与电梯句

**定位**：竞品编译文献的**内容**（LKM=论文内部推理图、ScholarStack=
事实资产、GraphRAG 系=查询时图组织）；我们编译研究领域的**世界状态**
——治理化维度空间上的定位主张、物化的可比性等价类、类型化未知、时间
切片、更新语义。领域级问题从"综合项目"变成"状态查询"。

**电梯句（EN）**：A scientific agent acts in a world — the accumulated
research record of its field. We compile that world into a state: every
measurement located in a governed space of entities, subjects, metrics,
and conditions; every gap typed by its burden of proof; every relation
carrying its evidence basis; every past moment queryable. Field-level
questions become state queries, not synthesis projects.

**电梯句（中）**：科学 agent 行动于一个世界——它领域的累积研究记录。
我们把这个编译成状态：每条测量定位在治理化的实体×对象×指标×条件
空间中，每个缺口带举证责任类型，每条关系携带证据基础，每个历史时刻
可查询。领域级问题由此成为状态查询，而非每次现场的综合项目。

## 2. 故事链（intro 骨架）

- **Problem**：科学 agent 要答的是领域级问题（条件 C 下谁最好/什么
  从没被做过/截至 T 年已知什么/这两个数字能不能比）。文献是行存文档，
  检索式系统让 agent 每次查询现场拼装答案。
- **先行工作（正面引用，不回避）**：编译路线已被确立——LKM 以 40M 篇
  产品规模证明编译优于检索（ScholarQA citation F1 配对显著）；
  ScholarStack/ASKS/Agents-K1/CTIFoundry 同向；Karpathy LLM Wiki 占了
  大众叙事（"compiled once, kept current"）。**范式有效性不再是我们的
  主张，是我们的前提**。
- **Gap**：所有已发表系统编译的都是**内容**（论文里有什么：推理链/
  事实/三元组/摘要社区）；没有人编译领域的**状态**——可比性没有被
  物化（跨论文数值相遇靠答题时赌）、未知没有被类型化（"没做过"无处
  查询）、时间没有被切片（as-of 不存在）、状态不可更新（源失效无连坐、
  库不生长）。结果：组织好的知识仍被误用——跨带比较、转述当实证、
  没查过就说没有（外部实锤：引文核验器 unsupported 率 3%→18% 漂移
  [2607.20527]；AstaBench 官方自认 citation 指标不验真实性）。
- **Challenge**：状态必须保真（LLM 抽取器不可靠→确定性验证门+canary
  哨兵，工程支柱）；可比性必须可计算（条件散落在文本里→治理化维度
  词表+编译期等价类）；未知必须有认识论分档（absence 三态+穷尽性
  依据+推导空洞的 paper-anchored 规则防爆炸）；状态必须可更新（失效
  代数+生长回流）。
- **Insight**：编译单元=治理化维度空间中的**定位主张**；视图=状态空间
  的投影（矩阵/覆盖/谱系/档案）；领域级查询=投影上的直接读取。
- **Method**：§3 系统概览。
- **Evidence**：五槽主实验（v5 设计档）+横切分析。
- **Limitation**：状态是语料相对的（"compiled from corpus C"，推导
  空洞自带"语料内"限定）；单一答题模型族；深抽成本真实（~2min/篇，
  Tier 分层是经济学答案）；更新语义演示规模小（n 披露）。

## 3. 系统概览（方法章骨架，部件→状态语义映射）

| 部件 | 状态语义 | 关键数字（盘上复核过） |
|---|---|---|
| 类型化记录（7 类，quote+loc 100%） | 定位主张（值+条件维度+认识类型+逐字存根） | 54,486 条/430 篇；epistemic 94.2%；result dims 90.4% |
| 确定性验证门+repair-or-drop+canary | 状态准入与年检（**工程支柱，非矛尖**） | 95% first-pass；canary 7 事实+6 陷阱×3 套跨域 |
| 注册表+词表治理（仲裁/使用审计/版本化） | 坐标轴立法 | 8,968 实体；人工审计混淆 0.11% |
| 分带比较矩阵 | **可比性等价类物化**（旗舰机制） | 5,079 表；带内 drift→仲裁 |
| 覆盖网格+三态 absence+推导空洞 | **类型化未知** | 1,939+2,014=3,953 缺口坐标 |
| 谱系+as_of | 时间切片 | 971 类型化边（evidence_basis 100%） |
| typed tools（13 个） | 状态查询界面 | F12 判决：纯检索访问有损 |
| 答题接地门（回指/值锚定/absence 断言） | 查询时合法性强制 | 误用病例档案=检测器来源 |
| 失效代数（溯源索引→blast radius→两阶段提交） | 状态更新语义·吊销 | 查新判决：完整闭环无先例（"组合首次"措辞） |
| broker+粗抽回流+Tier 阶梯 | 状态更新语义·生长 | live 验证（Mamba 12.9s→29.9ms 复利案例） |

## 4. 防走偏清单（被否叙事，禁止复活）

| # | 废弃方向 | 废弃理由（裁定人/依据） |
|---|---|---|
| 1 | verified compilation/验证当矛尖 | 用户裁：验证=工程支柱；读作 QC |
| 2 | zero-trust 框架 | 用户裁：还是把可验证性当主要内容 |
| 3 | Terms of Use/licensed claims | 用户裁：许可=验证换马甲 |
| 4 | 需求驱动答案结构编译当矛尖 | v6 五席判 C5 凑数+用户裁不够新（降方法论节可用） |
| 5 | 计量学框架 | 用户裁：只覆盖数值子系统（finding=半个库） |
| 6 | 记录优先 vs 图 当总纲 | 用户裁：只打得动传统 KG，打不动 LKM/ScholarStack |
| 7 | "首个编译 KB 打败 RAG" | LKM 09-23 已发表+产品化（占位判决） |
| 8 | n-ary 超边/schema 共演化/方法演化关系 首创 | 旧卖点，LKM Def7+Buehler+Intern-Atlas 占位 |
| 9 | gap 记录首创 | LKM research-gap records 占广义领地；仅"字段级三态+外向闭环交点"存活 |
| 10 | 消融当主实验 | 用户裁：主实验=对外权威 PK，消融封存至定型 |
| 11 | Multi-108 闭卷数字进论文 | 用户裁：oracle 语料攻击面；0.6503 退内部证据 |
| 12 | ScholarStack/Lacuna 数字引用 | 用户裁：预印本+实验粗糙，仅 related work 提及 |

## 5. 近邻划界措辞（写作即用）

- **LKM**：编译论文**内部推理**（premise→conclusion），领域组织=语义
  聚类**地图**（34,360 clusters）；我们编译领域**状态**=可运算的
  **仪表盘**（可比性等价类/类型化空洞/时间切片/更新语义）。其 belief
  propagation=数值可信度；我们=类型化认识态+条件结构。数字永不混表
  （开集 40M+GPT-5.4 vs 受控闭卷+27B）
- **ScholarStack**：条件记录为上下文串+答题时推理，verification
  status=模型语义核验，Knowledge View=任务时组装；我们=编译期物化
  等价类+确定性验证+物化视图。其 MDAQA 18.41/8.46 **不引用**
- **HippoRAG 2**：记忆=联想检索索引，持续学习=只加不撤；我们=带吊销
  的状态（必引其持续学习实验为最近邻）
- **Eigenius**："epistemic 不变量"措辞先占（DBMS 提交期赋值）；我们=
  从论文修辞抽取+语料规模+答题期强制
- **ASKS**："scientific knowledge compilation" 术语先占；其确定性
  checks=图变更转换非保真验证
- **EGT-KG**：最近邻 baseline，实跑对比+引用
- **DIAL-KG**：治理最近亲（无 QA/不开源/无演化 ablation）
- **EGM 系**（Diabetes Camp 2026 最逼近）：概念词汇拥挤，机制四点
  （记录级三态/推导空洞/持久可查询/机器构建）无先例——定位为"其
  自动化+记录级升级"
- **GRADE**：正交（证据体级人工分档 vs 记录级自动修辞类型）——用
  "强度轴"措辞必引声明粒度
- **GiantsBench/Hakken/BackTrend**：条目级未来预测 vs 我们报告级+
  状态驱动+客观时间验证（FieldState-Bench 划界）
- **盟友证据**：2607.20527（核验器不可靠）、AstaBench 官方自认、
  GraphRAG 维护模式、Agent Traces 综述（selective invalidation 列
  开放挑战）、SIGMOD26 provenance tutorial 零提 LLM
- **DB 线**：multi-truth（PVLDB 2018=远亲已核销：domain=源专长度非
  条件）、DeepDive（头号划界：概率化 vs 离散认识态+仲裁）、c-tables/
  PDB（概率 vs 类型化）、IVM/Enzyme（可靠 Q vs 不可靠 Q）、SUPG/Lotus/
  Text2KGBench/WikiMonitor-Onto（失效代数措辞="组合首次"）

## 6. DB 审稿人四连杀防御（预写）

1. "DeepDive 换不确定性表征"→ Q 的概率校准本身不可信（与核验器不可信
   同源），故离散认识态+制度化仲裁；正交路线论证
2. "multi-truth 已做条件依赖真值"→ 已核销（其 domain=源专长度；
   自动消解 vs 我们仲裁不消解）
3. "workload-driven=auto-admin 搬到 LLM"→ 失败模式信号 vs 代价统计
   信号，配实测案例（card 空 42%→12% 等修复史）；且已降为方法论节
4. "DocETL/Palimpzest/LOTUS 覆盖需求驱动"→ 它们查询时优化用户给定
   pipeline；我们建库期生成结构本身

## 7. 措辞纪律（硬规则）

1. "deterministic adjudication, with LLM proposers/repairers under
   re-adjudication"——不写裸 "deterministic verification"
2. 失效代数="组合首次"，不写"无人做过"（WikiMonitor-Onto 必引）
3. 状态永远带语料限定："state compiled from corpus C"；推导空洞=
   "语料内未见"
4. 条件槽不夸大：不写"每条记录带条件"（实测 dims 全库 32.9%）；写
   "每类记录带各自的门强制许可结构，条件在决策关键处密集（result
   90.4%），空槽=无限定主张（合法状态）"
5. LKM/ScholarStack 数字永不同表；外部数字仅引已发表论文
6. canary=三仪器面板之一（+人工审计+门统计），不单独承重
7. "world state" 术语源引 STRIPS/规划文献；不用 "world model"（避免
   动力学预期）

## 8. 贡献声明结构（intro 末尾列表用）

1. **表示主张**：研究领域世界状态的首个编译实现——定位主张+物化
   可比性+类型化未知+时间切片+更新语义（对照内容/推理/资产编译范式）
2. **实证主张**：五槽 matched-model PK——状态查询胜零件综合，增益
   集中于领域级题型（外部权威基准：PeerQA/MDAQA/CS2/ReportBench）
3. **基准贡献**：FieldState-Bench——时间剖分客观 gold 的领域认知
   基准（零标注、不可 game、随论文发布）
4. **支柱**：确定性保真门+canary 测量学（工程基础，如实定位）
5. **分析贡献**：综合合法性误用分类学+全臂误用率（首个测量仪器）

## 9. 论文章节骨架（成文时细化）

1 Intro（张力：访问已解决、状态缺失；误用病例开场）
2 Related Work（五轴：编译式科学 KB/图与记忆 RAG/归因与验证/缺口与
负知识/DB 与 EBM 线）
3 World State 表示（记录=许可结构；治理维度；视图=投影；时间；负空间）
4 编译管线（抽取+门+canary+治理+失效代数——验证为支柱）
5 状态访问与强制（typed tools+答题接地门）
6 主实验（五槽，v5 档）
7 分析（误用分类学/宽度梯度/成本三本账/失败分析）
8 更新语义（失效注入+生长纵向）
9 Limitations（语料相对性/单模型/成本/更新演示规模）
附录：消融（定型后）/披露全集/FieldState-Bench 规格

## 10. 未决项（新 session 与用户裁）

1. 标题终裁（候选：The World State of a Research Field: Compiling
   Scientific Literature into Queryable State for Agents / A Compiled
   World State for Scientific Agents / The Field, Compiled；系统名暂沿
   CompileScholar）
2. 会场：EMNLP 2027 六月档（五槽全量，推荐）vs ACL 2027 一月档（砍槽4或5）
3. FieldState 领域数 3 vs 5；CS2 test 100 冲不冲；MDAQA 子集 200/300
4. 槽 3/4 API 预算额度
5. C4 delta 继承解析复活与否（五席旧评"最干净新机制"，未实现；复活则
   归入可比性条款的"条件传递"机制）
6. 投稿前竞品复核硬前置：ScholarStack 全文复核/LKM 新版本监视/
   SciAtlas+超图系+GraphRAG 家族补核/M3 需求驱动 2025-26 地毯排查
7. ISNĀD-Rijāl claim grading 论文（Google 扫到的远亲）核一眼
