# 顶会叙事分析（2026-09-26，草稿）

> 方法：第一步直接精读三处实现（kb_compiler 11.4k 行 / 答题 harness 7.7k 行 /
> sci-evo 检索栈 18k 行，核心文件全文精读，注释中的实测教训作一手设计理由）；
> 第二步 CDP 发现层 + 一手 ID 验证层调研生态位。本文件 = 第一步结论 +
> （待并入）调研判决 + 叙事方向。实验数字全部来自盘上产物（scores.jsonl /
> views.json stats / postcheck_stats / 判档文档），非设计文档转述。

## 一、系统真正建成了什么（代码精读结论）

### 1. 建库侧：可验证的科学文献"编译器"

**管线**（records/，全部读过）：
- Stage 0 manifest：零 LLM 年份（arXiv API published 权威戳，杜绝年份漂移）
- Stage 1 skeleton（cards.py）：单遍全文通读 → 论文卡片（方法身份/机构/贡献自述/
  实验矩阵/图表普查/维度候选/相关方法+关系措辞+被引年份/章节语义标签）。
  卡片 = Stage 2 的路由表 + Stage 1.5 的注册输入
- Stage 1.5 registry+vocab（registry.py, 1273 行）：表面名收集 → 实体合并 →
  维度词表。反病理设计密集区：
  - LLM 输出 index groups 而非字符串 + 机器覆盖检查（截断→singleton 兜底，
    永不静默丢实体）；渠道死亡大声中止（ChannelDeadError）而非伪造全 singleton
  - own-paper 类型确定性覆盖 LLM 标签；origin_year_cited 由 mention 级年份
    投票确定性得出（LLM 从不碰年份）
  - 规模防御：embedding 语义分块（余弦连通分量 + 缩写/包含边通道，legacy
    registry 上校准 recall 0.896）→ 块内 LLM 合并 → 跨块 canonical 二次合并；
    co-mention 通道只做定向 singleton 回收（单向，不进连通分量——实测传递
    闭包会把一切链成 8-23 个巨型块）
  - 确定性合并守卫：'+' 变体守卫（mCLIP+ ≠ mCLIP）、超短 token 守卫、
    词缀扩展守卫——"碎片化安全，混淆有毒"
  - 懒输出合规环（>30% 未分配→只补漏项的追加调用）
- Stage 2 deep_extract：section 感知分块 + 卡片主/正则兜底的记录类型路由
  （带 odometer 计数）；quote-first 纪律（先逐字抄 quote，字段只从抄件填；
  数值/指标名/方法名必须在 quote 中逐字出现；"即使你认识这篇论文也严禁用
  记忆"——DSF memory-blurt 实测教训）；kind-slice 注入（schema 体量与单次
  prompt 体量解耦）；表格/公式记录协议（逐字符照抄，跨格式转写=拒绝）；
  枚举纪律；宽表列位对齐规则；限定条件操作化（condition/dims_new）；
  overflow 残差队列（"可见的残差，静默丢弃才是错误"）；生长纪律（LLM 只写
  表面名，规范化解析确定性完成；未解析→entity_queue，新维度值→dims_new
  仲裁队列，抽取时永不发明）；全局 chunk 池 + WAL 检查点 + repair 补洞；
  inline postcheck（篇级流水线，边抽边检）
- Stage 3 postcheck：五道确定性门（必填字段/枚举合法/引文逐字→loc 锚定
  [精确/归一/模糊锚+LaTeX 折叠+表格宽松通道]/数值逐字[双通道数字形态]/
  词表合法[非致命→dims_new]）+ repair-or-drop（一次修复调用带违规项+原文
  chunk，复检，仍坏→drop+log）+ 证据分诊（近零救活率的违规类直接 drop，
  全档案 n~2000 实测）+ FG7 确定性迁移（schema v1.4 的最大违规类不是 bug
  而是缺失语义槽位——strength='cited' 迁移到 epistemic 而非丢弃）
- 独立通道：table_channel（F24 零 LLM 确定性表格解析：HTML/管道表、
  rowspan/colspan 网格展开、多层列头路径、banner 行组追踪、融合单元格
  诚实跳过→可见残差、单元格级引用 epistemic 覆盖、行级逐字 quote 最佳
  匹配、注册表链接仅在唯一匹配时）+ table_semantic（F35：LLM 只 PROPOSE
  角色，确定性结构门是唯一写权限——注册表匹配/基准泄漏 blocklist[从不进
  prompt]/caption 溯源/canary 地面真值硬停）+ figure_channel（VLM 识图）+
  notation_harvest（公式符号）+ absence 全文遍（三态缺失性事实）
- canary：合成哨兵论文（7 事实+6 陷阱）走同一管线，规则判分（无 LLM），
  预注册门（fact_recall≥6/7, trap_fired==0）；canary_granular 验陷阱检测器
  域不变性；table_semantic 自带 canary（role-reversed + normal 两案）
- 治理：schema 冻结+版本化（v1.0→v1.4 每次修订有仲裁记录与实测证据）；
  usage audit（填充率+枚举使用+视图 FIELD_USES 计数→退休候选，报告不自动删）；
  arbitration list（新实体预聚类供人工仲裁）；consistency 四检（同篇带内漂移/
  cited-vs-source 数值对照[MultiHiertt 容差]/带等价类/关系对称性）→仲裁队列，
  语义决议永不自动
- views 编译（compiler.py）：matrix（subject×metric，单元格带可比性 band=
  setup 规范+预算量级桶，epistemic，delta）、genealogy（类型化边+祖先闭包+
  as-of 时间旅行）、coverage（抽取三态 absence 与推导空洞严格分开；paper-
  anchored 推导规则防多域爆炸——实测 13115 爆炸守卫）、cards（跨论文证据
  档案；IL-B4 round-robin"每篇先发声一次"+直接点名 finding 优先）、
  narrative（shift 链+主题家族线，非编年流水账）、pair_deltas（推导，
  provenance 分离）、notation index。FIELD_USES 喂治理减法
- maintenance（失效代数）：provenance index（record_id→每个消费它的视图
  位置）→行级失效→blast radius 报告→dream queue（失效/源漂移待重抽）→
  两阶段提交（staging→validate→原子替换）

**Multi-430 实测数字**（盘上产物）：
- 430 篇 → 46,726 条抽取记录；postcheck first-pass 95.0%、repaired 0.9%、
  dropped 4.1%、overflow 残差 2.7%
- views：54,486 条记录（含 table 通道）、5,079 矩阵表、971 类型化谱系边、
  1,779 覆盖实体、1,939 经验缺失 + 2,014 推导空洞（=3,953 缺口坐标）、
  335 实体档案、1,375 成对差值、214 符号
- registry_v3：8,968 实体（335 in-corpus / 3,784 out_of_corpus / 1,182
  mechanism / 424 practice）；人工审判决档：真实混淆 3 处/2659=0.11%
- 全流程 12 篇九站复验全绿（09-25）；135+19 测试

### 2. 答题侧：确定性 typed tools 优先 + 反 fabrication 门 + 循环健康装置

- KBTools 13 个本地工具：compare（空结果返回 vocab_hint"你是不是想查"+
  实体在/不在诊断+轴混淆提示——G1-B1 实测 15/27 空调用因自由文本撞编译
  词表）、lineage（方向/关系/as_of/paper_id 域定）、find_gap、config
  （item-only 降级带标记）、findings（'|' 多措辞 OR、短名整词边界）、card
  （miss→nearest_in_corpus 重定向+字面 next_action——G1-B2 实测 146/153
  空 card 是库外实体）、as_of、entities（类别→成员扩展）、search（embedding
  兜底，same-model-same-dim 纪律+provider 失配拒答）、fetch_chunk（record_id
  →原文窗口）、describe_kb（零 LLM 定向——B6 实测语料 4.5x 后 compare 使用
  率降 36% 的对症）、search_text（A7 BM25+vector RRF 混合全文检索=
  "编译区逃生阀"，对"预编译覆盖不全"反驳的功能性回答）
- F12 hybrid 臂判决是设计权威：纯检索访问有损（记录在库、检索没中→agent
  答"不存在"，1 vs 5 分）→ typed tools 优先、embedding 仅长尾兜底
- 笔记协议：Markovian 全量重写 notes（固定 schema：N 行带 [record_id] 回指+
  anchor 逐字数值+conditions+epistemic；X 行被推翻主张保留不擦除）+ gaps
  清单 + queried 去重表；笔记是唯一记忆
- 接地门（全部确定性）：backref gate（N 行必带已知 id；宽松复合解析+匹配
  顺序教训 P0-1：2,898 假拒绝/85 题硬停的根因修复）；F27 值锚定一致
  （锚定行内每个小数/百分比/倍数必须逐字出现在所锚记录——堵"真 record id+
  编造数值"洞，d070 实测五个编造 AP）；F26 标题篡改门（模糊匹配真标题但
  非逐字=参数知识渗透的编造副标题形态）；答案数值门（边界守卫+hex id 剥离）
  +未溯源句检查（[pid] 或 Author-year 双引用形等价）+absence 断言门（F9：
  断言"文献里没有"必须真查过 absence 通道；"我的检索没有浮现"认知语域放行）
  +citation lock+repair≤1
- 循环健康装置（纪律：只用可观测状态，无 gold、无题型感知——"过拟合闸"）：
  F22 中程覆盖审计（capacity-forward 措辞——budget-panic 实测教训）、F25
  转写提示、F28 无进展梯（3 步无证据→自动 findings(paper_id) 补检索[max 3]
  →目标尽→强制诚实收答）、F31V2 自动证据（系统从观测转写 A 行持久块，
  消费/控制面分离，伪造 A 行剥除——mode-A 转写失败实测：富观测到了笔记
  没接住→honest-zero）、动态步数帽（语料规模 R-A + 每题实体需求 A2，
  零 LLM token 重叠计数）、反循环（dedup/指纹>3 击杀/零 delta 冗余/2 次
  冗余回滚）、预算耗尽注入（Search-o1 模板，不弃答）、早停守卫（零工具
  短答案拒收一次强制检索）
- 判分纪律：官方协议逐字（extract_citations/AutoAIS 官方 prompt/rubric 生成+
  判分）、判分独立账本、数据 hash 版本锁、gold 探针天花板（rubric 0.598）、
  配对 bootstrap CI、判分器冻结快照

### 3. 开集侧：知识模型驱动检索 + 库生长飞轮（建成、live 验证、未量化）

- 分层检索引擎（sci-evo）：fast/full/agent 三档延迟分级；熔断+递进降级
  （实测 arXiv 406→S2 429→OpenAlex 1.3s 接住且目标第一）；语义通道打头
  （A6 判决：唯一跨词汇鸿沟的通道）；A3 查询理解（一次小调用→意图/域/
  子查询；agent 档跳过——循环已精炼，重分解纯浪费 5s/次）；A4 嵌入重排
  （目标论文 4→1）+LRU/DOI 双缓存；P15 子查询并行
- gap_search：KB 的 absence 记录+推导空洞（3,953 条缺口坐标）→与问题匹配
  →缺口坐标合成定向 query（**词汇是题面没有的**——知识模型贡献盲检索无法
  表达的查询）→同一分层引擎执行→候选对缺口坐标做 fill 标注
- lineage_walk：类型化谱系 BFS（extends/improves/replaces/generalizes）→
  边界检测（frontier=库内谱系终点）→外部续走（后继 query 由谱系上下文
  合成）→successor 标注；backflow 已挂的外部论文零检索成本优先浮现
- citation_graph：S2 双向+OpenAlex 兜底+语料回标（in_corpus=paper_id）
- extract_paper（Tier-1 粗抽）：摘要→同 schema 子集轻记录（finding/method/
  limitation+mentions；coarse: 前缀防撞；epistemic=stated 默认；postcheck
  结构门保留、quote 锚定跳过）；11-16s/篇
- mentions 回流（backflow.py）：粗抽 mentions=外部论文视角的引用意图→
  两级确定性匹配（精确+连续 token 包含；blocklist 501 条防跨域同形；
  歧义即弃=诚实 miss>错边）→external_mention 弱边（provenance=coarse，
  升级留给 Tier 2）→JSONL 持久化+重放→**库在答题中外扩**；活体：Mamba
  摘要 12.9s 粗抽→2 边挂 Transformer 实体→lineage_walk 29.9ms 零成本召回
- TierStore（SQLite）：粗/深状态账（重抽跨会话拒绝）+确定性升级规则
  （limitation 记录或≥2 谱系锚→promotion-candidate）；promote_to_deep
  离线跑（获取链 arXiv 身份验证→mineru→篇级 kb 链→深级入账），永不进
  延迟受限的答题循环
- A6 结构性发现：闭卷单发召回 FAIL（0.062，门 0.7）——诊断链完整（池召回
  ≈top-10 召回=gold 从未被检索到；天花板探针正常；根因=词汇鸿沟：多跳
  gold 靠特定发现连题，题面不含这些词；physics 0.247 最好/cs_nlp 0.0）
  →**含义：检索必须进 agent 循环迭代，单发只是地板**——这正是 ReAct 架构
  与 broker 设计的依据

### 4. 成绩单（Multi-108，ScholarQA-Multi 官方判分链）

- 设定：闭卷 430 篇、三臂同答题模型（local Qwen3.8-27B）、判分 GLM-5.3、
  官方 extract_citations 逐字+AutoAIS 官方 prompt+rubric 官方 judge prompt
- Citation F1（预注册主判据）：**ours 0.6503** vs LightRAG 0.5146
  （+0.136 CI[+0.071,+0.200] 配对显著）vs PaperQA2 取优 0.4784
  （+0.172 CI[+0.104,+0.239] 配对显著）
- Rubric 质量轨：ours 0.5113 vs 0.4588/0.3722；gold 天花板 0.598（ours 达 86%）
- 必须披露：0.6503 是 P1-C 机械引用格式修复后终值（修复前 0.5183，三臂
  罕见接近、CI 跨零）。修复性质=纯格式（编译路径把 49 个 [hex/stem] 源
  重编号成 [1]..[44]，官方桥只能翻译 hex/stem→43/44 越界；修复=从该题
  自己的笔记重建映射还原），零内容改动零 LLM 重跑；基线核查无同类格式病
  （PaperQA 5 越界、LightRAG 0）
- **0926 离线复核**：从盘上修复后 answers_ours.json + 官方 gold 按官方
  extract_citations 逐字重算 = F1 0.6503 / P 0.7535 / R 0.6237，与报告
  一致。⚠️ 产物管理缺口：answers_ours.scores.jsonl 停在修复前状态
  （0923 21:08），修复后判分未落盘——论文前必须重跑 judge 落盘+冻结快照
- 分域分解（修复后，12 域）：rulin_cs 0.98 / benjamin_bio 0.87 /
  weijia_cs 0.85 / pan_biophysics 0.85 / hao_photonics 0.82 /
  shengyan_photonics 0.74 / minyang_physics 0.69 / jacqueline_cs 0.67 /
  akari_cs 0.63 / yanyu_photonics 0.49 / norman_bio 0.49 / **bohao_cs
  0.00（三臂全零：语料对 cs_hci 覆盖弱=基准死区，披露项非我方缺陷**；
  id_mapping 覆盖已核实正常）
- AutoAIS 参考轨：ours 0.0704 低于 PaperQA2 0.1332——段落级验证材料
  系统性压分（gold 天花板同轨也只有 0.42），用户裁定砍掉留档披露
- 效率：P0-1+P1-2 修复后步数尾部塌缩 p90 40→21、max 47→30、40+步烧尽
  3→0；F1 配对无损（-0.04 噪声带内）
- 行为探针 5/5 出答案：主动外扩①/迭代精炼②/Tier-1 升级④/引用存活⑤
  成立；驱动式工具选择③部分成立（有锚点场景正确调 lineage_walk_ext/
  gap_search；缺口型题模型偏好库内答）；发现 P16 转写自旋（已修）

### 4b. 规模轴证据（PaperScope 16→53，档案判决）

- PS16（16 篇 gold-only，检索平凡）：ours 3.100 vs A9（RAG 对照臂）3.147
  ——略落后，语料小到检索不吃力时编译无优势
- PS53（53 篇=16 gold+37 硬负例干扰）：ours 3.053 vs A9 2.887，配对
  +0.167 CI[−0.049,+0.382] W16/L8——**转正但不显著**；ΔD 交互 +0.213
  （语料 3.3x 后 A9 掉 0.26、ours 掉 0.05）
- 分题型：trend（演化/趋势类）+0.440 最强优势格；gap/resu 修复后全部
  转正；fresh16 泛化闸 PASS（题型画像同构同量级，无过拟合警报）
- 判读：**规模是编译路线的朋友**——检索臂随语料增长退化、编译臂保持；
  16 篇平局是"检索平凡"伪影。与 Multi-430 显著胜构成同一条规模趋势线
  （16 平→53 转正→430 显著）
- 注意：PS 判分是 rubric 1-5 分制（Kimi 主判+官方 rubric 逐字），与
  Multi 的 citation F1 不同轨；PS 判分器天花板 ~3.15（全场无臂超过）

### 5. 独到 vs 常规（评估）

**精心且有独到之处**（可作论文贡献候选）：
1. 缺失性事实一等公民：三态 absence + 推导空洞 + paper-anchored 推导规则
   + 抽取/推导 provenance 严格分离——知识模型的"负空间"
2. 机器可验证抽取质量：quote-first + 五道确定性门 + repair-or-drop +
   证据分诊 + canary 事实/陷阱双向自校验——全程无 gold
3. 治理式 schema/registry 演化：冻结+仲裁通道+usage audit 减法+overflow
   可见残差+dims_new+版本化生长（抽取时永不发明）——FG7 是范本案例
   （最大违规类=缺失语义槽位，仲裁加槽+确定性迁移而非丢弃）
4. provenance 贯穿+失效代数：record→view→answer→判分全链 record_id/quote，
   行级失效+blast radius+两阶段提交
5. 答题循环的"只用可观测状态"装置纪律：F22/F25/F27/F28/F31V2/动态帽——
   每个装置有实测病理+修法+复测数字，且过"无 gold 无题型感知"闸
6. 缺口驱动检索+回流飞轮：库知道自己缺什么→缺口坐标变检索词汇→外部
   论文粗抽→mentions 挂边→库外扩→下次走谱系零成本到达
7. typed tools 优先/检索兜底的明确对立立场（F12 判决+A7 逃生阀+G2 A/B
   全暴露判决）

**扎实但常规**（不当贡献卖）：BM25+vector RRF、嵌入重排、熔断降级、
WAL/shard、缓存、ReAct 循环本体、查询分解→多源→重排的组合

### 6. 诚实弱点（叙事必须处理）

- W1 核心差异化机制（gap_search/lineage_walk/backflow 飞轮）**无量化增益
  数字**（P2）——只有 live 验证+单例轶事（Mamba）；"知识模型驱动 vs 盲
  检索"目前是设计论证不是测量
- W2 开集基准未跑（SQA2 待裁点 2-5；外扩反面冒烟 P3 未做）
- W3 单基准胜+引用修复披露（0.5183→0.6503）需要极谨慎的呈现方式
- W4 PaperScope 平局（16 篇语料太小=伪影，但对外是"另一个基准上没赢"）
- W5 AutoAIS 弱（已披露砍掉，但审稿人可能追问）
- W6 建库成本（深抽 ~2min/篇 + 全链）必须诚实报；Tier 分层是解法但
  Tier-2 经济学没系统测
- W7 两仓未合并（P5）、粗记录未全进 views（P8）、7,759 未仲裁实体（P10）
- W8 答案模型单一（Qwen3.8-27B 三臂同模型是公平设定，但"换个更强模型
  编译优势是否还在"没有数据）

## 二、调研判决（4 路 agent，滚动并入）

### 2.0 发现层先行成果（基准调研 agent 的发现子代理，0926）

**新基准（2026，全部一手 arXiv ID 已核 abs）**：
- SAGE（2602.05975）：1,200 queries×4 域、20 万篇开放语料、reasoning-
  intensive retrieval，6 个 agent 系统全部挣扎——"放大语料→我方优势显现"
  的 scaling 方向靶场
- MUSES（2609.00313）：2.33M 篇固定语料、prospective intellectual-roots
  检索、作者确认 paper-level 标签——百万级固定语料=预编译范式理想战场
- ScholarQuest（2606.20235）：taxonomy 引导 agentic 论文搜索、4 种研究意图
- AutoResearchBench（2604.25256）：PaperScope 同团队（BAAI）续作、自主
  文献发现双任务
- LitTraceQA（2608.07370）：找对论文→定位证据→忠于证据答案三段联动判分
  ——与 typed tools+证据溯源同构
- MAPLE（2608.15624）：multi-aspect 全文检索（motivation/method/findings
  三面一致性），ML/NLP 主场域
- RATIO（2608.27394）：按类型化 ideation 操作（Address/Broaden/zoom-in）
  定义相关性——typed 关系一等公民理念同频
- ADRA-Bank（2512.00986）：学术 DR agent 模块化基准（检索/规划/推理分评）
- SciExplore（2607.20926）：科研信息 seeking 工作流 103 专家任务

**⚠️ 新竞品/同生态位（已转竞品 agent 一手深核）**：
- **LKM（2609.27297，09-23 发布）**："文献→共享计算可访问推理资源；
  source-grounded reasoning graphs；结构遍历+语义检索耦合"——正面同生态位
- SimScholarSearch（trillion-labs repo）：1.12M CS 论文本地检索 RL 环境、
  9 工具接口——"本地语料+typed tools"结构同构（但为 RL 训练环境）
- SciLENS（2609.03338）：~12M 学术记录双层索引全本地 agent
- Crase（2608.24809）：引文图 1.5-hop 有界探索+entailment 剪枝胜 DR agents
- EGT-KG（2609.00479）：typed KG 检索+小模型科学 QA
- **2607.20527**：引文核验器本身不可靠审计（unsupported 率 3%-18% 随
  verifier 漂移）——我们"验证层"叙事的外部佐证

**发现层含义（初判，待验证层确认）**：
1. 2026 下半年基准爆发期，"reasoning-intensive/agentic 科学检索"是新热点；
   固定大语料（MUSES 2.33M / SAGE 200k）与我们 scaling 叙事对口
2. 尚无任何新基准以"缺失性事实/absence"为评测轴（待 agent D 确认）
3. LKM 是最大撞车威胁，判决前叙事不定稿

### 2.1 gap 驱动检索新颖性（agent D 中间简报，机制 1 已核）

**判定：组合成立，无完全占位者。**
- 最近邻（学术）：EviMem（2604.27695）、S2G-RAG（2604.23783）——做到
  "结构化缺口→下轮检索 query"，但缺口是**查询时临时诊断**，非 KB 持久化
  absence records（无三态/无逐字引文/无矩阵空洞/无 fill 标注）→部分重叠
- 最近邻（开源）：history-research-agent——gap→query→本地档案链同形，
  检索对象是本地史料非外部文献，无 fill 标注→部分重叠
- ⚠️ "LLM 检测 research gap"赛道已饱和：GAPMAP 已区分 explicit/implicit
  gap（与 explicitly_stated 撞概念）——**gap 检测本身不能当独立卖点**
- 差异化轴（论文措辞方向）：transient query-time gap diagnosis vs.
  **persistent KB-level absence coordinates + fill 标注**；应主动引上述
  三个部分重叠工作并沿此轴区分
- 机制 2（谱系边界续走）、机制 3（回流飞轮）核查进行中

### 2.1c 机制 2/3 判决（主协调员亲自一手核验，轻扫口径披露：arXiv 精确
检索 6 组 + CDP Google 发现 2 组 + abs 一手 2 篇；置信度中——轻于机制 1
的 17 组重扫，但两组独立通道均零占位命中）

**机制 2【类型化谱系边界续走】**：
- arXiv 精确检索 "citation graph traversal agent retrieval" 零命中、
  "method genealogy" 零命中；Google 发现层无占位
- 最近邻=Crase（2608.24809，agent A 已核：查询时引文图有界探索，
  无持久库、无类型化关系——entailment 剪枝走的是无类型引文边）+
  Intern-Atlas SGT-MCTS（占"lineage reconstruction"名，但服务 idea
  生成非检索续走）
- 判定：**"类型化关系 BFS→语料边界检测→定向外部后继检索→successor
  标注→已回流论文优先浮现"的组合无占位者**；引用链遍历这个一般动作
  当然普遍（Litmaps/ResearchRabbit 级常识），差异化在类型化边+边界
  交接+backflow 优先。撞车程度=部分重叠（仅远亲），组合成立

**机制 3【答题中库生长/回流飞轮】**：
- arXiv 精确检索 "test-time knowledge expansion"/"growing knowledge
  base"/"knowledge base growth agent" 零相关命中（DIAL-KG 是唯一
  incremental KG+QA 命中，已判治理近亲、无 QA 接口）；Google 发现层
  无占位
- 最近邻（abs 一手已核）：Dual-Layer Agentic Memory（2608.22215）——
  write-phase cost-aware 路由（non-write/write-new/write-update 小→大
  模型级联）+定期参数化 consolidation（SFT 写回）；通用对话记忆域、
  LLM 路由非确定性规则、consolidation 是参数内化非 schema 记录深抽、
  无实体解析挂边。Sleep-time Compute（2504.13171，Letta）——离线预
  计算通用上下文，"编译一次"的通用记忆版。LKM §6.2——回灌愿景层占位
  （§6.3 自认 first stages 无实验）
- 判定：**"答题循环内 Tier-1 粗抽+mentions→注册表确定性解析挂弱边+
  JSONL 持久化跨会话重放+确定性升级规则+零成本复利召回"的组合无
  占位者**。撞车程度=部分重叠（记忆生命周期话术近亲存在），组合成立
- 引用义务：Dual-Layer/Sleep-time compute 作记忆生命周期最近邻；LKM
  §6.2 作愿景先行；差异化轴=科学 schema 记录+确定性实体解析挂边+
  可检查性（consolidation 后仍是可审计记录而非参数）

### 2.1b 机制 1 最终判决（agent D，LKM 叠加版，一手已核）

**判定：完整链条仍未被占位，但卖点收窄、创新论证形态必须改变。**

被占死的：
- "KB 持久化缺口记录"单点——LKM 以 40M 规模把 open question/weak point/
  insufficient 档做成一等对象（原文："connected problems, subproblems,
  premises, reasoning chains, conclusions, highlights, weak points, and
  open questions"）。从"我们的首创"变成"必须引用的先行验证"
  （反面好处：大厂重注=设计方向被独立证实）

仍然空着的（我们坐在两类先行工作的交点上，交点无占位者）：
- **持久缺口坐标（三态+逐字引文+矩阵空洞）→ 合成题面没有的检索词汇 →
  库外定向搜索 → fill 关系标注回挂缺口** 的完整闭环
- LKM：检索向内不向外、无 fill 标注、insufficient 是证据分组档位非三态
  认识论、无矩阵空洞编译、回灌是愿景（自认 first stages）
- EviMem（2604.27695）/S2G-RAG（2604.23783）：闭环向外但缺口是查询时
  临时诊断，不持久；无 fill 标注
- 其它部分重叠：SGHA（2608.17501，gap 检测→研究问题，不做外部检索）、
  CROWN-QA（2608.04591，三态负答案评测无 records）、history-research-
  agent（开源，本地档案非外部文献）
- 层次(i) gap 检测赛道饱和（GAPMAP 2510.25055 已分 explicit/implicit）
  ——检测环节不可作独立卖点

**组合创新纪律警告**：这现在是"两个各有前作的半环的组合"——必须做
ablation 证明组合收益（只有持久缺口无外部闭环 vs 只有外部闭环无持久
缺口 vs 完整链），否则会被审稿人拆成"LKM 子集+EviMem 子集"。
**动作建议**：抢时间窗口；把"fill 标注+缺口词汇合成"做成可验证的独立
贡献点；叙事重心不放 absence records 本身。
威胁排序：LKM（若其 §6.2 愿景落地+外向检索=直接覆盖）> EviMem/S2G-RAG
（若把临时 gap 持久化即撞车）

### 2.2 LKM 深核判决（agent A，全文+官网+SKILL.md 一手，档案 lkm-check-0926.md）

**LKM（arXiv:2609.27297，2026-09-23，深势科技/AISR+鄂维南系 22 人，
ICLR 2027 在审，lkm.bohrium.com 已产品化计费）——威胁级：极高**

被占位（一手引文已核）：
- **预编译 KB→typed tools→benchmark 报数的范式主干**：40M 篇 reasoning
  graph + 70M 摘要层离线编译；REST typed API（10 endpoints/9 scopes）+
  CLI + agent skill 三通道；"expose the same operations as tool calls
  inside an agent runtime"
- **n 元推理结构**：Def 7 inference factor = "maps an ordered premise set
  to a conclusion and records its method, parameters"——与我们 n-ary 超边
  语义等价（连 parameters 都带），旧卖点"n-ary 形式化真空"已失
- **负知识一等公民**：research-gap records（6 类）+ insufficient 证据档 +
  weakpoint_of 边 + negative outcomes + open_question 检索 scope
- **ScholarQA citation F1 报数**：CS 54.06 / Multi 57.79（固定 GPT-5.4，
  graph vs search +5.71/+6.67pp，paired CI 不含零）——"预编译图上下文
  提升引用忠实度不损答案质量"的核心主张已被抢先发表
- 规模与产品化远在我前（40M vs 430/2,355；计费 API v4 已上线）

我们仍独有（其论文+产品文档 grep 零证据）：
- **抽取保真的机器可验证质检**：其 "original wording" 只是文档声明；
  全文无 verbatim 回定位/entailment/抽检描述；抽取 LLM 未披露；质检流程
  空白。我们=五道确定性门+repair-or-drop+canary+95% first-pass+0.11%
  混淆审计——且 2607.20527（核验器普遍不可靠审计）恰证此维度紧迫
- **schema 治理式演化**：其本体冻结（产品 schema 比论文更窄且固定）；
  我们=仲裁通道+usage audit 减法+dims_new+版本化生长+FG7 迁移案例
- **演化/回灌收益的严格 ablation**：其 §6.2 回灌是愿景、§6.3 自认
  "first stages"无实验；我们 backflow 飞轮 live 验证（但同样欠量化=P2）
- **受控科学设定**：同答题模型/同语料/同预算的三臂对照+规模轴
  （16→53→430）；LKM 是产品级评测（自家 40M KB+GPT-5.4，无语料控制）

口径警示：LKM Multi 57.79 与我们 65.03 **不可直接比**——他们开集 40M KB
+GPT-5.4 答题，我们闭卷 430 篇+本地 27B；子集规模/harness/引用匹配规则
均需对齐后才可比。论文中避免数字对撞，主打受控设定差异。

**叙事冲击与重构方向**：
1. "首个编译文献 KB 打败 RAG"的大故事**没了**——LKM 三天前以 1000x 规模
   发表并产品化。任何投稿都必须引用并区分它（concurrent 辩护弱：对方
   先挂出+在审 ICLR 2027）
2. 但 LKM 反向验证了范式正确性——可转为顺风："独立同期工作以产品规模
   证实编译路线；此类 KB 的**内容保真度无人验证**——我们提供验证层与
   受控证据"
3. quote-first 主张必须升级措辞：从"有逐字引文"→"有**机器验证**的逐字
   引文纪律"（引文存在性不独有，机器验证才独有）
4. 我们的差异化生存空间收窄至三点：验证（质检门/canary/治理）、受控
   ablation（哪部分编译买到什么收益+规模轴）、生长闭环（答题中库生长的
   最小实现+量化）

### 2.3 竞品最终判决表（agent A，一手核验；详档 .research_tmp/*-check-0926.md）

**已核完的关键行**（引文均一手）：
- **LKM**：见 §2.2。范式逐环节同构+规模/产品化碾压。威胁极高
- **DIAL-KG**（2603.20059+Springer 正式刊出）：MKB="governance hub and
  evolutionary memory"，auditable add/modify/retire batch 粒度+三重验证
  （LLM judge）——治理演化最接近我们；但无对外 QA、不开源、非科学域。
  威胁中
- **Intern-Atlas v2**（2604.28158）：940 万条边全带 verbatim span、三
  typed operators、hosted API——(a)引文(d)typed 访问正面重叠；但 binary/
  static/无演化/无 QA 轨；验证=LLM audit 非机器校验。威胁中高
- **AgentCAT**（2602.18479）：⚠️ 旧"ADD-only"标签需修正——有 Schema-
  Evolution Agent+backward compatibility；催化域专用（Gemini-2.5-Pro），
  "头号必比"可比性需重估。威胁中低
- **EGT-KG**（2609.00479）：typed KG+evidence 节点溯源+两步检索答题，
  llama3:8b +14.67% vs vanilla RAG、QASPER cross-paper F1 +5.9%——最近邻
  路线，**须引为 baseline**；但 30 篇单域、无演化、无验证。威胁中
- **Crase**（2608.24809）：查询时引文图 1.5-hop 有界探索+entailment 剪枝，
  LitSearch recall@50 0.3659 vs DR agents 0.1220（3x）成本 1/5——检索时
  策略非编译库，与 lineage_walk 局部重叠。威胁低中
- **SimScholarSearch**：verl-based RL 训练环境非 KB（源码实锤 9 工具全为
  检索 API，S2Graph=无类型引文边）。威胁低
- **SciLENS**：repo 源码实锤"KG"=paper cites paper 单关系，元数据级零内容
  抽取——"12M 记录 KG"名实不符。威胁低
- **2607.20527（盟友证据）**：同一输出 unsupported 率随 verifier 严格度
  3%→18% 漂移、negative agreement 仅 0.27-0.30、OpenScholar/PaperQA2
  "take the checker for granted"——**"LLM 验 LLM 不可信"的外部实锤**

**发现层新高威胁（abs 级已核，细节未核）**：
- **ASKS**（2608.29612）："scientific knowledge compilation"——LLM 产
  Wiki 视图+确定性检查→GraphDelta→持久状态整合，每次 ingest 是可检查
  状态迁移；56 篇时序编译实证。**与我们几乎同构，须优先补核**
- **ScholarStack**（2609.23735，09-20）：三层资产（source-grounded
  statements/domain organization/evidence-grounded syntheses）+统一接口
  按任务返回证据粒度视图+保留 study conditions/verification status
- **Agents-K1**（2606.13669）：全文→agent-native 科学 KG、typed 关系、
  4B IE backbone GRPO 训练；动机叙事与我们一字不差（批评"论文压扁成
  abstract+flat cites"）
- **Buehler 组超图**（2601.04878）：1100 篇→161k 节点/320k 超边，明确
  论证 pairwise KG 无法表达高阶——n-ary 占位（与 LKM Def 7 双重）
- **Tree-of-Concerns**（2608.20777）：多 agent 辩论抽 unstated
  limitations+ToC-Bench（414 篇/1,905 条）——absence 相邻最强占位，但
  语义推断型局限挖掘非字段级三态记录、无逐字引文纪律
- **Karpathy LLM Wiki gist**（04-05，1600 万浏览传说+5 个开源实现）：
  "compiled once and then kept current, not re-derived on every query"
  ——"编译 vs 检索"大众叙事名分已被占（个人知识库语境，无验证/无
  benchmark/无溯源评测）
- CTIFoundry（2608.18613，CTI 域 build-time 物化同款论证框架）、
  Valhalla（2608.15193 五层封装）、CoEvoKG（2608.01904 写回图）、
  ArticleMiner（2609.25607 ontology-guided 表格定量）、Hakken/HyGRAIL
  （gap=预测目标非 absence 记录）——中低威胁各就位
- gap 驱动检索家族补全：SEAL-RAG/GRAIL/AdaGATE/EviMem 均为答题时即时
  gap→micro-query，非建库期持久 absence

**未核实（三路断连，旧认知不可直接引用）**：SciAtlas/Mechanist、
Hyper-KGGen/HGNet/SCION、GraphRAG 家族 2026 现状

### 2.4 空白点判决（agent A 综合，LKM 冲击后）

**已被占死（不能再当独有卖点）**：
1. "论文语料→预编译 KB→typed 接口→benchmark"范式本身（LKM 发表+产品化；
   ASKS/ScholarStack/Agents-K1/CTIFoundry 在涌；Karpathy 占大众名分）
   ——范式新颖性归零，只剩执行深度可辩护
2. n 元高阶结构形式化（LKM Def 7 + Buehler 32 万超边双重占位）
3. gap/负知识广义领地（LKM research-gap records + ToC unstated limitations）
4. "图上下文提升引用忠实度不损答案质量"（LKM ScholarQA paired CI 已发表）

**仍然真空（一手证据支撑）**：
1. **机器可验证的抽取质检门**——全部已核系统无一做确定性机器验证
   （LKM=文档声明/Intern-Atlas=LLM audit/DIAL-KG=LLM judge/AgentCAT=
   LLM review/EGT-KG=无）；叠加 2607.20527"验证器不可靠"实锤——
   **"确定性回原文定位+数值逐字校验+repair-or-drop"是我们最硬的独有位**
2. **字段级三态 absence 一等公民**（某论文对某字段 not_reported/
   explicitly_stated/cannot_tell+逐字引文）——LKM 是开放问题级、ToC 是
   语义推断级、SEAL-RAG 系是查询时级；细分仍空，但措辞必须与 LKM 显式
   区分
3. **治理式演化+usage audit 减法+演化收益的严格 ablation**——DIAL-KG
   最近但无 QA 不开源无演化 ablation；"演化收益的严格对照证明"仍真空
   （与旧定位"首次严格验证 in-loop 收益"一致，现在更值钱）
4. **canary 合成哨兵自校验**——所有已核系统均无
5. 确定性 typed tools 优先/检索兜底的纪律形式——仍独有但被 LKM typed
   API 面弱化

**最紧急行动项**：ScholarQA 数字口径对撞——LKM 引 Asai et al. 2024
（OpenScholar）的 ScholarQA-CS/Multi；我们的 Multi-108 来自 AI2 asta-bench
数据形态（scholarqa_bio/multi/neuro jsonl）。很可能根本不是同一数据集
（待 agent B 的 asta-bench 核验确认）；即便同源，语料设定（开集 40M vs
闭卷 430）与答题模型（GPT-5.4 vs 27B）也不同。论文必须先讲清口径。

**0926 离线核（数据形态）**：我们的 scholarqa_multi.json 字段=annotator
（人名：norman/benjamin/bohao/...12 人）/ctxs（含 citation_count/url）/
input/output/subject——**FutureHouse ScholarQA-Bench 形态**（AI2 asta-bench
包装的 Multi 轨）。LKM 的 ScholarQA 引 OpenScholar（Asai et al.）谱系，
其评测细节论文未给（agent A 已核："细节未给→未核实"）。判定：两数字
**当前不可比**（题集谱系待确认+语料设定不同+答题模型不同），我方预注册
纪律本来就规定"官方数字永不进对比表、协议不同不混表"——沿用即可，论文
中引用 LKM 数字时标注设定差异。

### 2.5 基准格局判决表（agent B，全部一手核验）

**主战场判定**：
- **ScholarQA-Bench（OpenScholar）已发 Nature 正刊**（s41586-025-10072-4，
  arXiv:2411.14199）——我们 Multi-108 主场基准的公信力已坐实。判分=
  citation F1 + ingredient；AstaBench 论文承认其局限（ingredient 可被
  gaming、反映两位标注者偏好）
- **AstaBench/ScholarQA-CS2（AI2，arXiv:2510.21652，ICLR 2026 Oral）**：
  11 基准 2400+ 题；ScholarQA-CS2=长文综述报告（test 100 题，源自
  OpenSciLM 真实用户查询）；判分四维平均（citation recall/precision+
  answer relevance/coverage），judge=gemini-2.5-flash；工具面=Asta
  Scientific Corpus（S2 底座）经 MCP 提供 8 工具，**snippet_search 是
  全文级且可限定 paper IDs（=支持闭卷语料设定）**；日期 cutoff 2025-05-01；
  **test split 未被扣留可本地跑**（上榜走 HF 提交）——与我们 broker 同构，
  迁移成本低
  - test 榜现状：ScholarQA-CS2 最高=ReAct+gpt-5.4 0.904；总榜第一
    ReAct+claude-opus-4-7 0.580。**榜上全部是检索式 agent，无预编译 KB
    系统上场**
  - 判分软肋（论文自认）："citation metrics do not verify that citations
    refer to real papers or that quoted snippets actually appear in the
    cited sources"——与我们"检测仪器不可信"纪律同向
- **⭐ 生态位判决：全生态预编译 KB 路线公开报数的只有两家——我们
  （Multi F1 0.6503）与 LKM（CS 54.06/Multi 57.79）**。"预编译结构化
  KB+typed tools"在 AstaBench 生态仍是无人上场的空位=差异化窗口

**其它基准**：
- PaperArena（2510.10909 v4，**非 FutureHouse——旧记忆归属修正**）：
  tool-augmented 跨论文 agentic 推理，最强 agent 仅 38.78%（hard 18.47%）
- PaperScope（2604.11307+**ACL 2026 Findings** 2026.findings-acl.394）：
  已升级 2,000+ 篇 AI 论文 KG+2,000+ QA，闭卷固定语料——与我们范式形态
  最合的第二目标；旧"仅 16 篇 gold"结论需用正式版重验
- PaperQA2：CalVer v2025.12.17，新增表/图/公式/非英语多模态解析，9.2k
  stars 活跃——必比 baseline 仍有效
- LightRAG：无跟进论文，工程迭代（RagAnything 合并等），39.8k stars——
  baseline 对比在 2026-09 仍有意义
- LitSearch：停在 EMNLP 2024，被实质取代；SciConBench/MDAQA：本轮未核实
  （旧结论引用前须重验）

**判分可信度警报（上场前必读）**：
- Deep Research, Shallow Evaluation（2603.06942，**AI2 自家元评测**）：
  ScholarQA-CS2 人机判分一致性 system 级 Kendall τ=0.467（剔除 Elicit 后
  0.800）、instance 级 68.1%
- Unraveling the Ai2 Asta Citation System（2606.08301）：第三方实测 Asta
  引文"notable instability across identical queries"
- 含义：上 AstaBench 需双仪器披露；ingredient rubric 管线开源在
  ai2-scholarqa-eval，我们的 Citation F1 需换算四维+补 coverage

**新发现清单（顺带，一手 ID）**：AutoResearchBench（2604.25256）、
ScholarQuest（2606.20235）、PaperMind（2604.21304）、Improving Attributed
Long-form QA（2603.27435）、DR Tulu（2511.19399，rubric 演化训练）

### 2.5b ScholarStack 全文深核判决（主协调员亲核，2609.23735v2 HTML 全文
下载+关键词普查+定点提取；ScholarSeed AI Team/Wotao Yin 系 19 人）

**它占了（比 abs 级判断严重得多）**：
- 三层编译资产：L1 Scientific Fact f=(statement, context, evidence)（四型：
  finding/method/hypothesis/limitation；定量值绑定 subject+setting）；
  L2 域组织（taxonomy+canonical entities+typed relations，确定性规则+
  模型仲裁合并，未解析保持 uncertain）；L3 带 scope 的跨篇综合
  （s=(account,scope,basis)，证据标 supporting/opposing/limiting）
- **per-fact verification status**——但为模型语义核验（归因正确性+内容/
  限定保持），失败保留在状态里；无逐字回锚/字符串/数值校验（原文
  确认："does not describe verbatim quote anchoring, string matching,
  or numeric validation"）；"a fact records what the source asserts
  rather than certifying its truth"
- **条件感知可比性（概念级）**：context 槽记录"assumptions and conditions
  under which the report holds"；"comparability is still decided by the
  contexts recorded in F"；"shared membership alone does not justify
  aggregating their results"；答案"combines compatible findings and
  preserves contextual differences and unresolved conflicts"——但为
  **记录的上下文串+答题时推理**，非编译期物化的等价类
- 受控评测设计与我们同款：matched base model（Qwen3.8-Max+Codex）、
  闭卷、层消融（L1/L1+L2/full）、token 成本账；**MDAQA 797 题多论文 QA
  18.41 vs Full-text 8.46**（五维 0-4 判分含 calibration 3.90 vs 0.92，
  GPT-6 Astra judge）；QASPER 416 篇/PeerQA 208 篇单论文 QA（质量略输
  Full-text=93% F1 但 34% tokens）；LitSearch 固定语料 64,183 篇（资产
  建在 55,631 篇上）；实验设计生成/想法生成/NLPCC 主张核验共 10 场景
- 失效语义雏形：源修订→"its dependents require rechecking"；L3 不继承
  证据的 verification
- 轻度生长：task-derived objects 可选注册（身份+引用检查后）
- 中心问题设定："task-specific research bottleneck"（跨任务复用研究资产）

**它没有（关键词普查零命中+定点确认）**：
- absence/not_reported 零命中——**无缺失性事实记录**（limitation 是内容
  型 fact，非覆盖负空间）；无推导空洞；无缺口驱动检索
- as-of/time-travel/temporal 查询零命中——**无时间轴**
- growth 零命中——**无库生长飞轮**（外部检索只在 agentic search 基线
  工作流里，不回流进库）
- 无 canary/哨兵自校验；无确定性机器验证（核验是模型语义级）
- 无物化视图（Knowledge View=任务时从三层资产**选组合装**，五通用操作
  retrieving/selecting/expanding/comparing/synthesizing，非编译期物化的
  矩阵/覆盖网格/谱系闭包/成对差值）
- 无治理式 schema 演化（taxonomy revision prompt 存在但无仲裁通道/
  使用审计减法/残差驱动版本化）
- 无答题层机器强制链（笔记回指门/值锚定门/absence 断言门——其
  calibration 靠 judge 打分不靠门拦截）
- 无需求测量→接口演化环路记录

**对叙事的含义**：分层资产+条件记录+verification status+matched-model
多任务评测+层消融=全部被占；**"编译资产提升跨论文任务"的受控证据也
被占（MDAQA 18.41 vs 8.46）**。剩余 delta 收窄到：机器强制（确定性
验证+答题门）vs 模型语义核验；物化等价类（分带）vs 记录的上下文串；
三态 absence+负空间；时间轴；治理演化；canary 测量学；生长飞轮。
ScholarStack 的 MDAQA 数字未来可作同基准对话对象（其 full-text 臂
=gold 论文全文，设定与我们闭卷 union 语料不同，需口径说明）。

### 2.5c v6 时代竞品更新判决（agent F B 轴，一手全文/repo 核验）

- **SciAtlas（2605.22878）**：scoop 硬阈值**未触发**（内容四轴仍未补）✅
- **Lacuna（2606.26246）**：维护语义/冲突感知/内容四轴三问**全否**✅
- **Agents-K1（2606.13669，全文核）**：混合偏供给侧——schema 一次定型后
  对 2.46M 篇无差别抽取，demand/query-driven 全文零命中，无查询需求
  回流；主线**严格二元**（⟨head,type,rel,tail,type⟩+verbatim span+
  confidence），n-ary 超边只是 General-KG 的派生 upgrade view（合并共现
  二元边），非原生、无 qualifier 语义槽；验证只在评测/训练期（LLM-judge
  抽样 F1 79-87），**无构建期逐字回锚/数值校验**；演化=实体级增量+用户
  闸门命令，contradict/invalidate/obsolete 零命中——**无冲突处理、无
  命题级失效传播**。我们的失效代数+构建期确定性验证对它均成立
  - ⚠️ 警示①：其 Prop 2/3 已形式化论证超边价值（"binary projection
    collapses…temporal qualifier or multi-argument experimental
    condition"）并占住 "agent-native KG" 话语位——我们的条件/维度语义
    须以"**原生 n-ary 记录+dims 槽+使用条件强制** vs 派生投影视图"
    划界并正面引用
  - ⚠️ 警示②：repo 建成即停更 3.5 个月（唯一 commit，42 star），evolve
    命令不在公开 repo——可复现性弱，对比只能用其 HF 1M 子集
  - 影响：S2=3（2 自引）/OpenAlex=0，占位威胁有限

### 2.5d 限定词/认识态轴查新（主协调员亲核，arXiv API+abs 一手）

- **negation/speculation、hedge detection**：arXiv 近年零相关命中——经典
  BioNLP _scope detection（BioScope/NegBio 时代）是**句级语义标注**，
  与"库级使用条件+三重强制+测量"粒度与目的均不同，划界容易
- **qualifier+claim extraction**：无直接近邻；TRACE-CTI（2607.24563，
  CTI 域"auditable post-extraction governance"）为异域旁证
- **⚠️ Eigenius（2608.04457，2026-08-05，abs 一手已核）**："A Typed
  Knowledge-Graph DBMS with Epistemic Stratification"——**必引近邻**：
  "turns data provenance into a structural invariant"、"epistemic status
  (declared/observed/derived/verified) is enforced as a strict
  commit-time invariant"、Nature 研究端到端重算（52 结论保持+4 处机器
  核验差异）。它先占了"认识态作为强制不变量"的**措辞**。
  delta：①它是通用类型化 KG DBMS 内核（依赖类型论+institution 边界+
  内容寻址存储+Lean 4 证明检查），非文献抽取编译管线；②其认识态由
  记录系统在提交时赋值，我们的 stated/demonstrated/cited 是**从论文
  修辞证据里抽出来的**（本篇声称 vs 本篇实验支持 vs 转述他篇），带
  逐字引文与违规残差治理（FG7 案例）；③无可比性条件、无 absence、
  无 QA 评测、无语料规模（单研究重算）。
  **叙事用途**：独立同期收敛（DBMS 内核方向 vs 文献编译方向都走向
  "认识态不变量"）=方向正确性的第三方佐证；引用划界后反而加强
  "使用条件"叙事的时代性。v6 编辑部决议当年已把它列必引——延续。

### 2.5e 失效代数轴查新判决（agent E 子代理落盘，NOVELTY-AXIS3-
PROVENANCE-VIEW-MAINTENANCE-20260926.md，四渠道交叉一手核）

**判决：「provenance/lineage 驱动的 LLM 抽取管线下游视图失效检测+局部
重算」完整闭环的先例不存在**（截至 2026-09）。
- 唯一部分占位=AIET 2026（单作者非主流期刊、传播止步打标、人工触发、
  无重算无事务、代码未释放）——占位强度弱，必引并区分
- 三条近亲线各缺一环：attribution 线（RARR/ALCE/AIS=一次性文本修复/
  度量/人评协议，无持久派生轨迹）；agent 记忆线（Zep/Graphiti 有
  lineage 持久化+删除传播，无下游物化视图）；经典 IVM 线（Enzyme/
  OmniTable 闭环完整但全部假设确定性可重放算子——我们的 Q 不可靠）
- **Agent Traces 综述（2026-06）把"provenance 使能的 selective
  invalidation+下游影响定位"明确列为开放挑战**——我们的四元闭环
  （轨迹持久化×失效检出率作为一等测量对象×blast radius×局部重算+
  两阶段提交）在该综述问题地图上属空白
- 借力证据三处独立文献指向我们机制必要性：Agent Traces §5.2/§7、
  Valhalla §11.5 future-work 自认、RAID "cascading false revisions"
  风险自述；STALE 基准=检测端已基准化、维护端无人做
- GraphRAG 论文全文 provenance/invalidat/incremental 等 **0 命中**
  （repo 层有 text_unit_ids 溯源列+日期合并增量，无失效反查）；
  LightRAG "incremental update" 逐字核实=**只做新数据合并插入**，
  无 lineage/失效/传播——related work 现成引文
- 对 Terms-of-Use 叙事的含义：失效代数=「条款被破坏时的召回程序」，
  现在是**带独立查新判决的开放机制**，可从支柱升为机制组第四件

### 2.5f 轴4查新判决：lineage 完整性/探测器召回作为测量对象
（agent E 子代理，全部 DOI/arXiv 一手验证，证据在 .research_tmp/axis4/）

**判决：部分先例存在、完整组合不存在。**
- DB 理论侧对 lineage 表示的质疑成熟：Ré&Suciu 近似 lineage（PVLDB
  2008）、Cheney et al. 依赖溯源**不可计算**（MSCS 2011）、Amsterdamer
  semiring 在差/否定下 soundness 不可能保全（TaPP 2011）——但测的是
  表示保真/复杂度，非探测召回；SIGMOD 2026 tutorial 已把"近似
  provenance 实践可接受"列常态、全文零提 LLM/不可靠抽取器
- "探测器召回被系统测量/统计保证"两半各有先例：DB 侧 SUPG（PVLDB
  2020，proxy 选择带 recall 保证）+Lotus（LLM 算子对 gold 有精度保证）；
  LLM 侧 Text2KGBench（ISWC 2023，抽取器 P/R 基准化）+FActCC/RAGTruth
  （检测器可靠性评测范式）
- ⚠️ **必引并正面区分**：WikiMonitor-Onto（JAAI 2026，单作者小刊）已把
  "LLM 维护 KB 的 staleness 探测召回对 gold 测量并报 P/R"窗户纸捅破
  （P 0.824/R 0.560，62 概念单标注者 gold）——但其传播图=概念级本体
  非 record 级派生轨迹、staleness=时间过时非源失效触发、无 blast
  radius→局部重算闭环
- **无先例的完整组合**：record 级派生轨迹+源失效反查→blast radius→
  局部重算的物化视图维护回路+零 token 地面真值自审验证失效探测器
  召回。新颖性措辞纪律：从"完全无人做过"收敛为"**组合首次**+与
  WikiMonitor-Onto/SUPG/Text2KGBench 的明确差异"
- 附带证伪两条错误线索（"Decision problems for provenance semirings"/
  "Putting lips on cowbirds"三渠道零命中不存在）——引用纪律执行到位

### 2.5g DB 理论轴收口稿（agent E，**训练知识级、全部未一手核实**，
引用前须逐条落 ID；轴号注意错位：其轴2=provenance/轴3=uncertain DB/
轴4=IVM，与已入档的轴3/轴4文件按各自标题区分）

**初步判决（未核实状态）**：
- Truth discovery/data fusion：整条线公理=冲突即值分歧、消解即源可靠性
  投票——**不存在"条件等价性"作为冲突判定前提**；伪冲突在该框架会被
  投票静默消解，我们反向（先判带、带内才谈冲突、永不自动消解）。
  ⚠️ 唯一威胁线索：VLDB 2018 "domain-aware multi-truth discovery"
  （多域多真值）——若坐实为语境键控多真值，机制(1)降为改进型 delta
  - **✅ 已核销（0926 主协调员 OpenAlex 一手，DOI 10.1145/3187009.
    3177739，PVLDB 2018）**：其 "domain"=**源的领域专长度**（按数据
    丰富度推断源在不同实体域的可靠性），非测量条件；输出仍为贝叶斯
    置信度打分的多真值集（自动消解路线）；全文无条件/语境等价概念。
    **威胁降级为远亲必引**（multi-truth 线承认多真值，我们承认条件化
    多值+拒绝自动消解），机制(1)"条件等价性作为冲突判定前提"在
    truth discovery 线内确认无占位。同线必引：SmartMTD（1708.02018）、
    Truth Discovery from Conflicting Multi-Valued Objects（DOI
    10.1145/3041021.3053374）、LTM VLDB 2009、Li et al. survey 2016
- Uncertain DB/c-tables/PDB/DeepDive：全部概率化路线；我们的离散认识态
  +仲裁制度化+探测器召回仪器化在 PDB 框架无对应物。DeepDive=头号
  划界必引（同"抽取→KB"，不确定性处理正交：概率推理消解 vs 仲裁+仪器）
- 语义查询系统（DocETL 2410.12189/Palimpzest 2412.10422/LOTUS
  2407.11418/CAESURA）：全部查询时算子编排/优化，无一做建库期需求
  驱动物化
- Workload-driven 经典线（Chaudhuri-Narasayya/Agrawal VLDB 2000）：
  SQL 代价统计→物理设计；我们=agent 失败日志→语义视图+接口。载体/
  信号/产物三重不同，作 M3 谱系锚点必引。**M3 占位排查=全案承重墙，
  未经 2025-2026 地毯核实不得写入论文声称**
- OLAP-KG：Papers With Code=形态最近占位（method×dataset×metric 人工
  表格，无带/无溯源/无失效维护）；RDF cubes=远亲。"条件带+溯源+失效
  维护的比较立方"未发现占位（未核实）

**DB 审稿人四个一击必杀点（防御预案）**：
1. "DeepDive 换个不确定性表征"→防御：Q 的概率校准本身不可信（与
   核验器不可信同源），故用离散认识态+制度化仲裁，正交路线论证做实
2. "multi-truth discovery 已做条件依赖真值"→VLDB 2018 线索必须先核
3. "M3=auto-admin 搬到 LLM"→防御：失败模式信号 vs 代价统计信号的本质
   差异配实例（card 空返回 42%→12% 这类修复史）
4. "DocETL 系 agentic 优化已覆盖需求驱动"→一句话斩断：它们优化用户
   给定的 pipeline，我们从 agent 失败日志生成 KB 结构本身

### 2.5h 循证医学轴判决（agent F，全部一手 DOI/Europe PMC/官方文档核验）

**核心结论：概念词汇层已挤满，机制层四点均无先例——差异化靠机制不靠
概念，且概念窗口在收窄。**

- **Evidence Gap Maps**：最逼近一篇=Diabetes Camp Paradox EGM（2026，
  DOI 10.1111/hex.70833）——已有 "five epistemic layers"（含 what do we
  not know）、未报告 vs 报告了区分、缺失按风险分级。但四个机制点无
  先例：①记录级三态 absence+穷尽性依据（EGM 格态=研究计数 0-2/3-9/
  ≥10，非认识态）②推导空洞（同文兄弟推断、无需外部注册库）零先例
  ③持久化机器可查询缺口资源（EGM=一次性论文图）④机器自动构建
  （所获 EGM 全人工，LLM 自动构建 EGM 未见发表）。gap 驱动行动本身
  不独有（WHO Living EGM 驱动肿瘤分类迭代，DOI 10.12688/
  openreseurope.23926.1）
- **Living SR**：更新粒度=论文纳入/review 级（JBI 2024 协议+
  RobotReviewer LIVE 试点 JCE 2023）——**无命题级失效传播先例**，
  失效代数无对应物
- **GRADE**：证据体级/outcome 粒度/人工分档（Guyatt JCE 2011，
  citedBy 7831）vs 我们记录级/修辞类型/构建期自动——两轴正交；
  **论文用"强度轴"措辞须主动引 GRADE 声明粒度差异**，防"重新发明
  GRADE"误判
- **Trialstreamer/RobotReviewer/Elicit**：全线**无构建期机器保真门**
  ——全部是存 snippet 供人核+抽样人工评测（Trialstreamer JAMIA 2020
  质检=事后人工抽检 100 篇；Elicit 官方立场=裁决权交人）；**跨研究
  数值可比性分带：检索范围内零命中**
- **ORB 线**：全部先例须外部注册库作 ground truth——我们的推导空洞
  从论文内部结构推断，**恰好覆盖无注册库领域（ML/AI）的空白**；
  三态缺失分类无先例（审计只有二元 discrepancy）

**行动含义**：论文必须主动引 EGM 系并定位为其"自动化+记录级升级"；
引 GRADE 声明粒度正交；护城河=机制（三态坐标/推导空洞/持久可查询/
机器构建），概念层若被抢注（尤其 LLM 自动构建 EGM 发表）会进一步变薄
→ 时间窗论据再添一条。

## 2.8 查新总闭合状态（0926）

Terms-of-Use 三条款+召回程序+验证支柱，每条的占位判决均有≥2 路一手
证据支撑：
| 机制 | 占位判决 | 必引划界对象 |
|---|---|---|
| 可比性条款（条件等价分带） | **无占位**（truth discovery 线无条件等价前提；EBM 线分带零命中；ScholarStack=条件串非等价类；PwC=形态近亲无机制） | multi-truth 线（PVLDB 2018 已核销为远亲）/ScholarStack/PwC/RDF cubes |
| 归因条款（记录级认识类型） | **无占位**（GRADE 正交；Eigenius=DBMS 提交期赋值；LKM=belief 数值） | GRADE/Eigenius/LKM/ALCE-AIS 线 |
| 否定条款（三态 absence+推导空洞） | **机制无占位、概念词汇拥挤** | EGM 系（尤其 Diabetes Camp 2026）/LKM gap records/ToC/CROWN-QA/ORB 线 |
| 召回程序（失效代数） | **完整组合无先例**（措辞纪律：组合首次） | WikiMonitor-Onto/SUPG/Lotus/Text2KGBench/Zep/Enzyme/LSR 线/Agent Traces 综述 |
| 验证支柱（机器保真门+canary） | **无占位**（竞品全线+EBM 线双确认） | 2607.20527/AstaBench 自认/Intern-Atlas LLM audit/ScholarStack 语义核验/DeepDive |
| 生长飞轮（机制 2/3） | **组合无占位**（轻扫口径） | LKM §6.2 愿景/EviMem 系/Dual-Layer Memory/Sleep-time compute |
挂账（非阻塞）：M3 需求驱动 2025-2026 地毯排查；DocETL 系 2026 形态；
ScholarStack 全文复核（投稿前硬前置）。

### 2.6 三线叙事占位判决（agent C，一手核验）

**已被说满（放弃首创主张）**：
1. 范式主干（语料→离线编译→KB→agent 消费→QA 验证）：LKM 全环节占据。
   **"预编译图上下文提升 citation F1 且配对显著"已被抢先发表**——我们赢
   LightRAG/PaperQA2 的结论方向被预支
2. "编译"术语与比喻：Karpathy（大众，"compiled once, kept current"）+
   **ASKS（学术，"scientific knowledge compilation"，2608.29612）**+
   CTIFoundry（跨域）三层占满——不能再以术语首创自居
3. n-ary 超边+论文语料：LKM Def 7 + Buehler（2601.04878）+ HyperRAG
   （2602.14470）
4. 负知识/gap 检测环节：LKM gap records + ToC（2608.20777）+ CROWN-QA +
   GAPMAP——检测不稀奇
5. typed tools 查结构化 KB：LKM 生产级（10 endpoints+SKILL.md 工程范本）+
   Intern-Atlas 算子级

**还没人说（按强度排序=我们的剩余空间）**：
1. **抽取保真的机器验证**——LKM 全文 grep verbatim/entail/hallucin 零命中、
   抽取模型未披露；EGT-KG 无验证循环；Intern-Atlas=LLM audit；DIAL-KG=
   LLM judge；AgentCAT=LLM review。**"带机器可验证溯源的编译 KB"无人
   占据**，且 2607.20527（unsupported 率 3%→18% 随 verifier 漂移）是
   现成方法论盟友。最硬差异化
2. **schema 并进演化×下游收益的严格 ablation**——全部竞品本体冻结
   （LKM/Intern-Atlas/EGT-KG）；DIAL-KG 有治理无"演化 vs 冻结"直接
   ablation；AgentCAT 演化但域专用无下游 QA。真空成立但必须配 ablation
3. **持久 absence 坐标+fill 标注完整链**——17 组检索无人做全（EviMem/
   S2G-RAG/AdaGATE/GRAIL/SEAL-RAG 全是查询时临时 gap）；措辞轴=
   "transient query-time gap diagnosis vs persistent KB-level absence
   coordinates"
4. **"图式 RAG 在多跳/聚合上系统性失败"的受控实证**——未发现专门研究
   （未核实项中价值最高）；**微软 GraphRAG 官方自认 maintenance mode
   （repo README 实测："won't be accepting new PRs or implementing new
   features"）=叙事顺风**；我们闭卷固定语料对照设计可补此实证位
5. **失效代数/violation 传播**——零占位

**线二 agent memory（部分未核实）**：Materials agent memory（2608.11224，
"inspectable facts+executable skills"）部分重叠；CoEvoKG 写回图但主线
RL；Letta/Mem0/A-MEM 未一手核查；科学文献域"经验编译成记录+视图"=
LKM §6.2 愿景层被占、实证层空

**线三 typed tools**：LKM 生产级占位主体；Intern-Atlas 三算子（SGT-MCTS
谱系重建撞"谱系视图"名）；SciLENS/S3 工具化答题是共识形态但知识层无
内容级结构（反衬）；Text2Cypher vs 预定义 typed tools 的区分论证无人占

**姿态建议（agent C 原话）**：不再讲"我们发现编译范式优于检索"，改讲
"编译范式已被确立，但没人回答**编译的可信度**（机器验证溯源）与
**编译的生长性**（schema 演化收益+持久 absence 坐标）——我们在受控
闭卷设定下给出这两者的严格证据"

### 2.7 记录优先 vs 图优先：架构级对比分析（0926，用户问题驱动）

**领域静默收敛的证据**（各家竞品的一手架构事实）：
- LKM：基本单元不是三元组而是 inference factor（有序前提集→结论+
  method+parameters）=**穿了图外衣的记录**（reification）
- ScholarStack：L1 基本单元 f=(statement, context, evidence)=记录形状；
  图只是 L2 的组织关系
- Agents-K1：2.46M 篇规模的"agent-native KG"主线**退回严格二元**
  （⟨head,type,rel,tail,type⟩），n-ary 超边只作派生 upgrade view——
  最大规模玩家用脚投票放弃了原生 n-ary
- ASKS：wiki 视图+GraphDelta，非三元组库
- 没有任何一家把"为什么离开三元组"说破——**明确的架构立场文是空位**

**三元组/图基底的七个结构限制 × 我们的对应设计**：

| # | KG 结构限制 | 我们的记录优先设计 | 实测/文献支撑 |
|---|---|---|---|
| 1 | n 元性：科学事实天生带限定（方法×指标×值×条件×规模），二元边要么 reify 要么丢 qualifier；超图形式上解决但身份/匹配/查询语言全不成熟 | 记录=原生 n 元类型化结构（dims 槽位系统），无需超图机制；reification done right：每条记录有身份/溯源/引文/认识态 | 自家史：超图时代 validate 全拒 n-ary 边、split 命名跨篇不收敛；Agents-K1 二元退却 |
| 2 | 身份脆弱性：图质量被实体消解上限卡死，合并错误沿遍历/社区检测静默传播，不可逆 | 身份与数据分层：registry 独立治理（确定性守卫+版本化+仲裁队列），记录保留 surface_raw——身份错误时记录层优雅降级（未解析 ref 仍可按表面名查；覆盖视图限治理实体、矩阵保留原始表面=PSV4 分级决策） | KG 无此分层：节点错=挂在上面的一切错 |
| 3 | 外延语义：边要么存在要么不存在，至多挂 confidence 浮点；无法表达 stated/demonstrated/cited、可比条件、否定举证责任 | 记录带认识态轴+条件槽+强度，视图与答题层消费它们（归因措辞/溯源追查/分带纪律） | confidence 浮点表达不了"此数值在条件 X 下测得、仅与同带数值可比"；Eigenius 在 DBMS 侧独立收敛到认识态不变量（佐证） |
| 4 | 负空间不可表达：图存"有什么"；"没做什么"在开放世界假设下不可查，闭世界假设对抽取语料不健全 | absence 一等记录（三态+穷尽性依据）+推导空洞（paper-anchored 规则防爆炸）+provenance 分离 | 实测：无治理时推导空洞 375→15,512 爆炸；Reiter 1978 CWA 老问题在文献 KB 的新解 |
| 5 | 数值可比性非图概念：值挂属性上，"A 的 87.5 和 B 的 86.2 能不能比"图拓扑表达不了；KG-QA 把子图丢给 LLM 现场推理 | 编译期物化可比性等价类（分带），compare 直接返回带对齐的行+带内排名；带内冲突→drift 仲裁 | 对比是 cube 操作不是遍历操作；ScholarStack 停在"记录条件串+答题时推理" |
| 6 | 溯源粒度与失效：KG 溯源通常边级（哪篇论文）；无字符级锚点就无法机器验证，无派生轨迹就无法失效传播 | quote+loc 字符级锚点→五道确定性门；provenance index（record_id→每个视图位置）→行级失效+blast radius+两阶段提交 | 全生态无机器验证（LKM 文档声明/ScholarStack 模型语义核验/Intern-Atlas LLM audit/Agents-K1 评测期抽检） |
| 7 | 本体两难：冻结（LKM/Intern-Atlas/EGT-KG）或运行时自由生长（污染）；ontology evolution 是 DB 老领域但 LLM-KG 系无人实践 | 中间道路：冻结 schema+残差驱动仲裁升格（FG7 案例：最大违规类=缺失语义槽→v1.4）+使用审计减法+版本化 registry 生长 | DIAL-KG 最近亲但无 QA/不开源/无演化 ablation |

**访问层的架构后果**：KG 访问=遍历/模式匹配/Text2Cypher（脆弱）或
子图+LLM 现场推理（每查询重付综合成本）；我们=typed tools 返回
**物化的答案结构**（compare 给对齐行+排名、card 给档案、find_gap 给
双层负空间、as_of 给时点快照）——跨论文综合在编译期算一次（pair
deltas/祖先闭包/带内排名），查询期零 LLM。F12 判决实测：检索式访问
记录层仍有损（记录在库、检索没中→答"不存在"）。

**KG 的真实优势（诚实清单，防稻草人）**：多跳遍历表达力（我们保留
——谱系视图就是图，lineage 工具做遍历+祖先闭包）；成熟查询语言与
生态（我们用 typed tools 换确定性，代价是接口演化负担——G1 实测
空返回问题及修复）；图算法库（社区检测/中心性——我们不需要，因为
分析类问题由视图物化）。

**叙事含义**：这段分析给主方法叙事提供了**架构层的根**——
"记录优先、图为视图"是结构基础，"使用条件编译"是记录承载的语义纲领，
两者合成一个立场：**科学文献知识的正确单元不是三元组，而是带使用
条件、可机器验证的类型化记录；图/矩阵/覆盖网格是它的编译视图**。
我们自家超图史（建过、测过天花板、转过来）是"从内部知道 KG 弊端"
的可信度资产；领域的静默收敛（各家都退向记录形状但无人说破）是
时代性证据；明确的架构立场文+受控证据=空位。

## 三、叙事方向建议（草案，待 D 机制 2/3 判决后定稿）

### 3.0 前提判断（生态位重构后的现实）

1. **范式本身不再新颖**：LKM 已发表+产品化；"编译"术语三层被占
   （Karpathy 大众/ASKS 学术/CTIFoundry 跨域）；同范式在涌（ScholarStack/
   Agents-K1）。"首个编译 KB 打败 RAG"与"n-ary 形式化"与"gap 记录首创"
   三个旧卖点全部失效
2. **但 LKM 们把"范式有效"变成了共识**——这替我们扫清了"为什么要编译"
   的论证负担，把论文问题推进到下一层：**编译的可信度**（内容保真谁来
   验证）与**编译的生长性**（schema 演化/缺口坐标/库生长闭环）——这两轴
   经 4 路调研确认仍空
3. **时间窗紧迫**：LKM 09-23 挂出、ICLR 2027 在审、产品在迭代（API v2→
   v4）；ASKS/ScholarStack/Agents-K1 都是近两月的 abs。我们投稿时
   （最早 ACL 2027 一月截稿）这些都将是确立的 prior work。差异化主张
   必须在它们全文细节核清后定稿（尤其 ASKS 的"deterministic checks"
   与我们验证层的重叠度——**定稿前唯一必须补的深核**）
4. **我们手里的独占资产**（4 路调研交叉确认后）：
   - 机器可验证抽取质检（五道确定性门+repair-or-drop+证据分诊）——
     全生态无一家做确定性机器验证（⚠️ ASKS/ScholarStack 全文未核，
     见第 3 条）
   - canary 合成哨兵自校验（事实+陷阱双向）——所有已核系统均无
   - 治理式演化全套（冻结+仲裁+usage audit 减法+FG7 迁移案例+版本化
     registry 生长）——DIAL-KG 最近但无 QA 不开源无演化 ablation
   - 字段级三态 absence + 持久缺口坐标 + fill 标注 + mentions 回流的
     完整闭环——LKM 占"持久化"半环、EviMem 系占"检索闭环"半环，
     交点无人（D 机制 1 终判）
   - 受控闭卷三臂对照 + 规模轴（16→53→430）+ 判分纪律（官方协议逐字/
     gold 探针天花板/数据版本锁/配对 CI）——产品级评测（LKM）做不到的
     科学严谨性
   - 失效代数/violation 传播——零占位（但目前只是机制存在，无收益实验）

### 3.1 方向一（主推）："Verified Compilation"——可信的科学知识编译

**核心论点**：编译范式已被确立（LKM 等），但没人回答"编译产物可信吗"。
现有系统全部用 LLM 验 LLM（或根本不验），而外部审计已证明 LLM 核验器
本身不可靠（2607.20527：同一输出 unsupported 率随 verifier 严格度 3%→
18% 漂移、negative agreement 仅 0.27-0.30；AstaBench 官方自认 citation
指标"不核验引文是否指向真实论文"）。我们给出第一个**机器可验证保真**的
科学文献编译框架：quote-first 逐字纪律+五道确定性门+repair-or-drop+
canary 哨兵+治理通道，并在受控闭卷设定下证明：验证过的编译 KB 在同答题
模型下显著优于检索式系统，且规模越大优势越明显。

**创新性**：①确定性机器验证替代 LLM 自评（全生态首家，有盟友证据撑
动机）②canary 事实/陷阱双向自校验（首家）③治理式 schema 演化+减法
审计（首家完整实现）④"图式/ chunk 检索在多跳聚合负载下随规模退化、
编译保持"的受控实证（填补未发现专门研究的实证位；GraphRAG 官方
maintenance mode 作背景）

**实验支撑**：
- 已有：Multi-108 三臂显著（+0.136/+0.172 paired CI）；PS16→53 规模轴
  （+0.167 转正、ΔD 交互 +0.213）；canary 跨域记录（RL/granular/table
  三套）；postcheck 全量统计（95% first-pass/4.1% drop）；registry 人工
  审计（0.11% 混淆）；G2 A/B；F12 判决
- 需补：质检门 ablation（关各门→下游 F1/编造率变化，预注册 search-only
  消融臂已在 MULTI-PREREG 设计）；canary 敏感度分析；答案层 fabrication
  率三臂对照（我们门拦下了多少）；建库成本表

**风险**：①"验证=工程不是研究"的批评——对策：把 canary+门+治理升格为
方法论贡献，用 2607.20527 和 AstaBench 自认软肋做动机，用 ablation 证明
验证层有下游因果收益（不只是描述）；②门 ablation 还没跑——这是本方向
的实验债核心；③引用修复披露（见 3.4）；④单答题模型——加一个模型臂
更稳（或明示 scope）

### 3.2 方向二："知道自己不知道的 KB"——absence 坐标与生长闭环

**核心论点**：编译 KB 的独有价值不止"知道什么"，而是"知道自己不知道
什么"。三态 absence 持久记录+矩阵空洞推导=3,953 条缺口坐标；负空间
支撑三件事：诚实回答（"corpus 未报告"≠"文献中不存在"）、gap-driven
外部检索（用题面没有的词汇定向找填补文献+fill 标注）、库生长飞轮
（Tier-1 粗抽 mentions 回流挂边→下次零成本召回）。LKM 占了"gap 记录"
广义领地、EviMem 系占了"查询时 gap 闭环"，但**持久坐标→外向检索→
fill 标注→回流挂边**的完整链无人占（机制 1 终判；机制 2/3 待 D 确认）。

**创新性**：①字段级三态 absence（vs LKM open-question 级/ToC 语义推断级/
CROWN-QA 评测级）②缺口坐标合成检索词汇（知识模型贡献盲检索无法表达的
query）③答题中库生长最小闭环（LKM §6.2 仅愿景、自认 first stages）
④投资阶梯的确定性升级规则

**实验支撑**：几乎全部要新跑——三臂驱动式检索 ablation（P2 欠账：盲
关键词 vs +语义 vs +驱动式，且在正确靶场=外扩场景）、外扩冒烟（P3）、
多问题流生长曲线（库随答题增长、边际检索成本下降、后续题准确率变化）、
absence 回答校准（三态措辞的诚实性对照）。**实验最重的方向**

**风险**：①W1 无量化数字是本方向死穴——不跑实验不能写；②LKM 产品迭代
快，愿景落地即覆盖（时间窗最窄）；③"两个半环的组合"必须证明组合收益
（顶会纪律，agent D 明确警告）；④行为探针显示缺口型题模型偏好库内答
（驱动式工具的触发率是真实行为风险）

### 3.3 方向三："编译的受控科学"——what does compilation buy?

**核心论点**：产品级论证（LKM 40M 篇）与受控科学论证互补。社区缺一个
严格回答"编译的哪部分买到什么收益"的分解：同模型同语料同预算下的
消融阶梯（raw text 检索→+记录层→+视图/typed tools→+absence→+验证门）
×规模轴（16→53→430）。产出"编译收益分解表"。

**创新性**：消融阶梯+规模趋势本身是贡献；"检索臂随语料退化、编译臂
保持"是干净的科学结论；受控设定（同答题模型）是 LKM 评测没有的

**实验支撑**：一半在手（三臂+PS 规模轴+G2+F12+search-only 臂已预注册）；
需补完整阶梯（views-off/absence-off/门-off 各臂）+至少一个更大语料点

**风险**：①分析型论文在 main conference 的位置偏窄（但 ACL/EMNLP 有
传统，且"benchmark+analysis"混合形态可投 findings 保底）；②阶梯若
显示某层收益为零必须诚实报（也是发现，但削弱系统论文形态）；③与
方向一高度耦合——单独成篇需要更完整的阶梯

### 3.4 推荐组合与投稿策略

**推荐**：方向一为矛尖（verified compilation），方向三的消融阶梯为
实验主干（验证层 ablation 天然是阶梯一级），方向二的 absence/生长作
差异化亮点章节（实验来得及就放，来不及就 future work+第二篇）。
一句话论文定位：**"编译范式已被确立；我们证明编译必须可验证——
第一个带机器可验证保真与治理演化的科学文献编译框架，在受控闭卷
设定下给出验证层与结构层的收益分解。"**

**投稿落点**（待用户裁）：
- 最快：ACL 2027 main（~1 月截稿）——Multi-108+规模轴+门 ablation+
  canary 成篇；SQA2/AstaBench 若跑得动是第二战场加分项
- AstaBench ScholarQA-CS2：预编译路线无人上场的空位主场，test split
  可本地跑，broker 与其 MCP 工具面同构——但判分可信度有警报（AI2 自家
  元评测 τ=0.467），需双仪器披露；且需 gemini judge 渠道
- 方向二独立第二篇（库生长飞轮）：先补 P2/P3 量化债

**相关工作写法（正面引，不回避）**：LKM=范式先行+口径差异说明（开集
产品 vs 闭卷受控）；ASKS/Karpathy/CTIFoundry=编译术语先例；EGT-KG=
最近邻 baseline；2607.20527+AstaBench 自认=验证动机；EviMem/S2G-RAG/
AdaGATE/GRAIL/SEAL-RAG=transient gap 对照轴；DIAL-KG=治理最近亲；
Intern-Atlas=verbatim 最近亲（LLM audit vs 机器验证）；ToC/CROWN-QA/
GAPMAP=absence 概念近亲；微软 GraphRAG maintenance mode=背景板

**必须处理的诚实项**：
1. P1-C 引用修复：主表报修复后 0.6503，同表披露修复前 0.5183+修复的
   机械性质+基线无同类病核查；敏感性分析两值都报
2. AutoAIS 弱轨：披露段落级验证材料的系统性压分归因+gold 天花板同轨
   测量（0.42），或干脆双轨披露
3. bohao_cs 三臂全零域：披露语料覆盖死区
4. 建库成本：深抽 ~2min/篇+全链账本（token/墙钟）如实报；Tier 分层是
   经济学答案但 Tier-2 经济学未系统测
5. LKM 数字口径：不混表（预注册纪律），引用时标注设定差异

### 3.5 定稿前必办（阻塞项）

1. ~~ASKS/ScholarStack 深核~~ **已完成（0926，主协调员 WebFetch abs 一手）**：
   - **ASKS**（2608.29612，"LLMs Interpret, Embeddings Organize, Graphs
     Emerge: Agent-Driven Compilation of Scientific Knowledge"，08-30，
     Shi-Ju Ran 等，张量网络领域）：其 "Deterministic checks convert the
     latter into document-local GraphDelta" 指 **LLM 语义→图变更的结构
     完整性转换**，非对源文本的抽取保真验证；无 verbatim quotes/质量门/
     源文本校验（abs 级确认）；56 篇单课题组时序编译，无 QA benchmark。
     **与方向一主张不撞车**；重叠=术语（"scientific knowledge
     compilation"）+ingest-as-inspectable-state-transition 框架（与我们
     两阶段提交/失效代数精神近）——须正面引用
   - **ScholarStack**（2609.23735，ScholarSeed AI Team/Wotao Yin 系 19 人，
     v2 09-22）：三层资产+共享接口；"verification status" 摘要未定义、
     无机制描述；无 absence；无具名 benchmark 数字（自称 matched-model
     baselines 上跨论文任务增益+token 成本下降）；schema 演化未提。
     **不占验证位**（caveat：全文未核，投稿前建议复核一次）；其
     "matched-model 对照+跨论文增益+成本下降"主张形态与我们受控对照
     相似——列为方向三的中度威胁
   - **结论：'机器可验证抽取保真'位确认无人占据，方向一核心主张存活**
2. D 的机制 2/3 终判（lineage_walk/回流飞轮撞车度）——进行中
3. SciAtlas/Mechanist、超图系、GraphRAG 家族的补核（agent A 三路断连
   未核完；GraphRAG maintenance mode 已核，其余为旧认知）——非阻塞，
   写 related work 前补
4. 用户裁决：主叙事选择/目标会议/LKM 口径策略/AstaBench 是否上场

### 2.9 可比性/条件比较类测试集专项判决（调研子代理，报告
comparability-track2-tables.md，18 候选逐一一手核）

- **(d) 数值可比性判定的直接 gold：不存在**（Google×14+arXiv×7 双通道
  地毯 2020-2026 零命中）。能力被拆散在三个互不连通社区：计量抽取
  （有条件标注无比较任务）/表格事实核查（有判定无单位条件概念）/
  科学复现（有一致性语义但绑代码执行）——**三者交集=真空位=定位机会**
- **条件比较 QA 无直接现成品**：最近似=SCITAT（ACL 2025 Findings
  2025.findings-acl.199，953 题/871 篇 arXiv CS，Comparison 子类 ~78 题，
  EM/F1/BERTScore 无 GPT 锁，数据实拉验证）——但条件隐式（同表内比较）
- **合成 gold 配方（现成资源）**：SciREX（ACL 2020，438 篇 ML 论文，
  κ≈0.95，{Method,Metric,Task,Material,Score} 五元组=唯一现成"条件
  向量→数值"结构化 gold；条件全同=正例/单维扰动=负例）+MeasEval
  （SemEval-2021 T8，单位+修饰符+Qualifier 层，α=0.866/Qualifier 0.334
  需收紧）+SciTab（counter-claim 生成校验管线+NEI 档可搬）——全链路
  规则+GLM-5.3 判分，零 OpenAI/Gemini 依赖
- 纠错：CondAmbigQA 真但通用域+GPT 锁 judge=C 级仅 schema 参考；
  SciGen 真 ID=2104.08296；QUANTA/SciReplication 定位不到（判定名字
  记混）；表格数值推理系 7 个全核无一含可比性判定题型
- **对设计的影响**：G4 可比性判定自建的成本大降（SciREX 五元组直接
  程序化生成正负例，零人工标注）；SCITAT Comparison 子类可当外部
  微锚；"可比性判定无基准"写进论文=真空位声明（双通道地毯背书）
