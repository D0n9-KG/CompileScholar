# 全量战役总规划（CAMPAIGN MASTER PLAN）— 2026-09-10

> **本文件是 compact 后的执行圣经**。所有已定决策带拍板人（用户/预注册/我方提案待定），所有档案给盘上路径。讨论中状态见 §10。

## 0. 一句话战略

PaperScope 线收束为内部资产（用户裁定不进论文）→ 战役重心=**三考场全量 PK（小模型同底座）+ 双分辨率知识架构（核心库+外围层）+ 均衡器主张（模型越弱 KB 增益越大）**；论文叙事五章：机制（主场）→ 考试（AirQA）→ 溯源（LitTraceQA）→ 真实工作流（SurGE）→ 讨论（真空轴案例演示）。

## 1. 已定决策（拍板记录）

| # | 决策 | 拍板 |
|---|---|---|
| D1 | PaperScope 不进论文；dev30+fresh16+三轮档案=内部决策资产（台账保留；审稿人问起备一句诚实脚注选项） | 用户 09-10 |
| D2 | 三考场：**AirQA 主战场**（客观子集 886 题，逐题代码判分）+ **SurGE 下游**（综述写作，MIT，判分重确定性）+ **LitTraceQA dev55 溯源专项**（唯一官方考证据定位；兼外围检索层考场） | 用户 09-10 |
| D3 | GSAP-ERE（~4M，记录层确定性验证，数据已发布免鉴权）挂观察，默认不进，用户点头随时加 | 待用户 |
| D4 | 假说生成位=案例演示不进主表（ResearchBench/MOOSE-Chem 材料在档备用）；时点快照/KB演化/逐值超参=实测真空→案例演示+内部工具=论文占位主张 | 用户 09-10 方向确认 |
| D5 | MetaSyn/SurveyBench/SciCiteVal/MOOSE-Chem/SciPaths/ScholarQABench（窗口10-23）=二期或审稿响应储备 | 用户 09-10 |
| D5b | **AstaBench/PaperFindingBench=二期储备**（另一会话源码级核验 09-10，档案 scratch/ai2_baseline_2026-09-08/{asta,apf}/）：任务形态高度对口（333 题=查询→有序 paper_id+逐字证据片段，与 LitTraceQA 检索+证据段重叠，主阵容已有同功能考场不叠加）；硬依赖=Ai2 托管 MCP 键（需申请，~4req/s）+S2 batch+HF gated 数据集，外部服务依赖三重=不进主力。**APF（asta-paper-finder）基线=砍**：冻结快照（实质提交止于 2025-08）、底座非配置级可插拔（硬编码注册表+responses_api 坑+gemini 步要改码）、四家 API 键依赖（OpenAI/S2/Cohere/Google）、线上版内部索引已剥离——与 OpenScholar 同判：降级形态不值一套冻结栈 | 核验+我方判决 09-10 |
| D6 | 小模型路线：全量战役以小模型同底座跑（"全量可跑"的前提）；旗舰模型只留均衡器消融一列；小模型标准=**相对优势**（同底座下我们>外部基线），非逼近旗舰 | 用户 09-09/09-10 |
| D7 | gpt-oss-120b 出候选（不够普遍）；GraphRAG 砍（古早+最贵）；OpenScholar 砍（746GB datastore+API 从未上线+停更）；PaSa-7B 砍（NC 权重+形态错位）；RAG-Anything 降备用（pin 冲突；LightRAG 1.5.x 已原生多模态） | 用户+调研 09-10 |
| D8 | 通道防护规格（修正版）：重试只在同模型同通道（指数退避+抖动+Retry-After）；跨通道备援仅当同模型同版本已验证；通道持续挂→停等+断点续跑**不换模型**；**臂纯度断言**（账本逐调用记 provider+model，跑批结束断言全臂同模型，违例=该臂作废） | 用户 09-10 |
| D9 | 引文密度=呈现档案参数：系统默认/agent 档案=逐句引用全保留（可验证性=产品功能）；审计面（笔记+answer_raw 存档）永远全量；prose 档案（人类/LLM裁判场地显式选用）才聚合 | 用户 09-10 |
| D10 | 比赛资产仅供参考：继承实测事实（API 通道行为/成本率）+待重验假设（"召回瓶颈在措辞"），代码零搬运；外围层从需求+前沿重新设计（设计铁律追加条款已入 memory） | 用户 09-10 |
| D11 | 台账 RESULTS-LEDGER 只录现役方法；直读臂成绩不入表（仅成本倍率行）；迭代过程数字不入账（存档在报告+IL 日志） | 用户 09-10 |

## 2. 自跑基线阵容（通用层已定，场景对口层待调研收尾）

**已定（通用 RAG 层）**：LightRAG 1.5.7（已跑通，MIT，嵌入建库后不可换）+ HippoRAG 2（MIT，端点全可插，OpenIE 每 chunk 2 调用）+ 自建裸RAG/直读臂（AirQA 官方基线代码实为未发布，必须自建）。
**场景对口层（调研终版已闭环，sci_baseline_survey.md 252 行全档）**：
- **PaperQA2 进 AirQA 主对口臂**（唯一活跃学术QA专用系统，2026-08 发版，Apache-2.0，repo 已改名 Future-House/paper-qa 经 PyPI 反查确认；审稿人必问名字）。**⚠️ 与用户 09-09 "PaperQA2 出局"裁定的关系**：当年出局理由=旗舰价位下答题成本 2.7× 于我方且预算不可行（PaperScope 战役语境）；现在是免费/廉价通道下当对照基线（88.6M tokens 墙钟 1.2-15 天按通道速度，工程可行），性质不同——**需用户确认翻案**。前置=embedding 本地化+断点续跑+50 题 smoke；降本旋钮（top_n/evidence_k 减半）备用。
- **AutoSurvey + SurveyForge 进 SurGE 写作臂**：均为 SurGE 官方评过的 agent 基线=venue 原生可比（统一端点重跑即同协议对比）；SurveyForge 引用召回全场最强（0.0868）=最硬对手，赢它才有说服力；两家无 LICENSE 文件（内部评测可用、发布披露，SurGE repo 自带其产物有先例）；SurveyForge 自建库路径需 smoke，不过则臂位让 SurveyX（闭语料 .md 喂入最友好，991★）。
- **LitTraceQA 无场景对口主臂**——深查确认**没有任何开源系统原生支持证据定位输出**=我方卖点空位实证；SPAR（MIT+端点可插+Qwen 原生）降级备用（检索段，live→闭池改造 1 天级）。
- 砍单终版：AutoScholar（前提纠正：实为 PaSa 论文的 35k 合成评测集，非系统）、OpenScholar（746GB+API 从未上线）、SciAgents（材料域零对口）、PaSa（本地 7B×2+logits+付费 SerpApi=底座不可插拔，agent_prompt.json 可借素材）、ScholarQA lib（live S2 架构与闭语料错位）、STORM（停更 11.5 月+非学术专用，降 SurGE 通用对照备用）、PaperBot/FutureHouse 托管 agents（无开源本体）、SciAtlas（头号竞品，引用对象非对照臂）、ScholarAgent/Tolar/DeepScholar（查无官方 repo，如实记录）。
- **统一纪律**：全臂锁死同一 OpenAI 兼容端点；闭语料适配一律"换检索后端不动推理管线"；judge 与生成端点分离+双 judge 抽检；四个"未完成核验"项（PaperQA2 深核细节/ScholarQA/STORM 改造量/OpenScholar 复核）列入进场 smoke test 清单兜底。
**借行参照（不自跑）**：AirQA 论文 8 配置+官网排行榜（不同底座，披露脚注）；SurGE 官方 3 配置基线产物本地重算；LitTraceQA 排行榜 4 队+主办方 RUNBOOK dev 基线（paper_f1 0.338/evidence 0.029）。
**分考场配置**：AirQA=全阵容主 PK（论文主表）；SurGE=写作对口阵容+官方重算；LitTraceQA=轻量（检索段确定性基线自建+我们）。

## 3. 小模型定档闸（大钱之前，全部便宜）

- **闸0（完成）**：Paratera 86 模型清单+CST 7 免费模型已在档（scratch/paratera_models.txt）。**遗留核查已闭环（09-10 precheck）**：CST `qwen3.5` = **Qwen3.5 家族 397B 全量档**（CST 官方文档逐字"上下文128K，支持多模态输入，397B全量"；对应 Paratera 命名 `Qwen3.5-397B-A17B`，快照版本未标注=中置信同版，跨通道备援启用前须行为一致性验证）。**框架影响：CST 免费臂不是小模型而是上代旗舰 MoE**——闸1 两臂语义精确化：CST 臂=免费通道可用性，Paratera Qwen3.6-27B 臂=真小模型可用性（D6 本体）。档案：experiments/stageB/precheck_2026-09-10/PRECHECK-REPORT.md §1。
- **闸1 前置工程已全部完成（09-10 precheck，四项全过）**：①CST 加固落地 src/kb_infra/llm.py（sock 150s/重试5次退避+抖动+Retry-After/信号量4/臂纯度断言 check_arm_purity；活体 8/8 含 3 次重试救回；顺手修复 call_json→call_cst 的 enable_thinking 潜伏 TypeError）②思考关实锤：qwen3.5 必须带 chat_template_kwargs={"enable_thinking":false}（TTFB 77-176s→1.3s，reasoning 0；DeepSeek 形态 422；间歇 422 flake 由 call_json 外层吸收）③嵌入三通道实测：CST 18条/s(4096维) >> Paratera 2.65(1024) > 本地Ops-MM 1.22(1536,确定性)——**待用户拍板定死**④VLM 通道 PASS：GLM-4V-Flash 合成图转写+标号柱+无标号柱估计全对/真实图正确识别/1.5-6s（GLM-4.6V 需大思考预算；CST qwen3.5 多模态备选）。
- **闸1 建库保险丝**（~6.6M，两独立单模型臂）：CST qwen3.5 臂 + Paratera Qwen3.6-27B 臂（可加 Qwen3.5-35B-A3B 凑梯度），各把 PS16 篇重跑抽取。验收三条（跑前冻结）：①金丝雀不劣化（5事实+4陷阱）②写门通过率/修复率不崩（对照 Qwen3.8-Max 基线带 first_pass 0.86/repair 0.089）③记录对齐率（与现有 KB 的实体/主张重合度）。标准=无崩坏非逼近旗舰（D6）。前置工程：CST 防护加固（D8）+嵌入通道吞吐实测（三通道：CST embed/Paratera API/本地 Ops-MM 2B——**全系统统一嵌入端点，建库前定死**，LightRAG/HippoRAG 均不可换）。
- **闸2 瘦身版（用户 09-10 深夜修订，原三臂作废）**：**dev30 分层抽 12 题（trend/gap/results 各 4，种子冻结）× {我们 / LightRAG} 两臂 × 同 27B 底座**，~6M。**直读臂不跑**（用户多次裁定，D15 延伸到内部闸）。核心问题=**27B 答题端 agentic 驾驭力**（历史教训：旧 DSF 抽取尚可但答题侧 FAIL −0.92，抽取好≠答题好）。**答题模型后备梯（用户指令）：Qwen3.6-27B → DSF-0731 → Qwen3.8-Max**，每级失败如实记档再降级。前置=抽取层修复包 FG1–FG5 + 27B 重建 PS16 库 + **人工实读验收**（用户指令：裁决门不能直接信，要实际看抽取效果；门只当粗筛）。标准=相对优势成立（配对+CI 如实报，n=12 功效弱如实披露）。
- 双闸过 → §4 全量开工；任一闸败 → 回旗舰档重核算或缩组合。

## 4. 三考场执行账（小模型档 token 量级）

| 考场 | 建库 | 答题 | 基线 | 合计 |
|---|---|---|---|---|
| AirQA 客观子集 | ~900 篇×205k≈185M（多文档子集 gold 论文+采样干扰；全池 14k 篇=2.9B 不可行）| 886 题×~140k≈124M | LightRAG ingest+query≈200M；HippoRAG2 建索引 5-10k 调用+查询便宜；裸RAG 便宜 | **~510M**（裁剪版=多文档子集 333 题≈250M）|
| SurGE | 元数据 1.5GB 本地检索层（便宜）+top-K 晋升建库 | 41 题长文（成文贵，~200k/题级）| LightRAG/裸RAG/官方3配置重算 | **~40-80M** |
| LitTraceQA | 70 篇 gold×205k≈14M+外围层嵌入 27k 摘要≈14M | 55 题×140k≈8M | 检索段 BM25/向量（零 LLM）| **~36M**+图通道另计 |
| **总盘** | | | | **~600M**（旗舰档同规模≈6-13B=不可行，故 D6）|

均衡器消融（旗舰×小模型双底座一列）：选一个考场（建议 AirQA 子集抽样）×{我们/LightRAG}×{小/旗舰} ≈ +100-150M。

## 5. 系统侧待修清单（战役版本，全部已诊断在档）

F16 强参数解析层（标题/描述串/pid 当实体名→确定性代解析；fresh-16 两例+dev30 一例证据）；F17 v1 prose 档案接线（编译提示词升级，离线验证 +0.16）；R2 数字闸无损归一（85.30 vs 85.3）；R3 contains 零命中附表面索引候选；R4 J1 词表扩双语；resu 范围纪律（编译层"只答所问条目、数字只取所问对象"）；F15 已落地。规则层元条款四条即日生效（RULE-LAYER-AUDIT.md）。
图通道（LitTraceQA 25% 图证据）：RAG-Anything 双层思路（离线 VLM 描述入库+在线按需原图实读）；MinerU chart 数字化表禁用（已证伪）；前置=VLM 通道核查（并行科技视觉模型清单+传图实测）。
外围层（开放检索）：双分辨率架构（核心类型化库+外围摘要级轻记录+"已注册未读"覆盖态+晋升梯）；流程=前沿调研（PaSa/DeepRetrieval/Search-o1/deep research 系）→设计规格→用户评审→代码（src/ 正规目录，知识层一等组件）。

## 6. 数据/工程前置（可并行，零决策依赖）

①AirQA：**第一步已完成（09-10）**——metadata.zip+processed_data.zip(463MB,全池 13,957 条目)+test_data(1246题)+uuid2title 已下到 experiments/stageB/airqa/；结构分析实锤：886 客观题=885/886 纯代码判分、gold 论文并集 **1064 篇**（建库账修正 1064×205k≈**218M**，原估 185M）、**processed_data 含 TOC 分节全文+表格HTML+图注/bbox/页码 → 文本建库免 PDF 解析+图通道免费清单层官方直接给**；PDF 只需选择性下载（image-tag 题绑定论文），60GB 全量免下。子集选定（886 vs 333+干扰采样方案）写进预注册；②SurGE：Google Drive 整包镜像（1.5GB 单点防失联）+自定 topic 输入字符串披露；③LitTraceQA：70 篇 gold PDF 获取（OpenReview 38 篇走 API/arXiv 标题匹配+CVF 20+ACL 9+ECVA 3）+MinerU 解析；④CST 防护加固+qwen3.5 身份查证；⑤嵌入端点吞吐三通道实测。

## 7. 判分与协议纪律（沿承）

异厂分离（作答/抽取=定档小模型，判分=Kimi-K2.6 或考场官方确定性判分优先）；AirQA 主结果=886 题纯代码判分子集（LLM judge 子集若跑须自证对齐）；SurGE=引用召回（集合运算）+引用准确（本地 NLI）为主锚、judge 分（τ=0.805 人工对齐）为辅；LitTraceQA=官方 evaluate.py 零改动；盲评乱序+gold 锚定混杂披露照旧；预注册先行（每考场一份，判据跑前冻结，IL 日志披露迭代）；波次协议（分段跑+体检扳机+随时停车权）；臂纯度断言（D8）；成本分阶段报账、帽=实测单价+margin（IL-P8 教训）、超帽+10% 停线。

## 8. 论文资产盘点（已有，零新增成本）

机制章：主场 gold 50 题（B3 主臂 3.42+同循环对照配对 +0.462+循环增益 +0.58）+LightRAG 现行协议重判（3.080，配对 +0.340 CI[+0.07,+0.61]，配置+1.00/覆盖+0.70/聚合−0.13）+组件消融三臂义务（评审面板桶B清单）。外部验证章（若启用脚注选项）：PaperScope 三轮+fresh-16 全档案。裁判体系方法学：三发现可独立成文。失效分析章素材：PS 三轮诊断链（访问层断裂→修复→风格天花板）+gold 锚定混杂三例+规则层审计方法论（IL-P9/F17v2 案例）。台账：RESULTS-LEDGER.{md,docx} 持续维护。

## 9. 尽调档案索引（全部一手核验）

benchmark_borrow_scan_2026-09-09/：airqa_deep_check.md / metasyn_deep_check.md / survey_venues_deep_check.md / claim_verify_band_deep_check.md / hypo_repro_band_deep_check.md / baseline_repro_check.md / function_slot_sweep_A.md / function_slot_sweep_B.md / relatedwork_mine_{mdaqa,littraceqa}.md / second_arena_recon.md / MERGED-VERDICT.md。设计档案：PRESENTATION-LAYER-SPEC.md / RULE-LAYER-AUDIT.md / CAMPAIGN-PORTFOLIO-2026-09-10.md（含用户拍板记录§五）。PaperScope 档案：paperscope_r2/{PILOT-PREREG(IL-P1~P12), PILOT-REPORT{,-R2,-R3,-F16}, PROMPT-AUDIT-R2, DIAGNOSIS-GAP-R2, PILOT-VERDICT-COMPUTED}.md。

## 9b. 用户拍板增补（2026-09-10 深夜）

- **D12 三考场锁定，其余全转备选**（优先序：GSAP-ERE > SciCiteVal > MetaSyn > 其他；主考场出事按序顶替）。
- **D13 ScholarQABench 放掉**（QA 轴已有 AirQA、写作轴已有 SurGE，两头不占+10-23 死线）。
- **D14 多模态/图通道升格为系统级工作项**（此前漏项，用户点名）：RAG-Anything 双层思路为参照+我方三层设计（免费清单层/选择性描述层/按需实读层）；两个特有设计难题=图生主张的证据机制（无逐字原文可引→locator+VLM读数+质量旗标平行证据形态）与 VLM 幻觉闸（独立认识论标记，不与文本记录同权重）；时序=LitTraceQA 试跑可文本+清单层先进场（证据定位吃 locator 不吃读图），读图层建成后补全量；前置=VLM 通道核查（并行科技视觉模型+传图实测）。正式规格文档走评审流程（先讨论再动笔）。
- **D15 自跑臂瘦身**（排行榜借行只当参照，主表同底座自跑不变）：AirQA=真系统阵容（LightRAG+HippoRAG2+PaperQA2）；SurGE=写作对口臂（AutoSurvey+SurveyForge）+LightRAG 一个通用代表；LitTraceQA=轻量（SPAR备用）。**裸RAG/直读不进主实验**（用户 09-10 深夜裁定）：下界参照由 AirQA 官方发表行（RAG/Direct 基线）承担。**消融实验的具体设计（做哪些臂、含不含裸RAG/直读）=后议专项，本条不预设**（用户明示）；已有素材（E2 均衡器、主场 flat 臂、组件消融三臂义务=评审面板桶B清单）留档备用。
- **D16 PaperQA2 翻案确认**（性质变化：PS 战役出局理由=旗舰价答题成本不可行；现=免费/廉价通道下的最相近外部参照臂，Apache+活跃+审稿人必问）。A6 时代复现四坑（索引配置/并发竞态/100k单题/断路器题级执法）预置进 50 题 smoke 清单；进度保险=通道限速卡脖子时只跑 333 多文档子集并披露。
- **D17 迭代纪律两条**：每考场预注册写死迭代轮次帽 ≤2（到顶升级用户决策）；主场 gold 50 题转岗回归测试集（每轮修复后抽题重跑，防修 A 砸 B）。
- **D18 三阶段考场推进纪律（用户 09-10 裁定，硬约束）**：任何 benchmark **禁止直接跑全量**。流程=①小规模随机抽样试跑（暴露问题）→②及时修复+再随机抽样迭代几轮（迭代帽 D17 管住）→③试跑无大问题+**用户确认**后才开全量。理由（用户原话要义）：系统当前成熟度不支持直接全量。每考场预注册必须把三阶段写死：试跑样本量、迭代轮次、全量放行条件。
- **D19 闸1 落章+过夜授权（用户 09-10 深夜）**：①闸1 判决=B 臂 Qwen3.6-27B 晋级战役底座（判决书 GATE1-VERDICT.md；A/C 用户终止、C′ 淘汰）②执行顺序=**先修抽取层（FG1 公式协议/FG2 chunk 计数闸/FG3 修复分诊/FG4 27B 首过率调优/FG5 成本护栏）→ 27B 重建 PS16 库 → 人工实读验收（"裁决门不能直接信"教训：门只当粗筛，判决前实读记录对原文）→ 达预期 → 闸2 瘦身版**③答题模型后备梯 27B→DSF-0731→Qwen3.8-Max④**读图模块=抽取层达预期后启动；外围论文检索层=基础打牢后启动**（都走调研→规格→用户评审→代码）⑤过夜自主执行授权（"你干吧"），成本纪律照旧（帽+停线+晨间报账）。

## 10. 讨论中/待决（compact 后先核本节）

1. ~~场景对口基线终选~~ **已闭环并入 §2**；遗留拍板：**PaperQA2 翻案**（09-09 出局裁定 vs 现在当 AirQA 对照基线，性质不同，等用户确认）；
2. 闸1 放行（候选=CST qwen3.5[已查实=Qwen3.5-397B-A17B 上代旗舰 MoE，免费/不稳/预计 3-8h 墙钟] + Paratera Qwen3.6-27B[真小模型]；**前置四项已全部完成 09-10**，PRECHECK-REPORT.md §7 待用户两项拍板：嵌入端点定死+闸1 放行）；
3. ~~GSAP-ERE~~（D12 已决：转备选第一位）；~~ScholarQABench~~（D13 已决：放）；~~PaperQA2 翻案~~（D16 已决：进）；
4. 外围层设计规格提纲评审（§5，前沿调研→规格→评审→代码）；
5. AirQA 子集方案（886 全客观集 ~510M vs 333 多文档子集 ~250M）——闸1 后定；
6. 图通道/多模态正式规格（D14 前置：VLM 通道核查；规格文档走评审）；
7. ~~嵌入端点定死~~ **已拍板（用户 09-10）：CST qwen3-embedding:8b（4096 维，18条/s，免费）**；本地 Ops-MM 留确定性审计通道、Paratera GLM 三级备用；建库后不可换，嵌入 provider 标签纪律照旧。
