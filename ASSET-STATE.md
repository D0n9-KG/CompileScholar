# ASSET-STATE — 单一事实源（2026-09-16 建立，随进展更新）

> **仓库更名（2026-09-20）**：LogicKG → **CompileScholar**（GitHub 仓库+本地目录同步改名，旧链接自动重定向；范式主张=编译 vs 检索）。历史文档中的 LogicKG 字样均指本仓库。

> 目的：终结"状态散在 memory 恢复点/台账/判决档/LINEAGE 四处互相引用"的混乱。
> 任何"现在有什么/什么状态"的问题先查这页。路径缩写：`STAGEB` = `.research_tmp/experiments/stageB`。
> 更新纪律：每次换装 staging / 完成战役 / 冻结解冻一条线，当轮更新本页。

## 0. 系统实况勘误（2026-09-17 代码一手核验，**本节优先于此前一切叙述**）

> 背景：用户指出既往记录文档造成认知混乱。本节=通读 src/ 全部模块+答题栈 harness+PS-53 真实产出+65 测试实跑后的结论。与旧文档冲突处以本节为准。

- **系统不是知识图谱，也不是"五层认知底座"**。现存系统=类型化记录层（冻结 schema v1.4）+确定性视图编译+9 个 typed tools+检索兜底+ReAct 答题栈。"五层（Find/Get/Understand/Cognize/Trust）"是愿景叙事非现状——Find 不存在、Understand 仅表格通道、Grow 不存在。论文叙事不得按五层写。
- **王冠资产=抽取管线的"确定性三明治"纪律**（11 步：manifest→skeleton→registry→slot→postcheck 五门→表格通道→F35→round2→canary→视图→tools；LLM 只逐字抄，一切判定走确定性代码，每个守卫带实测事故背书）。这是系统区别于一切 KG-RAG/科学文献系统的指纹。
- **关系层薄，事实层厚**（PS-53 实测）：result 54%/finding 29%/config 13%，但 lineage 仅 ~8 条/篇、shift 全库 3 条——genealogy/演化只能作辅助视图，撑不起主叙事。
- **实体链接=头号工程短板（P0）**：PS-53 建库时 registry_round2=OFF → gold16 method_ref 链接率 46.3%、distractor37 仅 **6.3%**。typed tools（compare/card/config 走 registry 解析）在 53 篇语料上被实体层卡脖子；matrix 表键碎片化同源。任何 PK/规模实验**之前**必须先跑 round2+仲裁消化。
- **答题栈=演化产物非设计物**（笔记写入路径 6 层补丁/monkey-patch 三层包装/AirQA 文件伪装 PS 名）：工程可用，发表形态不合格。论文定位候选=评测消费端（非系统组件），待用户裁。
- 抽取管线/守卫/门的一手技术事实报告存档：`ccfa-workfiles/idea/logickg-idea-plan-2026-09-17/`（IDEA-PLAN v4 附录）。

## 1. 活代码（src/，git 追踪，65/65 测试过，commit 35219f44 后干净）

| 模块 | 职责 | 状态 |
|---|---|---|
| `kb_compiler/records/schema.py` | 记录层 schema **v1.4 冻结**（7 记录类型+overflow；epistemic 三态；verbatim 引文纪律；改动走仲裁+版本号） | 活跃·冻结 |
| `kb_compiler/records/skeleton.py` `slot.py` `postcheck.py` `chunk_retry.py` `canary*.py` | LLM 抽取链（骨架→槽位→后检门；canary=金丝雀回归守卫） | 活跃 |
| `kb_compiler/records/table_channel.py` | F24v2+F32+F33 确定性表格通道（解析器零 LLM；subject 落位/实体链接/引文剥离） | 活跃 |
| `kb_compiler/records/table_semantic.py` | F35 语义表格层（LLM 提案+确定性结构门）| **排队**（15 单测过，canary/apply 未跑） |
| `kb_compiler/records/figure_channel.py` | 识图通道（GLM-4V-Flash 读值已验证） | **排队**（未集成） |
| `kb_compiler/records/registry.py` `registry_round2.py` | 实体注册表+round2 生长（F34 "+"变体守卫在位）；治理=仲裁队列非自演化 | 活跃 |
| `kb_compiler/records/manifest*.py` | 论文元数据层 | 活跃 |
| `kb_compiler/records/usage_audit.py` `arbitration_list.py` | 治理诊断（字段利用率→退休候选；仲裁议程） | 活跃 |
| `kb_compiler/views/compiler.py` `render.py` `tools.py` | 四视图编译（matrix 分带/genealogy as_of/coverage derived-absence/cards 档案）+ typed tools API（compare/lineage/find_gap/config/findings/card/as_of/entities/search） | 活跃 |
| `kb_compiler/verification/consistency.py` `maintenance.py` | 一致性验证/维护 | 活跃 |
| `kb_infra/` | LLM provider 网关（Paratera/CSTCloud）+ embedding | 活跃 |
| `claim_coverage.py` | 独立 stdlib 工具 | 活跃 |
| `tests/`（65 个） | 含 `test_kb_compiler_import_gate`=机器闸（AST 扫描 src/ 全部 import：禁 granular_agent+kb_infra 仅 stdlib+kb_compiler 白名单+schema 冻结守卫）。**闸边界=仅 src/**；.research_tmp 实验脚本不在闸内，靠物理隔离（旧栈在 archive/legacy，sys.path 不含）+纪律：实验脚本禁加 archive 路径，Type B 等若升格进 src/ 必须 fresh rewrite 过闸（2026-09-16 依赖图全链 grep 核验：识图/F35/TypeB/答题栈零旧栈引用） | 活跃 |
| `archive/legacy/`（gitignored） | 死代码：granular_agent（Stage A 超图）/contest/run_kernel/run_cases/8 死测试 | 死·可逆归档 |

## 2. 知识库与 staging（三套）

| KB | 位置 | 内容 | 状态 |
|---|---|---|---|
| **PS16**（PaperScope 16 gold 篇） | `STAGEB/psfix_2026-09-11/psv5/`（血统见其中 STAGING-LINEAGE.md） | records v323（4206）+views v323（matrix 308/cards 1205/derived 8）+registry v35_f34（1725 实体） | 冻结（历史 +0.133/+0.83 判决口径） |
| **AirQA**（25 篇） | `STAGEB/airqa/s1_qa/`（staging）+ `airqa/s1_build/`（构建产物） | records v34（5515+表格）+views v34+registry v2s_f34 | 冻结（阶段一+批3a 收口） |
| **PS-53**（16 gold+37 干扰，规模实验专用） | `STAGEB/paperscope_r2/ps53/` | records_checked_ps53（**17822**=LLM 10691+表格 7131）+ views_ps53_full（matrix 1026/cards 1205/genealogy 416）+ views_ps53_16（同 KB 过滤 16 篇：4269 条/matrix 302）+ build_summary_ps53.json；建库 15.78M/2532 calls 全 Qwen3.6-27B | **活跃**（视图 2026-09-16 建成，F13 双臂 PASS；round2=OFF 已冻结[预注册 v1.0]；distractor37 实体链接仅 6.3%=披露项；**观察项 2026-09-16**：overflow 队列 1294 条被 References 节协议噪声主导[非 schema 压力]，下次建库 text prep 应过滤 References[IL-2 先例]或 triage 加确定性过滤） |

## 3. 答题栈（冻结态，解冻需用户授权）

- 位置：`STAGEB/rebuild27b_2026-09-11/`（harness）+ `STAGEB/airqa/`（airqa_run.py runner）
- 形态：evidence_gate2f harness + **F31V2 案A**（auto_transcribe 机械转写装置，G2_F31=off 字节等价开关在位）+ **NAV**（miss 时跨索引提示）+ IL-9b（元数据投影）+ PROJ_LEDGER（投影台账）
- 纪律：hybrid 冻结停止加补丁（09-17 判决：补丁堆=agentic 固有成本非设计病）
- 离线验证工具（零 token 可复用）：`STAGEB/airqa/s1_qa/` 下 e5_v2（五层严格仪器=唯一可信值级追踪）/f31v2_tests/nav_verify/f32_obs_level_verify/f28_forensics/recall_census2/format_forensics

## 4. 五条工作线（状态+解冻条件）

| 线 | 状态 | 解冻条件 |
|---|---|---|
| **PS-53 规模实验** | **活跃**：**P0 已完成（2026-09-17，总 1.95M）：round2+relink+F35 落地，链接率 gold16 46.3%→76.1%/distractor37 6.3%→66.2%，matrix 表 302→413（gold16）/1026→1900（full53），staging=v38**→ 下一步=D2 硬负例重建（~10M）→四臂闸门轮（ours×{16,53}+A9×{16,53}）。详档 ps53/P0A-ROUND2-REPORT.md + P0B-F35-REPORT.md | round2 建库时未开=既成事实（见 §0 勘误）；gold16 残留 3.9pt=复合串 F16 领地不阻塞；仲裁队列 2418 新实体待人工消化（~741 垃圾+1677 真实方法名，不阻塞 P1） |
| AirQA | 冻结：阶段一 4/20+批3a 持平收口；批3b（7.3M）缓跑；拦路石=语义层 56% 残留/0b1cad92 数据质量/2a 读图缓 | 论文定位定稿后按证据缺口决定是否重启 |
| F35 语义表格层 | 排队：代码+15 单测 ready，触发面 ~80 表（PS）/124 表（AirQA），canary+apply ~1-2M | 定位定稿+预算批 |
| 识图 figure_channel | 排队：模块已建未集成；2a 战役（~22M）无限期缓 | 同上 |
| Type B 海量检索 | 排队：pool v2（1272 篇）+通道消融完成（query-decomp 最大杠杆 R@10 0.697；bm25 0.610>vector 0.530）；channel-4 结构化未测（~1.8M+） | 用户定价值+定位需要 |

## 5. 权威数字（只认 RESULTS-LEDGER.md：`.research_tmp/paper_drafts/RESULTS-LEDGER.md`）

- 主场 gold（50 题/DRL 40 篇/Kimi 判+J1）：ours 3.42 vs LightRAG 3.080 = **+0.340，CI[+0.07,+0.61]，18W24T8L**；vs 同循环原文块对照臂 +0.462
- PaperScope dev30（16 篇）：−0.073 打平（**已诊断=语料太小伪影**，PS-53 重审中）；fresh-16：−0.062 泛化失败条款触发（同诊断待重审）
- AirQA 阶段一/批3a：4/20 官方分；机理台账=修复季真成绩（recall_or_projection 3→1/hard_stop 13→9）
- 建库经济性：Qwen3.6-27B ~280k tok/篇（PS-53 实测 15.78M/52 篇）
- 稳定画像：内容维打平略胜/风格维恒亏（流畅 −0.5~−0.71）；编造 0/178（机械锚门）
- 纪律：官方 leaderboard 数字永不进对比表

## 6. 关键档案地图（绝对定位）

> 工作区两级索引：repo 根=本页（资产状态）；`.research_tmp/INDEX.md`=实验区内部地图（2026-09-16 整理后建立，新目录必须登记）。根目录死形态已归档 `archive/pre_stageB_root_2026-09-16/`（清单+恢复方法见 archive/MANIFEST-pre_stageB_root_2026-09-16.md，commit bc72a742）。

- 预注册/判决档：`STAGEB/airqa/`（AIRQA-PREREG/AIRQA-S1-VERDICT/AIRQA-B3A-VERDICT/MATERIAL-LEDGER/F35-SPEC/NAV-SPEC/F31V2-SPEC/TYPEB-RETRIEVAL-SPEC）+ `STAGEB/paperscope_r2/`（PILOT-PREREG/PAPERSCOPE-R2-PREREG/OFFICIAL-REPO-NOTES）
- 创新定位材料（转向后，只认这些）：memory 目录 `v6-review-round1-verdict` / `research-direction-pivot-need-driven` / `workload-catalog-2026-09-03` / `competitive-landscape-2026-09-03` / `sciatlas-v2-mechanist-recheck` / `e2-r2-scaling-verdict`；全文 `.research_tmp/docs_decisions/PROPOSAL-2026-09-03-v6-*`
- 基准调研：`.research_tmp/literature/survey_homecourt_benchmark_2026-09-17/`（判决表+§五追记）+ `survey_harness_design_2026-09-15/`；**本轮查新（进行中）**：`.research_tmp/literature/survey_positioning_2026-09-16/`（AGENT-A 规模条件性/B demand-driven content/C 保真度治理/D delta 继承）
- 废弃轴清单（禁当现系统创新）：n-ary 超图表达力/schema 自演化/预计算本身/覆盖地图概念/冲突消解首创/provenance 概念——详见 memory stageB-impl-progress ⑧

## 7. 模型与 provider 现状

- 建库：Qwen3.6-27B（Paratera，便宜，已授权小规模直接跑）| 答题：Qwen3.8-Max（钱大头，>1M 需批）| 判分：Kimi-K2.6（主判）+gpt-oss-120b（二判，多 judge κ 立信用）
- 本地 Qwen3.8-27B：用户部署中（部署好后答疑成本近零，可上忠实规模）
- VLM：GLM-4V-Flash（读值准）；GLM-4.6V 思考型读值全错（禁用教训）
