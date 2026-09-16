# ASSET-STATE — 单一事实源（2026-09-16 建立，随进展更新）

> 目的：终结"状态散在 memory 恢复点/台账/判决档/LINEAGE 四处互相引用"的混乱。
> 任何"现在有什么/什么状态"的问题先查这页。路径缩写：`STAGEB` = `.research_tmp/experiments/stageB`。
> 更新纪律：每次换装 staging / 完成战役 / 冻结解冻一条线，当轮更新本页。

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
| `tests/`（65 个） | 含 `test_kb_compiler_import_gate`=机器闸（kb_compiler 禁 import 旧栈+schema 冻结守卫） | 活跃 |
| `archive/legacy/`（gitignored） | 死代码：granular_agent（Stage A 超图）/contest/run_kernel/run_cases/8 死测试 | 死·可逆归档 |

## 2. 知识库与 staging（三套）

| KB | 位置 | 内容 | 状态 |
|---|---|---|---|
| **PS16**（PaperScope 16 gold 篇） | `STAGEB/psfix_2026-09-11/psv5/`（血统见其中 STAGING-LINEAGE.md） | records v323（4206）+views v323（matrix 308/cards 1205/derived 8）+registry v35_f34（1725 实体） | 冻结（历史 +0.133/+0.83 判决口径） |
| **AirQA**（25 篇） | `STAGEB/airqa/s1_qa/`（staging）+ `airqa/s1_build/`（构建产物） | records v34（5515+表格）+views v34+registry v2s_f34 | 冻结（阶段一+批3a 收口） |
| **PS-53**（16 gold+37 干扰，规模实验专用） | `STAGEB/paperscope_r2/ps53/` | records_checked_ps53（**17822**=LLM 10691+表格 7131）+ views_ps53_full（matrix 1026/cards 1205/genealogy 416）+ views_ps53_16（同 KB 过滤 16 篇：4269 条/matrix 302）+ build_summary_ps53.json；建库 15.78M/2532 calls 全 Qwen3.6-27B | **活跃**（视图 2026-09-16 建成，F13 双臂 PASS；round2 实体生长=预注册决策点：distractor37 实体链接仅 6.3%） |

## 3. 答题栈（冻结态，解冻需用户授权）

- 位置：`STAGEB/rebuild27b_2026-09-11/`（harness）+ `STAGEB/airqa/`（airqa_run.py runner）
- 形态：evidence_gate2f harness + **F31V2 案A**（auto_transcribe 机械转写装置，G2_F31=off 字节等价开关在位）+ **NAV**（miss 时跨索引提示）+ IL-9b（元数据投影）+ PROJ_LEDGER（投影台账）
- 纪律：hybrid 冻结停止加补丁（09-17 判决：补丁堆=agentic 固有成本非设计病）
- 离线验证工具（零 token 可复用）：`STAGEB/airqa/s1_qa/` 下 e5_v2（五层严格仪器=唯一可信值级追踪）/f31v2_tests/nav_verify/f32_obs_level_verify/f28_forensics/recall_census2/format_forensics

## 4. 五条工作线（状态+解冻条件）

| 线 | 状态 | 解冻条件 |
|---|---|---|
| **PS-53 规模实验** | **活跃**：建库+视图完成 → 下一步=答疑相位预注册→预算批→四臂跑（ours×{16,53}+A9×{16,53}） | — |
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
