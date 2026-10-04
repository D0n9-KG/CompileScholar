# 实验设计 v5（终版）：领域世界状态论文（2026-09-26）

> 本文件=新 session 实施交接的权威设计档。决策链 v1→v5 见文末附录。
> 叙事档：PAPER-NARRATIVE-0926.md（世界状态框架）+ NARRATIVE-ANALYSIS-
> 0926-draft.md（全部查新判决与一手引文）+ eval-anatomy-0926.md（证据
> 语法）+ benchmark-audit-0926/FINAL-VERDICT.md（基准审计）。
> 用户已裁（0926 晚）：槽3 Multi→**ScholarQA-CS2**；槽2 加图方法对手
> （LightRAG）；全文臂降参考臂（槽1除外）；开放任务=**领域热启动基库+
> 题内增量生长**；执行序=**CS2 先跑实战体检边测边调**；本地算力拉满+
> 并行设计；检索=Sciverse 主打国内已定；ScholarStack/Lacuna 数字不引用；
> OpenScholar 整体放弃；MDAQA 用社区 oracle 形态；SciREX 弃用；
> Multi-108 闭卷全部数据（0.6503 三臂）退为内部开发证据不进论文。

---

## 一、主张与槽位对应

论文主张：科学 agent 需要研究领域的**世界状态**（治理化维度空间上的
定位主张+可比性等价类+类型化未知+时间切片+更新语义），不是文献零件。

| 主张 | 槽位 |
|---|---|
| S0 单篇能力守恒（防御非劣） | 槽 1 PeerQA |
| S1 状态优越·固定语料（同信息访问，知识组织定胜负；含对图方法） | 槽 2 MDAQA |
| S1' 状态优越·开放世界（broker 实战：热启动库+检索+编译+答题闭环） | 槽 3 CS2 |
| S2 领域认知建模（状态→综述/领域报告：全景/比较/缺口/动态） | 槽 4 ReportBench + 槽 5 FieldState |
| S3 综合合法（引用忠实+误用率低） | 槽 3/4 判分维度+检测器横切 |
| S4 状态保真 | 横切审计章（不进 PK） |
| S5 状态可更新（吊销+生长复利） | 横切演示章+槽 3 生长日志 |

## 二、五槽主实验矩阵（终版）

### 槽 1｜PeerQA（单篇 QA，防御非劣）
- 数据：208 篇/579 题（495 可答性标注，含不可答子集顺带喂否定分析）；
  官方 answerability 指标+ROUGE，全确定性判分
- 对手×2：**PaperQA2（单篇模式）+ 全文直读臂**（单篇 ~15k tokens 直接
  入窗，此槽全文臂=正牌对手=非劣参照系）
- 我们的臂：单篇 mini-KB（记录+卡片，跳过跨篇视图）同一答题循环
- 判据（预注册非劣）：Answer-F1 ≥ 全文臂 −5pp 且 token 不高于全文臂；
  Evidence-F1 期望反超（loc 锚定主场）；失守→诚实报+主张收窄
- 成本：轻（~1 周含建库）

### 槽 2｜MDAQA 社区 oracle（固定文献多篇 QA，含图方法对手）
- 数据：6,804 题中预注册分层子集 200-300（比较型优先）；每题 gold
  社区 2-13 篇全文**全臂同喂**（ScholarStack 同款公平形态，oracle 设定
  披露）；SPIQA 25k 固定池背景
- 对手×2：**PaperQA2（社区模式）+ LightRAG（社区模式，图式 RAG 代表，
  基建现成）**；HippoRAG 2=可选第三臂（P2）；全文直读=参考臂（天花板
  探针，回答"为什么编译"，不计入对手）
- 判分：主=我方 rubric 装置（GLM-5.3+多 judge κ+gold 探针天花板）；辅=
  官方四重叠指标双报（已知低估我方=披露不当主判据）；MDAQA 弱 gold
  （LLM 合成参考）披露
- 可引已发表数字：MDAQA 论文（EMNLP 2025 Findings 在刊）检索轨天花板
  BM25 R@10 0.35 / BGE-neighbor 0.55（引用行，我们检索不单独测）
- 附加：信息访问宽度三档梯度（社区 oracle→+20 池内干扰→全池检索，
  同题三档=规模故事载体，P2 可裁）
- 成本：按题 mini-KB×250≈3-5 GPU 天+四臂答题≈1.5-2 周

### 槽 3｜ScholarQA-CS2（开放文献 QA/证据报告，**先跑·系统实战体检**）
- 数据：AstaBench（ICLR 2026+AI2）dev 100+test 100；HF gated-auto 已
  实拉；真实 OpenSciLM 用户查询，输出=分节报告 JSON（sections+inline
  citations+真实 snippets，解析失败=0 分）；知识截止 2025-05-01
- 判分：官方四维等权（ingredient_recall/answer_precision/citation_
  recall/citation_precision，citation 类占 50%）；judge=GLM-5.3 经
  InspectAI `-T scorer_model=openai-api/glm/glm-5.3` 零代码替换+
  json_schema 兼容冒烟+与官方 gemini 分数相关性小验（披露替代判分）
- 对手×2：**①memorized baselines 重判**（AI2 官方预存答案：Elicit/
  Perplexity Sonar DR/OpenScholar/SciSpace=真实产品数字，同 GLM judge
  重判=matched-judge 锚，零跑动成本、说服力最强）②**Claude-Code 级
  harness+27B+检索 MCP（开网）**（matched-model 深度研究臂；先例=
  ScholarStack 用 Claude Code DeepResearch 当对手）
- 我们的臂：热启动基库（§三）+broker 循环内检索（Sciverse 主打）+每题
  增量编译+报告适配器（主张→分节 JSON+KB 真实 snippets=1.0 档 citation
  recall；官方自认不验 snippets 真实性→我们干净优势）
- **熔断纪律**：dev 20 题先跑测形态税（AirQA 前科；损耗过大→叙事编译
  层优先级提到最前或启用备胎 LitTraceQA）；dev 100 主跑；test 100 视
  dev 结果+预算裁
- 预注册诊断：每题检索召回质量日志（A6 教训=词汇鸿沟，循环内检索是
  生死线）；生长日志（每题粗抽/挂边/升级数=飞轮 S5 的过程证据）
- 成本：报告适配器+每题检索建库管线=阶段 0 主工程；答题 100 题×三源
  ≈1-2 周墙钟+API 预算

### 槽 4｜ReportBench-ML 25 任务（开放文献综述）
- 数据：ByteDance-BandAI（arXiv 2508.15804，Apache-2.0，HF 可下）；
  600 篇专家综述 gold→逆向 prompt（**带时间约束**=as_of 原生对齐）；
  ML 子集 25 任务（Lacuna/ScholarStack 使用先例=生态位认可，**数字不引**）
- 对手×2：**STORM（Stanford 31.5k★，27B 后端）**+ Claude-Code 级
  harness+27B（开网，同槽 3 臂复用）
- 判分：micro citation P/R/F1（对 gold 引用集，确定性）+RACE point-wise
  （独立 judge=GLM-5.3，装置同槽 2）+时间泄漏检测（引截止后文献判负）
- 我们的臂：热启动基库（按每题 temporal constraint 做 as_of 过滤）+
  broker 检索增补+叙事编译层写综述
- 成本：STORM 本地化冒烟+25 题×4 臂≈1-1.5 周

### 槽 5｜FieldState-Bench（自建·领域认知·固定文献断网）
- 规格（v4 §四沿承）：3-5 领域×~300 篇 arXiv OA（跨度≥5 年）；一次
  编译多切面（as_of 免费切片，每领域 2 截止点→6-10 报告单元/臂×4 节）；
  四节报告（全景/比较/缺口/动态）；**未来集（T+1~T+2）当 gold 全臂
  隐藏**：缺口节=未来填补命中率（时间箭头判分，零标注）/全景节=未来
  引用覆盖率/动态节=manifest 年份确定性核对/grounding=引用 P/R+泄漏
  判负/数值纪律=误用检测器/RACE 辅轨
- 对手×2：**STORM（语料限定检索后端）+ Claude-Code 级 harness+27B+
  语料 MCP（断网）**；可选第三臂 LightRAG（300 篇语料上图方法有意义，
  P2）
- 固定文献+断网理由（用户已确认）：防未来集泄漏/公平归因（同语料同
  信息访问）/与槽 3、4 开放形态分工
- 占位划界必引：GiantsBench/Hakken/BackTrend（条目级预测 vs 我们报告级
  +状态驱动+客观时间验证）
- 发布：语料 ID+prompt+scorer+未来集随论文发布（benchmark 贡献）
- 成本：语料采集+编译 30-50 GPU 时一次+gold 管线工程 ~1 周+四臂报告
  生成≈数天

## 三、开放任务初始 KB：领域热启动基库+题内增量生长（用户已裁）

- **构成**：Tier-1 粗抽 2-5k 篇 CS/ML 通域论文（coarse_extract ~12s/篇
  ≈10-16 GPU 时）+ Tier-2 深抽 200-500 篇高被引核心（~2min/篇≈7-17
  GPU 时）+ 回流边+注册表合并
- **时间纪律（硬）**：基库按任务截止过滤——CS2 全部 2025-05-01 前；
  ReportBench 按每题 temporal constraint 以 as_of 切片供给；泄漏检测
  判分侧兜底
- **题内生长**：broker 检索（Sciverse 主打）→Tier-1 粗抽→mentions
  回流挂边→确定性升级规则标记 Tier-2 候选→（答题循环外）深抽入账；
  每题生长日志=飞轮过程证据（S5）
- **披露**：基库构成/建库日期/截止过滤方法；冷启动对照=消融章（封存）
- **理由**（记录在案）：空白起步=每题从零建库延迟爆炸+状态退化为
  临时小库（与按题编译无差别）；热启动=产品真实形态+飞轮有初始库+
  成本摊销进账本叙事

## 四、横切章（不进 PK 表）

1. **S4 保真审计**：分层人工抽检 200 单元+canary 三套+registry 审计
   （0.11%）+postcheck 统计（95% first-pass）——6/6 近邻全空位
2. **S3 误用检测器总表**：五检测器扫五槽全部臂答案/报告→误用率对比
   +50 样本人工复核+检测器-人工一致性披露（裁判依赖披露）
3. **S5 更新演示**：失效注入（k=10/50/100 分层，blast radius 查全查准
   vs 全量重建差集=地面真值+重算等价性+摊销比+对抗案例）+槽 3 生长
   日志汇总（复利指标）
4. **成本三本账**：建库（GPU 时/篇分 Tier+热启动基库一次性）/查询
   （per-question tokens 全臂）/更新（摊销比）
5. **失败分析**（预注册呈现义务）：检索召回上限案例/形态税实测/死区/
   残留空返回

## 五、本地算力拉满与并行设计（用户指令：拉满）

- **GPUStack 27B 服务**：先修 P9（单流 10 vs 20 tok/s=副本疑缺）——
  查副本数/显存分配；vLLM continuous batching 开启；答题 marathon 按
  A2 教训用**单一全局并行池**（workers×chunk-threads 复合会尾部饿死）；
  池宽从现网吞吐实测校准（8 连发零失败的 Sciverse 限流经验同法用于
  本地并发压测）
- **embedding 独立服务**：qwen3-embedding-8b 与推理分卡/分服务，避免
  互抢（same-model-same-dim 纪律不变）
- **judge 走 Paratera GLM-5.3**：不占本地卡；判分与答题流水线并行
  （增量判分 loop 现成：答一题判一题）
- **崩溃安全**：WAL/分片检查点/增量落盘全套复用（deep_extract/
  postcheck/judge 三处现成）；marathon 显式传长 timeout（0925 全流程
  测试教训：默认 240s 装不下 9k-token 输出）
- **墙钟预算**：每阶段跑前出预算表（题数×步数×tok/s×并行度），跑后
  对账（账本现成）

## 六、执行序（用户裁定：CS2 先跑体检）

```
阶段 0（前置工程，~2 周）
  0.1 harness 代理层冒烟：Claude Code/Codex CLI × 27B 胜任力
      （多步工具循环稳定性+格式合格率；不胜任→双试→降级自制
      ReAct 臂并披露）
  0.2 报告适配器（主张→分节 JSON/Markdown，槽 3/4/5 共用=叙事编译层）
  0.3 检索/语料 MCP 服务器（search_text/fetch_chunk/Sciverse 包装）
  0.4 InspectAI+GLM judge 冒烟（json_schema 兼容+官方分数相关性小验）
  0.5 GPUStack 拉满调优（P9 修复+并行池校准）
  0.6 热启动基库构建（粗抽 2-5k+深抽 200-500，截止过滤）
阶段 1（槽 3 CS2 先跑=系统实战体检）
  dev20 熔断 → 问题归因修复 → dev100 主跑 → 边测边调迭代
  （修复走归因驱动纪律；论文数字出自定型后冻结 run）
阶段 2（系统迭代收敛后铺其余槽）
  槽 2 MDAQA → 槽 1 PeerQA → 槽 5 FieldState 构建+跑 → 槽 4 ReportBench
阶段 3（定型冻结）→ 阶段 4（消融章：阶梯/band-off/门开关/冷启动对照）
阶段 5（横切 3 失效注入+生长汇总）→ 成文 → 投稿
```
工期：阶段 0-2 ≈ 8-10 周；六月 EMNLP 档全量从容；一月 ACL 档需砍
槽 4 或槽 5 之一（不推荐砍 5）。**会场未裁**。

## 七、诚信与披露装置（全案统一）

逐槽预注册（判据冻结/可证伪预测/诚实预案：CS2 形态税过大→熔断披露；
harness 不胜任→降级披露；FieldState 缺口命中无差异→主张降描述性；
PeerQA 非劣失守→主张收窄）；gold-blind 建库；子集化披露（MDAQA/
ReportBench）；oracle 设定披露（MDAQA 社区）；MDAQA 弱 gold 披露；
judge 替换效度小验（CS2/RACE）；热启动基库构成披露；自建基准全规格
发布；检测器裁判依赖+人工一致性；预算对账口径；外部数字仅引已发表
论文（ScholarStack/Lacuna 预印本数字一律不引）；所有 TBD 保持 TBD。

## 八、开工前必核清单（新 session 第一件事）

1. harness 代理层×27B 胜任力冒烟（决定 Claude-Code 臂成立性）
2. CS2 数据/判分链路端到端冒烟（HF 数据已在盘：benchmark-audit-0926/
   实拉件；InspectAI+GLM json_schema）
3. ReportBench HF 实拉+judge 可换性
4. MDAQA license 精确串+数据完整性（mdaqa-data.json 5.2MB 已在盘）
5. PeerQA 数据与官方评测器实拉（Baumgärtner et al. 2025 出处落实）
6. STORM 本地后端兼容冒烟
7. GPUStack 副本/吞吐体检（P9）
8. Sciverse key 配额与限流实测（两 key 分仓）
9. 热启动基库的论文来源清单（自家 2,355 库+arXiv 补充，截止过滤脚本）
10. FieldState 领域选定 3 vs 5（用户裁）+采集清单

## 附录：决策链（v1→v5）

v1=verified compilation 论点+消融为中心（废：论点换）→ v2=世界状态
论点+四条腿（废：形态矩阵抄 ScholarStack+过重）→ v3=八形态矩阵
（废：仍非从论文主张推导+量大做不完）→ v4=五槽+具名对手（用户逐条
修正）→ **v5=CS2 换 Multi+图方法对手+热启动基库+CS2 先跑体检+算力
拉满（本档，用户定稿）**。
关键用户裁定原文位置：本 session 对话 0926 晚（Multi 弃用/ScholarStack
数字不引/OpenScholar 放弃/FieldState 固定文献确认/执行序 CS2 先行）。
