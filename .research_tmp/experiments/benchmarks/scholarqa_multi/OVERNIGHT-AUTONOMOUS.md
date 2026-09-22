# 09-23 凌晨自治执行计划（用户就寝授权）

> 用户原话授权（compact 前）："lightrag和paperqa2建完之后就进答题，我们的系统也是全部弄完之后就进答题。看门狗以及效率问题，设置好并行参数等，稳定的前提下把效率拉到最高。答完题就直接判分，或者可以先把答题和判分做成并行的，每个系统都是出一个题就判一个题的分，不要都等全答完题了再判分。我醒了会通知你，compact后你就一直推进吧。"

## 当前状态（01:10 快照）
- registry_growth run5 跑中（52/103 块，解析修复后）→ views → 建库收官
- PaperQA 索引 83/427（稳定门守护，建完自动开跑 108 题 fanout=3）
- LightRAG 复活中（200 篇重入队，保守参数 MAX_ASYNC_LLM=16/EMBED=4）
- watchdog.py 常驻（5 分钟轮询三线物理进度，WEDGED/DEAD 经 Monitor 中继）

## 执行队列（依序自动推进，无需请示）

### 阶段A：三线建设收官
1. registry_growth → views（build.py 断点续跑；views 落盘=我方 KB 就绪）
2. LightRAG 入库 430/430（watchdog 盯；若再嵌入超时：降 EMBEDDING_FUNC_MAX_ASYNC=2 重试）
3. PaperQA 索引稳定完成（磁盘稳定门：连续 3 次读一致）→ 自动开跑 108 题

### 阶段B：三臂答题（各自就绪即开，不等其他臂）
- 我方：evidence_gate2r harness（views.json 就绪后启动，OURS_QUERY_FANOUT=4；答题栈=_shared/tools/evidence_gate2r_harness.py+ps53r_run.py，需接 Multi qfile/records/views——数据加载层改造，参考 PS 时代 staging 约定）
- LightRAG：入库完 → multi_baseline_lightrag.py 查询阶段（LRAG_QUERY_FANOUT=6，断点 answers json）
- PaperQA：索引稳定完 → 自动进入（PQA_QUERY_FANOUT=3）

### 阶段C：答题判分流水线（用户明确指示：出一题判一题，不等全答完）
- Citation F1：确定性脚本（_shared/tools/citation_correctness_eval.py），每出新答案行即增量跑该题判分（写 per-question 分数文件，非一次性全批）
- rubric judge：judge_answer.py 框架（16 workers，GLM-5.3 走 Paratera 独立账本）；判分器接线前必须完成：
  ① Multi-108 官方 rubric 版本一手核验（手里 sqa2_rubrics_v2 是 S2 重组版——查 ScholarQABench repo human_answers.json 是否带 Multi rubric）
  ② GLM-5.3 判分接线按预注册（官方提示词逐字，Paratera 通道独立 ledger）
- 增量判分实现形态：watchdog 式循环——每 5 分钟扫三臂 answers 文件新增行 → 对新行跑判分 → 分数落盘 answers/*.scores.jsonl（append-only，可断点）

### 阶段D：判分汇总
- 三臂×108 题齐后：配对差统计（Citation F1 配对 t/CI，预注册主判据）→ 首版结果表

## 纪律（不豁免）
- 看门狗 WEDGED/DEAD 告警必须响应（取证→修→重启，断点全在）
- 杀进程前必须 Get-CimInstance 验命令行身份（今晚两次误杀教训）
- 纯度断言：每臂答题结束后 check_arm_purity（判分走独立账本不混）
- 出现无法自治修复的问题：取证入档（POSTRUN-FIXES 或新档），跳过该臂继续其他臂，晨报说明
- 效率：capacity.yaml 参数为准；GPU 空闲段（如一臂完成另一臂未起）可临时提高在跑臂的 fanout

## 晨报要素（用户醒时交付）
- 三臂建设/答题/判分各到哪、卡在哪
- 首版结果（若已出）：Citation F1 配对差表
- 夜间事故清单（POSTRUN-FIXES 增量）
- 待用户裁事项（rubric 版本核验结果、判分偏离披露等）
