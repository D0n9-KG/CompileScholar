# VERDICT-L1-0928 — deep_read L1 定向模式判决（批 10）

日期：2026-09-28（凌晨-清晨自主推进段）
状态：**L1 成立（批 10 判据全过）**；判分补齐进行中

## 一句话

四级下潜阶梯的 L1（定向深读）落地：批 10 同五题 5/5 全答、弃答 0/5
（预注册判据 ≤2/5）、deep_read 观测全部为真实记录数（720-1029 字符，
零 248 错误码）、模型 10/10 主动使用 sections 参数——**Agent 决定读哪
部分的设计被真实行使**。

## 预注册判据 vs 实测

| 判据（批前冻结） | 目标 | 实测 | 判 |
|---|---|---|---|
| deep_read obs | 大数值非 248 | 720-1029ch，0 个 248 | ✅ |
| 弃答率 | ≤2/5 | 0/5 | ✅ |
| sections 参数使用 | Agent 会用 | 10/10 次显式传参 | ✅ |
| L1 时延 | ~3min/篇 | 缓存命中 0s；未命中 3-5min | ✅ |
| L2 不阻塞答题 | 后台 | 批内 10 篇 L2 后台补齐 1397 条 | ✅ |
| 批 10 分数 | 纵向上升 | **待判分**（2/5 已出） | ⏳ |

## 三大断口（本日发现+修复，全部是消费端断裂）

1. **deep_read 全灭（批 5-9）**：只进 TOOL_SIGS，没进 TOOL_WHITELIST
   也没挂 kb+records_target——40+ 次调用全撞 'unknown tool'（248ch obs）。
   → attach 三件套齐挂。
2. **runner 载错 records 文件**：载 coarse_records.json（title 键），而
   1896 键桥接落在 records_merged.json——需求池 793 篇+hub 深抽对模型
   不可见（Q1 三篇 in-topic 论文查 0 记录→弃答）。→ 改载 merged。
3. **适配器看不见深记录**：deep_read 终化记录只活在答题进程内存，
   report_adapter（独立进程）解析不了深记录回指→空壳报告。→
   deep_read_records.json 持久化+适配器叠加层（合并非替换）+
   recover_deep_records.py 离线重放（批 10 的 10 篇 1397 条已恢复）。

**共性教训**：修复必须追到消费端。断口 2/3 都是"修在 A 文件、B 端读
的是另一个文件/进程"——键桥接修在 merged 里而 runner 读 coarse；深
记录入账在进程内存而适配器读盘上文件。

## 行为观察（下一批的改进输入）

- **捞取断口**：0919a852 深读 magneto（56 条）+DL 论文（67 条）但笔记
  只落 1 条——obs 只报数量，模型不再花一步 findings 捞内容。修复=
  L1 obs 嵌 sample_records（6 条最有信息量记录，数值类优先，带
  record_id 可回指）——批 11 生效。
- sections 关键词形态（"pps"/"oscillator stability"）比精确节名常用——
  匹配器按节题+前 300 字符 token 重叠（≥0.6）+子串，双向兼容。

## 纵向分数（GLM-5.3 direct_judge）

| batch | judged | ingredient | precision | citF1 | global |
|---|---|---|---|---|---|
| **batch10（L1 生效）** | **5/5** | **0.31** | **0.80** | **0.75** | **0.621** |
| batch5（深读灭） | 4/4 | 0.20 | 0.88 | 0.54 | 0.539 |
| batch6（深读灭） | 2/3 | 0.26 | 0.71 | 0.56 | 0.510 |
| batch7（深读灭） | 4/5 | 0.27 | 0.65 | 0.51 | 0.478 |

参照：harness dev18=0.733；ours batch1=0.571（断键基线）/ batch4=0.623
（2/5 真答，inspect-eval 口径）。

**批 10 单题**：EVM 0.672 / schema 0.444（捞取断口）/ active-learning
**0.729**（run1 同题完全弃答→precision 1.0+citF1 0.90，KB 修复+深读
的数字兑现）/ GNSS 0.605 / embedding 0.661。

**判读**：深读生效 vs 深读全灭的同题带 = **+8~14pt global**；对
harness 的剩余差距集中在 ingredient recall（0.31 vs harness 随时读
原文）——下一杠杆是 sample_records 预览（捞取断口修复）+ L1 覆盖面
（每题 2 篇×3-5 节）。precision 0.80/citF1 0.75 说明 verbatim quote
经编译层落进了报告——citation 维是我们 KB 路线的出击点判断成立。

## 工程资产（本日新增）

- external_tools.py：deep_read L1/L2/缓存/持久化/预览全链
- recover_deep_records.py：缓存→终化记录离线重放（确定性）
- adapt_batches.py + score_compare.py：批次判分管线
- direct_judge.py：AsyncClient 补丁修复（elicit 判分挂死根因）
- base_kb/deep_read_cache.jsonl + deep_read_texts/ + deep_read_records.json

## 遗留（按优先级）

1. 判分补齐：b5/b6/b7/b10/elicit（GLM 重试螺旋是税，21 次耗尽计
   None 可删条目重判）
2. 批 11：验证 sample_records 预览对捞取断口的修复效果（判据：
   0919a852 同题 notes 回指数 1→5+）
3. absence 定向化（L2 已异步，优先级降）
4. GLM 判分 JSON 畸形（"criteria" 字段缺失型）——系统性观察在案，
   需向 scorer 侧报（或换 judge 时披露）
