# 预注册 0.6：CS2 热启动基库构建方案（2026-09-27 冻结；修订案 A'' 见文末）

> 依据：EXPERIMENT-DESIGN-WORLDSTATE-0926 §三（开放任务初始 KB）+§八-9。
> 用户裁定（09-27 对话在案）：采集=arXiv bulk 主打；Tier-1 规模=2k。

## 修订案 B（2026-09-27 深夜二次修订，取代 A''，深抽启动前生效）

**Tier-2 骨干=综述为主（survey-first backbone）。用户核心裁定：**
"热库的关键需求是把领域骨架搭起来，综述正是这个任务最好的信息源"；
"应该为综述论文专门开发一套抽取来适配系统，识别到综述不用跳过
直接用这个模块"；"把综述论文的价值尽可能发掘出来为系统所用"。

**构成（三层分工）：**

1. **综述层（100-200 篇，survey_extract 专门抽取）**：骨架——
   谱系边/族结构/领域快照/缺口地图/registry 空心实体条目
2. **hub 供血层（~80 篇，标准深抽）**：顶级方法实体的第一方
   记录（config/result）——空心锚点变实心；选取=跨综述共识
   频次（references 交集）+高引补盲
3. **题内生长（答题时）**：中长尾+题目特定实体供血

**综述选取**：六大类子领域核心综述（OpenAlex type=review+引用
排序+截止 ≤2025-04-30+领域白名单 Physical Sciences）。

**hub 选取**：综述池 references 的跨综述共识频次（被 ≥N 篇综述
同时引用=领域共识骨干，纯领域信号、与题无关）top ~250-300 中
取非综述高引 ~80 +引用 top 但零综述收录的盲区补充。

**A'' 的领域闸教训（保留）**：OpenAlex 概念+Physical Sciences
白名单+全文可得硬筛（三道闸实测把社科渗漏清零）。

**历史注记**：A'' 的纯引用骨干池（backbone_pool.json 300 篇）
作废留档——社工渗漏三道闸修复过程保留为方法记录；其"综述排除"
规则被本修订案反转（综述从被排除者变为一等公民，代价=专门
抽取器，见下）。

## 三、综述抽取器 survey_extract（修订案 B 核心新件）

四类记录（S1-S4）+语义分节路由，详见
base_kb_build/survey_extract_design.md（设计稿）。

关键设计点：
- 综述无预设论文结构——分节用语义标签（taxonomy/comparison/
  chronology/challenges/methodology）替代论文六标签
- 转述不进事实层：S4 受控转述记录 epistemic="survey-claimed"+
  claims_about 被转述方，与一手记录分轨
- 缺口四档：综述判断（survey-claimed absence）与语料推导缺席
  严格分开
- 识别路由：粗抽层打综述标→深抽自动分流（题内生长摸到综述
  同样受益）

## 三、产物落位

```
experiments/benchmarks/cs2/base_kb/
  manifest.json          # 2k 行：paper_id=arxiv_id 优先/title-year 兜底
  demand_rehearsal.json  # dev100×每题命中（Sciverse 原始返回存档）
  cutoff_filter_log.json # 被滤论文+理由
  coarse_records.json    # Tier-1 粗抽产物（coarse: 前缀 id）
  tier2_gain_table.json  # 增益表（夜2 冻结）
  tier2_pool.json        # Tier-2 池终版
```

## 四、披露义务（论文用）

基库构成（来源分层/规模/截止过滤方法/构建日期）+Tier-2 增益判据全文
+预演检索仅用 dev 题声明+冷启动对照留消融章（封存纪律）。

## 五、可证伪预测（预注册）

- P1：需求核心池对 dev 题的覆盖显著高于通域随机层（否则预演检索无效）
- P2：Tier-2 族内密度对 compare/lineage 工具可查结构数的增益为超线性
  （否则"族整体"逻辑不成立）
- P3：粗抽 limitation 信号与深抽 absence 记录数正相关（否则 gap_signal
  权重无意义）
