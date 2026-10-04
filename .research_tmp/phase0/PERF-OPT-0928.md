# 性能优化档：CS2 双臂慢因分析与修复（2026-09-28）

## ours 臂画像（995 调用 / 395 分钟实测）

- p50 延迟 105s；prompt p50=**76k tokens**；55% 调用 >30k prompt
- 步数中位 47（cap=48 顶格）——模型在磨
- 题间并行 fanout=4

## 根因（按贡献排序）

1. **grounding 全量注入**（已修）：build_grounding 在 CS2 大库
  （2,100 篇/15,100 实体）产出 148k tokens（papers 全量清单 164k
  chars 为主）——Multi-108 的 430 篇库上无感，库一大线性爆炸；
  每步重复 prefill
2. 步数 cap 48 偏大（行为参数，待用户裁）
3. fanout=4 偏保守

## 已做修复（下一轮生效，当前 dev20 不打断）

- **papers grounding 按题裁剪**（evidence_gate2r_harness.py）：
  题面 token 命中 title/paper_id 的论文优先，上限 60 篇
  （GROUND_PAPERS_CAP 可调）；GROUND_PAPERS_FULL=1 回滚旧行为
- **实测**：dropout 题 grounding 148k→2.3k tokens（-98%）
- 语义安全性：grounding 是导航目录不是库本身——typed tools
  （search/card/compare）查的仍是全库；裁目录≠裁库。词汇鸿沟
  风险（标题不含题面词的相关论文从目录消失）由工具层兜底
  （search 嵌入检索全库）+ 下一轮 A/B 抽查验证

## 待用户裁

- 步数 cap 48→30？CS2 是新基准无历史可比包袱；中位 47=大量题
  磨到顶格，理论上 cap 收紧能强制早收敛。风险：复杂题被截断。
  建议：30（与 p90 对齐）

## harness 臂（串行 17 分钟/题）

- 根因：单题内 25-40 轮工具循环串行（Claude Code 会话本质）+
  题间也串行（runner 没开并行）
- 修复：harness_arm_run.py 加题间并行（2 路——每题是独立
  claude -p 子进程，服务器余量已验证足够）
- 预期：20 题墙钟 5.7h→~3h

## 批 2 验收（2026-09-28，三项修改验证轮）——全部生效

| 指标 | 批 1 | 批 2 | 变化 |
|---|---|---|---|
| 批时长 | ~180 min | **16 min** | -91% |
| 步数 | 12-48（3/5 顶格） | 7-16（零触 cap） | 自然早收敛 |
| 重复率 | 36-77% | 0-14% | 饱和反馈生效 |
| 工具形态 | search_text 独占 | findings 主力（59>34） | typed tools 觉醒 |

**新暴露问题（批 3 修复清单）**：
1. notes 格式漂移：1/5 题自由格式无回指（短循环下笔记门容错轮次不足，
   格式漂移未被拒写纠正）→ 适配失败
2. 答案偏短（958-1905ch vs 批 1 3300-5100）+引用偏少（2 题 1 引用）
   ——精简过头？判分裁决
3. 批 1 遗留：880129/e03c49 的 notes 回指 id 与 records_merged 键不
   匹配（coarse 桥接），引用落低档

**判分对齐**：批 2 与 harness 批 2 同 5 题——判分出即三系统同题对比
（含批 1 的 5 题后共 10 题 ours 样本）。

## dev100 配置备忘

- OURS_QUERY_FANOUT=8
- grounding 裁剪生效
- harness 2 路并行
- （若用户裁 cap）--cap 30
