# 隔夜自主执行计划（2026-09-29 凌晨）

用户指令：睡觉，帮跑。外部方法直接在 benchmark 上跑不用讨论；ours 先跑 CS2 测试，
跑完看问题→修→迭代；题目逐步拓展；本地模型拉满+效率拉满，稳定前提。

## 已在跑（自动完成）
1. ✅ 批 16a/16b（V1-V3 验证双样本）——答题+适配完成，DeepSeek 判分进行中
2. ✅ STORM CS2 5 题——判分完成（均值 ~0.495，DeepSeek 尺子）
3. ✅ DeepScholar STORM 5 题——答题完成，判分管线待接
4. 🔄 harness dev20 重判（DeepSeek）——进行中

## 队列（判分完成后依次推进）
A. 批 16 双样本判读（V1-V3 效果：方差带收窄判据=DSL 题极差 <0.24）
B. ours 批 15a/15b 用 DeepSeek 重判（换尺全量重判纪律）
C. CS2 拓展题：批 17 = offset 15, limit 5（新 5 题——题目拓展第一步）
D. harness 双样本？暂缓（先看 16 结果）
E. DeepScholar 判分管线核验（LOTUS judge 需 OPENAI_API_KEY——明早问用户）
F. 问题清单更新：16 轮暴露的新问题（判读后归档）

## 纪律（不越权）
- 不改系统代码（修复要用户批准——除非纯 bug fix 接线类）
- 外部方法（STORM）各 benchmark 直接跑
- 每步结果落盘，早上汇总一份战报
- 判分模型=DeepSeek-V4.1-Flash（新尺子）；GLM 判的历史数字标记作废待重判
