# 抽取栈重写验收 — Run 1（联合抽取 + 严格 gate 版）

日期：2026-08-25。口径：DECISION-extraction-rewrite.md 预注册。run：REWRITE_PPR_24493BE6E8C2（261s=4.3min，注入探针同款 gold pattern）。

## 数字（不粉饰）

| 指标 | 结果 | 门柱 | 判定 |
|---|---|---|---|
| 语义正确率（双 judge both-pass 严格口径） | **9/37 = 24.3%** | ≥70% | **未过** |
| 单 judge（GLM-5-Turbo / qwen3.5） | 各 14/37 = 37.8% | — | — |
| judge 一致率 | 27/37 = 73% | — | 尚可 |
| 效率 | 261s = 4.3min/篇 | <5min | **过** |
| verbatim/locatability | 规则层 100% 保证 | 100% | 过 |
| 机制不回退 | 演化 v0.3→0.4 + pattern 采纳（4 条注入边进 kept） | 采纳仍工作 | **过** |

## 对照基线（探针宽松口径 ~20%）——重要 caveat

探针基线 20% 是**宽松口径**（我人工裁），本轮 24.3% 是**双 judge 严格口径**。口径不可直接比；可比的是失效模式：

## 关键正面结果：探针失效模式已治愈

1. **槽位绑定（探针核心病灶）治愈**：4 条注入 pattern 边全部语义合格（3 条 ablates 分列三组件 + 1 条 outperforms_on 同句绑定正确 loser）——探针 0/5 的失效（跨句偷换/绑类别/范围泛化）在联合抽取下**零复发**。ablates 被 composed_of 抢走的探针病灶也治愈（3 条 ablates 正确）。
2. gate 审计确认 8/38 拦截是正确拦截（真错边：方向反/算法伪代码行 mash/绑错节点）。

## 失败归因（18 条 both-fail 的结构）

**14/18 是 `influences`，且 reason 高度一致**：把"属性陈述/定义/时序条件/视角限制"绑成 influences（"task is partially observed"、"feedback received after thousands of steps"、"states are perceptually aliased"）。这不是绑定错——是 **pattern 语义选择错**：influences 的 boundary（"X functionally depends on Y"）没被遵守，joint prompt 里没有 step3 时代"X depends on Y"的直接判据示例。

→ **修复路径明确且便宜**：joint prompt 的 schema 段把 influences 的 boundary 直接写进判据位（或 _retrieved_schema_prompt 渲染 boundary 时对 influences 这类高频误用 pattern 加判据示例）。这是 prompt 层修复，不是架构问题。

其余 4 条失败：composed_of 2（"combines paradigms"是分类不是结构组成）、defines 2（"selected by"是选择不是定义）——同属 pattern 语义选择，同路径修复。

## 次序说明

本轮 judge 评的是**严格 gate** 存活边；gate 放宽（commit 6c0085a3，误杀审计后按预注册放宽）会让 ~17-19 条误杀边回流。**Run 2 = prompt 修复 + 放宽 gate 后重跑 + 重 judge**，才是终验口径。

## Run 2 计划

1. `_JOINT_PROMPT` 加 pattern-选择判据强化（influences/composed_of/defines 三个高频误用 pattern，从 seed boundary 提炼判据句）
2. 重跑验收 run（同 DQN + 同注入）
3. 重跑 judge 交叉（同双 judge 同口径）
4. 预注册判定不变：both-pass ≥70% 过门柱
