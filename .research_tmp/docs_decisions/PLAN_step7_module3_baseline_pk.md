# PLAN: step7 模块3 SOTA baseline 公平 PK

Date: 2026-08-14

## baseline 开源情况（已调研本地全文）
| baseline | github | 开源 | 我们的对标 |
|---|---|---|---|
| AutoSchemaKG | github (code+data fully available) | ✅ | schema 抽取+RAG，无自进化升层 |
| ASEE (Adaptive-EE) | github USTC-StarTeam/ASEE | ✅ | 事件 IE schema，cross-lingual，无升层 |
| IncSchema (hier_schema) | github raspberryice/inc-schema | ✅ | 直接 probe 不从低阶升，防幻觉 |
| Hyper-KGGen | 需查正文（arxiv 2602.19543） | ? | n-ary+自进化skill，最近邻 |

## 公平 PK 协议（同 LLM 同数据受控）
控制变量（纪律6）：
- 同 LLM：被测臂 deepseek，judge GLM-5，gold GLM-5.2
- 同数据：ARFM2024 19篇 / SciFact corpus
- 同 judge/指标：NMR/ERR/PSC + gold 覆盖

## 诚实方案（两条线）
### 线1: 消融臂模拟 baseline 核心（可立即执行）
我们的 pipeline 关掉各创新点 = 模拟 baseline 核心行为：
- F frozen = "无升层" baseline（对标 IncSchema 直接 probe 不升层）
- T text-only = "纯文本 schema 抽取" baseline（对标 AutoSchemaKG 文本驱动）
- 已有数据：F 0/3 gold, T 43 relations
这是**受控同 LLM 同数据**对比，最公平。

### 线2: 文献数字对比（baseline 论文报告值 vs 我们）
AutoSchemaKG/ASEE/IncSchema 各自论文报告的 NMR/ERR 等数字，
与我们同数据上的数字对比。诚实约束：数据/LLM 不同，数字不完全可比，
但趋势（升层 vs 不升层）可对比。注明 caveat。

## 我们独有的对比维度（差异化）
- 结构信号消融（citation/quals 各关）—— baseline 没有
- 自演化升层（F vs T/B）—— IncSchema 不从低阶升
- n-ary 超图 + pattern 拓扑（dep/con/comp）—— AutoSchemaKG 无

## 待执行
1. clone AutoSchemaKG/ASEE/IncSchema，确认能在同数据跑（需配置环境，工程量大）
2. 或用线1（消融模拟）作主对比 + 线2（文献数字）作佐证
3. Hyper-KGGen 开源情况待查（最近邻，重点对比）
