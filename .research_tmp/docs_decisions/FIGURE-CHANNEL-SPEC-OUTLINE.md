# 图通道（多模态）设计规格提纲 v0.1 — 2026-09-11 凌晨（供用户评审，非最终规格）

> 地位：D14 授权（"抽取层效果达预期后开始做"——重建验收已过，本提纲=规格流程第一步）。
> 流程纪律：前沿调研（部分已在档）→ **本提纲评审** → 正式规格 → 用户批准 → 代码进 src/。
> 本文只列设计骨架+已实测的证据+开放问题，不预设实现。

## 1. 需求锚定（为什么做）

- LitTraceQA dev55：~25% 题证据在图表；AirQA：image 标签题 207/1246（客观子集内亦占一成以上）
- 现有 KB 是纯文本闭语料：图内容完全缺席（卡片层只记"存在哪几张图表"）
- 差异化机会：无任何开源竞品系统原生支持图证据定位（LitTraceQA 尽调实证）

## 2. 三层架构（D14 骨架，实测数据已支撑选型）

**L1 免费清单层（零 VLM 成本）**
- 内容：每张图/表的 locator（caption+bbox+页码+类型）
- 数据源实测：AirQA processed_data **官方直接给**（figures: caption/bbox/page；tables 含 HTML 全文）；PS16 mineru_or 有 images/ 目录+content_list.json；自有管线=MinerU 输出同构
- 入 KB 形态：paper_card.figtab 槽扩展 → 结构化 figure_inventory 记录（新 kind 或卡片字段，规格阶段定）
- **表格特殊通道**：mineru/AirQA 已把表格转 HTML 文本——表格内容走现有文本抽取（重建已验证 A7/A8 教科书级表格记录），**表格不需要 VLM**；图通道只管"图"（曲线/散点/架构示意）

**L2 选择性描述层（批量 VLM）**
- 对象：正文引用过的图（引用图注匹配→优先级）+ 清单层筛出的数据图
- 模型：**GLM-4V-Flash**（实测 3.5s/张、修正后 19/20、max_tokens≤1024 硬上限）
- 产物：每图一条 figure_finding 类记录（描述+读数+质量旗标），嵌入检索可及
- 幻觉闸（设计难题②）：描述层输出与 L1 清单/caption 交叉校验；数值读数必须带"从哪根轴/哪个点读出"的定位串；批量抽检协议（合成 ground-truth 图集——今晚的 vlm_compare 五图组直接转产为回归测试集）

**L3 按需实读层（答题循环内工具）**
- 工具：read_figure(figure_id, question)——答题 agent 遇到图证据需求时调用
- 模型：**GLM-4.6V**（实测 20/20、需大思考预算、25s/次可接受因为是按需单发）
- 证据机制（设计难题①）：图生主张无逐字原文可引 → 证据三元组 = **locator（bbox+页码+图注逐字）+ VLM 读数 + 质量旗标（模型/时刻/单次或多数投票）**；答案引用形态 "(Author et al., Year, Fig.3 读数)"；审计面保存原图裁片路径
- 编号纪律沿用：图生记录有独立 id 空间，写入闸=locator 确定性校验（bbox 在清单层存在+页码在范围内）——不靠 LLM 自证

## 3. 已实测的工程事实（今晚前置核查产出）

- VLM 终选数据：Flash 3.5s/19分、4.6V 17.4s/20分、ERNIE 6.2s/19分（折线图一处网格误读）、CST 多模态不可用（4/5 超时）
- 幻觉陷阱题（不存在的柱子）：四家 Paratera 模型全部正确拒答——单发拒答能力已验证；**批量幻觉率未测**（规格阶段必做，n≥50 图集）
- **判分员读图不可信实锤**（今晚事故）：我把 Table 9 误认成 Table 2 做真值——图真值锚定必须走"caption/locator→原文表格 ID"的确定性映射，禁止人眼/LLM 读图定真值
- 传图通道：Paratera OpenAI 兼容 base64 可用；CST 官方推荐 URL/内网文件中转（生产若走 CST 需要中转服务，当前选型 Paratera 无此问题）
- 裁图能力：bbox+页码→PDF 裁片需要 pdf 工具链（pymupdf 级，本地零成本）——AirQA 场景只需对 image 题绑定论文选择性下载 PDF

## 4. 开放问题（评审要拍的）

1. figure 记录的 schema 形态：新 kind（figure_finding）还是 finding+source_type=figure？（schema v1.3 仲裁一并处理——finding 的 cited 缺口也在同一轮）
2. L2 描述层的进入门槛：全部数据图 vs 仅正文引用过的图（成本差 ~5-10×）
3. LitTraceQA 试跑用哪层进场：文本+L1（既有裁定）→ L2/L3 何时补
4. 图证据的判分协议：官方 evaluate.py 的 evidence 定位吃 locator——L1 就够还是要 L3 读数
5. 多数投票/双模型复核的适用线（哪些读数值得花两次 VLM）

## 5. 排期建议

规格细写（含 schema 提案+批量幻觉测试协议）→ 用户批准 → 实现 L1（一天级，纯确定性）→ LitTraceQA 试跑带 L1 → L2/L3 按试跑缺口决定深度。

## 6. 禁做清单（既有裁定）

- MinerU chart 数字化表禁用（按轴刻度编造，已证伪）
- RAG-Anything 的 chart 走数字化表路线不抄（其实测=GenericModalProcessor 吃表不看原图，我们绕过=差异化）
- 比赛资产代码零搬运（D10）
