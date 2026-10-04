# PLAN: step7 评测设计（顶会论文强评测要求）

Date: 2026-08-14 (细化 2026-08-14)
Goal step: 阶段顺序 step 7

## 评测方向（已定）
Intern-Atlas 式"综述当专家共识 gold 代理"。Intern-Atlas 协议（本地全文确认）：
- 30 篇高影响力综述建 gold（2268 nodes/1462 edges/133 evolution chains）
- NMR(节点匹配 91%)/ERR(边可达 89.7%)/PSC(路径语义 92%) + 人工核验边
- 附录 D.1 benchmark/D.2 lineage-search baselines/D.3 dataset

**我们的差异化**：dep/con/comp 多类语义边（不只 lineage 演化链）。
本机 MinerU 验证 PPR_017AD90905AA（本构定律+公式）dep/con/comp 全触发。
颗粒流域天然适合（综述论文含本构定律内容）。

## 四个评测模块

### 模块1: 综述复杂学科问题（创新评测，schema 路由发力处）
- 题型：局限-解决因果链 / 方向归纳 / 方法对比判断
- 设计：需高阶 schema 才答，frozen/低阶答不出（证明升层价值）
- gold：从综述论文抽取（综述=专家共识代理）
- 指标：answer accuracy + schema 路由命中率 + 跨方法关系引用准确率
- 对比 ResearchQA 已排除（朴素 RAG 就解，schema 发不出力）

### 模块2: 公开数据集（外部复现可信度）
- SciFact：段落级事实核查（已本地 s_scifact.html + s_SciFact_retrieval.html）
  - 已有 scifact-external-replication-succeeds 工作：MiniCheck 段落级 0.0000 vs rationale 0.428
  - 我们的超图抽取 vs MiniCheck baseline，复现已成功
- SciREX：科学 IE（方法/数据集抽取）——测 n-ary 超图抽取质量
- 借 Hyper-KGGen 的 HyperDocRED 技术（semantic+Hungarian matching 判 n-ary 对错）

### 模块3: SOTA baseline 公平 PK（同 LLM 同数据受控）
- AutoSchemaKG / Adaptive-EE / Hyper-KGGen / IncSchema
- 控制变量：同 LLM(deepseek 抽取臂)、同数据、同 judge(GLM-5)
- 指标：NMR/ERR/PSC + 我们独有的结构信号消融

### 模块4: 消融（各关看贡献，已有数据）
- split / lift / feedback / citation 先验(step3) / quals(step4)
- step5 全量 A/B 是消融核心两臂（纯文本 vs 全信号）：
  - A pure-text: 56 relations(extends11/improves5/compares37)
  - B full: 48 relations(extends5/improves9/compares31)
  - 结构信号: improves+80%识别真继承, extends-55%抑制过度extends
- step3 聚焦: 引用先验 3/15对 compares→improves/extends/background
- step4 聚焦: quals 4/15对 extends→更准 compares/background

## 现状衔接
- step5 ✅ 全量 A/B 数字已拿（模块4消融核心数据）
- step3/step4 ✅ 聚焦 A/B 单信号证据
- step6 ✅ 超图存储闭环（评测时超图按 paper_id 存取，复现性）

## 待执行（step7 实现工作）
1. 建 Intern-Atlas 式综述 gold（30 篇综述，dep/con/comp 多类边）
2. 跑 SciFact/SciREX 外部复现
3. 实现/跑 baseline（AutoSchemaKG/IncSchema 等开源情况待查）
4. 消融全表（step5 + step3/step4 组合）

## 诚实约束
- baseline PK 需同 LLM 受控（否则不可比）
- 综述 gold 标注规模小但可人工核验（专家闸门批处理）
- 公开数据集复现已部分成功（SciFact）
