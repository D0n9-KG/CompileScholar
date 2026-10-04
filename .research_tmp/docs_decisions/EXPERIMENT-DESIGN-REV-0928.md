# EXPERIMENT-DESIGN-REV-0928 — 实验设计修订版（用户终裁 2026-09-28 深夜）

> 取代 EXPERIMENT-DESIGN-WORLDSTATE-0926 的基准选型部分（叙事五要素/
> 世界状态框架/防走偏清单不变，仍为权威）。本档只记录基准组合与臂配置
> 的变更，避免后续遗忘。

## 最终基准组合（三基准，全部现成判分，零自造装置）

### 槽 1｜Multi-108（ScholarQA-Multi，闭卷固定文献）——✅ 已完成
- gold=人类专家真实综述答案（Nature 650:857-863 正刊血统），质量有保障
- 我们 F1 0.6503 双显著（vs LightRAG +0.004 CI 跨零=平；vs PaperQA2
  +0.136 CI[+0.071,+0.200] 显著）
- LightRAG/PaperQA2/memorized 基线全套在案
- **角色：固定文献受控形态（处理方式隔离）**

### 槽 3｜ScholarQA-CS2（开放文献 QA/证据报告）——🔄 主战场
- 臂配置（三范式全覆盖）：
  1. **ours**：热启动基库（综述骨干 2,100 篇）+broker 循环内检索+L1/L2
     深读（当前批 10-14 纵向 0.62→0.76，harness 干净口径 0.79）
  2. **harness 臂**：Claude Code+本地 27B+检索 MCP（matched-model DR
     agent 形态，dev20 答案在盘，同尺子判分 0.634/0.792）
  3. **PaperQA2 臂**（新增，用户裁定）：专用 RAG 管线形态——经典基线，
     Multi-108 管线改造（题目格式+pqac 引用桥接+Sciverse 开放检索），
     约 1 天工程。风险披露：Multi-108 时它 0.4784 偏弱，若 CS2 也弱
     需防"挑软柿子"质疑（应对：三范式对照的完整性论证+如实报数）
- **判分优化**（用户裁定方向，详见下节）：不再对齐官方 gemini judge
  口径——官方榜不比了，判分自由度释放，按成本重设计

### 槽 4｜DeepScholar-Bench（综述/related work 生成）——🆕 替换 ReportBench-ML
- 用户裁定：比 ReportBench-ML 好（gold 中位 23 引文/题 vs 137=现实
  可达；官方全自动判分三维度七指标；live 防污染；Berkeley Sky Lab
  出品学术公信力强；官方 baseline 表齐全：OpenAI DR 0.309 最强/
  STORM 0.073/OpenScholar 0.042 可直接引）
- 数据+判分代码：github.com/guestrin-lab/deepscholar-bench（CC BY 4.0）
- **待办**：拉数据核验+判分管线跑通（Nov-2025 版 200 题或先 63 题冒烟）
- 与我们能力的对齐点：verifiability 维度（引用精度/主张覆盖）=我们
  verbatim quote 纪律的强项；retrieval 质量维度=知识模型引导检索的
  发挥空间；live/月更=as_of 时间切片叙事的新战场

### 归档/降级
- **MDAQA：彻底归档**（用户裁定 09-28 深夜）——判分装置坑深（官方重叠
  指标对弱 gold 敏感+自造 judge 无时间预算）；直读臂 302 题+PaperQA
  176 题+HippoRAG2 部署方案全部留档不进论文
- **FieldState 自建 benchmark：低优先级**（用户裁定：时间不够，先保证
  现成 benchmark 跑出优势）
- ReportBench-ML：被 DeepScholar-Bench 取代

## CS2 判分优化方案（待批）

背景：不与官方榜对表后，判分只需臂间同尺子可比——优化空间全释放。

现行慢的解剖（单题 40-60 次 GLM 调用，30-60 分钟）：
1. ingredient（1 大调用）：题目+答案全文+13 条 criteria → GLM 思考
   开销 ~2.6k tok/次+schema 畸形重试最多 21 次
2. precision（1 大调用）：同类开销
3. citation（3-N 调用）：**sent_tokenize 逐段 LLM 分句（10-20 次调用/
   题）+propagate_citations 逐句传播+每 citation group 1 次**——调用
   数大头

优化候选（从低风险到高风险）：
- A（保守）：题间并行 4→8 路（GLM 并发上限试探）+重试上限 21→5
  （重试大多无救，5 次后记 None 补判）——预期 2-3x 提速，零语义风险
- B（中等）：citation 的 sent_tokenize 前置换**本地 NLTK 分句**（零
  LLM 调用），只保留 citation group 判分的 LLM 环节——语义近似（官方
  分句也是为了确定 citation 归属句；NLTK 分句+句内 citation 标记匹配
  可确定性完成同一目的）——预期再 2x 提速，需小样本验证与官方口径的
  一致性（同答案双跑对照）
- C（激进）：三 scorer 并行化（现在串行）——每题墙钟≈最慢 scorer
  而非三者之和，预期 2-3x 提速，零语义风险但并发压力×3
- 建议组合：A+C 立即做，B 做 10 题对照验证后决定

## 判分口径声明（论文写法）
- 所有 GLM-5.3 判分数字=臂间相对比较（同尺子同题同批），不对表官方
  gemini 榜（判分器不同，披露）
- 可选防御出口：答案对可打包 inspect eval 格式供 gemini 重判（如果
  审稿人要求官方口径）
- GLM 判崩率 10%（大答案 JSON 解析失败路径）——判分排雷规则预注册：
  全 facet 零分题重判一次，仍零分按真零计（不手工剔除）

## CS2 剩余执行清单（优先序）
1. 判分优化落地（A+C → 验证 → B 可选）
2. 批 14 补判收尾（ontology 题行尾 ref 修复后重判中）
3. 批 15（适配器三形态盲区+批 12/14 双样本均值的干净 A/B）
4. PaperQA2 臂改造+冒烟（~1 天）
5. dev100 全量轮（批 15 稳定后）
6. DeepScholar-Bench 拉数据+判分管线核验（与 CS2 并行推进）
