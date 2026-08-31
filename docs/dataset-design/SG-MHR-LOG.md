# SG-MHR 实现日志

> 目标：实现 Schema-Guided Multi-Hop Hypergraph Retrieval，验证能否区分"schema/抽取器问题" vs "下游策略太粗糙"。在 ResearchQA multi_hop 上对比 C1/C2/naive-RAG/bare-LLM。

## 起点资产核查（2026-08-14）

- 缓存超图：4 篇（W3086667591/W4251560328/W4385245566/W4292779060），共 8 个 multi_hop 题，每题跨 2 section。
- 超图结构：edges=[{pat, nodes:[[surface,role]], ev, quals}], pat_edges={pat:[idx]}, schema_topo=[{rel,src,tgt}], sections, raw_text。
- pattern 词汇（实测 6 种）：defines / composed_of / influences / measures / claim_relation / constitutive_law。
- role 词汇：defines→subject/object；composed_of→whole/component；influences→source/target；claim_relation→from/to/parameter；measures→object/instrument；constitutive_law→input/output/parameter。
- **发现：缓存论文 schema_topo 只有 depends_on + composes，没有 constrains**（4 篇全无 constrains 边）。Phase 2 路由只能用 dep/compose 两类语义边——这是约束，记下。
- C1/C2 旧结果（phase1b.json, 8 题）：C1/C2 在多数题上 section_coverage 相等，净效应微弱（与背景一致）。

## P0 设计（Phase 1 需求分解 + Phase 3 实体中心检索 + gate）

实现文件：`.research_tmp/sgmhr.py`

- Phase 1 Demand Decomposition：LLM 拆 query 为 2-4 个 demand，每个带 demand_text / pattern_hints（从 6 种 pattern 选）/ role_hint / entity_hints（query 中的关键实体）。
- Phase 3 Entity-Centric Retrieval：
  1. 实体 linking：entity_hints 与边 node surface 做 token-overlap 匹配
  2. pattern 过滤：优先保留 pat ∈ demand.pattern_hints 的边（schema 路由参与决策）
  3. query-aware gate：cos(demand_emb, edge_emb) ≥ thr
  4. 实体 join key：沿共享实体扩展到相邻超边
  5. 不分解 triple，保留 n-ary 超边整体
- 输出：每个 demand 的检索超边 + 是否覆盖 expected section（内容审视，非只看数字）。

## P0 首轮结果（8 题，sgmhr_run.log）

**混淆变量发现（剔除 W4251560328 两题）**：该 Handbook 的 raw_text 不含 "Chapter 18"——section 解析把全部章节合并成 "Methods"，TOC/Contributors 内容从未进入超图。这是**上游 PDF 解析问题**，非策略也非抽取器。该篇 2 题 5 arm 全 0，策略对比中应剔除。

**6 个有效题的初步观察**：
- Q1 W3086667591：A=0.50 B=0.50 C1=1.00 C2=0.50 **SG=1.00**（与 C1 持平、超 C2），两 ref COVERED。答案 final=29/initial=64 来自真实边（"only 29 (45%) mentioned..."），非幻觉。
- Q2 W3086667591：A=0.00 B=0.50 C1=0.50 C2=0.50 SG=0.50（持平），METHODS MISS。
- 实体匹配退化已确认：单 token 实体（"devices"/"chapter"）em=1.0 泛滥，pattern 过滤被绕过（D1 hint=measures 却全是 composed_of 边）。

**待调优**：entity match 对单 token 实体过松；pattern_hints 在实体强匹配时未真正参与过滤。

## P0 首轮完整结论（8 题 → 剔除 2 混淆题后 6 有效题）

**Aggregate（6 有效题，剔除 W4251560328 PDF 解析失败）**：

| arm | sect_cov | cite_acc | cite_prec |
|-----|----------|----------|-----------|
| A bare-LLM | 0.083 | 0.667 | 0.083 |
| B naive-RAG | **0.417** | 0.833 | **0.472** |
| C1 边相似度 | 0.333 | 1.000 | 0.361 |
| C2 +拓扑扩展 | 0.333 | 0.833 | 0.333 |
| **SG-MHR** | 0.333 | **1.000** | 0.306 |

**核心诊断结论（回答目标的问题：schema/抽取器问题 vs 策略太粗糙）**：

1. **SG-MHR 没在 section_coverage 上胜过 C1/C2（均 0.333）** — 策略升级未带来 coverage 提升。
2. **但 SG cite_accuracy=1.000（与 C1 并列最高，超 B/C2 的 0.833）** — verbatim grounding 工作，几乎零幻觉。策略在"防幻觉"上有效。
3. **naive-RAG(B) section_coverage=0.417 > 所有超图臂(0.333)** — 关键信号：原始 chunk 比抽取的超图边覆盖更多 expected section。说明**抽取器漏抽**了 expected reference 所在的内容。

**内容审视验证（W4385245566 Q5，深挖校正）**：
- 初判误读：ref"1.1 Contributions" alt 的"长版本 in raw_text=False"——实因 alt 是含尾部"in Section 6.6, i.e..."的长句，尾部不在原文；其核心 "Our strongest single AI/TP method now proves in 120s 60%..." **in raw_text=True（verbatim）**。**不是 paraphrase bug**。
- 校正后真因：SG 对 Q5 生成的 citation = 检索到的边的 verbatim（"outperforms by 58.2%" / "56.1% on Mizar40"），是诚实的——但这些边**不是 expected ref 所指的那条**。expected ref 指的"60% in 120s bushy"和"56.4% in 30s bushy"两条关键论断，**原文有但没有任何超图边的 ev 含它们**——**抽取器漏抽**。
- 即：SG-MHR 检索策略正确（找相关边、verbatim grounding 正常、零幻觉），但超图里压根没存 expected 论断，策略再好也检索不到。naive-RAG(B) 更高正是因它直接用原始 chunk，不依赖抽取器抽全。

**诊断定论（校正版）**：
- **当前瓶颈是抽取器覆盖率，不是检索策略**。非 paraphrase bug（edges 是 verbatim 的），非生成端问题（SG citation 是检索边 verbatim、诚实）。
- SG-MHR 已能清晰区分"策略问题 vs 抽取器问题"——答案是抽取器漏抽关键论断。cite_acc=1.0 + retrieval 找得到相关边 = 策略 OK；coverage 卡住 = 超图内容不全。
- naive-RAG(B) sect_cov 0.417 > 超图臂 0.333，是因为 chunk 不经抽取器、信息无损，覆盖到漏抽内容。这本身就是抽取器有信息丢失的旁证。

**漏抽量化（6 有效题，13 个 expected ref-alt）**：
- 抽到 9 / **原文有但漏抽 4** / 原文也没有 0。
- **31% 的 expected 论断原文有但抽取器漏抽，0 个原文缺失**（100% 抽取器责任，非数据问题）。
- 漏抽 4 条全是数值结果类关键论断："56.4% bushy" / "1000 hard theorems" / "512 segments" / spatial join 定义。
- 与 schema 设计呼应：`measures` pattern（数值结果）在缓存论文出现极少（W3086667591=1, W4385245566=1 条），抽取器把数值结果大量误分到 composed_of/influences，关键数值论断没被独立抽出。

**下一步方向（P1）**：
- 不再调检索策略超参（瓶颈不在那），转向修复抽取器：(1) 提高 measures/数值结果 pattern 的抽取覆盖（关键数值论断独立成边）(2) 验证修复后 SG-MHR 是否反超 naive-RAG(B)。
- 兜底方案：SG-MHR + raw chunk 混合（hybrid retrieval），用 raw chunk 补抽取器漏抽内容，看是否反超纯 naive-RAG。
- 实体匹配单 token 退化已修（sgmhr.py 改为单 token 弱信号），下一轮验证。

## 当前状态（诚实）

SG-MHR P0 完成并诚实报告：
- 实现跑通（.research_tmp/sgmhr.py，8 题 end-to-end，代码非空跑）。
- 核心目标"区分 schema/抽取器问题 vs 策略太粗糙"**已达成**：答案=抽取器漏抽（31% expected 论断原文有没成边，0 个原文缺失）。
- SG-MHR 未在 section_coverage 上超过 C1/C2（均 0.333），但 cite_accuracy=1.000（零幻觉，超 B/C2）。
- 创新定位"精细化创新"暂未证明 capability lift——因瓶颈在抽取器上游，策略的精细化优势被超图内容不全掩盖。需修复抽取器后重新验证。

## 抽取器漏抽 bug 根因定位（2026-08-14）

不是 LLM 抽取质量差，是 **PDF 表格解析 + section 级抽取覆盖不均**：
- W4385245566 有 3 个 section 边数=0（Methods@28580/36204/43617），漏抽的"56.4% bushy"/"1000 hard theorems"全在这几个里。
- pymupdf 把表格抽成短数字行碎片，section 文本变表格垃圾 → LLM 抽不出结构化边 → 夹在表格里的关键结果句一起丢。
- 关键证据：56.4 那行是完整散文（PROSE），只是被夹在时间线表格 section 里。
- naive-RAG(B) 不经抽取器反而更高，正是因为它用原始 chunk 不丢这些。

## 关键验证：dep/con/comp 三类边全部触发（2026-08-14）

用本机 MinerU 跑颗粒论文 PPR_017AD90905AA（Scientific Reports，含本构定律+公式），MinerU content_list → 适配器分段 → build_hypergraph_from_sections 抽取。

**结果：49 边，6 schema_topo，三类边全触发**：
- depends_on 3（influences→defines）
- **constrains 2**（constitutive_law→influences, constitutive_law→measures）★ con 首次触发
- composes 1（composed_of→measures）
- patterns: composed_of 11 / influences 22 / measures 3 / defines 8 / constitutive_law 5

**意义**：
1. 之前 W4385245566 没有 constitutive_law 所以 con 触发不了、差异化只剩 dep+comp。这次 MinerU 保留 equation 块进 prompt → LLM 抽出 constitutive_law → con 推断触发。
2. con 触发证明"constitutive_law 约束 measures"语义成立——差异化不再空。
3. **Intern-Atlas 式综述 gold 思路能测全我们三类边**，前提是 gold 论文含本构定律类内容 → 颗粒流域（2355 语料）天然适合。
4. 评测域定为颗粒流（和 memory [paper-goal-constraints] 策略 + 专家闸门对齐）。

**遗留**：本机 MinerU 慢（~133s/篇），等远端 MinerU 恢复后批量。con 触发率仍低（2 条），可考虑增强 constitutive_law 抽取 prompt。

**修复1：_filter_prose（researchqa_phase0.py）** — 保留散文行删表格碎片。验证：正常 section 保留 75-94%，56.4 存活。但重建后 56.4 仍 0 边——表格碎片是表层，深层是 LLM 在 block5 没抽这句成边（prompt 覆盖问题，待继续）。

**修复2：call_llm 加超时重试+退避+模型备选（src/granular_agent/llm_client.py）** — 用户指出 deepseek 卡死应换模型/重试/退避而非靠运气。已改：单次 timeout 45s，3 次指数退避（2s/6s），全失败 fallback Paratera GLM-5-Turbo。验证 0.6s 正常返回。备选模型：GLM-5-Turbo/Kimi-K2.6/DeepSeek-R1/Qwen3。

**修复3：MinerU 替换 pymupdf（进行中）** — 用户指出该用 MinerU 处理 PDF（表格结构化是它的强项）。MinerU 3.1.11 装在 science_evo/.venv-mineru，模型已缓存（PDF-Extract-Kit-1.0，1.1G）。
- 障碍：MinerU 要访问 huggingface.co 校验模型 revision，但 HF 直连被墙（SSL 失败）。
- 解决：设 HF_HUB_OFFLINE=1 用本地缓存。
- 2,355 篇语料已跑过 MinerU（remote_mineru 目录是结果落地，不是可调服务——核实后确认无远端服务入口）。
- ResearchQA 的 PDF（W4385245566 等）不在其中，单独跑 MinerU，验证表格不再成碎片、56.4 是否被抽到。

**注**：用户指出之前推进太快（P0 简版→跳找新 benchmark），纠正方向为"先修方法本身再谈测评"。同时调研 GraphRAG 系 SOTA 用的 benchmark（agent 后台跑）。
