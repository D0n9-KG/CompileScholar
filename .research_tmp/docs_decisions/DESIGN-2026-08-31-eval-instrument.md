# DESIGN 2026-08-31：评测仪器（分层金字塔）

状态：设计定稿待实现（用户 2026-08-31 批准顺序：仪器 → 全身体检 → 迭代到停止线 → 判决实验）。
本文档是仪器的预注册设计：实现与验收都以此为准，改动须更新本文并注明原因。

## 0. 定位与动机

一句话：把"每轮迭代靠 LLM 逐条判边"换成"确定性层日常给出召回+精确率仪表盘，judge 只出现在有界位置"，同时补上**从未测过的抽取召回**。

动机（用户 2026-08-31 指出，成立）：
1. 现有 judge 交叉只度量精确率（对已抽出的边判对错），召回（漏了什么）没有分母——judge 方法本质上给不了召回。
2. judge 逐条判慢 → 迭代周期长 → 问题发现滞后，不利于快速迭代优化。
3. 判决实验（跨篇 in-loop，主终点 gold recall）与日常迭代需要的召回指标是**同一件基建**——一份工两处用。

设计原则：
- **确定性优先**：能用代码算的绝不用 LLM；LLM 只出现在有界位置（歧义匹配仲裁、里程碑抽样）。
- **gold 从构造上机器可匹配**：verbatim 锚 + 端点名 + pattern_type + role 映射，沿用 2026-08-25 gold 构造纪律。
- **仪器先自证再上岗**：单调性检验——人为退化系统，指标必须朝正确方向动，否则仪器失效。
- **预注册**：主终点/判读标准先落盘再跑，防事后挑指标（P0 多 seed 教训）。

## 1. 五层金字塔

| 层 | 回答什么问题 | LLM 用量 | 耗时 | 频率 |
|---|---|---|---|---|
| L0 确定性结构检查 | 结构纪律有没有破 | 零 | <1min | 每次 run |
| L1 金丝雀注入探针 | 该抽到的是否抽到（受控召回+忠实性） | 零（评测侧） | 分钟级 | 每轮迭代 |
| L2 探针篇 gold-lite | 真实显著事实的 P/R | 有界仲裁 | 匹配<5min，整体被抽取时间支配 | 每轮迭代 |
| L3 judge 交叉分层抽样 | extras 与变更边的语义正确性 | 抽样 50 条 | 小时级 | 里程碑 |
| L4 判决实验终点 | 跨篇 in-loop 收益 | — | 天级 | 只跑几次 |

### L0 确定性结构检查
- 指标：verbatim 率（边 evidence 是否原文严格子串）、端点锚定率、gate 拦截统计（按 gate 类型分：binding-locality / slot-discipline / role-legality / slot-type / competition）、每篇调用数/时长/成本。
- 数据源：`runs/kernel_v2/<RUN>/{kept,dropped}_edges,new_patterns,ledger,meta_snapshot}.json` + `data/graphs/<name>/kb.json`（超边带 provenance: paper_id/evidence/section/cited_from）。
- 实现锚点：`_deterministic_gate`（`src/granular_agent/extraction_agent.py:295`）已产出拦截记录，L0 只需聚合呈现，不重写检查逻辑。

### L1 金丝雀注入探针
- **正向金丝雀**：往真实论文文本注入合成事实句（方法演化边/参数边/概念边），每条即一条机器可匹配的 gold 边 → 受控召回率 = 抓到数/注入数。
- **负向金丝雀**：注入极性反转/偷换主体的陷阱句 → 不该照抽，测忠实性（误抽率）。
- 难度分层：显式单句 / 跨句隐式 / 带干扰项。
- **轮换制**：金丝雀池分份轮换，防抽取 prompt 对探针过拟合。
- 边界：金丝雀测的是"灵敏度"不是"真实召回"——真实分布由 L2 锚定，两层互补。

### L2 探针篇 gold-lite
- 3-5 篇固定论文（DQN 演替序列内选），每篇 30-60 条 gold，由 Claude 构造。
- gold 内部分两类，口径分开报：
  - **方法演化族 = 穷尽**：该篇支持的全部 lineage 边（outperforms_on/ablates/extends/cites 类，数量少可枚举）——这是判决实验主终点的直接前驱。
  - **其余显著事实 = 抽样**（重要概念/参数/成分边）。
- 指标命名诚实：**salient-recall（显著事实召回）**，不称全量召回。
- 匹配：确定性优先——verbatim 严格子串 + 端点名规范化（小写/复数/别名表）+ pattern_type 一致 + role 对齐；残余歧义 LLM 仲裁（有界，逐条记录）。
- precision 侧：匹配上 gold 的边记对；未匹配的 **extras 抽样送 L3**（gold 非穷尽，extras 不必然错——禁止把 extras 一律算错）。
- 匹配器歧义不记 miss：确定性匹配不上且仲裁未过才记漏，防止把匹配器缺陷算成系统召回差。

### L3 judge 交叉（分层抽样 + 差分）
- 50 条按 pattern_type × 论文分层；双 judge（judge ≠ 抽取模型）、8 路并发、快端点（gpt-oss-120b 前科提速 25×）。
- 只判两类：L2 extras 抽样 + **差分评测**（与上一版 bundle diff，只判新增/修改/消失的边——大多数迭代间不变的边不重判，这是迭代提速的关键）。
- 锚点：`.research_tmp/experiments/eval_judge/judge_cross.py` 已有。
- 口径前科：终裁实证 judge 过严 35%——L3 数字永远与终裁抽样对照报告，只作里程碑不作日常。

### L4 判决实验终点
- 两域 gold recall + frozen / evolving（/bookkeeping）臂对照、≥4 seed、paired bootstrap 95% CI。
- ML 域 gold = L2 gold-lite 同一工件扩到 10 篇（构造纪律不变）。

## 2. Schema 仪表盘（确定性，零 LLM）

| 指标 | 定义 | 数据源 |
|---|---|---|
| pattern 总数曲线 | 随篇数的增长轨迹（收敛/爆炸/衰退） | meta_snapshot 序列 |
| 跨篇复现率 | pattern 首现后被后续篇复用的比例 | ledger + pattern 首现篇 |
| 冗余度 | 表面重叠 + embedding 相似度超阈的 pattern 对占比（代理） | tbox.patterns |
| 槽位具体度 | role_slots 中 type=THING 的占比（已知弱点） | tbox.patterns |
| 采纳/复用计数 | 哪些边用了哪篇演化出的 pattern | ledger |

这五个数打出来，"自然演化 schema 质量"从感觉变成可见物。embedding 仅作相似度计算，不作判断。

## 3. gold-lite 构造纪律（沿用 2026-08-25 已定，重申）

1. Claude 构造，与抽取管线**完全隔离**：只读论文原文，不看任何抽取产物。
2. 每条过机器闸：verbatim 严格子串 + 端点名 + pattern_type + role 映射，规则可验证非判断。
3. 流程版本戳落盘；冻结前给用户"最可疑 20 条"做 10 分钟快审。
4. 探针篇候选：DQN(2015) / Double / Dueling / Prioritized / Rainbow 中选 3-5 篇（与判决实验语料重叠，一份数据两用）。实现第一步先核验 5 篇本地内容可用性（sci-evo API 内容目录）。

## 4. 仪器自身的验收标准（预注册）

- **A1 单调性检验**：对现有 bundle 人为退化（随机丢 30% 边 / 交换 role / 破坏 verbatim），各层指标必须相应单调变差。不动 = 仪器失效，禁止上岗。
- **A2 匹配一致率**：L2 确定性匹配 vs Claude 手工匹配，50 条试点一致率 ≥95%；不一致逐条归因（匹配器缺陷 / gold 缺陷分开记）。
- **A3 端到端时效**：5 篇重跑（论文级并行）后，评测面板（L0 + schema 仪表盘 + L2 匹配）≤5 分钟出全。
- **A4 金丝雀区分度**：现系统正向金丝雀召回不得为 0% 或 100%（否则无区分度，调整难度分层直至有区分度）。

验收报告落盘 docs_decisions/，通过后仪器才上岗。

## 5. 停止线（占位：体检后与用户定稿）

默认提案（待体检基线出来后校准绝对值）：
- L2 方法演化族 salient-recall ≥70% 且 L1 金丝雀召回 ≥80%；
- 达到后连续两轮迭代 L2 主指标移动 <2pt → 停止打磨，进判决实验。

体检的意义之一就是给停止线提供真实基线，避免拍脑袋定绝对值。

## 6. 实现顺序（每步带验收）

1. 核验 5 篇探针语料可用性 + L0 聚合面板（纯读 bundle/kb.json）→ 验收：对任一现有 bundle 一键出面板。
2. gold-lite 构造：先 2 篇试点 → A2 → 扩到 3-5 篇 → 用户 20 条快审后冻结。
3. L2 匹配器 + L1 金丝雀池 → 验收 A1/A3/A4。
4. L3 差分 judge 接线。
5. **全身体检**：现系统跑一轮全套 → 体检报告（含：抽取召回首个真实数字、自然演化 schema 首个仪表盘）→ 停止线定稿。

代码位置：`.research_tmp/experiments/eval_instrument/`（研究基建，不进 src/）。

## 7. 实施记录（2026-08-31 当日，滚动追加）

### 7.1 已落地组件（`.research_tmp/experiments/eval_instrument/`）
- `l0_panel.py` — L0 面板（verbatim 双档/端点锚定/gate 拦截直方图/pattern 分布/槽位类型一致性/calls 统计）。验收：REWRITE + A2G + ML_DQN 三种形态 bundle 一键出面板。
- `a1_degrade.py` — A1 单调性检验。**PASS**（REWRITE + A2G 双 bundle：D1 丢边→edges_kept 降、D2 坏证据→verbatim 降、D3 换 role→slot_type 一致性降）。
- `gold_lite_v1.json` — 3 篇 gold（DQN 25 边 + DoubleDQN 20 + Dueling 25 = 70 边，方法演化族穷尽+显著抽样）。
- `gold_gate.py` — gold 机器闸（G1 verbatim/G2 锚点/G3 pattern/G4 role）。**70/70 全过**。
- `l2_match.py` — L2 确定性匹配器（surface 三规则匹配/证据重叠含引用标记剥离/四级判定 full-partial-arbitration-missed/跨 pattern 分歧仲裁）。
- A2 验收：两 bundle × 25 gold 全量人工核对，**判定性结论（matched/missed）100% 一致；仲裁队列零噪声**（修掉单 token 包容与 token 子集两处过松规则后）。A2 **PASS**。

### 7.2 首批真实召回数字（历史 bundle，仅示诊断价值）
| bundle | 演化族 recall_full | 显著族 recall_full | 备注 |
|---|---|---|---|
| REWRITE (DQN) | 0.40 (6/15) | 0.20 (2/10) | 15-pattern 种子（含注入 outperforms_on/ablates） |
| MSEED_S1R0 (DQN) | 0.067 (1/15) | 0.00 | 旧栈，outperforms_on/ablates 零候选 |
| PPR_DD48410E18B3 (DoubleDQN) | 0.00 | 0.00 | **12-pattern 种子无 outperforms_on/ablates** |
| PPR_FC0F6B04EEBF (Dueling) | 0.067 (1/15) | 0.00 | 同上 |

### 7.3 重大发现：schema 种子混杂（比数字本身重要）
跨 run 召回对比被 **schema 种子不一致**混杂：REWRITE 15-pattern（含演化专用 pattern）vs 08-27 晚 ML 臂 12-pattern（不含）→ 演化族 recall 0.40 vs 0.0/0.067 的差异主要是**种子天花板**，不是抽取质量。
**设计修正（新增，视同预注册条款）**：
1. gold_gate 的 G3 以**固定参考 ML schema**（15-pattern 版，含 outperforms_on/ablates）为基准，不随 run 的 meta_snapshot 变。
2. 全身体检与判决实验的**所有臂（含 frozen）一律从同一参考种子起步**——frozen 臂的种子即其 pattern 发射上限，两臂同种子才是公平对照；"prior knowledge"基线的含义 = 参考种子。
3. 历史跨 run 数字只作诊断叙事，不进论文对比表。

### 7.4 首个 run 间方差信号
REWRITE vs MSEED 同论文同 gold：0.40 vs 0.067（混杂：栈版本+种子不同）。干净的同种子多 seed 方差量化留给全身体检。

### 7.5 L1 金丝雀组件（已落地，A4 验证随首轮抽取运行）
- `canary_pool.json` — 12 正向（Tier A 显式单句 / B 跨句 / C 干扰项）+ 6 负向（极性反转/假设句/移除组件/纯对比/未观察到效果/开放问题），全部使用**虚构实体**（QRN/SV-advantage/temporal-binned replay/calibration gap 等）→ surface 精确匹配、不与真实事实冲突。双轮换集 S1/S2 防过拟合。
- `canary_inject.py` — 在 MINERU_BASE 下建 `CANARY_*` 假 pid（清一色前缀、可随时删），按锚点把金丝雀句追加进宿主 block；走 `load_paper_blocks` 完全相同路径（零 src 改动）；自带自检（所有句子经管线加载器可见）。已验证 S1/S2 双集注入自检 OK。
- `canary_eval.py` — 确定性评估（复用 l2_match 的 surface/evidence 匹配）：正向按档召回率 + 负向陷阱触发率。
- `canary_run.py` — 抽取驱动，**固定参考种子**（§7.3 的 reference seed，base+outperforms_on/ablates）起步。

### 7.6 Schema 仪表盘（已落地，`schema_dashboard.py`）
输入=按时序排列的共享 KB per-paper bundle 序列，输出五指标。**颗粒流 A2G 10 篇序列首个读数**：
- pattern 曲线 15→32（末端 32→32 收敛）
- THING 占比 0.944→0.714（槽位随演化变具体——正面信号）
- 跨篇复现率 26/32=0.812
- 冗余度（token-Jaccard≥0.6 代理）0 对——代理粗糙，语义冗余可能藏在阈值下
- 采纳率 0.975（含种子 pattern；种子=先验知识计入"born earlier"合理，但解读须分开：most-reused 全是种子 pattern）
- born-but-never-reused 6 个（adapts/assesses_accuracy_of/substitutes_in/lacks_comparison/develops_during_flow/serves_as_building_block）= 演化碎屑，prune 治理的天然评测对象

**这是"自然演化 schema 质量"第一次有数字。**

### 7.7 L1 首轮 A4 判定（S1，2026-08-31）
S1 抽取运行：329s，71 边，16 patterns（=参考种子 14+2，接线正确）。canary_eval 三层口径：
- **事实级召回 1.0**（6/6 事实全捕获，surface 任意节点匹配、不受 role 命名约束）
- **精确 pattern 召回 0.833**（P03 被抽为 influences 而非 improves——绑定/证据全对，pattern 选择分歧。improves 与 influences 语义重叠是 schema 固有灰区）
- **陷阱率 0.0**（极性反转/假设句/移除组件三类陷阱全部正确未抽）

**A4 判定：PASS**（召回严格介于 0-1，有区分度）。L1 仪器上岗。
实现中发现并修复：pattern 分歧连带 role 命名分歧（improves from/to vs influences source/target）——事实级匹配必须 role 名无关，否则分层口径失效。

### 7.8 L1 双轮完整读数（S1+S2，2026-08-31 收官）

陷阱检测修正：只查精确 forbidden pattern 太窄（漏检 N06 开放问题被抽为 compares）——改为**任意 pattern + 实体对**触发。修正后双轮：

| 指标 | S1（全 Tier A） | S2（A/B/C 混合） |
|---|---|---|
| 事实级召回 | 1.0 | 0.667 |
| 精确 pattern 召回 | 0.833 | 0.0 |
| 陷阱率 | 0.333 | 0.667 |
| 边数 | 71 | 47 |

**S2 六个 miss 的诚实分解**：
- role-binding disagreement ×3（P07/P08/P10）：事实捕获、pattern 正确，但 surface 字面不同（P08/P10 抽取器把 "the same suite" 解析为 "49 games"——**抽取器比 gold 更对**，gold 的 surface 太字面）或 role 方向分歧（P07 defines 主客体反转，两可）
- pattern-disagreement ×1（P09）：句子 1 的 replaces 抽到、**句子 2 的 improves 真漏**——Tier B 跨句事实失效实锤
- true miss ×2（P11 条件参数事实 / P12 replaces）

**忠实性泄漏（新发现，此前不可见）**：假设句（N02 "might eventually surpass" 被抽为 extends）、开放问题（N06 "remains an open question" 被抽为 compares）、纯对比句（N04 "Unlike X, Y is not..." 被抽为 compares）三类泄漏；极性反转（N01）/移除组件（N03）/未观察到效果（N05）三类守住。**抽取器对"明确否定"鲁棒、对"非断言语境"（假设/疑问/纯对比）泄漏**——这是明确的迭代目标，也是负向金丝雀的存在价值证明。

**run 间方差**：同基线论文 71 vs 47 边——LLM 非确定性；金丝雀指标要稳定须多轮（全身体检按 3 轮取均值）。

A4 最终判定：**PASS**（双层召回有区分度 + 陷阱检测能抓真实泄漏）。

### 7.9 gold 全集完成 + 五篇快照（2026-08-31 收官补）
- **5/5 篇 gold 完成：120 边全过机器闸**（DQN 25 / DoubleDQN 20 / Dueling 25 / PER 25 / Rainbow 25；每篇方法演化族 13-15 条 + 显著族 7-10 条）。PER/Rainbow 构造时机器闸各抓出 6/11 条 anchor 缺陷并修正——闸本身在发挥作用（anchor 必须在证据内这条纪律靠它强制执行）。
- 五篇历史 bundle 的 L2 快照（**均为 12-pattern 旧种子 run，除 REWRITE 外演化 pattern 不在种子内——数字反映的是种子天花板不是抽取质量**）：

| 论文 | 演化族 full | 显著族 full | 仲裁数 |
|---|---|---|---|
| DQN (REWRITE, 15-pattern 种子) | 0.40 | 0.20 | 5 |
| DoubleDQN | 0.00 | 0.00 | 8 |
| Dueling | 0.067 | 0.00 | 11 |
| PER | 0.067 | 0.00 | 8 |
| Rainbow | 0.067 | 0.133 | 3 |

- **含金量最高的观察**：仲裁队列被填满（3-11 条/篇）——pattern 分歧/role 分歧是系统性失效模式，正是 L3 差分 judge 的目标工作面；仲裁不解决，"召回"数字系统性低估真值。
- 下一步 = 全身体检：五篇以**参考种子**重抽（3 轮取均值）+ 金丝雀 + 仪表盘全套。

### 7.10 L3 落地 + 全身体检启动（2026-08-31 晚收官）
- `l3_arbitrate.py` 已实现并冒烟通过（REWRITE bundle：5 仲裁 + 12 extras 抽样，双 judge GLM-5-Turbo/gpt-oss-120b）：
  - **仲裁折算后 recall 上修**：演化族 0.40→0.467、显著族 0.20→0.30（匹配器卡住的真捕获被 judge 放行——分层口径按设计工作）
  - **extras precision 首个信号**：12 条分层抽样 strict 4 / lenient 8——与 known ~50-58% 精确率口径自洽，judge 过严前科（35%）下的真实值应在两者之间
  - 双 judge 一致率问题留待全身体检的分歧样本分析
- `healthcheck_run.py` 启动：五篇 × 3 轮，全部从**冻结参考种子**起步（§7.3 修正条款的第一次执行）。每轮 fresh agent（无演化残留），bundle 落 `HC_R{r}_{name}`。
- `healthcheck_report.py`（L0+L2 跨轮聚合，mean±range）+ `gold_review20.py`（快审材料生成器）就绪。
- **review20.md 已生成**：top-20 可疑 gold 按五启发式排序（paraphrased-anchor 16 / gray-pattern 20 / constructor-flagged 8 / medium-confidence 14 / low-confidence 2 / ocr-damaged 1）——gold 冻结前的用户快审输入就位。

## 8. 风险与开放问题

- **金丝雀 ≠ 真实文本分布**：只作受控灵敏度；真实性由 L2 锚定。两层结论方向不一致时以 L2 为准并记录。
- **探针篇过拟合**：固定 3-5 篇会被 prompt 迭代隐性调参污染 → 金丝雀轮换 + 里程碑时用非探针篇 spot-check。
- **同名异义/别名匹配歧义**：仲裁层必须有，且仲裁记录留痕可审计。
- **extras 的 precision 口径**：gold 非穷尽，extras 抽样 judge 是唯一诚实口径，成本须受 A3 约束。
- judge 过严前科（35%）：所有 L3 结论必须附终裁抽样对照。
