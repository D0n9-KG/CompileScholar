# DECISION: B+ 重建（抽取栈重写 + KB 内核保留简化 + 单管线 + legacy 物理封存）

日期：2026-08-25
依据：六路全面审查最终判决（AUDIT_2026-08-25_glm53_full_review.md）+ 用户愿景对齐 session 全部拍板（memory: audit-2026-08-25-verdict / final-vision-and-crossdomain-schema / arfm-gold-audit-rebuild-principles）。
本文档是 B+ 重建期的**最高执行参照**。铁律：**改核心设计前必须先 commit + 写 DECISION**——本文档即该铁律的第一次执行。

---

## 0. 一句话判决

方向不调（n-ary 超图 + schema 并进演化 + 科学论文，四元组合真空已终检确认），但 kernel_v2 全部现存 bundle 数字作废（单块 mock 输入 + 同篇抽 5 遍 + 修复前代码状态三重污染）；三 bug 因果链（斜杠 prompt → 复合 pattern_type → role gate 整边丢弃）同时杀死两个核心卖点的旗舰证据。判决：**抽取栈推倒重写，KB 内核保留并简化，砍掉全部设计断点死代码，旧管线物理封存，此后单管线**。

## 1. B+ 范围（做什么/不做什么）

**做**：
1. 采纳探针（2-3 天，判决性实验前置闸门）：手工注入 ML 域 gold pattern（`outperforms_on` / `ablates`）+ [NEW] 强标记 + usage 统计，三分支判读定 in-loop 卖点生死。探针结果直接决定 schema 卖点走向，如实报告不粉饰——两个分支是坏消息分支，坏消息照报。
2. 确定性修复（1-2 天）：真实 MinerU 块输入、确定性按标题切 section（替代 LLM 切块）、单 writer、ablation arm 真开关、model provenance、seed 参数化。
3. 抽取栈重写：联合抽取（按句/短段，绑定局部发生）+ 语义 verifier（逐边对 evidence span、方向/极性显式校验）。验收（开发期内部指标，非论文评测指标）：judge 交叉（GLM+qwen，judge≠抽取模型）语义正确率 43%→**≥70%** + 确定性子维度（成员合法性/verbatim/可定位性）+ 固定样本逐条对照原文。
4. 跨篇判决性实验（两域：ML DQN→Double→Dueling→Prioritized→Rainbow 演替序列 + 颗粒流 ARFM 锚点）：共享 KB 串行演化臂 vs frozen 对照臂。
5. gold 重建（排在抽取栈重写之后，新 pattern 体系定型再建，避免返工）：按 arfm-gold-audit-rebuild-principles 四道闸 + 三件套。
6. 下游评测（靠后）：IG-Bench 最对口 + ScholarQABench 类；consumer agent 论文必须（至少一个能跑 benchmark 的下游检索/推理 agent）。

**不做**：QA/综述/冲突三个 consumer、HITL 闭环、衰减 reaper、LangGraph 完善度、Skill Library、ML_DQN 单篇深挖（debug 使命完成）、第三域跨篇实验、人工标注测试集（用户定调：不做）。

**时间窗**：主投 WWW 2027（10 月截稿），保底 CIKM 2027，冲高 ACL 2027。IG-Bench 占位速度每月 1-2 个邻接工作，6 个月硬时间窗。

## 2. 保留 - 推倒 - 砍（模块级清单）

**保留（原地不动）**：
| 模块 | 理由 |
|---|---|
| `knowledge_base.py` | 内核 Mutation/两阶段 commit/ledger，6 套测试全过，论文机制贡献本身 |
| `hypergraph_schema.py` | T-box + seed；Pattern 结构设计 sound（微调：加 domain + 分层 namespace 字段） |
| `concept_graph.py` | A-box + 概念合并；结构设计 sound（微调：definition/central） |
| `llm_client.py` | 加 per-call model provenance 后继续用 |
| 全部评测资产 | P0 脚本 / gold / baseline PK / 7 维评测 / 逐条语义审脚本（升格正式指标） |

**推倒重写**：
| 对象 | 重写为 |
|---|---|
| `structure_mapper.py` 的 LLM 切块 | 确定性按标题切 section + 退化分节守卫 |
| 三步分解抽取（step1 实体→step2 关系→step3 绑边） | 联合抽取（单 prompt 按句/短段出完整边，绑定上下文不丢失，方向/极性 prompt 显式约束） |
| 双 writer（process_paper_via_kernel 内置 writer + extraction_agent writer 并存） | 单 writer |
| 语义校验（规则 gate 只查 role∈声明集 + verifier 截断默认 keep） | 语义 verifier：逐边对 evidence span，方向/极性/成员确定性检查 |
| metadata 管道（paper_registry） | 重接 sci-evo-extract；先解决前科 PPR_xxx 0 重叠再依赖 |

**砍（设计断点死代码，不封存直接删）**：CAS、统一 propose queue、Skill 层、边级 5-outcome route、replay/time-travel、`process_batch_via_kernel` 死出口（prune 唯一 caller，全仓库零调用——注意：砍的是死代码本身，prune 机制在跨篇实验中重接到真 caller）。

**Hyperedge 微调**：provenance + paper_id/DOI 锚（结构 sound，只加锚字段）。

## 3. legacy 封存方案（物理隔离）

- 旧管线路径（文件级清单由依赖地图确定，见封存执行 commit）**物理移入 `legacy/` 目录**，不留在 src/ 里靠自觉。
- kernel_v2 现存 bundle（.research_tmp/runs/kernel_v2/）标注 **"审计作废，仅供诊断"**（加 INVALIDATED.md 说明三重污染原因：单块 mock 输入 / 同篇抽 5 遍 / 部分跑在修复前代码上）。不删除——逐条语义审的诊断价值保留。
- 根目录 `_tmp_*.py`（128 个未跟踪文件中的脚本部分）清理：有留存价值的移入 .research_tmp，无价值的删除；`_tmp_*.html` 抓取缓存全删。
- 封存动作本身单独 commit，commit message 列出移动清单。

## 4. 单管线原则

- **主代码不 import legacy**。`src/granular_agent/` 里只允许存在一条抽取→演化→维护管线（kernel 路径）。
- 旧管线封存后如需受控对比（重写后新 vs 旧一次公平对比），从 legacy/ 显式引用，用完即弃，对比结果写进 .research_tmp。
- 严禁新旧混合：新代码不调用 legacy 模块，legacy 模块不改（发现 bug 记录不修）。

## 5. 跨域 schema：分层 namespace + 显式升格（用户拍板"可以走"）

```
meta:global（跨域通用 pattern）
  ├── meta:ml / meta:granular / meta:molecular / ...（域专属）
```

- 输入语料可跨领域，每篇带 domain 标签。
- **演化默认域内**：同域 distinctness/merge/prune，互不污染。
- **跨域升格 = 显式治理操作**：pattern 被 ≥2 域独立复用且语义一致才升 global（LLM 判 + 可人工闸）。跨域结构是挣出来的不是混出来的。
- 跨域下游（知识发现/范式借鉴）在 global 层 + 跨域概念对齐层做，不要求 pattern 全跨域通用。
- 附赠机制层卖点：pattern 生命周期（域内诞生→域内复用→跨域升格）是 Intern-Atlas/DIAL-KG 都没有的结构。
- 落地纪律：两域判决性实验免费观察"有没有 pattern 自然满足升格条件"——有就写，没有不硬造。

## 6. metadata 一等公民 + sci-evo-extract 管道

- 每篇论文 DOI/标题/作者可查（下游基础操作）；Hyperedge 加 paper_id/DOI 锚。
- 获取管道利用本地 **sci-evo-extract** 项目（多来源取论文+信息）。
- ⚠️ 前科：paper_registry 上次接 sci-evo 两边 PPR_xxx **0 重叠**（memory paper-registry-client-step1-done）。重接时**先解决 0 重叠问题再依赖**该管道，未验证前 metadata 获取走独立通道。

## 7. 效率目标（一等设计目标，与效果/成本并列）

- **单篇抽取 < 5min**（基线 7.4min）。路径：联合抽取省 2/3 调用 + chunk 并发 4→8/16 + verifier 批量 + schema prompt 检索式（非全量塞）。
- **硬约束：跨篇共享 KB 演化臂必须串行**（第 N 篇要看到前 N-1 篇的 schema，这是"并进"的定义）。论文级并行只适用于 frozen/static 对照臂和不同 seed 之间。
- 效率进机制层指标：单篇时间 / 每边调用数 / 每边成本**三项都报**。
- Provider：Paratera + DeepSeek-V4-Flash（thinking 关）。论文报 API 调用数/费用（卖点：尽可能少的调用/费用实现尽可能好的效果）。

## 8. 卖点定位（重锚后）

- 主故事：以 b 的证据结构讲 a 的故事。下限（已验证方向，数字待冻结 gold 重算）：A1 n-ary vs binary + A3 富拓扑直读 + SCION/Hyper-KGGen PK。上限（唯一赌注）：A2 跨篇 in-loop 演化判决性实验。
- schema 演化定位 = "首次严格验证 DIAL-KG 声明但未验证的 in-loop 收益"（正面引 DIAL-KG；其 ablation 无 w/o-schema-evolution 臂）。
- mutable hypergraph memory 降级为架构描述（正面引 EvoGraph-R1/Co-E，不与争概念）。
- 方法演化关系一等公民卖点待采纳探针判决——探针是判决性实验的前置闸门。
- 旧评测数字（ERR_cond=0.667 / fair_recall=0.833 / SCION PK / P0 多 seed 表）全部待冻结版 gold 重算，此前不进论文。

## 9. 执行序（唯一优先级）

1. 本文 DECISION commit（铁律第一次执行）
2. legacy 物理封存 + _tmp 清理 + bundle 作废标注（单独 commit）
3. **采纳探针**（2-3 天）：三分支判读——管道可破→修采纳机制 / pattern 质量问题→转 governance 故事 / LLM 能力不够→砍 in-loop 卖点
4. 确定性修复（1-2 天）
5. 抽取栈重写（验收=judge 交叉≥70% + 固定样本逐条对照原文）
6. gold 重建（新 pattern 体系定型后）
7. 跨篇两域判决性实验 → A1/A2/A3 × 4 seed → baseline 扩 3 个 → 下游 benchmark → 写作

---

*本文档取代 DECISION-step7-kernel-pipeline.md 的"两条管线并存"决策——并存使命（ablation 对照）已完成，双管线时代结束。*
