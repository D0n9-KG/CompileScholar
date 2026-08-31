# 采纳探针设计（PROBE: pattern adoption）+ 接线修复决策

日期：2026-08-25。依据：DECISION-2026-08-25-b-plus-rebuild 执行序第 3 步（采纳探针=判决性实验前置闸门）。
探针问题：**schema 演化的 pattern 到底能不能被抽取管线采纳（"并进"闭环是否活）？** 三分支判读定 in-loop 卖点生死，坏消息照报。

---

## 0. 探针前发现：两个确定性接线 bug（修正审计因果故事）

准备探针时逐行读码发现，比审计"planner 惯性"的解释更硬的根因：

### Bug W1（致命）：schema_prompt 在 multistep 核心内被静默丢弃
- `ExtractionAgent.execute()`（extraction_agent.py:244）正确构建 schema_prompt = `_retrieved_schema_prompt(tbox)` + planner relation_outline，传给 `_execute_core` → `_run_hg_node_multistep(schema_prompt=...)`
- **`_run_hg_node_multistep`（hypergraph_extractor.py:657）收到 schema_prompt 后函数体内从未使用**——step1/step2/step3 三个 prompt 都不含它。`_STEP3_PROMPT`（边定型调用）的模式清单是**硬编码**的（"constitutive_law, influences, ..., extends/improves/compares"），与 T-box 无关。
- 老版单调用 `_run_hg_node` 是用 schema_prompt 的（chunk_schema → 抽取 prompt）——multistep 性能重构（commit 22d7281 时期）引入回归。
- **推论：0/215 采纳的直接根因是接线断裂**。演化新 pattern 进 T-box 后，边定型 LLM 调用根本看不见它。planner 的 relation_outline（schema-in-context 设计）也随 schema_prompt 一起被丢弃——planner 看得见 schema、executor 看不见，计划与执行脱节。
- verifier 是看见 schema 的（`verify()` 用 `tbox.to_prompt()` 全量渲染 + per-edge allowed_roles）——审的人有图、干活的人没图。

### Bug W2：executor 模型硬编码绕过定案 provider
- `_execute_core`（extraction_agent.py:298）硬编码 `llm="deepseek"` → `_call()` 路由到官方 `call_llm(model="deepseek-chat")`。
- 设计与定案 provider（Paratera DeepSeek-V4-Flash）不符：现状 planner 用 V4-Flash、真正抽边的 executor 用 deepseek-chat。也是审计"model provenance"点的实锤。

### 与审计因果链的关系
审计的三 bug 链（斜杠 prompt → 复合 pattern_type → role gate 丢弃）**仍然成立**，但它在 W1 下游：即使 LLM 想用 schema pattern，它也看不见 schema，只能照硬编码清单（含斜杠字面量）输出。W1 修复后斜杠 bug 的影响面会在探针中直接显现。

---

## 1. 修复决策（铁律：先文档后改码）

**修 W1 + W2（最小 surgical，恢复设计行为非改设计）**：
1. `_STEP3_PROMPT` 加 `{schema_prompt}` 段（labeled entities 之前），并调整 role 指令：schema 中 pattern 声明的 role_slots 优先于硬编码清单。
2. `_run_hg_node_multistep` 的 step3 format 传 schema_prompt。影响所有调用者（kernel 路径 + eval scifact/scirex + legacy 封存路径不跑）——eval 行为变化是**正确方向**（旧数字已判作废待重挣）。
3. `_execute_core` 的 llm 由 ExtractionAgent 新参数 `executor_model`（默认 "deepseek" 保持兼容）传入；agent.py 构造时传 `self.llms[0]`。

**不修（留给确定性修复/重写阶段）**：`_locate_chapter_text` 退化路径、切块确定性、单 writer。探针只修到"schema 可见"这一层，保持归因干净。**一个例外**：重写 pattern_type 指引为"schema 优先"时，原句"extends/improves/compares"斜杠字面量保留会与新指引自相矛盾（一边教优先 schema、一边教复合字面量），故顺带加了一句"these are SEPARATE pattern names, never write them joined with slashes"消歧——这是一句话的一致性修补，不是完整斜杠清理（qualifier 清单等处的斜杠残留留给确定性修复阶段）。

**探针臂设置**：
- ~~Arm A（bug 复现臂：现状代码+注入）~~——**不跑**。代码级证据（上文 W1）+ 6 个现存 bundle（0/215）已足够证明断线态不可采纳，烧一次 run 无增量信息。
- **Arm B（主臂）**：W1+W2 修复后 + 注入 2 个手工 gold pattern + 真实 MinerU 块。测：可见的 gold pattern 是否被用、用得对不对。
- **Arm C（对照臂）**：W1+W2 修复后、不注入。测 base rate：同样的论文事实在无注入 pattern 时去了哪（不存在/被丢/以其他 type 存在）——区分"召回增益"与"重标注"。

## 2. Gold pattern 注入规格（手工构造，排除 pattern 质量干扰）

ML 域两个 pattern（用户指认候选），构造时给足定义/boundary/roles：

```python
outperforms_on:  # family: claim (评测比较语义) — A 方法在任务 T 上以指标 M 胜过 B
  description: "method A outperforms method B on task/benchmark T, measured by metric M"
  semantic_boundary: "quantitative comparison result between two methods on a shared
    task with a metric; NOT a general claim_relation (which is discourse-level
    supports/contrasts), NOT extends/improves (which is building-on, no numbers)"
  role_slots: [{role: winner, type: METHOD}, {role: loser, type: METHOD},
               {role: task, type: TASK_OR_BENCHMARK}, {role: metric, type: PROPERTY}]
  allowed_qualifiers: [method, evidence_strength, cited_from]

ablates:  # family: claim (消融语义) — 论文消融方法 M 的组件 C 得到效果 E
  description: "paper ablates component C of method M, observing effect E on the result"
  semantic_boundary: "an explicit ablation study statement (removing/disabling a
    component and measuring the consequence); NOT composed_of (structural parts),
    NOT influences (functional dependence in a law)"
  role_slots: [{role: method, type: METHOD}, {role: component, type: METHOD},
               {role: effect, type: PROPERTY}]
  allowed_qualifiers: [method, evidence_strength, cited_from]
```

- **[NEW] 强标记**：注入 pattern 的 description 前缀 `[NEW]`（to_prompt 渲染会带出来），强化可见性——探针要测的是"能不能采纳"，先排除"没注意到"。
- 注入方式：构造 `meta_hg = seed_meta_hypergraph_general()` + `add_pattern(...)` 两个 pattern → `agent.meta_hg = meta_hg`（探针脚本内，不改主码）。
- DQN 论文里两类事实都真实大量存在（49 个 Atari 游戏上的对比表、消融实验 section），有足够抽取素材。

## 3. 输入与运行

- **真实 MinerU 块**：paper_id = `PPR_24493BE6E8C2`（DQN Nature 2015，143 块/74 text），直接跑（load_paper_blocks 从 MINERU_BASE 读），**零 monkey-patch**（对审计前科的正面回应）。
- domain='ml'，arm='full'，DeepSeek-V4-Flash（Paratera，关 thinking）。
- 每臂 1 run（探针是方向判决不是统计断言；方差问题留给 4-seed 阶段）。

## 4. 测量（usage 直方图 + 语义逐条审）

1. **usage 计数**：kept_edges + dropped_edges 的 pattern_type 直方图；`outperforms_on`/`ablates` 的边数（B 臂 vs C 臂）。
2. **语义质量**（B 臂注入 pattern 的每条边 + C 臂同位置事实）：
   - 我（Claude 子代理）逐条对照原文审：方向对不对（winner/loser 不反）、绑定对不对（节点真是那个方法/任务/指标）、evidence 是否支撑
   - LLM judge 交叉（judge≠抽取模型）
3. **planner 联动**：plan 的 expected_patterns/relation_outline 是否含注入 pattern（planner 一直看得见 schema——它有没有选？）。
4. **副作用**：修复后全 pattern_type 分布 vs 修复前 bundle 的分布（schema 可见是否改变整体行为——比如 claim_relation 泛滥是否缓解）。

## 5. 三分支持判读（预注册，不许事后挪门柱）

| 结果 | 判读 | 行动 |
|---|---|---|
| B 臂 usage>0 且语义合格率 ≥70% | **管道可修活**：接线修好后 gold pattern 被采纳且用对 → in-loop 卖点复活，斜杠 bug 修复后演化边也有望复活 | 进确定性修复→重写，A2 跨篇实验按计划 |
| B 臂 usage>0 但语义合格率 <70% | **LLM 能力不够**：看得见也用不好（绑定/方向错） | in-loop 卖点砍或重定位为 governance；抽取栈重写时 pattern-following 是验收项 |
| B 臂 usage=0（schema 已可见+[NEW] 标记仍不用） | **采纳机制更深的问题**（prompt 结构/三步分解本身抑制使用） | 直接进抽取栈重写，联合抽取把 schema 嵌进单 prompt；in-loop 卖点等重写后再判 |
| C 臂对照 | 若 C 中同类事实以 claim_relation/compares 存在 → 注入=重标注（弱卖点）；若 C 中缺失/被丢 → 注入=真召回增益（强卖点） | 写进 A2 实验设计 |

## 6. 诚实声明

- 两个 wiring bug 是**好消息偏向的发现**（修好可能复活卖点）——但探针判读仍按预注册标准执行，不为复活卖点点软化。
- 单篇内 in-loop 天花板低（审计三证），探针通过的结论是"机制可活"，最终卖点证据要靠 A2 跨篇实验。
- 探针结果无论哪支都如实写进 bundle + 向用户报告。
