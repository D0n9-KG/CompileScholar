# DECISION: 确定性修复（B+ 执行序第 4 步）

日期：2026-08-25。依据：DECISION-2026-08-25-b-plus-rebuild 第 1.2 节 + 探针病灶清单（PROBE_adoption_result.md 第五节）。铁律：先 commit 本文档再改码。

## 修复范围（五项）

### D1 确定性切块（structure_mapper，最大项）

**现状病灶**：map_structure 是 LLM 调用（BIO 前科：7 DAG 节点切成 1 section；探针方差：n6/c1 解析 0 节点）；路由硬编码 Kimi-K2.6（`llm=="deepseek"` 字面量判断，探针 run 实际走 Kimi——provenance 违规实锤）；loader 过滤 `len<10` 文本块**把 "METHODS" 等短标题丢掉**。

**设计**（全部确定性，零 LLM）：
1. `load_paper_blocks` 保留标题块：content_list 中带 `text_level` 的 text 块（或匹配标题 regex 的短行）即使 <10 字符也保留，打 `is_heading=True` 标记；过滤元数据伪标题（doi / Received…accepted / 纯日期行）。语料实测：25/25 论文有 text_level 标记（中位 24 字符）。
2. `map_structure` 重写为确定性：按 heading 边界切 section；<80 字符的碎 section 并入相邻；单个 section >12k 字符按块边界二分（"name (part N)"）；全篇无标题 → 固定 10k 字符切块兜底（永不退化为"整篇一 section"）。
3. DAG=线性链（deps=[前一 section]）——predecessor context 保留且可复现；discourse_role 由标题名 regex 归类（abstract/introduction/methods/results/discussion/conclusion/related work/appendix），未匹配用标题原文。
4. 删 STRUCTURE_PROMPT + `_call_paratera_structure`（Kimi 硬编码随死）。签名不变（llm/domain 参数保留忽略——eval scifact/scirex 调用方零改动）。
5. 附带收益：省 1 次 LLM 调用/篇（效率目标）+ Kimi 依赖消失。

**验收**：同篇 3 次 run 的 sections 完全一致（字节级）；DQN 143 块切出的 section 覆盖全部 text 块。

### D2 ablation arm 真开关（agent.process_paper_via_kernel）

**现状病灶**：add_only/no_intra_dag 与 full 行为完全相同（audit E）。arm 语义（对齐 A1/A2/A3 ablation）：
- `frozen`：无演化、无 align、无 repair（既有，正确）
- `full`：per-section 演化 drain + align + repair（既有，正确）
- `add_only`：per-section 演化 drain + align，**跳过 repair**（split/merge/rename detect）
- `no_intra_dag`：演化推迟到篇末批量 drain（batch 模式）+ align + repair——单篇内 schema 不并进、篇末一次性演化，对照"in-loop 时机"轴
- 未知 arm：**报错退出**（不静默当 full）

### D3 model provenance（llm_client，按子代理审计方案）

**现状病灶**：零调用记录（usage 字段三处被丢）；call_llm 失败静默 fallback GLM-5-Turbo 且返回值不可区分（hypergraph_lifter 外层再 retry 3 次 → 单次"调用"最多混 3 模型全不可见）。

**设计**（仅动 llm_client.py，~50 行，零调用方改动）：
1. 模块级 CALL_LOG + `_log_call`：`{ts, run_id, provider, model, ok, latency_ms, prompt_tokens, completion_tokens, attempt, fallback_for}`；env `LLM_CALL_LOG` 给路径则逐 call append JSONL。
2. `_chat_once` 返回 usage（tokens 唯一来源）。
3. fallback 分支记录 `fallback_for="deepseek-chat"`——静默换模型变可见。
4. 新增 env `LLM_ALLOW_FALLBACK=0`：科学 run 关 fallback（行为开关，默认开保持鲁棒性但日志可见）。

### D4 seed 参数化

三个 chat 入口加 `seed: int | None = None` 透传 payload（OpenAI-compat 大概率支持，首次 run 实测验证；不支持则如实记录"API 不支持 seed"并靠多 seed 报告）。agent/入口层透传。

### D5 run_kernel.py 正式入口

替代全部 _tmp 脚本的官方 runner：`python src/run_kernel.py --papers PPR_xxx [--papers ...] --domain ml --arm full --seed 1 --tag mytag`。设 LLM_RUN_ID、跑完 bundle 自动存（既有 _save_kernel_bundle）、call log 落盘 bundle 目录、打印 provenance 摘要（模型×次数×tokens×耗时）。

### 不做（诚实范围）

- **单 writer**：审计后确认现状已是单 writer（唯一写路径=KB ledger，探针 run ledger 144 add_edge + 9 align_concept_merge，无旁路直改）——无需修。
- 联合抽取/语义 verifier：下一步抽取栈重写的事，不混进本步。
- `_locate_chapter_text` 保留作安全网（D1 后基本不再触发）。

## 执行序与验收

D3（独立，先行）→ D1（最大）→ D2 → D5 → **终验：DQN 重跑 3 次**——sections 字节级一致（D1）、call log 完整（D3/D4）、arm 行为可区分（D2）、单篇时长与边数报告（效率基线）。
