# 夜间自主执行计划（09-21 深夜，用户授权）

> **用户原话授权**："你一直跑吧，我去睡觉了，门过了就直接放门，然后你去做外部基准的准备。"
> = slot 冒烟门过后**无需等用户**直接放行全量；建库链推进；随后做基线（PaperQA2/LightRAG）适配准备。
> 纪律不豁免：每 stage 启动后活体确认；异常先取证再修；失败不掩盖；产物进 git。

## 阶段 0：slot 冒烟门判决（当前在跑，task bnbxw6vlf）

- 日志 `kb/slot_smoke.log`，产物 `kb/records_smoke.json`，账本 `kb/ledger_build.jsonl`（run_id=multi-slot-smoke）
- **判决三项**（全过才放行）：
  1. **契约金丝雀**：ledger 里 ok=False 比例 <5%；无整篇 parse 崩溃；salvage 触发次数记录
  2. **记录实读**：抽 3 篇（QLoRA 表格/GPT-3 长文/一篇 bio）各读 5-8 条记录——quote 逐字回原文核对（grep 验证）、数值保真、subject/measure 填充、实体链接（entity_id 解析）
  3. **PS-53 基线对照**：每篇记录数（PS-53 ~40-80/篇量级）、quote 锚定率（基线 100%）、kind 分布合理（finding/claim/comparison 等都有）
- 不过门：取证→修→重跑冒烟，**不放行全量**；连续两次不过门则停下留给用户

## 阶段 1：slot 全量马拉松（过门即发，~2h）

```
cd .research_tmp/experiments/benchmarks/scholarqa_multi
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python run_stage.py --name slot -- \
  python -m kb_compiler.records.slot \
  --texts <abs>/corpus/texts --manifest <abs>/corpus/manifest.json \
  --cards <abs>/kb/cards.json --registry <abs>/kb/registry.json \
  --vocab <abs>/kb/dim_vocab_v1.json --out <abs>/kb/records_slot.json \
  --model {MODEL} --workers 12
```
（<abs> = C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/scholarqa_multi）
- 逐篇断点续跑（records_slot.json 已有篇跳过）；run_stage 自动 LOCAL_MAX_CONCURRENT=32+allowlist+纯度断言
- 冒烟的 10 篇结果可复用：先把 records_smoke.json 内容并入 records_slot.json（同 pid 跳过重抽）——省 10 篇
- 监控：每 ~20 分钟看一次 ledger 速率与失败数；结束后核对 430/430 篇齐

## 阶段 2：建库链后段（依序，每个 stage 过 run_stage）

1. **postcheck**（五门修复）：`--records kb/records_slot.json --texts ... --vocab ... --out-dir kb/postcheck --model {MODEL}` → 产出 records_checked（确认实际文件名）
2. **table_channel F24**（确定性表格通道）：`--texts --checked <postcheck产物> --cards kb/cards.json --registry kb/registry.json --out kb/records_tables.json`
3. **table_semantic F35**：`--records kb/records_tables.json --texts ... --registry kb/registry.json --out-dir kb/f35 --provider local --model Qwen3.8-27B --canary`（金丝雀已复验 PASS；G5 硬停线生效）
4. **notation_harvest**（公式收割）：`--texts --records <F35后记录> --out kb/notation.json --provider local --model Qwen3.8-27B`
5. **registry_round2**（实体链接增长）：`--registry kb/registry.json --records kb/records_slot.json --out-dir kb --model {MODEL} --embed-cache kb/embed_cache`（topk 锚点自动启用）
6. **views 编译**：`python -m kb_compiler.views.compiler --records <最终记录> --registry <round2后registry> --vocab kb/dim_vocab_v1.json --manifest corpus/manifest.json --out kb/views.json`（+ render/search_text 索引，按 views 模块 CLI 实况）
- 每个 stage：启动活体确认 + 结束核对产物 + ledger 记账；出错=取证→修→重跑（stage 级断点）

## 阶段 3：外部基线准备（建库链跑着就能开始写代码，ingestion 等 GPU 空）

- 库已就位且最新：paper-qa 2026.8.12 / lightrag-hku 1.5.7
- 模板：PS 时代 `.research_tmp/experiments/archive/paperscope_2026-09/paperscope_r2/` 下的 lightrag harness（gate2_lightrag.py）与 paperqa2_check
- 要做：`benchmarks/_shared/tools/multi_baseline_*.py` 两个 harness——语料=430 texts；LLM 端点=本地 Qwen3.8-27B（与我们同模，公平性）；嵌入=本地 qwen3-embedding-8b-local；题目=108 题 input；输出=答案+[序号]引用（对齐官方 ctx 序号，参考 id_mapping.json 反向）
- LightRAG ingestion（建图）是 LLM 大户：排在 slot 马拉松之后或用户副本就位后
- PaperQA2：索引=嵌入为主（轻），问答时 LLM

## 阶段 4：收尾记账

- 全部产物 git 提交（records/views 大文件先看体积，>50MB 考虑 gitignore+落盘）
- memory 恢复点更新（session-2026-09-21-scaling-and-corpus.md 追加夜间战果段）
- 给用户的晨报：各 stage 结果+QC 仪表+异常与处置+基线准备状态

## 红线（不豁免）

- 冒烟门判决必须实读记录（不是跑通就算过）
- 编造红线：记录 quote 抽验必须 grep 原文核对
- 任何 stage 纯度断言失败=停下取证，不带病推进
- 判分（judging）不在今夜范围——首跑答题要等预注册 v1.0（36 题清单用户过目）
