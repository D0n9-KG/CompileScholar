# 必核清单外部槽核验进度（2026-09-28 凌晨，dev20 双臂跑动期间并行完成）

## #3 ReportBench ✅（详见 VERDICT-reportbench-pull.md）

## memorized baselines ✅（2026-09-28 晨，用户过 gated 后全量到手）

- **公开渠道找到**：`allenai/asta-bench-solver-data`（HF，gated:auto
  ——用户点同意即解锁；比 S3/AWS 凭证路线简单一个量级，S3 路线废弃）
- 拉到 4 件（.research_tmp/.../cs2/arm_memorized/）：
  - Elicit `sqa/elicit/responses.json` **201 条标准 sections 形态**
    （citations+snippets 完整——直接进官方 scorer）
  - Perplexity DR `sqa/openai_dr/dev.json`+`test.json` 各 100
    （markdown 报告——format_solver 转形态，官方 repo 有转换器）
  - SciSpace `sqa/scispace/test.json` 100（response 文本）
- OpenScholar 预存答案不在此库（在 DVC）——它是模型答案非真产品，
  锚定价值低，暂不追
- 重判管线：三个基线的 dev 答案 → GLM judge（0.4 已验证的
  scorer_model=openai-api/glm/GLM-5.3 通道）→ matched-judge 对照

- `YeloDriver/MDAQA`：**license = CC-BY-4.0**（HF API 亲验），
  ungated——槽 2 使用合规（署名即可）
- 数据已在盘（mdaqa-data.json 5.2MB，benchmark-audit-0926/）
- 遗留：ScholarStack 五维 judge prompt 可否复用（未核，
  槽 2 执行时再查——其 MDAQA 协议是改造版非官方）

## #5 PeerQA ✅（全量核验补完 2026-09-28 晨）

- papers.jsonl=**24,265 篇语料池**（不是 208——208 是评测锚定
  论文数；24k 是全语料）、qa.jsonl 579 题、answerable：
  **True 414 / False 112 / None 53**（不可答子集=否定分析素材，
  设计档"495 可答性标注"≈414+53+若干边界，口径以实测为准）
- augmented answers 579 条（augmented_answer_free_form）——
  生成参考形态（官方判分的重叠指标用）
- 槽 1 完全就绪：数据+标注齐；判分=官方 answerability 指标
  （全确定性）+ROUGE；论文出处=arXiv 2502.13668（已核）

## #2 CS2 数据/判分链路端到端 ✅（0.4 冒烟即此——四 facet 全产）

## 效率诊断与优化决定（dev20 期间）

- ours 臂慢根因：**客户端并行不足**（默认 OURS_QUERY_FANOUT=4；
  33% 调用 >100s 是答题循环的大上下文重推理调用，76k prompt→652
  completion 形态）——服务器远未拉满（双臂负载上叠 4 路探针
  1-3.8s 全过）
- **决定**：dev100 主跑 OURS_QUERY_FANOUT=8；当前 dev20 不重启
  （resume 安全，改环境变量需重启进程）
- harness 臂串行（17.2 分钟/题）——下一轮改 2 路题间并行
  （Claude Code 子进程独立会话，服务器余量足够）
- 全局 cap 教训已修：cs2_runner 已设 G2_CAP 无限（首轮 dev20 因
  TOKEN_CAP_R3=4.9M 全局预算在第 1 题后耗尽，16/20 题被
  budget_abort 杀——已重跑）
