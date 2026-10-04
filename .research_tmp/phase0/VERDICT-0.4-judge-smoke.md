# 阶段 0.4 判决：InspectAI × GLM-5.3 judge 冒烟通过（2026-09-27）

## 结论

**官方 AstaBench sqa scorer × GLM-5.3（Paratera）零代码替换成立。**
端到端实跑：memorized solver（手搓 CS2 形态报告）→ 官方三 scorer
（rubric/precision/citation，`sqa(scorer_model="openai-api/glm/GLM-5.3")`）
→ 四 facet 全部产出，json_schema strict 兼容+重试路径实测有效。

| facet | 得分（手搓报告） |
|---|---|
| global_avg | **0.723** |
| ingredient_recall | 0.464 |
| answer_precision | 1.000 |
| citation_precision | 0.857 |
| citation_recall | 0.571 |

（分数值本身无意义——报告是手搓的；冒烟验证的是判分管线。）

## 关键事实（阶段 1 规划用）

1. **模型名大小写敏感**：Paratera 服务端要 `GLM-5.3`（`glm-5.3` 报
   400 "no healthy deployments"）。InspectAI 模型串=
   `openai-api/glm/GLM-5.3` + env `GLM_API_KEY`/`GLM_BASE_URL`
2. **GLM-5.3 是思考模型**：json_schema strict 调用思考 ~2.6k token 后
   吐合法 JSON——**GenerateConfig 必须留足预算**（官方 scorer 主路径
   不设 max_tokens=服务端默认，OK；`_score_evidence` 的 max_tokens=100
   调用会被思考吃掉，但 simplified_eval=True 默认路径不走它——记录在案）
3. **thinking 显式关闭反而破坏 schema 约束**（实测 thinking:disabled +
   json_schema → content 变纯文本）——保持思考开启
4. **延迟与成本实测（单题）**：9.5 min 墙钟 / 88.8k judge tokens
   （O 82.7k 含 R 21.6k reasoning）。dev100 串行 ≈16h → 必须并行
   （inspect eval -j + Paratera 并发上限待测）
5. astabench 安装：editable 本地 repo（scratch/ai2_baseline_2026-09-08/
   asta/asta-bench-main）+ inspect-ai 0.3.258；import 注册表要求
   OPENAI_API_KEY 存在（占位值即可，judge 不走它）
6. .eval 日志=zip 封装（新版格式），细读用 inspect_ai.log 官方 API

## 与官方 gemini 分数相关性小验：无法执行

无 gemini 渠道（硬约束本就排除了 GPT/Gemini）。披露口径按设计档：
"替代判分"披露 + memorized baselines 同 GLM 重判=matched-judge 对照锚
（Elicit/Perplexity/Sonar 预存答案下载待阶段 1）。

## 工件

- 冒烟任务：`phase0/judge_smoke/smoke_task.py`（memorized solver 模式
  =阶段 1 灌预存答案的模板）
- GLM 探针：`glm_probe.py`/`glm_probe2.py`（连通+json_schema 三形态）
- eval 日志：`phase0/judge_smoke/logs/*.eval`

## 追记（2026-09-28 晨）：GLM-5.3 判分关思考判决——不可关

用户提议测试 thinking:disabled 提速判分。严格对照实验（官方 scorer
同款 json_schema strict + 5 道判分题）：

- **schema 兼容性**：关思考后 4 题中 3 题 content 为空（约束被破坏，
  与 0.4 冒烟"关思考 content 变纯文本"同根因且更糟）
- **语义准确性**：dropout 计数类 criterion（"至少两个副作用"）：
  开思考判 2（正确），关思考判 0（漏数）
- 开思考 4/4 全对；思考 ~600-1100 ch/次（判分题比生成题思考短）

**判决：判分端思考必须开**（质量必需，速度代价接受）。提速正道=
流水线化+sample 并行。答题端（本地 Qwen）本来就关思考（输出预算
判决），两端策略相反是各自模型的性质差异，不冲突。
