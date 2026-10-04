# 阶段 0.1 冒烟判决：harness 代理层 × 本地 Qwen3.8-27B（2026-09-27）

## 结论

**Claude Code 臂成立。** 子进程 Claude Code 2.1.246 经 GPUStack
Anthropic 兼容端点直连本地 Qwen3.8-27B，六任务工具循环套件
**两轮均 6/6 PASS**，零畸形工具调用、零格式失败、零超时。

- 多步循环稳定性：最长 8 turns（T5 编译报告），全部收敛
- 输出格式合格率：inline JSON 4/4、文件工件 JSON 3/3（schema 校验过）
- 单任务墙钟：9.9–74.8s（27B 单流 ~26 tok/s 下）

**Codex 臂弃用（用户裁定 09-27：只用 Claude Code）。** 技术记录：
Codex CLI 0.157 已移除 `wire_api="chat"`，仅支持 responses API；
GPUStack `/v1/responses` 裸调用 200，但 codex 实际请求形态稳定 502
（经转发代理捕获前被裁，不再排查）。论文披露口径：harness 臂=
Claude Code 作 agent 层 + matched 27B 模型，单一 harness 披露。

## 接法（复用配方）

- `ANTHROPIC_BASE_URL=http://192.168.199.73`（GPUStack 原生
  Anthropic 兼容 `/v1/messages`，零代理层）
- `ANTHROPIC_AUTH_TOKEN=<LOCAL_API_KEY>` + `--model Qwen3.8-27B`
- **Windows 坑**：必须直调
  `npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe`——
  npm 的 `claude.cmd` 壳会二次解析 prompt 里的 `<>&|`（cmd 重定向
  符），"The system cannot find the file specified" 即此因
- 未识别模型警告（auto-compact 按 200k 假设）无害；如需 1M 窗口
  传 `--model "Qwen3.8-27B [1m]"` 或设 CLAUDE_CODE_MAX_CONTEXT_TOKENS

## 套件构成（smoke_runner.py，可重跑）

T1 单文件读+inline JSON｜T2 多文件读+结构化文件写｜T3 跨文件检索+
JSON 数组｜T4 纯 shell 计数｜T5 迷你编译报告（分节+逐 claim
file:line 锚定，CS2 形态微缩）｜T6 迭代修复坏 JSON（读-改-验证环）

## 观察与注意

1. **限权是设计行为**：T5 中模型想跑 `python -m json.tool` 自验被
   allowlist 挡（该任务只给了 Read/Glob/Grep/Write）——正式臂放开
   工具集即无此问题；反过来说明 `-p` 模式 allowlist 硬隔离有效
2. GPUStack 在 Claude 套件跑批期间曾对 codex 请求 502（并发+请求
   形态叠加疑），单流 26.4 tok/s 健康未受影响——0.5 并发压测时
   一并观察
3. 单流 26.4 tok/s（TTFT 3.1s，512 token 流式实测）≈记忆中 20
   tok/s 档，P9"10 tok/s 降速"本次未复现——0.5 复核项保留
4. T5 的 claim 锚定（source_file+source_line 全部真实存在）一次
   通过——这对槽 3 的 citation 真实性主张（官方不验 snippets→我们
   验）是直接正信号

## 工件

- 跑批器：`smoke_runner.py`（任务重置+schema 校验+结果 JSON 落盘）
- 结果：`results/claude_results.json`（run2）、
  `results/claude_results_run2.json`（同）；run1 数字见本档结论行
  （9.9/32.0/19.7/25.8/74.8/35.1s，6/6）
- 后续臂的生产化：正式 harness 臂配置（工具集全开+检索 MCP 0.3
  产出接入）在槽 3 实施时定稿

## 对设计档的回写

EXPERIMENT-DESIGN §八-1"不胜任→双试→降级自制 ReAct 臂"路径
**未触发**——Claude Code 胜任，harness 臂按原设计成立。
