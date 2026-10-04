# 附：GPUStack 网关 httpx 兼容性问题档案（2026-09-28 凌晨）

## 现象

所有 httpx 系客户端打 GPUStack（192.168.199.73）稳定 502/超时：
openai SDK / litellm / STORM VLLMClient（+ 早前 codex CLI 的 502
大概率同根因）。而 urllib（每请求新 TCP 短连接）全通——我们全部
自建管线（call_local 等）走 urllib，从未受影响。

## 定位过程

- 头部对齐实验：UA/x-stainless/Accept-Encoding 全排除（urllib 加
  httpx 全套头仍通）
- **httpx + `Connection: close` 头 → 3 连 200**（复现关键证据：
  连接池复用是触发条件之一）
- 但 openai SDK 类层 patch（httpx.Client.send 强制 Connection:
  close）仍 502——**不止 keep-alive 一个因素**（openai SDK 还有
  某种行为差异未定位，候选：流式响应处理/连接超时语义）

## 影响面

- 主线管线：**零影响**（全 urllib）
- 外部工具：STORM（槽 4 对手）LM 通道受阻；codex 已裁弃无碍

## 待选解法（槽 4 执行期决定）

1. 给 STORM 写 urllib 后端 LM 子类（半小时活，确定性最高）
2. openai SDK 反代（本地 127.0.0.1 urllib-proxy 翻译请求）
3. GPUStack 网关侧修复（需要服务器管理员=用户）

修复模块留存：cs2/gpustack_httpx_fix.py（httpx 全局
Connection:close patch——对纯 httpx 场景有效，对 openai SDK 不够）
