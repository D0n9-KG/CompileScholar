# DECISION: Thornton2026 gold 用 GLM-4-Flash 抽取 (非 GLM-5.2)

Date: 2026-08-14

## 决策
Thornton2026(第2篇segregation综述)gold 用 GLM-4-Flash 抽取, 非 ARFM2024 用的 GLM-5.2.

## 原因 (诚实, 多次尝试)
Paratera 上 GLM-5.2 / GLM-5-Turbo / Kimi-K2.6 **都是 reasoning model**, 在 Thornton2026 dense segregation 文本(86k字符)上 reasoning_tokens 达 16383-16387 爆 max_tokens(16384) → empty content → 卡住/超时。小window(3500)也爆。
- deepseek-chat: 非reasoning 不爆token, 但 socket 间歇 hang (llm_client已知问题), fallback到GLM-5-Turbo又爆token。
- **GLM-4-Flash**: Paratera 非reasoning 模型(实测 reasoning_tokens 无值), 不爆token, endpoint稳定。最终方案。

## 诚实标注
- ARFM2024 gold: GLM-5.2 (reasoning, 高质量)
- Thornton2026 gold: GLM-4-Flash (非reasoning, 牺牲质量换可行性)
- 两 gold 抽取模型不同 → 跨综述一致性评测需标注此差异
- 不假装两 gold 同质量抽取

## 验证
Thornton gold 抽完后, gold_audit_check.py 复用(改 SURVEY 路径)核验 Thornton gold verbatim 率, 与 ARFM 0.85 对比。若 Thornton 显著低, 诚实报"GLM-4-Flash抽取质量低, 仅作第2篇验证用"。

## 改动
.research_tmp/gold_extract.py: _call_gold_llm 加 GLM-4-Flash/deepseek/Kimi 分支(非reasoning用call_paratera, max_tokens 6000); extract_gold 加 --model 参数。
