# -*- coding: utf-8 -*-
"""判分器 bug 严格复核:不依赖任何推测,逐行追代码+构造对照实验。

要验证的命题:
  P1(半信用惩罚蒸发):grader 返回的 supporting 是 md 标题文本而非 [n] id,
     citation_intersection(supporting, half_credit) 恒空 → 五折惩罚恒 0
  P2(对照组):若引用 id 正确出现在 supporting 里,惩罚会正常生效

实验设计:
  A. 构造最小报告:一句 claim + 2 条 title-only 引用(与 GPTR 同形态:
     正文 md 标题链接, snippets 空)
  B. 同一报告两种正文引用形态:
     B1 = md 标题链接(GPTR 形态)
     B2 = [1][2] 编号引用(我们形态)
     两者引用条目完全相同(title-only)
  C. 分别判分,抓 prompt_logs 里的 supporting 返回形态+最终 recall/precision
  若 B1 无惩罚、B2 有惩罚(或 supporting 形态不同) → P1 实锤
  若两者都无惩罚 → 我们的解释错,bug 在别处,回溯
"""
import asyncio
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import httpx


def _patch():
    _orig = httpx.Client.send

    def _send(self, request, **kw):
        request.headers["Connection"] = "close"
        return _orig(self, request, **kw)

    httpx.Client.send = _send
    _orig_a = httpx.AsyncClient.send

    async def _asend(self, request, **kw):
        request.headers["Connection"] = "close"
        return await _orig_a(self, request, **kw)

    httpx.AsyncClient.send = _asend


_patch()

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\scratch\ai2_baseline_2026-09-08\asta\asta-bench-main")
ENVF = r"C:\Users\D0n9\Desktop\CompileScholar\.env"
for line in open(ENVF, encoding="utf-8"):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()
os.environ["OPENAI_API_KEY"] = "sk-placeholder"
os.environ["DEEPSEEK_API_KEY"] = os.environ.get("PARATERA_API_KEY", "")
os.environ["DEEPSEEK_BASE_URL"] = os.environ.get("PARATERA_BASE_URL", "")

from inspect_ai.model import get_model  # noqa: E402
from inspect_ai.scorer import Target  # noqa: E402
from astabench.evals.sqa.task import score_citation  # noqa: E402


class _State:
    def __init__(self, completion, meta):
        self.output = type("O", (), {"completion": completion})()
        self.metadata = meta


CITS = [
    {"id": "[1]", "snippets": [], "title": "continuous dropout"},
    {"id": "[2]", "snippets": [], "title": "biased dropout"},
]
# 两条引用条目都没 snippets → filter_citation=False → 走 JUST_HAS_A_TITLE 分支
# → half_credit=[ '[1]','[2]' ] (两条都 title-only)

# B1: GPTR 形态——正文里用 md 标题链接,id 字段也是 [1](适配器塞的),
# 但正文里 [1]/[2] 字符串不出现(只有 md 链接文本)
TEXT_B1 = ("Dropout is widely recognized as an effective regularization "
           "technique for training deep networks because it prevents "
           "co-adaptation of feature detectors ([continuous dropout]"
           "(https://scholar.google.com/scholar?q=continuous+dropout)) "
           "and works across network sizes ([biased dropout]"
           "(https://scholar.google.com/scholar?q=biased+dropout)).")

# B2: 我们形态——正文里用 [1][2] 编号
TEXT_B2 = ("Dropout is widely recognized as an effective regularization "
           "technique for training deep networks because it prevents "
           "co-adaptation of feature detectors [1] and works across "
           "network sizes [2].")


async def run_case(text, label):
    sections = [{"title": "s", "text": text,
                 "citations": [dict(c) for c in CITS]}]
    report = json.dumps({"sections": sections}, ensure_ascii=False)
    model = get_model("openai-api/paratera/DeepSeek-V4.1-Flash")
    if hasattr(model, "api") and hasattr(model.api, "model_name"):
        model.api.model_name = "DeepSeek-V4.1-Flash"
    sc = score_citation(model, sentence_wise_cit_eval=False, all_at_once=True,
                        temperature=0.5, top_p=0.95)
    state = _State(report, {"initial_prompt": "test"})
    s = await sc(state, Target(["{}"]))
    pl = (s.metadata or {}).get("prompt_logs") or {}
    print(f"\n===== {label} =====")
    print("score:", json.dumps(s.value, ensure_ascii=False))
    for ent in (pl.get("claims") or [])[:1]:
        claims = (ent.get("data") or {}).get("claims") or []
        for c in claims:
            print(f"  supporting={c.get('supporting')!r} "
                  f"non={c.get('non_supporting')!r} "
                  f"fully={c.get('is_fully_supported')}")
    for ent in (pl.get("half_credit") or [])[:1]:
        t = ent.get("data") or ent
        print(f"  half_credit清单: {json.dumps(t, ensure_ascii=False)[:120]}")


async def main():
    # 判分是随机的(temp 0.5)——每形态跑 3 次看稳定性
    for i in range(3):
        print(f"\n########## 第 {i+1} 轮 ##########")
        await run_case(TEXT_B1, "B1: md标题链接(GPTR形态)")
        await run_case(TEXT_B2, "B2: [n]编号(我们形态)")


asyncio.run(main())
