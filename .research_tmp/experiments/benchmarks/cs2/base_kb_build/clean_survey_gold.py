# -*- coding: utf-8 -*-
"""留出综述金标清洗（在看到全量系统结果之前定下，10-03 晚）。

金标局限来自综述抽取记录（challenges 主张 + survey_claimed 缺口），审计发现 41% 像碎片、不少 challenges 条目根本不是
局限（例："use the graph recurrent neural networks (GRNN) to model the sequential behaviors of users"）。噪声对所有系统
同等压低召回、掩盖差异。清洗只看金标单条文本（不接触任何系统输出），由独立判分器（DeepSeek）判：
  limitations：是否是一个自足的、可理解的局限 / 挑战 / 开放问题陈述；
  properties ：是否是一个自足的、关于某类方法的性质陈述。
写回 survey_gold.json 每条的 "clean": true/false；评测同时报全部金标与 clean 金标（匹配结果复用，不重判）。
用法：python clean_survey_gold.py
"""
import concurrent.futures as cf
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "..", "base_kb_v2")
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..", "src")))
from kb_infra.llm import call_paratera, parse_json_response  # noqa: E402

JUDGE = os.environ.get("FIELD_JUDGE", "DeepSeek-V4.1-Flash")
ASK = {
    "limitations": "a self-contained statement of a limitation, weakness, challenge or open problem (of a method, a "
                   "family of methods, data, evaluation or the research area), understandable without its context",
    "properties": "a self-contained statement of a property of a method or family of methods (how it works, what "
                  "distinguishes it, a strength, or when it applies), understandable without its context",
}
PROMPT = """For each numbered statement, decide whether it is {ask}. Fragments, citations of what one paper did,
and statements whose subject is missing do not qualify.

{items}

Return JSON only: {{"qualifies": {{"1": true, "2": false}}}}"""


def judge(texts, role):
    out = {}
    for b in range(0, len(texts), 30):
        chunk = texts[b:b + 30]
        items = "\n".join(f"{b + i + 1}. {t[:300]}" for i, t in enumerate(chunk))
        obj = None
        for _ in range(3):
            obj = parse_json_response(call_paratera(PROMPT.format(ask=ASK[role], items=items), model=JUDGE,
                                                    max_tokens=2000, enable_thinking=False) or "")
            if isinstance(obj, dict) and isinstance(obj.get("qualifies"), dict):
                break
        q = (obj or {}).get("qualifies") or {}
        for i in range(len(chunk)):
            out[b + i] = bool(q.get(str(b + i + 1)))
    return out


def main():
    p = os.path.join(V2, "survey_gold.json")
    gold = json.load(open(p, encoding="utf-8"))
    jobs = []
    with cf.ThreadPoolExecutor(6) as ex:
        for pid, g in gold.items():
            for role in ("limitations", "properties"):
                jobs.append((pid, role, ex.submit(judge, [x["text"] for x in g[role]], role)))
        for pid, role, f in jobs:
            for i, ok in f.result().items():
                gold[pid][role][i]["clean"] = ok
    json.dump(gold, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for role in ("limitations", "properties"):
        n = sum(len(g[role]) for g in gold.values())
        c = sum(1 for g in gold.values() for x in g[role] if x.get("clean"))
        print(f"{role}: clean {c}/{n} = {c / n:.0%}")


if __name__ == "__main__":
    main()
