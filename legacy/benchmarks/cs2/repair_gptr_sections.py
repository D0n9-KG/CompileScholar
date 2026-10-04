# -*- coding: utf-8 -*-
"""存量 GPTR 答案的确定性修复（不重跑 GPTR、不改内容），对齐 cs2_gptr.py 的适配器修复：
正文里的 md 链接 [title](url) / ([title](url)) 替换为该节 citations 中同 url 的 id（[k]），
使 citation id 出现在正文（官方 task.py:81 契约），半信用扣减按官方设计生效。
snippet 无法事后补（运行时检索摘要未落盘）——保持空，即 title-only 档，与官方 retrieverless 口径一致。
输出 arm_gptr/answers_gptr_cs2.fixed.json，并报告 id-in-text 率修复前后。"""
import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SRC = "arm_gptr/answers_gptr_cs2.json"
OUT = "arm_gptr/answers_gptr_cs2.fixed.json"
PAREN = re.compile(r"\(\[([^\]\[]{2,120})\]\((https?://[^)\s]+)\)\)")
BARE = re.compile(r"\[([^\]\[]{2,120})\]\((https?://[^)\s]+)\)")


def rate(rows):
    tot = hit = 0
    for r in rows:
        for s in r["sections"]:
            for c in s["citations"]:
                tot += 1
                hit += c["id"] in s["text"]
    return tot, hit


a = json.load(open(SRC, encoding="utf-8"))
rows = a if isinstance(a, list) else list(a.values())
t0, h0 = rate(rows)
for r in rows:
    for s in r["sections"]:
        by_url = {(c.get("metadata") or {}).get("url"): c["id"] for c in s["citations"]}
        sub = lambda m: by_url.get(m.group(2)) or m.group(0)
        s["text"] = BARE.sub(sub, PAREN.sub(sub, s["text"]))
t1, h1 = rate(rows)
json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"citations {t0} | id-in-text before {h0} ({h0 / max(1, t0):.3f}) -> after {h1} ({h1 / max(1, t1):.3f})")
