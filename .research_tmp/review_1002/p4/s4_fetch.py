# -*- coding: utf-8 -*-
"""P4 stage 4: fetch cited-paper HTML for all sampled items (cached). Output: s4_fetch.json"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import P4, calls, fetch_html  # noqa: E402

d = json.load(open(os.path.join(P4, "s3_sample.json"), encoding="utf-8"))
stat = {}
for s, v in d.items():
    ok = 0
    for p in v["picked"]:
        t = fetch_html(p["arxiv_id"])
        p["html_chars"] = len(t)
        ok += bool(t)
        print(s, p["arxiv_id"], len(t), flush=True)
    stat[s] = {"n": len(v["picked"]), "html_ok": ok}
json.dump(d, open(os.path.join(P4, "s3_sample.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(stat, open(os.path.join(P4, "s4_fetch.json"), "w"), indent=1)
print(stat, calls())
