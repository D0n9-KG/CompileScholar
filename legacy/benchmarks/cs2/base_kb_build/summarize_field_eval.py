# -*- coding: utf-8 -*-
"""汇总留出综述检验（只读 heldout_eval/*.scores.json + survey_gold.json，不调用模型）。
报：每系统宏平均召回（全部金标 / clean 金标）、候选条数、配对差（state_v2 − 其他，按综述 bootstrap）。
用法：python summarize_field_eval.py"""
import glob
import json
import os
import random
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "..", "base_kb_v2")
gold = json.load(open(os.path.join(V2, "survey_gold.json"), encoding="utf-8"))
rows = {os.path.basename(p).split(".scores")[0]: json.load(open(p, encoding="utf-8"))
        for p in glob.glob(os.path.join(V2, "heldout_eval", "*.scores.json"))}
systems = sorted({s for r in rows.values() for s in r})


def rec(pid, s, part, clean):
    items = gold[pid][part] if part != "families" else [{"clean": True}] * len(gold[pid]["families"])
    idx = [i for i, x in enumerate(items) if (x.get("clean") or not clean)]
    if not idx or s not in rows[pid]:
        return None
    m = rows[pid][s][part]["matches"]
    return sum(1 for i in idx if m.get(f"G{i + 1}")) / len(idx)


print(f"surveys scored: {len(rows)}")
for part in ("families", "properties", "limitations"):
    for clean in (False, True):
        if part == "families" and clean:
            continue
        line = []
        for s in systems:
            v = [x for x in (rec(p, s, part, clean) for p in rows) if x is not None]
            nc = [rows[p][s][part]["n_cand"] for p in rows if s in rows[p]]
            line.append(f"{s} {st.mean(v):.3f}(n{len(v)},cand {st.mean(nc):.0f})" if v else f"{s} -")
        print(f"{part:11s} {'clean' if clean else 'all  '} | " + " | ".join(line))
for a in ("state_v2", "state"):
    for b in [x for x in systems if x != a]:
        for part in ("limitations", "properties"):
            d = [rec(p, a, part, True) - rec(p, b, part, True) for p in rows
                 if rec(p, a, part, True) is not None and rec(p, b, part, True) is not None]
            if len(d) < 3:
                continue
            random.seed(0)
            m = sorted(st.mean(random.choice(d) for _ in d) for _ in range(4000))
            print(f"  {a}-{b} {part} (clean): {st.mean(d):+.3f} CI[{m[100]:+.3f},{m[3900]:+.3f}] n={len(d)}")
