# -*- coding: utf-8 -*-
"""P5 s2: re-run the OFFICIAL nuggetizer assign prompt (window=10, temp 0) to
recover per-nugget labels (official run only persisted per-question scores).

usage: python s2_assign.py <arm: harness|storm> <run_id>
output: assign_<arm>_r<run>.json  {idx: [label,...]}
"""
import ast, json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\deepscholar\dsb\eval\nuggetizer\src")
from nuggetizer.prompts.assigner_prompts import create_assign_prompt  # official prompt
from nuggetizer.core.types import ScoredNugget, NuggetAssignMode
from llm import chat

OUT = os.path.dirname(os.path.abspath(__file__))
arm, run = sys.argv[1], sys.argv[2]
bundles = json.load(open(os.path.join(OUT, "bundles.json"), encoding="utf-8"))


def one(b):
    text = b["harness_text"] if arm == "harness" else b["storm_text"]
    nugs = [ScoredNugget(text=n["text"], importance=n.get("importance", "vital")) for n in b["nuggets"]]
    labels = []
    for s in range(0, len(nugs), 10):
        win = nugs[s:s + 10]
        msgs = create_assign_prompt(b["query"], text, win, NuggetAssignMode.SUPPORT_GRADE_3)
        got = None
        for t in range(3):
            resp = chat(msgs, f"assign_{arm}_r{run}_{b['idx']}_{s}", temperature=0.0 if t == 0 else 0.2)
            try:
                lab = ast.literal_eval(resp.replace("```python", "").replace("```", "").strip())
                if isinstance(lab, list) and len(lab) == len(win):
                    got = [str(x).lower() for x in lab]
                    break
            except Exception:
                pass
        labels.extend(got or ["failed"] * len(win))
    return b["idx"], labels


with ThreadPoolExecutor(max_workers=6) as ex:
    res = dict(ex.map(one, bundles))
json.dump(res, open(os.path.join(OUT, f"assign_{arm}_r{run}.json"), "w", encoding="utf-8"), indent=1)
tot = sum(len(v) for v in res.values())
sup = sum(x == "support" for v in res.values() for x in v)
print(arm, run, "nuggets", tot, "support", sup, "strict cov(micro)", round(sup / tot, 3),
      "failed", sum(x == "failed" for v in res.values() for x in v))
import statistics
print("macro strict cov", round(statistics.mean(sum(x == "support" for x in v) / len(v) for v in res.values()), 3))
