# -*- coding: utf-8 -*-
"""P5 s0: map gt nugget dirs -> qid, harness per-q score, num_turns."""
import csv, json, os, statistics, re
csv.field_size_limit(10**9)
B = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks"
G = os.path.join(B, "deepscholar", "dsb", "dataset", "gt_nuggets_outputs")
idx = {}
for i in os.listdir(G):
    p = os.path.join(G, i, "res.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        idx[i] = (d["qid"], len(d.get("supported_nuggets", [])), len(d.get("nuggets", [])))
sc = {}
for r in csv.DictReader(open(os.path.join(B, "deepscholar", "dsb", "results_harness_dsb", "nugget_coverage", "storm.csv"), encoding="utf-8")):
    sc[re.split(r"[\\/]", r["folder_path"])[-1]] = float(r["nugget_coverage"])
st = {}
for f in ["results_storm_rest"]:
    for r in csv.DictReader(open(os.path.join(B, "deepscholar", "dsb", f, "nugget_coverage", "storm.csv"), encoding="utf-8")):
        st[re.split(r"[\\/]", r["folder_path"])[-1]] = float(r["nugget_coverage"])
ans = {r["qid"]: r for r in json.load(open(os.path.join(B, "cs2", "arm_harness_dsb", "answers_harness_dsb.json"), encoding="utf-8"))}
storm_dirs = set(os.listdir(os.path.join(B, "cs2", "arm_storm_dsb")))
harn_idx = set(os.listdir(os.path.join(B, "cs2", "arm_harness_dsb", "indexed")))
rows = []
for i, (q, ns, nn) in sorted(idx.items(), key=lambda x: int(x[0])):
    rows.append((int(i), q, ns, nn, sc.get(i), ans.get(q, {}).get("num_turns"), q in storm_dirs, i in harn_idx, st.get(i)))
for r in rows:
    print(r)
v = [r[4] for r in rows if r[4] is not None]
print("harness mean", statistics.mean(v), len(v))
print("n gt dirs", len(idx), "unique qids", len(set(q for q, _, _ in idx.values())))
