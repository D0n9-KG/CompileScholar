# -*- coding: utf-8 -*-
import json, os, re, sys, collections, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
V2 = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb_v2"
P = json.load(open(os.path.join(V2, "papers.json"), encoding="utf-8"))
R = json.load(open(os.path.join(V2, "records.json"), encoding="utf-8"))
C = collections.Counter
recs = [(pid, r) for pid, v in R.items() for r in v["records"]]
res = [(pid, r) for pid, r in recs if r.get("kind") == "result"]
print("result papers", len({p for p, _ in res}), "by layer", C(P[p]["layer"] for p, _ in res))
print("result sample", json.dumps(res[0][1], ensure_ascii=False)[:600])
dimk = C(k for _, r in res for k in (r.get("dims") or {}))
print("dims keys", dimk.most_common(12))
def g(d, *ks):
    for k in ks:
        v = (d or {}).get(k)
        if isinstance(v, dict): v = v.get("canonical") or v.get("surface")
        if v: return str(v).lower().strip()
    return ""
cells = collections.defaultdict(set)
for pid, r in res:
    d = r.get("dims") or {}
    ds, mt = g(d, "dataset", "benchmark", "task"), g(d, "metric") or str(r.get("measure") or "").lower()
    if ds and mt: cells[(ds, mt)].add(pid)
print("(dataset,metric) cells", len(cells), "with >=2 papers", sum(1 for v in cells.values() if len(v) >= 2))
print(" top", [(k, len(v)) for k, v in sorted(cells.items(), key=lambda x: -len(x[1]))[:8]])
# comparison tables in surveys: rows?
ct = [(pid, r) for pid, r in recs if r.get("snapshot_type") == "comparison_table"]
print("comparison_table", len(ct), "fields", C(k for _, r in ct for k in r).most_common(20))
print(" sample", json.dumps(ct[0][1], ensure_ascii=False)[:700])
tn = [(pid, r) for pid, r in recs if r.get("snapshot_type") == "taxonomy_node"]
print("taxonomy_node", len(tn), "surveys", len({p for p, _ in tn}), "fields", C(k for _, r in tn for k in r).most_common(20))
print(" sample", json.dumps(tn[3][1], ensure_ascii=False)[:900])
# do taxonomy nodes carry member method names / cited works?
mem = C(len(r.get("members") or r.get("children") or []) for _, r in tn)
print(" members dist", mem.most_common(8))
