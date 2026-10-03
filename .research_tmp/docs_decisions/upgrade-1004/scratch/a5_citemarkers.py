# -*- coding: utf-8 -*-
import json, os, re, sys, collections, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
V2 = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb_v2"
P = json.load(open(os.path.join(V2, "papers.json"), encoding="utf-8"))
R = json.load(open(os.path.join(V2, "records.json"), encoding="utf-8"))
C = collections.Counter
recs = [(pid, r) for pid, v in R.items() for r in v["records"]]
NUM = re.compile(r"\[\s*\d+(?:\s*[,\-–]\s*\d+)*\s*\]")
AY = re.compile(r"\b[A-Z][A-Za-z'\-]+(?: et al\.?| and [A-Z][A-Za-z'\-]+)?,?\s*\(?(?:19|20)\d\d[a-z]?\)?")
by = C(); tot = C()
for pid, r in recs:
    q = r.get("quote") or ""
    k = r.get("kind")
    tot[k] += 1
    if NUM.search(q): by[(k, "num")] += 1
    elif AY.search(q): by[(k, "authyear")] += 1
for k in ("survey_claim", "lineage", "domain_snapshot", "absence", "finding", "method", "limitation", "result"):
    print(k, tot[k], "num", by[(k, "num")], "authyear", by[(k, "authyear")])
# comparison_table claim rows
ct = [r for _, r in recs if r.get("snapshot_type") == "comparison_table"]
rows = sum(len(r.get("claims") or []) for r in ct)
numeric = sum(1 for r in ct for c in r.get("claims") or [] if re.search(r"\d+(\.\d+)?\s*%|\b\d+\.\d+\b", c.get("claim") or ""))
print("comparison_table rows", rows, "numeric rows", numeric, "tables", len(ct), "surveys", len({r['paper_id'] for r in ct}))
# heldout refs
H = os.path.join(V2, "heldout_refs")
fs = sorted(glob.glob(os.path.join(H, "*.json")))
print("heldout_refs files", len(fs))
sizes = []
for f in fs:
    x = json.load(open(f, encoding="utf-8"))
    n = len(x); ab = sum(1 for p in x if (p.get("abstract") or "").strip())
    sizes.append((os.path.basename(f), n, ab))
print(sizes[:25])
print("ref keys", list(json.load(open(fs[0], encoding="utf-8"))[0].keys()))
G = json.load(open(os.path.join(V2, "survey_gold.json"), encoding="utf-8"))
g = next(iter(G.values()))
print("gold n", len(G), "keys", list(g.keys()), "fam", len(g["families"]), "props", len(g["properties"]), "lims", len(g["limitations"]))
print(" prop sample", json.dumps(g["properties"][0], ensure_ascii=False)[:300])
# do heldout surveys overlap KB?
print("heldout surveys in KB papers:", sum(1 for k in G if k in P))
