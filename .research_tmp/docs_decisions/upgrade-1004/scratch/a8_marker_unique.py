# -*- coding: utf-8 -*-
import json, os, re, sys, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
V2 = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb_v2"
R = json.load(open(os.path.join(V2, "records.json"), encoding="utf-8"))
P = json.load(open(os.path.join(V2, "papers.json"), encoding="utf-8"))
NUM = re.compile(r"\[\s*(\d{1,4}(?:\s*[,\-–]\s*\d{1,4})*)\s*\]")
u = set(); per_rec = []
for pid, v in R.items():
    if P[pid]["layer"] != "survey": continue
    for r in v["records"]:
        if r.get("kind") not in ("survey_claim", "lineage", "domain_snapshot"): continue
        n = 0
        for g in NUM.findall(r.get("quote") or ""):
            for tok in re.split(r"\s*,\s*", g):
                if tok.strip().isdigit(): u.add((pid, int(tok))); n += 1
        if n: per_rec.append(n)
print("unique (survey, [n]) pairs referenced by records:", len(u), "records with markers", len(per_rec))
# lineage: family names that are survey method_entry claims_about
me = collections.Counter(r.get("semantic_label") for v in R.values() for r in v["records"] if r.get("kind") == "survey_claim")
print(me)
