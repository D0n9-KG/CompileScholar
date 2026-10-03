# -*- coding: utf-8 -*-
"""Read-only audit of base_kb_v2 for DESIGN-CROSSPAPER (10-04). No LLM, no network."""
import json, os, re, sys, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
V2 = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb_v2"
P = json.load(open(os.path.join(V2, "papers.json"), encoding="utf-8"))
R = json.load(open(os.path.join(V2, "records.json"), encoding="utf-8"))
C = collections.Counter
kinds, fields = C(), collections.defaultdict(C)
recs = []
for pid, v in R.items():
    for r in v["records"]:
        recs.append((pid, r))
        kinds[r.get("kind")] += 1
        for k in r:
            fields[r.get("kind")][k] += 1
print("papers", len(P), "papers_with_records", len(R), "records", len(recs))
print("kinds", kinds.most_common())
for k in ("lineage", "result", "absence", "method", "limitation", "finding"):
    print(k, "fields:", [f for f, n in fields[k].most_common(40)])
# provenance sample
pv = next(iter(R.values()))["provenance"]
print("provenance sample:", json.dumps(pv, ensure_ascii=False)[:400])
# records per layer
lay = C();
for pid, r in recs:
    lay[(P.get(pid) or {}).get("layer"), r.get("kind")] += 1
print("by layer x kind", sorted(lay.items()))
# arXiv id coverage
print("papers with arxiv_id", sum(1 for p in P.values() if p.get("arxiv_id")), "doi", sum(1 for p in P.values() if p.get("doi")),
      "abstract", sum(1 for p in P.values() if (p.get("abstract") or "").strip()))
# quotes
q = sum(1 for _, r in recs if (r.get("quote") or "").strip())
print("records with quote", q, "/", len(recs))
# location fields
loc = C(k for _, r in recs for k in r if k in ("location", "loc", "section", "char_span", "span", "page", "anchor"))
print("location-ish fields", loc)
# epistemic
print("epistemic", C(str(r.get("epistemic"))[:30] for _, r in recs).most_common(10))
json.dump({"kinds": kinds}, open(os.path.join(os.path.dirname(__file__), "a1_out.json"), "w"), indent=1)
