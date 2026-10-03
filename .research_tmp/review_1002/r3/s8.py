import json, sys, os, collections as C
sys.stdout.reconfigure(encoding="utf-8")
from kbload import load
kb,records,views,man=load()
B=r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb"
d=json.load(open(os.path.join(B,"deep_read_records.json"),encoding="utf-8"))
deep_ids={r["id"] for p in d.values() for r in p["records"]}
vrec=set()
for e in views["genealogy"]["edges"]: vrec.add(e.get("record_id"))
for t in views["matrix"]["tables"].values():
    for cells in t.values():
        for c in cells: vrec.add(c.get("record_id"))
for a in views["coverage"]["absences_extracted"]: vrec.add(a.get("record_id"))
print("deep record ids", len(deep_ids), "appearing in views genealogy/matrix/coverage", len(deep_ids & vrec))
# claim_type purity
os.environ["CS2_PPR"]="0"
for q in [dict(contains="contrastive learning", claim_type="criticism"), dict(contains="transformer", claim_type="criticism"), dict(contains="large language model", claim_type="criticism|limitation")]:
    r=kb.findings(k=100000, **q); ct=C.Counter(e.get("claim_type") for e in r["entries"])
    r40=kb.findings(**q); ct40=C.Counter(e.get("claim_type") for e in r40["entries"][:15])
    print(q, "all", dict(ct), "| first15 shown", dict(ct40))
# manifest gaps
miss=[p for p in records if p not in man]; print("records papers not in manifest.json", len(miss), C.Counter(p.split("_")[0] for p in miss).most_common())
# lineage edge sources
src=C.Counter(e["paper_id"].split("_")[0] for e in views["genealogy"]["edges"]); print("lineage edges by source prefix", src.most_common())
kinds_by_src=C.Counter()
for pid,p in records.items():
    for r in p["records"]:
        if r.get("kind")=="lineage": kinds_by_src[(pid.split("_")[0], r.get("provenance") or (p.get("provenance") if isinstance(p,dict) else None))]+=1
print("merged payload provenance sample", C.Counter(str(p.get("provenance"))[:40] for p in json.load(open(os.path.join(B,"records_merged.json"),encoding="utf-8")).values()).most_common(6))
# backflow-attached degree of runtime-grown entity resolution
print("KBTools.byid size", len(kb.byid))
