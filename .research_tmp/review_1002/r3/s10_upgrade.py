import json, sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")
sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\_shared\tools")
from kb_compiler.records.backflow import build_backflow
from external_tools import _record_mentions
B=r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb"
reg=json.load(open(os.path.join(B,"registry_v2.json"),encoding="utf-8"))
d=json.load(open(os.path.join(B,"deep_read_records.json"),encoding="utf-8"))
tot_prod={"deep":0,"coarse":0}; tot_full={"deep":0,"coarse":0}
for pid,p in d.items():
    recs=p["records"]
    prod={"records":[{"id":r.get("id"),"kind":r.get("kind"),"subject":r.get("subject"),"claim":r.get("claim"),"quote":r.get("quote"),"mentions":_record_mentions(r)} for r in recs]}
    full={"records":[{**r,"mentions":_record_mentions(r)} for r in recs]}
    for nm,pl,tot in (("prod",prod,tot_prod),("full",full,tot_full)):
        bf=build_backflow(pl,{"title":pid,"year":None},reg)
        for e in bf["edges"]: tot[e["provenance"]]+=1
print("production payload shape -> edges by provenance", tot_prod)
print("full record payload     -> edges by provenance", tot_full)
