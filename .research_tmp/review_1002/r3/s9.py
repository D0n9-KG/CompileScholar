import json, sys, os, time, random, collections as C
sys.stdout.reconfigure(encoding="utf-8")
from kbload import load
kb,records,views,man=load()
os.environ["CS2_PPR"]="1"
for q in [dict(entity="transformer"), dict(contains="large language model"), dict(contains="reinforcement learning")]:
    r=kb.findings(**q); p15=[e["paper_id"] for e in r["entries"][:15]]
    print("PPR on", q, "n", r["n"], "top15 prefixes", dict(C.Counter(p.split("_")[0] for p in p15)), "first", p15[:3])
idx=kb._entity_papers()
br=[(len(p),eid) for eid,p in idx.items() if 2<=len(p)<=100 and eid in kb.byid]
br.sort(reverse=True)
print("top40 bridge entities:", [kb.byid[e]["canonical"] for n,e in br[:40]])
random.seed(1); smp=random.sample(br,30); print("random30 bridges:", [kb.byid[e]["canonical"] for n,e in smp])
top50_pairs=sum(n*(n-1)//2 for n,e in br[:50]); allp=sum(n*(n-1)//2 for n,e in br); print("pair share of top50 bridges", round(top50_pairs/allp,3))
single=[e for n,e in br if len(kb.byid[e]["canonical"].split())==1]; print("single-token bridges", len(single), "/", len(br))
# semantic fallback cost
t=time.time(); r=kb.findings(contains="reward hacking in preference optimization"); print("semantic fallback", r.get("n"), r.get("semantic_match"), r.get("returned"), "secs", round(time.time()-t,1))
t=time.time(); r=kb.findings(contains="seismic p-wave picking"); print("semantic fallback2", r.get("n"), r.get("semantic_match"), r.get("returned"), [e.get("semantic_score") for e in r["entries"][:3]], "secs", round(time.time()-t,1))
# carrier coverage
n=0; has=0; haskey=0
for p in records.values():
    for x in p["records"]:
        n+=1; haskey+= ("entity_refs" in x)
        if x.get("entity_refs") or any(isinstance(x.get(f),dict) and x[f].get("entity_id") for f in ("method_ref","from_method_ref","to_method_ref","scope_ref_ref","target_ref_ref")): has+=1
print("runtime records", n, "bound carrier", round(has/n,3), "entity_refs key present", round(haskey/n,3))
