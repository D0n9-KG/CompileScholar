import json, sys, os, collections as C
sys.stdout.reconfigure(encoding="utf-8")
from kbload import load
kb,records,views,man=load()
os.environ["CS2_PPR"]="0"
# 1. ordering bias
for q in [dict(entity="transformer"), dict(contains="large language model"), dict(contains="graph neural network")]:
    r=kb.findings(k=100000, **q); allp=[e["paper_id"] for e in r["entries"]]
    r40=kb.findings(**q); p40=[e["paper_id"] for e in r40["entries"]]
    pre=lambda ps: C.Counter(p.split("_")[0] for p in ps)
    yr=lambda ps: C.Counter((man.get(p) or {}).get("year") for p in ps)
    print(q, "n", r["n"], "distinct papers", len(set(allp)), "all prefix", dict(pre(set(allp))), "| returned40 prefix", dict(pre(p40)), "distinct", len(set(p40)))
    print("   first5 pids", p40[:5], "| last pid among all sorted", sorted(set(allp))[-1][:30])
# 2. as_of matrix
T=views["matrix"]["tables"]; mp={c["paper_id"] for ents in T.values() for cells in ents.values() for c in cells}
print("matrix source papers", len(mp), "in manifest.json", sum(1 for p in mp if p in man), sample:=list(mp)[:3])
# 3. semantic fallback
try:
    from kb_infra.embedding import embed_local
    v=embed_local(["reward hacking"]); print("embed_local ok dim", len(v[0]))
except Exception as e: print("embed_local FAIL", repr(e)[:200])
r=kb.findings(contains="reward hacking"); print("reward hacking ->", r.get("n"), r.get("semantic_match"), r.get("returned"), str(r.get("note"))[:60])
# 4. generic entities
idx=kb._entity_papers()
top=sorted(((len(p),eid) for eid,p in idx.items()), reverse=True)[:25]
print("top entities by paper count:", [(kb.byid[e]["canonical"][:28],n) for n,e in top if e in kb.byid])
n_papers=len(records)
generic={e for n,e in top if n>0.05*n_papers}
print("entities >5% papers:", len(generic))
g=views["genealogy"]["edges"]
GEN={"model","models","baseline","baselines","existing literature","method","methods","approach","framework","system","algorithm","network","dataset","data","large language model","deep learning","machine learning","neural network","neural networks","llm","llms","ai","artificial intelligence"}
isgen=lambda name: str(name).strip().lower() in GEN
print("genealogy edges w/ generic endpoint (lexicon):", sum(1 for e in g if isgen(e["from_name"]) or isgen(e["to_name"])), "/", len(g))
bf=[e for l in open(r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb\backflow_edges.jsonl",encoding="utf-8") for e in json.loads(l).get("edges") or []]
print("backflow edges w/ generic from (lexicon):", sum(1 for e in bf if isgen(e["from_name"])), "/", len(bf), " top from_names", C.Counter(e["from_name"] for e in bf).most_common(8))
# bridge entities: entities spanning >=2 papers and <=100
bridges=[eid for eid,p in idx.items() if 2<=len(p)<=100]
gb=[eid for eid in bridges if eid in kb.byid and isgen(kb.byid[eid]["canonical"])]
print("bridge entities", len(bridges), "generic-lexicon among them", len(gb), [kb.byid[e]["canonical"] for e in gb][:10])
pairs=sum(len(idx[e])*(len(idx[e])-1)//2 for e in bridges); print("paper pairs via bridges", pairs)
