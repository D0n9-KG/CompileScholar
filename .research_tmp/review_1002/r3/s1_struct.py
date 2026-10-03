import json, os, collections
B = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb"
def L(n): return json.load(open(os.path.join(B,n),encoding="utf-8"))
m = L("records_merged.json")
print("merged type", type(m).__name__, "n_keys", len(m))
k0 = next(iter(m)); print("sample key", k0[:80], "payload keys", list(m[k0].keys())[:10] if isinstance(m[k0],dict) else type(m[k0]))
d = L("deep_read_records.json"); print("deep n_keys", len(d)); k1=next(iter(d)); print("deep sample", k1[:80], list(d[k1].keys())[:10])
v = L("views_cs2.json"); print("views keys", list(v.keys())); print("stats", json.dumps(v.get("stats",{}),ensure_ascii=False)[:1500])
r = L("registry_v2.json"); print("registry keys", list(r.keys())); print("n_ent", len(r.get("entities",[])), "surface_index", len(r.get("surface_index",{})))
e0 = r["entities"][0]; print("ent sample keys", list(e0.keys()))
man = L("manifest.json"); mall = L("manifest_all.json"); print("manifest", type(man).__name__, len(man), "manifest_all", type(mall).__name__, len(mall))
print("man sample keys", list(man[0].keys()) if isinstance(man,list) else None)
voc = L("dim_vocab_cs2.json"); print("vocab", voc)
