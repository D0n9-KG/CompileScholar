import json, os, sys
sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")
B = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb"
ARM = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\arm_ours"
def load():
    from kb_compiler.views.tools import KBTools
    L=lambda n: json.load(open(os.path.join(B,n),encoding="utf-8"))
    records={k:v for k,v in L("records_merged.json").items() if isinstance(v,dict) and v.get("records")}
    for pid,p in L("deep_read_records.json").items():
        base=records.get(pid,{}).get("records") or []
        seen={r.get("id") for r in base}
        records[pid]={"records":list(base)+[r for r in p["records"] if r.get("id") not in seen]}
    views=L("views_cs2.json"); reg=L("registry_v2.json"); voc=L("dim_vocab_cs2.json")
    man={r["paper_id"]:r for r in L("manifest.json")}
    return KBTools(views,reg,voc,man,records,emb_cache_path=os.path.join(ARM,"emb_cache_records.bin")), records, views, man
