import json, os, collections as C
B = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb"
def L(n): return json.load(open(os.path.join(B,n),encoding="utf-8"))
m = L("records_merged.json"); d = L("deep_read_records.json")
man = {r["paper_id"]:r for r in L("manifest.json")}
mall = {r["paper_id"]:r for r in L("manifest_all.json")}
def stats(name, db):
    kinds=C.Counter(); ct=C.Counter(); refs_bound=0; refs_total=0; eref_rec=0; n=0
    absn=C.Counter(); resolved=0; about=0; para=0; noid=0
    REFF=("method_ref","from_method_ref","to_method_ref","scope_ref_ref","target_ref_ref")
    rel=C.Counter(); kind_ct=C.Counter()
    for pid,p in db.items():
        for r in p.get("records",[]):
            n+=1; k=r.get("kind"); kinds[k]+=1
            if k in ("finding","method","limitation","survey_claim","domain_snapshot"):
                kind_ct[(k, r.get("claim_type"))]+=1
            if not r.get("id"): noid+=1
            has_bound=False
            for f in REFF:
                ref=r.get(f)
                if isinstance(ref,dict):
                    refs_total+=1
                    if ref.get("entity_id"): refs_bound+=1; has_bound=True
            if any(isinstance(a,dict) and a.get("entity_id") for a in r.get("entity_refs") or []): has_bound=True
            if has_bound: eref_rec+=1
            if k=="absence":
                absn[r.get("absence_type")]+=1
                if r.get("resolved_by"): resolved+=1
            if r.get("about_paper_id"): about+=1
            if r.get("paraphrase_rel"): para+=1
            if k=="lineage": rel[r.get("relation")]+=1
    print(f"== {name}: papers {len(db)} records {n} noid {noid}")
    print(" kinds", kinds.most_common())
    print(" ref slots", refs_total, "bound", refs_bound, f"{refs_bound/max(1,refs_total):.3f}", " records w/ any bound entity", eref_rec, f"{eref_rec/max(1,n):.3f}")
    print(" absence types", dict(absn), "resolved_by", resolved)
    print(" about_paper_id", about, "paraphrase_rel", para)
    print(" lineage relations", rel.most_common())
    tot=C.Counter()
    for (k,c),v in kind_ct.items(): tot[k]+=v
    for k in tot:
        none=kind_ct[(k,None)]
        print(f"  claim_type for {k}: total {tot[k]} None {none} ({none/tot[k]:.2f}) top", [ (c,v) for (kk,c),v in kind_ct.most_common() if kk==k][:6])
stats("merged", m); stats("deep", d)
print("merged pids not in manifest.json", sum(1 for p in m if p not in man), "not in manifest_all", sum(1 for p in m if p not in mall))
print("deep pids not in manifest.json", sum(1 for p in d if p not in man), "in merged", sum(1 for p in d if p in m))
yrs=C.Counter((man[p].get("year") if p in man else "NOMAN") for p in m)
print("year coverage among merged papers (manifest.json):", sum(v for k,v in yrs.items() if isinstance(k,int)), "/", len(m), " missing", yrs.get(None,0), "noman", yrs.get("NOMAN",0))
print("year dist", sorted([(k,v) for k,v in yrs.items() if isinstance(k,int)])[-10:], "min", min(k for k in yrs if isinstance(k,int)))
print("published>cutoff 2025-05-01:", sum(1 for p in m if p in man and str(man[p].get("published") or "")>"2025-05-01"))
