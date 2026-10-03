import json, sys, os, collections as C
sys.stdout.reconfigure(encoding="utf-8")
from kbload import load
kb,records,views,man=load()
os.environ["CS2_PPR"]="0"
def short(r, keys=("n","returned","truncated","semantic_match","note","ppr_reranked")):
    return {k:(str(r.get(k))[:90]) for k in keys if k in r}
qs=[dict(entity="transformer"), dict(contains="contrastive learning"), dict(contains="contrastive learning", claim_type="criticism"),
    dict(contains="reward hacking"), dict(entity="RLHF"), dict(contains="mixture of experts|moe"), dict(entity="lora")]
for q in qs:
    r=kb.findings(**q); top=[(e["paper_id"][:30], e.get("claim_type"), e["claim"][:60]) for e in r["entries"][:3]]
    print("findings",q, short(r)); [print("   ",t) for t in top]
# PPR on vs off
os.environ["CS2_PPR"]="1"
for q in [dict(entity="transformer"), dict(entity="bert", contains="fine-tuning"), dict(contains="diffusion model")]:
    os.environ["CS2_PPR"]="0"; a=kb.findings(**q); os.environ["CS2_PPR"]="1"; b=kb.findings(**q)
    ia=[e["record_id"] for e in a["entries"][:10]]; ib=[e["record_id"] for e in b["entries"][:10]]
    print("PPR",q,"reranked",b.get("ppr_reranked"),"top10 overlap",len(set(ia)&set(ib)),"same order",ia==ib)
    print("   seeds resolved:", [ (n, bool(kb.resolve(n))) for n in [q.get("entity")]+[c for c in str(q.get("contains") or "").split("|") if c]])
for ent in ["transformer","bert","resnet","lora","proximal policy optimization"]:
    r=kb.lineage(entity=ent); print("lineage",ent,"n_edges",r["n_edges"],"anc",len(r["transitive_ancestors"]), "rels",C.Counter(e["relation"] for e in r["edges"]).most_common(4), "ext nodes", sum(1 for e in r["edges"] if str(e.get("to")).startswith("ext:")))
for ent in ["transformer","bert","large language models", None]:
    r=kb.find_gap(entity=ent) if ent else kb.find_gap(subject_family="vision")
    print("find_gap",ent,"extracted",len(r["absences_extracted"]),"derived",len(r["absences_derived"]),"grid",len(r["grid"]),"supply",len(r.get("papers_discussing_entity") or []))
    if r["absences_extracted"]: print("   ex:", r["absences_extracted"][0].get("absence_type"), str(r["absences_extracted"][0].get("subject"))[:40],"|",str(r["absences_extracted"][0].get("missing"))[:80])
for y in (2019, 2022, 2024):
    r=kb.as_of(y); ys=[e["year"] for e in r["genealogy"]["edges"]]
    print("as_of",y,"visible_papers",len(r["visible_papers"]),"edges",len(ys),"max edge year",max(ys) if ys else None,"nodes",len(r["genealogy"]["nodes"]),"matrix",len(r["matrix_tables"]))
r=kb.compare(entities=["bert"]); print("compare bert", r["n"], r.get("vocab_hint",{}).get("entities") if not r["n"] else "")
r=kb.compare(subject="imagenet"); print("compare imagenet", r["n"], "papers", len({x["paper_id"] for x in r["rows"]}))
r=kb.compare(metric="accuracy"); print("compare accuracy", r["n"], "papers", len({x["paper_id"] for x in r["rows"]}), "tables", len({(x['subject'],x['metric']) for x in r['rows']}))
