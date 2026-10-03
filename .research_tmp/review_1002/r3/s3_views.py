import json, os, collections as C
B = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb"
def L(n): return json.load(open(os.path.join(B,n),encoding="utf-8"))
v = L("views_cs2.json"); m = L("records_merged.json"); reg=L("registry_v2.json")
man = {r["paper_id"]:r for r in L("manifest.json")}
g=v["genealogy"]; nodes=g["nodes"]; edges=g["edges"]
print("nodes", len(nodes), "with year", sum(1 for n in nodes.values() if n.get("year")), "in_corpus", sum(1 for n in nodes.values() if n.get("in_corpus")))
print("edges", len(edges), "with year", sum(1 for e in edges if e.get("year")), "edge keys", sorted(edges[0].keys()))
print("edge provenance", C.Counter(e.get("provenance") for e in edges).most_common())
print("edge relation", C.Counter(e.get("relation") for e in edges).most_common(8))
fe = sum(1 for e in edges if e["from"] in nodes); te=sum(1 for e in edges if e["to"] in nodes)
both=sum(1 for e in edges if e["from"] in nodes and e["to"] in nodes)
print("from is entity_id", fe, "to is entity_id", te, "both", both)
print("edges with from_paper", sum(1 for e in edges if e.get("from_paper")), "to_paper", sum(1 for e in edges if e.get("to_paper")))
cross = [e for e in edges if (e.get("to_paper") and e.get("to_paper")!=e.get("paper_id")) or (e.get("from_paper") and e.get("from_paper")!=e.get("paper_id"))]
print("edges pointing to another paper (via ref paper_id)", len(cross))
# cross-paper via entity: the 'to' entity has records in another paper
idx=C.defaultdict(set)
for pid,p in m.items():
    for r in p["records"]:
        for f in ("method_ref","from_method_ref","to_method_ref","scope_ref_ref","target_ref_ref"):
            ref=r.get(f)
            if isinstance(ref,dict) and ref.get("entity_id"): idx[ref["entity_id"]].add(pid)
        for a in r.get("entity_refs") or []:
            if isinstance(a,dict) and a.get("entity_id"): idx[a["entity_id"]].add(pid)
xp=sum(1 for e in edges if e["to"] in nodes and (idx.get(e["to"],set())-{e["paper_id"]}))
print("edges whose 'to' entity is discussed by >=1 other paper", xp)
print("ancestor_closure", len(g.get("ancestor_closure",{})))
# matrix
T=v["matrix"]["tables"]; multi=0; xpaper=0; xpaper_multi_ent=0
for k,ents in T.items():
    papers={c["paper_id"] for cells in ents.values() for c in cells}
    if len(ents)>=2: multi+=1
    if len(papers)>=2: xpaper+=1
    if len(papers)>=2 and len(ents)>=2: xpaper_multi_ent+=1
src_papers={c["paper_id"] for ents in T.values() for cells in ents.values() for c in cells}
print("matrix tables", len(T), ">=2 entities", multi, ">=2 papers", xpaper, ">=2 papers&>=2 ents", xpaper_multi_ent, "source papers", len(src_papers))
print("pair_deltas", len(v["pair_deltas"]))
cards=v["cards"]["cards"]; print("cards", len(cards))
cn=[len(c.get("findings",[])) for c in cards.values()]; print(" cards w/ >=1 finding", sum(1 for x in cn if x), "cards paper_id None", sum(1 for c in cards.values() if not c.get("paper_id")), "year None", sum(1 for c in cards.values() if not c.get("year")))
ck=C.Counter(); [ck.update(c.keys()) for c in cards.values()]; print(" card keys", ck.most_common())
cov=v["coverage"]; print("coverage grid ents", len(cov["grid"]), "absences_extracted", len(cov["absences_extracted"]), "derived", len(cov["absences_derived"]))
print(" abs resolved_by in view", sum(1 for a in cov["absences_extracted"] if a.get("resolved_by")), "missing_zh", sum(1 for a in cov["absences_extracted"] if a.get("missing_zh")))
print(" abs types view", C.Counter(a.get("absence_type") for a in cov["absences_extracted"]).most_common())
nar=v["narrative"]; print("narrative shifts", len(nar["shifts"]), "family_lines", len(nar["family_lines"]))
# registry
ents=reg["entities"]
print("registry version", reg.get("version"), "entity_type", C.Counter(e.get("entity_type") for e in ents).most_common(8))
print(" in_corpus_paper_id set", sum(1 for e in ents if e.get("in_corpus_paper_id")), "origin_year_cited", sum(1 for e in ents if e.get("origin_year_cited")))
print(" provenance", C.Counter(str(e.get("provenance"))[:40] for e in ents).most_common(5))
