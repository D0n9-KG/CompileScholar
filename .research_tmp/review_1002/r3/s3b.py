import json, os, collections as C
B = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb"
def L(n): return json.load(open(os.path.join(B,n),encoding="utf-8"))
v = L("views_cs2.json"); reg=L("registry_v2.json")
c=v["cards"]; print("cards type", type(c).__name__, (list(c.keys())[:5] if isinstance(c,dict) else len(c)))
if isinstance(c,dict):
    k=next(iter(c)); print("first key", k, type(c[k]).__name__, (list(c[k].keys())[:20] if isinstance(c[k],dict) else None))
    print("n", len(c))
    vals=[x for x in c.values() if isinstance(x,dict)]
    print(" w/ findings", sum(1 for x in vals if x.get("findings")), "paper_id None", sum(1 for x in vals if not x.get("paper_id")), "year None", sum(1 for x in vals if not x.get("year")))
    print(" avg findings", sum(len(x.get("findings") or []) for x in vals)/max(1,len(vals)))
cov=v["coverage"]; print("coverage grid ents", len(cov["grid"]), "absences_extracted", len(cov["absences_extracted"]), "derived", len(cov["absences_derived"]))
print(" abs resolved_by in view", sum(1 for a in cov["absences_extracted"] if a.get("resolved_by")), "missing_zh", sum(1 for a in cov["absences_extracted"] if a.get("missing_zh")))
print(" abs types view", C.Counter(a.get("absence_type") for a in cov["absences_extracted"]).most_common())
nar=v["narrative"]; print("narrative shifts", len(nar["shifts"]), "family_lines", len(nar["family_lines"]))
ents=reg["entities"]
print("registry version", reg.get("version"), "entity_type", C.Counter(e.get("entity_type") for e in ents).most_common(8))
print(" in_corpus_paper_id set", sum(1 for e in ents if e.get("in_corpus_paper_id")), "origin_year_cited", sum(1 for e in ents if e.get("origin_year_cited")), "mention_papers nonempty", sum(1 for e in ents if e.get("mention_papers")))
print(" provenance", C.Counter(str(e.get("provenance"))[:60] for e in ents).most_common(5))
