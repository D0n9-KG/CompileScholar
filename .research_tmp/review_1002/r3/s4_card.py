import json, sys
sys.stdout.reconfigure(encoding="utf-8")
from kbload import load
kb, records, views, man = load()
for ent in ["transformer","bert","BERT","contrastive learning","RLHF","LoRA"]:
    r = kb.card(ent); print(ent, "->", {k:(v if not isinstance(v,list) else len(v)) for k,v in r.items() if k in ("n","error","nearest_in_corpus","next_action","canonical","findings")})
d = kb.describe_kb(); print("describe_kb cards:", d["views"]["cards"], "lineage", d["views"]["lineage"], "coverage", d["views"]["coverage"])
e = kb.entities(contains="transformer", k=5); print("entities in_corpus flags:", [(x["canonical"], x["in_corpus"]) for x in e["entries"]])
print("resolve transformer:", (kb.resolve("transformer") or {}).get("entity_id"), "in flat cards:", (kb.resolve("transformer") or {}).get("entity_id") in views["cards"])
