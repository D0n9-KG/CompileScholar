# -*- coding: utf-8 -*-
"""P5 s4: preliminary stats + review cards for manual audit (missed nuggets)."""
import json, os, random
from collections import Counter
OUT = os.path.dirname(os.path.abspath(__file__))
B = {b["idx"]: b for b in json.load(open(os.path.join(OUT, "bundles.json"), encoding="utf-8"))}
J = json.load(open(os.path.join(OUT, "layer_judgments.json"), encoding="utf-8"))
A = {f"{a}{r}": json.load(open(os.path.join(OUT, f"assign_{a}_r{r}.json"))) for a in ("harness", "storm") for r in (1, 2)}
rows = []
for idx, v in J.items():
    b = B[idx]
    for j in v["judg"]:
        i = j["n"]
        h1, h2 = A["harness1"][idx][i], A["harness2"][idx][i]
        s1, s2 = A["storm1"][idx][i], A["storm2"][idx][i]
        rows.append(dict(j, idx=idx, qid=b["qid"], text=b["nuggets"][i]["text"], imp=b["nuggets"][i].get("importance"),
                         h1=h1, h2=h2, stable_miss=(h1 != "support" and h2 != "support"),
                         stable_cov=(h1 == "support" and h2 == "support"),
                         storm_miss=(s1 != "support" and s2 != "support")))
json.dump(rows, open(os.path.join(OUT, "rows.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
miss = [r for r in rows if r["stable_miss"]]
cov = [r for r in rows if r["stable_cov"]]
print("stable missed", len(miss), "stable covered", len(cov), "unstable", len(rows) - len(miss) - len(cov))
print("layer (LLM):", Counter(r.get("layer") for r in miss).most_common())
print("type missed:", Counter(r["type"] for r in miss).most_common())
print("type covered:", Counter(r["type"] for r in cov).most_common())
print("gold_ref_in_system:", Counter(r.get("gold_ref_in_system") for r in miss).most_common())
print("topic_in_system:", Counter(r.get("topic_in_system") for r in miss).most_common())
print("info_in_abs:", Counter(r.get("info_in_gold_abstract") for r in miss).most_common())
# review sample: stratified by layer, seed fixed, 36 cards
random.seed(7)
by = {}
for r in miss:
    by.setdefault(r.get("layer"), []).append(r)
sample = []
quota = {"a": 12, "c1": 8, "c2": 8, "b": 4, "x": 4, "d": 4}
for L, q in quota.items():
    pool = by.get(L, [])
    sample += random.sample(pool, min(q, len(pool)))
lines = []
for k, r in enumerate(sample):
    b = B[r["idx"]]
    refs = "; ".join(f'{x}:{b["legend"][x]["title"][:60]}|in_sys={J[r["idx"]]["refmatch"][x]["matched"]}' for x in r.get("source_refs", []) if x in b["legend"])
    lines.append(f"### C{k} idx={r['idx']} n={r['n']} LLM layer={r.get('layer')} type={r['type']} assigner={r['h1']}/{r['h2']} storm_miss={r['storm_miss']}\n"
                 f"NUGGET: {r['text']}\nSRC: {r.get('source_sentence')}\nREFS: {refs}\n"
                 f"gold_in_sys={r.get('gold_ref_in_system')} topic={r.get('topic_in_system')} info_in_abs={r.get('info_in_gold_abstract')}\n"
                 f"SYSQ: {r.get('closest_system_quote')}\nWHY: {r.get('reason')}\n")
open(os.path.join(OUT, "review_cards.md"), "w", encoding="utf-8").write("\n".join(lines))
json.dump([(r["idx"], r["n"]) for r in sample], open(os.path.join(OUT, "review_sample.json"), "w"))
print("cards", len(sample))
