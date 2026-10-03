# -*- coding: utf-8 -*-
"""P5 s5: final stats. Manual review overrides (37 stratified cards, see review_cards.md)
+ calibrated layer shares (LLM label counts x reviewed P(true|LLM)), cluster bootstrap CI."""
import json, os, random
from collections import Counter, defaultdict
OUT = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(OUT, "rows.json"), encoding="utf-8"))
sample = [tuple(x) for x in json.load(open(os.path.join(OUT, "review_sample.json")))]
# manual labels for the 37 reviewed cards (C0..C36 order); 'pe' = partial-enumeration flag
MAN = ["a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a",          # C0-C11 (LLM a)
       "c1", "a", "d", "c1", "c1", "c1", "d", "c1",                         # C12-C19 (LLM c1)
       "c2", "a", "c2", "c2", "c1", "c2", "c2", "c1",                       # C20-C27 (LLM c2)
       "a", "c1", "x", "b",                                                 # C28-C31 (LLM b)
       "x", "x", "x", "x",                                                  # C32-C35 (LLM x)
       "d"]                                                                 # C36 (LLM d)
PE = {6, 3, 9, 22, 24, 25, 26}  # cards where some listed items present, others absent
assert len(MAN) == len(sample)
man = {sample[k]: MAN[k] for k in range(len(sample))}
miss = [r for r in rows if r["stable_miss"]]
cov = [r for r in rows if r["stable_cov"]]
L = ["a", "b", "c1", "c2", "d", "x"]
for r in miss:
    r["llm_layer"] = r.get("layer")
    r["final_layer"] = man.get((r["idx"], r["n"]), r["llm_layer"])
rev = [r for r in miss if (r["idx"], r["n"]) in man]
agree = sum(r["llm_layer"] == r["final_layer"] for r in rev)
print(f"review: {len(rev)} cards, LLM-vs-manual agreement {agree}/{len(rev)} = {agree/len(rev):.2f}")
conf = Counter((r["llm_layer"], r["final_layer"]) for r in rev)
print("confusion (llm->manual):", sorted(conf.items()))


def calibrated(miss_rows, rev_rows):
    """share_true[L] = sum_l n_llm[l] * P(true=L | llm=l) / N, P from reviewed stratum."""
    n_llm = Counter(r["llm_layer"] for r in miss_rows)
    P = defaultdict(Counter)
    for r in rev_rows:
        P[r["llm_layer"]][r["final_layer"]] += 1
    out = Counter()
    for l, n in n_llm.items():
        tot = sum(P[l].values())
        if tot == 0:
            out[l] += n
        else:
            for t, c in P[l].items():
                out[t] += n * c / tot
    N = sum(n_llm.values())
    return {k: out[k] / N for k in L}


point = calibrated(miss, rev)
# cluster bootstrap over questions (resample qids) x review stratum bootstrap
random.seed(0)
qids = sorted(set(r["idx"] for r in miss))
byq = defaultdict(list)
for r in miss:
    byq[r["idx"]].append(r)
revby = defaultdict(list)
for r in rev:
    revby[r["llm_layer"]].append(r)
boots = defaultdict(list)
raw_boots = defaultdict(list)
for _ in range(2000):
    qs = [random.choice(qids) for _ in qids]
    mr = [r for q in qs for r in byq[q]]
    rr = [random.choice(v) for k, v in revby.items() for _ in v]
    s = calibrated(mr, rr)
    rc = Counter(r["llm_layer"] for r in mr)
    for k in L:
        boots[k].append(s[k])
        raw_boots[k].append(rc[k] / len(mr))


def ci(v):
    v = sorted(v)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


raw = Counter(r["llm_layer"] for r in miss)
print(f"\nTable1 layer shares over {len(miss)} stable-missed nuggets ({len(qids)} questions)")
print("layer | LLM raw n (%) [95%CI] | calibrated % [95%CI]")
for k in L:
    lo, hi = ci(raw_boots[k]); clo, chi = ci(boots[k])
    print(f"{k:3s} | {raw[k]:3d} ({raw[k]/len(miss):.1%}) [{lo:.1%},{hi:.1%}] | {point[k]:.1%} [{clo:.1%},{chi:.1%}]")
# excluding judge false negatives
nx = 1 - point["x"]
print("calibrated shares among true misses (excl x):", {k: round(point[k] / nx, 3) for k in L if k != "x"})

# Table 2: type x final layer (reviewed -> manual, else LLM)
T = ["single_paper_fact", "cross_paper_comparison", "taxonomy", "temporal_evolution", "target_positioning"]
print("\nTable2 type x layer (missed; final labels)")
print("type | n | " + " ".join(L))
for t in T:
    rs = [r for r in miss if r["type"] == t]
    c = Counter(r["final_layer"] for r in rs)
    print(f"{t} | {len(rs)} | " + " ".join(str(c[k]) for k in L))
print("\ntype distribution: missed vs covered, and per-type miss rate (all 233)")
allc = Counter(r["type"] for r in rows)
mc = Counter(r["type"] for r in miss); cc = Counter(r["type"] for r in cov)
for t in T:
    print(f"{t}: missed {mc[t]} ({mc[t]/len(miss):.0%}) covered {cc[t]} ({cc[t]/len(cov):.0%}) miss-rate {mc[t]/max(1,allc[t]):.0%} of {allc[t]}")
print("\nimportance: missed", Counter(r["imp"] for r in miss), "covered", Counter(r["imp"] for r in cov))
print("vital miss-rate", sum(r["imp"] == "vital" for r in miss) / max(1, sum(r["imp"] == "vital" for r in rows)))
print("\ngold_ref_in_system (LLM) among missed:", Counter(r.get("gold_ref_in_system") for r in miss))
print("info_in_gold_abstract among final=a:", Counter(r.get("info_in_gold_abstract") for r in miss if r["final_layer"] == "a"))
print("storm also misses: overall", sum(r["storm_miss"] for r in miss), "/", len(miss))
for k in L:
    rs = [r for r in miss if r["final_layer"] == k]
    if rs:
        print(f"  layer {k}: storm also misses {sum(r['storm_miss'] for r in rs)}/{len(rs)}")
pe_n = len(PE)
print(f"\npartial-enumeration among reviewed: {pe_n}/{len(rev)}")
json.dump(miss, open(os.path.join(OUT, "missed_final.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
