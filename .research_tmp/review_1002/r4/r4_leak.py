# -*- coding: utf-8 -*-
"""R4 leakage: share of final-answer citations landing on demand_core papers
(KB built from dev100 question-text Sciverse top-10) and on the question's OWN
rehearsal hits; ours (31a-34e) vs harness / GPTR / perplexity on same qids.
Matching = normalized title (lowercase alnum, first 80 chars)."""
import json, os, re, collections, statistics as st, math, random

CS2 = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2"
BASE = os.path.join(CS2, "base_kb")


def nt(s):
    s = re.sub(r"\[.*?\]", " ", str(s or "").lower())
    return re.sub(r"[^a-z0-9]+", "", s)[:80]


man = json.load(open(os.path.join(BASE, "manifest_all.json"), encoding="utf-8"))
demand = {nt(m.get("title")) for m in man if m.get("primary_category") == "demand_core" and m.get("title")}
demand.discard("")
reh = json.load(open(os.path.join(BASE, "demand_rehearsal.json"), encoding="utf-8"))
own = {r["case_id"][:24]: {nt(h.get("title")) for h in r.get("hits") or [] if h.get("title")} for r in reh}
merged = json.load(open(os.path.join(BASE, "records_merged.json"), encoding="utf-8"))
mtitle = {m["paper_id"]: m.get("title") for m in man}
in_kb_demand = {nt(mtitle.get(p)) for p in merged if nt(mtitle.get(p)) in demand}
print("demand_core titles", len(demand), "| in records_merged", len(in_kb_demand))


def qavg(s):
    ir = (s.get("ingredient_recall") or {}).get("ingredient_recall")
    ap = (s.get("answer_precision") or {}).get("answer_precision")
    cf = (s.get("citation") or {}).get("f1")
    v = [x for x in (ir, ap, cf) if isinstance(x, (int, float))]
    return {"G": sum(v) / len(v) if v else None, "IR": ir, "CT": cf}


def load_scores(p):
    d = json.load(open(p, encoding="utf-8"))
    return {k[:24]: qavg(v) for k, v in d.items() if isinstance(v, dict) and v.get("ingredient_recall")}


def cit_titles(item):
    out = []
    for sec in item.get("sections") or []:
        for c in sec.get("citations") or []:
            t = c.get("title") or (c.get("metadata") or {}).get("title")
            if t:
                out.append(nt(t))
    return out


def arm_stats(judge_path, qfilter=None):
    J = json.load(open(judge_path, encoding="utf-8"))
    per = {}
    for it in J:
        q = it["qid"][:24]
        if qfilter and q not in qfilter:
            continue
        ts = cit_titles(it)
        uniq = set(ts)
        per[q] = {"n": len(ts), "uniq": len(uniq),
                  "demand": sum(1 for t in ts if t in demand),
                  "own": sum(1 for t in ts if t in own.get(q, set())),
                  "own_uniq": len(uniq & own.get(q, set())),
                  "demand_uniq": len(uniq & demand)}
    return per


def summarize(name, per):
    n = sum(v["n"] for v in per.values())
    d = sum(v["demand"] for v in per.values())
    o = sum(v["own"] for v in per.values())
    qd = sum(1 for v in per.values() if v["own"] > 0)
    print(f"{name:14s} q={len(per):3d} cites={n:5d} demand={100*d/max(1,n):5.1f}% "
          f"own-rehearsal={100*o/max(1,n):5.1f}% | q with >=1 own-hit cite={qd}/{len(per)} "
          f"| mean own_uniq papers/q={st.mean([v['own_uniq'] for v in per.values()]) if per else 0:.2f}")
    return {"q": len(per), "cites": n, "demand_share": d / max(1, n), "own_share": o / max(1, n),
            "q_with_own": qd, "own_uniq_mean": st.mean([v['own_uniq'] for v in per.values()]) if per else 0}


out = {"ours": {}, "cross": {}}
BATCHES = ['31a', '31b', '32a', '32b', '33a', '33b', '34a', '34b', '34c', '34d', '34e']
rows = []
for b in BATCHES:
    per = arm_stats(os.path.join(CS2, f"judge_input_ours_batch{b}.json"))
    out["ours"][b] = summarize("ours_" + b, per)
    S = load_scores(os.path.join(CS2, f"direct_scores_ours_batch{b}_ds.json"))
    for q, v in per.items():
        if q in S and S[q]["G"] is not None:
            rows.append({"b": b, "q": q, **v, **S[q]})

# (b) score difference: rows with >=1 own-hit citation vs none; also within-q
print("\n(b) own-rehearsal-cite vs none (all ours rows)")
for fac in ("G", "IR", "CT"):
    a = [r[fac] for r in rows if r["own"] > 0 and r[fac] is not None]
    z = [r[fac] for r in rows if r["own"] == 0 and r[fac] is not None]
    print(f"  {fac}: with={st.mean(a):.3f} (n={len(a)})  without={st.mean(z):.3f} (n={len(z)})")
# within-question: same q, runs with more own-hit cites vs fewer
byq = collections.defaultdict(list)
for r in rows:
    byq[r["q"]].append(r)
num = den = 0
dx, dy = [], []
for q, rs in byq.items():
    if len(rs) < 2:
        continue
    mo = st.mean(r["own_uniq"] for r in rs)
    mg = st.mean(r["G"] for r in rs)
    for r in rs:
        dx.append(r["own_uniq"] - mo)
        dy.append(r["G"] - mg)
cov = sum(a * b for a, b in zip(dx, dy)) / len(dx)
vx = sum(a * a for a in dx) / len(dx)
vy = sum(b * b for b in dy) / len(dy)
print(f"  within-question corr(own_uniq papers, G) = {cov/math.sqrt(vx*vy):.3f}; slope={cov/vx:.4f} G per extra own-hit paper")
# between-question: question-level mean own_uniq vs mean G
qm = [(st.mean(r["own_uniq"] for r in rs), st.mean(r["G"] for r in rs)) for rs in byq.values()]
mx, my = st.mean(a for a, _ in qm), st.mean(b for _, b in qm)
c = sum((a - mx) * (b - my) for a, b in qm)
print(f"  between-question corr = {c/math.sqrt(sum((a-mx)**2 for a,_ in qm)*sum((b-my)**2 for _,b in qm)):.3f} (n_q={len(qm)})")

# (c) cross-arm on same qid sets
setA = set(r["q"] for r in rows if r["b"] in BATCHES[:6])
setB = set(r["q"] for r in rows if r["b"] in BATCHES[6:])
for name, p in [("harness", os.path.join(CS2, "judge_input_harness_100.json")),
                ("gptr", os.path.join(CS2, "arm_gptr", "judge_input_gptr_cs2.json")),
                ("perplexity", os.path.join(CS2, "judge_input_perplexity_dev.json"))]:
    for sname, qs in (("setA", setA), ("setB", setB), ("all100", None)):
        per = arm_stats(p, qs)
        out["cross"][f"{name}|{sname}"] = summarize(f"{name}_{sname}", per)
for sname, bs in (("setA", BATCHES[:6]), ("setB", BATCHES[6:])):
    agg = collections.Counter()
    for b in bs:
        for k in ("cites",):
            pass
    n = sum(out["ours"][b]["cites"] for b in bs)
    o = sum(out["ours"][b]["own_share"] * out["ours"][b]["cites"] for b in bs)
    d = sum(out["ours"][b]["demand_share"] * out["ours"][b]["cites"] for b in bs)
    print(f"ours_{sname} pooled: demand={100*d/n:.1f}% own={100*o/n:.1f}%")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "r4_leak_out.json"), "w"), indent=1)
