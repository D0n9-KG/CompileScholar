# -*- coding: utf-8 -*-
"""R4: termination flag combos, cross-arm paired diffs w/ bootstrap CI on the
common question sets, and power / required-n from replicate SDs."""
import json, os, collections, statistics as st, math, random

CS2 = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2"
ARM = os.path.join(CS2, "arm_ours")
BATCHES = ['31a', '31b', '32a', '32b', '33a', '33b', '34a', '34b', '34c', '34d', '34e']


def qavg(s):
    ir = (s.get("ingredient_recall") or {}).get("ingredient_recall")
    ap = (s.get("answer_precision") or {}).get("answer_precision")
    cf = (s.get("citation") or {}).get("f1")
    v = [x for x in (ir, ap, cf) if isinstance(x, (int, float))]
    return {"G": sum(v) / len(v) if v else None, "IR": ir, "AP": ap, "CT": cf}


def load(p):
    d = json.load(open(p, encoding="utf-8"))
    return {k[:24]: qavg(v) for k, v in d.items() if isinstance(v, dict) and v.get("ingredient_recall")}


# termination combos
combo = collections.Counter()
steps_by = collections.defaultdict(list)
for b in BATCHES:
    for a in json.load(open(os.path.join(ARM, f"answers_pilot_cs2batch{b}.json"), encoding="utf-8")):
        g = a.get("gate") or {}
        key = (bool(g.get("f28_hard_stop")), bool(g.get("forced_notes_submit")),
               bool(g.get("fallback_compiled")), a.get("steps", 0) >= a.get("cap", 36) - 1)
        combo[key] += 1
        steps_by[key].append(a.get("steps"))
print("(f28_hard_stop, forced_submit, fallback_compiled, steps>=cap-1): count, mean steps")
for k, v in combo.most_common():
    print("  ", k, v, round(st.mean(steps_by[k]), 1))

# ours per-question mean over batches, per set
ours = collections.defaultdict(list)
for b in BATCHES:
    for q, s in load(os.path.join(CS2, f"direct_scores_ours_batch{b}_ds.json")).items():
        if s["G"] is not None:
            ours[q].append((b, s))
setA = {q for q, v in ours.items() if v[0][0] in BATCHES[:6]}
setB = {q for q, v in ours.items() if v[0][0] in BATCHES[6:]}
arms = {"harness": load(os.path.join(CS2, "direct_scores_harness_100_ds.json")),
        "gptr": load(os.path.join(CS2, "arm_gptr", "direct_scores_gptr_cs2_ds.json")),
        "perplexity": load(os.path.join(CS2, "direct_scores_perplexity_ds.json"))}
for n, d in arms.items():
    print(n, "judged q:", len(d), "| covers setA", len(setA & set(d)), "setB", len(setB & set(d)))


def boot(diffs, B=10000, seed=0):
    rnd = random.Random(seed)
    m = [st.mean(rnd.choices(diffs, k=len(diffs))) for _ in range(B)]
    m.sort()
    return m[int(.025 * B)], m[int(.975 * B)]


print("\nPaired (ours per-q mean over its batches) - arm, per set")
for sname, qs, bs in (("setA", setA, BATCHES[:6]), ("setB", setB, BATCHES[6:]),
                      ("A+B", setA | setB, BATCHES)):
    for arm, d in arms.items():
        for fac in ("G", "IR", "AP", "CT"):
            diffs = []
            for q in qs:
                if q not in d or d[q][fac] is None:
                    continue
                ov = [s[fac] for b, s in ours[q] if b in bs and s[fac] is not None]
                if ov:
                    diffs.append(st.mean(ov) - d[q][fac])
            if len(diffs) < 3:
                continue
            lo, hi = boot(diffs)
            if fac == "G" or arm != "perplexity":
                print(f"  {sname:5s} ours-{arm:10s} {fac:2s} n={len(diffs):2d} mean={st.mean(diffs):+.3f} "
                      f"95%CI[{lo:+.3f},{hi:+.3f}]")
    # single-run (each batch separately) vs harness G: shows dispersion of single-run comparisons
    single = []
    for b in bs:
        dd = [s["G"] - arms["harness"][q]["G"] for q, v in ours.items() for bb, s in v
              if bb == b and q in arms["harness"] and q in qs]
        if dd:
            single.append(round(st.mean(dd), 3))
    print(f"  {sname} single-batch ours-harness G deltas: {single}")

# ---- power ----
# within-q replicate SD (ours), from r4_stats: G~0.093. Between-arm paired diff SD across questions:
diffs_all = []
for q in setA | setB:
    if q in arms["harness"]:
        diffs_all.append(st.mean(s["G"] for _, s in ours[q]) - arms["harness"][q]["G"])
sd_pair_mean = st.stdev(diffs_all)
# single-run paired-diff SD (one ours run vs one harness run)
sr = []
for q in setA | setB:
    if q in arms["harness"]:
        for _, s in ours[q]:
            sr.append(s["G"] - arms["harness"][q]["G"])
sd_single = st.stdev(sr)
print(f"\nSD of paired diff (ours-q-mean - harness) across q = {sd_pair_mean:.3f} (n={len(diffs_all)}); "
      f"single-run paired SD = {sd_single:.3f}")
z = 1.96 + 0.8416
for sd_lbl, sd in (("single-run", sd_single), ("ours n=2 avg (approx)", math.sqrt(max(0, sd_single**2 - 0.093**2 / 2)))):
    for delta in (0.02, 0.03, 0.05):
        print(f"  [{sd_lbl}] paired-q needed for Δ={delta}: {math.ceil((z*sd/delta)**2)}")
# SE of a 100-question mean
allG = [st.mean(s["G"] for _, s in v) for v in ours.values()]
print(f"between-question SD of ours G (q-means) = {st.stdev(allG):.3f} -> SE(100-q mean, independent) = "
      f"{st.stdev(allG)/10:.3f}; paired SE(100) single-run = {sd_single/10:.3f}")
