"""Step4-6: 归一统计、对比边、网络统计、一致性/缩水/复现率。"""
import json, glob, sys, collections, itertools, statistics
sys.path.insert(0, ".")
from common import P3
from norm import norm_method, norm_dataset, norm_metric

pap = {c["id"]: c for c in json.load(open(P3 + "/s1_candidates.json", encoding="utf-8"))}


def unit(v):
    return v / 100 if v > 1 else v


T = []
n_tab = n_res = n_fail = 0
for f in glob.glob(P3 + "/tuples/*.json"):
    d = json.load(open(f, encoding="utf-8"))
    n_tab += 1
    if not d["raw_ok"]:
        n_fail += 1
        continue
    if d["result"] and d["result"].get("is_result_table"):
        n_res += 1
    T += d["tuples"]
raw_m = {t["method"] for t in T if t["method"]}
raw_d = {t["dataset"] for t in T if t["dataset"]}
for t in T:
    t["m"] = norm_method(t["method"]) if t.get("role") != "ablation" else None
    t["d"] = norm_dataset(t["dataset"])
    t["k"] = norm_metric(t["metric"])
    t["year"] = pap.get(t["paper"], {}).get("year")
S = {"tables_sent": n_tab, "parse_fail": n_fail, "result_tables": n_res, "tuples": len(T),
     "papers_with_tuples": len({t["paper"] for t in T}),
     "methods_raw": len(raw_m), "methods_norm": len({t["m"] for t in T if t["m"]}),
     "datasets_raw": len(raw_d), "datasets_norm": len({t["d"] for t in T if t["d"]}),
     "ablation_tuples": sum(t.get("role") == "ablation" for t in T)}

# 列 = (paper, table, dataset, metric, setting)；列内两两成边
import re as _re
SENS = _re.compile(r"hyper-?param|sensitiv|varying|effect of (the )?(number|length|dimension|size)|impact of|"
                   r"different (values|number|length)|analysis of the (hyper|parameter)", _re.I)
caps = {}
for f in glob.glob(P3 + "/tuples/*.json"):
    d = json.load(open(f, encoding="utf-8"))
    caps[(d["paper"], d["table_id"])] = d["caption"]
S["sensitivity_tables_excluded"] = sum(bool(SENS.search(c)) for c in caps.values())
cols = collections.defaultdict(list)
for t in T:
    if t["m"] and t["d"] and t["k"] and not SENS.search(caps.get((t["paper"], t["table_id"]), "")):
        cols[(t["paper"], t["table_id"], t["d"], t["k"], t["setting"])].append(t)
E = []
for key, rows in cols.items():
    best = {}
    for r in rows:
        best.setdefault(r["m"], r)
    for a, b in itertools.combinations(sorted(best), 2):
        va, vb = best[a]["value"], best[b]["value"]
        if va == vb:
            continue
        hib = best[a].get("higher_is_better", True) is not False
        w = a if (va > vb) == hib else b
        E.append({"paper": key[0], "year": pap.get(key[0], {}).get("year"), "a": a, "b": b, "w": w,
                  "margin": abs(va - vb) / max(abs(va), abs(vb), 1e-9), "d": key[2], "k": key[3]})
S["edges_column_level"] = len(E)

# 论文级聚合：(paper, pair) 多数票
pp = collections.defaultdict(collections.Counter)
for e in E:
    pp[(e["paper"], e["a"], e["b"])][e["w"]] += 1
pair_papers = collections.defaultdict(set)
adj = collections.defaultdict(set)
for (p, a, b), c in pp.items():
    pair_papers[(a, b)].add(p)
    adj[a].add(b)
    adj[b].add(a)
S["edges_paper_pair"] = len(pp)
S["distinct_pairs"] = len(pair_papers)
S["pairs_direct_ge2_papers"] = sum(len(v) >= 2 for v in pair_papers.values())
seen, comps = set(), []
for n in adj:
    if n in seen:
        continue
    st, comp = [n], 0
    while st:
        x = st.pop()
        if x in seen:
            continue
        seen.add(x)
        comp += 1
        st += list(adj[x] - seen)
    comps.append(comp)
S["network_nodes"] = len(adj)
S["lcc_frac"] = round(max(comps) / len(adj), 3) if adj else 0
S["indirect_pairs_1hop"] = sum(1 for a, b in itertools.combinations(sorted(adj), 2)
                               if b not in adj[a] and adj[a] & adj[b])
deg = sorted(((len(v), k) for k, v in adj.items()), reverse=True)
S["top_degree"] = deg[:8]

# 一致性：(baseline 方法, d, k) 跨论文值离散度（每篇取均值）
vals = collections.defaultdict(dict)
for t in T:
    if t["m"] and t["d"] and t["k"] and t.get("role") == "baseline":
        vals[(t["m"], t["d"], t["k"])].setdefault(t["paper"], []).append(unit(t["value"]))
disp = []
for key, byp in vals.items():
    if len(byp) < 3:
        continue
    v = [statistics.mean(x) for x in byp.values()]
    if min(v) <= 0:
        continue
    disp.append({"key": key, "n_papers": len(v), "min": round(min(v), 4), "max": round(max(v), 4),
                 "ratio": round(max(v) / min(v), 2),
                 "cv": round(statistics.pstdev(v) / statistics.mean(v), 3),
                 "papers": sorted(byp)[:6]})
disp.sort(key=lambda x: -x["n_papers"])
S["consistency_keys_ge3"] = len(disp)
if disp:
    r = sorted(x["ratio"] for x in disp)
    cv = sorted(x["cv"] for x in disp)
    S["consistency_ratio_median"] = r[len(r) // 2]
    S["consistency_ratio_p90"] = r[int(len(r) * .9)]
    S["consistency_cv_median"] = cv[len(cv) // 2]

# 缩水：方法在其提出论文(proposed) vs 他人 baseline 的报告值
own = collections.defaultdict(list)
asb = collections.defaultdict(list)
for t in T:
    if not (t["m"] and t["d"] and t["k"]):
        continue
    if t.get("role") == "proposed":
        own[(t["m"], t["d"], t["k"])].append((t["paper"], unit(t["value"])))
    elif t.get("role") == "baseline":
        asb[(t["m"], t["d"], t["k"])].append((t["paper"], unit(t["value"])))
shrink = []
for key in own:
    op = {p for p, _ in own[key]}
    later = [(p, v) for p, v in asb.get(key, []) if p not in op]
    if not later:
        continue
    ov = statistics.mean(v for _, v in own[key])
    lv = statistics.mean(v for _, v in later)
    shrink.append({"key": key, "own_paper": sorted(op)[0], "own": round(ov, 4),
                   "as_baseline_mean": round(lv, 4), "n_later": len({p for p, _ in later}),
                   "rel": round((lv - ov) / ov, 3) if ov else None,
                   "later_papers": sorted({p for p, _ in later})[:4]})
S["shrink_keys"] = len(shrink)
if shrink:
    rr = sorted(x["rel"] for x in shrink if x["rel"] is not None)
    S["shrink_rel_median"] = rr[len(rr) // 2]
    S["shrink_frac_lower"] = round(sum(x < 0 for x in rr) / len(rr), 3)
    S["shrink_methods"] = sorted({x["key"][0] for x in shrink})

# 复现率：P 中 proposed 胜 baseline；后续论文 Q（两者都是 baseline）是否仍同向
role = {}
for t in T:
    if t["m"]:
        role.setdefault((t["paper"], t["m"]), t.get("role"))
res_pp = {}
for k, c in pp.items():
    mc = c.most_common(2)
    if len(mc) == 1 or mc[0][1] > mc[1][1]:
        res_pp[k] = mc[0][0]
rep = []
for (p, a, b), w in res_pp.items():
    if role.get((p, w)) != "proposed":
        continue
    lo = b if w == a else a
    for (q, a2, b2), w2 in res_pp.items():
        if (a2, b2) == (a, b) and q != p and role.get((q, a)) == "baseline" and role.get((q, b)) == "baseline":
            rep.append({"claim_paper": p, "winner": w, "loser": lo, "later_paper": q, "replicated": w2 == w})
S["replication_checks"] = len(rep)
S["replication_rate"] = round(sum(r["replicated"] for r in rep) / len(rep), 3) if rep else None
# 自家胜率：proposed 在本论文列内对 baseline 的胜率
own_w = [e for e in E if role.get((e["paper"], e["a"])) == "proposed" or role.get((e["paper"], e["b"])) == "proposed"]
S["proposed_self_winrate"] = round(sum(role.get((e["paper"], e["w"])) == "proposed" for e in own_w) / len(own_w), 3) if own_w else None

json.dump({"stats": S, "dispersion_top": disp[:20], "shrink": sorted(shrink, key=lambda x: x["rel"] or 0)[:20],
           "replication": rep}, open(P3 + "/s4_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump({"E": E, "T": T}, open(P3 + "/s4_edges.json", "w", encoding="utf-8"), ensure_ascii=False)
for k, v in S.items():
    print(k, v)
