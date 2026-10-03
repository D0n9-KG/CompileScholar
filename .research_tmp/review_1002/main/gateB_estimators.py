"""门槛 B 补测：预先固定 5 个标准估计器，全部报告（探索性，不挑选）。
相关性=金标 oracle；捕获=种子 top-K 参考文献表 + 种子自身。
  Chao1；ACE（稀有阈值 10）；一阶/二阶 jackknife（以种子为抽样单元，incidence 形式）；
  Good-Turing 样本覆盖 Ĉ=1-Q1/U·((t-1)Q1/((t-1)Q1+2Q2))（Chao&Jost incidence 版）。
同时报告：真实覆盖的题间方差，以及"被捕总次数""种子有参考文献比例"两个朴素代理的相关，作参照。"""
import json
import re
import statistics as st
from collections import Counter

P8 = "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/review_1002/p8/"
P6 = "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/review_1002/p6/"
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
S = json.load(open(P8 + "seeds.json", encoding="utf-8"))
R = json.load(open(P8 + "seed_refs.json", encoding="utf-8"))
O = {e["qid"]: e for e in json.load(open(P6 + "oracle_inputs.json", encoding="utf-8"))}


def estimators(inc, t):
    """inc: 每个已见物种的出现单元数(incidence)；t: 抽样单元数(种子数)。返回 N̂ 字典。"""
    s = len(inc)
    q = Counter(inc)
    Q1, Q2 = q.get(1, 0), q.get(2, 0)
    U = sum(inc)
    k = (t - 1) / t if t > 1 else 1
    chao2 = s + k * (Q1 * Q1 / (2 * Q2) if Q2 > 0 else Q1 * (Q1 - 1) / 2)
    jk1 = s + Q1 * k
    jk2 = s + (Q1 * (2 * t - 3) / t - Q2 * (t - 2) ** 2 / (t * (t - 1))) if t > 2 else jk1
    rare = [x for x in inc if x <= 10]
    s_rare, s_abund = len(rare), s - len(rare)
    n_rare = sum(rare)
    c_ace = 1 - (q.get(1, 0) / n_rare) if n_rare else 1
    if c_ace <= 0:
        ace = chao2
    else:
        g2 = max(0.0, (s_rare / c_ace) * sum(i * (i - 1) * q.get(i, 0) for i in range(1, 11)) / (n_rare * (n_rare - 1)) - 1) if n_rare > 1 else 0
        ace = s_abund + s_rare / c_ace + Q1 / c_ace * g2
    denom = (t - 1) * Q1 + 2 * Q2
    cov_gt = 1 - (Q1 / U) * (((t - 1) * Q1 / denom) if denom > 0 else 1) if U else 0
    return {"chao2": chao2, "ace": ace, "jack1": jk1, "jack2": max(s, jk2)}, cov_gt


K = 30
rows = []
for qid, sd in S.items():
    if qid not in O:
        continue
    gold = {norm(r["title"]) for r in O[qid]["refs"] if r.get("title")}
    pub = (O[qid].get("published_date") or "")[:10]
    tgt = norm(O[qid]["title"])
    seeds = [x for x in sd["rec"] if norm(x.get("title")) != tgt and (not pub or (x.get("publicationDate") or "0") < pub)][:K]
    units = []
    for x in seeds:
        u = {norm(c.get("title")) for c in (R.get(x.get("paperId")) or {}).get("refs") or []}
        u.add(norm(x.get("title")))
        units.append(u & gold)
    t = len(units)
    inc = Counter(g for u in units for g in u)
    if len(inc) < 2 or not gold:
        continue
    nh, cov_gt = estimators(list(inc.values()), t)
    true_cov = len(inc) / len(gold)
    has_refs = sum(1 for x in seeds if (R.get(x.get("paperId")) or {}).get("refs")) / max(1, t)
    rows.append({"true": true_cov, **{k: len(inc) / v for k, v in nh.items()}, "goodturing": cov_gt,
                 "proxy_total_captures": sum(inc.values()), "proxy_has_refs": has_refs})
print(f"K={K} n={len(rows)} | true coverage mean {st.mean(r['true'] for r in rows):.3f} sd {st.pstdev(r['true'] for r in rows):.3f}")
for k in ("chao2", "ace", "jack1", "jack2", "goodturing", "proxy_total_captures", "proxy_has_refs"):
    x = [r[k] for r in rows]
    y = [r["true"] for r in rows]
    b = st.mean(a - c for a, c in zip(x, y)) if not k.startswith("proxy") else float("nan")
    print(f"  {k:22s} pearson {st.correlation(x, y):+.3f} | mean bias {b:+.3f}")
