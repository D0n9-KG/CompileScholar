"""门槛 B 粗测（相关性用金标当 oracle，只测估计器本身）：
对每题，取种子 top-K 的参考文献表作为 K 次"捕获"，统计每篇金标论文被捕到的次数 → 频次 f1,f2,...
Chao1 估计金标总数 N̂ = S_obs + f1²/(2 f2)（f2=0 时用 f1(f1-1)/2）；估计覆盖 = S_obs/N̂；真实覆盖 = S_obs/|gold|。
报告：估计覆盖 vs 真实覆盖的相关、偏差；并与"天真覆盖=1"（系统默认自认找全）对比。
注：金标=人类重要引用（非全集），故此处只检验"估计器能否跟踪真实覆盖的高低"，不检验绝对校准。"""
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


def chao1(freqs):
    s = len(freqs)
    c = Counter(freqs)
    f1, f2 = c.get(1, 0), c.get(2, 0)
    extra = f1 * f1 / (2 * f2) if f2 > 0 else f1 * (f1 - 1) / 2
    return s + extra


for K in (10, 20, 30):
    est, true = [], []
    for qid, s in S.items():
        if qid not in O:
            continue
        gold = {norm(r["title"]) for r in O[qid]["refs"] if r.get("title")}
        pub = (O[qid].get("published_date") or "")[:10]
        tgt = norm(O[qid]["title"])
        seeds = [x for x in s["rec"] if norm(x.get("title")) != tgt and (not pub or (x.get("publicationDate") or "0") < pub)][:K]
        cnt = Counter()
        for x in seeds:
            for c in (R.get(x.get("paperId")) or {}).get("refs") or []:
                t = norm(c.get("title"))
                if t in gold:
                    cnt[t] += 1
        # 种子本身命中金标也算一次捕获
        for x in seeds:
            t = norm(x.get("title"))
            if t in gold:
                cnt[t] += 1
        sobs = len(cnt)
        if sobs < 2 or not gold:
            continue
        n_hat = chao1(list(cnt.values()))
        est.append(sobs / n_hat)
        true.append(sobs / len(gold))
    r = st.correlation(est, true) if len(est) > 3 else float("nan")
    bias = st.mean(e - t for e, t in zip(est, true))
    print(f"K={K}: n={len(est)} | true cov mean {st.mean(true):.3f} | est cov mean {st.mean(est):.3f} "
          f"| pearson(est,true) {r:.3f} | mean bias {bias:+.3f} | naive 'found everything'=1.0 bias {1 - st.mean(true):+.3f}")
