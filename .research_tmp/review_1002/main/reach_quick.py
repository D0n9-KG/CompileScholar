"""门槛 A 粗估（独立于 P8，只读其缓存）：语义/推荐种子 top-N 直接命中金标 vs 种子一跳参考文献按频次排序 top-N 命中金标。
金标匹配=标题归一精确匹配（保守，低估两者，但对两者同口径）。种子须早于目标论文。"""
import json
import random
import re
from collections import defaultdict

P8 = "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/review_1002/p8/"
P6 = "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/review_1002/p6/"
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
S = json.load(open(P8 + "seeds.json", encoding="utf-8"))
R = json.load(open(P8 + "seed_refs.json", encoding="utf-8"))
O = {e["qid"]: e for e in json.load(open(P6 + "oracle_inputs.json", encoding="utf-8"))}
NS = (30, 50, 100, 200)
res = {f"sem@{n}": [] for n in NS} | {f"cite@{n}": [] for n in NS} | {"cite_union_K30": []}
cov_refs = []
for qid, s in S.items():
    if qid not in O:
        continue
    gold = {norm(r["title"]) for r in O[qid]["refs"] if r.get("title")}
    if not gold:
        continue
    pub = (O[qid].get("published_date") or "")[:10]
    tgt = norm(O[qid]["title"])
    seeds = [x for x in s["rec"] if norm(x.get("title")) != tgt and (not pub or (x.get("publicationDate") or "0") < pub)]
    sem_titles = [norm(x.get("title")) for x in seeds]
    for n in NS:
        res[f"sem@{n}"].append(len(gold & set(sem_titles[:n])) / len(gold))
    freq = defaultdict(int)
    got = 0
    for x in seeds[:30]:
        refs = (R.get(x.get("paperId")) or {}).get("refs") or []
        got += bool(refs)
        for c in refs:
            t = norm(c.get("title"))
            if t and t != tgt:
                freq[t] += 1
    cov_refs.append(got / max(1, min(30, len(seeds))))
    order = sorted(freq, key=lambda t: -freq[t])
    for n in NS:
        res[f"cite@{n}"].append(len(gold & set(order[:n])) / len(gold))
    res["cite_union_K30"].append(len(gold & set(order)) / len(gold))


def boot(x, B=3000):
    random.seed(0)
    m = sorted(sum(random.choice(x) for _ in x) / len(x) for _ in range(B))
    return m[int(.025 * B)], m[int(.975 * B)]


print(f"questions={len(res['sem@30'])}  seeds-with-refs fraction (top30) mean={sum(cov_refs)/len(cov_refs):.2f}")
for k, v in res.items():
    lo, hi = boot(v)
    print(f"{k:16s} recall {sum(v)/len(v):.3f} [{lo:.3f},{hi:.3f}]")
