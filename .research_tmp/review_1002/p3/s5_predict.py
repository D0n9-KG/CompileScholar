"""Step7: 迷你时间剖分预测。训练网=≤2022 论文的受控对比；测试=2023-2024 论文中
首次直接对比、且 ≤2022 无直接边但经 1 个中介可达的方法对（论文内多数票为真值）。"""
import json, sys, math, collections, itertools, statistics, re
import concurrent.futures as cf
sys.path.insert(0, ".")
from common import P3, local27b, deepseek, parse_json, DS_CALLS

D = json.load(open(P3 + "/s4_edges.json", encoding="utf-8"))
E, T = D["E"], D["T"]
CUT = 2022


def paper_votes(edges):
    pp = collections.defaultdict(collections.Counter)
    for e in edges:
        pp[(e["paper"], e["a"], e["b"])][e["w"]] += 1
    out = []
    for (p, a, b), c in pp.items():
        mc = c.most_common(2)
        if len(mc) == 2 and mc[0][1] == mc[1][1]:
            continue
        out.append((p, a, b, mc[0][0]))
    return out


tr = paper_votes([e for e in E if e["year"] and e["year"] <= CUT])
te = paper_votes([e for e in E if e["year"] and e["year"] > CUT])
adj = collections.defaultdict(set)
wins = collections.Counter()   # (winner, loser) 计数（论文级）
for p, a, b, w in tr:
    lo = b if w == a else a
    adj[a].add(b); adj[b].add(a)
    wins[(w, lo)] += 1
direct = {(a, b) for _, a, b, _ in tr}

# 测试对：同对多篇 2023+ 论文 → 多数票；需 ≤2022 无直接边 & 有 1 跳中介
tv = collections.defaultdict(collections.Counter)
tpapers = collections.defaultdict(set)
for p, a, b, w in te:
    tv[(a, b)][w] += 1
    tpapers[(a, b)].add(p)
test = []
for (a, b), c in tv.items():
    if (a, b) in direct or a not in adj or b not in adj or not (adj[a] & adj[b]):
        continue
    mc = c.most_common(2)
    if len(mc) == 2 and mc[0][1] == mc[1][1]:
        continue
    test.append({"a": a, "b": b, "truth": mc[0][0], "papers": sorted(tpapers[(a, b)])})

# Bradley-Terry（MM 算法，带弱先验：每对加 0.5 虚拟平局）
nodes = sorted(adj)
s = {n: 1.0 for n in nodes}
W = collections.Counter()
N = collections.Counter()
for (w, lo), c in wins.items():
    W[w] += c
    N[tuple(sorted((w, lo)))] += c
for n in nodes:
    W[n] += 0.5 * len(adj[n])
for _ in range(300):
    new = {}
    for i in nodes:
        den = 0.0
        for j in adj[i]:
            nij = N[tuple(sorted((i, j)))] + 1.0
            den += nij / (s[i] + s[j])
        new[i] = W[i] / den if den else s[i]
    g = math.exp(statistics.mean(math.log(v) for v in new.values()))
    s = {k: v / g for k, v in new.items()}


def bt(a, b):
    return a if s[a] > s[b] else b


def mediator(a, b):
    sc = 0
    for m in adj[a] & adj[b]:
        sc += (wins[(a, m)] - wins[(m, a)]) - (wins[(b, m)] - wins[(m, b)])
    return None if sc == 0 else (a if sc > 0 else b)


# 绝对值基线：≤2022 论文中两方法在共同 (d,k) 的跨论文平均报告值
absv = collections.defaultdict(list)
for t in T:
    if t.get("m") and t.get("d") and t.get("k") and t.get("year") and t["year"] <= CUT:
        v = t["value"] / 100 if t["value"] > 1 else t["value"]
        absv[(t["m"], t["d"], t["k"])].append(v)


def absolute(a, b):
    va, vb = 0, 0
    for (m, d, k), v in absv.items():
        if m != a:
            continue
        if (b, d, k) in absv:
            x, y = statistics.mean(v), statistics.mean(absv[(b, d, k)])
            if x != y:
                va += x > y
                vb += y > x
    return None if va == vb else (a if va > vb else b)


# 新近性基线：方法首次出现年份（样本内 proposed 年份优先，否则最早出现）
first = {}
for t in sorted(T, key=lambda t: (t.get("year") or 9999)):
    m = t.get("m")
    if not m:
        continue
    if t.get("role") == "proposed":
        first.setdefault(("p", m), t["year"])
    first.setdefault(("a", m), t["year"])


def recency(a, b):
    ya = first.get(("p", a), first.get(("a", a)))
    yb = first.get(("p", b), first.get(("a", b)))
    if ya is None or yb is None or ya == yb:
        return None
    return a if ya > yb else b


PR = ("In sequential recommendation, two methods are named '{a}' and '{b}'. Based only on your prior knowledge "
      "of these methods, which one typically achieves better top-K ranking accuracy (e.g. NDCG@10/HR@10) on "
      "standard benchmarks like MovieLens-1M or Amazon Beauty when both are evaluated under the same protocol? "
      'Answer JSON only: {{"winner": "<exact name>", "confidence": 0-1}}')


def llm(fn, a, b):
    try:
        j = parse_json(fn(PR.format(a=a, b=b)) if fn is local27b else fn(PR.format(a=a, b=b), max_tokens=1500))
        w = str((j or {}).get("winner", "")).lower().replace(" ", "")
        if w == a.replace(" ", ""):
            return a
        if w == b.replace(" ", ""):
            return b
        if a in w and b not in w:
            return a
        if b in w and a not in w:
            return b
    except Exception:
        pass
    return None


def wilson(k, n):
    if n == 0:
        return (None, None)
    z = 1.96
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(c - h, 3), round(c + h, 3))


with cf.ThreadPoolExecutor(8) as ex:
    p27 = list(ex.map(lambda x: llm(local27b, x["a"], x["b"]), test))
with cf.ThreadPoolExecutor(4) as ex:
    pds = list(ex.map(lambda x: llm(deepseek, x["a"], x["b"]), test))
preds = {"bradley_terry": [bt(x["a"], x["b"]) for x in test],
         "mediator_vote": [mediator(x["a"], x["b"]) for x in test],
         "absolute_values": [absolute(x["a"], x["b"]) for x in test],
         "recency(newer wins)": [recency(x["a"], x["b"]) for x in test],
         "27B_zero_shot": p27, "DeepSeek_zero_shot": pds}
res = {}
for k, pv in preds.items():
    cov = [(p, x["truth"]) for p, x in zip(pv, test) if p is not None]
    hit = sum(p == t for p, t in cov)
    res[k] = {"n_covered": len(cov), "coverage": round(len(cov) / len(test), 3) if test else None,
              "acc": round(hit / len(cov), 3) if cov else None, "ci95": wilson(hit, len(cov))}
# BT 在 中介投票也能判的子集上（同子集比较）
both = [i for i, x in enumerate(test) if preds["mediator_vote"][i] is not None and p27[i] is not None]
sub = {k: round(sum(preds[k][i] == test[i]["truth"] for i in both) / len(both), 3) if both else None
       for k in ("bradley_terry", "mediator_vote", "27B_zero_shot", "DeepSeek_zero_shot")}
out = {"train_paper_edges": len(tr), "train_nodes": len(nodes), "test_paper_edges": len(te),
       "n_test_pairs": len(test), "results": res, "same_subset_n": len(both), "same_subset_acc": sub,
       "ds_calls": DS_CALLS["n"], "test": [dict(x, **{k: preds[k][i] for k in preds}) for i, x in enumerate(test)]}
json.dump(out, open(P3 + "/s5_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "test"}, ensure_ascii=False, indent=1))
