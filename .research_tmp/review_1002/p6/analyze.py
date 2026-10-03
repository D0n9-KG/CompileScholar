"""汇总 judged/*.json：均值+bootstrap 95% CI（题级重抽样）、vital/okay 分开、配对差、harness 复现一致性。"""
import json, os, random, statistics as st

random.seed(0)


def boot(xs, B=4000):
    n = len(xs)
    ms = sorted(sum(random.choice(xs) for _ in range(n)) / n for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


def load(name):
    p = os.path.join("judged", name + ".json")
    return {r["gt_dir"]: r for r in json.load(open(p, encoding="utf-8"))} if os.path.exists(p) else {}


def cov(r, imp):
    ns = [n for n in r["nuggets"] if n["importance"] == imp]
    return sum(1 for n in ns if n["assignment"] == "support") / len(ns) if ns else None


systems = {k: load(k) for k in ("harness", "B0", "B1")}
common = sorted(set.intersection(*[set(v) for v in systems.values() if v]), key=int)
print(f"common questions: {len(common)}")
rows = []
for name, d in systems.items():
    if not d:
        continue
    xs = [d[i]["nugget_coverage"] for i in common]
    vit = [c for c in (cov(d[i], "vital") for i in common) if c is not None]
    oka = [c for c in (cov(d[i], "okay") for i in common) if c is not None]
    words = [d[i]["words"] for i in common]
    lo, hi = boot(xs)
    rows.append(name)
    print(f"{name:8s} NC={st.mean(xs):.3f} [{lo:.3f},{hi:.3f}]  vital={st.mean(vit):.3f} (n={len(vit)})  "
          f"okay={st.mean(oka):.3f} (n={len(oka)})  partial-credit all={st.mean(d[i]['all_partial'] for i in common):.3f}  "
          f"words med={st.median(words):.0f}  failed_nuggets={sum(d[i]['n_failed'] for i in common)}")

for a, b in (("B0", "harness"), ("B1", "harness"), ("B1", "B0")):
    if systems[a] and systems[b]:
        diff = [systems[a][i]["nugget_coverage"] - systems[b][i]["nugget_coverage"] for i in common]
        lo, hi = boot(diff)
        print(f"paired {a}-{b}: {st.mean(diff):+.3f} [{lo:+.3f},{hi:+.3f}]  wins/ties/losses="
              f"{sum(x > 0 for x in diff)}/{sum(x == 0 for x in diff)}/{sum(x < 0 for x in diff)}")

# harness 复现：本判分器 vs 官方 eval.main 早先跑出的 per-dir 分数（同判分模型，不同运行）
off = json.load(open("harness_official_scores.json"))
pairs = [(systems["harness"][i]["nugget_coverage"], off[i]) for i in common if i in off]
if pairs:
    a = [p[0] for p in pairs]; b = [p[1] for p in pairs]
    d = [x - y for x, y in pairs]
    r = st.correlation(a, b) if len(pairs) > 2 else float("nan")
    print(f"harness repro: n={len(pairs)} ours={st.mean(a):.3f} official-run={st.mean(b):.3f} "
          f"mean|diff|={st.mean(abs(x) for x in d):.3f} pearson={r:.3f}")
