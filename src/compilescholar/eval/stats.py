# -*- coding: utf-8 -*-
"""CS2 配对统计（官方四 facet 口径，缺答/报错按 0 计，复用 cs2_scoring.summarize）。

报：每臂均值（G + 四 facet）、缺答数；指定参照臂对其他臂的逐题配对差：
  均值、百分位 bootstrap 95% CI（B=10,000，seed 0）、符号翻转置换检验 p（双侧，10,000 次）、Holm 校正后的 p。
"ours" 可以是多个运行的逐题均值（--ref 传逗号分隔的多个文件）。

用法：
  python paired_stats.py test --ref direct_scores_vnext_test100_r1_ds.json,direct_scores_vnext_test100_r2_ds.json \
      --others harness=direct_scores_harness_test100_ds.json gptr=direct_scores_gptr_test100_ds.json ...
"""
import argparse
import random
import statistics as st

from .cs2 import scoring as S

KEYS = ("global", *S.FACETS)


def per_q(files: list[str], qids: list[str]) -> dict[str, dict[str, float]]:
    runs = [S.summarize(f, qids)["per_q"] for f in files]
    return {q: {k: sum(r[q][k] for r in runs) / len(runs) for k in KEYS} for q in qids}


def boot_ci(d: list[float], B: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(d)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


def perm_p(d: list[float], B: int = 10000, seed: int = 0) -> float:
    rng = random.Random(seed)
    obs = abs(sum(d))
    hits = sum(1 for _ in range(B) if abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs - 1e-12)
    return (hits + 1) / (B + 1)


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--ref", required=True, help="逗号分隔的分数文件（多个=逐题取均值）")
    ap.add_argument("--others", nargs="+", required=True, help="name=分数文件（逗号分隔则取均值）")
    a = ap.parse_args()
    qids = S.split_qids(a.split)
    ref_files = a.ref.split(",")
    ref = per_q(ref_files, qids)
    arms = {"ours": (ref, ref_files)}
    for spec in a.others:
        name, fs = spec.split("=", 1)
        arms[name] = (per_q(fs.split(","), qids), fs.split(","))

    print(f"split={a.split} n={len(qids)} (缺答按 0 计；判分报错行拒绝汇总，须先重判)")
    print(f"{'arm':10s} " + " ".join(f"{k[:6]:>7s}" for k in KEYS) + "  missing")
    for name, (pq, fs) in arms.items():
        miss = sum(S.summarize(f, qids)["n_missing_as_zero"] for f in fs)
        print(f"{name:10s} " + " ".join(f"{st.mean(pq[q][k] for q in qids):7.3f}" for k in KEYS) + f"  {miss}")

    rows = []
    for name, (pq, _) in arms.items():
        if name == "ours":
            continue
        for k in KEYS:
            d = [ref[q][k] - pq[q][k] for q in qids]
            rows.append((name, k, st.mean(d), *boot_ci(d), perm_p(d)))
    g_idx = [i for i, r in enumerate(rows) if r[1] == "global"]
    adj = holm([rows[i][5] for i in g_idx])
    adj_map = dict(zip(g_idx, adj))
    print("\nours − arm（逐题配对；Holm 只在 global 一族内校正）")
    for i, (name, k, m, lo, hi, p) in enumerate(rows):
        h = f"  holm p={adj_map[i]:.4f}" if i in adj_map else ""
        print(f"  vs {name:10s} {k:20s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  perm p={p:.4f}{h}")

    if len(ref_files) >= 2:
        r1, r2 = (per_q([f], qids) for f in ref_files[:2])
        d = [r1[q]["global"] - r2[q]["global"] for q in qids]
        x = [r1[q]["global"] for q in qids]
        y = [r2[q]["global"] for q in qids]
        mx, my = st.mean(x), st.mean(y)
        r = sum((a - mx) * (b - my) for a, b in zip(x, y)) / ((sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5)
        lo, hi = boot_ci(d)
        print(f"\nrun1 − run2 global {st.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]; per-question Pearson r = {r:.3f}; "
              f"|Δ| mean {st.mean(abs(v) for v in d):.3f}")


if __name__ == "__main__":
    main()
