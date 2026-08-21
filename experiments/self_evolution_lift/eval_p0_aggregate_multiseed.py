# -*- coding: utf-8 -*-
"""P0-2 多seed汇总: 各臂跨seed均值±方差 + paired bootstrap 显著性(arm vs baseline).

用法: 先跑 run_arms_shared_cluster --seeds N --suffix <s> + batch_err, 再跑本脚本.
读 seed{seed}{s}_{arm}_err.json + _psc.json, 汇总.
paired bootstrap: 对 (arm_x, arm_baseline) 每 seed 配对, 重采样 1000 次算差均值置信区间.
"""
import os, sys, json, glob, argparse, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

OUTDIR = os.path.join(os.path.dirname(__file__), "runs", "ARFM2024", "arms_err")
METRICS = ["err_uncond", "err_cond", "fair_recall"]


def load_arm(suffix, arms="TCQB"):
    """{arm: [per-seed dict of metrics]}."""
    out = {a: [] for a in arms}
    for a in arms:
        # seed{N}{suffix}_{arm}_err.json — suffix-aware, exclude _en when suffix=""
        pat = f"seed*{suffix}_{a}_err.json" if suffix else f"seed*_{a}_err.json"
        files = sorted(glob.glob(os.path.join(OUTDIR, pat)))
        if not suffix:
            # exclude seed0_en_T_err.json etc. (other suffixes)
            files = [f for f in files if not any(
                s in os.path.basename(f) for s in ("_en_", "_cn_")) and
                # base arm name must match exactly: seed0_<arm>_err, not seed0_en_<arm>_err
                re.match(rf"seed\d+_{a}_err\.json$", os.path.basename(f))]
        for f in files:
            e = json.load(open(f, encoding='utf-8'))
            fr = e.get("fair_recall") or {}
            pscf = f.replace("_err.json", "_psc.json")
            psc = None
            if os.path.exists(pscf):
                psc = json.load(open(pscf, encoding='utf-8')).get("psc")
            out[a].append({"err_uncond": e["err_uncond"], "err_cond": e["err_cond"],
                           "fair_recall": fr.get("fair_recall", 0.0), "psc": psc,
                           "coverable": e["coverable"], "edge_hit": e["counts"]["edge_hit"]})
    return out


def bootstrap_paired(x, y, n=1000):
    """paired bootstrap on (x_i - y_i); return mean, 2.5%/97.5% CI."""
    x, y = np.array(x), np.array(y)
    if len(x) != len(y) or len(x) == 0:
        return None
    diffs = x - y
    rng = np.random.default_rng(0)
    boots = [rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(n)]
    return float(np.mean(diffs)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="输出后缀 (英文版 _en)")
    ap.add_argument("--baseline", default="T", help="baseline 臂")
    args = ap.parse_args()
    sf = args.suffix
    data = load_arm(sf)
    print(f"\n=== 多seed汇总 (suffix='{sf}') ===")
    print(f"{'arm':<5} {'n':<3} {'ERR_uncond':<16} {'ERR_cond':<16} {'fair_recall':<16} {'PSC':<16}")
    series = {}
    for a, runs in data.items():
        if not runs:
            continue
        series[a] = runs
        def stat(m):
            vals = [r[m] for r in runs if r.get(m) is not None]
            if not vals: return "n/a"
            return f"{np.mean(vals):.3f}±{np.std(vals):.3f}(n={len(vals)})"
        print(f"{a:<5} {len(runs):<3} {stat('err_uncond'):<16} {stat('err_cond'):<16} {stat('fair_recall'):<16} {stat('psc'):<16}")

    bl = args.baseline
    if bl in series and len(series[bl]) >= 2:
        print(f"\n=== paired bootstrap vs baseline {bl} (1000x, 95%CI) ===")
        print(f"{'arm':<5} {'metric':<12} {'mean_diff':<10} {'CI2.5%':<8} {'CI97.5%':<8} {'sig':<4}")
        for a in series:
            if a == bl: continue
            n = min(len(series[a]), len(series[bl]))
            for m in METRICS + ["psc"]:
                x = [series[a][i][m] for i in range(n) if series[a][i].get(m) is not None]
                y = [series[bl][i][m] for i in range(n) if series[bl][i].get(m) is not None]
                if len(x) < 2: continue
                b = bootstrap_paired(x, y)
                if b is None: continue
                md, lo, hi = b
                sig = "yes" if (lo > 0 or hi < 0) else "no"
                print(f"{a:<5} {m:<12} {md:<10.3f} {lo:<8.3f} {hi:<8.3f} {sig:<4}")


if __name__ == "__main__":
    main()
