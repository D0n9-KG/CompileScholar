# -*- coding: utf-8 -*-
"""Multi-108 Stage D: paired-difference statistics for the prereg primary
criterion (MULTI-PREREG.md §6):

  主判据 = Citation F1 配对差 ours vs 各自跑对照（PaperQA2 / LightRAG），
  paired bootstrap 95% CI 下界 > 0 且点估计优势成立。

Reads the three arms' *.scores.jsonl (written incrementally by
multi_judge_incremental.py), pairs by qid, reports per-arm means and the two
paired contrasts on the COMMON question set (complete when all arms have
scored the same 108 qids; partial runs report the current intersection with
its size disclosed).

Usage: python multi_paired_stats.py [--boot 10000] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_MULTI = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "scholarqa_multi")
SCORES = {
    "ours": os.path.join(_MULTI, "baselines", "ours", "answers_ours.scores.jsonl"),
    "lightrag": os.path.join(_MULTI, "baselines", "lightrag", "answers_lightrag.scores.jsonl"),
    "paperqa": os.path.join(_MULTI, "baselines", "paperqa", "answers_paperqa.scores.jsonl"),
}


def load_scores(path: str) -> dict[str, float]:
    out = {}
    if not os.path.exists(path):
        return out
    for l in open(path, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except Exception:
            continue
        # policy=all rows only (first-policy rows are a separate record; the
        # prereg primary is scored on the all-policy translation)
        if r.get("policy") == "all" and r.get("qid"):
            out[r["qid"]] = r["f1"]   # last write wins (rescored rows)
    return out


def paired_bootstrap(diffs: list[float], n_boot: int = 10000, seed: int = 7):
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo, hi = means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]
    return round(lo, 4), round(hi, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    arm_scores = {a: load_scores(p) for a, p in SCORES.items()}
    for a, s in arm_scores.items():
        print(f"{a}: {len(s)} scored", flush=True)

    report = {"n_scored": {a: len(s) for a, s in arm_scores.items()},
              "contrasts": {}, "complete": False}
    n_common_all = set.intersection(*[set(s) for s in arm_scores.values()]) \
        if all(arm_scores.values()) else set()
    report["complete"] = len(n_common_all) == 108

    for baseline in ("lightrag", "paperqa"):
        common = sorted(set(arm_scores["ours"]) & set(arm_scores[baseline]))
        if not common:
            report["contrasts"][baseline] = {"n_pairs": 0}
            continue
        diffs = [arm_scores["ours"][q] - arm_scores[baseline][q] for q in common]
        mean_d = sum(diffs) / len(diffs)
        lo, hi = paired_bootstrap(diffs, args.boot)
        wins = sum(1 for d in diffs if d > 0)
        losses = sum(1 for d in diffs if d < 0)
        ties = len(diffs) - wins - losses
        verdict = ("PASS" if (lo > 0 and mean_d > 0)
                   else "FAIL" if mean_d <= 0 else "INCONCLUSIVE (CI spans 0)")
        c = {"n_pairs": len(common), "mean_diff": round(mean_d, 4),
             "ci95": [lo, hi], "wins": wins, "ties": ties, "losses": losses,
             "ours_mean": round(sum(arm_scores['ours'][q] for q in common) / len(common), 4),
             "baseline_mean": round(sum(arm_scores[baseline][q] for q in common) / len(common), 4),
             "prereg_verdict": verdict}
        report["contrasts"][baseline] = c
        print(f"\nours vs {baseline} (n={len(common)}):")
        print(f"  ours={c['ours_mean']}  {baseline}={c['baseline_mean']}")
        print(f"  paired diff mean={c['mean_diff']}  CI95=[{lo},{hi}]  "
              f"W/T/L={wins}/{ties}/{losses}")
        print(f"  prereg verdict: {verdict}")
    if not report["complete"]:
        print("\n(partial — common-set stats only, not the frozen verdict)",
              flush=True)

    if args.out:
        json.dump(report, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
