# -*- coding: utf-8 -*-
"""纵向分数对比：direct_scores_ours_batch*.json -> 四 facet 均值表。

global_avg = (ingredient_recall + answer_precision + citation F1) / 3
（官方 CS2 四 facet 的简化口径：citation_recall/precision 以 F1 计）。
"""
import glob
import json
import os
import re
import sys

CS2 = os.path.dirname(os.path.abspath(__file__))


def qavg(scores: dict) -> dict:
    """一题的分数 -> 扁平四值。None facet 计为缺（不进均值，但要报数）。"""
    out = {"ingredient": None, "precision": None, "cit_f1": None}
    ir = scores.get("ingredient_recall")
    if isinstance(ir, dict) and isinstance(ir.get("ingredient_recall"),
                                           (int, float)):
        out["ingredient"] = ir["ingredient_recall"]
    ap = scores.get("answer_precision")
    if isinstance(ap, dict) and isinstance(ap.get("answer_precision"),
                                           (int, float)):
        out["precision"] = ap["answer_precision"]
    cit = scores.get("citation")
    if isinstance(cit, dict) and isinstance(cit.get("f1"), (int, float)):
        out["cit_f1"] = cit["f1"]
    vals = [v for v in out.values() if v is not None]
    out["global"] = (sum(vals) / len(vals)) if vals else None
    return out


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(
            CS2, "direct_scores_ours_batch*.json"))):
        m = re.search(r"batch(\d+)", f)
        tag = f"batch{m.group(1)}" if m else os.path.basename(f)
        d = json.load(open(f, encoding="utf-8"))
        qs = [(qid, qavg(s)) for qid, s in d.items()
              if not (isinstance(s, dict) and s.get("error")
                      and not s.get("ingredient_recall"))]
        n_none = sum(1 for _, q in qs
                     if q["global"] is None)
        means = {}
        for k in ("ingredient", "precision", "cit_f1", "global"):
            vals = [q[k] for _, q in qs if q[k] is not None]
            means[k] = round(sum(vals) / len(vals), 4) if vals else None
        rows.append((tag, len(d), n_none, means))
    print(f"{'batch':<10}{'judged':>7}{'failed':>8}"
          f"{'ingred':>9}{'prec':>8}{'citF1':>8}{'global':>9}")
    for tag, n, nnone, m in rows:
        print(f"{tag:<10}{n:>7}{nnone:>8}"
              f"{str(m['ingredient']):>9}{str(m['precision']):>8}"
              f"{str(m['cit_f1']):>8}{str(m['global']):>9}")
    # harness 参照（同 5 题）
    print("\n参照: harness dev18=0.733 (GLM); ours batch1=0.571 批4=0.623")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
