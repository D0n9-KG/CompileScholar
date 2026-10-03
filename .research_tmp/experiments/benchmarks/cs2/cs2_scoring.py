# -*- coding: utf-8 -*-
"""CS2 官方口径汇总（单一真相源，所有臂共用）。

官方定义（astabench/evals/sqa/task.py:388-395, 435）：四个 facet 各占 1/4——
  ingredient_recall, answer_precision, citation_recall, citation_precision
（不是 (IR+AP+F1)/3；旧 score_compare.py 用的是后者，见 REBUILD-PLAN-1003 E6）。

缺题口径：题集 = 该 split 的全部题；某臂缺答 / 判分报错 → 该题四项按 0 计
（官方 inspect 跑法下无答案即 0 分；之前剔除超时题=幸存者偏差，E7）。
"""
import json
import os

CS2 = os.path.dirname(os.path.abspath(__file__))
FACETS = ("ingredient_recall", "answer_precision", "citation_recall", "citation_precision")


def facets(scores) -> dict | None:
    """一题判分 dict → 四 facet；结构缺失返回 None（由调用方按 0 计）。"""
    if not isinstance(scores, dict):
        return None
    try:
        ir = scores["ingredient_recall"]["ingredient_recall"]
        ap = scores["answer_precision"]["answer_precision"]
        cr = scores["citation"]["citation_recall"]
        cp = scores["citation"]["citation_precision"]
    except (KeyError, TypeError):
        return None
    if not all(isinstance(x, (int, float)) for x in (ir, ap, cr, cp)):
        return None
    return {"ingredient_recall": ir, "answer_precision": ap,
            "citation_recall": cr, "citation_precision": cp}


def official_global(f: dict) -> float:
    return sum(f[k] for k in FACETS) / 4


def split_qids(split: str = "dev") -> list[str]:
    rub = {"dev": "sqa2_rubrics_v1_recomputed.json",
           "test": "sqa2_rubrics_v2_recomputed.json"}[split]
    rows = json.load(open(os.path.join(CS2, "..", "scholarqa_multi", rub), encoding="utf-8"))
    return [r["case_id"][:24] for r in rows]


def summarize(score_file: str, qids: list[str]) -> dict:
    """在给定题集上按官方口径汇总；缺答/报错按 0 计并报数。"""
    d = json.load(open(score_file, encoding="utf-8")) if os.path.exists(score_file) else {}
    d = {k[:24]: v for k, v in d.items()}
    per, missing = {}, []
    for q in qids:
        f = facets(d.get(q))
        if f is None:
            missing.append(q)
            f = {k: 0.0 for k in FACETS}
        per[q] = {**f, "global": official_global(f)}
    n = len(qids)
    mean = {k: sum(per[q][k] for q in qids) / n for k in (*FACETS, "global")} if n else {}
    return {"n": n, "n_missing_as_zero": len(missing), "missing": missing, "mean": mean, "per_q": per}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    split = sys.argv[1]
    for f in sys.argv[2:]:
        s = summarize(f, split_qids(split))
        m = s["mean"]
        print(f"{os.path.basename(f):48s} n={s['n']} miss→0={s['n_missing_as_zero']:3d} "
              f"G={m['global']:.3f} IR={m['ingredient_recall']:.3f} AP={m['answer_precision']:.3f} "
              f"CR={m['citation_recall']:.3f} CP={m['citation_precision']:.3f}")
