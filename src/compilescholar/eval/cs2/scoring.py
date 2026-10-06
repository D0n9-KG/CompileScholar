# -*- coding: utf-8 -*-
"""CS2 官方口径汇总（单一真相源，所有臂共用）。

官方定义（astabench/evals/sqa/task.py:388-395, 435）：四个 facet 各占 1/4——
  ingredient_recall, answer_precision, citation_recall, citation_precision
（不是 (IR+AP+F1)/3；旧 score_compare.py 用的是后者，见 REBUILD-PLAN-1003 E6）。

缺题口径：题集 = 该 split 的全部题；某臂缺答 → 该题四项按 0 计（官方 inspect 跑法下无答案即 0 分；之前剔除超时题=幸存者偏差，E7）。
判分报错的行不计 0、拒绝汇总（W1-1，10-04）：判分器故障与系统没答出是两回事，必须重判到 0 错误行。
"""
import json
import os

from ...core import paths

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


RUBRIC_FILES = {"dev": "sqa2_rubrics_v1_recomputed.json", "test": "sqa2_rubrics_v2_recomputed.json"}


def rubric_path(split: str) -> str:
    """Official split -> rubric file: dev = rubrics v1, test = rubrics v2 (astabench task.py:444-451)."""
    return str(paths.benchmarks("cs2") / "rubrics" / RUBRIC_FILES[split])


def split_qids(split: str = "dev") -> list[str]:
    rows = json.load(open(rubric_path(split), encoding="utf-8"))
    return [r["case_id"][:24] for r in rows]


class JudgeErrorRows(RuntimeError):
    """Some questions have judge-error rows: re-judge them (they are never scored as 0)."""


def state(scores) -> str:
    """answered: all four facets present | missing: no row (the system gave no answer -> 0) |
    judge_error: a row exists but the judge failed (error / _errors, or facets missing)."""
    if scores is None:
        return "missing"
    if facets(scores) is not None:
        return "answered"
    return "judge_error"


def summarize(score_file: str, qids: list[str], allow_judge_errors: bool = False) -> dict:
    """Official aggregate over the given question set. A question with no row scores 0 (missing answer).
    W1-1 (2026-10-04): a judge-error row used to score 0 as well, mixing judge failures with system failures; it now
    raises JudgeErrorRows unless allow_judge_errors=True (then it scores 0 and is reported separately). A score file
    that does not exist raises FileNotFoundError (it used to silently score every question 0)."""
    if not os.path.exists(score_file):
        raise FileNotFoundError(score_file)
    d = json.load(open(score_file, encoding="utf-8"))
    d = {k[:24]: v for k, v in d.items()}
    per, missing, errors = {}, [], []
    for q in qids:
        st = state(d.get(q))
        if st == "missing":
            missing.append(q)
        elif st == "judge_error":
            errors.append(q)
        f = facets(d.get(q)) or {k: 0.0 for k in FACETS}
        per[q] = {**f, "global": official_global(f)}
    if errors and not allow_judge_errors:
        raise JudgeErrorRows(f"{score_file}: {len(errors)} judge-error rows, re-judge first: {errors[:5]}")
    n = len(qids)
    mean = {k: sum(per[q][k] for q in qids) / n for k in (*FACETS, "global")} if n else {}
    return {"n": n, "n_missing_as_zero": len(missing), "missing": missing, "n_judge_error": len(errors),
            "judge_error": errors, "mean": mean, "per_q": per}


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
