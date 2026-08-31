# -*- coding: utf-8 -*-
"""Paired offline A/B: authority-rerank vs llm-order on IDENTICAL grade output
(grader run-to-run variance measured at q2: 0.0 vs 0.148 — unpaired runs can't
isolate the ranker effect). Reads _grade_archive.jsonl, scores both arms."""
import json, math, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ARCHIVE = ".research_tmp/contest_survey/_grade_archive.jsonl"
QS = "_spar_tmp/benchmark/spar_bench.jsonl"


def _norm_title(s):
    return re.sub(r"[^a-z]", "", (s or "").lower())


def score(pred_titles, gold_titles):
    pred = {_norm_title(t) for t in pred_titles if t}
    gold = {_norm_title(t) for t in gold_titles if t}
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return tp, p, r, f1


def rerank_authority(high, some, cited, max_out=12):
    def _ck(c):
        return math.log10(1 + int(cited.get(c, 0) or 0))
    pool = high + some
    pool_max = max((_ck(c) for c in pool), default=1.0) or 1.0
    def _sc(c, w):
        return w * (0.65 + 0.35 * _ck(c) / pool_max)
    ranked = sorted([(_sc(c, 1.0), c) for c in high] + [(_sc(c, 0.55), c) for c in some],
                    key=lambda t: -t[0])
    return [c for _, c in ranked[:max_out]]


def main():
    golds = {}
    for line in open(QS, encoding="utf-8").read().strip().split("\n"):
        d = json.loads(line)
        golds[_norm_title(d["question"])] = d["answer"]

    # latest archive entry per query wins
    latest = {}
    for line in open(ARCHIVE, encoding="utf-8").read().strip().split("\n"):
        d = json.loads(line)
        latest[d["query"]] = d

    rows = []
    for query, d in latest.items():
        gold = golds.get(_norm_title(query)) or golds.get(query)
        if not gold:
            continue
        high, some = d["H"], d["S"]
        cited = dict(zip(d["titles"], d["cited"]))
        llm_pred = (high + some)[:12]
        auth_pred = rerank_authority(high, some, cited)
        s_llm = score(llm_pred, gold)
        s_auth = score(auth_pred, gold)
        rows.append((query, s_llm, s_auth))

    if not rows:
        print("no archived grades matched benchmark queries yet")
        return
    tp_l = sum(r[1][0] for r in rows); tp_a = sum(r[2][0] for r in rows)
    print(f"{'query':<50s} {'llm F1':>7s} {'auth F1':>8s}")
    for q, sl, sa in rows:
        print(f"{q[:48]:<50s} {sl[3]:>7.3f} {sa[3]:>8.3f}")
    print(f"\npaired micro-tp: llm={tp_l} auth={tp_a} over {len(rows)} queries")


if __name__ == "__main__":
    main()
