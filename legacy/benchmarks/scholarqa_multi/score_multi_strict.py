# -*- coding: utf-8 -*-
"""Multi-108 严格口径 citation F1：把"引用了语料内但不在本题 ctx 集里的论文"（翻译时被丢弃的引用）计为假阳性。
宽松口径（官方/预注册）只看留在答案里的 [ctx] 引用，丢弃的引用不计——对"引用多"的系统有利。两种口径都报。
用法：python score_multi_strict.py"""
import ast
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_shared", "tools"))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import multi_judge_incremental as J  # noqa: E402
import score_multi_f1 as S  # noqa: E402
from multi_baseline_common import load_questions  # noqa: E402

ARMS = {"vnext": "baselines/vnext/answers_vnext_v1.json", "vnext_v2": "baselines/vnext/answers_vnext_v2.json",
        "ours_old": "baselines/ours/answers_ours.json",
        "lightrag": "baselines/lightrag/answers_lightrag.json", "paperqa": "baselines/paperqa/answers_paperqa.json"}


def recs(r):
    c = r.get("citations_all")
    if isinstance(c, str):
        try:
            c = ast.literal_eval(c)
        except Exception:
            c = []
    return c or []


def strict(R, gold):
    out = {}
    for q, g in gold.items():
        r = R.get(q) or {}
        a = r.get("answer_official_all") or ""
        if not a:
            out[q] = 0.0
            continue
        base = J.citation_f1(a, g["output"])
        dropped = {s for x in recs(r) if not x.get("mapped") for s in (x.get("stems") or [])}
        npred = base["n_pred"] + len(dropped)
        p = base["n_correct"] / npred if npred else 0.0
        rr = base["recall"]
        out[q] = 2 * p * rr / (p + rr) if p + rr else 0.0
    return out


def main():
    gold = {q["id"]: q for q in load_questions()}
    res = {}
    for k, p in ARMS.items():
        R = S.load(os.path.join(HERE, p))
        res[k] = (S.f1s(R, gold), strict(R, gold))
        has = sum(1 for r in R.values() if recs(r))
        print(f"{k:9s} lenient F1 {st.mean(res[k][0].values()):.4f} | strict F1 {st.mean(res[k][1].values()):.4f} "
              f"| rows with citation records {has}/{len(R)}")
    # 10 题官方数据错配（bohao_cs_1-10：题面 NLP/HCI、金标光学微腔），另报去除后的 98 题
    q98 = [q for q in gold if not any(q == f"bohao_cs_{i}" for i in range(1, 11))]
    for k in ARMS:
        print(f"{k:9s} 98q lenient {st.mean(res[k][0][q] for q in q98):.4f} | strict {st.mean(res[k][1][q] for q in q98):.4f}")
    for a in ("vnext", "vnext_v2"):
        for b in [x for x in ARMS if x not in ("vnext", "vnext_v2")] + (["vnext"] if a == "vnext_v2" else []):
            for name, ix in (("lenient", 0), ("strict", 1)):
                d = [res[a][ix][q] - res[b][ix][q] for q in gold]
                random.seed(0)
                m = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(4000))
                print(f"  {name:7s} {a}-{b}: {st.mean(d):+.4f} CI[{m[100]:+.3f},{m[3900]:+.3f}]")


if __name__ == "__main__":
    main()
