# -*- coding: utf-8 -*-
"""Multi-108 Track 1（预注册主判据）citation F1——确定性，零 LLM，直接复用 multi_judge_incremental.citation_f1
与 official 引用抽取。对任意臂答案文件计算逐题 F1 与均值，并与指定基线做配对 bootstrap。
用法：python score_multi_f1.py baselines/vnext/answers_vnext_v1.json [--vs baselines/ours/answers_ours.json ...]"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_shared", "tools"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import multi_judge_incremental as J  # noqa: E402
from multi_baseline_common import load_questions  # noqa: E402


def load(path):
    a = json.load(open(path, encoding="utf-8"))
    rows = a if isinstance(a, list) else list(a.values())
    return {r["qid"]: r for r in rows}


def f1s(rows, gold):
    out = {}
    for q, g in gold.items():
        r = rows.get(q) or {}
        ans = r.get("answer_official_all") or r.get("answer_official_first") or ""
        out[q] = J.citation_f1(ans, g["output"])["f1"] if ans else 0.0
    return out


def boot(d, B=4000):
    random.seed(0)
    m = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(B))
    return m[int(.025 * B)], m[int(.975 * B)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("--vs", nargs="*", default=[])
    a = ap.parse_args()
    gold = {q["id"]: q for q in load_questions()}
    me = f1s(load(a.answers), gold)
    n_ans = sum(1 for q in gold if (load(a.answers).get(q) or {}).get("answer_official_all"))
    print(f"{os.path.basename(a.answers)}: answered {n_ans}/{len(gold)} | citation F1 mean {sum(me.values()) / len(me):.4f} (missing=0)")
    for v in a.vs:
        other = f1s(load(v), gold)
        d = [me[q] - other[q] for q in gold]
        lo, hi = boot(d)
        w = sum(x > 0 for x in d); t = sum(x == 0 for x in d); l = sum(x < 0 for x in d)
        print(f"  vs {os.path.basename(v)}: other {sum(other.values()) / len(other):.4f} | paired diff {sum(d) / len(d):+.4f} "
              f"CI[{lo:+.3f},{hi:+.3f}] W/T/L {w}/{t}/{l}")


if __name__ == "__main__":
    main()
