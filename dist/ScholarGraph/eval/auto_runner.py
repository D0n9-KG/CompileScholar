# -*- coding: utf-8 -*-
"""AutoScholarQuery 首测（GOAL 2026-08-29 晚）——官方建议数据集第一位
（bytedance/pasa → SPAR repo 的 test split，1000 题均值 2.4 gold/query）。

目的：量当前系统在"最可能的官方公开测试集形态"上的起点。管线原样
（fast：无②，CONTEST_EXPLORE 默认关；grade_window=150 显式传入，与
SPARBench 基线臂同配置），不改任何参数——cap 适配是小 gold 口径的
待测量，不是待预设。

输出：每题 top-20 排序（带 year/cited），离线算 F1@cap=3/5/10/20。
gold 仅含查询日期之前的论文；pred 的年份+published_time 落盘，供
离线做日期过滤分析（首测不在线过滤，如实记录）。

对照锚（SPAR 论文 Table 1，同基准同集合 F1 口径，已发表）：
  S2+LLM 0.0044 / OA+LLM 0.0045 / Google 0.2015 / G+GPT 0.2683 /
  PaSa 0.2449 / SPAR 0.3843

用法：python .research_tmp/auto_runner.py [--limit 50]
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(_REPO)

from contest.pipeline import recall, grade_and_rank, _norm_title

QS = "_spar_tmp/benchmark/AutoScholarQuery_test.jsonl"
OUT = ".research_tmp/contest_survey/auto_first50.jsonl"
CAPS = [3, 5, 10, 20]


def score_at(pred_titles, gold_titles, k):
    pred = {_norm_title(t) for t in pred_titles[:k] if t}
    gold = {_norm_title(t) for t in gold_titles if t}
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "n_pred": len(pred), "p": round(p, 3),
            "r": round(r, 3), "f1": round(f1, 3)}


def main(limit, offset=0, out_path=OUT):
    lines = open(QS, encoding="utf-8").read().strip().split("\n")[offset:offset + limit]
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(l)["qid"])
            except Exception:
                pass
    outf = open(out_path, "a", encoding="utf-8")
    t0 = time.time()
    for n, line in enumerate(lines):
        d = json.loads(line)
        if d.get("qid") in done:
            continue
        q, gold = d["question"], d["answer"]
        try:
            cands = recall(q)
            kept = grade_and_rank(q, cands, rank="authority", grade_window=150)
            pred = [c.get("title", "") for c in kept]
            meta = [{"title": c.get("title", ""), "year": c.get("year", ""),
                     "cited": c.get("citation_count") or c.get("citationCount") or 0}
                    for c in kept]
        except Exception as e:
            pred, cands, kept, meta = [], [], [], []
            print(f"  [err] {e!r}", flush=True)
        row = {"qid": d.get("qid"), "question": q, "gold": gold,
               "published_time": d.get("source_meta", {}).get("published_time", ""),
               "n_recall": len(cands), "pred": pred, "pred_meta": meta,
               "scores": {str(k): score_at(pred, gold, k) for k in CAPS}}
        outf.write(json.dumps(row, ensure_ascii=False) + "\n")
        outf.flush()
        s5 = row["scores"]["5"]
        print(f"[{n+1}/{len(lines)}] recall={len(cands)} pred={len(pred)} "
              f"F1@5={s5['f1']} (tp={s5['tp']})", flush=True)
    # summary over the first `limit` rows present in OUT
    rows = []
    for l in open(out_path, encoding="utf-8"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        try:
            if int(r["qid"].rsplit("_", 1)[-1]) < limit:
                rows.append(r)
        except Exception:
            pass
    print(f"\n[auto-first{limit}] {len(rows)} queries {time.time()-t0:.0f}s")
    for k in CAPS:
        tp = sum(r["scores"][str(k)]["tp"] for r in rows)
        np_ = sum(r["scores"][str(k)]["n_pred"] for r in rows)
        ng = sum(len({_norm_title(g) for g in r["gold"] if g}) for r in rows)
        mp = tp / np_ if np_ else 0
        mr = tp / ng if ng else 0
        mf1 = 2 * mp * mr / (mp + mr) if (mp + mr) else 0
        print(f"  cap={k:>2}: micro-F1={mf1:.4f} (P={mp:.4f} R={mr:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", type=str, default=OUT)
    a = ap.parse_args()
    main(a.limit, a.offset, a.out)
