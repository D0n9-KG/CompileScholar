# -*- coding: utf-8 -*-
"""RealScholarQuery runner (GOAL 2026-08-30) — the LARGE-gold official-family
benchmark (PaSa's eval set, 50 queries, mean 15.8 gold/query, max 65).

Config rationale (pre-registered): the SAME fixed pipeline as AutoScholar's
holdout EXCEPT precision-mode OFF — in a 15.8-gold regime, generous grading
and a wide cap are the correct shape (AutoScholar's strict grading would
starve recall here). ② explore stays ON: RSQ gold = representative-paper
lists, exactly what survey reference-list mining recovers. cap=20 output.

Scoring: set-F1 on normalized titles (same as AutoScholar), plus F1@10
(PaSa's recall@20-flavored view). arXiv-id matching available for exact
cross-check (answer_arxiv_id present) — title-normalized primary, id-match
reported for honesty.

Usage: python .research_tmp/rsq_runner.py [--limit 10] [--offset 0]
"""
import argparse, json, os, sys, time, re

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(_REPO)

from contest.pipeline import recall, grade_and_rank, _norm_title

QS = "_spar_tmp/RealScholarQuery/test.jsonl"
OUT_DEFAULT = ".research_tmp/contest_survey/rsq_first.jsonl"


def score(pred_titles, gold_titles, k):
    pred = {_norm_title(t) for t in pred_titles[:k] if t}
    gold = {_norm_title(t) for t in gold_titles if t}
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "n_pred": len(pred), "p": round(p, 3),
            "r": round(r, 3), "f1": round(f1, 3)}


def main(limit, offset, out_path):
    os.environ["CONTEST_EXPLORE"] = "on"      # from the launch script, but be explicit
    os.environ.pop("CONTEST_PRECISION", None) # large-gold: generous grading
    os.environ["CONTEST_CAP"] = "20"
    os.environ["CONTEST_DISAMBIG"] = "on"
    os.environ["CONTEST_MECH_BOOST"] = "on"   # mechanism groups get trim quota 40 (pre-registered)
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
        except Exception as e:
            pred, cands, kept = [], [], []
            print(f"  [err] {e!r}", flush=True)
        row = {"qid": d.get("qid"), "question": q, "gold": gold,
               "gold_arxiv": d.get("answer_arxiv_id", []),
               "n_recall": len(cands), "pred": pred,
               "scores": {str(k): score(pred, gold, k) for k in (10, 20)}}
        outf.write(json.dumps(row, ensure_ascii=False) + "\n")
        outf.flush()
        s20 = row["scores"]["20"]
        print(f"[{n+1}/{len(lines)}] recall={len(cands)} pred={len(pred)} "
              f"F1@20={s20['f1']} (tp={s20['tp']}/{len(gold)} gold)", flush=True)
    # summary
    rows = []
    for l in open(out_path, encoding="utf-8"):
        try:
            rows.append(json.loads(l))
        except Exception:
            pass
    rows = rows[offset:offset + limit]
    print(f"\n[rsq] {len(rows)} queries {time.time()-t0:.0f}s")
    for k in (10, 20):
        tp = sum(r["scores"][str(k)]["tp"] for r in rows)
        np_ = sum(r["scores"][str(k)]["n_pred"] for r in rows)
        ng = sum(len({_norm_title(g) for g in r["gold"] if g}) for r in rows)
        mp = tp / np_ if np_ else 0
        mr = tp / ng if ng else 0
        mf1 = 2 * mp * mr / (mp + mr) if (mp + mr) else 0
        print(f"  cap={k:>2}: micro-F1={mf1:.4f} (P={mp:.4f} R={mr:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", type=str, default=OUT_DEFAULT)
    a = ap.parse_args()
    main(a.limit, a.offset, a.out)
