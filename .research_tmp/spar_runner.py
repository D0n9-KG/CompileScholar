# -*- coding: utf-8 -*-
"""SPARBench practice runner — THIN WRAPPER over contest.pipeline (2026-08-28).
Was a diverged full copy: R9 ran stale code (semantic-output fix never loaded).
Now the pipeline lives ONLY in src/contest/pipeline.py; this file owns just the
benchmark loop + scoring.

Usage: python .research_tmp/spar_runner.py --limit 10 [--offset 0] [--rank authority]
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# resolve the repo root from this file's absolute location (background tasks
# may start in a different cwd — measured FileNotFoundError otherwise)
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(_REPO)

from contest.pipeline import recall, grade_and_rank, _norm_title

QS = "_spar_tmp/benchmark/spar_bench.jsonl"
NL = "\n"


def score(pred_titles, gold_titles):
    pred = {_norm_title(t) for t in pred_titles if t}
    gold = {_norm_title(t) for t in gold_titles if t}
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "n_pred": len(pred), "n_gold": len(gold),
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}


def run(limit, offset, out_path, rank="authority"):
    lines = open(QS, encoding="utf-8").read().strip().split(NL)[offset:offset + limit]
    results = []
    t0 = time.time()
    for n, line in enumerate(lines):
        d = json.loads(line)
        q, gold = d["question"], d["answer"]
        try:
            cands = recall(q)
            kept = grade_and_rank(q, cands, rank=rank)
            pred = [c.get("title", "") for c in kept]
        except Exception as e:
            pred, cands, kept = [], [], []
            print(f"  [err] {e!r}", flush=True)
        s = score(pred, gold)
        results.append({"question": q, "gold": gold, "pred": pred,
                        "n_recall": len(cands), "score": s})
        print(f"[{n+1}/{len(lines)}] recall={len(cands)} pred={len(pred)} "
              f"P={s['precision']} R={s['recall']} F1={s['f1']}", flush=True)
    tp = sum(r["score"]["tp"] for r in results)
    np_ = sum(r["score"]["n_pred"] for r in results)
    ng = sum(r["score"]["n_gold"] for r in results)
    mp = tp / np_ if np_ else 0
    mr = tp / ng if ng else 0
    mf1 = 2 * mp * mr / (mp + mr) if (mp + mr) else 0
    macro = sum(r["score"]["f1"] for r in results) / len(results) if results else 0
    print(f"{NL}[spar-practice rank={rank}] {len(results)} queries {time.time()-t0:.0f}s | "
          f"micro-F1={mf1:.3f} (P={mp:.3f} R={mr:.3f}) | macro-F1={macro:.3f}")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump({"results": results,
                   "summary": {"micro_f1": round(mf1, 4), "micro_p": round(mp, 4),
                               "micro_r": round(mr, 4), "macro_f1": round(macro, 4),
                               "n": len(results), "rank": rank}},
                  open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--rank", choices=["authority", "llm"], default="authority")
    a = ap.parse_args()
    # include the order-mode in the filename: all three ablation arms use
    # --rank authority, so without this the citation arm overwrites the
    # semantic arm's file (hybrid 0.035 was already clobbered-once; backed up).
    _mode = os.environ.get("CONTEST_ORDER", "hybrid")
    run(a.limit, a.offset,
        f".research_tmp/contest_survey/spar_practice_{a.rank}_{_mode}_{a.offset}_{a.limit}.json",
        rank=a.rank)
