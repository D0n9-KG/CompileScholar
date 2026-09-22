# -*- coding: utf-8 -*-
"""Multi-108 incremental judge loop (user directive 2026-09-23: answer one,
judge one — never wait for a full arm to finish).

Track 1 tonight: Citation F1 (deterministic, official extract_citations
verbatim, primary prereg criterion). Track 2 (AutoAIS) needs the local NLI
model on GPU — deferred while the inference server owns the GPU. Track 3
(rubric): VERIFIED no official rubric exists for Multi-108 (asta-bench HF
dataset ships rubrics_v1/v2_recomputed.json for the SQA track only) —
awaiting user ruling on ingredient generation, disclosed in the morning
report.

Per question q and arm a: preds = unique official [N] markers extracted from
answer_official_all; gold = unique [N] from the official gold output
(the official evaluator derives gold_ctxs exactly this way). Score rows
append to answers/<arm>.scores.jsonl (resume-safe, keyed by qid).

Usage: python multi_judge_incremental.py [--once] [--interval 120]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_MULTI = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "scholarqa_multi")
ARMS = {
    "ours": os.path.join(_MULTI, "baselines", "ours", "answers_ours.json"),
    "lightrag": os.path.join(_MULTI, "baselines", "lightrag", "answers_lightrag.json"),
    "paperqa": os.path.join(_MULTI, "baselines", "paperqa", "answers_paperqa.json"),
}
QFILE = os.path.join(_MULTI, "data", "scholarqa_multi.json")

# ---- official extract_citations (verbatim from asta-bench
# astabench/evals/sqa/citation_eval.py; the [0] / [2,3] zero-based form) ----
_citation_pattern = re.compile(r'\[(\d+(?:,\s*\d+)*)\]')


def extract_citations(text):
    matches = _citation_pattern.findall(text)
    citations = []
    for match in matches:
        citations.extend([int(num.strip()) for num in match.split(',')])
    return citations


def citation_f1(answer_official: str, gold_output: str) -> dict:
    preds = sorted(set(extract_citations(answer_official or "")))
    gold = sorted(set(extract_citations(gold_output or "")))
    n_pred, n_gold = len(preds), len(gold)
    inter = len(set(preds) & set(gold))
    prec = inter / n_pred if n_pred else 0.0
    rec = inter / n_gold if n_gold else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"n_pred": n_pred, "n_gold": n_gold, "n_correct": inter,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4)}


def load_gold() -> dict:
    qs = json.load(open(QFILE, encoding="utf-8"))
    return {q["id"]: q for q in qs}


def judge_arm(arm: str, path: str, gold: dict) -> list[str]:
    """Score new answer rows; append to <arm>.scores.jsonl. Returns qids
    scored this round."""
    out_path = path.rsplit(".", 1)[0] + ".scores.jsonl"
    if not os.path.exists(path):
        return []
    scored = set()
    if os.path.exists(out_path):
        for l in open(out_path, encoding="utf-8"):
            l = l.strip()
            if l:
                try:
                    scored.add(json.loads(l)["qid"])
                except Exception:
                    pass
    rows = json.load(open(path, encoding="utf-8"))
    newly = []
    for r in rows:
        qid = r.get("qid")
        # score ONLY substantively-answered rows: an err string (including the
        # EMPTY string that str(TimeoutError()) produces) or an empty official
        # answer means a system failure, not a real zero — scoring it would
        # poison the paired stats (2026-09-23 incident: 11 timed-out PaperQA
        # rows scored as F1=0)
        if not qid or qid in scored:
            continue
        if r.get("err") is not None and r.get("err") != "":
            continue
        ans = (r.get("answer_official_all")
               or r.get("answer_official_first") or "")
        if not ans.strip():
            continue
        g = gold.get(qid)
        if g is None:
            continue  # unknown qid — do not score
        s = citation_f1(ans, g.get("output") or "")
        newly.append({"qid": qid, "arm": arm, "policy": "all",
                      "n_cited_pids": len(r.get("cited_pids") or []),
                      "citation_dropped": r.get("citation_dropped"),
                      "citation_mapped": r.get("citation_mapped"),
                      "wall_s": r.get("wall_s"), **s})
    if newly:
        with open(out_path, "a", encoding="utf-8") as f:
            for row in newly:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return [r["qid"] for r in newly]


def summarize() -> dict:
    """Arm-level aggregates over scored rows (paired stats come later in
    Stage D once all arms complete)."""
    out = {}
    for arm, path in ARMS.items():
        sp = path.rsplit(".", 1)[0] + ".scores.jsonl"
        rows = []
        if os.path.exists(sp):
            for l in open(sp, encoding="utf-8"):
                l = l.strip()
                if l:
                    try:
                        rows.append(json.loads(l))
                    except Exception:
                        pass
        if rows:
            f1 = [r["f1"] for r in rows]
            prec = [r["precision"] for r in rows]
            rec = [r["recall"] for r in rows]
            out[arm] = {"n": len(rows),
                        "f1_mean": round(sum(f1) / len(f1), 4),
                        "prec_mean": round(sum(prec) / len(prec), 4),
                        "rec_mean": round(sum(rec) / len(rec), 4),
                        "zero_cite_rows": sum(1 for r in rows if r["n_pred"] == 0)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=120)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    gold = load_gold()
    print(f"[judge] gold loaded: {len(gold)} questions; arms: "
          f"{', '.join(ARMS)}", flush=True)
    while True:
        total_new = 0
        for arm, path in ARMS.items():
            try:
                new = judge_arm(arm, path, gold)
            except json.JSONDecodeError as e:
                print(f"[judge] {arm}: answers file mid-write ({e}) — retry "
                      f"next cycle", flush=True)
                continue
            if new:
                total_new += len(new)
                s = summarize().get(arm) or {}
                print(f"[judge] {arm}: +{len(new)} scored "
                      f"(n={s.get('n')} f1={s.get('f1_mean')})", flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
