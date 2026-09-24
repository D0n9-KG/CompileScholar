"""multi_closedbook_recall — P4-A A6: closed-book recall validation.

Hides the 430-paper corpus entirely: for each Multi-108 question, the
external retrieval module (sci-evo SearchService, full path: A3
understanding -> tiered discovery -> A4 rerank) must recall the gold
context papers from the question text ALONE. Gate: Recall@10 >= 0.7
(ROADMAP-v2 A6).

Matching: gold ctxs carry title only (no DOI) — normalized-title match
against retrieved candidates (title prefix/similarity tolerance for
truncated S2/OpenAlex titles).

Usage:
    python multi_closedbook_recall.py [--limit 20] [--mode full]
Output: closedbook_recall.json + per-subject breakdown on stdout.
"""

import argparse
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(
    r"C:\Users\D0n9\Desktop\sci-evo-extract", "src"))
sys.path.insert(0, os.path.join(
    r"C:\Users\D0n9\Desktop\CompileScholar", "src"))

# local GPUStack env for the A3 understand + A4 embed calls
os.environ.setdefault("LOCAL_BASE_URL", "http://192.168.199.73/v1")
os.environ.setdefault(
    "LOCAL_API_KEY",
    "***REMOVED-LOCAL_API_KEY***")
os.environ.setdefault("LOCAL_MODEL", "Qwen3.8-27B")
os.environ.setdefault("LOCAL_EMBED_MODEL", "qwen3-embedding-8b-local")
# Sciverse semantic channel (2026-09-25: it was ABSENT from the first
# A6 run - the token env was never set, so the sciverse tier silently
# blocked; 0.062 was measured WITHOUT the semantic channel)
if not os.environ.get("SCIVERSE_API_TOKEN"):
    _env = os.path.join(
        "C:" + os.sep + os.path.join(*["Users", "D0n9", "Desktop", "CompileScholar"]),
        ".env")
    try:
        for line in open(_env, encoding="utf-8"):
            if line.strip().startswith("SCIVERSE_API_TOKEN"):
                k, _, v = line.strip().partition("=")
                os.environ["SCIVERSE_API_TOKEN"] = v.strip()
    except Exception:
        pass
os.environ.setdefault("LLM_CALL_LOG", os.path.join(
    _HERE, "..", "judge", "ledger_closedbook.jsonl"))
os.environ.setdefault("LLM_RUN_ID", "multi-closedbook")

from sci_evo_extract.library.search_service import SearchService  # noqa: E402


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _title_match(gold_title: str, cand_title: str | None) -> bool:
    """Gold titles are full; candidate titles may truncate. Match on the
    shorter of the two (prefix containment after normalization)."""
    if not cand_title:
        return False
    g, c = _norm_title(gold_title), _norm_title(cand_title)
    if not g or not c:
        return False
    shorter, longer = (g, c) if len(g) <= len(c) else (c, g)
    return shorter in longer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", default="full", choices=["full", "fast", "auto"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(
        _HERE, "..", "..", "scholarqa_multi",
        "closedbook_recall.json"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    qs = json.load(open(os.path.join(
        _HERE, "..", "..", "scholarqa_multi", "data",
        "scholarqa_multi.json"), encoding="utf-8"))
    if args.limit:
        qs = qs[:args.limit]

    svc = SearchService(limit=args.k, doi_cache_path=os.path.join(
        _HERE, "..", "..", "scholarqa_multi", "judge",
        "closedbook_doi_cache.jsonl"))
    # circuit hygiene for long runs: a shared circuit lets a mid-run rate
    # limit (Sciverse 429 at q80) trip sources OPEN for 600s and every
    # later question silently loses that tier — measured: 17/40 early
    # questions had degraded pools vs 26/28 late, 7 with pool=0. Reset
    # per question: recall evaluation measures retrieval capability, not
    # cross-question state carryover.
    from sci_evo_extract.library.circuit import SourceCircuit
    _fresh_circuit = SourceCircuit(failure_threshold=3, window_s=120,
                                   cooldown_s=60)
    svc.circuit = _fresh_circuit
    rows = []
    for i, q in enumerate(qs):
        # fresh circuit per question (see comment above)
        svc.circuit = SourceCircuit(failure_threshold=3, window_s=120,
                                    cooldown_s=60)
        gold = [c.get("title") for c in q.get("ctxs") or []]
        t0 = time.time()
        r = svc.search(q["input"], mode=args.mode, limit=args.k)
        dt = time.time() - t0
        cands = r.candidates[:args.k]
        hit_flags = [
            any(_title_match(g, c.title) for c in cands) for g in gold
        ] if gold else []
        rec = sum(hit_flags) / len(hit_flags) if hit_flags else 0.0
        # pool recall (diagnostic): was gold in the full pre-rerank pool?
        pool = getattr(r, "pool", None) or cands
        pool_flags = [
            any(_title_match(g, c.title) for c in pool) for g in gold
        ] if gold else []
        pool_rec = sum(pool_flags) / len(pool_flags) if pool_flags else 0.0
        rows.append({
            "qid": q["id"], "subject": q.get("subject"),
            "n_gold": len(gold), "n_hit": sum(hit_flags),
            "recall": round(rec, 3), "pool_recall": round(pool_rec, 3),
            "pool_size": len(pool),
            "latency_s": round(dt, 1),
            "mode_used": r.mode_used, "reranked": r.reranked,
            "latency_breakdown": r.latency,
            "subqueries": [s["q"] for s in (r.understood.subqueries
                                            if r.understood else [])],
        })
        print(f"[{i+1}/{len(qs)}] {q['id'][:24]:24s} recall={rec:.2f} "
              f"({sum(hit_flags)}/{len(gold)}) {dt:.1f}s "
              f"mode={r.mode_used}", flush=True)

    overall = sum(r["recall"] for r in rows) / max(1, len(rows))
    pool_overall = sum(r.get("pool_recall", 0.0) for r in rows) / max(1, len(rows))
    by_subj = {}
    for r in rows:
        by_subj.setdefault(r["subject"], []).append(r["recall"])
    subj_means = {s: round(sum(v) / len(v), 3)
                  for s, v in sorted(by_subj.items())}
    summary = {
        "n_questions": len(rows),
        "recall_at_10": round(overall, 4),
        "pool_recall": round(pool_overall, 4),
        "gate": "PASS" if overall >= 0.7 else "FAIL",
        "by_subject": subj_means,
        "mean_latency_s": round(
            sum(r["latency_s"] for r in rows) / max(1, len(rows)), 1),
    }
    out = {"summary": summary, "rows": rows}
    json.dump(out, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
