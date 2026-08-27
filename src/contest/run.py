# -*- coding: utf-8 -*-
"""One-shot contest entry point (2026-08-28).

    python -m contest.run "complex academic query" [--mode fast|full]

fast: rewrite -> multi-source recall (S2 bulk + crossref) -> chunked grading
      -> authority-in-prompt ranking. ~30s, <$0.001.
full: fast + graph expansion (if domain graph exists) + snowball (needs S2 key)
      — cost ledger prints per-layer breakdown at the end.
"""
import argparse, json, sys

from .cost_ledger import ledger
from .pipeline import recall, grade_and_rank, rewrite_query


def search(query: str, mode: str = "fast", max_out: int = 15) -> dict:
    """The one-shot API: returns papers + relations + cost report."""
    t_all = ledger.t0 if ledger.events else None
    cands = recall(query)
    kept = grade_and_rank(query, cands, rank="authority")
    out = {
        "query": query,
        "mode": mode,
        "papers": [{"title": c.get("title", ""), "year": c.get("year"),
                    "citations": c.get("citation_count") or c.get("citationCount"),
                    "source": c.get("source_name", "s2")}
                   for c in kept[:max_out]],
        "n_recall": len(cands),
    }
    print(ledger.pretty())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    result = search(a.query, a.mode)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(result, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
