# -*- coding: utf-8 -*-
"""0.6 step 1: dev-rehearsal retrieval for the CS2 hot-start base KB.

CS2 dev 100 questions x Sciverse semantic search (top 10 each) ->
demand-core paper pool (title/abstract-ish/year/doi). Raw responses
archived for the prereg. Test split is NEVER touched.

Usage: python rehearsal_dev100.py
Output: base_kb/demand_rehearsal.json (+ demand_pool.json)
"""
import json
import re
import sys
import time
from pathlib import Path

CS_REPO = Path(r"C:\Users\D0n9\Desktop\CompileScholar")
SEE_SRC = Path(r"C:\Users\D0n9\Desktop\sci-evo-extract\src")
for p in (str(CS_REPO / "src"), str(SEE_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sci_evo_extract.library.sources import SciverseClient  # noqa: E402

RUBRICS = (CS_REPO / ".research_tmp/experiments/benchmarks/scholarqa_multi/"
           "sqa2_rubrics_v1_recomputed.json")
OUT_DIR = CS_REPO / ".research_tmp/experiments/benchmarks/cs2/base_kb"
K = 10


def _token() -> str | None:
    for line in (CS_REPO / ".env").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*SCIVERSE_API_TOKEN\s*=\s*(\S+)", line)
        if m:
            return m.group(1)
    return None


def main():
    import os
    tok = _token()
    assert tok, "SCIVERSE_API_TOKEN not found in CompileScholar/.env"
    os.environ["SCIVERSE_API_TOKEN"] = tok  # request_json reads env only
    client = SciverseClient(token=tok)

    questions = json.load(open(RUBRICS, encoding="utf-8"))
    print(f"[rehearsal] {len(questions)} dev questions", flush=True)

    rehearsal = []
    pool: dict[str, dict] = {}   # key: title-lower | doi
    for i, q in enumerate(questions):
        t0 = time.time()
        try:
            cands = client.semantic_search(q["question"], limit=K)
        except Exception as e:
            cands = []
            print(f"  q{i}: SEARCH ERROR {str(e)[:100]}", flush=True)
        hits = []
        for c in cands:
            if c.status != "ready":
                continue
            raw = c.raw or {}
            year = c.year
            hit = {
                "title": c.title,
                "year": year,
                "doi": None,  # agentic-search payload carries no DOI
                "doc_id": raw.get("doc_id"),
                "matched_text": (raw.get("chunk") or raw.get("abstract")
                                 or "")[:600],
            }
            hits.append(hit)
            key = (c.title or "").strip().lower()[:120] or raw.get("doc_id")
            if key and key not in pool:
                pool[key] = dict(hit)
        rehearsal.append({
            "case_id": q["case_id"],
            "question": q["question"],
            "n_hits": len(hits),
            "hits": hits,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(questions)}] pool={len(pool)} "
                  f"last={time.time()-t0:.1f}s", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(rehearsal, open(OUT_DIR / "demand_rehearsal.json", "w",
                              encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(pool, open(OUT_DIR / "demand_pool.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    n_year = sum(1 for p in pool.values() if p.get("year"))
    after_cutoff = sum(1 for p in pool.values()
                       if (p.get("year") or 0) > 2025)
    print(f"[rehearsal] pool={len(pool)} unique papers | "
          f"year known: {n_year} | >2025 (to filter): {after_cutoff}",
          flush=True)


if __name__ == "__main__":
    main()
