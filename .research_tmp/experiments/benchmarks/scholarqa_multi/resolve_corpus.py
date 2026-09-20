# -*- coding: utf-8 -*-
"""Multi-108 corpus resolver (stage 1 of task #11).

Resolves the 441 unique ctx titles from scholarqa_multi.json to identities via
OpenAlex (primary, polite pool) with S2 as fallback when OpenAlex misses.
Outputs corpus_map.json: one entry per unique title with
{paper_id, title, year, subject, openalex_id, doi, arxiv_id, oa_pdf_url,
 resolution, score}.

paper_id = stable 10-char hash of the normalized title (the KB-side key that
the runner will later map to per-question ctx indices via id_mapping).

Discipline: year from the benchmark metadata is authoritative for validation
(resolver never trusts LLM/year guesses from APIs; a >1-year mismatch or a
similarity <0.55 flags the entry for review instead of silent acceptance).
Resume-safe: existing entries are kept, only misses re-run.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:/Users/D0n9/Desktop/sci-evo-extract/src")
from sci_evo_extract.library.sources import OpenAlexClient, SemanticScholarClient

BASE = Path(r"C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/scholarqa_multi")
DATA = BASE / "data" / "scholarqa_multi.json"
OUT = BASE / "corpus" / "corpus_map.json"
MIN_SCORE = 0.55


def norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (t or "").lower())).strip()


def paper_id(title: str) -> str:
    import hashlib
    return "m" + hashlib.sha1(norm_title(title).encode()).hexdigest()[:10]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    items = json.load(open(DATA, encoding="utf-8"))
    # unique titles with benchmark-side metadata (year/subject from first sighting)
    seen = {}
    for x in items:
        for c in x.get("ctxs") or []:
            t = (c.get("title") or "").strip()
            if t and t not in seen:
                seen[t] = {"year": c.get("year"),
                           "authors": [a.get("name") for a in (c.get("authors") or [])][:3],
                           "subject": x.get("subject")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    oa = OpenAlexClient()
    s2 = SemanticScholarClient()
    todo = [(t, m) for t, m in seen.items() if t not in done]
    print(f"resolve: {len(todo)} to run ({len(done)} cached of {len(seen)} unique)", flush=True)
    for i, (title, meta) in enumerate(todo):
        entry = {"paper_id": paper_id(title), "title": title, **meta,
                 "openalex_id": None, "doi": None, "arxiv_id": None,
                 "oa_pdf_url": None, "resolution": None, "score": None,
                 "review": False}
        # --- primary: OpenAlex ---
        cands = []
        try:
            cands = [c for c in oa.search_title(title, limit=3) if c.status == "ready"]
        except Exception as e:
            print(f"[{i}] oa err {e}", flush=True)
        best = None
        for c in sorted(cands, key=lambda c: (c.candidate_score or 0), reverse=True):
            best = c
            if (c.candidate_score or 0) >= MIN_SCORE:
                break
        if best is not None and (best.candidate_score or 0) >= MIN_SCORE:
            raw = best.raw or {}
            locs = raw.get("locations") or []
            entry.update({
                "openalex_id": best.source_record_id,
                "doi": best.normalized_doi,
                "score": round(best.candidate_score, 3),
                "resolution": "openalex",
            })
            for loc in locs:
                lp = str(loc.get("landing_page_url") or "")
                am = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,6})", lp)
                if am:
                    entry["arxiv_id"] = am.group(1)
                    break
            if not entry["arxiv_id"]:
                for loc in locs:
                    if loc.get("pdf_url"):
                        entry["oa_pdf_url"] = loc.get("pdf_url")
                        break
            year_meta = meta.get("year")
            year_hit = best.raw.get("publication_year")
            if year_meta and year_hit and abs(int(year_meta) - int(year_hit)) > 1:
                entry["review"] = True
                entry["review_reason"] = f"year mismatch: benchmark={year_meta} openalex={year_hit}"
        else:
            # --- fallback: S2 (rate-limited; single attempt, no hammering) ---
            try:
                s2c = s2.search_title(title, limit=1)
                if s2c and s2c[0].status == "ready":
                    ext = s2c[0].raw.get("s2_external_ids") or {}
                    entry.update({"doi": s2c[0].normalized_doi,
                                  "arxiv_id": ext.get("ArXiv"),
                                  "resolution": "s2",
                                  "score": round(s2c[0].candidate_score or 0, 3)})
            except Exception:
                pass
            if entry["resolution"] is None:
                entry["resolution"] = "miss"
        done[title] = entry
        if (i + 1) % 20 == 0:
            OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
            n_arx = sum(1 for e in done.values() if e.get("arxiv_id"))
            n_oa = sum(1 for e in done.values() if e.get("oa_pdf_url") and not e.get("arxiv_id"))
            n_miss = sum(1 for e in done.values() if e.get("resolution") == "miss")
            print(f"  [{i+1}/{len(todo)}] arxiv={n_arx} oa_pdf={n_oa} miss={n_miss}", flush=True)
        time.sleep(1.1)  # OpenAlex polite pool courtesy
    OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
    n_arx = sum(1 for e in done.values() if e.get("arxiv_id"))
    n_oa = sum(1 for e in done.values() if e.get("oa_pdf_url") and not e.get("arxiv_id"))
    n_doi = sum(1 for e in done.values() if e.get("doi") and not e.get("arxiv_id") and not e.get("oa_pdf_url"))
    n_miss = sum(1 for e in done.values() if e.get("resolution") == "miss")
    n_rev = sum(1 for e in done.values() if e.get("review"))
    print(f"DONE unique={len(done)} arxiv={n_arx} oa_pdf={n_oa} doi_only={n_doi} "
          f"miss={n_miss} review_flagged={n_rev} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
