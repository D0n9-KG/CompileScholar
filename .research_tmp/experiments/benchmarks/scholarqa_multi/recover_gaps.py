# -*- coding: utf-8 -*-
"""Recover the 59 gap papers: re-resolve DOIs via Crossref/Sciverse, then
re-run the four-level acquisition chain with the found identities."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:/Users/D0n9/Desktop/sci-evo-extract/src")
from sci_evo_extract.library.sources import CrossrefClient, SciverseClient
from sci_evo_extract.library.acquisition_chain import acquire_fulltext

BASE = Path(r"C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/scholarqa_multi")
REPORT = BASE / "corpus" / "fetch_report.json"
CMAP = BASE / "corpus" / "corpus_map.json"
FETCH_DIR = BASE / "corpus" / "pdfs"
STOP = {"the", "a", "an", "of", "for", "and", "or", "in", "on", "with",
        "to", "by", "from", "at", "is", "are", "as", "its"}


def title_words(t):
    return [w for w in __import__("re").findall(r"[a-z]{3,}", (t or "").lower()) if w not in STOP]


def sim(a, b):
    import re
    wa = title_words(a)
    if not wa:
        return 0.0
    hay = " ".join(re.findall(r"[a-z]{3,}", (b or "").lower()))
    return sum(1 for w in wa if w in hay) / len(wa)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    cmap = json.loads(CMAP.read_text(encoding="utf-8"))
    cr = CrossrefClient()
    sv = SciverseClient()
    gaps = [(pid, v) for pid, v in report.items() if not v.get("ok")]
    print(f"recovering {len(gaps)} gap papers", flush=True)
    recovered = 0
    for i, (pid, v) in enumerate(gaps):
        title = v.get("title", "")
        # 1. find DOI via Crossref bibliographic search
        doi = None
        cands = []
        try:
            cands = [c for c in cr.search_title(title, limit=3) if c.status == "ready"]
        except Exception:
            pass
        for c in sorted(cands, key=lambda c: (c.candidate_score or 0), reverse=True):
            if (c.candidate_score or 0) >= 0.5 and c.normalized_doi:
                doi = c.normalized_doi
                break
        # 2. try Sciverse meta-search for arxiv/OA info
        arxiv_id = None
        oa_url = None
        try:
            mc = sv.search_title(title, limit=2)
            for x in mc:
                if x.status == "ready" and sim(title, x.title or "") >= 0.5:
                    raw = x.raw or {}
                    locs = raw.get("locations") or []
                    for loc in locs:
                        lp = str(loc.get("landing_page_url") or "")
                        import re
                        am = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,6})", lp)
                        if am:
                            arxiv_id = am.group(1)
                            break
                    if loc.get("pdf_url") and not arxiv_id:
                        oa_url = loc.get("pdf_url")
                    break
        except Exception:
            pass
        # 3. update corpus_map entry
        entry = cmap.get(title) or {"paper_id": pid, "title": title}
        if doi:
            entry["doi"] = doi
        if arxiv_id:
            entry["arxiv_id"] = arxiv_id
        cmap[title] = entry
        # 4. run acquisition chain with recovered identity
        res = acquire_fulltext(title=title, out_dir=FETCH_DIR,
                               doi=doi, arxiv_id=arxiv_id, oa_pdf_url=oa_url)
        ok = res.get("status") == "ready"
        report[pid].update(ok=ok, channel=res.get("channel"),
                           path=res.get("path"), format=res.get("format"))
        if not ok:
            report[pid]["attempts"] = [
                (a.get("channel"), a.get("reason") or "no route")
                for a in res.get("attempts", [])]
        if ok:
            recovered += 1
            ch = res.get("channel")
            print(f"  OK [{ch}] {title[:50]}", flush=True)
        if (i + 1) % 10 == 0:
            REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
            CMAP.write_text(json.dumps(cmap, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  [{i+1}/{len(gaps)}] recovered={recovered}", flush=True)
        time.sleep(1.5)  # Crossref courtesy
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    CMAP.write_text(json.dumps(cmap, ensure_ascii=False, indent=1), encoding="utf-8")
    total_ok = sum(1 for v in report.values() if v.get("ok"))
    print(f"DONE recovered={recovered}/{len(gaps)} | total_ok={total_ok}/{len(report)}", flush=True)


if __name__ == "__main__":
    main()
