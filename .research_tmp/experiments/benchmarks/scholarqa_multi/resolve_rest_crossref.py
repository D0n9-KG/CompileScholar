# -*- coding: utf-8 -*-
"""Resolve remaining Multi titles via Crossref + arXiv API (2026-09-21 凌晨).

OpenAlex hard-blocked us (429 sustained >1h — anonymous-pool abuse detection).
Crossref (polite pool, generous) + the arXiv API (generous) cover the
remaining resolution need: DOI per title + arxiv_id where the paper is on
arXiv. PDF URLs are NOT needed here — the acquisition chain discovers them
at fetch time (S2 openAccessPdf + Crossref links by DOI).

Identity discipline: Crossref bibliographic match must clear title-similarity
+ year checks; arXiv matches must clear title similarity. Failures are
logged, never guessed.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, r"C:/Users/D0n9/Desktop/sci-evo-extract/src")
from sci_evo_extract.library.sources import CrossrefClient

BASE = Path(r"C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/scholarqa_multi")
CMAP = BASE / "corpus" / "corpus_map.json"
STOP = {"the", "a", "an", "of", "for", "and", "or", "in", "on", "with",
        "to", "by", "from", "at", "is", "are", "as", "its"}


def title_words(t):
    return [w for w in re.findall(r"[a-z]{3,}", (t or "").lower()) if w not in STOP]


def sim(expected, observed):
    ws = title_words(expected)
    if not ws:
        return 0.0
    hay = " ".join(re.findall(r"[a-z]{3,}", (observed or "").lower()))
    return sum(1 for w in ws if w in hay) / len(ws)


def arxiv_search(title):
    """arXiv API title search -> [(arxiv_id, doi, year)] best-effort."""
    q = urllib.parse.quote(f'ti:"{title}"')
    url = (f"http://export.arxiv.org/api/query?search_query={q}"
           f"&max_results=5")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            ns = {"a": "http://www.w3.org/2005/Atom",
                  "arxiv": "http://arxiv.org/schemas/atom"}
            root = ET.fromstring(r.read())
            out = []
            for e in root.findall("a:entry", ns):
                aid = (e.findtext("a:id", "", ns) or "").rsplit("/", 1)[-1]
                t = e.findtext("a:title", "", ns) or ""
                pub = e.findtext("a:published", "", ns) or ""
                doi = ""
                for d in e.findall("arxiv:doi", ns):
                    doi = d.text or ""
                out.append((aid, doi, pub[:4], re.sub(r"\s+", " ", t)))
            return out
    except Exception:
        return []


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmap = json.loads(CMAP.read_text(encoding="utf-8"))
    cr = CrossrefClient()
    # unresolved = titles without doi AND without arxiv
    todo = [(t, e) for t, e in cmap.items()
            if not e.get("doi") and not e.get("arxiv_id")]
    print(f"crossref+arxiv resolve: {len(todo)} of {len(cmap)}", flush=True)
    for i, (title, e) in enumerate(todo):
        got_doi = got_ax = None
        # --- arXiv first (gives both arxiv_id and often DOI) ---
        for aid, adoi, ayear, atitle in arxiv_search(title):
            if sim(title, atitle) >= 0.6:
                got_ax = aid
                if adoi and not e.get("doi"):
                    got_doi = adoi.replace("doi:", "")
                break
        # --- Crossref bibliographic search for DOI ---
        if not got_doi:
            cands = []
            try:
                cands = [c for c in cr.search_title(title, limit=3)
                         if c.status == "ready"]
            except Exception:
                pass
            for c in cands:
                if (c.candidate_score or 0) >= 0.6 and c.normalized_doi:
                    year_meta, year_hit = e.get("year"), c.year
                    if year_meta and year_hit and abs(int(year_meta) - int(year_hit)) > 1:
                        continue   # year mismatch: likely a different paper
                    got_doi = c.normalized_doi
                    break
        if got_ax:
            e["arxiv_id"] = got_ax
        if got_doi:
            e["doi"] = got_doi
        e["resolution"] = (e.get("resolution") if (e.get("doi") or got_ax)
                            else e.get("resolution"))
        if got_ax or got_doi:
            e["resolution"] = e.get("resolution") or "crossref_or_arxiv"
            if got_ax:
                e["resolution"] = "arxiv_api" if not e.get("doi") else "arxiv_api+crossref"
            else:
                e["resolution"] = "crossref"
        else:
            e["resolution"] = e.get("resolution") if e.get("resolution") not in (
                None, "openalex") else "miss"
        cmap[title] = e
        if (i + 1) % 15 == 0:
            CMAP.write_text(json.dumps(cmap, ensure_ascii=False, indent=1),
                            encoding="utf-8")
            ok = sum(1 for v in cmap.values() if v.get("doi") or v.get("arxiv_id"))
            print(f"  [{i+1}/{len(todo)}] total_resolvable={ok}/{len(cmap)}", flush=True)
        time.sleep(1.2)   # Crossref courtesy
    CMAP.write_text(json.dumps(cmap, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(1 for v in cmap.values() if v.get("doi") or v.get("arxiv_id"))
    n_doi = sum(1 for v in cmap.values() if v.get("doi"))
    n_ax = sum(1 for v in cmap.values() if v.get("arxiv_id"))
    n_miss = sum(1 for v in cmap.values()
                 if not v.get("doi") and not v.get("arxiv_id"))
    print(f"DONE unique={len(cmap)} resolvable={ok} doi={n_doi} arxiv={n_ax} miss={n_miss}",
          flush=True)


if __name__ == "__main__":
    main()
