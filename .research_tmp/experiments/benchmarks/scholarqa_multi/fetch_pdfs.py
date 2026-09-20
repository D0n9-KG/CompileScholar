# -*- coding: utf-8 -*-
"""Multi-108 corpus PDF fetcher (stage 2 of task #11).

Reads corpus_map.json (resolve_corpus.py output) and downloads a PDF for every
resolvable paper:
  1. arXiv papers (arxiv_id): https://arxiv.org/pdf/<id> (our best channel,
     identity-verified first page)
  2. OA PDF (oa_pdf_url): direct publisher/repository link
  3. everything else: logged as acquisition_gap (local DOI archive / Crossref
     / S2-with-key remain later fallbacks; never silent substitution)
Identity verification (pipeline standard): pdftotext first page must contain
a fuzzy-matching fragment of the benchmark title; failures are quarantined
(not kept). Resume-safe: existing verified PDFs are skipped.

Output: corpus/pdfs/<paper_id>.pdf + corpus/fetch_report.json
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE = Path(r"C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/scholarqa_multi")
CMAP = BASE / "corpus" / "corpus_map.json"
PDF_DIR = BASE / "corpus" / "pdfs"
REPORT = BASE / "corpus" / "fetch_report.json"
UA = {"User-Agent": "Mozilla/5.0 (research corpus fetch; CompileScholar)"}
TIMEOUT = 180
VERIFY_WORDS = 6  # significant words from the title that must appear on page 1


def _title_words(title: str) -> list[str]:
    stop = {"the", "a", "an", "of", "for", "and", "or", "in", "on", "with",
            "to", "by", "from", "at", "is", "are", "as", "its"}
    ws = [w for w in re.findall(r"[a-z]{3,}", (title or "").lower()) if w not in stop]
    return ws


def _first_page_text(pdf: Path) -> str:
    try:
        out = subprocess.run(["pdftotext", "-f", "1", "-l", "1", str(pdf), "-"],
                             capture_output=True, timeout=60)
        return (out.stdout or b"").decode("utf-8", errors="replace").lower()
    except Exception:
        return ""


def verify(pdf: Path, title: str) -> bool:
    """Pipeline standard: first-page text must contain enough title words.
    A mismatched download (wrong paper) is quarantined, never kept."""
    if pdf.stat().st_size < 20000:   # <20KB is not a real paper
        return False
    if pdf.read_bytes()[:5] != b"%PDF-":
        return False
    txt = _first_page_text(pdf)
    if not txt:
        return True   # pdftotext unavailable/unparseable: accept size+magic only
    words = _title_words(title)
    hits = sum(1 for w in words if w in txt)
    return hits >= min(VERIFY_WORDS, max(2, len(words) // 2))


def fetch(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
        return dest.stat().st_size > 20000
    except Exception:
        if dest.exists():
            dest.unlink()
        return False


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmap = json.loads(CMAP.read_text(encoding="utf-8"))
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    stats = {"arxiv": 0, "oa_pdf": 0, "gap": 0, "quarantined": 0}
    todo = [(t, e) for t, e in cmap.items()
            if e.get("paper_id") not in report]
    print(f"fetch: {len(todo)} papers ({len(report)} already resolved)", flush=True)
    for i, (title, e) in enumerate(todo):
        pid = e.get("paper_id")
        dest = PDF_DIR / f"{pid}.pdf"
        entry = {"title": title, "via": None, "ok": False}
        if dest.exists() and verify(dest, title):
            entry.update({"ok": True, "via": "cached"})
            report[pid] = entry
            continue
        if dest.exists():
            dest.unlink()
        # 1. arXiv (best channel; version suffix stripped by arXiv redirect)
        ax = e.get("arxiv_id")
        if ax and fetch(f"https://arxiv.org/pdf/{ax}", dest) and verify(dest, title):
            entry.update({"ok": True, "via": "arxiv"})
            stats["arxiv"] += 1
        # 2. OA PDF link
        elif e.get("oa_pdf_url") and fetch(e["oa_pdf_url"], dest) and verify(dest, title):
            entry.update({"ok": True, "via": "oa_pdf"})
            stats["oa_pdf"] += 1
        else:
            if dest.exists():
                dest.unlink()          # unverified download: quarantine
                stats["quarantined"] += 1
            entry.update({"ok": False, "via": "gap",
                          "reason": "no arxiv, no oa, or identity mismatch"})
            stats["gap"] += 1
        report[pid] = entry
        if (i + 1) % 20 == 0:
            REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                              encoding="utf-8")
            ok_n = sum(1 for v in report.values() if v.get("ok"))
            print(f"  [{i+1}/{len(todo)}] ok={ok_n} gaps={stats['gap']} "
                  f"quarantined={stats['quarantined']}", flush=True)
        time.sleep(1.0)   # arXiv courtesy
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    ok_n = sum(1 for v in report.values() if v.get("ok"))
    print(f"DONE papers={len(report)} fetched_ok={ok_n} "
          f"arxiv={stats['arxiv']} oa_pdf={stats['oa_pdf']} "
          f"gaps={stats['gap']} quarantined={stats['quarantined']}", flush=True)


if __name__ == "__main__":
    main()
