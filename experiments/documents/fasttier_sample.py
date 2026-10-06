# -*- coding: utf-8 -*-
"""Phase B fast-tier comparison, step 1: sample cs papers and collect the gold standard.

Sample: primary cs.*, v1 dated 2018-01..2025-08 (the NAS per-version mirror ends at 2509), 9 per year, fixed seed.
Per paper: copy <mirror>/<yymm>/<id>v1.pdf, fetch https://arxiv.org/html/<id>v1 (LaTeXML; every in-text citation
is an anchor to its bibitem, so "which sentence cites which entry" is exact). 15 s pacing per arxiv.org robots.txt.
Papers without a v1 HTML rendering or a mirrored v1 PDF are dropped. Run from PowerShell (UNC access).
Output: runs/phaseB_fasttier/{pdf/<id>v1.pdf, html/<id>v1.html, sample.jsonl}
"""
from __future__ import annotations

import json
import random
import shutil
import sqlite3
import sys
import time
import urllib.error
import urllib.request

from compilescholar.core import paths

OUT = paths.runs() / "phaseB_fasttier"
UA = "CompileScholar-research (citation-context gold from arXiv HTML; 15 s pacing)"
PER_YEAR = 9


def main(per_year: int = PER_YEAR, seed: int = 20261006) -> None:
    (OUT / "pdf").mkdir(parents=True, exist_ok=True)
    (OUT / "html").mkdir(parents=True, exist_ok=True)
    mirror = paths.resource("arxiv_pdf_mirror")
    con = sqlite3.connect(f"file:{paths.derived() / 'papers.sqlite'}?mode=ro", uri=True)
    rng = random.Random(seed)
    sample = []
    for y in range(2018, 2026):
        hi = "2025-08-31" if y == 2025 else f"{y}-12-31"
        ids = [r[0] for r in con.execute("SELECT arxiv_id FROM papers WHERE primary_cat LIKE 'cs.%' AND v1_date "
                                         "BETWEEN ? AND ? ORDER BY arxiv_id", (f"{y}-01-01", hi))]
        sample += [(y, a) for a in rng.sample(ids, per_year)]
    rows, last = [], 0.0
    for y, aid in sample:
        pdf = mirror / aid.split(".")[0] / f"{aid}v1.pdf"
        rec = {"arxiv_id": aid, "year": y, "pdf": None, "html": None}
        if pdf.exists():
            shutil.copyfile(pdf, OUT / "pdf" / f"{aid}v1.pdf")
            rec["pdf"] = f"pdf/{aid}v1.pdf"
        hp = OUT / "html" / f"{aid}v1.html"
        if not hp.exists():
            time.sleep(max(0.0, 15.0 - (time.time() - last)))
            last = time.time()
            try:
                req = urllib.request.Request(f"https://arxiv.org/html/{aid}v1", headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=90) as r:
                    body = r.read()
                if b"ltx_bibitem" in body:
                    hp.write_bytes(body)
                else:
                    rec["html_missing"] = "no bibitems"
            except urllib.error.HTTPError as e:
                rec["html_missing"] = f"http {e.code}"
        if hp.exists():
            rec["html"] = f"html/{aid}v1.html"
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    with open(OUT / "sample.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok = sum(1 for r in rows if r["pdf"] and r["html"])
    print(f"done: {len(rows)} sampled, {ok} with both v1 PDF and v1 HTML", flush=True)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))
