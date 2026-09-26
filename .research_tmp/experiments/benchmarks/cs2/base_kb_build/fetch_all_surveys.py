# -*- coding: utf-8 -*-
"""Bulk-fetch all 149 survey full texts from arXiv HTML.

Polite: 3.5s sleep between fetches (~9 min for 149). Fallback per
paper: html unavailable -> try abs page abstract only (mark partial).
Output: base_kb/survey_texts/<paper_id>.md + fetch_log.json
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_arxiv_html import fetch_text  # noqa: E402

BASE = Path(r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp"
            r"\experiments\benchmarks\cs2\base_kb")
OUT = BASE / "survey_texts"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pool = json.load(open(BASE / "survey_pool.json", encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    log = []
    ok = fail = 0
    for i, w in enumerate(pool):
        pid = w["paper_id"]
        aid = w["arxiv_id"]
        dest = OUT / f"{pid}.md"
        if dest.exists() and dest.stat().st_size > 20000:
            ok += 1
            continue  # resume
        try:
            text = fetch_text(aid)
            if len(text) < 5000:
                raise ValueError(f"too short: {len(text)}")
            dest.write_text(text, encoding="utf-8")
            ok += 1
            log.append({"paper_id": pid, "arxiv_id": aid, "ok": True,
                        "chars": len(text)})
        except Exception as e:
            fail += 1
            log.append({"paper_id": pid, "arxiv_id": aid, "ok": False,
                        "err": str(e)[:120]})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(pool)}] ok={ok} fail={fail}", flush=True)
        time.sleep(3.5)
    json.dump(log, open(BASE / "survey_fetch_log.json", "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[fetch] done: ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
