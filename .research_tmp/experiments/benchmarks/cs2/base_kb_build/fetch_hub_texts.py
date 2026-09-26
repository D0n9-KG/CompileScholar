# -*- coding: utf-8 -*-
"""Fetch hub-pool full texts. Hub entries carry doi / openalex_id /
(arxiv id if any). Route: arXiv HTML (if we can resolve an arxiv id)
-> else DOI landing is paywalled for many; fallback = OpenAlex
primary_location PDF (OA) -> nfo text extraction is heavy; smoke
route: use arXiv HTML where the DOI resolves to an arXiv version,
else record as unavailable (hub pool sized with slack for this).

Usage: python fetch_hub_texts.py
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_arxiv_html import fetch_text  # noqa: E402

BASE = Path(r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp"
            r"\experiments\benchmarks\cs2\base_kb")
OUT = BASE / "hub_texts"


def arxiv_id_for_doi(doi: str) -> str | None:
    """DOI -> arXiv id via OpenAlex locations (arXiv version exists?)."""
    if not doi:
        return None
    try:
        url = ("https://api.openalex.org/works/doi:" + doi)
        req = urllib.request.Request(url, headers={
            "User-Agent": "CompileScholar-basekb/0.1"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
        for loc in d.get("locations") or []:
            src = (loc.get("source") or {}).get("display_name", "")
            if src == "arXiv (Cornell University)":
                landing = loc.get("landing_page_url") or ""
                m = re.search(r"abs/([0-9]{4}\.[0-9]{4,5})", landing)
                if m:
                    return m.group(1)
    except Exception:
        return None
    return None


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    hub = json.load(open(BASE / "hub_pool.json", encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    log = []
    ok = fail = 0
    for i, w in enumerate(hub):
        pid = "hub_" + w["openalex_id"]
        dest = OUT / f"{pid}.md"
        if dest.exists() and dest.stat().st_size > 20000:
            ok += 1
            continue
        aid = arxiv_id_for_doi(w.get("doi"))
        time.sleep(0.8)
        if not aid:
            # try title search on arXiv? too noisy — mark unavailable
            fail += 1
            log.append({"paper_id": pid, "ok": False,
                        "err": "no arxiv version"})
            continue
        try:
            text = fetch_text(aid)
            if len(text) < 5000:
                raise ValueError(f"too short: {len(text)}")
            dest.write_text(text, encoding="utf-8")
            w["arxiv_id"] = aid
            ok += 1
            log.append({"paper_id": pid, "ok": True, "arxiv_id": aid,
                        "chars": len(text)})
        except Exception as e:
            fail += 1
            log.append({"paper_id": pid, "ok": False,
                        "err": str(e)[:100]})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(hub)}] ok={ok} fail={fail}", flush=True)
        time.sleep(3.5)
    json.dump(log, open(BASE / "hub_fetch_log.json", "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(hub, open(BASE / "hub_pool.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[hub-fetch] ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
