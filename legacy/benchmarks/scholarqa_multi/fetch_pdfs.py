# -*- coding: utf-8 -*-
"""Multi-108 corpus fetcher (stage 2 of task #11) — chain edition.

Every paper goes through sci-evo-extract's automatic acquisition chain
(arXiv -> OA PDF -> Sciverse -> local DOI archive, identity-verified at each
PDF channel, automatic fallback on not-found/timeout/mismatch). The report
maps paper_id -> {channel, path, format}; gaps are logged for the prereg
disclosure list, never silently substituted.

Output: corpus/pdfs/<name>.pdf|.md (chain-named, path recorded) + corpus/fetch_report.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:/Users/D0n9/Desktop/sci-evo-extract/src")
from sci_evo_extract.library.acquisition_chain import acquire_fulltext

BASE = Path(r"C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/scholarqa_multi")
CMAP = BASE / "corpus" / "corpus_map.json"
FETCH_DIR = BASE / "corpus" / "pdfs"
REPORT = BASE / "corpus" / "fetch_report.json"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmap = json.loads(CMAP.read_text(encoding="utf-8"))
    FETCH_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    todo = [(t, e) for t, e in cmap.items()
            if e.get("paper_id") not in report]
    print(f"fetch(chain): {len(todo)} papers ({len(report)} already resolved)", flush=True)
    for i, (title, e) in enumerate(todo):
        pid = e.get("paper_id")
        entry = {"title": title, "ok": False, "channel": None, "path": None,
                 "format": None}
        # cached: previously acquired file still exists
        prev = report.get(pid) or {}
        if prev.get("ok") and prev.get("path") and Path(prev["path"]).exists():
            continue
        try:
            res = acquire_fulltext(
                title=title, out_dir=FETCH_DIR, doi=e.get("doi"),
                arxiv_id=e.get("arxiv_id"), oa_pdf_url=e.get("oa_pdf_url"))
        except Exception as ex:  # noqa: BLE001 — one paper never kills the batch
            res = {"status": "gap", "attempts": [{"channel": "chain",
                                                 "ok": False, "reason": str(ex)[:120]}]}
        entry.update(ok=res.get("status") == "ready",
                     channel=res.get("channel"), path=res.get("path"),
                     format=res.get("format"))
        if not entry["ok"]:
            entry["attempts"] = [(a.get("channel"), a.get("reason") or "no route")
                                 for a in res.get("attempts", [])]
        report[pid] = entry
        if (i + 1) % 10 == 0:
            REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                              encoding="utf-8")
            ok_n = sum(1 for v in report.values() if v.get("ok"))
            by_ch = {}
            for v in report.values():
                if v.get("ok"):
                    by_ch[v["channel"]] = by_ch.get(v["channel"], 0) + 1
            print(f"  [{i+1}/{len(todo)}] ok={ok_n}/{len(report)} channels={by_ch}", flush=True)
        time.sleep(1.0)   # arXiv courtesy between papers
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    ok_n = sum(1 for v in report.values() if v.get("ok"))
    by_ch = {}
    for v in report.values():
        if v.get("ok"):
            by_ch[v["channel"]] = by_ch.get(v["channel"], 0) + 1
    gaps = [v["title"][:60] for v in report.values() if not v.get("ok")]
    print(f"DONE papers={len(report)} ok={ok_n} channels={by_ch} gaps={len(gaps)}", flush=True)
    for g in gaps[:20]:
        print(f"  GAP: {g}", flush=True)


if __name__ == "__main__":
    main()
