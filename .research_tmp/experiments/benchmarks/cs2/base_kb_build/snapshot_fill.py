# -*- coding: utf-8 -*-
"""0.6 step 2 (snapshot route): fill the base KB from the LOCAL arXiv
official metadata snapshot on the share drive (no API, no rate limit).

Reads (read-only): \\\\192.168.199.138\\Share400T\\pub\\LLM_Data\\data\\
JournalPapers\\arXiv_Dataset\\arxiv-metadata-oai-snapshot.json
Writes (local only): manifest.json (merged), snapshot_fill_stats.json.

Sampling: six categories x year bands 2021-2025-04, stratified random,
dedup vs the demand pool (by title-norm and arxiv id).

Also: back-resolve demand-pool papers to arxiv ids via title match against
the snapshot (needed for Tier-2 full-text fetch).
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def parse_created(s: str) -> str:
    """'Thu, 5 Apr 2007 02:57:15 GMT' -> '2007-04-05'（快照实测格式）"""
    m = re.search(r"(\d{1,2})\s+([A-Z][a-z]{2})\s+(\d{4})", s or "")
    if not m:
        return ""
    return f"{m.group(3)}-{_MONTHS.get(m.group(2), 0):02d}-{int(m.group(1)):02d}"

SNAPSHOT = (Path(r"\\192.168.199.138\Share400T\pub\LLM_Data\data"
                 r"\JournalPapers\arXiv_Dataset"
                 r"\arxiv-metadata-oai-snapshot.json"))
CS_REPO = Path(r"C:\Users\D0n9\Desktop\CompileScholar")
BASE_KB = (CS_REPO / ".research_tmp/experiments/benchmarks/cs2/base_kb")

CATS = ["cs.LG", "cs.CL", "cs.CV", "cs.AI", "cs.IR", "stat.ML"]
YEAR_BANDS = ["2021", "2022", "2023", "2024", "2025"]  # 2025 -> <=04-30
CUTOFF = "2025-04-30"
TARGET = 2000

_norm_title = re.compile(r"[^a-z0-9]+")


def norm_title(t: str) -> str:
    return _norm_title.sub(" ", (t or "").lower()).strip()[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    random.seed(20260927)

    # ---- load existing demand pool manifest
    manifest = json.load(open(BASE_KB / "manifest.json", encoding="utf-8"))
    demand_rows = [r for r in manifest
                   if r.get("primary_category") == "demand_core"]
    seen_titles = {norm_title(r["title"]) for r in demand_rows}
    print(f"[fill] demand pool: {len(demand_rows)} papers", flush=True)

    fill_need = args.target - len(demand_rows)
    per_cell = max(1, fill_need // (len(CATS) * len(YEAR_BANDS)) + 2)
    print(f"[fill] need {fill_need}; per cell {per_cell}", flush=True)

    # ---- one streaming pass over the snapshot: collect per-cell pools
    # (over-collect 3x, then sample per cell)
    cells: dict[tuple, list] = {}
    demand_by_title: dict[str, dict] = {}
    n_scanned = 0
    with open(SNAPSHOT, encoding="utf-8", errors="replace") as f:
        for line in f:
            n_scanned += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            # demand back-resolve (any category): title-norm match
            tn = norm_title(d.get("title"))
            if tn in seen_titles and tn not in demand_by_title:
                date = parse_created(
                    (d.get("versions") or [{}])[0].get("created", ""))
                demand_by_title[tn] = {
                    "arxiv_id": d.get("id"),
                    "published": date,
                    "year": int(date[:4]) if date[:4].isdigit() else None,
                }
            # six-cat cell collection
            cats = (d.get("categories") or "").split()
            cat = next((c for c in cats if c in CATS), None)
            if not cat:
                continue
            date = parse_created(
                (d.get("versions") or [{}])[0].get("created", ""))
            if not date or date > CUTOFF:
                continue
            year = date[:4]
            if year not in YEAR_BANDS:
                continue
            tn = norm_title(d.get("title"))
            if tn in seen_titles:
                continue  # dedup vs demand pool
            cell = (cat, year)
            # over-collect 3x per cell cap
            if len(cells.get(cell, [])) < per_cell * 3:
                cells.setdefault(cell, []).append({
                    "paper_id": "arxiv_" + (d.get("id") or "").replace("/", "_"),
                    "arxiv_id": d.get("id"),
                    "title": re.sub(r"\s+", " ", d.get("title") or "").strip(),
                    "abstract": re.sub(r"\s+", " ",
                                       d.get("abstract") or "").strip(),
                    "published": date,
                    "year": int(year),
                    "authors": [a for a in re.split(r",\s*",
                               d.get("authors") or "")][:6],
                    "primary_category": cat,
                })
            if n_scanned % 500000 == 0:
                print(f"  scanned {n_scanned/1e6:.1f}M rows", flush=True)

    print(f"[fill] snapshot scanned: {n_scanned} rows", flush=True)

    # ---- stratified sample per cell
    bulk_rows = []
    for cell, pool in sorted(cells.items()):
        k = min(per_cell, len(pool))
        bulk_rows.extend(random.sample(pool, k))
    print(f"[fill] bulk sampled: {len(bulk_rows)} "
          f"(cells: {len(cells)})", flush=True)

    # ---- back-resolve demand pool arxiv ids
    resolved = 0
    cutoff_drops = []
    for r in demand_rows:
        tn = norm_title(r["title"])
        hit = demand_by_title.get(tn)
        if hit:
            r["arxiv_id"] = hit["arxiv_id"]
            r["published"] = hit.get("published")
            if hit.get("year") and r.get("year") != hit["year"]:
                r["year_snapshot"] = hit["year"]
            resolved += 1
            # cutoff enforcement with snapshot date (more reliable than
            # Sciverse year): drop if v1 after cutoff
            if (hit.get("published") or "") > CUTOFF:
                cutoff_drops.append({
                    "title": r["title"],
                    "arxiv_id": hit["arxiv_id"],
                    "published": hit["published"],
                    "reason": "demand hit v1 after cutoff (snapshot)"})

    rows = []
    dropped_titles = {d["title"] for d in cutoff_drops}
    for r in demand_rows:
        if r["title"] in dropped_titles:
            continue
        # unresolved 论文保留（Sciverse 年份已滤过一轮；快照缺失的保守放行）
        rows.append(r)
    rows.extend(bulk_rows)

    random.shuffle(rows)
    rows = rows[:args.target]
    json.dump(rows, open(BASE_KB / "manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    stats = {
        "snapshot_rows_scanned": n_scanned,
        "demand_pool": len(demand_rows),
        "demand_resolved_to_arxiv": resolved,
        "demand_cutoff_drops": cutoff_drops,
        "bulk_rows": len(bulk_rows),
        "final_manifest": len(rows),
        "category_mix": dict(Counter(
            r.get("primary_category") for r in rows)),
        "abstract_ok": sum(1 for r in rows
                           if len(r.get("abstract") or "") >= 150),
    }
    json.dump(stats, open(BASE_KB / "snapshot_fill_stats.json", "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[fill] manifest: {len(rows)} | demand->arxiv resolved: "
          f"{resolved} | abstracts ok: {stats['abstract_ok']}", flush=True)
    print(f"[fill] category mix: {stats['category_mix']}", flush=True)


if __name__ == "__main__":
    main()
