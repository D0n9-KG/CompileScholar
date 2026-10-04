# -*- coding: utf-8 -*-
"""Survey pool v3: from the LOCAL arXiv metadata snapshot (share drive).

Route decision: OpenAlex concept queries leak engineering domains
(IoT/microgrid/solar reviews measured) — the arXiv snapshot is the
clean domain boundary (six categories) AND gives arxiv ids for free
(full-text channel).

Selection: titles matching survey/review patterns, six-category papers,
cutoff <= 2025-04-30, then per-subfield quota by citation proxy —
the snapshot has NO citation counts, so we rank by recency-window
stratification (surveys' value is freshness for the modern era;
foundational surveys with proven staying power are captured by the
2018-2020 window) and dedupe near-identical topics.

Citation ranking will be backfilled from OpenAlex in a second pass
(batch doi/title lookups) for the top candidates only (~300 lookups).

Usage: python survey_arxiv.py [--target 150]
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

SNAPSHOT = (Path("//192.168.199.138/Share400T/pub/LLM_Data/data"
                 "/JournalPapers/arXiv_Dataset"
                 "/arxiv-metadata-oai-snapshot.json"))
BASE = Path(r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp"
            r"\experiments\benchmarks\cs2\base_kb")

CATS = {"cs.LG", "cs.CL", "cs.CV", "cs.AI", "cs.IR", "stat.ML"}
CUTOFF = "2025-04-30"
SURVEY_TITLE = re.compile(
    r"^(a\s+)?(systematic\s+)?(literature\s+)?(critical\s+)?"
    r"(survey|review|overview)\b|:\s*a\s+(survey|review|overview)\b"
    r"|(survey|review)\s+of\s", re.I)
# 排除：会议系统综述（shared task overview 等非领域综述）
NOT_SURVEY = re.compile(
    r"shared task|system description|overview of the .* shared|"
    r"camera.ready|extended abstract", re.I)

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def parse_created(s):
    m = re.search(r"(\d{1,2})\s+([A-Z][a-z]{2})\s+(\d{4})", s or "")
    if not m:
        return ""
    return f"{m.group(3)}-{_MONTHS.get(m.group(2), 0):02d}-{int(m.group(1)):02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=150)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows = []
    n = 0
    with open(SNAPSHOT, encoding="utf-8", errors="replace") as f:
        for line in f:
            n += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            title = re.sub(r"\s+", " ", d.get("title") or "").strip()
            if not title or not SURVEY_TITLE.search(title):
                continue
            if NOT_SURVEY.search(title):
                continue
            cats = (d.get("categories") or "").split()
            primary = next((c for c in cats if c in CATS), None)
            if not primary:
                continue
            date = parse_created(
                (d.get("versions") or [{}])[0].get("created", ""))
            if not date or date > CUTOFF:
                continue
            abstract = re.sub(r"\s+", " ", d.get("abstract") or "").strip()
            rows.append({
                "paper_id": "arxiv_" + (d.get("id") or "").replace("/", "_"),
                "arxiv_id": d.get("id"),
                "title": title,
                "abstract": abstract,
                "published": date,
                "year": int(date[:4]),
                "primary_category": primary,
                "doi": d.get("doi"),
            })
    print(f"[survey-arxiv] scanned {n} rows -> {len(rows)} surveys",
          flush=True)

    # 近重复主题去重（标题前 8 词为键，保最新）
    by_topic: dict[str, dict] = {}
    for r in sorted(rows, key=lambda r: r["published"]):
        key = " ".join(r["title"].lower().split()[:8])
        by_topic[key] = r
    deduped = list(by_topic.values())
    print(f"[survey-arxiv] topic-deduped: {len(deduped)}", flush=True)

    # 年代分层×子领域配额：综述覆盖不同时期知识状态（CS2 题目分布
    # 跨年代；太新的综述没经过时间检验，太旧的缺现代内容）
    #   2018-2020: 25%  2021-2022: 30%  2023-2024-04: 45%
    BANDS = {"2018-2020": ("2018-01-01", "2020-12-31"),
             "2021-2022": ("2021-01-01", "2022-12-31"),
             "2023-2024": ("2023-01-01", "2024-12-31")}
    BAND_FRAC = {"2018-2020": 0.25, "2021-2022": 0.30, "2023-2024": 0.45}
    per_cat = args.target // len(CATS) + 10
    final = {}
    for band, (lo, hi) in BANDS.items():
        band_quota = int(args.target * BAND_FRAC[band])
        in_band = [r for r in deduped if lo <= r["published"] <= hi]
        n_band = 0
        for r in sorted(in_band, key=lambda r: r["published"],
                        reverse=True):
            if n_band >= band_quota:
                break
            cat = r["primary_category"]
            cnt = sum(1 for x in final.values()
                      if x["primary_category"] == cat)
            if cnt >= per_cat:
                continue
            final[r["paper_id"]] = r
            n_band += 1
    pool = list(final.values())

    json.dump(pool, open(BASE / "survey_pool.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[survey-arxiv] pool {len(pool)} | "
          f"cats: {dict(Counter(r['primary_category'] for r in pool))} | "
          f"years: {dict(Counter(r['year'] for r in pool))}",
          flush=True)


if __name__ == "__main__":
    main()
