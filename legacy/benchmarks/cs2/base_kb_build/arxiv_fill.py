# -*- coding: utf-8 -*-
"""0.6 step 2: arXiv bulk fill for the CS2 hot-start base KB.

Stratified random sample over 6 categories x year bands (2021-2025-04),
filling the base KB from the demand-core pool size up to N=2000.
Metadata only (title/abstract/date/authors/arxiv id) — Tier-1 coarse
extraction is abstract-based; full texts are only acquired for Tier-2.

arXiv API: 100/page per category-year query, polite 3s sleep.
Cutoff: papers with a v1 submission date after 2025-04-30 are dropped
(official inserted_before=2025-05 mapped to the safe side; the arXiv
API's submittedDate range filter does the primary filtering).

Usage: python arxiv_fill.py [--target 2000] [--pool demand_pool.json]
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

CS_REPO = Path(r"C:\Users\D0n9\Desktop\CompileScholar")
OUT_DIR = CS_REPO / ".research_tmp/experiments/benchmarks/cs2/base_kb"

NS = {"a": "http://www.w3.org/2005/Atom"}
CATEGORIES = ["cs.LG", "cs.CL", "cs.CV", "cs.AI", "cs.IR", "stat.ML"]
YEAR_BANDS = ["2021", "2022", "2023", "2024", "2025"]  # 2025 capped at 04-30
PER_QUERY = 100
SLEEP_S = 3.2
CUTOFF = "2025-04-30"


def arxiv_query(cat: str, year: str, start: int) -> tuple[str, ET.Element | None]:
    if year == "2025":
        date_range = f"[202501010000 TO 202504302359]"
    else:
        date_range = (f"[{year}01010000 TO {year}12312359]")
    q = (f"cat:{cat} AND submittedDate:{date_range}")
    # A2 纪律（sci-evo sources.py 实测在案）：urlencode safe=":" 防 %3A；
    # 406=arXiv 临时封禁信号（burst 后分钟级窗口），长退避而非快重试
    url = ("http://export.arxiv.org/api/query?"
           + urllib.parse.urlencode(
               {"search_query": q, "start": str(start),
                "max_results": str(PER_QUERY),
                "sortBy": "submittedDate", "sortOrder": "descending"},
               safe=":[]"))
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "CompileScholar-basekb/0.1 (research use)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return url, ET.fromstring(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 406:
                wait = 60 * (attempt + 1)   # 封禁窗口分钟级
                print(f"  406 throttle: wait {wait}s", flush=True)
                time.sleep(wait)
                continue
            print(f"  retry {attempt+1}: HTTP {e.code}", flush=True)
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            print(f"  retry {attempt+1}: {str(e)[:100]}", flush=True)
            time.sleep(5 * (attempt + 1))
    return url, None


def parse_entries(root: ET.Element) -> list[dict]:
    out = []
    for e in root.findall("a:entry", NS):
        arxiv_id = (e.findtext("a:id", "", NS) or "").split("/abs/")[-1]
        title = re.sub(r"\s+", " ", e.findtext("a:title", "", NS)).strip()
        abstract = re.sub(r"\s+", " ",
                          e.findtext("a:summary", "", NS)).strip()
        published = e.findtext("a:published", "", NS) or ""
        authors = [a.findtext("a:name", "", NS)
                   for a in e.findall("a:author", NS)][:6]
        # primary category
        cats = [c.get("term") for c in e.findall("a:category", NS)]
        if not (arxiv_id and title and abstract and published):
            continue
        out.append({
            "paper_id": "arxiv_" + arxiv_id.split("v")[0].replace("/", "_"),
            "arxiv_id": arxiv_id.split("v")[0],
            "title": title,
            "abstract": abstract,
            "published": published[:10],
            "year": int(published[:4]),
            "authors": authors,
            "primary_category": cats[0] if cats else None,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--pool", default=str(OUT_DIR / "demand_pool.json"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    random.seed(20260927)

    pool = json.load(open(args.pool, encoding="utf-8")) if Path(
        args.pool).exists() else {}
    print(f"[fill] demand pool: {len(pool)} papers", flush=True)

    fill_need = args.target - len(pool)
    if fill_need <= 0:
        print("[fill] pool already at target")
        return
    per_cell = fill_need // (len(CATEGORIES) * len(YEAR_BANDS)) + 2
    print(f"[fill] need {fill_need}; per category-year cell ~{per_cell}",
          flush=True)

    fetched: dict[str, dict] = {}
    for cat in CATEGORIES:
        for year in YEAR_BANDS:
            got = 0
            start = random.randint(0, 3) * PER_QUERY  # 随机起点防最新偏置
            while got < per_cell:
                url, root = arxiv_query(cat, year, start)
                if root is None:
                    print(f"  [{cat}/{year}] query failed at start={start}",
                          flush=True)
                    break
                entries = parse_entries(root)
                if not entries:
                    break
                for e in entries:
                    if e["paper_id"] not in fetched:
                        fetched[e["paper_id"]] = e
                        got += 1
                start += PER_QUERY
                time.sleep(SLEEP_S)
                if len(entries) < PER_QUERY:
                    break
            print(f"  [{cat}/{year}] +{got}", flush=True)

    # 截止过滤（arXiv 查询已按日期带过滤；双保险再滤一次）
    dropped = [p for p in fetched.values() if p["published"] > CUTOFF]
    fetched = {k: v for k, v in fetched.items()
               if v["published"] <= CUTOFF}

    all_rows = list(fetched.values())
    # 需求池并入（paper_id 规范化：title-year 键 → 保持原 title 做 paper_id）
    for key, p in pool.items():
        if (p.get("year") or 9999) > 2025:
            dropped.append({"title": p.get("title"),
                            "reason": f"year {p.get('year')} > cutoff"})
            continue
        row = {
            "paper_id": "sciverse_" + re.sub(
                r"[^a-z0-9]+", "_", (p.get("title") or key).lower())[:80],
            "arxiv_id": None,
            "title": p.get("title"),
            "abstract": p.get("matched_text") or "",
            "published": None,
            "year": p.get("year"),
            "authors": [],
            "primary_category": "demand_core",
            "doc_id": p.get("doc_id"),
        }
        all_rows.append(row)

    random.shuffle(all_rows)
    rows = all_rows[:args.target]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT_DIR / "manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(dropped, open(OUT_DIR / "cutoff_filter_log.json", "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)
    cats = {}
    for r in rows:
        cats[r.get("primary_category") or "?"] = \
            cats.get(r.get("primary_category") or "?", 0) + 1
    with_abs = sum(1 for r in rows if len(r.get("abstract") or "") >= 150)
    print(f"[fill] manifest: {len(rows)} papers "
          f"(arxiv {len(fetched)}, demand {len(pool)-len(dropped)}; "
          f"abstracts>=150ch: {with_abs}) | dropped: {len(dropped)}",
          flush=True)
    print(f"[fill] category mix: {cats}", flush=True)


if __name__ == "__main__":
    main()
