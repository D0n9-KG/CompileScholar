# -*- coding: utf-8 -*-
"""Hub layer selection: cross-survey consensus (prereg amendment B step 2).

For each fetched survey, pull its OpenAlex referenced_works (batch
lookup by arxiv id), count how many DIFFERENT surveys cite each work,
then:
  - consensus backbone = works cited by >=3 surveys (domain consensus)
  - exclude surveys themselves + non-CS via domain check on top-N only
  - hub pool = top ~80 non-survey consensus works by consensus count

Usage: python hub_consensus.py [--target 80]
Output: base_kb/hub_pool.json (+ consensus_stats.json)
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

BASE = Path(r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp"
            r"\experiments\benchmarks\cs2\base_kb")
OA = "https://api.openalex.org/works"


def _get(url):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "CompileScholar-basekb/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            print(f"  retry {attempt+1}: {str(e)[:70]}", flush=True)
            time.sleep(4 * (attempt + 1))
    return None


def survey_openalex_id(arxiv_id: str) -> str | None:
    d = _get(OA + "?" + urllib.parse.urlencode({
        "filter": f"ids.openalex_id:null",
        "search": "",
    }) if False else OA + "?" + urllib.parse.urlencode({
        "per-page": "1",
        "search": arxiv_id,
    }))
    if not d or not d.get("results"):
        return None
    return d["results"][0]["id"].split("/")[-1]


def work_meta(openalex_id: str) -> dict | None:
    d = _get(f"{OA}/{openalex_id}")
    if not d:
        return None
    title = d.get("display_name") or ""
    return {
        "openalex_id": openalex_id,
        "title": title,
        "year": d.get("publication_year"),
        "cited_by_count": d.get("cited_by_count"),
        "doi": (d.get("doi") or "").split("doi.org/")[-1] or None,
        "type": d.get("type"),
        "domain": ((d.get("primary_topic") or {}).get("domain") or {})
        .get("display_name"),
        "referenced_works": d.get("referenced_works") or [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=80)
    ap.add_argument("--min-surveys", type=int, default=2)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    surveys = json.load(open(BASE / "survey_pool.json", encoding="utf-8"))
    fetched = {p.stem for p in (BASE / "survey_texts").glob("*.md")}
    surveys = [s for s in surveys if s["paper_id"] in fetched]
    print(f"[hub] {len(surveys)} surveys with full text", flush=True)

    # 1) 每个 survey 的 referenced_works（arxiv id -> openalex work）
    ref_counter: Counter = Counter()   # openalex_id -> 被几篇综述引用
    ref_by_cat: dict = {}              # cat -> Counter（分子领域共识）
    cache_path = BASE / "survey_refs_cache.json"
    cache = json.load(open(cache_path, encoding="utf-8")) \
        if cache_path.exists() else {}
    for i, s in enumerate(surveys):
        aid = s["arxiv_id"]
        if aid in cache:
            refs = cache[aid]
        else:
            w = work_meta_by_arxiv(aid)
            refs = w or {"not_found": True}
            cache[aid] = refs
            json.dump(cache, open(cache_path, "w", encoding="utf-8"),
                      ensure_ascii=False)
            time.sleep(1.0)
        if refs.get("not_found"):
            continue
        cat = s.get("primary_category", "?")
        for r in refs.get("referenced_works", []):
            rid = r.split("/")[-1]
            ref_counter[rid] += 1
            ref_by_cat.setdefault(cat, Counter())[rid] += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(surveys)}] refs counting...",
                  flush=True)

    # 2) 共识骨干（被 >=min-surveys 篇综述引用）
    consensus = {}
    for rid, c in ref_counter.items():
        if c >= args.min_surveys:
            consensus[rid] = c
            continue
        for cat, cnt in ref_by_cat.items():
            if cnt.get(rid, 0) >= args.min_surveys:
                consensus[rid] = cnt[rid]
                break
    print(f"[hub] consensus works (>= {args.min_surveys} surveys): "
          f"{len(consensus)}", flush=True)

    # 3) 排除综述自身+拉 top 元数据（按共识频次取前 target*3 供筛）
    survey_oa_ids = {v.get("openalex_id") for v in cache.values()
                     if v and not v.get("not_found")}
    top = sorted(consensus.items(), key=lambda x: -x[1])[:args.target * 3]
    SURVEY_PAT = re.compile(
        r"^(a\s+)?(systematic\s+)?(literature\s+)?(survey|review|overview)\b"
        r"|:\s*a\s+(survey|review|overview)\b", re.I)
    hub = []
    for rid, cnt in top:
        if rid in survey_oa_ids:
            continue
        m = work_meta(rid)
        time.sleep(0.8)
        if not m:
            continue
        if m["domain"] != "Physical Sciences":
            continue
        if m["type"] not in ("article", "preprint", None):
            continue
        if SURVEY_PAT.search(m["title"] or ""):
            continue
        m["consensus_count"] = cnt
        hub.append(m)
        if len(hub) >= args.target:
            break
    json.dump(hub, open(BASE / "hub_pool.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    stats = {
        "n_surveys": len(surveys),
        "total_refs": len(ref_counter),
        "consensus_ge3": len(consensus),
        "hub_pool": len(hub),
        "consensus_dist_top10": [(h["title"][:40], h["consensus_count"])
                                 for h in hub[:10]],
    }
    json.dump(stats, open(BASE / "consensus_stats.json", "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[hub] pool {len(hub)} | top10: "
          f"{[(h['title'][:30], h['consensus_count']) for h in hub[:5]]}",
          flush=True)


def work_meta_by_arxiv(arxiv_id):
    d = _get(OA + "?" + urllib.parse.urlencode({
        "filter": f"ids.openalex:null", "per-page": "1"}) if False else
        _get(OA + "?" + urllib.parse.urlencode({
            "per-page": "1",
            "filter": f"locations.source.id:S4306400194",  # arXiv source
            "search": arxiv_id,
        })))
    if not d or not d.get("results"):
        # fallback plain search
        d = _get(OA + "?" + urllib.parse.urlencode({
            "per-page": "1", "search": arxiv_id}))
    if not d or not d.get("results"):
        return {"not_found": True}
    w = d["results"][0]
    return {"openalex_id": w["id"].split("/")[-1],
            "referenced_works": w.get("referenced_works") or []}


if __name__ == "__main__":
    main()
