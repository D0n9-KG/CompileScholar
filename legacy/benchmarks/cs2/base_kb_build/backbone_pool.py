# -*- coding: utf-8 -*-
"""Tier-2 backbone pool via citation rank (prereg amendment A'').

Domain-facing backbone: top-cited CS/ML papers from OpenAlex, cutoff
<= 2025-04-30, surveys excluded, era-stratified (foundational <=2019 /
modern 2020-2025 quota). Zero question contact.

Usage: python backbone_pool.py [--target 300] [--modern-frac 0.5]
Output: base_kb/backbone_pool.json (+ backbone_stats.json)
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
MAILTO = "mailto=research@compile-scholar.local"

# OpenAlex topic/concept ids: we probe by concept name filter instead —
# simpler: use concepts.id for Machine Learning / AI / CV / NLP / IR.
# Fallback verified: primary_topic filtering is noisy; use arxiv category
# mirror via "topics" is complex. Practical route: query per concept id.
CONCEPTS = {
    "machine_learning": "C119857082",       # 5.18M works
    "deep_learning": "C108583219",          # 763k
    "computer_vision": "C31972630",         # 5.37M
    "natural_language_processing": "C204321447",  # 2.19M
    "artificial_intelligence": "C154945302",      # 34M
}

SURVEY_PAT = re.compile(
    r"\b(survey|review|tutorial|overview|systematic literature|"
    r"bibliometric|meta[- ]analysis)\b", re.I)

_t = lambda t: re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()[:120]


def fetch_page(params):
    url = OA + "?" + urllib.parse.urlencode(params) + "&" + MAILTO
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "CompileScholar-basekb/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  retry {attempt+1}: {str(e)[:80]} (wait {wait}s)",
                  flush=True)
            time.sleep(wait)
    return None


def abstract_from_inv(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def collect_era(era: str, n_needed: int) -> list[dict]:
    """era: 'foundational' (<=2019) or 'modern' (2020-2025-04).
    Query top-cited per concept, merge, dedupe, exclude surveys."""
    if era == "foundational":
        date_filt = "1990-01-01|2019-12-31"
    else:
        date_filt = "2020-01-01|2025-04-30"
    per_concept = max(60, n_needed // len(CONCEPTS) + 20)
    pool: dict[str, dict] = {}
    for cname, cid in CONCEPTS.items():
        # cite rank within concept+era; page through if needed
        cursor = "*"
        got = 0
        while got < per_concept:
            params = {
                "filter": (f"concepts.id:{cid},"
                           f"publication_date:{date_filt},"
                           f"type:article,has_abstract:true"),
                "sort": "cited_by_count:desc",
                "per-page": "100",
                "cursor": cursor,
            }
            d = fetch_page(params)
            if not d:
                break
            for w in d.get("results", []):
                title = w.get("display_name") or ""
                if not title or SURVEY_PAT.search(title):
                    continue
                # 领域白名单（primary_topic.domain）：社科/人文/医学
                # 渗漏的权威闸门（实测：motivated reasoning=Social
                # Sciences，SOM=Physical Sciences）
                domain = ((w.get("primary_topic") or {})
                          .get("domain") or {}).get("display_name", "")
                if domain not in ("Physical Sciences",):
                    continue
                # 全文可得性硬筛：必须 arXiv 版或 DOI 可解析
                has_arxiv = any(
                    (l.get("source") or {}).get("display_name", "")
                    == "arXiv (Cornell University)"
                    for l in (w.get("locations") or []))
                doi = w.get("doi")
                if not has_arxiv and not doi:
                    continue
                pid = w.get("id", "").split("/")[-1]
                if pid in pool:
                    continue
                pool[pid] = {
                    "paper_id": "oa_" + pid,
                    "openalex_id": pid,
                    "title": title,
                    "abstract": abstract_from_inv(
                        w.get("abstract_inverted_index")),
                    "published": w.get("publication_date"),
                    "year": w.get("publication_year"),
                    "cited_by_count": w.get("cited_by_count"),
                    "doi": doi.split("doi.org/")[-1] if doi else None,
                    "primary_location": (
                        (w.get("primary_location") or {}).get("source")
                        or {}).get("display_name"),
                    "concept": cname,
                    "era": era,
                }
                got += 1
                if got >= per_concept:
                    break
            cursor = (d.get("meta") or {}).get("next_cursor")
            if not cursor:
                break
        print(f"  [{era}/{cname}] +{got} (pool {len(pool)})", flush=True)
        time.sleep(1.0)
    # rank by citations, take needed
    ranked = sorted(pool.values(),
                    key=lambda w: -(w.get("cited_by_count") or 0))
    return ranked[:n_needed]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--modern-frac", type=float, default=0.5)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    n_modern = int(args.target * args.modern_frac)
    n_found = args.target - n_modern
    print(f"[backbone] modern {n_modern} + foundational {n_found}",
          flush=True)

    modern = collect_era("modern", n_modern)
    foundational = collect_era("foundational", n_found)

    # dedupe across eras by title-norm
    seen = set()
    pool = []
    for w in modern + foundational:
        t = _t(w["title"])
        if t in seen:
            continue
        seen.add(t)
        pool.append(w)
    # 与现有 Tier-1 manifest 去重（保留骨干标记，paper_id 沿用旧 id）
    manifest = json.load(open(BASE / "manifest.json", encoding="utf-8"))
    man_titles = {_t(r["title"]) for r in manifest}
    overlap = [w for w in pool if _t(w["title"]) in man_titles]
    print(f"[backbone] pool {len(pool)} | already in Tier-1: "
          f"{len(overlap)}", flush=True)

    json.dump(pool, open(BASE / "backbone_pool.json", "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    stats = {
        "n_pool": len(pool),
        "era_mix": dict(Counter(w["era"] for w in pool)),
        "concept_mix": dict(Counter(w["concept"] for w in pool)),
        "cite_dist": {
            "top": pool[0]["cited_by_count"] if pool else 0,
            "median": sorted(
                w["cited_by_count"] or 0 for w in pool)[len(pool)//2]
            if pool else 0,
            "min": min((w["cited_by_count"] or 0 for w in pool),
                       default=0),
        },
        "with_abstract": sum(1 for w in pool
                             if len(w.get("abstract") or "") >= 150),
        "in_tier1_overlap": len(overlap),
    }
    json.dump(stats, open(BASE / "backbone_stats.json", "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[backbone] stats: {stats}", flush=True)


if __name__ == "__main__":
    main()
