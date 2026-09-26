# -*- coding: utf-8 -*-
"""Survey pool selection (prereg amendment B, step 1).

Core surveys per arXiv six-category domain, via OpenAlex:
  - type=review (OpenAlex authoritative survey flag)
  - cited_by_count desc within concept x era
  - domain whitelist Physical Sciences; fulltext availability (arXiv or DOI)
  - cutoff <= 2025-04-30
  - English (CS2 corpus language)

Also collects each survey's referenced_works (for cross-survey consensus
in step 2).

Usage: python survey_pool.py [--target 150]
Output: base_kb/survey_pool.json (+ survey_refs.json raw)
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

CONCEPTS = {
    "machine_learning": "C119857082",
    "deep_learning": "C108583219",
    "computer_vision": "C31972630",
    "natural_language_processing": "C204321447",
    "artificial_intelligence": "C154945302",
    "information_retrieval": "C23123220",  # 亲验：Information retrieval
}

_t = lambda t: re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()[:120]


def fetch(params):
    url = OA + "?" + urllib.parse.urlencode(params) + "&mailto=research@cs.local"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "CompileScholar-basekb/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            print(f"  retry {attempt+1}: {str(e)[:80]}", flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def abstract_from_inv(inv):
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=150)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # concept id 核验（防 backbone_pool 的 404 覆辙）
    for name, cid in list(CONCEPTS.items()):
        try:
            req = urllib.request.Request(
                f"https://api.openalex.org/concepts/{cid}",
                headers={"User-Agent": "CS/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read())
            if name.replace("_", " ").lower() not in d[
                    "display_name"].lower().replace("  ", " "):
                print(f"  [warn] {name} id points to: {d['display_name']}")
        except Exception as e:
            print(f"  [BAD ID] {name}: {str(e)[:60]} — removing", flush=True)
            del CONCEPTS[name]

    per_concept = max(20, args.target // len(CONCEPTS) + 10)
    SURVEY_TITLE = re.compile(
        r"^(a\s+)?(systematic\s+)?(literature\s+)?(critical\s+)?"
        r"(survey|review|overview)\b|:\s*a\s+(survey|review|overview)\b"
        r"|(survey|review)\s+of\s", re.I)
    pool: dict[str, dict] = {}
    for cname, cid in CONCEPTS.items():
        cursor = "*"
        got = 0
        pages = 0
        while got < per_concept and pages < 30:  # 高引榜里综述稀疏，多翻页
            pages += 1
            # type:review 覆盖差（实测 25 篇 vs 目标 150，arXiv 综述几乎
            # 不标 review 类型）——改标题模式为主识别信号
            d = fetch({
                "filter": (f"concepts.id:{cid},"
                           f"publication_date:2015-01-01|2025-04-30,"
                           f"has_abstract:true"),
                "sort": "cited_by_count:desc",
                "per-page": "50",
                "cursor": cursor,
            })
            if not d:
                break
            for w in d.get("results", []):
                title = w.get("display_name") or ""
                if not title or not SURVEY_TITLE.search(title):
                    continue
                domain = ((w.get("primary_topic") or {})
                          .get("domain") or {}).get("display_name", "")
                if domain != "Physical Sciences":
                    continue
                has_arxiv = any(
                    (l.get("source") or {}).get("display_name", "")
                    == "arXiv (Cornell University)"
                    for l in (w.get("locations") or []))
                doi = w.get("doi")
                if not has_arxiv and not doi:
                    continue
                # 语言：标题含非拉丁字符比例高的跳过（OpenAlex language
                # 字段不可过滤，标题启发式）
                latin = len(re.findall(r"[a-zA-Z]", title)) / max(1, len(title))
                if latin < 0.7:
                    continue
                pid = w.get("id", "").split("/")[-1]
                if pid in pool:
                    continue
                pool[pid] = {
                    "paper_id": "survey_" + pid,
                    "openalex_id": pid,
                    "title": title,
                    "abstract": abstract_from_inv(
                        w.get("abstract_inverted_index")),
                    "published": w.get("publication_date"),
                    "year": w.get("publication_year"),
                    "cited_by_count": w.get("cited_by_count"),
                    "doi": (doi.split("doi.org/")[-1] if doi else None),
                    "has_arxiv": has_arxiv,
                    "concept": cname,
                    "referenced_works": [
                        r.split("/")[-1]
                        for r in (w.get("referenced_works") or [])],
                }
                got += 1
                if got >= per_concept:
                    break
            cursor = (d.get("meta") or {}).get("next_cursor")
            if not cursor:
                break
        print(f"  [{cname}] +{got} (pool {len(pool)})", flush=True)
        time.sleep(1.0)

    ranked = sorted(pool.values(),
                    key=lambda w: -(w.get("cited_by_count") or 0))
    # 语义去重（同题近重复综述）
    seen, deduped = set(), []
    for w in ranked:
        t = _t(w["title"])
        if t in seen:
            continue
        seen.add(t)
        deduped.append(w)
    deduped = deduped[:args.target]

    json.dump(deduped, open(BASE / "survey_pool.json", "w",
                            encoding="utf-8"),
              ensure_ascii=False, indent=1)
    n_refs = sum(len(w["referenced_works"]) for w in deduped)
    stats = {
        "n_surveys": len(deduped),
        "total_references": n_refs,
        "cite_dist": {
            "top": deduped[0]["cited_by_count"] if deduped else 0,
            "median": sorted(w["cited_by_count"] for w in deduped)
            [len(deduped)//2] if deduped else 0,
            "min": min((w["cited_by_count"] for w in deduped),
                       default=0)},
        "with_arxiv": sum(1 for w in deduped if w["has_arxiv"]),
    }
    json.dump(stats, open(BASE / "survey_pool_stats.json", "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[survey] pool {stats}", flush=True)


if __name__ == "__main__":
    main()
