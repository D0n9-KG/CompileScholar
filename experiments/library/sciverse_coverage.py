# -*- coding: utf-8 -*-
"""How much of a benchmark corpus Sciverse already holds as parsed (MinerU 2.5) full text, and which version.

Sample: papers drawn at random (fixed seed) from the corpus members in the registry, stratified by year of first
public date. Per paper: semantic search with the exact title (Sciverse has no id lookup); a hit counts only when its
title_key equals the registry title's (core.ids.title_key). For a hit: the Sciverse date against the arXiv version
dates (v1 / latest), and the full text read through /content: length, whether the reference list is present, whether
tables are markdown or HTML. Account limit 30 req/min (the shared token bucket paces the calls).
Output: runs/sciverse_coverage/{sample.jsonl, report.json}"""
from __future__ import annotations

import json
import random
import re
import sqlite3
import sys
import time

from compilescholar.core import ids, paths
from compilescholar.sources import sciverse as S

OUT = paths.runs() / "sciverse_coverage"


def sample(benchmark: str, n: int, seed: int = 20261007) -> list[dict]:
    con = sqlite3.connect(f"file:{(paths.library() / 'registry.sqlite').as_posix()}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT p.paper_id, p.first_hi, (SELECT title FROM records r WHERE r.paper_id=p.paper_id "
        "ORDER BY r.source != 'arxiv' LIMIT 1) FROM members m JOIN papers p ON p.paper_id=m.paper_id "
        "WHERE m.benchmark=?", (benchmark,)).fetchall()
    by = {}
    for pid, fh, t in rows:
        if t and fh:
            by.setdefault(fh[:4], []).append((pid, fh, t))
    rng = random.Random(seed)
    years = sorted(y for y in by if len(by[y]) >= 50)
    per = max(1, n // len(years))
    out = []
    for y in years:
        for pid, fh, t in rng.sample(by[y], min(per, len(by[y]))):
            vs = con.execute("SELECT version, hi FROM dates WHERE paper_id=? AND kind='arxiv_v' ORDER BY version",
                             (pid,)).fetchall()
            out.append({"paper_id": pid, "first_hi": fh, "title": t, "versions": vs})
    return out


def probe(p: dict) -> dict:
    r = S.request_json("POST", "/agentic-search", payload={"query": p["title"], "page_size": 10}, timeout_seconds=60)
    key = ids.title_key(p["title"])
    hit = next((h for h in r.get("hits") or [] if ids.title_key(re.sub(r"^title:", "", h.get("title") or "")) == key),
               None)
    out = {**p, "found": hit is not None}
    if hit is None:
        return out
    out.update(doc_id=hit["doc_id"], sv_date=hit.get("publication_published_date"),
               sv_venue=hit.get("publication_venue_name_unified"), model=f"{hit.get('model_name')} {hit.get('model_version')}")
    text, off = "", 0
    for _ in range(40):                                    # up to ~400 kB
        c = S.request_json("GET", "/content", query={"doc_id": hit["doc_id"], "offset": off, "limit": 10000},
                           timeout_seconds=60)
        text += c.get("text") or ""
        if str(c.get("more")) != "True":
            break
        off = int(c.get("next_offset") or off + 10000)
    out.update(chars=len(text), has_refs=bool(re.search(r"(?im)^#*\s*(references|bibliography)\s*$", text)),
               html_tables=text.count("<table"), md_tables=len(re.findall(r"(?m)^\|.*\|\s*$\n^\|[\s:-]+\|", text)),
               latex_display=text.count("$$") // 2, images=len(re.findall(r"!\[[^\]]*\]\(", text)))
    vs = dict(p["versions"])
    if vs and out["sv_date"]:
        v1, latest = vs.get(1), vs[max(vs)]
        out["sv_vs_v1_days"] = (time.mktime(time.strptime(out["sv_date"][:10], "%Y-%m-%d")) -
                                time.mktime(time.strptime(v1, "%Y-%m-%d"))) / 86400 if v1 else None
        out["n_versions"] = max(vs)
        out["sv_matches"] = ("v1" if out["sv_date"][:10] == v1 else "latest" if out["sv_date"][:10] == latest else
                             next((f"v{k}" for k, d in vs.items() if d == out["sv_date"][:10]), "other"))
    return out


def main(benchmark: str = "scholarcatalyst", n: int = 60) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sample(benchmark, n):
        try:
            rows.append(probe(p))
        except Exception as e:
            rows.append({**p, "error": f"{type(e).__name__}: {e}"[:200]})
        print(json.dumps({k: rows[-1].get(k) for k in ("paper_id", "first_hi", "found", "sv_date", "sv_matches",
                                                        "chars", "has_refs", "html_tables", "md_tables", "error")}),
              flush=True)
    (OUT / "sample.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    found = [r for r in rows if r.get("found")]
    rep = {"benchmark": benchmark, "sampled": len(rows), "found": len(found),
           "errors": sum("error" in r for r in rows),
           "by_year": {y: [sum(1 for r in rows if r["first_hi"][:4] == y and r.get("found")),
                           sum(1 for r in rows if r["first_hi"][:4] == y)] for y in sorted({r["first_hi"][:4] for r in rows})},
           "version_match": {k: sum(1 for r in found if r.get("sv_matches") == k) for k in
                             {r.get("sv_matches") for r in found}},
           "multi_version_found": sum(1 for r in found if (r.get("n_versions") or 1) > 1),
           "multi_version_v1": sum(1 for r in found if (r.get("n_versions") or 1) > 1 and r.get("sv_matches") == "v1"),
           "median_chars": sorted(r["chars"] for r in found)[len(found) // 2] if found else 0,
           "with_refs": sum(1 for r in found if r.get("has_refs")),
           "with_html_tables": sum(1 for r in found if r.get("html_tables")),
           "with_md_tables": sum(1 for r in found if r.get("md_tables"))}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["scholarcatalyst"]), *(int(x) for x in sys.argv[2:3]))
