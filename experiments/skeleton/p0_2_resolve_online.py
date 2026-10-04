# -*- coding: utf-8 -*-
"""W2 P0-2, network stage: resolve the remaining stubs in data/state/cites.jsonl through OpenAlex batched title search
(6 titles per OR query = 10 credits; budget guard from refgraph). Acceptance is the same as the snapshot stage:
normalized-title 40-char prefix equal, year within ±1 when both known. Ambiguous -> stays a stub. Resumable: results
cached per query in refgraph's HTTP cache; rewrites cites.jsonl in place with method="openalex"."""
import json
import re
import sys

from compilescholar.core import paths
from compilescholar.sources import refgraph as RG
from compilescholar.sources.arxiv_snapshot import norm

STATE = paths.data() / "state"
CHUNK = 6


def main(max_queries: int):
    rows = [json.loads(l) for l in open(STATE / "cites.jsonl", encoding="utf-8")]
    todo = [i for i, r in enumerate(rows) if r["paper"] is None and len(norm(r["title"])) >= 12]
    print(f"stubs with a usable title: {len(todo)}; queries needed: {(len(todo) + CHUNK - 1) // CHUNK}", flush=True)
    done = queries = 0
    for s in range(0, len(todo), CHUNK):
        if queries >= max_queries:
            break
        part = todo[s:s + CHUNK]
        q = "|".join(re.sub(r"[,:|()\"?!]", " ", rows[i]["title"])[:200] for i in part)
        d = RG._oa_get({"filter": f"title.search:{q}", "per-page": 50,
                        "select": "id,display_name,publication_year,doi"}, 10)
        queries += 1
        res = (d or {}).get("results") or []
        for i in part:
            t, y = norm(rows[i]["title"]), rows[i]["year"]
            c = [x for x in res if norm(x.get("display_name"))[:40] == t[:40]
                 and (not y or not x.get("publication_year") or abs(int(x["publication_year"]) - int(y)) <= 1)]
            ids = {x["id"].split("/")[-1] for x in c}
            if len(ids) == 1:
                x = c[0]
                doi = (x.get("doi") or "").replace("https://doi.org/", "").lower()
                ax = doi.split("arxiv.")[-1] if "10.48550/arxiv." in doi else None
                rows[i]["paper"] = f"arxiv:{ax}" if ax else (f"doi:{doi}" if doi else f"openalex:{ids.pop()}")
                rows[i]["method"] = "openalex"
                done += 1
        if queries % 50 == 0:
            print(f"  {queries} queries, +{done} resolved, OpenAlex remaining credits {RG._OA_REMAINING[0]}", flush=True)
    tmp = STATE / "cites.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(STATE / "cites.jsonl")
    n = len(rows)
    res_n = sum(1 for r in rows if r["paper"])
    print(f"queries {queries}; newly resolved {done}; total resolved {res_n}/{n} ({res_n / n:.1%}); "
          f"OpenAlex remaining {RG._OA_REMAINING[0]}; stats {dict(RG.STATS)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10_000)
