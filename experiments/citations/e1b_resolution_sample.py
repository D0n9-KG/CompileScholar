# -*- coding: utf-8 -*-
"""Step-1 acceptance check (DESIGN-UPGRADE §10.1): marker -> entry -> arXiv id precision on IdeaForecastBench Markdown.

Samples papers across months (seed 1005), extracts citation sentences, resolves each cited entry with
citations.resolve.EntryResolver, and draws 200 resolved (sentence, entry, resolved paper) items stratified by citation
style (numeric / author-year) and resolution method. Writes runs/e1b-resolution-20261005/{sample.jsonl,stats.json};
labelling is done by two models of different families (experiments/citations/e1b_label.py), as in E1."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from compilescholar.citations import markdown as M
from compilescholar.citations.resolve import EntryResolver
from compilescholar.corpus.papers import Papers

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "external" / "ideaforecast"
OUT = REPO / "runs" / "e1b-resolution-20261005"
SEED, PAPERS_PER_MONTH, N_SAMPLE = 1005, 4, 200


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    papers = Papers()
    res = EntryResolver(papers)
    pool, stats = [], Counter()
    for f in sorted(SRC.glob("*.parquet")):
        rows = pq.read_table(f, columns=["arxiv_id", "text"]).to_pylist()
        for r in rng.sample(rows, min(PAPERS_PER_MONTH, len(rows))):
            bib, cs = M.citation_sentences(r["text"])
            stats["papers"] += 1
            if bib is None:
                stats["no_bib"] += 1
                continue
            stats[f"style_{bib.style}"] += 1
            used = {}
            for c in cs:
                used.setdefault(c.key, c)
            for k, c in used.items():
                e = res.resolve(bib.entries[k]["raw"])
                stats["entries_cited"] += 1
                stats[f"method_{e.method}"] += 1
                if e.arxiv_id:
                    pool.append({"citing": r["arxiv_id"], "style": bib.style, "method": e.method,
                                 "sentence": c.sentence, "entry": bib.entries[k]["raw"], "resolved": e.arxiv_id,
                                 "resolved_title": (papers.get(e.arxiv_id) or {}).get("title", "")})
    rng.shuffle(pool)
    by = {}
    for x in pool:
        by.setdefault(x["style"] if x["style"] in ("numeric", "author-year") else "other", []).append(x)
    sample = []
    for style, n in (("numeric", 100), ("author-year", 100)):
        sample += by.get(style, [])[:n]
    for i, x in enumerate(sample):
        x["id"] = f"E1B-{i + 1:03d}"
    with open(OUT / "sample.jsonl", "w", encoding="utf-8", newline="\n") as fo:
        for x in sample:
            fo.write(json.dumps(x, ensure_ascii=False) + "\n")
    stats["resolved_pool"] = len(pool)
    stats["sample"] = len(sample)
    stats["sample_methods"] = dict(Counter(x["method"] for x in sample))
    json.dump(dict(stats), open(OUT / "stats.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(dict(stats), indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
