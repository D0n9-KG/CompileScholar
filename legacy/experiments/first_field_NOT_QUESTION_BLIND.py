# -*- coding: utf-8 -*-
"""First end-to-end slice (DESIGN §12a "打通全链路，再加量"): one field through every stage.

Field = the cs.LG core queries of ScholarCatalyst (the benchmark we adapt first). Targets = their positive and hard
negative documents that are arXiv papers (the works whose field cognition a tool would be asked about), plus the
source papers' own references when we have their full text. The citations stage processes the full text of every
IdeaForecastBench paper that cites at least one target (so target reception comes from the whole corpus).

Steps: citations.build(selection) -> extract.build(targets) -> tool smoke test. Each step writes its manifest;
rerunning resumes."""
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from compilescholar.citations import build as CB
from compilescholar.citations import markdown as M
from compilescholar.citations.resolve import EntryResolver
from compilescholar.corpus.papers import Papers
from compilescholar.extract import build as EB

REPO = Path(__file__).resolve().parents[2]
SC = REPO / "data" / "external" / "scholarcatalyst"
DOMAIN = "cs.LG"


def targets() -> set[str]:
    q = {x["id"]: x for x in map(json.loads, open(SC / "queries.jsonl", encoding="utf-8"))}
    out = set()
    for line in open(SC / "rels" / "core_query.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if q[r["query_id"]]["paper_domain"] != DOMAIN:
            continue
        for d in [p["id"] for p in r["positive_docs"]] + r["hard_negatives"]:
            if d.startswith("arxiv_"):
                out.add("paper:" + d[6:])
    return out


def citing_selection(tg: set[str]) -> set[str]:
    """IdeaForecastBench papers whose bibliography resolves to at least one target (one pass, no LLM)."""
    res = EntryResolver(Papers())
    sel = set()
    for f in sorted(CB.ideaforecast_dir().glob("*.parquet")):
        for r in pq.read_table(f, columns=["arxiv_id", "text"]).to_pylist():
            bib = M.parse(r["text"])
            if not bib:
                continue
            for k in {m["key"] for m in bib.markers if m["key"] in bib.entries}:
                e = res.resolve(bib.entries[k]["raw"])
                if e.arxiv_id and f"paper:{e.arxiv_id}" in tg:
                    sel.add(r["arxiv_id"])
                    break
        print(f"[first_field] scanned {f.name}: {len(sel)} citing papers", flush=True)
    return sel


def main(stage: str):
    tg = targets()
    print(f"[first_field] {DOMAIN}: {len(tg)} target papers", flush=True)
    if stage in ("citations", "all"):
        sel_p = REPO / "runs" / "dfc-first-field-citing.json"
        sel = set(json.load(open(sel_p))) if sel_p.exists() else citing_selection(tg)
        json.dump(sorted(sel), open(sel_p, "w"))
        print(json.dumps(CB.build(selection=sel, log=lambda s: print(s, flush=True))), flush=True)
    if stage in ("extract", "all"):
        print(json.dumps(EB.build(tg, log=lambda s: print(s, flush=True))), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
