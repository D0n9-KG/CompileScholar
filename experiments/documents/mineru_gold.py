# -*- coding: utf-8 -*-
"""MinerU careful-tier parses of the fast-tier gold sample (the cs v1 PDFs that have an arXiv v1 HTML), for
fulltext_compare.py. Skips PDFs already parsed (runs/phaseB_fasttier/mineru/<arxiv_id>.zip). Shared service: 5 in
flight, documents.parse.mineru waits while the router queue is long."""
from __future__ import annotations

import json

import httpx

from compilescholar.core import paths
from compilescholar.dfc.store import parallel
from compilescholar.documents import parse as P

OUT = paths.runs() / "phaseB_fasttier"


def main(slots: int = 5) -> None:
    rows = [json.loads(l) for l in open(OUT / "sample.jsonl", encoding="utf-8")]
    gold = [r for r in rows if r["pdf"] and r["html"]]
    dest = OUT / "mineru"
    dest.mkdir(exist_ok=True)
    todo = [r for r in gold if not (dest / f"{r['arxiv_id']}.zip").exists()]
    print(f"{len(todo)} of {len(gold)} gold PDFs to parse", flush=True)
    base = P._url("mineru_file_parse")
    client = httpx.Client(timeout=httpx.Timeout(60, read=900), trust_env=False)

    def one(r):
        try:
            z = P.mineru((OUT / r["pdf"]).read_bytes(), base, client)
            (dest / f"{r['arxiv_id']}.zip").write_bytes(z)
            print("ok", r["arxiv_id"], flush=True)
        except Exception as e:
            print("fail", r["arxiv_id"], type(e).__name__, e, flush=True)
    parallel(one, todo, slots)


if __name__ == "__main__":
    main()
