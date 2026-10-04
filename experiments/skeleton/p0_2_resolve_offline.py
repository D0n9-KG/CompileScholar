# -*- coding: utf-8 -*-
"""W2 P0-2, offline stages only (explicit ids, OAI snapshot, KB titles): resolve every bibliography entry that a KB
record's quote cites. Writes data/state/{bib_entries,cites}.jsonl and prints coverage by method and marker style.
No network."""
import collections
import json
import os
import sys

from compilescholar.compile.skeleton import bib as B
from compilescholar.compile.skeleton.resolve import Resolver
from compilescholar.core import paths
from compilescholar.sources.arxiv_snapshot import TitleIndex

KB = paths.legacy_bench() / "cs2" / "base_kb_v2"
TEXTS = paths.legacy_bench() / "cs2" / "base_kb" / "survey_texts"
OUT = paths.data() / "state"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    papers = json.load(open(KB / "papers.json", encoding="utf-8"))
    records = json.load(open(KB / "records.json", encoding="utf-8"))
    res = Resolver(papers, TitleIndex())
    by_method, by_style = collections.Counter(), collections.defaultdict(collections.Counter)
    n_entries = 0
    with open(OUT / "bib_entries.jsonl", "w", encoding="utf-8") as fb, open(OUT / "cites.jsonl", "w", encoding="utf-8") as fc:
        for f in sorted(os.listdir(TEXTS)):
            sid = f[:-3]
            b = B.parse(open(TEXTS / f, encoding="utf-8", errors="replace").read())
            if b is None:
                continue
            consumed = set()
            for r in (records.get(sid) or {}).get("records", []):
                for k in B.markers_in(r.get("quote") or "", b):
                    consumed.add(k)
            for k, e in b.entries.items():
                key = list(k) if isinstance(k, tuple) else k
                fb.write(json.dumps({"survey": sid, "key": key, "style": b.style, **e,
                                     "consumed_by_records": k in consumed}, ensure_ascii=False) + "\n")
                n_entries += 1
                if k not in consumed:
                    continue
                rr = res.resolve(e)
                fc.write(json.dumps({"survey": sid, "key": key, "paper": rr.paper, "method": rr.method,
                                     "title": rr.title, "year": rr.year}, ensure_ascii=False) + "\n")
                by_method[rr.method] += 1
                by_style[b.style]["resolved" if rr.paper else "stub"] += 1
    total = sum(by_method.values())
    print(f"entries {n_entries:,}; consumed by records {total:,}")
    for m, c in by_method.most_common():
        print(f"  {m:20s} {c:6,} ({c / total:.1%})")
    for s, c in by_style.items():
        print(f"  style {s:13s} resolved {c['resolved'] / (c['resolved'] + c['stub']):.1%} of {c['resolved'] + c['stub']:,}")
    kb_hits = sum(c for m, c in by_method.items() if m.startswith("kb_"))
    print("resolved to a KB paper (any stage):",
          sum(1 for l in open(OUT / "cites.jsonl", encoding="utf-8") if '"paper": "kb:' in l))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
