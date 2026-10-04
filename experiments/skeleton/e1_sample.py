# -*- coding: utf-8 -*-
"""E1 gate sample (DESIGN-W2 §2): 200 resolved (marker -> paper) cases, stratified 100 numeric-style / 100
author-year, seed 1004. Each row shows the citing passage (record quote), the bibliography entry text the marker points
to, and the resolved paper's title, so a human can judge: does the marker refer to that paper?

Writes data/state/e1_sample.jsonl and e1_sample.md (annotation sheet) + their sha256 in e1_sample.sha256. Run once,
before any annotation; the sample is committed (results/e1/) before labels are made."""
import hashlib
import json
import random
import sys
from collections import defaultdict

from compilescholar.compile.skeleton import bib as B
from compilescholar.core import paths
from compilescholar.sources.arxiv_snapshot import index_path

KB = paths.legacy_bench() / "cs2" / "base_kb_v2"
TEXTS = paths.legacy_bench() / "cs2" / "base_kb" / "survey_texts"
STATE = paths.data() / "state"
OUT = paths.REPO / "results" / "e1"
SEED, PER = 1004, 100


def main():
    papers = json.load(open(KB / "papers.json", encoding="utf-8"))
    records = json.load(open(KB / "records.json", encoding="utf-8"))
    cites = {(c["survey"], json.dumps(c["key"])): c for c in map(json.loads, open(STATE / "cites.jsonl", encoding="utf-8"))}
    entries = {(e["survey"], json.dumps(e["key"])): e for e in map(json.loads, open(STATE / "bib_entries.jsonl", encoding="utf-8"))}
    pool = defaultdict(list)
    for sid in sorted({k[0] for k in cites}):
        b = B.parse(open(TEXTS / f"{sid}.md", encoding="utf-8", errors="replace").read())
        style = "author-year" if b.style == "author-year" else "numeric"
        for r in (records.get(sid) or {}).get("records", []):
            q = r.get("quote") or ""
            for k in B.markers_in(q, b):
                key = json.dumps(list(k) if isinstance(k, tuple) else k)
                c = cites.get((sid, key))
                if c and c["paper"]:
                    pool[style].append({"survey": sid, "record_id": r.get("id"), "quote": q[:900], "key": json.loads(key),
                                        "entry": entries[(sid, key)]["raw"][:600], "paper": c["paper"], "method": c["method"]})
    rng = random.Random(SEED)
    sample = []
    for style in ("numeric", "author-year"):
        rows = sorted(pool[style], key=lambda x: (x["survey"], str(x["record_id"]), json.dumps(x["key"])))
        pick = rng.sample(rows, min(PER, len(rows)))
        for x in pick:
            x["style"] = style
        sample += pick
    # resolved paper titles (KB or snapshot) for the sheet
    want = {x["paper"].split(":", 1)[1].lower() for x in sample if x["paper"].startswith("arxiv:")}
    snap = {}
    for line in open(index_path(), encoding="utf-8"):
        p, aid, y, s, full = line.rstrip("\n").split("\t")
        if aid.lower() in want:
            snap[aid.lower()] = f"{full} ({y})"
    for i, x in enumerate(sample, 1):
        x["id"] = f"E1-{i:03d}"
        kind, ident = x["paper"].split(":", 1)
        x["resolved_title"] = ((papers.get(ident) or {}).get("title") if kind == "kb" else
                               snap.get(ident.lower()) if kind == "arxiv" else f"({kind} {ident})")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "e1_sample.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for x in sample:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    md = ["# E1 annotation sheet (marker -> paper)\n",
          "For each row: does the bibliography entry the marker points to describe the same paper as **Resolved**? "
          "Label `Y` (same paper), `N` (different paper), or `?` (cannot tell). Leave `label` empty until annotating.\n"]
    for x in sample:
        md.append(f"\n## {x['id']} ({x['style']}, via {x['method']})\n\n**Marker** `{x['key']}` in survey `{x['survey']}`\n\n"
                  f"**Quote**: {x['quote']}\n\n**Entry**: {x['entry']}\n\n**Resolved**: `{x['paper']}` — {x['resolved_title']}\n\n"
                  f"label: \n")
    (OUT / "e1_sheet.md").write_text("".join(md), encoding="utf-8", newline="\n")
    h = {n: hashlib.sha256((OUT / n).read_bytes()).hexdigest() for n in ("e1_sample.jsonl", "e1_sheet.md")}
    (OUT / "e1_sample.sha256").write_text("".join(f"{v}  {k}\n" for k, v in h.items()), encoding="utf-8", newline="\n")
    print({s: len(pool[s]) for s in pool}, "-> sample", len(sample), h)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
