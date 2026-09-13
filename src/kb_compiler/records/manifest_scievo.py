# -*- coding: utf-8 -*-
"""Stage 0 manifest adapter for the sci-evo-extract registry (journal-domain
corpora; second-domain granular pilot).

Non-arXiv adaptation point (honest disclosure, spec §3 Stage 0):
- chronology anchor = DB year/published_year + year_source (DOI-registry
  curated), NOT arxiv_id (absent for journal-native papers)
- MANIFEST_FIELDS extended with doi/year_source (v1.2.1 patch: metadata-layer
  only, record-layer semantics untouched; same RC6 lineage — non-LLM fields)
- authors/affiliations not in DB -> null + flag (skeleton harvests from text
  headers, as with RL40)
- source_version = sha256 of the EXPORTED text copy (the exact bytes the
  pipeline consumes — material-source audit discipline)

Usage:
  python -m kb_compiler.records.manifest_scievo --db PATH/scievo_registry.sqlite \
      --ids "PPR_XXX=slug1,PPR_YYY=slug2" --texts-out DIR --out manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys


def build(db_path: str, id_map: dict, texts_out: str) -> list:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    os.makedirs(texts_out, exist_ok=True)
    out = []
    for pid, slug in id_map.items():
        row = cur.execute(
            "select title, year, venue, normalized_doi, published_year, year_source "
            "from papers where paper_id=?", (pid,)).fetchone()
        if not row:
            print(f"[MISS] {pid} not in registry", flush=True)
            continue
        title, year, venue, doi, pub_year, year_source = row
        mdrow = cur.execute(
            "select local_path from mineru_artifacts where local_path like '%.md' "
            "and local_path like ?", (f"%{pid}%",)).fetchone()
        if not mdrow:
            print(f"[MISS] {pid} no md artifact", flush=True)
            continue
        base = os.path.dirname(db_path)  # artifact paths are relative to the DB dir (data/library)
        src = os.path.join(base, mdrow[0].replace("\\", "/"))
        if not os.path.exists(src):
            print(f"[MISS] {pid} md file absent: {src}", flush=True)
            continue
        dst = os.path.join(texts_out, slug + ".md")
        shutil.copyfile(src, dst)
        rec = {
            "paper_id": slug, "title": re.sub(r"\s+", " ", str(title or "")).strip(),
            "arxiv_id": None, "arxiv_year": None,
            "venue": str(venue or "") or None, "venue_year": None,
            "year": year or pub_year, "year_source": year_source,
            "doi": doi, "openreview_id": None,
            "authors": None, "affiliations": None,
            "source_file": dst,
            "source_version": hashlib.sha256(open(dst, "rb").read()).hexdigest()[:16],
            "parser": "mineru", "scievo_paper_id": pid,
            "flags": ["affiliations_unavailable", "chronology_anchor=db_year"],
        }
        if not rec["year"]:
            rec["flags"].append("metadata_incomplete:year")
        out.append(rec)
        print(f"[{slug}] {rec['title'][:60]} | {rec['year']} | {rec['venue']} | doi={rec['doi']}",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ids", required=True, help="PPR_XXX=slug,PPR_YYY=slug,...")
    ap.add_argument("--texts-out", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    id_map = {}
    for part in args.ids.split(","):
        pid, slug = part.split("=", 1)
        id_map[pid.strip()] = slug.strip()
    recs = build(args.db, id_map, args.texts_out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(recs, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nsaved {len(recs)} papers -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
