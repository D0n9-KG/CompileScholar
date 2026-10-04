# -*- coding: utf-8 -*-
"""Build the Multi-108 KB manifest: bridge texts/ file stems to corpus_map.

Naming reality (audited 2026-09-21):
- corpus_map.json: keyed by TITLE (439), values carry paper_id (m-hash), subject, doi, year, authors
- fetch_report.json: keyed by m-hash (436), path may be STALE (pre-rename
  LogicKG dir, one broken relative join, one wrong path m40db89bfaf)
- texts/: named by title slug (fetch-time), NOT by m-hash

Manifest convention (frozen for the Multi build):
  paper_id (KB pid) = texts/ file stem; m-hash kept as `corpus_pid` for the
  judge-time id_mapping bridge. One text may serve multiple m-hashes
  (2022_Islet shared by two corpus entries) -> corpus_pid is a LIST.

Match strategy per corpus_map entry:
  1. fetch_report path stem (normalized: LogicKG->CompileScholar, basename), if in texts/
  2. slugified title exact match
  3. alphanumeric-prefix match (slugs truncate ~80 chars)
Unmatched entries are reported (expected: 4 unreachable + parse-pending).

Usage: python build_manifest.py [--out corpus/manifest.json]
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(BASE, "corpus")


def slugify(title):
    s = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    return s


def alnum(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(CORPUS, "manifest.json"))
    args = ap.parse_args()

    cmap = json.load(open(os.path.join(CORPUS, "corpus_map.json"), encoding="utf-8"))
    rep = json.load(open(os.path.join(CORPUS, "fetch_report.json"), encoding="utf-8"))
    texts = {f[:-3] for f in os.listdir(os.path.join(CORPUS, "texts"))
             if f.endswith(".md")}

    # report stem per m-hash (normalized)
    rep_stem = {}
    for h, v in rep.items():
        if v.get("ok") and v.get("path"):
            base = os.path.basename(v["path"].replace("\\", "/"))
            rep_stem[h] = os.path.splitext(base)[0]

    # slug index of texts for prefix matching
    text_alnum = {stem: alnum(stem) for stem in texts}

    entries = []          # manifest rows
    stem2rows = defaultdict(list)
    unmatched = []
    conflicts = []        # title-match vs report-path disagreement (kept title, review!)
    for title, cv in cmap.items():
        h = cv.get("paper_id")
        stem = None
        how = None
        # Title-based match FIRST, report path stem only as fallback:
        # fetch_report has at least one wrong path (m40db89bfaf points at
        # the quadruple-knockout PDF but is a different paper — the Cooper
        # Frontiers review, whose own text exists under its title slug).
        # 1. exact slug
        s2 = slugify(title)
        if s2 in texts:
            stem, how = s2, "slug_exact"
        s1 = rep_stem.get(h)
        # 2. alnum prefix (slugs truncate ~80 chars); report path breaks ties
        if stem is None:
            ta = alnum(title)
            cands = [st for st, a in text_alnum.items()
                     if a and ta and (a.startswith(ta[:60]) or ta.startswith(a[:60]))]
            if len(cands) == 1:
                stem, how = cands[0], "prefix"
            elif len(cands) > 1:
                if s1 in cands:
                    stem, how = s1, "prefix+report_tiebreak"
                else:
                    unmatched.append((title, h, f"AMBIGUOUS prefix: {cands}"))
                    continue
        # 3. report path stem (fallback); conflict with a title match keeps
        # the title match (audited: report paths can be wrong) but warns.
        if stem is None and s1 and s1 in texts:
            stem, how = s1, "report_stem"
        elif stem is not None and s1 and s1 in texts and s1 != stem:
            conflicts.append((title, h, stem, s1))
        if stem is None:
            unmatched.append((title, h, "no text"))
            continue
        row = {
            "paper_id": stem,
            "corpus_pid": [h],
            "title": title,
            "subject": cv.get("subject"),
            "year": cv.get("year"),
            "doi": cv.get("doi"),
            "arxiv_id": cv.get("arxiv_id"),
            "authors": cv.get("authors") or [],
            "match": how,
        }
        stem2rows[stem].append(row)
        entries.append(row)

    # merge rows sharing a stem (one text serving multiple corpus entries)
    merged = {}
    for stem, rows in stem2rows.items():
        base = dict(rows[0])
        base["corpus_pid"] = sorted({h for r in rows for h in r["corpus_pid"]})
        if len(rows) > 1:
            base["title"] = rows[0]["title"]
            base["alias_titles"] = [r["title"] for r in rows[1:]]
        merged[stem] = base
    manifest = sorted(merged.values(), key=lambda r: r["paper_id"])

    # texts with no corpus entry (orphans: duplicates / misnamed)
    mapped_stems = set(merged)
    orphans = sorted(texts - mapped_stems)

    json.dump(manifest, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"manifest: {len(manifest)} rows -> {args.out}")
    print(f"corpus entries matched: {len(entries)}/{len(cmap)}")
    print(f"unmatched corpus entries: {len(unmatched)}")
    for t, h, why in unmatched:
        print(f"  [{why}] {t[:70]} ({h})")
    print(f"orphan texts (no corpus entry): {len(orphans)}")
    for o in orphans:
        print(f"  {o}")
    print(f"CONFLICTS (title match kept, report path disagreed): {len(conflicts)}")
    for t, h, ts, rs in conflicts:
        print(f"  {t[:60]} ({h}): title->{ts[:50]} vs report->{rs[:50]}")
    shared = [r for r in manifest if len(r["corpus_pid"]) > 1]
    print(f"stems shared by multiple corpus entries: {len(shared)}")
    for r in shared:
        print(f"  {r['paper_id'][:60]} <- {r['corpus_pid']}")


if __name__ == "__main__":
    main()
