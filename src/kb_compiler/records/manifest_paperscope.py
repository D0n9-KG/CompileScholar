# -*- coding: utf-8 -*-
"""Stage 0 manifest for the PaperScope-53 second-domain corpus.

Sources (all pre-existing local assets + arXiv API; OpenReview API is 403 on
this IP — probed 2026-09-05):
- paper_ids.json / paper_titles.json      : 53 OpenReview ids + titles
- download.log / fallback*.log            : ACTUAL material source per paper
                                            (arXiv id + version) — parsed
- download_report.json                    : same, structured (47 ok)
- arXiv API batch                         : authors / published / comment /
                                            journal_ref (venue hints)
- mineru/<id>/<id>.md                     : parsed text (sha256 = source_version)

Protocol-audit fields (lesson #1 from the PaperScope verdict: material source
!= gold source): gold_source is CONSTANT "openreview_pdf" (official pdf_links
are all openreview.net); material records the arXiv version we actually parsed.
venue/venue_year ONLY from arXiv comment/journal_ref regex — never guessed
(the PBADet/Dale's Law year-drift root cause). Unresolved -> null + flag.

Usage:
  python -m kb_compiler.records.manifest_paperscope --ps-dir DIR \
      --out manifest_ps53.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

from .manifest import _arxiv_batch

_VENUE_PAT = re.compile(
    r"\b(ICLR|NeurIPS|ICML|COLM|EMNLP|ACL|NAACL|CVPR|ICCV|ECCV|AAAI|KDD|WWW|"
    r"SIGIR|TMLR|JMLR|UAI|AISTATS|CoRL|RSS)\b[ ,]*((?:19|20)\d{2})?", re.I)


def _parse_download_logs(ps_dir: str) -> dict:
    """openreview_id -> arxiv_id (with version if logged)."""
    src = {}
    for fn in ("download.log", "fallback.log", "fallback2.log"):
        p = os.path.join(ps_dir, fn)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8", errors="replace"):
            m = re.search(r"\[OK[^\]]*\]\s+(\w{8,12})\s+<-\s+(\d{4}\.\d{4,5}v?\d*)", line)
            if m:
                src.setdefault(m.group(1), m.group(2))
                continue
            m = re.search(r"\[arXiv-all\]\s+(\w{8,12})\s+->\s+(\d{4}\.\d{4,5}v?\d*)", line)
            if m:
                src.setdefault(m.group(1), m.group(2))
    return src


def build(ps_dir: str) -> list:
    ids = json.load(open(os.path.join(ps_dir, "paper_ids.json"), encoding="utf-8"))
    titles = json.load(open(os.path.join(ps_dir, "paper_titles.json"), encoding="utf-8"))
    log_src = _parse_download_logs(ps_dir)
    rep = json.load(open(os.path.join(ps_dir, "download_report.json"), encoding="utf-8"))
    rep_src = {r["id"]: r.get("arxiv") for r in rep.get("ok", []) if r.get("arxiv")}

    resolved = {}
    for pid in ids:
        aid = log_src.get(pid) or rep_src.get(pid)
        if aid:
            resolved[pid] = aid.split("v")[0]  # API wants bare id
    versions = {pid: (log_src.get(pid) or rep_src.get(pid) or "") for pid in ids}

    entries = _arxiv_batch(sorted(set(resolved.values())))
    print(f"arXiv API: {len(set(resolved.values()))} requested, {len(entries)} got",
          flush=True)

    out = []
    for pid in ids:
        md = os.path.join(ps_dir, "mineru", pid, pid + ".md")
        rec = {
            "paper_id": pid, "title": titles.get(pid), "arxiv_id": resolved.get(pid),
            "arxiv_year": None, "venue": None, "venue_year": None,
            "openreview_id": pid, "authors": None, "affiliations": None,
            "source_file": md if os.path.exists(md) else None,
            "source_version": None, "parser": "mineru",
            "gold_source": "openreview_pdf",
            "material_source": f"arxiv:{versions[pid]}" if versions.get(pid) else "unknown",
            "flags": ["affiliations_unavailable", "gold_source_ne_material_source"],
        }
        if os.path.exists(md):
            rec["source_version"] = hashlib.sha256(
                open(md, "rb").read()).hexdigest()[:16]
        else:
            rec["flags"].append("mineru_text_missing")
        ent = entries.get(resolved.get(pid)) if resolved.get(pid) else None
        if ent:
            if ent.get("published"):
                rec["arxiv_year"] = int(ent["published"][:4])
            rec["authors"] = ent.get("authors") or None
            if ent.get("title") and not rec["title"]:
                rec["title"] = ent["title"]
            for field in (ent.get("comment") or "", ent.get("journal_ref") or ""):
                m = _VENUE_PAT.search(field)
                if m:
                    rec["venue"] = m.group(1).upper()
                    rec["venue_year"] = int(m.group(2)) if m.group(2) else None
                    rec["venue_source"] = "arxiv_comment" if field == (ent.get("comment") or "") else "arxiv_journal_ref"
                    break
        if not rec["arxiv_id"]:
            rec["flags"].append("no_arxiv_source")
        missing = [k for k in ("title", "arxiv_year", "venue", "authors")
                   if rec.get(k) in (None, "", [])]
        if missing:
            rec["flags"].append("metadata_incomplete:" + ",".join(missing))
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recs = build(args.ps_dir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(recs, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n_venue = sum(1 for r in recs if r["venue"])
    n_auth = sum(1 for r in recs if r["authors"])
    n_arx = sum(1 for r in recs if r["arxiv_id"])
    print(f"saved {len(recs)} -> {args.out}")
    print(f"arxiv_id: {n_arx}/53 | venue resolved: {n_venue}/53 | authors: {n_auth}/53",
          flush=True)
    from collections import Counter
    print("venues:", Counter(r["venue"] for r in recs if r["venue"]).most_common(), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
