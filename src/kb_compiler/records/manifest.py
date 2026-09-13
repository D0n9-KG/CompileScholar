# -*- coding: utf-8 -*-
"""Stage 0: paper metadata manifest (non-LLM). Spec v1.1 §3 Stage 0 + RC6.

paper_id -> {title, arxiv_id, arxiv_year, venue, venue_year, openreview_id,
             authors, affiliations, source_file, source_version, parser, flags}

Discipline:
- NO LLM-guessed years anywhere (v2 year-drift root cause: PBADet/Dale's Law).
  arxiv_year comes from the arXiv API `published` stamp (v1 date); venue
  fields from journal_ref / manual-curated overrides only.
- Honest nulls + flags: metadata no source gives stays null and is flagged;
  never fabricated. Manual-curated entries carry source="manual_curated".
- source_version = sha256 prefix of the parsed-text file — the data basis
  for the "material source == gold source" protocol audit.

Source tiers (2026-09-05):
1. arXiv API batch (export.arxiv.org id_list — one call for the whole corpus)
2. local acquisition manifest (contest-era S2 matches: matched_title/year)
3. MANUAL_OVERRIDES: seeds + pre-arXiv-era papers, human-curated facts only
S2 graph API removed: instant 429 wall on this IP (probed 2026-09-05, 0.4s);
affiliations therefore unavailable from current tiers -> flagged, not faked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# Human-curated facts for papers the arXiv tier cannot resolve. Every entry is
# a well-established bibliographic fact; source tagged for audit. Authors left
# null where not curated (flagged) rather than typed from memory.
MANUAL_OVERRIDES = {
    "dqn2013":         {"arxiv_id": "1312.5602", "source": "manual_curated"},
    "a3c":             {"arxiv_id": "1602.01783", "source": "manual_curated"},
    "gorila":          {"title": "Massively Parallel Methods for Deep Reinforcement Learning",
                        "venue": "ICML", "venue_year": 2015, "source": "manual_curated"},
    "seed_DQN_nature": {"title": "Human-level control through deep reinforcement learning",
                        "venue": "Nature", "venue_year": 2015, "source": "manual_curated"},
    "seed_DDQN":       {"arxiv_id": "1509.06461", "source": "manual_curated"},
    "seed_PER":        {"arxiv_id": "1511.05952", "source": "manual_curated"},
    "seed_Dueling":    {"arxiv_id": "1511.06581", "source": "manual_curated"},
    "seed_Rainbow":    {"arxiv_id": "1710.02298", "source": "manual_curated"},
    # R2D2 = ICLR 2019, NO arXiv version exists. The contest-era acq manifest
    # mismatched tag "r2d2" onto IMPALA's arXiv id (1802.01561) — caught by the
    # title-consistency guard below on first builder run (2026-09-05).
    "r2d2":            {"title": "Recurrent Experience Replay in Distributed Reinforcement Learning",
                        "venue": "ICLR", "venue_year": 2019, "source": "manual_curated"},
}


def _tok(s: str | None) -> set:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _jaccard(a: str | None, b: str | None) -> float:
    ta, tb = _tok(a), _tok(b)
    return len(ta & tb) / max(1, len(ta | tb))


def _arxiv_batch(ids: list[str], tries: int = 3) -> dict[str, dict]:
    """One batched arXiv API call -> {arxiv_id: entry}. Honest {} on failure."""
    if not ids:
        return {}
    url = f"{ARXIV_API}?id_list=" + ",".join(ids) + f"&max_results={len(ids)}"
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kb-compiler-stage0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                root = ET.fromstring(r.read())
            out = {}
            for e in root.findall("a:entry", _NS):
                eid = (e.findtext("a:id", "", _NS) or "")
                m = re.search(r"(\d{4}\.\d{4,5})", eid)
                if not m:
                    continue
                jref = e.findtext("arxiv:journal_ref", "", _NS) or ""
                out[m.group(1)] = {
                    "title": re.sub(r"\s+", " ", e.findtext("a:title", "", _NS)).strip(),
                    "authors": [n.text for n in e.findall("a:author/a:name", _NS)],
                    "published": e.findtext("a:published", "", _NS),
                    "journal_ref": jref.strip(),
                    "comment": (e.findtext("arxiv:comment", "", _NS) or "").strip(),
                    "doi": (e.findtext("arxiv:doi", "", _NS) or "").strip(),
                }
            return out
        except Exception:
            time.sleep(4.0 * (i + 1))
    return {}


def _blank(paper_id: str, source_file: str) -> dict:
    ext = os.path.splitext(source_file)[1].lower()
    digest = hashlib.sha256(open(source_file, "rb").read()).hexdigest()[:16]
    return {
        "paper_id": paper_id, "title": None, "arxiv_id": None, "arxiv_year": None,
        "venue": None, "venue_year": None, "openreview_id": None,
        "authors": None, "affiliations": None,
        "source_file": source_file, "source_version": digest,
        "parser": "mineru" if ext == ".md" else "plain_txt",
        "flags": ["affiliations_unavailable"],  # no current tier provides them (RC6 partial)
    }


def build_manifest(texts_dir: str, acq_path: str | None) -> list[dict]:
    acq = {}
    if acq_path and os.path.exists(acq_path):
        for line in open(acq_path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") == "ok":
                acq[r["tag"].replace("-", "_")] = r

    pids = sorted(f.rsplit(".", 1)[0] for f in os.listdir(texts_dir)
                  if f.endswith((".md", ".txt")))

    # resolve arxiv_id per paper: override -> acq externalIds -> acq pdf filename
    ids, id_src = {}, {}
    for pid in pids:
        ov = MANUAL_OVERRIDES.get(pid, {})
        aid = ov.get("arxiv_id")
        src = "override"
        if not aid and pid in acq:
            row = acq[pid]
            aid = (row.get("externalIds") or {}).get("ArXiv")
            if not aid:
                m = re.search(r"arxiv[_/](\d{4}\.\d{4,5})", row.get("pdf_path") or "")
                aid = m.group(1) if m else None
            src = "acq"
        if aid:
            ids[pid] = aid
            id_src[pid] = src

    entries = _arxiv_batch(sorted(set(ids.values())))
    print(f"arXiv API: requested {len(set(ids.values()))} ids, got {len(entries)} entries",
          flush=True)

    # title-consistency guard on acq-derived ids: the contest-era local S2
    # match is not trusted blindly (r2d2->IMPALA contamination, caught 2026-09-05).
    # Ground truth = max(jaccard vs acq query_title, jaccard vs corpus text
    # heading) — the text heading survives arXiv cross-version title drift
    # (efficientzero v1 "Mastering Atari Games with Limited Data" false-reject,
    # caught same day: heading matched API title exactly, query_title didn't).
    def _heading(pid: str) -> str:
        for ext in (".md", ".txt"):
            p = os.path.join(texts_dir, pid + ext)
            if os.path.exists(p):
                with open(p, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip().lstrip("#").strip()
                        if line:
                            return line[:200]
        return ""

    rejected = []
    for pid in [p for p, s in id_src.items() if s == "acq"]:
        ent = entries.get(ids[pid])
        if not ent:
            continue
        j = max(_jaccard(ent["title"], acq[pid].get("query_title")),
                _jaccard(ent["title"], _heading(pid)))
        if j < 0.5:
            rejected.append((pid, ids[pid], ent["title"]))
            del ids[pid]
    for pid, aid, got in rejected:
        print(f"[guard] rejected acq arxiv id: {pid} -> {aid} (API title: {got[:50]})",
              flush=True)

    out = []
    for pid in pids:
        rec = _blank(pid, os.path.join(texts_dir, pid + (".md" if os.path.exists(
            os.path.join(texts_dir, pid + ".md")) else ".txt")))
        ov = MANUAL_OVERRIDES.get(pid, {})
        aid = ids.get(pid)
        ent = entries.get(aid) if aid else None
        if ent:
            rec["arxiv_id"] = aid
            rec["arxiv_year"] = int(ent["published"][:4]) if ent.get("published") else None
            rec["title"] = ent["title"] or None
            rec["authors"] = ent["authors"] or None
            jref = ent.get("journal_ref") or ""
            if jref:
                rec["venue"] = jref
                m = re.search(r"(19|20)\d{2}", jref)
                rec["venue_year"] = int(m.group(0)) if m else None
        elif aid:
            rec["arxiv_id"] = aid
            rec["flags"].append("arxiv_api_miss")
        # manual-curated overrides (fill gaps only; arXiv stays authoritative)
        for k in ("title", "venue", "venue_year"):
            if rec.get(k) in (None, "") and ov.get(k) is not None:
                rec[k] = ov[k]
        # local acquisition fallback for title
        if not rec["title"] and pid in acq:
            rec["title"] = acq[pid].get("matched_title")
            rec["flags"].append("title_from_acq_local")
        if any(rj[0] == pid for rj in rejected):
            rec["flags"].append("acq_id_rejected_title_mismatch")
        if ov:
            rec.setdefault("provenance", []).append(ov["source"])
        rec.setdefault("provenance", []).append("arxiv_api" if ent else "local_only")
        missing = [k for k in ("title", "arxiv_year", "venue", "authors")
                   if rec.get(k) in (None, "", [])]
        if missing:
            rec["flags"].append("metadata_incomplete:" + ",".join(missing))
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--acq", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recs = build_manifest(args.texts, args.acq)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(recs, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for r in recs:
        print(f"[{r['paper_id']}] {(r['title'] or 'TITLE?')[:58]} | arXiv {r['arxiv_id']} "
              f"{r['arxiv_year']} | venue {(r['venue'] or '-')[:30]} {r['venue_year'] or ''} | "
              f"auth {len(r['authors'] or [])} | {';'.join(r['flags'])}", flush=True)
    complete = sum(1 for r in recs if not any(f.startswith("metadata_incomplete") for f in r["flags"]))
    print(f"\nsaved {len(recs)} papers -> {args.out}")
    print(f"complete(title+arxiv_year+venue+authors): {complete}/{len(recs)}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
