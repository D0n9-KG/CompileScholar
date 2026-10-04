# -*- coding: utf-8 -*-
"""Local title index over the arXiv OAI metadata snapshot (one JSON object per line; ~2.9M records, read-only on the
share). Built once by streaming the file and kept as a compact TSV in the cache dir:
  norm_title_prefix40 \t arxiv_id \t year \t first_author_surname \t norm_title
Used by W2 P0-2 to resolve bibliography entries to arXiv ids at zero API cost.

Location: CS_ARXIV_SNAPSHOT (default: the share path used since 09-27).
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from ..core import paths

DEFAULT = r"\\192.168.199.138\Share400T\pub\LLM_Data\data\JournalPapers\arXiv_Dataset\arxiv-metadata-oai-snapshot.json"


def snapshot_path() -> Path:
    return Path(os.environ.get("CS_ARXIV_SNAPSHOT") or DEFAULT)


def index_path() -> Path:
    return paths.cache() / "arxiv_snapshot_title_index.tsv"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (s or "").lower())).strip()


def _year(rec: dict) -> str:
    v = (rec.get("versions") or [{}])[0].get("created") or ""
    m = re.search(r"(\d{4})", v)
    if m:
        return m.group(1)
    m = re.match(r"(\d{2})(\d{2})\.", rec.get("id") or "")
    return ("20" + m.group(1)) if m else ""


def _first_surname(rec: dict) -> str:
    ap = rec.get("authors_parsed") or []
    if ap and ap[0]:
        return norm(ap[0][0]).split(" ")[-1] if ap[0][0] else ""
    a = (rec.get("authors") or "").split(",")[0].strip()
    return norm(a).split(" ")[-1] if a else ""


def build(limit: int | None = None, report_every: int = 500_000) -> dict:
    """Stream the snapshot once and write the index. Returns timing / count stats."""
    src, dst = snapshot_path(), index_path()
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0, n, nbytes = time.time(), 0, 0
    tmp = dst.with_suffix(".tmp")
    with open(src, "rb") as fi, open(tmp, "w", encoding="utf-8", newline="\n") as fo:
        for line in fi:
            nbytes += len(line)
            try:
                rec = json.loads(line)
            except Exception:
                continue
            t = norm(rec.get("title") or "")
            if not t:
                continue
            fo.write(f"{t[:40]}\t{rec.get('id')}\t{_year(rec)}\t{_first_surname(rec)}\t{t}\n")
            n += 1
            if report_every and n % report_every == 0:
                print(f"  {n:,} records, {nbytes / 1e9:.2f} GB, {time.time() - t0:.0f}s", flush=True)
            if limit and n >= limit:
                break
    os.replace(tmp, dst)
    return {"records": n, "bytes": nbytes, "seconds": round(time.time() - t0, 1)}


class TitleIndex:
    def __init__(self, path: Path | None = None):
        self.by_prefix: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
        with open(path or index_path(), encoding="utf-8") as f:
            for line in f:
                p, aid, year, sur, full = line.rstrip("\n").split("\t")
                self.by_prefix[p].append((aid, year, sur, full))

    def lookup(self, title: str, year: int | str | None = None, raw_entry: str = "") -> str | None:
        """arXiv id when exactly one snapshot record matches: same normalized-title 40-char prefix, year within ±1
        (when both known), and the record's first-author surname appears in the raw bibliography entry (when given).
        Ambiguous or unmatched -> None."""
        t = norm(title)
        if len(t) < 12:
            return None
        cands = self.by_prefix.get(t[:40], [])
        out = []
        raw = norm(raw_entry)
        for aid, y, sur, full in cands:
            if len(t) >= 40 and not (full.startswith(t[:60]) or t.startswith(full[:60])):
                continue
            if year and y and str(year).isdigit() and abs(int(y) - int(year)) > 1:
                continue
            if raw and sur and len(sur) > 1 and sur not in raw.split():
                continue
            out.append(aid)
        out = list(dict.fromkeys(out))
        return out[0] if len(out) == 1 else None
