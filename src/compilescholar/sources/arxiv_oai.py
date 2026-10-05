# -*- coding: utf-8 -*-
"""arXiv OAI-PMH harvester (set=cs, metadataPrefix=arXivRaw) -> paper metadata with first-version dates.

The local OAI snapshot (sources.arxiv_snapshot) ends at 2404.03658, so everything after it comes from here.
Endpoint: https://oaipmh.arxiv.org/oai. Records are harvested by datestamp windows and written as JSONL, one file per
window, under cache/arxiv_oai/<set>/<from>_<until>.jsonl; a window whose file exists is skipped (resumable).

Why arXivRaw and not arXiv: the `arXiv` format's <created> is not the first-version date (1601.04794 reports
created=2026-08-30, while its v1 is 2016-01-19; measured 10-05). arXivRaw lists every version with its date, so
v1_date is exact.

Per record we keep: arxiv_id, v1_date (YYYY-MM-DD), versions (count), title, abstract, authors (surname list, first
author first), categories (list, primary first), doi.
OAI-PMH asks for no more than one request every few seconds; we pause PAUSE seconds between requests and back off on
503 with the server's Retry-After."""
from __future__ import annotations

import email.utils
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

from ..core import paths

ENDPOINT = "https://oaipmh.arxiv.org/oai"
NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "raw": "http://arxiv.org/OAI/arXivRaw/"}
PAUSE = 3.0
UA = "CompileScholar-harvester (research; respects OAI-PMH flow control)"


def out_dir(set_spec: str = "cs") -> Path:
    return paths.cache() / "arxiv_oai" / set_spec


def _text(el, path: str) -> str:
    x = el.find(path, NS)
    return re.sub(r"\s+", " ", x.text).strip() if x is not None and x.text else ""


def _rfc2822_day(s: str) -> str:
    """'Tue, 19 Jan 2016 04:10:52 GMT' -> '2016-01-19' ('' when unparseable)."""
    try:
        return email.utils.parsedate_to_datetime(s).date().isoformat()
    except (TypeError, ValueError):
        return ""


def surnames(authors: str) -> list[str]:
    """arXivRaw author string ('A. Smith, B. Jones and C. Lee (MIT)') -> ['Smith', 'Jones', 'Lee']."""
    s = re.sub(r"\([^()]*\)", " ", authors or "")
    parts = re.split(r",\s*|\s+and\s+", s)
    out = []
    for p in parts:
        w = [x for x in p.strip().split() if x]
        if w:
            out.append(w[-1].strip(".,;"))
    return [x for x in out if x]


def parse_record(rec: ET.Element) -> dict | None:
    """One <record> (arXivRaw) -> dict, or None for deleted records."""
    header = rec.find("oai:header", NS)
    if header is not None and header.get("status") == "deleted":
        return None
    md = rec.find("oai:metadata/raw:arXivRaw", NS)
    if md is None:
        return None
    versions = md.findall("raw:version", NS)
    v1 = next((v for v in versions if v.get("version") == "v1"), versions[0] if versions else None)
    v1_date = _rfc2822_day(_text(v1, "raw:date")) if v1 is not None else ""
    return {"arxiv_id": _text(md, "raw:id"), "v1_date": v1_date, "versions": len(versions),
            "title": _text(md, "raw:title"), "abstract": _text(md, "raw:abstract"),
            "authors": surnames(_text(md, "raw:authors")), "categories": _text(md, "raw:categories").split(),
            "doi": _text(md, "raw:doi") or None}


def _get(url: str, tries: int = 8) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 503:
                wait = int(e.headers.get("Retry-After") or 10)
                time.sleep(min(300, max(wait, 5)))
                continue
            if e.code >= 500:
                time.sleep(min(120, 5 * 2 ** i))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(min(120, 5 * 2 ** i))
    raise RuntimeError(f"OAI-PMH request kept failing: {url}")


def harvest_window(start: date, end: date, set_spec: str = "cs") -> tuple[Path, int]:
    """All records with datestamp in [start, end] (inclusive, per OAI-PMH) -> one JSONL file. Skips if it exists."""
    d = out_dir(set_spec)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{start.isoformat()}_{end.isoformat()}.jsonl"
    if p.exists():
        return p, sum(1 for _ in open(p, encoding="utf-8"))
    params = {"verb": "ListRecords", "metadataPrefix": "arXivRaw", "set": set_spec,
              "from": start.isoformat(), "until": end.isoformat()}
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    n, tmp = 0, p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fo:
        while url:
            root = ET.fromstring(_get(url))
            err = root.find("oai:error", NS)
            if err is not None:
                if err.get("code") == "noRecordsMatch":
                    break
                raise RuntimeError(f"OAI-PMH error {err.get('code')}: {err.text}")
            lr = root.find("oai:ListRecords", NS)
            for rec in lr.findall("oai:record", NS) if lr is not None else []:
                r = parse_record(rec)
                if r and r["arxiv_id"]:
                    fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
            tok = lr.find("oai:resumptionToken", NS) if lr is not None else None
            url = (ENDPOINT + "?" + urllib.parse.urlencode({"verb": "ListRecords", "resumptionToken": tok.text})
                   if tok is not None and tok.text else None)
            time.sleep(PAUSE)
    tmp.replace(p)
    return p, n


def harvest(start: date, end: date, days: int = 7, set_spec: str = "cs", log=print) -> int:
    """Harvest [start, end] in windows of `days` days. Returns the number of records written in this call."""
    total, cur = 0, start
    while cur <= end:
        w_end = min(end, cur + timedelta(days=days - 1))
        p, n = harvest_window(cur, w_end, set_spec)
        total += n
        log(f"[oai] {cur}..{w_end}: {n} records -> {p.name}")
        cur = w_end + timedelta(days=1)
    return total


def iter_records(set_spec: str = "cs"):
    """All harvested records; a paper re-appears when it is updated, the latest window's copy wins (v1_date is the
    same in every copy)."""
    seen: dict[str, dict] = {}
    for p in sorted(out_dir(set_spec).glob("*.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            seen[r["arxiv_id"]] = r
    return seen.values()
