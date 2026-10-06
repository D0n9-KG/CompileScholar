# -*- coding: utf-8 -*-
"""Full-text channels. Each is fetch(want) -> Fetched | None (None = this channel does not have it; a transient failure
raises). `want` carries the paper's arXiv id, version, DOI.

  arxiv_nas     <arxiv_pdf_mirror>/<yymm>/<id>vN.pdf (new style) — 0704..2509, cs v1 coverage ~99 % (measured);
                old-style ids are not in the mirror. Pointer only, nothing copied.
  arxiv_gcs     storage.googleapis.com/arxiv-dataset/arxiv/{arxiv/pdf/<yymm>/<id>vN.pdf | <archive>/pdf/<yymm>/<num>vN.pdf}
                — public, no auth, to the current month; the PDF is downloaded into data/library/pdf/.
  scihub_local  the local Sci-Hub archive by DOI (internal use only; never in CS main experiments or a publication).
  oa_pdf        OpenAlex works/doi:<doi> best_oa_location.pdf_url (needs a DOI); downloaded.
Order per paper (acquire.run.plan): arXiv papers read the requested arXiv version from NAS, then GCS; papers with
only a DOI try oa_pdf, then scihub_local (unless the run excludes it)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..core import paths
from ..sources import http, scihub

GCS = "https://storage.googleapis.com/arxiv-dataset/arxiv"


@dataclass
class Want:
    paper_id: str
    arxiv: str | None = None
    version: int = 1
    doi: str | None = None


@dataclass
class Fetched:
    channel: str
    version: int            # arXiv version, or 0 for the version of record
    data: bytes
    pointer: dict           # how to find the bytes again without copying them


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _new_style(aid: str) -> bool:
    return bool(re.fullmatch(r"\d{4}\.\d{4,5}", aid or ""))


def arxiv_nas(w: Want) -> Fetched | None:
    if not w.arxiv or not _new_style(w.arxiv):
        return None
    p = paths.resource("arxiv_pdf_mirror") / w.arxiv.split(".")[0] / f"{w.arxiv}v{w.version}.pdf"
    try:
        b = p.read_bytes()
    except FileNotFoundError:
        return None
    rel = f"{w.arxiv.split('.')[0]}/{w.arxiv}v{w.version}.pdf"
    return Fetched("arxiv_nas", w.version, b, {"channel": "arxiv_nas", "path": rel})


def gcs_url(aid: str, version: int) -> str:
    if _new_style(aid):
        return f"{GCS}/arxiv/pdf/{aid.split('.')[0]}/{aid}v{version}.pdf"
    arch, _, num = aid.partition("/")              # cs/0408007 -> cs/pdf/0408/0408007v1.pdf
    return f"{GCS}/{arch.split('.')[0]}/pdf/{num[:4]}/{num}v{version}.pdf"


def arxiv_gcs(w: Want) -> Fetched | None:
    if not w.arxiv:
        return None
    url = gcs_url(w.arxiv, w.version)
    r = http.get("gcs", url, cache=False, timeout=120)
    if r.status in (403, 404):                       # GCS answers 404 (or 403 without listing rights) for no object
        return None
    if not r.ok:
        raise http.Transient(f"gcs {r.status} {url}")
    return Fetched("arxiv_gcs", w.version, r.content, {"channel": "arxiv_gcs", "url": url})


def scihub_local(w: Want) -> Fetched | None:
    if not w.doi:
        return None
    hit = scihub.lookup(w.doi)
    if hit is None:
        return None
    return Fetched("scihub_local", 0, scihub.read(hit), scihub.pointer(hit))


def oa_pdf(w: Want) -> Fetched | None:
    if not w.doi:
        return None
    r = http.get("openalex", f"https://api.openalex.org/works/doi:{w.doi}",
                 params={"select": "best_oa_location,open_access"})
    if r.status == 404 or not r.ok:
        return None
    loc = (r.json() or {}).get("best_oa_location") or {}
    url = loc.get("pdf_url")
    if not url:
        return None
    try:
        f = http.get("oa_pdf", url, cache=False, timeout=120)
    except http.Transient:
        return None                                  # a publisher host that keeps failing is not a transient of ours
    if not f.ok or not f.content.startswith(b"%PDF"):
        return None
    return Fetched("oa_pdf", 0, f.content, {"channel": "oa_pdf", "url": url})


CHANNELS = {"arxiv_nas": arxiv_nas, "arxiv_gcs": arxiv_gcs, "scihub_local": scihub_local, "oa_pdf": oa_pdf}
