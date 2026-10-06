# -*- coding: utf-8 -*-
"""Is this PDF the paper we asked for? (FITNESS-LIBRARY-ACQUIRE: the old check let scanned PDFs through, failed every
non-ASCII title, matched "Deep learning" to a protein-folding paper and α- to β-synuclein.)

Evidence from the first two pages (PyMuPDF text layer; PDF metadata title):
  1. identifier stamp — the arXiv id (arXiv:<id>vN in the margin) or the DOI printed on the page:
       the paper's own id -> ok; a different arXiv id / the paper's DOI absent while another DOI is the only one -> a
       strong sign of a different paper, but only checked together with the title (journal pages print DOIs of
       cited works too)
  2. title — every token of the registry title (norm_title, Unicode) in order inside the page text, or token coverage
  3. authors — registry surnames found on the page
Decision: ok when (own id) or (title coverage >= 0.9 and a surname, or no authors known) ; mismatch when title
coverage < 0.4 and no own id; anything else (a scanned PDF with no text layer, partial coverage) -> "unsure", which
the caller sends to the LLM verdict (llm_verdict) with the page text and the registry record."""
from __future__ import annotations

import re

from ..core import ids

PAGES = 2
OK_COVER, BAD_COVER = 0.9, 0.4


def first_pages(data: bytes, pages: int = PAGES) -> tuple[str, str]:
    """(text of the first pages, metadata title)."""
    import pymupdf
    with pymupdf.open(stream=data, filetype="pdf") as d:
        text = "\n".join(d[i].get_text() for i in range(min(pages, d.page_count)))
        return text, (d.metadata or {}).get("title") or ""


def _cover(title_tokens: list[str], page_tokens: set[str]) -> float:
    toks = [t for t in title_tokens if len(t) > 1 or t.isdigit()]
    return sum(t in page_tokens for t in toks) / len(toks) if toks else 0.0


def check(data: bytes, title: str, surnames: list[str], arxiv: str | None, doi: str | None) -> dict:
    """{'verdict': ok | mismatch | unsure, 'why': ..., 'cover': .., 'surname': .., 'text': first-page text}"""
    try:
        text, meta_title = first_pages(data)
    except Exception as e:                                    # not a PDF / broken PDF
        return {"verdict": "mismatch", "why": f"unreadable pdf: {type(e).__name__}", "text": ""}
    flat = re.sub(r"-\s*\n\s*", "", text)                    # line-end hyphenation
    page_norm = ids.norm_title(flat + " " + meta_title, greek=True)
    # also split letters from digits: affiliation marks glue to names ("Chen1,4", "Müller*2")
    ptoks = set(page_norm.split()) | set(re.findall(r"[^\W\d_]+|\d+", page_norm))
    stamps_ax = set(ids.find_arxiv(flat))
    stamps_doi = set(ids.find_dois(flat))
    own = (arxiv and arxiv in stamps_ax) or (doi and doi in stamps_doi)
    cover = _cover(ids.norm_title(title, greek=True).split(), ptoks)
    sur = [s for s in (ids.norm_title(x) for x in surnames) if s]
    sur_hit = any(all(p in ptoks for p in s.split()) for s in sur) if sur else None
    out = {"cover": round(cover, 2), "surname": sur_hit, "own_id": bool(own), "text": flat[:4000]}
    if len(flat.strip()) < 200 and not meta_title:
        return {**out, "verdict": "unsure", "why": "no text layer"}
    if own and cover >= BAD_COVER:
        return {**out, "verdict": "ok", "why": "own identifier on the page"}
    if cover >= OK_COVER and (sur_hit or sur_hit is None):
        return {**out, "verdict": "ok", "why": "title and author on the page"}
    if cover < BAD_COVER and not own:
        return {**out, "verdict": "mismatch", "why": f"title coverage {cover:.2f}"}
    return {**out, "verdict": "unsure", "why": f"title coverage {cover:.2f}, surname {sur_hit}, own id {bool(own)}"}


PROMPT = """A PDF was downloaded for the bibliographic record below. Decide from the PDF's first page whether it is that
paper (any version: preprint or published, possibly with a slightly different title), or a different document (another
paper, a journal issue's table of contents, an erratum, a cover page, a book containing many chapters).

Record
title: {title}
authors: {authors}
identifiers: {idents}

First page of the PDF (text layer, may be garbled)
{text}

Answer with JSON only: {{"same": true|false, "confidence": 0.0-1.0, "reason": "<one sentence>"}}"""


def llm_verdict(res: dict, title: str, surnames: list[str], idents: list[str]) -> dict:
    """Resolve an "unsure" check with the local model: ok_llm when it says same with confidence >= 0.8, else mismatch."""
    from ..llm import client as LC
    if not res.get("text", "").strip():
        return {**res, "verdict": "mismatch", "why": res["why"] + "; no text for a verdict"}
    o = LC.call_json(PROMPT.format(title=title, authors=", ".join(surnames[:12]), idents=", ".join(idents),
                                   text=res["text"][:3000]), provider="local", max_tokens=200,
                     template="acquire.same_pdf.v1",
                     validate=lambda o: isinstance(o, dict) and isinstance(o.get("same"), bool))
    if o and o["same"] and float(o.get("confidence", 0)) >= 0.8:
        return {**res, "verdict": "ok_llm", "why": f"llm: {o.get('reason', '')}"}
    return {**res, "verdict": "mismatch", "why": f"llm: {(o or {}).get('reason', 'no verdict')}"}
