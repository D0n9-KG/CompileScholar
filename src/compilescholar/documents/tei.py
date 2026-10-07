# -*- coding: utf-8 -*-
"""GROBID TEI (+ the PDF's hyperref citation links) -> units, sentences, bibliography and citation pairs: the fast
tier's document (INTEGRATED-SYSTEM-1005 v2.2 item 1).

  units      section headings, paragraphs (prose), figure / table captions, formulas — each with page and bbox
  sentences  every <s> of a body paragraph, with a stable id <doc>#s<n>, its unit, page and bbox; sentences in figures,
             tables and their notes are not prose and are kept out, and so is a table row GROBID left inside a
             paragraph (table_like) (measured: about half of the "false" citation pairs against the arXiv v1 HTML
             gold were table / figure text)
  entries    bibliography: xml:id -> raw reference text, title, year, DOI / arXiv id (from GROBID's biblStruct)
  cites      (sentence id, entry key): in a paper with hyperref links (>= 5 resolved to entries) a link inside the
             sentence decides the entry and GROBID's refs that no link covers are dropped; otherwise GROBID's ref
             targets. Measured on 61 cs v1 papers: recall 0.934 / precision 0.919 (links) vs 0.873 / 0.913 (GROBID).
Coordinates: GROBID "page,x,y,w,h;..." (1-based page, top-left origin, PDF points); links: PyMuPDF rect (0-based page,
top-left origin) — compared in one frame."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from lxml import etree

from ..core import ids

T = "{http://www.tei-c.org/ns/1.0}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
MIN_LINKS = 5


@dataclass
class Unit:
    uid: str
    kind: str              # section | para | caption | formula
    section: str
    text: str
    page: int | None = None
    bbox: list | None = None


@dataclass
class Sentence:
    sid: str
    unit: str
    text: str
    page: int | None = None
    bbox: list | None = None


@dataclass
class Doc:
    title: str = ""
    abstract: str = ""
    units: list = field(default_factory=list)
    sentences: list = field(default_factory=list)
    entries: dict = field(default_factory=dict)      # key -> {raw, title, year, doi, arxiv}
    cites: list = field(default_factory=list)        # (sid, key, n_keys_in_sentence)
    links_used: bool = False

    def to_dict(self) -> dict:
        return {"title": self.title, "abstract": self.abstract, "units": [asdict(u) for u in self.units],
                "sentences": [asdict(s) for s in self.sentences], "entries": self.entries,
                "cites": [list(c) for c in self.cites], "links_used": self.links_used}


def boxes(coords: str | None) -> list[tuple]:
    out = []
    for part in (coords or "").split(";"):
        f = part.split(",")
        if len(f) == 5:
            p, x, y, w, h = int(f[0]), *map(float, f[1:])
            out.append((p - 1, x, y, x + w, y + h))
    return out


def _bbox(bx: list[tuple]) -> tuple[int | None, list | None]:
    if not bx:
        return None, None
    p = bx[0][0]
    same = [b for b in bx if b[0] == p]
    return p, [round(min(b[1] for b in same), 1), round(min(b[2] for b in same), 1),
               round(max(b[3] for b in same), 1), round(max(b[4] for b in same), 1)]


def _text(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _prose(el) -> str:
    """Prose text with a space at every sentence boundary: GROBID writes no whitespace between <s> elements,
    so the bare itertext join fuses them ("...learning.This...") — measured on the 260-paper smoke: abstracts
    arrived as one unbreakable blob and 98.4% of T1 quotes degenerated to the whole abstract. Used for the
    abstract and prose units; per-sentence records and bibliography entries keep _text (their content is one
    <s>/one bibl, no boundary to miss)."""
    ss = el.findall(f".//{T}s")
    return re.sub(r"\s+", " ", " ".join(_text(s) for s in ss)).strip() if ss else _text(el)


def _entry(b) -> dict:
    raw = b.find(f"{T}note[@type='raw_reference']")
    raw = _text(raw) if raw is not None else _text(b)
    t = b.find(f"{T}analytic/{T}title")
    if t is None or not _text(t):
        t = b.find(f"{T}monogr/{T}title")
    d = b.find(f".//{T}date[@type='published']")
    doi = next((x.text for x in b.iter(f"{T}idno") if x.get("type") == "DOI" and x.text), None)
    ax = next((x.text for x in b.iter(f"{T}idno") if x.get("type") == "arXiv" and x.text), None)
    return {"raw": raw, "title": _text(t) if t is not None else "",
            "year": (d.get("when") or "")[:4] if d is not None else "",
            "doi": ids.normalize_doi(doi) if doi else (ids.find_dois(raw) or [None])[0],
            "arxiv": ids.normalize_arxiv(ax) if ax else (ids.find_arxiv(raw) or [None])[0]}


def _dest_entry(starts: list[tuple], page: int, x: float, y: float) -> str | None:
    """The entry a link destination points at: the entry whose first line starts nearest the anchor on that page, in
    the same column (within 60 pt across, 25 pt up or down)."""
    best, bd = None, 1e9
    for key, (p, x0, y0) in starts:
        if p != page or abs(x0 - x) > 60 or not (-25 <= y0 - y <= 25):
            continue
        dd = abs(y0 - y) + 0.1 * abs(x0 - x)
        if dd < bd:
            best, bd = key, dd
    return best


_NUMTOK = re.compile(r"[-+±]?[\d.,%]+[a-zA-Z%]?|[-–]+|\[\d+(?:,\d+)*\]")
TABLE_NUM_SHARE = 0.4


def table_like(text: str) -> bool:
    """A "sentence" that is a table row run into the text (GROBID leaves some tables inside paragraphs): at least 8
    tokens, >= TABLE_NUM_SHARE of them numbers / dashes / bracketed markers. Structural, not semantic. Measured on the
    61-paper gold: citation precision 0.919 -> 0.930, recall unchanged (0.930), the same for shares 0.3-0.5."""
    toks = text.split()
    return len(toks) >= 8 and sum(bool(_NUMTOK.fullmatch(t)) for t in toks) / len(toks) >= TABLE_NUM_SHARE


def _inside(px, cx, cy, bx) -> bool:
    return any(p == px and a0 - 1 <= cx <= a1 + 1 and b0 - 2 <= cy <= b1 + 2 for p, a0, b0, a1, b1 in bx)


def parse(tei: bytes, links: list | None = None, doc_id: str = "doc") -> Doc:
    root = etree.fromstring(tei)
    doc = Doc()
    h = root.find(f".//{T}teiHeader")
    if h is not None:
        t = h.find(f".//{T}titleStmt/{T}title")
        doc.title = _text(t) if t is not None else ""
        a = h.find(f".//{T}profileDesc/{T}abstract")
        doc.abstract = _prose(a) if a is not None else ""
    starts = []
    for b in root.iter(f"{T}biblStruct"):
        key = b.get(XML_ID)
        if not key or b.getparent() is None or b.getparent().tag != f"{T}listBibl":
            continue
        doc.entries[key] = _entry(b)
        bx = boxes(b.get("coords"))
        if bx:
            starts.append((key, (bx[0][0], bx[0][1], bx[0][2])))
    linked = []
    for pg, (x0, y0, x1, y1), tp, dx, dy in (links or []):
        key = _dest_entry(starts, tp, dx, dy)
        if key:
            linked.append((pg, (x0 + x1) / 2, (y0 + y1) / 2, key))
    doc.links_used = len(linked) >= MIN_LINKS
    body = root.find(f".//{T}text/{T}body")
    back = root.find(f".//{T}text/{T}back")
    n = {"section": 0, "para": 0, "caption": 0, "formula": 0, "s": 0}

    def unit(kind, section, text, el):
        n[kind] += 1
        pg, bb = _bbox(boxes(el.get("coords")))
        u = Unit(f"{doc_id}#{kind}{n[kind]}", kind, section, text, pg, bb)
        doc.units.append(u)
        return u

    for part in (body, back):
        if part is None:
            continue
        for div in part.iter(f"{T}div"):
            if div.get("type") in ("references", "acknowledgement", "funding", "availability"):
                continue
            head = div.find(f"{T}head")
            sec = (f"{head.get('n')} " if head is not None and head.get("n") else "") + (_text(head) if head is not None
                                                                                          else "")
            if head is not None and sec.strip():
                unit("section", sec, sec, head)
            for el in div:
                if el.tag == f"{T}p":
                    u = unit("para", sec, _prose(el), el)
                    for s in el.iter(f"{T}s"):
                        n["s"] += 1
                        pg, bb = _bbox(boxes(s.get("coords")))
                        sent = Sentence(f"{doc_id}#s{n['s']}", u.uid, _text(s), pg, bb)
                        doc.sentences.append(sent)
                        if not table_like(sent.text):          # a table row is not a citing sentence
                            doc.cites += [(sent.sid, k, 0) for k in _keys(s, linked, doc)]
                elif el.tag == f"{T}formula":
                    unit("formula", sec, _text(el), el)
    for fig in root.iter(f"{T}figure"):
        d = fig.find(f"{T}figDesc")
        if d is not None and _text(d):
            lab = fig.find(f"{T}head")
            unit("caption", "", ((_text(lab) + " ") if lab is not None else "") + _prose(d), fig)
    per = {}
    for sid, k, _ in doc.cites:
        per.setdefault(sid, []).append(k)
    doc.cites = [(sid, k, len(per[sid])) for sid, k, _ in doc.cites]
    return doc


def _keys(s, linked, doc: Doc) -> list[str]:
    if doc.links_used:
        bx = boxes(s.get("coords"))
        keys = [k for pg, cx, cy, k in linked if _inside(pg, cx, cy, bx)]
    else:
        keys = [r.get("target", "").lstrip("#") for r in s.iter(f"{T}ref") if r.get("type") == "bibr"]
    return list(dict.fromkeys(k for k in keys if k in doc.entries))
