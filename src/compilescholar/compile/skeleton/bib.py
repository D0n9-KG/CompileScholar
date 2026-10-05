# -*- coding: utf-8 -*-
"""W2 P0-1: survey bibliography splitting and in-text marker resolution (deterministic, no LLM).

Input: a survey's full text as rendered by LaTeXML -> markdown (base_kb/survey_texts/*.md). Three bibliography
renderings occur (measured on 135 surveys): numeric "[12]" headers (68), author-year "Name et al. [2018]" /
"Name et al. (2018)" headers (60), bare numeric "12" headers whose in-text markers are superscripts rendered inline (7).

Outputs, per survey:
  entries : key -> {"raw": entry text, "arxiv": id | None, "doi": doi | None}
            key = int for numeric styles, (surname_lower, year) for author-year
  markers : list of {"key", "start", "end"} for every in-text citation marker in the body
  style   : "numeric" | "author-year" | "numeric-bare"
Author-year keys that occur more than once in the bibliography are ambiguous and never resolved (no guessing).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from ...core import ids

REF_HEADING = re.compile(r"\n#+\s*(?:\d+\.?\s*)?(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n")
NUM_ENTRY = re.compile(r"^\s*\[?(\d{1,4})\]?\s*$", re.M)
BRACKET_NUM_ENTRY = re.compile(r"^\s*\[\d+\]\s*$", re.M)
AY_ENTRY = re.compile(r"^\s*([A-Z][^\n\[\]()]{0,80}?)\s*[\[(]((?:19|20)\d\d[a-z]?)[\])]\s*$", re.M)
NUM_MARK = re.compile(r"\[\s*(\d+(?:\s*[,–\-]\s*\d+)*)\s*\]")
BARE_MARK = re.compile(r"(?<=[A-Za-z\)]) (\d{1,3}(?:\s?,\s?\d{1,3})*)(?=\s*[ .,;:)])")
AY_MARK = re.compile(r"([A-Z][A-Za-z'\-]+(?:\s(?:et\s?al\.|and\s[A-Z][A-Za-z'\-]+))?)\s?[\[(]((?:19|20)\d\d[a-z]?)[\])]")
AY_PAREN_GROUP = re.compile(r"\(([^()]{4,400})\)")
AY_PAREN = re.compile(r"([A-Z][A-Za-z'\-]+)(?:\s+(?:et\s?al\s?\.?|and\s+(?:et\s?al\.?|[A-Z][A-Za-z'\-]+)))?\s*,\s*((?:19|20)\d\d[a-z]?)")
MAX_RANGE = 20


@dataclass
class Bibliography:
    style: str
    entries: dict = field(default_factory=dict)
    markers: list = field(default_factory=list)
    ambiguous_keys: set = field(default_factory=set)

    def resolved_markers(self) -> list:
        return [m for m in self.markers if m["key"] in self.entries and m["key"] not in self.ambiguous_keys]


def _expand(s: str) -> list[int]:
    out = []
    for part in re.split(r"\s*,\s*", s):
        m = re.match(r"(\d+)\s*[–\-]\s*(\d+)$", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if 0 < b - a <= MAX_RANGE:
                out += list(range(a, b + 1))
        elif part.strip().isdigit():
            out.append(int(part))
    return out


def _ay_key(name: str, year: str) -> tuple[str, str]:
    return re.sub(r"\s.*", "", name.strip()).lower(), year


def _ids(raw: str) -> tuple[str | None, str | None]:
    """First arXiv id and first (non-arXiv) DOI in an entry, normalised by core.ids (old-style arXiv ids, DOI URLs,
    trailing punctuation and SICI DOIs are handled there)."""
    a, d = ids.find_arxiv(raw), ids.find_dois(raw)
    return (a[0] if a else None), (d[0] if d else None)


def split_body_refs(text: str) -> tuple[str, str] | None:
    """Body and reference section (the LAST References heading; earlier ones are usually a table of contents)."""
    m = None
    for m in REF_HEADING.finditer(text):
        pass
    if not m:
        return None
    return text[:m.start()], text[m.end():]


def parse(text: str) -> Bibliography | None:
    text = text.replace("\xa0", " ")
    parts = split_body_refs(text)
    if parts is None:
        return None
    body, refs = parts
    n_num, n_ay = len(NUM_ENTRY.findall(refs)), len(AY_ENTRY.findall(refs))
    if n_num >= n_ay:
        heads = list(NUM_ENTRY.finditer(refs))
        bare = not BRACKET_NUM_ENTRY.search(refs)
        bib = Bibliography(style="numeric-bare" if bare else "numeric")
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(refs)
            raw = re.sub(r"\s+", " ", refs[h.end():end]).strip()
            ax, doi = _ids(raw)
            bib.entries[int(h.group(1))] = {"raw": raw, "arxiv": ax, "doi": doi}
        found = [(m.start(), m.end(), n) for m in NUM_MARK.finditer(body) for n in _expand(m.group(1))]
        if bare and len(found) < 10:
            found = [(m.start(1), m.end(1), n) for m in BARE_MARK.finditer(body) for n in _expand(m.group(1))]
        bib.markers = [{"key": n, "start": s, "end": e} for s, e, n in found]
        return bib
    heads = list(AY_ENTRY.finditer(refs))
    bib = Bibliography(style="author-year")
    counts = Counter()
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(refs)
        k = _ay_key(h.group(1), h.group(2))
        counts[k] += 1
        raw = re.sub(r"\s+", " ", refs[h.end():end]).strip()
        ax, doi = _ids(raw)
        bib.entries[k] = {"raw": raw, "arxiv": ax, "doi": doi}
    bib.ambiguous_keys = {k for k, c in counts.items() if c > 1}
    found = [(m.start(), m.end(), _ay_key(m.group(1), m.group(2))) for m in AY_MARK.finditer(body)]
    for g in AY_PAREN_GROUP.finditer(body):
        for m in AY_PAREN.finditer(g.group(1)):
            found.append((g.start(1) + m.start(), g.start(1) + m.end(), _ay_key(m.group(1), m.group(2))))
    seen = set()
    for s, e, k in sorted(found):
        if (s, k) not in seen:
            seen.add((s, k))
            bib.markers.append({"key": k, "start": s, "end": e})
    return bib


def markers_in(text: str, bib: Bibliography) -> list:
    """Keys cited inside an arbitrary passage (e.g. a record's verbatim quote) using the survey's marker style."""
    text = text.replace("\xa0", " ")
    if bib.style == "author-year":
        keys = [_ay_key(a, y) for a, y in AY_MARK.findall(text)]
        for g in AY_PAREN_GROUP.findall(text):
            keys += [_ay_key(a, y) for a, y in AY_PAREN.findall(g)]
    else:
        keys = [n for g in NUM_MARK.findall(text) for n in _expand(g)]
    return [k for k in dict.fromkeys(keys) if k in bib.entries and k not in bib.ambiguous_keys]
