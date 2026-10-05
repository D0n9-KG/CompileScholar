# -*- coding: utf-8 -*-
"""Citation sentences from full-text Markdown (MinerU / LaTeXML renderings), deterministic, no LLM.

Input: a paper's Markdown. Output: the bibliography (reusing compile.skeleton.bib for entry splitting and marker
styles) and one CitationSentence per (sentence, cited entry key). Handles, beyond bib.parse:
  - reference headings that bib.REF_HEADING misses: no '#', a numbered heading ("7. REFERENCES"), or the first entry
    glued to the heading line ("References Bi, J.; ..."); the reference block need not be the last thing in the paper
    (appendices may follow) — the block ends at the next top-level heading after it;
  - numeric entries written "1. Name ..." (MinerU) in addition to "[1]" lines;
  - false numeric markers inside LaTeX ("w_{2}[0]", "$x[3]$"): a marker directly preceded by '_', '^', '}' or a
    letter/digit without a space, or inside $...$, is dropped;
  - alpha-label markers ("[AZLS19]") are recognized when the reference list uses the same labels.
A sentence is cut at '.', '?', '!' followed by whitespace and an upper-case letter / bracket, not after common
abbreviations (et al., e.g., i.e., Fig., Eq., Sec., vs.).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..compile.skeleton import bib as B

HEADING_ANY = re.compile(
    r"(?m)^(?:#+\s*)?(?:\d+\.?\s*)?(References|REFERENCES|Reference|Bibliography|BIBLIOGRAPHY)\b[ \t]*:?[ \t]*")
TOP_HEADING = re.compile(r"(?m)^#{1,2}\s+\S")
DOT_ENTRY = re.compile(r"(?m)^\s*(\d{1,4})\.\s+(?=[A-ZÀ-ɏ])")
BRACKET_ENTRY = re.compile(r"(?m)^\s*\[(\d{1,4})\]\s*")
ALPHA_ENTRY = re.compile(r"(?m)^\s*\[([A-Z][A-Za-z+]{1,8}\d{2}[a-z]?)\]\s*")
ALPHA_MARK = re.compile(r"\[([A-Z][A-Za-z+]{1,8}\d{2}[a-z]?(?:\s*,\s*[A-Z][A-Za-z+]{1,8}\d{2}[a-z]?)*)\]")
MATH = re.compile(r"\$[^$\n]{0,300}\$")
ABBREV = re.compile(r"(?:\bet al|\be\.g|\bi\.e|\bFig|\bFigs|\bEq|\bEqs|\bSec|\bvs|\bcf|\bResp|\bNo|\bApp|\bTab)\.$")


@dataclass
class CitationSentence:
    sentence: str
    key: object           # bibliography key (int, (surname, year) or alpha label)
    n_keys: int           # number of distinct keys cited in the same sentence (>1 = grouped citation)
    group: tuple          # all keys cited in the sentence, in order
    offset: int           # character offset of the sentence in the body


YEAR_CAND = re.compile(r"(?<![\d/\-.])((?:19|20)\d\d)([a-z]?)(?![\d/\-])")


def _entry_lines(block: str) -> list[str]:
    """One-entry-per-line reference lists (MinerU renders each entry as a line ending in '   \\n')."""
    return [l.strip() for l in block.split("\n") if len(l.strip()) >= 30 and YEAR_CAND.search(l)]


def find_refs(text: str) -> tuple[str, str] | None:
    """(body, reference block). Uses the last reference heading whose following text looks like a reference list
    (at least 3 entries of any style, or 3 one-per-line entries carrying a year); the block ends at the next
    level-1/2 heading after it, if any."""
    best = None
    for m in HEADING_ANY.finditer(text):
        tail = text[m.end():]
        nxt = TOP_HEADING.search(tail, 1)
        block = tail[:nxt.start()] if nxt else tail
        n = (len(DOT_ENTRY.findall(block)) + len(BRACKET_ENTRY.findall(block)) + len(ALPHA_ENTRY.findall(block))
             + len(B.AY_ENTRY.findall(block)) + len(B.NUM_ENTRY.findall(block)))
        if n >= 3 or len(_entry_lines(block)) >= 3:
            best = (text[:m.start()], block)
    return best


def _numeric_entries(refs: str) -> dict:
    heads = list(BRACKET_ENTRY.finditer(refs)) or list(DOT_ENTRY.finditer(refs)) or list(B.NUM_ENTRY.finditer(refs))
    out = {}
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(refs)
        raw = re.sub(r"\s+", " ", refs[h.end():end]).strip()
        k = int(h.group(1))
        if raw and k not in out:
            ax, doi = B._ids(raw)
            out[k] = {"raw": raw, "arxiv": ax, "doi": doi}
    return out


def _alpha_entries(refs: str) -> dict:
    heads = list(ALPHA_ENTRY.finditer(refs))
    out = {}
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(refs)
        raw = re.sub(r"\s+", " ", refs[h.end():end]).strip()
        ax, doi = B._ids(raw)
        out[h.group(1)] = {"raw": raw, "arxiv": ax, "doi": doi}
    return out


def _mask_math(body: str) -> str:
    """Blank out $...$ spans (same length) so markers inside math never match."""
    return MATH.sub(lambda m: " " * (m.end() - m.start()), body)


def _numeric_markers(body: str) -> list[tuple[int, int, int]]:
    masked = _mask_math(body)
    out = []
    for m in B.NUM_MARK.finditer(masked):
        prev = masked[m.start() - 1] if m.start() else " "
        if prev in "_^}" or prev.isalnum():
            continue
        for n in B._expand(m.group(1)):
            out.append((m.start(), m.end(), n))
    return out


def parse(text: str) -> B.Bibliography | None:
    """Bibliography with style, entries and markers (same structure as compile.skeleton.bib.parse)."""
    text = text.replace("\xa0", " ")
    parts = find_refs(text)
    if parts is None:
        return None
    body, refs = parts
    alpha = _alpha_entries(refs)
    if len(alpha) >= 3:
        bib = B.Bibliography(style="alpha", entries=alpha)
        for m in ALPHA_MARK.finditer(_mask_math(body)):
            for lab in re.split(r"\s*,\s*", m.group(1)):
                bib.markers.append({"key": lab, "start": m.start(), "end": m.end()})
        return bib
    num = _numeric_entries(refs)
    n_ay = len(B.AY_ENTRY.findall(refs))
    numeric_marks = _numeric_markers(body)
    if num and (len(num) >= n_ay or len(numeric_marks) >= 10):
        bib = B.Bibliography(style="numeric", entries=num)
        bib.markers = [{"key": n, "start": s, "end": e} for s, e, n in numeric_marks]
        return bib
    # author-year: entries from bib.parse when its entry pattern fits ("Name (2018)" header lines), otherwise the
    # one-entry-per-line MinerU rendering; markers are scanned here (bib.parse misses bracketed groups)
    sub = B.parse(body + "\n# References\n" + refs)
    if sub is not None and sub.style == "author-year" and len(sub.entries) >= 3:
        entries, amb = sub.entries, sub.ambiguous_keys
    else:
        ay = _author_year_entries(refs)
        if not ay:
            return None
        entries, amb = ay["entries"], ay["ambiguous"]
    bib = B.Bibliography(style="author-year", entries=entries, ambiguous_keys=amb)
    bib.markers = author_year_markers(body)
    return bib


AY_BRACKET_GROUP = re.compile(r"\[([^\[\]]{4,400})\]")


def author_year_markers(body: str) -> list[dict]:
    """'Name et al. (2018)' / 'Name (2018)', '(Name et al., 2018; Other, 2019)' and '[Name and Other, 2006]'."""
    found = [(m.start(), m.end(), B._ay_key(m.group(1), m.group(2))) for m in B.AY_MARK.finditer(body)]
    for grp in (B.AY_PAREN_GROUP, AY_BRACKET_GROUP):
        for g in grp.finditer(body):
            for m in B.AY_PAREN.finditer(g.group(1)):
                found.append((g.start(1) + m.start(), g.start(1) + m.end(), B._ay_key(m.group(1), m.group(2))))
    out, seen = [], set()
    for s, e, k in sorted(found):
        if (s, k) not in seen:
            seen.add((s, k))
            out.append({"key": k, "start": s, "end": e})
    return out


def first_surname(raw: str) -> str:
    """First author's surname from an entry line. 'Surname, I.' / 'Surname, Given' -> Surname; otherwise the last
    capitalized word of the first author chunk ('N. Cesa-Bianchi and G. Lugosi.' -> Cesa-Bianchi;
    'Durmus Alp Emre Acar, Yue Zhao' -> Acar; 'J. de Vilmarest.' -> Vilmarest)."""
    s = re.sub(r"^\s*(?:\[\w+\]|\d{1,4}\.)\s*", "", raw)
    # the first author chunk ends at ', ', ';', ' and ', or a period that does not close an initial ("N. Cesa")
    first = re.split(r",\s|;\s|\sand\s|&|(?<![A-Z])\.\s", s, maxsplit=1)[0].strip()
    if re.fullmatch(r"[A-ZÀ-ɏ][A-Za-zÀ-ɏ'\-]+", first):
        return first
    words = [w.strip(".") for w in first.split()]
    words = [w for w in words if re.fullmatch(r"[A-ZÀ-ɏ][A-Za-zÀ-ɏ'\-]+", w) and not re.fullmatch(r"[A-Z]", w)]
    return words[-1] if words and words[-1].lower() not in ("url", "in", "the") else ""


def _author_year_entries(refs: str) -> dict | None:
    """Author-year lists rendered one entry per line (MinerU) — the case bib.AY_ENTRY does not cover. Key =
    (first-author surname lower, year+suffix), the same key bib._ay_key builds from in-text markers; the year is the
    last plausible year in the entry (not inside a URL, DOI or page range)."""
    entries, counts = {}, {}
    for l in _entry_lines(refs):
        ys = YEAR_CAND.findall(l)
        sur = first_surname(l)
        if not ys or not sur:
            continue
        y = ys[-1][0] + ys[-1][1]
        k = (sur.lower(), y)
        counts[k] = counts.get(k, 0) + 1
        raw = re.sub(r"\s+", " ", l).strip()
        ax, doi = B._ids(raw)
        entries.setdefault(k, {"raw": raw, "arxiv": ax, "doi": doi})
    if len(entries) < 3:
        return None
    return {"entries": entries, "ambiguous": {k for k, c in counts.items() if c > 1}}


def sentence_spans(body: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in re.finditer(r"[.?!](?=\s+[A-Z\[(])", body):
        if ABBREV.search(body[max(0, m.start() - 8):m.end()]):
            continue
        spans.append((start, m.end()))
        start = m.end()
    spans.append((start, len(body)))
    return [(s, e) for s, e in spans if body[s:e].strip()]


def citation_sentences(text: str, bib: B.Bibliography | None = None) -> tuple[B.Bibliography | None,
                                                                              list[CitationSentence]]:
    text = text.replace("\xa0", " ")
    bib = bib or parse(text)
    if bib is None:
        return None, []
    parts = find_refs(text)
    body = parts[0] if parts else text
    marks = sorted((m for m in bib.markers if m["key"] in bib.entries and m["key"] not in bib.ambiguous_keys),
                   key=lambda m: m["start"])
    out, j = [], 0
    for s, e in sentence_spans(body):
        keys = []
        while j < len(marks) and marks[j]["start"] < e:
            if marks[j]["start"] >= s:
                keys.append(marks[j]["key"])
            j += 1
        keys = list(dict.fromkeys(keys))
        if not keys:
            continue
        sent = re.sub(r"\s+", " ", body[s:e]).strip()
        if len(sent) < 25 or len(sent) > 1500:
            continue
        for k in keys:
            out.append(CitationSentence(sentence=sent, key=k, n_keys=len(keys), group=tuple(keys), offset=s))
    return bib, out
