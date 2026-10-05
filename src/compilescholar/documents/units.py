# -*- coding: utf-8 -*-
"""L1 document structure: one paper's full text -> addressable units (DESIGN-LITERATURE-LAYER N1).

Unit kinds: section (heading), para (paragraph text), table (raw table block), caption (figure/table caption).
Every unit has a stable id "<arxiv_id>#<kind><n>", its section path ("3 Method > 3.2 Encoder"), and its text verbatim
from the source, so every downstream quote can point at a unit. Two sources are normalised to the same units:
  Markdown (MinerU, IdeaForecastBench): '#' headings, blank-line paragraphs, <table> blocks / pipe tables,
                                        'Table N:' / 'Figure N:' caption lines
  arXiv HTML (LaTeXML):                 <section>/<h2-h6> headings, <p class="ltx_p">, <table class="ltx_tabular">,
                                        <figcaption>
The reference block is not split into units (citations.* owns it); the body ends where the reference block starts.

Also: `section_of(units, offset)` and the paper-level `abstract_and_intro` used by the self pass."""
from __future__ import annotations

import html as _html
import re
from dataclasses import dataclass

from ..citations import markdown as CM

HEAD = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
TABLE_BLOCK = re.compile(r"<table>.*?</table>", re.S)
CAPTION = re.compile(r"(?mi)^\s*((?:table|figure|fig\.)\s*\d+[.:]\s*.+)$")
INTRO = re.compile(r"(?i)\bintroduction\b")


@dataclass
class Unit:
    uid: str
    kind: str          # section | para | table | caption
    section: str       # heading path at this unit
    text: str
    start: int         # char offset in the body string the units were cut from


def _clean(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()


def from_markdown(arxiv_id: str, text: str) -> list[Unit]:
    parts = CM.find_refs(text)
    body = parts[0] if parts else text
    units, path, n = [], [], {"section": 0, "para": 0, "table": 0, "caption": 0}

    def add(kind, txt, start):
        n[kind] += 1
        units.append(Unit(f"{arxiv_id}#{kind}{n[kind]}", kind, " > ".join(path), txt, start))

    # tables first, so their text is not re-split into paragraphs
    spans = [(m.start(), m.end()) for m in TABLE_BLOCK.finditer(body)]
    for s, e in spans:
        pass
    pos = 0
    chunks = []  # (start, end, is_table)
    for s, e in spans:
        if s > pos:
            chunks.append((pos, s, False))
        chunks.append((s, e, True))
        pos = e
    chunks.append((pos, len(body), False))
    for s, e, is_table in chunks:
        seg = body[s:e]
        if is_table:
            add("table", seg, s)
            continue
        cursor = 0
        for m in HEAD.finditer(seg):
            _paras(seg[cursor:m.start()], s + cursor, add)
            level, title = len(m.group(1)), _clean(m.group(2))
            # MinerU often renders every heading as '#'; a numbered title ("3.2 Encoder", "III.", "A.") carries
            # the real depth, so use it when present
            num = re.match(r"^(\d+(?:\.\d+)*)\.?\s", title)
            if num:
                level = num.group(1).count(".") + 1
            path[:] = path[:max(0, level - 1)] + [title]
            add("section", title, s + m.start())
            cursor = m.end()
        _paras(seg[cursor:], s + cursor, add)
    return units


def _paras(seg: str, base: int, add) -> None:
    for m in re.finditer(r"(?:[^\n]+\n?)+", seg):
        t = _clean(m.group(0).replace("\n", " "))
        if len(t) < 20:
            continue
        if CAPTION.match(t):
            add("caption", t, base + m.start())
        else:
            add("para", t, base + m.start())


H_HEAD = re.compile(r"<h([2-6])[^>]*>(.*?)</h\1>", re.S)
H_PARA = re.compile(r'<p\b[^>]*class="ltx_p"[^>]*>(.*?)</p>', re.S)
H_TABLE = re.compile(r'<table\b[^>]*class="[^"]*ltx_tabular[^"]*"[^>]*>.*?</table>', re.S)
H_CAP = re.compile(r"<figcaption[^>]*>(.*?)</figcaption>", re.S)
TAG = re.compile(r"<[^>]+>")
MATH = re.compile(r"<math\b.*?</math>", re.S)


def _htext(fr: str) -> str:
    return re.sub(r"\s+", " ", _html.unescape(TAG.sub(" ", MATH.sub(" [math] ", fr)))).strip()


def from_html(arxiv_id: str, page: str) -> list[Unit]:
    cut = page.find('class="ltx_bibliography')
    body = page[:cut] if cut > 0 else page
    events = []
    for m in H_HEAD.finditer(body):
        events.append((m.start(), "section", int(m.group(1)), _htext(m.group(2))))
    for m in H_PARA.finditer(body):
        events.append((m.start(), "para", 0, _htext(m.group(1))))
    for m in H_TABLE.finditer(body):
        events.append((m.start(), "table", 0, m.group(0)))
    for m in H_CAP.finditer(body):
        events.append((m.start(), "caption", 0, _htext(m.group(1))))
    units, path, n = [], [], {"section": 0, "para": 0, "table": 0, "caption": 0}
    for start, kind, level, txt in sorted(events):
        if kind == "section":
            path[:] = path[:max(0, level - 2)] + [txt]
        if kind != "table" and len(txt) < (1 if kind == "section" else 20):
            continue
        n[kind] += 1
        units.append(Unit(f"{arxiv_id}#{kind}{n[kind]}", kind, " > ".join(path), txt, start))
    return units


def abstract_and_intro(units: list[Unit], max_chars: int = 7000) -> str:
    """Front matter (paragraphs before the introduction heading, i.e. the abstract) + the introduction section, up to
    the next section heading; capped. The self pass's full-text input."""
    out, phase = [], "front"          # front -> intro -> done
    for u in units:
        if u.kind == "section":
            if INTRO.search(u.text) and phase == "front":
                phase = "intro"
            elif phase == "intro":
                break
            continue
        if u.kind == "para" and phase in ("front", "intro"):
            out.append(u.text)
        if sum(len(x) for x in out) >= max_chars:
            break
    return "\n\n".join(out)[:max_chars]


SKIP_SEC = re.compile(r"(?i)\b(abstract|introduction|related works?|prior works?|background|preliminar\w*|"
                      r"conclusions?|discussion|limitations?|acknowledg\w*|appendix|ethics|broader impacts?|"
                      r"future works?|references)\b")


def method_and_experiments(units: list[Unit], max_chars: int = 9000) -> str:
    """Body paragraphs after the introduction, excluding related work / background / conclusion / appendix sections
    (heading-keyword exclusion is more robust than keyword inclusion: method sections are often named after the
    method), capped — deep pass input."""
    out, after_intro = [], False
    for u in units:
        if u.kind == "section" and INTRO.search(u.text):
            after_intro = True
        if u.kind == "para" and after_intro and not SKIP_SEC.search(u.section or "") \
                and not INTRO.search(u.section or ""):
            out.append(u.text)
    return "\n\n".join(out)[:max_chars]
