# -*- coding: utf-8 -*-
"""Citation sentences from arXiv HTML (LaTeXML rendering of arxiv.org/html/<id>), deterministic, no LLM.

LaTeXML wraps every in-text citation in <cite class="ltx_cite ..."> whose <a href="#bib.bibN"> links point at
<li id="bib.bibN" class="ltx_bibitem">, so marker -> entry is an anchor lookup (no style detection). Measured on
2404.01039 (10-05): 238 cites, 177 bibitems.

Output uses the same CitationSentence record as citations.markdown (key = "bibN")."""
from __future__ import annotations

import html as _html
import re

from ..compile.skeleton import bib as B
from .markdown import CitationSentence, sentence_spans

BIBITEM = re.compile(r'<li[^>]*\bid="bib\.(bib\d+)"[^>]*class="[^"]*ltx_bibitem[^"]*"[^>]*>(.*?)</li>', re.S)
CITE = re.compile(r'<cite class="ltx_cite[^"]*">(.*?)</cite>', re.S)
BIBREF = re.compile(r'href="#bib\.(bib\d+)"')
PARA = re.compile(r'<p\b[^>]*class="ltx_p"[^>]*>(.*?)</p>', re.S)
TAG = re.compile(r"<[^>]+>")
MATH = re.compile(r"<math\b.*?</math>", re.S)
# placeholder for a <cite>: private-use characters cannot occur in the extracted text
TOK = re.compile("(\\d+)")


def _text(fragment: str) -> str:
    fragment = MATH.sub(" [math] ", fragment)
    return re.sub(r"\s+", " ", _html.unescape(TAG.sub(" ", fragment))).strip()


def parse(page: str) -> tuple[B.Bibliography, list[CitationSentence]] | None:
    """Bibliography (entries keyed by 'bibN') and citation sentences from body paragraphs (<p class="ltx_p">
    before the bibliography)."""
    entries = {}
    for m in BIBITEM.finditer(page):
        # the label span ("Benko et al<span>.</span> (2024)") nests spans; the entry text starts at the first bibblock
        item = m.group(2)
        j = item.find('class="ltx_bibblock')
        raw = _text(item[item.rfind("<", 0, j):] if j > 0 else item)
        ax, doi = B._ids(raw)
        entries[m.group(1)] = {"raw": raw, "arxiv": ax, "doi": doi}
    if not entries:
        return None
    bib = B.Bibliography(style="latexml", entries=entries)
    cut = page.find('class="ltx_bibliography')
    body = page[:cut] if cut > 0 else page
    out: list[CitationSentence] = []
    offset = 0
    for p in PARA.finditer(body):
        keys_by_tok: dict[str, list[str]] = {}

        def _sub(m, _k=keys_by_tok):
            i = str(len(_k))
            _k[i] = BIBREF.findall(m.group(1))
            return "" + i + "" + _text(m.group(1))
        text = _text(CITE.sub(_sub, p.group(1)))
        for s, e in sentence_spans(text):
            sent = text[s:e]
            keys = list(dict.fromkeys(k for i in TOK.findall(sent) for k in keys_by_tok.get(i, []) if k in entries))
            clean = re.sub(r"\s+", " ", TOK.sub("", sent)).strip()
            if not keys or len(clean) < 25 or len(clean) > 1500:
                continue
            for k in keys:
                out.append(CitationSentence(sentence=clean, key=k, n_keys=len(keys), group=tuple(keys),
                                            offset=offset + s))
        offset += len(text) + 1
        bib.markers += [{"key": k, "start": 0, "end": 0} for ks in keys_by_tok.values() for k in ks]
    return bib, out
