# -*- coding: utf-8 -*-
"""The match view (INTEGRATED-SYSTEM-1005 §7.3, unified final check rule 1): one lossy projection of a source text
used ONLY to locate quotes — stored quotes stay verbatim. Steps, in the design's order: NFKC, HTML tags out,
citation markers out, LaTeX folded, `$ { } ^ _ ~ \\` out, Greek spelled out, the MinerU/LaTeX glyph map, casefold,
whitespace out. Punctuation that carries numeric meaning (the decimal point, %) is kept — dropping '.' would make
87.5 and 8.75 collide.

Every view character remembers its origin index, so locate() maps a found quote back to a span of the ORIGINAL
text (loc.char_start / char_end in schema v2)."""
from __future__ import annotations

import re
import unicodedata

from ..core.ids import _GREEK, _LATEX_CMD, _LATEX_FORMAT

_HTML = re.compile(r"<[^>]+>")
_CITE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*[a-z]?\s*\]")
# ids._GREEK is the registry's title_key map and must not drift (stored keys); the view spells out the full
# alphabet so a LaTeX fold ("\zeta" -> "zeta") and the Unicode glyph meet
_GREEK_FULL = {**_GREEK, "ζ": "zeta", "η": "eta", "θ": "theta", "ϑ": "theta", "ι": "iota", "ν": "nu",
               "ξ": "xi", "ρ": "rho", "φ": "phi", "ϕ": "phi", "χ": "chi", "ψ": "psi", "ς": "sigma"}
# glyphs where the Unicode form and the LaTeX-command fold must meet (the fold keeps command NAMES: \times -> "times")
_GLYPH = str.maketrans({
    "−": "-", "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "±": "pm", "∓": "mp", "×": "times", "·": "cdot", "≤": "leq", "≥": "geq", "≈": "approx", "≡": "equiv",
    "≠": "neq", "∞": "infty", "→": "to", "←": "from", "∈": "in", "⊂": "subset", "⊆": "subseteq",
    "′": "'", "″": '"', "⁄": "/",
})


def _is_pua(c: str) -> bool:
    return 0xE000 <= ord(c) <= 0xF8FF      # private-use glyphs from PDF symbol fonts: unmatchable noise


def view_map(text: str) -> tuple[str, list[int]]:
    """(view, origins) — origins[i] is the index in `text` of view character i."""
    chars: list[str] = []
    orig: list[int] = []
    for i, ch in enumerate(text or ""):
        for c2 in unicodedata.normalize("NFKC", ch):
            chars.append(c2)
            orig.append(i)

    def rewrite(pattern, repl):
        """Drop/replace every regex span, keeping the char->origin arrays aligned."""
        nonlocal chars, orig
        s = "".join(chars)
        out_c: list[str] = []
        out_i: list[int] = []
        pos = 0
        for m in pattern.finditer(s):
            for j in range(pos, m.start()):
                out_c.append(chars[j])
                out_i.append(orig[j])
            for c2 in repl(m):
                out_c.append(c2)
                out_i.append(orig[m.start()])
            pos = m.end()
        for j in range(pos, len(chars)):
            out_c.append(chars[j])
            out_i.append(orig[j])
        chars, orig = out_c, out_i

    rewrite(_HTML, lambda m: "")                                   # HTML tags out
    rewrite(_CITE, lambda m: "")                                   # citation markers out
    rewrite(_LATEX_CMD, lambda m: "" if m.group(1) in _LATEX_FORMAT else m.group(1))   # LaTeX folded

    keep_c, keep_i = [], []
    for c, i in zip(chars, orig):
        if c in "${}^_~\\" or _is_pua(c) or c.isspace():
            continue
        g = _GREEK_FULL.get(c.casefold())
        if g is not None and not c.isascii():
            for c2 in g:
                keep_c.append(c2)
                keep_i.append(i)
            continue
        for c2 in c.translate(_GLYPH).casefold():     # per-char casefold: ß/İ-class expansions keep the map aligned
            keep_c.append(c2)
            keep_i.append(i)
    return "".join(keep_c), keep_i


def view(text: str) -> str:
    return view_map(text)[0]


def locate(quote: str, sent: str) -> tuple[int, int] | None:
    """The span of `quote` inside the ORIGINAL `sent` (start inclusive, end exclusive), matched through the view;
    None when the quote is not in the sentence. The span may cover view-dropped noise (tags, spaces) between the
    matched characters — it is a pointer for tools, not a re-extraction."""
    vs, orig = view_map(sent)
    vq = view(quote)
    if not vq:
        return None
    p = vs.find(vq)
    if p < 0:
        return None
    return orig[p], orig[p + len(vq) - 1] + 1
