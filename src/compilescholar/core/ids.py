# -*- coding: utf-8 -*-
"""Identifier normalisation: the one implementation for DOIs, arXiv ids and titles (INTEGRATED-SYSTEM-1005 v2 §3).
Replaces the three DOI rules (registry, skeleton bib, refgraph) and the four arXiv regexes that disagreed.

  normalize_doi("https://dx.doi.org/10.1000/ABC.")  -> "10.1000/abc"
  normalize_doi("10.48550/arXiv.2101.00001")       -> None   (an arXiv DOI is an arXiv id, see arxiv_from_doi)
  normalize_arxiv("arXiv:2101.00001v2")            -> "2101.00001"     (version stripped; arxiv_version() keeps it)
  normalize_arxiv("math.GT/0309136")               -> "math.GT/0309136" (old style keeps the subject-class case)
  norm_title("α-Synuclein: Étude")                 -> "α synuclein étude"  (NFKC + casefold, Unicode letters kept)

find_* return every identifier found in free text (a bibliography entry, a PDF first page)."""
from __future__ import annotations

import re
import unicodedata
from urllib.parse import unquote

_DOI_PREFIX = re.compile(r"^(?:doi\s*:\s*|https?://(?:dx\.|www\.)?doi\.org/|(?:dx\.|www\.)?doi\.org/)", re.I)
_DOI_CORE = re.compile(r"^10\.\d{4,9}/\S+$")
_DOI_IN_TEXT = re.compile(r"(?:doi\s*:\s*|doi\.org/)?(10\.\d{4,9}/[^\s\"'<>]+)", re.I)
_TRAIL = ".,;:)]}>'\""
_PAIRS = {")": "(", "]": "[", "}": "{", ">": "<"}
ARXIV_DOI_PREFIX = "10.48550/arxiv."

# new style: YYMM.NNNN (2007-04..2014-12) or YYMM.NNNNN (2015-01..); old style: archive[.SUBJ]/YYMMNNN
_ARXIV_NEW = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
_ARXIV_OLD = re.compile(r"^([a-z\-]+(?:\.[A-Za-z\-]{2})?/\d{7})(v\d+)?$", re.I)
_ARXIV_IN_TEXT = re.compile(
    r"(?:arxiv\s*[:.]?\s*|arxiv\.org/(?:abs|pdf|html)/|10\.48550/arxiv\.)"
    r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Za-z\-]{2})?/\d{7}(?:v\d+)?)", re.I)
_OLD_ARCHIVES = {"acc-phys", "adap-org", "alg-geom", "ao-sci", "astro-ph", "atom-ph", "bayes-an", "chao-dyn",
                 "chem-ph", "cmp-lg", "comp-gas", "cond-mat", "cs", "dg-ga", "funct-an", "gr-qc", "hep-ex",
                 "hep-lat", "hep-ph", "hep-th", "math", "math-ph", "mtrl-th", "nlin", "nucl-ex", "nucl-th",
                 "patt-sol", "physics", "plasm-ph", "q-alg", "q-bio", "q-fin", "quant-ph", "solv-int", "stat",
                 "supr-con"}


def _strip_trailing(s: str) -> str:
    """Trailing punctuation, but keep a closing bracket whose opener is inside the DOI (SICI DOIs, 10.1002/(SICI)...)."""
    while s and s[-1] in _TRAIL:
        c = s[-1]
        if c in _PAIRS and s.count(_PAIRS[c]) >= s.count(c):
            break
        s = s[:-1]
    return s


def _doi_core(raw: str | None) -> str | None:
    if not raw:
        return None
    s = unquote(str(raw)).strip()
    s = s.strip("()[]<>{} \t\r\n") if s[:1] in "([{<" else s.strip()
    s = _DOI_PREFIX.sub("", s).strip()
    s = _strip_trailing(s)
    s = s.lower()
    return s if _DOI_CORE.match(s) else None


def normalize_doi(raw: str | None) -> str | None:
    """Canonical DOI (lowercase, no prefix, no trailing punctuation), or None — also None for arXiv DOIs, which name
    an arXiv record, not a published version (use arxiv_from_doi)."""
    s = _doi_core(raw)
    if not s or s.startswith(ARXIV_DOI_PREFIX):
        return None
    return s


def arxiv_from_doi(raw: str | None) -> str | None:
    s = _doi_core(raw)
    if s and s.startswith(ARXIV_DOI_PREFIX):
        return normalize_arxiv(s[len(ARXIV_DOI_PREFIX):])
    return None


def _arxiv_parts(raw: str | None) -> tuple[str, str] | None:
    if not raw:
        return None
    s = unquote(str(raw)).strip()
    s = re.sub(r"^(?:https?://)?(?:www\.)?(?:export\.)?arxiv\.org/(?:abs|pdf|html)/", "", s, flags=re.I)
    s = re.sub(r"^arxiv\s*[:.]?\s*", "", s, flags=re.I)
    s = re.sub(r"\.pdf$", "", s, flags=re.I).strip().rstrip(".,;")
    if s.lower().startswith(ARXIV_DOI_PREFIX):
        s = s[len(ARXIV_DOI_PREFIX):]
    m = _ARXIV_NEW.match(s)
    if m:
        return m.group(1), m.group(2) or ""
    m = _ARXIV_OLD.match(s)
    if m:
        ident = m.group(1)
        arch, _, rest = ident.partition("/")
        a, _, subj = arch.partition(".")
        if a.lower() not in _OLD_ARCHIVES:
            return None
        return (f"{a.lower()}.{subj.upper()}/{rest}" if subj else f"{a.lower()}/{rest}"), m.group(2) or ""
    return None


def normalize_arxiv(raw: str | None) -> str | None:
    """Canonical arXiv id without version, or None."""
    p = _arxiv_parts(raw)
    return p[0] if p else None


def arxiv_version(raw: str | None) -> int | None:
    p = _arxiv_parts(raw)
    return int(p[1][1:]) if p and p[1] else None


def find_dois(text: str) -> list[str]:
    out = []
    for m in _DOI_IN_TEXT.finditer(text or ""):
        d = normalize_doi(m.group(1))
        if d and d not in out:
            out.append(d)
    return out


def find_arxiv(text: str) -> list[str]:
    out = []
    for m in _ARXIV_IN_TEXT.finditer(text or ""):
        a = normalize_arxiv(m.group(1))
        if a and a not in out:
            out.append(a)
    return out


_GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "κ": "kappa", "λ": "lambda",
          "μ": "mu", "π": "pi", "σ": "sigma", "τ": "tau", "ω": "omega"}


def norm_title(s: str | None, greek: bool = False) -> str:
    """NFKC + casefold; every run of non-letter/digit characters becomes one space (Unicode letters and digits kept:
    CJK, accents, Greek). greek=True spells Greek letters out (α-synuclein -> alpha synuclein) for matching against
    sources that transliterate. Empty input -> ""."""
    t = unicodedata.normalize("NFKC", s or "").casefold()
    if greek:
        t = "".join(_GREEK.get(c, c) for c in t)
    t = re.sub(r"[^\w]+|_", " ", t)
    return re.sub(r"\s+", " ", t).strip()


GENERIC_TITLES = {"editorial", "reply", "erratum", "corrigendum", "correction", "comment", "introduction",
                  "preface", "foreword", "index", "contents", "obituary", "book review", "letter to the editor",
                  "response", "retraction", "addendum", "news", "in this issue", "front matter", "back matter"}


def title_is_generic(s: str | None) -> bool:
    """Titles that may never be the only evidence for merging two papers (Editorial, Reply, ... or < 3 words)."""
    t = norm_title(s)
    return not t or t in GENERIC_TITLES or len(t.split()) < 3
