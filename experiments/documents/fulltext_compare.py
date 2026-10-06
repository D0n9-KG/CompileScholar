# -*- coding: utf-8 -*-
r"""Full-text quality for deep extraction: GROBID TEI vs MinerU vs the arXiv v1 LaTeXML HTML (rendered from the authors'
own LaTeX: the reference for body text, headings, tables and formulas).

Per paper (the fast-tier gold sample, cs v1, PDFs with an HTML rendering):
  body text   recall = gold body-paragraph word bigrams found in the system's body text; noise = system bigrams not in
              the gold page at all (page furniture, OCR garbage, text of figures). Bibliography excluded on all sides.
  sentences   share of gold body sentences found (>= 0.8 bigram coverage) in one system paragraph
  headings    share of gold section headings found among the system's headings
  tables      gold tables (ltx_tabular) -> cell texts; a gold table is recovered when one system table holds >= 70 % of
              its non-empty cells (structure, not only text: GROBID's table text inside a paragraph does not count)
  formulas    gold display equations (ltx_equation) -> system formula units present (count recall); and for MinerU
              the LaTeX: share of gold symbols (\alpha, \sum, digits, letters) present
  readability share of system paragraphs that are prose (>= 70 % alphabetic tokens) vs garbage
Usage: python fulltext_compare.py"""
from __future__ import annotations

import html as _html
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lxml import etree  # noqa: E402

from compilescholar.core import paths  # noqa: E402
from compilescholar.documents import mineru as MI  # noqa: E402
from compilescholar.documents import tei as TE  # noqa: E402

OUT = paths.runs() / "phaseB_fasttier"
TAG = re.compile(r"<[^>]+>")


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", re.sub(r"-\s+", "", (s or "").lower()))


def bigrams(s: str) -> set:
    w = words(s)
    return set(zip(w, w[1:]))


def _txt(fr: str) -> str:
    fr = re.sub(r"<math\b.*?</math>", " ", fr, flags=re.S)
    fr = re.sub(r'<cite class="ltx_cite[^"]*">.*?</cite>', " ", fr, flags=re.S)
    return re.sub(r"\s+", " ", _html.unescape(TAG.sub(" ", fr))).strip()


def gold(page: str) -> dict:
    cut = page.find('class="ltx_bibliography')
    if cut > 0:
        s, e = page.rfind("<", 0, cut), page.find("</section>", cut)
        page = page[:s] + (page[e + 10:] if e > 0 else "")
    paras = [_txt(m.group(1)) for m in re.finditer(r'<p\b[^>]*class="ltx_p"[^>]*>(.*?)</p>', page, re.S)]
    paras = [p for p in paras if len(p) > 40]
    heads = [_txt(m.group(2)) for m in re.finditer(r"<h([2-5])[^>]*class=\"ltx_title[^\"]*\"[^>]*>(.*?)</h\1>", page,
                                                   re.S)]
    heads = [re.sub(r"^\s*(?:[\dA-Z]+(?:\.\d+)*\.?\s+)", "", h) for h in heads if h and "Reference" not in h]
    tables = []
    for m in re.finditer(r'<table\b[^>]*class="[^"]*ltx_tabular[^"]*"[^>]*>(.*?)</table>', page, re.S):
        cells = [_txt(c) for c in re.findall(r"<td\b[^>]*>(.*?)</td>", m.group(1), re.S)]
        cells = [c for c in cells if c]
        if len(cells) >= 4:
            tables.append(cells)
    eqs = [m.group(1) for m in re.finditer(r'<table[^>]*class="[^"]*ltx_equation[^"]*"[^>]*>.*?<math[^>]*alttext="([^"]*)"',
                                           page, re.S)]
    sents = [s for p in paras for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", p) if len(words(s)) >= 6]
    return {"paras": paras, "heads": heads, "tables": tables, "eqs": [_html.unescape(e) for e in eqs], "sents": sents,
            "all": bigrams(_txt(page))}


def grobid_doc(stem: str) -> dict:
    tei = (OUT / "grobid_full_coords" / f"{stem}.tei.xml").read_bytes()
    d = TE.parse(tei, None, stem)
    root = etree.fromstring(tei)
    T = TE.T
    tables = []
    for fig in root.iter(f"{T}figure"):
        if fig.get("type") == "table":
            t = fig.find(f"{T}table")
            if t is not None:
                tables.append([TE._text(c) for c in t.iter(f"{T}cell") if TE._text(c)])
    # the abstract lives in the TEI header (profileDesc/abstract), not in <body>: count it as body text as the gold does
    absp = [TE._text(x) for x in root.iter(f"{T}abstract") for x in x.iter(f"{T}p")]
    return {"paras": absp + [u.text for u in d.units if u.kind == "para"], "heads": [u.text for u in d.units
                                                                           if u.kind == "section"],
            "tables": tables, "formulas": [u.text for u in d.units if u.kind == "formula"]}


def mineru_doc(aid: str) -> dict | None:
    p = OUT / "mineru" / f"{aid}.zip"
    if not p.exists():
        return None
    us, _ = MI.units(MI.load(p), aid)
    tables = []
    for u in us:
        if u.kind == "table" and u.html:
            tables.append([_txt(c) for c in re.findall(r"<td\b[^>]*>(.*?)</td>", u.html, re.S) if _txt(c)])
    return {"paras": [u.text for u in us if u.kind == "para"], "heads": [u.text for u in us if u.kind == "section"],
            "tables": tables, "formulas": [u.text for u in us if u.kind == "formula"]}


def _prose(p: str) -> bool:
    toks = p.split()
    return len(toks) >= 5 and sum(bool(re.fullmatch(r"[A-Za-z][A-Za-z'\-]*[.,;:]?", t)) for t in toks) / len(toks) >= 0.7


def _cell_norm(c: str) -> str:
    return " ".join(words(c))


def score(g: dict, s: dict) -> dict:
    gb = set().union(*(bigrams(p) for p in g["paras"])) if g["paras"] else set()
    sb = set().union(*(bigrams(p) for p in s["paras"])) if s["paras"] else set()
    recall = len(gb & sb) / max(1, len(gb))
    noise = len(sb - g["all"]) / max(1, len(sb))
    sys_par = [bigrams(p) for p in s["paras"]]
    found = 0
    for sent in g["sents"]:
        b = bigrams(sent)
        if b and any(len(b & p) / len(b) >= 0.8 for p in sys_par):
            found += 1
    hn = lambda h: " ".join(words(re.sub(r"^\s*(?:[\dA-Z]{1,3}(?:\.\d+)*\.?)\s+(?=\w)", "", h)))   # drop "4.1"
    hk = {hn(h) for h in s["heads"]}
    heads = sum(1 for h in g["heads"] if hn(h) in hk) / max(1, len(g["heads"]))
    tab_ok = 0
    for gt in g["tables"]:
        gc = {_cell_norm(c) for c in gt if _cell_norm(c)}
        if any(len(gc & {_cell_norm(c) for c in st}) / max(1, len(gc)) >= 0.7 for st in s["tables"]):
            tab_ok += 1
    sym = None
    if g["eqs"] and s["formulas"]:
        allf = " ".join(s["formulas"])
        toks = [t for e in g["eqs"] for t in re.findall(r"\\[A-Za-z]+|[A-Za-z]|\d", e)]
        sym = sum(t in allf for t in toks) / max(1, len(toks))
    return {"body_recall": recall, "noise": noise, "sent_recall": found / max(1, len(g["sents"])), "headings": heads,
            "tables_gold": len(g["tables"]), "tables_ok": tab_ok, "eqs_gold": len(g["eqs"]),
            "formulas": len(s["formulas"]), "formula_symbols": sym,
            "prose_share": sum(map(_prose, s["paras"])) / max(1, len(s["paras"]))}


def main() -> None:
    rows = [json.loads(l) for l in open(OUT / "sample.jsonl", encoding="utf-8")]
    gold_rows = [r for r in rows if r["pdf"] and r["html"]]
    res = {"grobid": [], "mineru": []}
    for r in gold_rows:
        aid, stem = r["arxiv_id"], f"{r['arxiv_id']}v1"
        m = mineru_doc(aid)
        if m is None:
            continue
        g = gold((OUT / r["html"]).read_text(encoding="utf-8", errors="replace"))
        res["grobid"].append({"id": aid, **score(g, grobid_doc(stem))})
        res["mineru"].append({"id": aid, **score(g, m)})
    (OUT / "fulltext_score.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    n = len(res["grobid"])
    print(f"{n} papers with both parses")
    for k in ("grobid", "mineru"):
        rs = res[k]
        med = lambda f: statistics.median(x[f] for x in rs)
        tg, to = sum(x["tables_gold"] for x in rs), sum(x["tables_ok"] for x in rs)
        eg, ef = sum(x["eqs_gold"] for x in rs), sum(min(x["formulas"], x["eqs_gold"]) for x in rs)
        fs = [x["formula_symbols"] for x in rs if x["formula_symbols"] is not None]
        print(f"{k:7s} body recall {med('body_recall'):.3f}  sentences {med('sent_recall'):.3f}  noise "
              f"{med('noise'):.3f}  headings {med('headings'):.3f}  prose share {med('prose_share'):.3f}  "
              f"tables {to}/{tg}  display formulas {ef}/{eg}  formula symbols "
              f"{statistics.median(fs) if fs else float('nan'):.3f}  "
              f"papers body<0.8 {sum(x['body_recall'] < 0.8 for x in rs)}")


if __name__ == "__main__":
    main()
