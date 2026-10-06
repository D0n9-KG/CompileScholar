# -*- coding: utf-8 -*-
"""Phase B fast-tier comparison, step 2: citation-sentence recall / precision of candidate full-text parsers against
the arXiv v1 LaTeXML gold (fasttier_sample.py collects the sample).

Gold (per paper, from html/<id>v1.html): bibliography entries (bibitem text) and, for every in-text <cite>, the cited
entry keys with the sentence that carries it. Prose citations (inside <p class="ltx_p">) are the recall target; every
other citation (tables, captions, footnotes) also counts as correct for precision, via its surrounding text.

Systems (each yields entries {key: raw text} and pairs [(key, sentence)]):
  pymupdf_asis   PyMuPDF page text joined, then citations.markdown as it is today
  grobid_crf     GROBID 0.9.1 (CRF models) processFulltextDocument, sentence segmentation on, no consolidation
  ...            further systems are added by name in SYSTEMS

Scoring per paper:
  entry alignment  system entry <-> gold entry, one-to-one greedy on word-set Jaccard >= 0.5
  pair match       same aligned entry and sentence word-bigram overlap: |G&S|/|G| >= 0.6 and |G&S|/|S| >= 0.4
  recall           matched gold prose pairs / gold prose pairs
  precision        system pairs matching any gold citation of that entry (prose sentence or other context) / pairs
Usage: python fasttier_compare.py run <system> [workers]   |   python fasttier_compare.py score
"""
from __future__ import annotations

import html as _html
import json
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from compilescholar.core import paths

OUT = paths.runs() / "phaseB_fasttier"

# ---------------------------------------------------------------- gold

BIBITEM = re.compile(r'<li[^>]*\bid="bib\.(bibx?\d+)"[^>]*class="[^"]*ltx_bibitem[^"]*"[^>]*>(.*?)</li>', re.S)
CITE = re.compile(r'<cite class="ltx_cite[^"]*">(.*?)</cite>', re.S)
BIBREF = re.compile(r'href="#bib\.(bibx?\d+)"')
PARA = re.compile(r'<p\b[^>]*class="ltx_p"[^>]*>(.*?)</p>', re.S)
TAG = re.compile(r"<[^>]+>")
MATH = re.compile(r"<math\b.*?</math>", re.S)
PH = re.compile("\x01(\\d+)\x02")          # placeholder for a <cite>: control chars never occur in text
CITE_SPAN = re.compile("\x01\\d+\x02.*?\x03")
# in-text markers on the system side: [1,2], [ 13 ], (Name et al., 2021; ...), [Name and Other, 2006]
SYS_MARK = re.compile(r"\[\s*\d+(?:\s*[,–\-]\s*\d+)*\s*\]|[(\[][^()\[\]]{0,300}?(?:19|20)\d\d[a-z]?[^()\[\]]{0,300}?[)\]]")


def _text(fragment: str) -> str:
    fragment = MATH.sub(" [math] ", fragment)
    return re.sub(r"\s+", " ", _html.unescape(TAG.sub(" ", fragment))).strip()


def gold(page: str) -> dict:
    from compilescholar.citations.markdown import sentence_spans
    entries = {}
    for m in BIBITEM.finditer(page):
        item = m.group(2)
        j = item.find('class="ltx_bibblock')
        entries[m.group(1)] = _text(item[item.rfind("<", 0, j):] if j > 0 else item)
    # the body is everything but the bibliography section itself: appendices often follow it
    cut = page.find('class="ltx_bibliography')
    if cut > 0:
        start = page.rfind("<", 0, cut)
        end = page.find("</section>", cut)
        body = page[:start] + (page[end + len("</section>"):] if end > 0 else "")
    else:
        body = page
    prose, other = [], {}
    in_para = [(p.start(1), p.end(1)) for p in PARA.finditer(body)]
    for p in PARA.finditer(body):
        keys_by: dict[str, list] = {}

        def _sub(m, _k=keys_by):
            i = str(len(_k))
            _k[i] = BIBREF.findall(m.group(1))
            return "\x01" + i + "\x02" + _text(m.group(1)) + "\x03"
        text = _text(CITE.sub(_sub, p.group(1)))
        for s, e in sentence_spans(text):
            sent = text[s:e]
            keys = list(dict.fromkeys(k for i in PH.findall(sent) for k in keys_by.get(i, []) if k in entries))
            clean = re.sub(r"\s+", " ", CITE_SPAN.sub(" ", sent)).strip()   # marker text removed
            for k in keys:
                prose.append((k, clean))
    for m in CITE.finditer(body):
        if any(s <= m.start() < e for s, e in in_para):
            continue
        ctx = _text(body[max(0, m.start() - 1500):m.end() + 1500])
        for k in BIBREF.findall(m.group(1)):
            if k in entries:
                other.setdefault(k, []).append(ctx)
    return {"entries": entries, "prose": prose, "other": other}


# ---------------------------------------------------------------- systems

def sys_pymupdf_asis(pdf: Path) -> dict:
    import pymupdf
    from compilescholar.citations import markdown as M
    with pymupdf.open(pdf) as d:
        txt = "\n".join(p.get_text() for p in d)
    bib, cs = M.citation_sentences(txt)
    if bib is None:
        return {"entries": {}, "pairs": []}
    return {"entries": {str(k): v["raw"] for k, v in bib.entries.items()},
            "pairs": [(str(c.key), c.sentence) for c in cs]}


GROBID = "http://127.0.0.1:8070/api/processFulltextDocument"


def _grobid_tei(pdf: Path, dest: Path, url: str = GROBID) -> Path:
    import httpx
    if dest.exists():
        return dest
    for attempt in range(8):
        with open(pdf, "rb") as f:
            r = httpx.post(url, files={"input": (pdf.name, f, "application/pdf")},
                           data={"segmentSentences": "1", "includeRawCitations": "1", "consolidateHeader": "0",
                                 "consolidateCitations": "0", "consolidateFunders": "0"}, timeout=600,
                           trust_env=False)       # a system proxy would intercept localhost
        if r.status_code == 503:          # server at its concurrency limit
            time.sleep(1 + attempt)
            continue
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(r.content)
        tmp.replace(dest)
        return dest
    raise RuntimeError("grobid busy")


TEI = "{http://www.tei-c.org/ns/1.0}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


def parse_tei(tei: bytes) -> dict:
    from lxml import etree
    root = etree.fromstring(tei)
    entries = {}
    for b in root.iter(f"{TEI}biblStruct"):
        bid = b.get(XML_ID)
        if not bid:
            continue
        raw = b.find(f"{TEI}note[@type='raw_reference']")
        entries[bid] = re.sub(r"\s+", " ", "".join(raw.itertext()) if raw is not None else
                              " ".join(b.itertext())).strip()
    pairs = []
    for s in root.iter(f"{TEI}s"):
        keys = [r.get("target", "").lstrip("#") for r in s.iter(f"{TEI}ref") if r.get("type") == "bibr"]
        keys = list(dict.fromkeys(k for k in keys if k in entries))
        sent = re.sub(r"\s+", " ", "".join(s.itertext())).strip()
        pairs += [(k, sent) for k in keys]
    return {"entries": entries, "pairs": pairs}


def sys_grobid_crf(pdf: Path) -> dict:
    tei = _grobid_tei(pdf, OUT / "grobid_crf" / (pdf.stem + ".tei.xml"))
    return parse_tei(tei.read_bytes())


def sys_grobid_full(pdf: Path) -> dict:
    """GROBID 0.9.1-full: DeLFT models for header, reference segmentation and citation parsing (body stays CRF)."""
    tei = _grobid_tei(pdf, OUT / "grobid_full" / (pdf.stem + ".tei.xml"), GROBID.replace(":8070", ":8071"))
    return parse_tei(tei.read_bytes())


def _boxes(coords: str | None) -> list[tuple]:
    """GROBID coords "page,x,y,w,h;..." (1-based page, top-left origin, PDF points) -> [(page0, x0, y0, x1, y1)]."""
    out = []
    for part in (coords or "").split(";"):
        f = part.split(",")
        if len(f) == 5:
            p, x, y, w, h = int(f[0]), *map(float, f[1:])
            out.append((p - 1, x, y, x + w, y + h))
    return out


def pdf_cite_links(pdf: Path) -> list[tuple]:
    """hyperref citation links: [(page0, rect, dest_page0, dest_x, dest_y_top)] for named destinations cite.*"""
    import pymupdf
    out = []
    with pymupdf.open(pdf) as d:
        names = d.resolve_names()
        for pg in d:
            for l in pg.get_links():
                nm = l.get("nameddest") or l.get("name") or ""
                dest = names.get(nm) if str(nm).startswith("cite.") else None
                if not dest or dest.get("page") is None or dest.get("page") < 0 or not dest.get("to"):
                    continue
                tp = dest["page"]
                x, y = dest["to"]
                out.append((pg.number, tuple(l["from"]), tp, x, d[tp].rect.height - y))
    return out


def _dest_entry(starts: list[tuple], page: int, x: float, y: float) -> str | None:
    """The bibliography entry a link destination points at: the entry whose first line starts nearest the anchor on
    that page, in the same column (within 60 pt across, 25 pt down or up). Checking the words printed at the anchor
    was tried and lost (recall 0.934 -> 0.905, 10x slower)."""
    best, bd = None, 1e9
    for bid, (p, x0, y0) in starts:
        if p != page or abs(x0 - x) > 60 or not (-25 <= y0 - y <= 25):
            continue
        dd = abs(y0 - y) + 0.1 * abs(x0 - x)
        if dd < bd:
            best, bd = bid, dd
    return best


def parse_tei_links(tei: bytes, links: list[tuple], fill_only: bool = False) -> dict:
    """GROBID sentences and entries; citation -> entry from the PDF's own hyperref links where the paper has them
    (a link inside a sentence box decides the entry), GROBID's ref targets otherwise."""
    from lxml import etree
    root = etree.fromstring(tei)
    entries, starts = {}, []
    for b in root.iter(f"{TEI}biblStruct"):
        bid = b.get(XML_ID)
        if not bid:
            continue
        raw = b.find(f"{TEI}note[@type='raw_reference']")
        entries[bid] = re.sub(r"\s+", " ", "".join(raw.itertext()) if raw is not None else
                              " ".join(b.itertext())).strip()
        bx = _boxes(b.get("coords"))
        if bx:
            starts.append((bid, (bx[0][0], bx[0][1], bx[0][2])))
    linked = []
    for pg, (x0, y0, x1, y1), tp, dx, dy in links:
        bid = _dest_entry(starts, tp, dx, dy)
        if bid:
            linked.append((pg, (x0 + x1) / 2, (y0 + y1) / 2, bid))
    use_links = len(linked) >= 5
    pairs = []
    for s in root.iter(f"{TEI}s"):
        keys = []
        if use_links and fill_only:
            # GROBID's resolved refs stand; links only fill refs GROBID left unresolved and markers it did not see
            refs = [r for r in s.iter(f"{TEI}ref") if r.get("type") == "bibr"]
            keys = [r.get("target").lstrip("#") for r in refs if r.get("target")]
            resolved = [b for r in refs if r.get("target") for b in _boxes(r.get("coords"))]
            for p, a0, b0, a1, b1 in _boxes(s.get("coords")):
                for pg, cx, cy, bid in linked:
                    if pg == p and a0 - 1 <= cx <= a1 + 1 and b0 - 2 <= cy <= b1 + 2 and not any(
                            q == pg and c0 - 1 <= cx <= c1 + 1 and d0 - 2 <= cy <= d1 + 2
                            for q, c0, d0, c1, d1 in resolved):
                        keys.append(bid)
        elif use_links and fill_only is None:
            # strict: in a hyperref paper every \cite is a link, so a GROBID ref no link covers is dropped
            for p, a0, b0, a1, b1 in _boxes(s.get("coords")):
                keys += [bid for pg, cx, cy, bid in linked if pg == p and a0 - 1 <= cx <= a1 + 1 and b0 - 2 <= cy <= b1 + 2]
        elif use_links:
            for p, a0, b0, a1, b1 in _boxes(s.get("coords")):
                keys += [bid for pg, cx, cy, bid in linked if pg == p and a0 - 1 <= cx <= a1 + 1 and b0 - 2 <= cy <= b1 + 2]
            # refs GROBID found that no link covers (a marker without a hyperlink) keep GROBID's target
            for r in s.iter(f"{TEI}ref"):
                if r.get("type") != "bibr" or not r.get("target"):
                    continue
                rb = _boxes(r.get("coords"))
                covered = any(pg == p and a0 - 1 <= cx <= a1 + 1 and b0 - 2 <= cy <= b1 + 2
                              for p, a0, b0, a1, b1 in rb for pg, cx, cy, _ in linked)
                if not covered:
                    keys.append(r.get("target").lstrip("#"))
        else:
            keys = [r.get("target", "").lstrip("#") for r in s.iter(f"{TEI}ref") if r.get("type") == "bibr"]
        keys = list(dict.fromkeys(k for k in keys if k in entries))
        sent = re.sub(r"\s+", " ", "".join(s.itertext())).strip()
        pairs += [(k, sent) for k in keys]
    return {"entries": entries, "pairs": pairs, "links": len(linked), "used_links": use_links}


def _grobid_tei_coords(pdf: Path, dest: Path, url: str) -> Path:
    import httpx
    if dest.exists():
        return dest
    for attempt in range(8):
        with open(pdf, "rb") as f:
            r = httpx.post(url, files={"input": (pdf.name, f, "application/pdf")},
                           data={"segmentSentences": "1", "includeRawCitations": "1", "consolidateHeader": "0",
                                 "consolidateCitations": "0", "consolidateFunders": "0",
                                 "teiCoordinates": ["s", "ref", "biblStruct"]},
                           timeout=600, trust_env=False)
        if r.status_code == 503:
            time.sleep(1 + attempt)
            continue
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(r.content)
        tmp.replace(dest)
        return dest
    raise RuntimeError("grobid busy")


def sys_grobid_crf_links(pdf: Path) -> dict:
    tei = _grobid_tei_coords(pdf, OUT / "grobid_crf_coords" / (pdf.stem + ".tei.xml"), GROBID)
    return parse_tei_links(tei.read_bytes(), pdf_cite_links(pdf))


def sys_grobid_full_links(pdf: Path) -> dict:
    tei = _grobid_tei_coords(pdf, OUT / "grobid_full_coords" / (pdf.stem + ".tei.xml"),
                             GROBID.replace(":8070", ":8071"))
    return parse_tei_links(tei.read_bytes(), pdf_cite_links(pdf))


def sys_grobid_crf_fill(pdf: Path) -> dict:
    tei = _grobid_tei_coords(pdf, OUT / "grobid_crf_coords" / (pdf.stem + ".tei.xml"), GROBID)
    return parse_tei_links(tei.read_bytes(), pdf_cite_links(pdf), fill_only=True)


def sys_grobid_full_fill(pdf: Path) -> dict:
    tei = _grobid_tei_coords(pdf, OUT / "grobid_full_coords" / (pdf.stem + ".tei.xml"),
                             GROBID.replace(":8070", ":8071"))
    return parse_tei_links(tei.read_bytes(), pdf_cite_links(pdf), fill_only=True)


def sys_grobid_full_strict(pdf: Path) -> dict:
    tei = _grobid_tei_coords(pdf, OUT / "grobid_full_coords" / (pdf.stem + ".tei.xml"),
                             GROBID.replace(":8070", ":8071"))
    return parse_tei_links(tei.read_bytes(), pdf_cite_links(pdf), fill_only=None)


SYSTEMS = {"pymupdf_asis": sys_pymupdf_asis, "grobid_crf": sys_grobid_crf, "grobid_full": sys_grobid_full,
           "grobid_full_strict": sys_grobid_full_strict,
           "grobid_crf_links": sys_grobid_crf_links, "grobid_full_links": sys_grobid_full_links,
           "grobid_crf_fill": sys_grobid_crf_fill, "grobid_full_fill": sys_grobid_full_fill}

# ---------------------------------------------------------------- scoring


def _norm(s: str) -> list[str]:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"-\s+", "", s)                      # line-end hyphenation in PDF text
    return [w for w in re.findall(r"[a-z0-9]+", s) if w != "math"]


def _bigrams(s: str) -> set:
    w = _norm(SYS_MARK.sub(" ", s))
    return set(zip(w, w[1:])) if len(w) > 1 else set(w)


def _words(s: str) -> set:
    return {w for w in _norm(s) if len(w) >= 3}


def align_entries(g: dict, s: dict) -> dict:
    gw = {k: _words(v) for k, v in g.items()}
    sw = {k: _words(v) for k, v in s.items()}
    cand = []
    for sk, a in sw.items():
        for gk, b in gw.items():
            if a and b:
                j = len(a & b) / len(a | b)
                if j >= 0.5:
                    cand.append((j, sk, gk))
    out, used = {}, set()
    for j, sk, gk in sorted(cand, reverse=True):
        if sk not in out and gk not in used:
            out[sk] = gk
            used.add(gk)
    return out


def _match(gs: str, ss: str) -> bool:
    """Same sentence up to segmentation: one is mostly contained in the other (a parser that splits a gold sentence
    in two, or joins two, still carries the citation's own context). Over-long system sentences are reported
    separately (sys_sentence_chars)."""
    a, b = _bigrams(gs), _bigrams(ss)
    if not a or not b:
        return False
    i = len(a & b)
    return i >= 4 and max(i / len(a), i / len(b)) >= 0.7


def _in_context(ctx: str, ss: str) -> bool:
    b = _bigrams(ss)
    return bool(b) and len(b & _bigrams(ctx)) / len(b) >= 0.6


def score_paper(g: dict, s: dict) -> dict:
    amap = align_entries(g["entries"], s["entries"])
    spairs = [(amap.get(k), t) for k, t in s["pairs"]]
    rec = sum(1 for gk, gt in g["prose"] if any(sk == gk and _match(gt, st) for sk, st in spairs))
    prec = 0
    for sk, st in spairs:
        if sk is None:
            continue
        if any(gk == sk and _match(gt, st) for gk, gt in g["prose"]) or \
                any(_in_context(c, st) for c in g["other"].get(sk, [])):
            prec += 1
    # citation-graph view: which cited entries does the paper cite in prose at all (independent of sentences)
    gold_cited = {gk for gk, _ in g["prose"]}
    sys_cited = {sk for sk, _ in spairs if sk is not None}
    return {"gold_entries": len(g["entries"]), "sys_entries": len(s["entries"]), "aligned": len(amap),
            "gold_pairs": len(g["prose"]), "sys_pairs": len(spairs), "recalled": rec, "correct": prec,
            "gold_cited": len(gold_cited), "cited_recalled": len(gold_cited & sys_cited),
            "sys_sentence_chars": sum(len(t) for _, t in s["pairs"])}


# ---------------------------------------------------------------- driver


def sample() -> list[dict]:
    rows = [json.loads(l) for l in open(OUT / "sample.jsonl", encoding="utf-8")]
    return [r for r in rows if r["pdf"] and r["html"]]


def run(system: str, workers: int = 1) -> None:
    fn = SYSTEMS[system]
    rows = sample()
    dest = OUT / "out" / system
    dest.mkdir(parents=True, exist_ok=True)

    def one(r):
        t0 = time.time()
        try:
            res = fn(OUT / r["pdf"])
            res["seconds"] = time.time() - t0
        except Exception as e:      # recorded, scored as zero
            res = {"entries": {}, "pairs": [], "error": f"{type(e).__name__}: {e}", "seconds": time.time() - t0}
        (dest / f"{r['arxiv_id']}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        return res
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, rows))
    wall = time.time() - t0
    errs = sum(1 for x in res if "error" in x)
    print(f"{system}: {len(rows)} papers, wall {wall:.1f}s ({wall / len(rows):.2f} s/paper at {workers} workers), "
          f"errors {errs}")


def score() -> None:
    rows = sample()
    golds = {r["arxiv_id"]: gold((OUT / r["html"]).read_text(encoding="utf-8", errors="replace")) for r in rows}
    systems = sorted(p.name for p in (OUT / "out").iterdir() if p.is_dir())
    table = {}
    for system in systems:
        tot = {"gold_entries": 0, "sys_entries": 0, "aligned": 0, "gold_pairs": 0, "sys_pairs": 0, "recalled": 0,
               "correct": 0, "gold_cited": 0, "cited_recalled": 0, "sys_sentence_chars": 0}
        per, zero, secs = {}, 0, []
        for aid, g in golds.items():
            f = OUT / "out" / system / f"{aid}.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text(encoding="utf-8"))
            secs.append(s.get("seconds", 0))
            m = score_paper(g, s)
            per[aid] = m
            for k in tot:
                tot[k] += m[k]
            zero += m["sys_pairs"] == 0 and m["gold_pairs"] > 0
        n = len(per)
        pr = [m["recalled"] / m["gold_pairs"] for m in per.values() if m["gold_pairs"]]
        table[system] = {"papers": n, **tot, "recall": tot["recalled"] / max(1, tot["gold_pairs"]),
                         "precision": tot["correct"] / max(1, tot["sys_pairs"]),
                         "entry_recall": tot["aligned"] / max(1, tot["gold_entries"]),
                         "cited_recall": tot["cited_recalled"] / max(1, tot["gold_cited"]),
                         "mean_sentence_chars": tot["sys_sentence_chars"] / max(1, tot["sys_pairs"]),
                         "papers_zero_pairs": zero,
                         "papers_recall_lt_50": sum(1 for x in pr if x < 0.5),
                         "median_paper_recall": sorted(pr)[len(pr) // 2] if pr else 0,
                         "mean_seconds": sum(secs) / max(1, len(secs)), "per_paper": per}
    (OUT / "score.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    for system, t in table.items():
        print(f"{system:14s} papers {t['papers']:3d}  recall {t['recall']:.3f}  precision {t['precision']:.3f}  "
              f"entry_recall {t['entry_recall']:.3f}  cited_recall {t['cited_recall']:.3f}  "
              f"sent_chars {t['mean_sentence_chars']:.0f}  zero-pair papers {t['papers_zero_pairs']:2d}  "
              f"papers recall<0.5 {t['papers_recall_lt_50']:2d}  median paper recall {t['median_paper_recall']:.3f}  "
              f"pairs gold/sys {t['gold_pairs']}/{t['sys_pairs']}  {t['mean_seconds']:.2f}s/paper")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    else:
        score()
