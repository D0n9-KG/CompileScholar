# -*- coding: utf-8 -*-
"""Build RESULTS-LEDGER.docx from RESULTS-LEDGER.md (living document).

Rebuild after every ledger update:  python build_ledger_docx.py
Style follows the method chapter pipeline: SimSun body, three-line tables,
A4, no vertical borders. Parses: #/##/### headings, pipe tables (caption =
the immediately preceding paragraph line), plain paragraphs.
"""
import io
import re
import sys

sys.path.insert(0, "C:/Users/D0n9/Desktop/LogicKG/.research_tmp/paper_drafts")
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = "C:/Users/D0n9/Desktop/LogicKG/.research_tmp/paper_drafts"
SRC = f"{HERE}/RESULTS-LEDGER.md"
OUT = f"{HERE}/RESULTS-LEDGER.docx"


def set_font(run, name_cn, size_pt, bold=False):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    r = run._element.rPr.rFonts
    r.set(qn("w:eastAsia"), name_cn)


def para(doc, text, size=10.5, bold=False, align=None, indent=False):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    set_font(run, "宋体", size, bold)
    return p


def heading(doc, text, level):
    sizes = {1: 15, 2: 13, 3: 11.5}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10 if level > 1 else 0)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    set_font(run, "黑体", sizes.get(level, 11), bold=True)


def cell_border_none(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        borders.append(el)
    tcPr.append(borders)


def hline(row, edge, sz):
    for cell in row.cells:
        tcPr = cell._tc.get_or_add_tcPr()
        borders = tcPr.find(qn("w:tcBorders"))
        if borders is None:
            borders = OxmlElement("w:tcBorders")
            tcPr.append(borders)
        old = borders.find(qn(f"w:{edge}"))
        if old is not None:
            borders.remove(old)
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:color"), "000000")
        borders.append(el)


def three_line_table(doc, caption, headers, rows):
    if caption:
        para(doc, caption, size=9, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    ncol = len(headers)
    t = doc.add_table(rows=1 + len(rows), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    # column widths weighted by max content length
    maxlen = [max([len(str(headers[i]))] + [len(str(r[i])) if i < len(r) else 0
                                          for r in rows]) for i in range(ncol)]
    total = sum(maxlen) or 1
    usable = 15.4
    for i in range(ncol):
        w = Cm(max(1.4, usable * maxlen[i] / total))
        for cell in t.columns[i].cells:
            cell.width = w
    for j, h in enumerate(headers):
        c = t.rows[0].cells[j]
        c.text = ""
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(str(h))
        set_font(run, "宋体", 9, bold=True)
    for i, row in enumerate(rows):
        for j in range(ncol):
            c = t.rows[i + 1].cells[j]
            c.text = ""
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(str(row[j]) if j < len(row) else "")
            set_font(run, "宋体", 9)
    for row in t.rows:
        for cell in row.cells:
            cell_border_none(cell)
    hline(t.rows[0], "top", 12)
    hline(t.rows[0], "bottom", 6)
    hline(t.rows[-1], "bottom", 12)
    para(doc, "", size=6)


def split_row(line):
    parts = [c.strip() for c in line.strip().strip("|").split("|")]
    return parts


def main():
    text = io.open(SRC, encoding="utf-8").read()
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.2)
    sec.left_margin = sec.right_margin = Cm(2.4)

    lines = text.split("\n")
    i = 0
    pending_caption = None
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("|"):
            # collect the whole table block
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(split_row(lines[i]))
                i += 1
            # drop separator rows (---)
            block = [r for r in block if not all(re.fullmatch(r":?-{2,}:?", c or "--") for c in r)]
            headers, rows = block[0], block[1:]
            three_line_table(doc, pending_caption, headers, rows)
            pending_caption = None
            continue
        if s.startswith("### "):
            heading(doc, s[4:], 3)
        elif s.startswith("## "):
            heading(doc, s[3:], 2)
        elif s.startswith("# "):
            heading(doc, s[2:], 1)
        elif re.match(r"^表\s*[A-Z]?\d+", s):
            pending_caption = s
        else:
            para(doc, s, indent=not s.startswith(("注：", "一、", "二、", "三、", "四、", "五、")))
            pending_caption = None
        i += 1

    doc.save(OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
