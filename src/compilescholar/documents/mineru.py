# -*- coding: utf-8 -*-
"""MinerU content_list -> units: the careful tier's document (deep-extraction subset; tables, formulas, layout).

content_list (one JSON list of blocks, reading order): type text (text_level = heading level), list (sub_type
ref_text = the bibliography), table (table_body HTML, table_caption, table_footnote), equation (LaTeX), image / chart
(captions), page_number / header / footer / aside_text / page_footnote (layout furniture: dropped); bbox is on a
0-1000 grid of the page, page_idx 0-based. Units: section, para, table (HTML kept for documents.tables), caption,
formula, footnote — each with page and bbox; the reference list becomes `references` (one string per entry) for the
citation stage. Sentences and citation pairs come from the fast tier of the same PDF; the careful tier adds structure,
not a second citation reading (INTEGRATED-SYSTEM-1005 v2.2)."""
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import asdict, dataclass

DROP = {"page_number", "header", "footer", "aside_text", "page_header", "page_footer"}


@dataclass
class Unit:
    uid: str
    kind: str              # section | para | table | caption | formula | footnote
    section: str
    text: str
    page: int | None = None
    bbox: list | None = None
    html: str | None = None


def _txt(s) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def load(path) -> list[dict]:
    with zipfile.ZipFile(path) as z:
        n = next(x for x in z.namelist() if x.endswith("_content_list.json"))
        return json.loads(z.read(n))


def units(blocks: list[dict], doc_id: str = "doc") -> tuple[list[Unit], list[str]]:
    out, refs, path = [], [], []
    n: dict[str, int] = {}

    def add(kind, text, b, html=None):
        n[kind] = n.get(kind, 0) + 1
        out.append(Unit(f"{doc_id}#{kind}{n[kind]}", kind, " > ".join(path), text, b.get("page_idx"), b.get("bbox"),
                        html))

    for b in blocks:
        t = b.get("type")
        if t in DROP:
            continue
        if t == "text":
            text = _txt(b.get("text"))
            if not text:
                continue
            lvl = b.get("text_level")
            if lvl:
                num = re.match(r"^(\d+(?:\.\d+)*)\.?\s", text)
                lvl = num.group(1).count(".") + 1 if num else int(lvl)
                path[:] = path[:max(0, lvl - 1)] + [text]
                add("section", text, b)
            else:
                add("para", text, b)
        elif t == "list":
            items = [_txt(x) for x in b.get("list_items") or [] if _txt(x)]
            if b.get("sub_type") == "ref_text":
                refs += items
            else:
                for x in items:
                    add("para", x, b)
        elif t == "table":
            cap = _txt(" ".join(b.get("table_caption") or []))
            add("table", cap, b, html=b.get("table_body") or "")
            if cap:
                add("caption", cap, b)
            for fn in b.get("table_footnote") or []:
                if _txt(fn):
                    add("footnote", _txt(fn), b)
        elif t == "equation":
            add("formula", _txt(b.get("text")), b)
        elif t in ("image", "chart"):
            cap = _txt(" ".join(b.get("image_caption") or b.get("chart_caption") or []))
            if cap:
                add("caption", cap, b)
        elif t == "page_footnote":
            if _txt(b.get("text")):
                add("footnote", _txt(b.get("text")), b)
    return out, refs


def to_dict(us: list[Unit], refs: list[str]) -> dict:
    return {"units": [asdict(u) for u in us], "references": refs}
