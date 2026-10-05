# -*- coding: utf-8 -*-
"""Result pass (L2c): a paper's own result tables -> single-paper result units (DESIGN-LITERATURE-LAYER N2).

Deterministic, zero LLM: documents.tables.extract_tables (the old table_channel, moved unchanged) parses every numeric
table, expands row/col spans and multi-tier headers, splits a dataset tier off the column path when it is a real
column group, and emits one record per (row x numeric column) whose quote is the verbatim table row. Rows it cannot map
safely (fused cells) are skipped and counted, never guessed.

Scope rule (user ruling 10-05): result units are valid only inside their own paper. They are written as self
statements about the paper (facet=result) and are shown in that paper's card; no stage aligns numbers across papers.

LaTeXML HTML tables (<table class="ltx_tabular" ...>) are normalised to bare <table> tags so the same parser reads
both renderings."""
from __future__ import annotations

import re

from ..documents import tables as T
from .schema import Statement

TABLE = re.compile(r"<table\b[^>]*>(.*?)</table>", re.S)
MATH = re.compile(r"<math\b.*?</math>", re.S)
ALT = re.compile(r'alttext="([^"]*)"')
ATTRS = re.compile(r"<(tr|td|th)\b([^>]*)>")
SPAN = re.compile(r'\b(rowspan|colspan)="(\d+)"')
INLINE = re.compile(r"</?(?:span|a|em|strong|b|i|sup|sub|thead|tbody|tfoot|p|div|br\s*/?)\b[^>]*>")


def normalise_html_tables(page: str) -> str:
    """LaTeXML tables -> the bare MinerU-style <table><tr><td> markup tables.extract_tables reads: math replaced by
    its TeX alttext, presentational tags and attributes removed (rowspan/colspan kept)."""
    def table(m):
        t = MATH.sub(lambda x: (ALT.search(x.group(0)) or [None, ""])[1] if ALT.search(x.group(0)) else " ", m.group(1))
        t = ATTRS.sub(lambda x: "<" + x.group(1) + "".join(f' {k}="{v}"' for k, v in SPAN.findall(x.group(2))) + ">", t)
        t = INLINE.sub("", t)
        t = re.sub(r"\s+", " ", t)
        return "<table>" + t + "</table>"
    return TABLE.sub(table, page)


def run(arxiv_id: str, date: str, raw: str, source: str, own_methods: tuple[str, ...] = ()) -> tuple[list[Statement], dict]:
    text = normalise_html_tables(raw) if source == "arxiv_html" else raw
    recs, residue = T.extract_tables(arxiv_id, text, own_methods=own_methods)
    out = []
    for r in recs:
        m = r["measure"]
        method = r["method_ref"]["surface"]
        dataset = ((r.get("dims_new") or {}).get("dims.subject") or [""])[0]
        txt = f"{method} — {dataset + ' / ' if dataset else ''}{m['metric']}: {m['value']}{(' ' + m['unit']) if m['unit'] else ''}"
        out.append(Statement(arxiv_id, date, "self", f"paper:{arxiv_id}", "describes", "result", txt[:300],
                             r["quote"], meta={"pass": "results", "method": method, "dataset": dataset,
                                               "metric": m["metric"], "value": m["value"], "unit": m["unit"],
                                               "direction": m["direction"], "row_role": r["role"],
                                               "epistemic": r["epistemic"], "table_header": r.get("table_header"),
                                               "section": r.get("section")}))
    return out, {"result_units": len(out), "tables_with_residue": len(residue)}
