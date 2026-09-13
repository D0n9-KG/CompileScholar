# -*- coding: utf-8 -*-
"""F24: deterministic table-parsing channel (PSFIX/PSV3 queue; brief at
.research_tmp/docs_decisions/F24-TABLE-CHANNEL-BRIEF.md).

Rationale (measured, COACH2/PSV3 archives): LLM extraction of benchmark
tables fails in four reproducible forms — row sampling (only representative
rows), husk collapse under enumeration pressure (quote+header perfect, all
fields empty x40), fused-cell fragments (mineru "91.691.4"), and column
misbinding (A7). All four are STRUCTURAL transport problems, and the LLM's
reliable skill is verbatim row copying — so: parse tables deterministically
(zero LLM), emit result records that pass the postcheck gates by
construction (quote = verbatim row, value = verbatim cell), and honestly
skip what cannot be parsed (fused/ambiguous -> visible overflow residue).

v0 scope (brief §2):
  - HTML <table> blocks (mineru) + pipe tables (markdown)
  - rowspan/colspan grid expansion; multi-tier header path ("NLP > QA RoBERTa-L")
  - numeric benchmark tables only (>=1 data row with >=2 numeric cells and a
    row-header column); taxonomy/notation/text tables skipped
  - one record per (data row x numeric column) cell
  - fused multi-value cells (ambiguous column mapping) -> row skipped -> residue
  - role: own-method anchor (identity card) -> main_result else
    baseline_comparison, both flagged role_provisional (brief §2.1)
  - epistemic: Related Work/References section -> cited else demonstrated
    (brief §2.2; canary T2 guards this)
  - direction/unit from header glyphs ((%), up/down arrows)
  - dedup vs existing KB records on (paper, row-quote, value): LLM record wins
Zero-LLM, corpus-generic (no benchmark-specific logic).

CLI:
  python -m kb_compiler.records.table_channel \
      --texts DIR --checked records_checked.json --cards cards.json \
      --out table_records.json [--merge-out merged.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from html.parser import HTMLParser

SECTION_CITED = re.compile(
    r"(related\s+work|references|bibliography|literature)", re.I)
NUMCELL = re.compile(r"^[^A-Za-z]*[-+]?\d")          # starts like a number
LATEX_NUM = re.compile(r"(?:(?<=\$)|(?<=\s))[-+]?\d[\d\s,]*\.?[\d\s]*(?=(?:\\pm|\$|\s|$))")


_FULLNUM = re.compile(r"^\d{2,}[\d,]*\.?\d*$|^\d+\.\d+$")


def _fold_num(cell: str) -> str:
    """Numeric cell -> clean value string, or '' when not cleanly numeric.

    PS16-validation fixes (2026-09-12): (1) FUSED-cell detection BEFORE space
    folding — '1536 85.3' / '39.7 33.6' are two values in one cell (mineru
    colspan artifacts); folding their space would mint a fake number
    ('153685.3'). LaTeX digit-spacing ('$8 8 . 9$') is distinguished by
    complete-number token count: spaced single chars fold, 2+ complete
    numbers fuse. (2) multi-dot tokens ('91.691.4') = fused -> ''. (3) digit
    spacing IS folded in the emitted value ('88.9 ± 0.01') so values are
    readable AND dedup-joinable against LLM records; the postcheck FG6
    ws-stripped channel still matches the verbatim quote."""
    t = cell.strip()
    if not t:
        return ""
    t2 = re.sub(r"^\$+|\$+$", "", t).strip()
    # scientific notation before brace strip: \times 10^{-5} -> e-5
    t2 = re.sub(r"\\times\s*10\s*\^\s*\{?\s*([-−]?)(\d+)\s*\}?",
                lambda m: "e" + ("-" if m.group(1) in "-−" else "") + m.group(2), t2)
    t2 = t2.replace("\\pm", "±").replace("\\%", "%")
    t2 = re.sub(r"\\(?:bf|mathbf|textbf|mathrm)\s*", "", t2)
    t2 = re.sub(r"[{}]", "", t2)
    t2 = re.sub(r"_\s*±", " ±", t2)   # subscript std: 0.84_± 0.11
    t2 = t2.replace("−", "-")
    # (1) fused-cell detection on the raw (pre-fold) cell
    toks = [x for x in re.split(r"\s+", t2) if x]
    full = [x for x in toks if _FULLNUM.match(x.replace(",", "").rstrip("."))]
    if len(full) >= 2:
        return ""                      # two+ complete numbers in one cell
    if t2.count(".") > 1 and "±" not in t2 and " " not in t2.strip():
        return ""                      # '91.691.4' single-token multi-dot fuse
    # (2) fold latex digit spacing
    v = re.sub(r"(?<=\d)\s+(?=\d)", "", t2)
    v = re.sub(r"(?<=\d)\s+(?=\.)", "", v)
    v = re.sub(r"(?<=\.)\s+(?=\d)", "", v)
    v = re.sub(r"\s*±\s*", " ± ", v)
    # (2b) unit suffix + parenthetical ("75.40 GB", "3.35 h", "0.00 GB (0%)")
    # — PS16-validation fix: nN8T §6 memory/time table rows were lost whole.
    # Parenthetical secondary values ("62.55 (70.34)") are dropped (disclosed
    # conservative loss; LLM channel may capture them separately).
    v = re.sub(r"\([^)]*\)", " ", v)
    m = re.match(r"^([-+]?[\d][\d,\.]*(?:\s*±\s*[\d][\d,\.]*)?)\s*"
                 r"([A-Za-zµ%‰][A-Za-z0-9/\.\-]{0,11})?$", v.strip())
    if m:
        v = m.group(1).strip()
    v = re.sub(r"\s+", " ", v).strip()
    # (3) validate: optionally ± std, each part exactly one number
    parts = re.split(r"±", v)
    for p in parts:
        p = p.strip().rstrip("%‰").strip()
        if not re.fullmatch(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", p):
            return ""
        if p.count(".") > 1:
            return ""
    return v


def _cell_unit(cell: str) -> str:
    """Trailing unit token of a numeric cell ('75.40 GB' -> 'GB')."""
    t = re.sub(r"\([^)]*\)", " ", re.sub(r"^\$+|\$+$", "", str(cell).strip()))
    t = re.sub(r"\\(?:bf|mathbf|textbf|mathrm)\s*|[{}]", "", t)
    m = re.search(r"[-+]?\d[\d,\.]*\s*([A-Za-zµ%‰][A-Za-z0-9/\.\-]{0,11})\s*$", t.strip())
    return m.group(1) if m else ("%" if "%" in t else "")


def _bare_numbers(cell: str) -> list:
    """All standalone numbers in a cell (fused-cell detection)."""
    s = re.sub(r"[{}$\\]", " ", cell)
    return re.findall(r"(?<![\d.])\d[\d,]*\.?\d*(?![\d])", s)


class _TableHTML(HTMLParser):
    """<table> -> list of rows; each row = list of (text, rowspan, colspan)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []          # list of tables
        self._cur_table = None
        self._cur_row = None
        self._cur_cell = None
        self._cell_attrs = None
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                self._cur_table = []
        elif tag == "tr" and self._cur_table is not None:
            self._cur_row = []
        elif tag in ("td", "th") and self._cur_row is not None:
            self._cell_attrs = a
            self._cur_cell = []
        elif tag == "br" and self._cur_cell is not None:
            self._cur_cell.append(" ")

    def handle_endtag(self, tag):
        if tag == "table" and self._depth > 0:
            self._depth -= 1
            if self._depth == 0 and self._cur_table is not None:
                if self._cur_table:
                    self.tables.append(self._cur_table)
                self._cur_table = None
        elif tag == "tr" and self._cur_row is not None:
            if self._cur_table is not None and self._cur_row:
                self._cur_table.append(self._cur_row)
            self._cur_row = None
        elif tag in ("td", "th") and self._cur_cell is not None:
            txt = re.sub(r"\s+", " ", "".join(self._cur_cell)).strip()
            span = lambda k: max(1, int(re.sub(r"\D", "", self._cell_attrs.get(k, "")) or 1))
            self._cur_row.append((txt, span("rowspan"), span("colspan")))
            self._cur_cell = None

    def handle_data(self, data):
        if self._cur_cell is not None:
            self._cur_cell.append(data)


def _expand_grid(rows):
    """rowspan/colspan expansion -> rectangular grid (list of rows of str|None)
    plus a parallel grid of 'is_row_header_continuation' flags (spanned cells
    repeat the origin value). Returns (grid, spanned) where spanned[r][c]=True
    when the cell is a span CONTINUATION (not the origin)."""
    grid, spanned = [], []
    pend = {}   # (r,c) -> (text, is_cont)
    for r, row in enumerate(rows):
        grow, srow = [], []
        c = 0
        cells = iter(row)
        while True:
            if (r, c) in pend:
                t, cont = pend.pop((r, c))
                grow.append(t); srow.append(cont)
                c += 1
                continue
            try:
                txt, rs, cs = next(cells)
            except StopIteration:
                break
            grow.append(txt); srow.append(False)
            for dr in range(rs):
                for dc in range(cs):
                    if dr == 0 and dc == 0:
                        continue
                    pend[(r + dr, c + dc)] = (txt, True)
            c += 1
        # drain pending for trailing columns
        while (r, c) in pend:
            t, cont = pend.pop((r, c))
            grow.append(t); srow.append(cont); c += 1
        grid.append(grow); spanned.append(srow)
    width = max((len(g) for g in grid), default=0)
    for g, s in zip(grid, spanned):
        while len(g) < width:
            g.append(None); s.append(False)
    return grid, spanned


def _is_data_cell(t):
    """Data (value) cell: starts numeric (after $/sign/latex) and letters are
    a small minority (unit suffixes like GB/h/%). Header labels like 'T5-B',
    'QA RoBERTa-L', 'mMR-2' contain digits but start with a letter or are
    letter-heavy — the digit-in-label form broke the naive numeric test
    (canary validation catch: 'Summ T5-B' misread as data, tier-2 header
    lost from the column path)."""
    if t is None:
        return False
    s = str(t).strip()
    if not s:
        return False
    s2 = s.lstrip("$+-−± ").strip()
    if not s2 or not (s2[0].isdigit() or s2[0] in ".<"):
        return False
    letters = sum(1 for ch in s if ch.isalpha())
    return letters / max(1, len(s)) < 0.34


def _is_header_cell(t):
    return t is not None and t != "" and not _is_data_cell(t)


def parse_pipe_tables(text):
    """Markdown pipe tables -> same row format as _TableHTML (no spans).
    Separator row optional (canary validation catch: synthetic/hand-written
    pipe tables without |---| were skipped whole); a pipe block qualifies
    when its first row is header-like (all non-data cells) and >=2 rows."""
    out = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|") and lines[i].count("|") >= 2:
            block = []
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                if re.match(r"^\s*\|[\s:\-|]+\|\s*$", lines[j]):
                    j += 1
                    continue   # separator row: skip, table continues
                cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                block.append([(c, 1, 1) for c in cells])
                j += 1
            if len(block) >= 2 and all(not _is_data_cell(x[0]) for x in block[0]):
                out.append((block, text.find(lines[i]), "\n".join(lines[i:j])))
            i = j if j > i else i + 1
        else:
            i += 1
    return out


def _nearest_heading(text, pos):
    h = ""
    for m in re.finditer(r"^#+\s*(.+)$", text[:pos], flags=re.M):
        h = m.group(1).strip()
    return h


def _rid(pid, kind, fp):
    return hashlib.md5(f"{pid}|{kind}|{fp}".encode("utf-8")).hexdigest()[:14]


def extract_tables(pid, text, own_methods=(), manifest_title=""):
    """Deterministic table -> result records + overflow residue.
    own_methods: normalized surfaces of the paper's own method (identity card
    canonical+aliases) for the provisional role decision."""
    records, residue = [], []
    blocks = []   # (rows, char_pos, raw_quote_getter)
    # HTML tables with their source spans (verbatim row quotes come from text)
    for m in re.finditer(r"<table>.*?</table>", text, flags=re.S):
        raw = m.group(0)
        p = _TableHTML()
        try:
            p.feed(raw)
        except Exception:
            continue
        for t in p.tables:
            blocks.append((t, m.start(), raw))
    for rows, pos, raw in parse_pipe_tables(text):
        blocks.append((rows, pos, raw))

    own = {re.sub(r"\s+", " ", o.strip().lower()) for o in own_methods if o}
    for rows, tpos, traw in blocks:
        if len(rows) < 2:
            continue
        grid, spanned = _expand_grid(rows)
        if not grid or len(grid) < 2:
            continue
        # ---- header split: leading rows whose cells are all non-numeric ----
        n_hdr = 0
        for gi, grow in enumerate(grid):
            body = [c for c in grow if c not in (None, "")]
            if body and all(_is_header_cell(c) for c in body):
                n_hdr = gi + 1
            else:
                break
        if n_hdr == 0 or n_hdr >= len(grid):
            continue
        headers = grid[:n_hdr]
        data = grid[n_hdr:]
        # ---- column header paths (multi-tier join) ----
        ncol = len(grid[0])
        colpath = []
        for c in range(ncol):
            parts = []
            for h in headers:
                v = h[c] if c < len(h) else None
                if v not in (None, "") and (not parts or parts[-1] != v):
                    parts.append(v)
            colpath.append(" > ".join(parts))
        # ---- row-header column count: leading columns that are non-numeric
        # in (almost) every data row ----
        rh = 0
        for c in range(ncol):
            col = [row[c] for row in data if c < len(row)]
            vals = [x for x in col if x not in (None, "")]
            if vals and sum(1 for x in vals if _fold_num(x)) <= len(vals) * 0.2:
                rh = c + 1
            else:
                break
        rh = min(rh, max(0, ncol - 2))   # keep >=2 value columns
        # ---- numeric-column detection + per-cell emission ----
        numeric_cols = []
        for c in range(rh, ncol):
            col = [row[c] for row in data if c < len(row)]
            vals = [x for x in col if x not in (None, "")]
            if vals and sum(1 for x in vals if _fold_num(x)) >= max(1, len(vals) // 2):
                numeric_cols.append(c)
        if len(numeric_cols) < 1:
            continue   # text/taxonomy table -> out of v0 scope
        section = _nearest_heading(text, tpos)
        epistemic = "cited" if SECTION_CITED.search(section or "") else "demonstrated"
        emitted_any = False
        fused_rows = []
        for row in data:
            rowhead = " ".join(str(row[c]) for c in range(rh)
                               if c < len(row) and row[c] not in (None, ""))
            if not rowhead.strip():
                continue
            # fused-cell guard: any numeric-position cell that is not cleanly
            # foldable AND carries multiple number tokens -> row ambiguous ->
            # visible residue (never guess column mapping)
            fused = False
            cells = {}
            for c in numeric_cols:
                if c >= len(row) or row[c] in (None, ""):
                    continue
                v = _fold_num(row[c])
                if v:
                    cells[c] = v
                elif len(_bare_numbers(str(row[c]))) >= 1 and \
                        re.search(r"\d\s+\d|\d\.\d+\.\d", str(row[c])):
                    fused = True
                elif re.search(r"[A-Za-z]{3,}", str(row[c])):
                    # PS16 manual-read catch: a TEXT cell inside the numeric
                    # zone = two logical rows fused into one <tr> (lXuB
                    # 'YoutubeSubtitles ... Wikipedia (en) ...' bound -0.0220
                    # to the wrong sub-row). Never guess — residue.
                    fused = True
            # mid-row text guard (PS16 manual-read catch #2): a TEXT cell
            # between the row-head zone and the last emitted numeric column
            # means multiple logical rows fused into one <tr> (lXuB table:
            # [ds1, v,v,v, 'Wikipedia (en)', v,v,v] — the whole text column
            # is non-numeric so per-column detection cannot see it; values
            # after the embedded label were bound to the FIRST sub-row).
            if cells and not fused:
                last_num = max(cells)
                for c in range(rh, last_num):
                    x = row[c] if c < len(row) else None
                    if x not in (None, "") and c not in cells and \
                            re.search(r"[A-Za-z]{3,}", str(x)):
                        fused = True
                        break
            if fused:
                fused_rows.append(rowhead)
                continue
            if not cells:
                continue   # row without clean numerics: out of v0 scope
            # verbatim row quote: the <tr>/pipe segment INSIDE this table's raw
            # span (PS16-validation fix: full-text probe found prose mentions
            # of the row-head word first -> wrong quote -> wrong provenance)
            cellvals = [str(row[c]) for c in cells]
            q = _row_quote(traw, rowhead, cellvals)
            if not q:
                continue
            hdr_q = _row_quote(traw, str(headers[-1][rh] if rh < len(headers[-1]) else ""),
                               [str(x) for x in headers[-1] if x]) or ""
            is_own = any(o and (o in rowhead.lower() or rowhead.lower() in o)
                         for o in own)
            for c, v in cells.items():
                head = colpath[c] if c < len(colpath) else ""
                metric = re.sub(r"\s+", " ", head).strip()[:80]
                direction = ""
                if "↑" in head or "higher" in head.lower():
                    direction = "higher_better"
                elif "↓" in head or "lower" in head.lower():
                    direction = "lower_better"
                unit = _cell_unit(row[c] if c < len(row) else "") or \
                    ("%" if "%" in head else "")
                fp = re.sub(r"\s+", " ", f"{rowhead}|{metric}|{v}").strip().lower()
                records.append({
                    "kind": "result",
                    "method_ref": {"surface": rowhead.strip()[:60], "canonical": None,
                                   "entity_id": None},
                    "measure": {"metric": metric or (rowhead.strip()[:40]),
                                "value": v, "unit": unit,
                                "direction": direction, "aggregation": "",
                                "timepoint": ""},
                    "role": "main_result" if is_own else "baseline_comparison",
                    "role_provisional": True,   # brief §2.1 (own-method anchor)
                    "dims": {},
                    "epistemic": epistemic,
                    "quote": q[:400],
                    "table_header": hdr_q[:300],
                    "paper_id": pid,
                    "chunk_id": f"{pid}#table",
                    "section": section or "(table)",
                    "chunk_char_start": tpos,
                    "id": _rid(pid, "result", fp),
                    "provenance": "table_channel_v1",
                })
                emitted_any = True
        if fused_rows:
            residue.append({"reason": "table_channel: fused/ambiguous cells — "
                                      "columns cannot be mapped deterministically",
                            "quote": _table_snippet(text, tpos),
                            "rows": fused_rows[:6]})
    return records, residue


def _row_quote(table_raw, rowhead, cellvals):
    """Verbatim row segment WITHIN the table's raw source span. Best-match
    scoring: the segment containing the MOST of this row's cell values wins
    (row-head word is a tie-breaker bonus, NOT a hard requirement — rowspan
    continuation rows legitimately lack the head in their physical segment).
    PS16-validation fix: the old head+first-value dual probe mis-matched
    rowspan continuation rows onto the group's first physical row whenever
    one value coincided (LLaMA-13B 4-bit's MMLU 47.4 == Original's 47.4 ->
    wrong provenance; postcheck correctly rejected 18 such records).
    Threshold: >=60% of values present, else None (honest skip, never guess).
    """
    vals = [re.sub(r"\s+", "", str(v)) for v in cellvals]
    vals = [v for v in vals if len(v) >= 2]
    if not vals:
        return None
    probe_head = ""
    if rowhead:
        w = [x for x in re.split(r"\s+", rowhead.strip()) if len(x) >= 3]
        probe_head = re.sub(r"\s+", "", w[0][:40]) if w else ""
    segs = re.findall(r"<tr>.*?</tr>", table_raw, flags=re.S)
    if not segs:
        segs = [l.strip() for l in table_raw.split("\n") if l.strip().startswith("|")]
    best, best_score = None, -1.0
    for seg in segs:
        segn = re.sub(r"\s+", "", seg)
        score = sum(1 for v in vals if v in segn)
        if probe_head and probe_head in segn:
            score += 0.5
        if score > best_score:
            best, best_score = seg, score
    if best is not None and best_score >= max(2, len(vals) * 0.6):
        return best.strip()
    return None


def _table_snippet(text, pos):
    return text[pos:pos + 300].replace("\n", " ")


def dedup_against_kb(new_records, checked):
    """(paper, row-quote-prefix, value) join: existing LLM record wins."""
    seen = set()
    for pid, b in (checked or {}).items():
        for r in b.get("records", []):
            if r.get("kind") != "result":
                continue
            v = str((r.get("measure") or {}).get("value") or "")
            q = re.sub(r"\s+", " ", str(r.get("quote") or ""))[:80].lower()
            seen.add((pid, re.sub(r"[\s,]", "", v).lower(), q))
    out, dupes = [], 0
    for r in new_records:
        key = (r["paper_id"],
               re.sub(r"[\s,]", "", r["measure"]["value"]).lower(),
               re.sub(r"\s+", " ", r.get("quote") or "")[:80].lower())
        if key in seen:
            dupes += 1
            continue
        out.append(r)
    return out, dupes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--checked", required=True, help="existing records_checked.json (dedup base)")
    ap.add_argument("--cards", default="", help="identity cards for own-method role anchor")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    checked = json.load(open(args.checked, encoding="utf-8"))
    cards = json.load(open(args.cards, encoding="utf-8")) if args.cards and os.path.exists(args.cards) else {}
    all_new, all_res, per_paper = [], [], {}
    for fn in sorted(os.listdir(args.texts)):
        if not fn.endswith((".md", ".txt")) or fn.rsplit(".", 1)[0] == "canary":
            continue
        pid = fn.rsplit(".", 1)[0]
        text = open(os.path.join(args.texts, fn), encoding="utf-8", errors="replace").read()
        mi = (cards.get(pid) or {}).get("method_identity") or {}
        own = [mi.get("canonical_name") or ""] + (mi.get("aliases") or [])
        recs, res = extract_tables(pid, text, own_methods=own)
        per_paper[pid] = {"records": len(recs), "residue": len(res)}
        all_new += recs
        all_res += res
    kept, dupes = dedup_against_kb(all_new, checked)
    out = {"records": kept, "residue": all_res, "per_paper": per_paper,
           "stats": {"parsed": len(all_new), "dedup_dropped": dupes,
                     "kept": len(kept), "residue_tables": len(all_res)}}
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("F24_TABLE_CHANNEL " + json.dumps(out["stats"], ensure_ascii=False))
    for pid, s in sorted(per_paper.items()):
        print(f"  {pid}: {s['records']} records, {s['residue']} residue")


if __name__ == "__main__":
    main()
