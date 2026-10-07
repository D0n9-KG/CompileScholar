# -*- coding: utf-8 -*-
"""Per-paper reading for the extract passes (INTEGRATED-SYSTEM-1005 v2.4 items 4–5, v2.5 item 1, §7.3).

full_text(D, pid) — the T2 reading, tier priority sciverse -> careful (local MinerU) -> fast (GROBID):
  sentences  every prose sentence with its `date` (v2.5: the earliest version it appears in), sid, unit and
             in_delta flag; the latest version's own content (the stored delta's new + revised sentences) is
             appended whenever the chosen body does not already carry it (an sv text judged to hold the latest
             version already contains it; a bigram check keeps the append from duplicating it when unsure)
  units      the chosen tier's units, normalised to {uid, kind, section, text} (+ html for tables)

t1_input(D, reg, pid) — title + numbered abstract sentences: the v1 GROBID abstract when the paper was parsed
(v2.5: the v1 abstract comes from the v1 parse; 38/72 two-version papers changed their abstract), else the
registry record's abstract as a marked fallback (registry abstracts are the latest version's — SC's official
口径 treats title/abstract as visible at v1, so it carries the v1 date).

chunks(units, sentences) — LLM-ready blocks <= CHUNK_CHARS: prose and caption sentences numbered per paper,
formulas and section headings as unnumbered context, table grids excluded (the results pass reads those),
acknowledgement / author / funding / availability blocks dropped (references are cut upstream).
"""
from __future__ import annotations

import re

from ..documents.sciverse import bigrams as _bigrams, split_sentences

CHUNK_CHARS = 8000
EXCLUDED = re.compile(r"(?i)\b(acknowledg\w*|author(?:s)?(?:\s*contributions?|\s*declaration)?|affiliations?|"
                      r"funding|ethics(?:\s*statement)?|disclosure\w*|conflicts?\s*of\s*interest|"
                      r"data\s*availability|code\s*availability|availability\s*of\s*data)\b")


def _norm_unit(u: dict) -> dict:
    out = {"uid": u.get("uid"), "kind": u.get("kind"), "section": u.get("section") or "", "text": u.get("text") or ""}
    if u.get("html"):
        out["html"] = u["html"]
    return out


def _base_latest(D, pid):
    """(base doc, latest doc or None) — base = v1 when held, else the version of record, else the newest."""
    vs = D.versions(pid)
    if not vs:
        return None, None
    base_v = 1 if 1 in vs else (0 if 0 in vs else max(vs))
    base = D.get(f"{pid}@v{base_v}")
    latest_v = max(vs)
    latest = D.get(f"{pid}@v{latest_v}") if latest_v != base_v else None
    return base, latest


def _delta_sentences(D, pid, latest_date):
    d = D.delta(pid)
    if not d:
        return []
    out = []
    for s in list(d.get("sentences") or []) + list(d.get("revised") or []):
        out.append({"sid": s["sid"], "unit": s.get("unit"), "text": s["text"], "date": latest_date, "in_delta": 1})
    return out


def _careful_sentences(doc):
    """The careful tier has units, not sentences: split its prose/caption paragraphs (the sv pass's conservative
    splitter). Every sentence carries the careful doc's own text_date — no cross-version contest is run here
    (the careful tier is read at the version it parsed)."""
    out, n = [], 0
    for u in doc["careful"]["units"]:
        if u["kind"] not in ("para", "caption"):
            continue
        for t in split_sentences(re.sub(r"\s+", " ", u["text"])):
            if len(t) < 20:
                continue
            n += 1
            out.append({"sid": f"{doc['key']}#cs{n}", "unit": u["uid"], "text": t,
                        "date": doc["text_date"], "in_delta": 0})
    return out


def full_text(D, pid: str) -> dict | None:
    """The T2 reading of one paper, or None when the library holds no full text for it."""
    base, latest = _base_latest(D, pid)
    sv = D.sv(pid)
    delta_extra_needed = False
    if sv and sv.get("sentences"):
        source = "sciverse"
        units = [_norm_unit(u) for u in sv["units"]]
        base_date = base["text_date"] if base else None
        sentences = [dict(s, in_delta=int(bool(base_date) and s["date"] != base_date)) for s in sv["sentences"]]
        delta_extra_needed = sv.get("held") != "latest"
    elif base and base.get("careful"):          # v2.4: T2 body priority sciverse -> mineru -> grobid
        source = "mineru"
        units = [_norm_unit(u) for u in base["careful"]["units"]]
        sentences = _careful_sentences(base)
        delta_extra_needed = True
    elif base and base.get("fast"):
        source = "grobid"
        units = [_norm_unit(u) for u in base["fast"]["units"]]
        sentences = [{"sid": s["sid"], "unit": s.get("unit"), "text": s["text"], "date": base["text_date"],
                      "in_delta": 0} for s in base["fast"]["sentences"]]
        delta_extra_needed = True
    elif latest and latest.get("careful"):
        source = "mineru"
        units = [_norm_unit(u) for u in latest["careful"]["units"]]
        sentences = _careful_sentences(latest)
        delta_extra_needed = False            # the latest careful text already is the latest version
    else:
        return None
    if delta_extra_needed and latest is not None:
        extra = _delta_sentences(D, pid, latest["text_date"])
        if source == "sciverse":
            have = _bigrams(" ".join(s["text"] for s in sentences))
            extra = [s for s in extra
                     if len(_bigrams(s["text"])) < 6 or len(_bigrams(s["text"]) & have) / len(_bigrams(s["text"])) < 0.8]
        sentences += extra
    if not sentences:
        return None
    return {"paper_id": pid, "source": source, "units": units, "sentences": sentences}


def t1_input(D, reg, pid: str) -> dict | None:
    """Title + numbered abstract sentences for T1 (v2.5: the v1 abstract from the v1 GROBID parse; the registry
    record's abstract — the latest version's — as a marked fallback)."""
    r = reg.execute("SELECT title FROM records WHERE paper_id=? ORDER BY source = 'arxiv' DESC LIMIT 1",
                    (pid,)).fetchone()
    title = (r[0] if r else "") or ""
    base, _ = _base_latest(D, pid)
    date = base["text_date"] if base else None
    abstract, source = "", "none"
    if base and base.get("fast") and (base["fast"].get("abstract") or "").strip():
        abstract, source = base["fast"]["abstract"].strip(), "grobid_v1"
    else:
        r = reg.execute("SELECT abstract FROM records WHERE paper_id=? AND abstract IS NOT NULL AND abstract != '' "
                        "ORDER BY source = 'arxiv' DESC LIMIT 1", (pid,)).fetchone()
        if r:
            abstract, source = r[0].strip(), "registry"
    if date is None:                          # no materialised doc: the registry's first public date
        r = reg.execute("SELECT first_hi FROM papers WHERE paper_id=?", (pid,)).fetchone()
        date = (r[0] if r else None)
    if not title or not abstract or not date:
        return None
    sents = split_sentences(re.sub(r"\s+", " ", abstract))
    if not sents:
        return None
    return {"paper_id": pid, "title": title, "date": date, "source": source,
            "sentences": [{"n": i + 1, "text": s} for i, s in enumerate(sents)]}


def chunks(units: list[dict], sentences: list[dict], max_chars: int = CHUNK_CHARS) -> list[dict]:
    """Pack the reading into LLM chunks. Numbering [n] runs across the whole paper; each chunk carries the
    n -> sid map for its own sentences. Table grids are left out (the results pass reads them); their captions
    are in (the adapter emits them as caption units / caption text)."""
    by_unit: dict[str, list] = {}
    for s in sentences:
        by_unit.setdefault(s["unit"], []).append(s)
    numbered = {s["sid"]: s for s in sentences}
    blocks = []                                # (rendered text, [sids])
    excluded_uids = set()
    last_section = None
    n = 0
    for u in units:
        kind, text = u["kind"], u["text"]
        if kind == "section":
            if not EXCLUDED.search(text):
                last_section = text
                blocks.append((f"## {text}", []))
            continue
        if (u["section"] and EXCLUDED.search(u["section"].split(" > ")[-1])) or kind in ("table", "footnote"):
            excluded_uids.add(u["uid"])        # their sentences must not resurface through the orphan block
            continue
        if kind == "formula":
            blocks.append((text, []))
            continue
        ss = by_unit.get(u["uid"]) or []
        if not ss:
            continue
        lines = []
        for s in ss:
            n += 1
            s["n"] = n
            lines.append(f"[{n}] {s['text']}")
        blocks.append(("\n".join(lines), [s["sid"] for s in ss]))
    # sentences whose unit is not in the unit list (a delta sentence from another tier): append as one block;
    # sentences of excluded units stay excluded
    placed = {sid for _, sids in blocks for sid in sids}
    orphans = [s for s in sentences if s["sid"] not in placed and s.get("unit") not in excluded_uids]
    if orphans:
        lines, sids = [], []
        for s in orphans:
            n += 1
            s["n"] = n
            lines.append(f"[{n}] {s['text']}")
            sids.append(s["sid"])
        blocks.append(("[later-version content]\n" + "\n".join(lines), sids))
    out, cur, cur_sids, cur_len = [], [], [], 0
    for text, sids in blocks:
        if cur and cur_len + len(text) > max_chars:
            out.append(_chunk(cur, cur_sids, numbered))
            cur, cur_sids, cur_len = [], [], 0
        cur.append(text)
        cur_sids += sids
        cur_len += len(text) + 1
    if cur:
        out.append(_chunk(cur, cur_sids, numbered))
    return out


def _chunk(parts, sids, numbered) -> dict:
    return {"text": "\n\n".join(parts), "sids": list(sids),
            "sid_by_n": {numbered[s]["n"]: s for s in sids if s in numbered}}


def tables_of(D, pid: str, full: dict) -> list[dict]:
    """The paper's table grids input for the results pass: (html|pipes, caption, context sentences) from the
    chosen tier — sv table units carry their HTML in `text`, careful units in `html`. Papers whose only text is
    the GROBID fast tier have no tables here (measured: GROBID rebuilds 60/218 table structures, MinerU 164/218 —
    the results pass reads MinerU-family grids only)."""
    out = []
    units = full["units"]
    sent_by_unit = {}
    for s in full["sentences"]:
        sent_by_unit.setdefault(s["unit"], []).append(s)
    for i, u in enumerate(units):
        if u["kind"] != "table":
            continue
        html = u.get("html") or (u["text"] if u["text"].lstrip().startswith("<table") else "")
        if not html:
            continue
        caption, cap_j = "", None
        for j in (i + 1, i - 1):               # the caption unit usually follows (sv) or precedes the grid
            if 0 <= j < len(units) and units[j]["kind"] == "caption" and re.match(r"(?i)\s*table\s*\d", units[j]["text"]):
                caption, cap_j = units[j]["text"], j
                break
        if not caption and u["text"] != html:   # careful tier: the table unit's text IS the caption
            m = re.match(r"(?i)\s*(table\s*\d+[.:].*)", u["text"])
            caption = m.group(1) if m else ""
        ctx_s = [s for j in (i - 1, i + 1) if 0 <= j < len(units) and units[j]["kind"] == "para" and j != cap_j
                 for s in sent_by_unit.get(units[j]["uid"], [])][:4]
        # the table's date: its caption sentence's date when dated (sv/careful tiers date sentences), else the
        # neighbouring prose; the caller's default (latest known) backs both up — later, never earlier
        cap_s = sent_by_unit.get(units[cap_j]["uid"], []) if cap_j is not None else []
        date = next((s["date"] for s in cap_s if s.get("date")), None) \
            or next((s["date"] for s in ctx_s if s.get("date")), None)
        out.append({"uid": u["uid"], "html": html, "caption": caption, "date": date,
                    "context": [s["text"] for s in ctx_s]})
    return out
