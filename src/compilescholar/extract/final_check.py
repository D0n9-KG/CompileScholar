# -*- coding: utf-8 -*-
"""The unified final check (INTEGRATED-SYSTEM-1005 §7.3) — ONE implementation, used by the offline build and by
the runtime deep_read alike:

  1. quote location: view(quote) must be a substring of view(source), through documents.matchview (NFKC,
     whitespace out, $ out, LaTeX folded, HTML tags out, citation markers out, glyph map). The view only matches;
     stored quotes stay verbatim. loc.char_start / char_end are filled here — the quote's span inside its
     anchor sentence's ORIGINAL text.
  2. numbers verbatim: every numeric token of `text` (and meta.config.value) must occur in view(quote).
  3. enums and required fields: schema.Statement.validate().
  4. failures get ONE LLM repair (the statement, its violations and its anchor sentence); still failing after
     the repair -> discarded with its violation code recorded.

Fuzzy matching is used for REPORTING only ("how far off" — the best sentence and its ratio land in the stats);
it never passes a statement."""
from __future__ import annotations

import json
import re
from collections import Counter

from rapidfuzz import fuzz, process

from ..documents import matchview as MV
from ..llm.client import call_local
from ..llm.jsonparse import parse_json_response
from . import prompts as PR
from .schema import EPISTEMIC, FACETS, RELATIONS, Statement

QUOTE_NOT_IN_SOURCE = "QUOTE_NOT_IN_SOURCE"
NUMBER_REWRITTEN = "NUMBER_REWRITTEN"
SCHEMA = "SCHEMA"
_NUM = re.compile(r"\d+(?:[.,]\d+)*(?:\s*[×x]\s*10\s*[-−–]?\s*\d+)?%?")


def violations(stmt: Statement, source_view: str) -> list[str]:
    errs = stmt.validate()
    if errs:
        return [f"{SCHEMA}: {'; '.join(errs)[:180]}"]
    vq = MV.view(stmt.quote)
    if not vq or vq not in source_view:
        return [QUOTE_NOT_IN_SOURCE]
    # numbers verbatim (§7.3 rule 2, 原文里逐字出现): for sentence-anchored statements the frame is the quote
    # itself; for results statements it is the whole table source — their labels come from header cells ("T5-B")
    # which are source text but not part of the row quote, and the value-to-object binding is structural there
    frame = source_view if stmt.pass_name == "results" else vq
    bad = [n for n in set(_NUM.findall(stmt.text)) if MV.view(n) not in frame]
    cfg = (stmt.meta or {}).get("config")
    if isinstance(cfg, dict) and cfg.get("value") and MV.view(str(cfg["value"])) not in frame:
        bad.append(str(cfg["value"]))
    if bad:
        return [f"{NUMBER_REWRITTEN}: {', '.join(sorted(bad)[:5])}"]
    return []


def _fill_loc(stmt: Statement, sent_text: str | None) -> None:
    if sent_text and stmt.loc.get("char_start") is None:
        span = MV.locate(stmt.quote, sent_text)
        if span:
            stmt.loc["char_start"], stmt.loc["char_end"] = span


def _repair(bad: list[tuple[Statement, list[str]]], sent_texts: dict, chat, item: str) -> list[Statement]:
    """One LLM repair attempt for the whole failing set of one item; returns repaired statement copies
    (same positions as `bad`). Unparseable answer -> the originals (they will fail the re-check and discard)."""
    cases = []
    for i, (s, v) in enumerate(bad, 1):
        cases.append(json.dumps({"i": i, "violations": v, "anchor_sentence": sent_texts.get(s.loc.get("sent_id"), ""),
                                 "statement": {"text": s.text, "quote": s.quote, "facet": s.facet, "role": s.role,
                                               "epistemic": s.epistemic, "condition": s.condition}},
                                ensure_ascii=False))
    raw = chat(PR.REPAIR.format(cases="\n".join(cases)), model=PR.MODEL, max_tokens=4000, temperature=0.0,
               enable_thinking=False, item=f"{item}#repair")
    obj = parse_json_response(raw or "")
    fixed = {o.get("i"): o for o in (obj or {}).get("fixed") or [] if isinstance(o, dict)} \
        if isinstance(obj, dict) else {}
    out = []
    for i, (s, _) in enumerate(bad, 1):
        o = fixed.get(i)
        if not o or o.get("drop"):
            out.append(s)                          # unrepairable: the re-check will discard it
            continue
        s2 = Statement(**{**{f: getattr(s, f) for f in
                            ("speaker", "date", "kind", "about", "target", "group", "function", "meta",
                             "schema_version", "pass_name", "item", "run_id", "model", "prompt_sha")},
                          "text": str(o.get("text") or s.text)[:400],
                          "quote": str(o.get("quote") or s.quote),
                          "facet": o.get("facet") if o.get("facet") in FACETS else s.facet,
                          "role": o.get("role") if o.get("role") in RELATIONS else s.role,
                          "epistemic": o.get("epistemic") if o.get("epistemic") in EPISTEMIC else s.epistemic,
                          "condition": str(o.get("condition") if o.get("condition") is not None else s.condition)[:300],
                          "loc": dict(s.loc)})
        s2.loc.pop("char_start", None)
        s2.loc.pop("char_end", None)
        out.append(s2)
    return out


def run(stmts: list[Statement], source_text: str, sent_texts: dict, chat=None, item: str = ""):
    """-> (kept, discarded, stats). sent_texts maps loc.sent_id -> the anchor sentence's verbatim text (the
    frame for char offsets); source_text is the whole text the pass read (the frame for the substring rule)."""
    chat = chat or call_local
    sv = MV.view(source_text)
    kept, bad = [], []
    for s in stmts:
        v = violations(s, sv)
        if v:
            bad.append((s, v))
        else:
            _fill_loc(s, sent_texts.get(s.loc.get("sent_id")))
            kept.append(s)
    st: Counter = Counter({"checked": len(stmts), "clean": len(kept)})
    discarded = []
    if bad:
        for s2, (s, v) in zip(_repair(bad, sent_texts, chat, item), bad):
            v2 = violations(s2, sv)
            if v2:
                discarded.append((s2, v2))
                st["discarded"] += 1
                st["v_" + v2[0].split(":")[0]] += 1
            else:
                _fill_loc(s2, sent_texts.get(s2.loc.get("sent_id")))
                if s2 is not s:
                    s2.meta = {**s2.meta, "repaired": True}
                    st["repaired"] += 1
                kept.append(s2)
        # fuzzy report only: where the discarded quotes almost matched (never a pass)
        sents = list(sent_texts.values())
        views = [MV.view(t) for t in sents]
        if discarded and views:
            samples = []
            for s2, v in discarded[:5]:
                if v[0] == QUOTE_NOT_IN_SOURCE:
                    m = process.extractOne(MV.view(s2.quote), views, scorer=fuzz.ratio)
                    if m:
                        samples.append({"ratio": round(m[1], 1), "nearest": sents[m[2]][:120]})
            if samples:
                st["fuzzy_samples"] = samples
    return kept, discarded, dict(st)
