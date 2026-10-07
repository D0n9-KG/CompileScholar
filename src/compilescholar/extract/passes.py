# -*- coding: utf-8 -*-
"""The four extraction passes as per-paper functions (INTEGRATED-SYSTEM-1005 §7.3; v2.4 item 1: the offline build
and the runtime deep_read share ONE implementation — the runtime never writes back).

  t1(inp)                    title + numbered v1-abstract sentences -> self statements + named artefacts
                             (relation proposes|uses — the pre-C proposes/uses confusion is a prompt field now)
  t2(pid, title, full, ...)  full-text chunks (<= 8k chars) -> self statements over ALL facets, each anchored to
                             the sentence it came from (quote = that sentence, date = that sentence's version
                             date); returns own_methods for the results pass; sentences describing other works
                             are skipped here (the other pass owns them)
  results(pid, tables, ...)  repaired table grids -> the LLM proposes AXIS ROLES only, a structural gate decides
                             whether the table is written at all, and every number comes from a cell (never from
                             the model); quote = the verbatim row text
  other_batch(citing, items) citation sentences of one citing paper -> one statement per (sentence, cited key),
                             role/function/facet per key, meta.self_cite from the citations stage; the old 0.6
                             lexical-support threshold is downgraded to a meta.supported signal (nothing dropped)
  deep_read(D, reg, pid)     the runtime composition: t1 + t2 + results for one paper, returns statements + stats

Deterministic discipline everywhere: closed vocabularies (an item with a value outside them is dropped, never
remapped), quotes are the numbered sentences themselves, surface names must occur literally in their sentence.
The unified final check (§7.3) runs over the assembled statements in the stage build, not inside each pass.
A pass returns None for its statements when the LLM gave no parseable answer — a failure of the item (retried by
the work discipline), never "the paper says nothing"."""
from __future__ import annotations

import re
from collections import Counter

from ..compile.skeleton import proposes as P
from ..documents import tables as T
from ..llm.client import call_local
from ..llm.jsonparse import parse_json_response
from . import prompts as PR
from . import reading
from .schema import (EPISTEMIC, FACETS, FUNCTIONS, LINEAGE, OTHER_RELATIONS, RELATIONS, Statement)

MODEL = PR.MODEL


def _n(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _literal(val, sentence: str) -> str | None:
    """A surface-name field survives only if it occurs literally (case-insensitive) in its sentence."""
    if not isinstance(val, str):
        return None
    v = _n(val).strip(" .,;:")
    return v if 2 <= len(v) <= 80 and v.lower() in " ".join(sentence.split()).lower() else None


def _item_common(it: dict, sent: dict, pid: str, date: str, st: Counter):
    """The vocabulary/anchor gate every (item, sentence) pair passes; -> kwargs for Statement or None."""
    facet, role = it.get("facet"), it.get("role")
    epi = it.get("epistemic") or "stated"
    if facet not in FACETS or role not in RELATIONS or epi not in EPISTEMIC:
        st["bad_vocab"] += 1
        return None
    text = _n(it.get("text"))
    if not text:
        st["empty_text"] += 1
        return None
    st["items"] += 1
    return dict(speaker=pid, date=date, kind="self", about=pid, role=role, facet=facet, text=text[:400],
                quote=sent["text"], epistemic=epi, condition=_n(it.get("condition"))[:300],
                loc={"unit_id": sent.get("unit") or f"{pid}#abstract", "sent_id": sent.get("sid") or ""})


# ---------------------------------------------------------------- T1
def t1(inp: dict, chat=None, model: str = MODEL) -> tuple[list[Statement] | None, list[str], dict]:
    """-> (statements | None, own names proposed, stats). inp = reading.t1_input(...) output."""
    chat = chat or call_local
    pid, date = inp["paper_id"], inp["date"]
    by_n = {s["n"]: s for s in inp["sentences"]}
    block = "\n".join(f"[{s['n']}] {s['text']}" for s in inp["sentences"])
    raw = chat(PR.T1.format(title=inp["title"][:300], sentences=block, facets=PR._FACETS, roles=PR._ROLES_SELF,
                            epistemic=PR._EPISTEMIC),
               model=model, max_tokens=3000, temperature=0.0, enable_thinking=False, item=pid)
    obj = parse_json_response(raw or "")
    if not isinstance(obj, dict):
        return None, [], {"parse_failed": 1}
    st: Counter = Counter()
    out = []
    source_text = " ".join(s["text"] for s in inp["sentences"])
    for it in obj.get("items") or []:
        if not isinstance(it, dict):
            continue
        n = it.get("n")
        s = by_n.get(n) if isinstance(n, int) else None
        if s is None:
            st["bad_n"] += 1
            continue
        s = {**s, "sid": f"{pid}@abs{n}"}
        kw = _item_common(it, s, pid, date, st)
        if kw is None:
            continue
        out.append(Statement(**kw, meta={"source": inp["source"]}, model=model, prompt_sha=PR.T1_SHA,
                             pass_name="t1", item=pid))
    names = []
    for p_ in obj.get("names") or []:
        if not isinstance(p_, dict):
            continue
        n = p_.get("n")
        s = by_n.get(n) if isinstance(n, int) else None
        rel = p_.get("relation")
        if s is None or rel not in ("proposes", "uses"):
            st["bad_name"] += 1
            continue
        name = _literal(p_.get("name"), s["text"])
        if name is None:
            st["name_not_literal"] += 1
            continue
        if rel == "proposes":
            v = P.validate({"name": p_.get("name"), "aliases": p_.get("aliases"), "evidence": s["text"]},
                           source_text)
            if not v:
                st["proposal_rejected"] += 1
                continue
            name, aliases, generic = v["method"], v["aliases"], v["generic"]
            text = f"proposes {name}"
        else:
            aliases, generic = [a for a in (_n(x) for x in p_.get("aliases") or []) if a], False
            text = f"uses {name}"
        st["names"] += 1
        if rel == "proposes":
            names.append(name)                      # own_methods: only what the paper proposes, never what it uses
        out.append(Statement(speaker=pid, date=date, kind="self", about=pid,
                             role="proposes" if rel == "proposes" else "uses", facet="contribution", text=text,
                             quote=s["text"], epistemic="stated",
                             loc={"unit_id": f"{pid}#abstract", "sent_id": f"{pid}@abs{n}"},
                             meta={"name": name, "aliases": aliases, "artefact": p_.get("artefact"),
                                   "generic": generic, "source": inp["source"]},
                             model=model, prompt_sha=PR.T1_SHA, pass_name="t1", item=pid))
    return out, names, dict(st)


# ---------------------------------------------------------------- T2
def t2(pid: str, title: str, full: dict, seed_own=(), chat=None,
       model: str = MODEL) -> tuple[list[Statement] | None, list[str], dict]:
    """-> (statements | None, own_methods, stats). full = reading.full_text(...) output; sentences carry their
    version dates. Every chunk must parse — a chunk without a parseable answer fails the whole item (dfc retries
    the paper, never half of it)."""
    chat = chat or call_local
    ch = reading.chunks(full["units"], full["sentences"])
    if not ch:
        return [], sorted(set(seed_own)), {"chunks": 0}
    sent_by_n = {s["n"]: s for s in full["sentences"] if s.get("n")}
    body_lower = " ".join(s["text"] for s in full["sentences"]).lower()
    out, own, st = [], set(seed_own), Counter()
    for i, c in enumerate(ch, 1):
        raw = chat(PR.T2.format(title=title[:300], chunk_i=i, chunk_n=len(ch), text=c["text"],
                                facets=PR._FACETS, roles=PR._ROLES_SELF, epistemic=PR._EPISTEMIC),
                   model=model, max_tokens=6000, temperature=0.0, enable_thinking=False, item=f"{pid}#c{i}")
        obj = parse_json_response(raw or "")
        if not isinstance(obj, dict):
            return None, [], {"parse_failed": 1, "chunks": len(ch), "failed_chunk": i}
        st["chunks"] += 1
        for it in obj.get("items") or []:
            if not isinstance(it, dict):
                continue
            n = it.get("n")
            s = sent_by_n.get(n) if isinstance(n, int) else None
            if s is None or s["sid"] not in c["sids"]:
                st["bad_n"] += 1
                continue
            kw = _item_common(it, s, pid, s["date"], st)
            if kw is None:
                continue
            mentions = [{"name": nm, "relation": m.get("relation")}
                        for m in it.get("mentions") or [] if isinstance(m, dict)
                        and m.get("relation") in (*LINEAGE, "uses", "compares")
                        for nm in [_literal(m.get("name"), s["text"])] if nm]
            kw["meta"] = {"mentions": mentions, "in_delta": s["in_delta"], "source": full["source"]}
            out.append(Statement(**kw, model=model, prompt_sha=PR.T2_SHA, pass_name="t2", item=pid))
        for cfg in obj.get("config") or []:
            if not isinstance(cfg, dict):
                continue
            n = cfg.get("n")
            s = sent_by_n.get(n) if isinstance(n, int) else None
            item_name, value = _n(cfg.get("item")), _n(cfg.get("value"))
            if s is None or not item_name or not value:
                st["bad_config"] += 1
                continue
            if value.lower() not in s["text"].lower():       # values are quoted from the sentence, never invented
                st["config_value_unsupported"] += 1
                continue
            st["config"] += 1
            out.append(Statement(speaker=pid, date=s["date"], kind="self", about=pid, role="describes",
                                 facet="config", text=f"{item_name} = {value}"[:400], quote=s["text"],
                                 epistemic="stated", condition=_n(cfg.get("applies_to"))[:300],
                                 loc={"unit_id": s.get("unit"), "sent_id": s["sid"]},
                                 meta={"config": {"item": item_name, "value": value,
                                                  "unit": _n(cfg.get("unit")), "applies_to": _n(cfg.get("applies_to"))},
                                       "in_delta": s["in_delta"], "source": full["source"]},
                                 model=model, prompt_sha=PR.T2_SHA, pass_name="t2", item=pid))
        for m in obj.get("own_methods") or []:
            m = _n(m)
            if 2 <= len(m) <= 80 and m.lower() in body_lower:   # a name the text does not contain is not its own
                own.add(m)
    return out, sorted(own), dict(st)


# ---------------------------------------------------------------- results pass
def _repair_grid(html: str):
    """HTML -> (grid, row quotes): spans expanded for the grid; the quotes come from the PRE-expansion rows'
    cell texts (a rowspan continuation would duplicate text the source has once), whitespace-collapsed — the
    final check's match view normalises the source the same way."""
    p = T._TableHTML()
    try:
        p.feed(html)
    except Exception:
        return None, []
    if not p.tables:
        return None, []
    rows = p.tables[0]
    grid, _ = T._expand_grid(rows)
    grid = [[("" if c is None else str(c)) for c in row] for row in grid]
    quotes = [" ".join(_n(txt) for txt, _, _ in row if _n(txt)) for row in rows]
    return (grid if len(grid) >= 2 else None), quotes


def results(pid: str, tables: list[dict], own_methods=(), default_date: str | None = None, chat=None,
            model: str = MODEL) -> tuple[list[Statement], dict]:
    """Deterministic grid repair + LLM axis roles + a structural gate; every value is read from a cell. Tables
    whose axes cannot be gated are skipped and counted — never guessed."""
    chat = chat or call_local
    out, st = [], Counter()
    own_norm = {re.sub(r"\s+", " ", str(m).strip().lower()) for m in own_methods if m}
    for tb in tables:
        grid, quotes = _repair_grid(tb["html"])
        if not grid:
            st["grid_unparsed"] += 1
            continue
        nrows, ncols = len(grid), max(len(r) for r in grid)
        render = "\n".join(f"r{ri}: " + " | ".join(f"[{ci}] {c[:60]}" for ci, c in enumerate(row))
                           for ri, row in enumerate(grid[:40]))
        raw = chat(PR.RESULTS_AXES.format(caption=(tb.get("caption") or "(none)")[:300],
                                          context="\n".join(f"- {c[:200]}" for c in tb.get("context") or []) or "(none)",
                                          n_rows=nrows, n_cols=ncols, grid=render[:6000],
                                          own_methods=", ".join(sorted(own_norm)[:10]) or "(unknown)"),
                   model=model, max_tokens=1500, temperature=0.0, enable_thinking=False, item=f"{pid}#{tb['uid']}")
        ax = parse_json_response(raw or "")
        if not isinstance(ax, dict):
            st["axes_parse_failed"] += 1
            continue
        if not ax.get("usable"):
            st["unusable"] += 1
            continue
        oaxis, oidx = ax.get("object_axis"), ax.get("object_index")
        if oaxis != "row" or not isinstance(oidx, int):
            # column-object (transposed) tables are skipped, not half-supported: their row quotes and skip/header
            # indices live in the other orientation, and a wrong number is worse than a missing one
            st["gate_column_object"] += 1
            continue
        measures = [m for m in ax.get("measures") or []
                    if isinstance(m, dict) and isinstance(m.get("index"), int) and 0 <= m["index"] < ncols
                    and _n(m.get("metric"))]
        if not measures:
            st["gate_no_measures"] += 1
            continue
        if not (0 <= oidx < ncols):
            st["gate_object_index"] += 1
            continue
        conds = [c for c in ax.get("conditions") or []
                 if isinstance(c, dict) and isinstance(c.get("index"), int) and 0 <= c["index"] < ncols
                 and c["index"] != oidx and _n(c.get("label"))]
        headers = {int(h) for h in ax.get("header_rows") or [] if str(h).lstrip("-").isdigit()} or {0}
        skip = {int(x) for x in ax.get("skip_rows") or [] if str(x).lstrip("-").isdigit()}
        date = tb.get("date") or default_date
        if not date:
            st["no_date"] += 1
            continue
        for ri, row in enumerate(grid):
            if ri in headers or ri in skip or ri >= len(quotes) or not quotes[ri]:
                continue
            obj = _n(row[oidx] if oidx < len(row) else "")
            if not obj:
                continue
            cond = "; ".join(f"{_n(c['label'])}={_n(row[c['index']])}" for c in conds
                             if c["index"] < len(row) and _n(row[c["index"]]))
            mine = any(o and (o in obj.lower() or obj.lower() in o) for o in own_norm)
            for m in measures:
                cell = _n(row[m["index"]]) if m["index"] < len(row) else ""
                if not cell or len(cell) > 40 or not re.search(r"\d", cell):
                    continue                        # no number in the cell: nothing to state, never infer one
                st["cells"] += 1
                out.append(Statement(
                    speaker=pid, date=date, kind="self", about=pid, role="describes", facet="result",
                    text=f"{obj}{(' — ' + cond) if cond else ''}: {_n(m['metric'])}: {cell}"[:400],
                    quote=quotes[ri], epistemic="demonstrated" if mine else "cited", condition=cond[:300],
                    loc={"unit_id": tb["uid"], "sent_id": f"{pid}@{tb['uid']}:r{ri}"},
                    meta={"object": obj, "metric": _n(m["metric"]), "value": cell, "unit": _n(m.get("unit")),
                          "direction": m.get("direction") if m.get("direction") in ("higher", "lower", "neutral")
                          else None, "own": mine, "table": tb["uid"], "row": ri},
                    model=model, prompt_sha=PR.RESULTS_SHA, pass_name="results", item=pid))
        st["tables_written"] += 1
    return out, dict(st)


# ---------------------------------------------------------------- other pass
_SUPPORT_STOP = set("a an the of in on for to and or with by from as is are was were be been this that these those "
                  "it its their they we our which such via using use used based into than also can may".split())


def _words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in _SUPPORT_STOP and len(w) > 1]


def supported(field: str | None, sentence: str) -> bool:
    """Lexical support as a SIGNAL (meta.supported), not a drop rule — the pre-C 0.6 threshold dropped real
    paraphrases; the final check owns what is written."""
    w = _words(field or "")
    if not w:
        return False
    sent = set(_words(sentence))
    return sum(x in sent for x in w) / len(w) >= 0.6


COMPARE_WORDS = re.compile(r"(?i)\b(outperform\w*|better|worse|superior|inferior|surpass\w*|beat\w*|exceed\w*|"
                           r"improv\w*|higher|lower|comparable|competitive|on par|gains?|drops?)\b")
OUTCOMES = ("citing_better", "cited_better", "mixed")
_MENTION_REL = (*LINEAGE,)


def other_batch(citing: str, items: list[dict], chat=None,
                model: str = MODEL) -> tuple[list[Statement] | None, dict]:
    """items: [{"s": int, "sentence", "date", "citing_key", "sid", "in_delta",
                "cites": [{"key", "cited", "title", "raw", "self_cite"}]}]
    -> one statement per (sentence, cited key) | None when the answer did not parse."""
    chat = chat or call_local
    lines = []
    for it in items:
        lines.append(f"[{it['s']}] \"{it['sentence'][:900]}\"")
        for c in it["cites"]:
            label = (c.get("title") or (c.get("raw") or "")[:160] or "(untitled)")[:200]
            lines.append(f"    cites \"{c['key']}\" = \"{label}\"")
    total = sum(len(it["cites"]) for it in items)
    raw = chat(PR.OTHER.format(sentences="\n".join(lines), epistemic=PR._EPISTEMIC), model=model,
               max_tokens=min(6000, 230 * total + 300), temperature=0.0, enable_thinking=False, item=citing)
    obj = parse_json_response(raw or "")
    pairs = obj.get("pairs") if isinstance(obj, dict) else None
    if not isinstance(pairs, list):
        return None, {"pairs": total, "failed": 1}
    got = {}
    for o in pairs:
        if isinstance(o, dict) and str(o.get("s", "")).lstrip("-").isdigit() and isinstance(o.get("key"), str):
            got.setdefault((int(o["s"]), o["key"]), o)
    out, st = [], Counter({"pairs": total})
    for it in items:
        by_key = {c["key"]: c for c in it["cites"]}
        group = tuple(c["cited"] for c in it["cites"])
        for c in it["cites"]:
            o = got.get((it["s"], c["key"]))
            if o is None:
                st["missing"] += 1
                continue
            rel, fn = o.get("role"), o.get("function")
            if rel not in OTHER_RELATIONS or fn not in FUNCTIONS:
                st["bad_vocab"] += 1
                continue
            epi = o.get("epistemic") if o.get("epistemic") in EPISTEMIC else "stated"
            about = _n(o.get("about")) or None
            cat = _n(o.get("category")) or None
            lim = _n(o.get("limitation")) or None
            facet = o.get("facet") if o.get("facet") in FACETS else \
                ("limitation" if lim else "categorization" if cat else "contribution")
            name = _literal(o.get("name"), it["sentence"])
            outcome = o.get("outcome") if o.get("outcome") in OUTCOMES else None
            if outcome and not COMPARE_WORDS.search(it["sentence"]):
                outcome = None                      # an outcome needs comparative wording in the sentence itself
            bo = _literal(o.get("builds_on"), it["sentence"])
            bo_rel = o.get("builds_on_relation") if bo and o.get("builds_on_relation") in _MENTION_REL else None
            st["parsed"] += 1
            common = dict(speaker=citing, date=it["date"], kind="other", about=c["cited"], role=rel,
                          quote=it["sentence"], epistemic=epi,
                          loc={"unit_id": it.get("citing_key") or citing, "sent_id": it.get("sid") or ""},
                          group=group, function=fn,
                          meta={"category": cat, "limitation": lim, "name": name, "outcome": outcome,
                                "builds_on": bo if bo_rel else None, "builds_on_relation": bo_rel,
                                "supported": supported(about or cat or lim, it["sentence"]),
                                "self_cite": int(c.get("self_cite") or 0), "in_delta": int(it.get("in_delta") or 0)},
                          model=model, prompt_sha=PR.OTHER_SHA, pass_name="other", item=citing)
            if lim and facet != "limitation":       # a stated limitation gets its own statement
                out.append(Statement(**common, facet="limitation", text=lim[:400]))
            out.append(Statement(**common, facet=facet,
                                 text=(about or lim or cat or "(cited without description)")[:400]))
    return out, dict(st)


# ---------------------------------------------------------------- runtime composition
def deep_read(D, reg, pid: str, chat=None) -> dict:
    """The per-paper runtime entry (v2.4: `deep_read(paper_id)` — same code as the offline build; the caller
    decides where the statements go, and during evaluation nothing is written back). Best effort: a failed pass
    is reported in stats, the others still return."""
    chat = chat or call_local
    stats: dict = {}
    out: list[Statement] = []
    inp = reading.t1_input(D, reg, pid)
    title = inp["title"] if inp else ""
    own: list[str] = []
    if inp:
        stmts, names, st = t1(inp, chat=chat)
        stats["t1"] = st
        if stmts is not None:
            out += stmts
            own += names
    full = reading.full_text(D, pid)
    if full:
        stmts, own2, st = t2(pid, title, full, seed_own=own, chat=chat)
        stats["t2"] = st
        if stmts is not None:
            out += stmts
            own = sorted(set(own) | set(own2))
        tbs = reading.tables_of(D, pid, full)
        if tbs:
            default_date = max((s["date"] for s in full["sentences"] if s.get("date")), default=None)
            stmts, st = results(pid, tbs, own_methods=own, default_date=default_date, chat=chat)
            stats["results"] = st
            out += stmts
    else:
        stats["t2"] = {"no_full_text": 1}
    return {"statements": out, "stats": stats, "own_methods": own, "source": full["source"] if full else None}
