# -*- coding: utf-8 -*-
"""L6 tools (phase D④): the only interface consumers (agents via MCP, the answer pipeline, benchmark adapters)
use. INTEGRATED-SYSTEM-1005 §9.2 / §9.4.

The contract:
  - every tool takes as_of (YYYY-MM-DD, inclusive) and never returns anything dated after it — statements and
    passages by their text-version date, papers by first public date, materialised cognition by its own
    visibility rules (snapshot <= T, closed shift windows, fact timelines cut at T);
  - returns separate `quote` (the verbatim source sentence — what an evidence claim cites, NEVER truncated)
    from `display` (truncated presentation text); evidence is always the quote;
  - ids are bare paper_ids ("arxiv:…" | "doi:…" | "title:…" | "stub:…"); display identifiers (DOI, arXiv)
    ride along from the registry's identifier table;
  - field tools (field_map / paper_profile / frontier / open_issues) read the materialised cognition tables;
    bounded tools (card / profile / compare / events) are computed online — target < 200 ms per call;
  - the ARM (none / flat / summary / full) and the ablations (-time / -reception / self_only) are fixed by
    server-side configuration at startup — an agent cannot change them through a tool call;
  - every call is ledgered (returned chars, estimated tokens, latency) for the same-budget audits (§9.4);
  - external tools (search_external / expand_citations) mark their hits `external` and NEVER write back during
    evaluation; an external hit that resolves to a local paper_id merges into the local results. A same-year
    external hit is merged or shown only after its exact date is resolved — an unresolvable one is excluded
    (§2.2: undecidable is invisible).

The pre-C version of this file (arXiv-id keys, "paper:" prefixes, truncated-only returns, online field
aggregation) is in git history (tag pre-integration-20261005 .. 8594109)."""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from functools import wraps

from ..cognition import comparisons as C
from ..cognition import families as F
from ..cognition import facts as FA
from ..cognition import lineage as L
from ..cognition import profiles as P
from ..cognition import shifts as SH
from ..cognition.asof import AsOf
from ..cognition.identity import Identity

__all__ = ["AsOf", "ToolConfig", "configure", "active_tools", "set_index"]
MAX_ITEMS = 8
TXT = 300
ARMS = ("none", "flat", "summary", "full")
ABLATIONS = ("-time", "-reception", "self_only")

_lock = threading.Lock()
_index = [None]
_docs = [None]          # the shared Documents reader (thread-safe ReadConn) + its papers set, memoised


@dataclass(frozen=True)
class ToolConfig:
    """Server-side configuration (fixed at startup; a tool call cannot change it — §9.4)."""
    arm: str = "full"                       # none | flat | summary | full
    ablations: tuple = ()                   # any of ABLATIONS
    budget_log: str | None = None           # jsonl ledger path (runs/<run_id>/tool_budget.jsonl)
    summary_model: str = ""                 # the summary arm's compressor model ("" = the local default)
    external: bool = True                   # external search enabled (needs SCIVERSE_API_TOKEN)
    deep_read_timeout_s: int = 900          # the MCP hard timeout for deep_read


_CFG = ToolConfig()


def configure(cfg: ToolConfig) -> None:
    global _CFG
    if cfg.arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {cfg.arm!r}")
    bad = [a for a in cfg.ablations if a not in ABLATIONS]
    if bad:
        raise ValueError(f"unknown ablations {bad} (allowed: {ABLATIONS})")
    _CFG = ToolConfig(**{**cfg.__dict__, "ablations": tuple(cfg.ablations)})


def config() -> ToolConfig:
    return _CFG


def _abl(name: str) -> bool:
    return name in _CFG.ablations


def _no_reception() -> bool:
    return _abl("-reception") or _abl("self_only")


def _ablated(what: str):
    return {"ablated": what, "note": "this capability is switched off by the server-side configuration"}


# ---------------------------------------------------------------- plumbing
def _idx():
    with _lock:
        if _index[0] is None:
            from ..index.search import Index
            try:
                from ..llm.embedding import embed_local
                emb = embed_local
            except Exception:                    # pragma: no cover — BM25-only fallback
                emb = None
            _index[0] = Index(embed=emb)
        return _index[0]


def set_index(index) -> None:
    """Inject an index (tests, BM25-only runs)."""
    _index[0] = index


def _documents():
    """The shared Documents reader + its papers set, memoised for the process. An MCP session is short-lived
    (§9.2: one process per harness session), so the memo never outlives a rebuild in practice; a long-lived
    consumer must call reset_caches() after a documents rebuild (the stage swaps its file atomically — an open
    reader would keep serving the old one)."""
    with _lock:
        if _docs[0] is None:
            from ..documents.build import Documents
            d = Documents()
            _docs[0] = (d, set(d.papers()))
        return _docs[0]


def reset_caches() -> None:
    """Drop the memoised readers (tests; a consumer that outlives a stage rebuild)."""
    with _lock:
        if _docs[0] is not None:
            _docs[0][0].close()
        _docs[0] = None


def _view(T) -> AsOf:
    return AsOf(T)


def _t(s, n=TXT) -> str:
    """The DISPLAY field: whitespace-collapsed and truncated (a quote is never passed through here)."""
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n - 1] + "…"


def _args_repr(a, k) -> str:
    try:
        return json.dumps({"args": [str(x)[:120] for x in a], "kw": {kk: str(vv)[:120] for kk, vv in k.items()}},
                          ensure_ascii=False)
    except (TypeError, ValueError):
        return "?"


def _ledger(tool: str, args: str, out, t0: float) -> None:
    """The §9.4 budget ledger: what each call RETURNED (chars + estimated tokens), for same-budget audits."""
    if not _CFG.budget_log:
        return
    try:
        s = json.dumps(out, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        s = str(out)
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tool": tool, "args": args, "chars": len(s),
           "tokens_est": max(1, len(s) // 4), "ms": round((time.monotonic() - t0) * 1000),
           "arm": _CFG.arm, "ablations": list(_CFG.ablations)}
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _lock:
        with open(_CFG.budget_log, "a", encoding="utf-8") as f:
            f.write(line)


_SUMMARY_PROMPT = """Compress the JSON result of a literature-search tool for an agent on a tight context budget.
Keep UNCHANGED: every "id", every date, every number, and every "quote" field verbatim (quotes are the evidence).
Shorten: "display" texts, and merge near-duplicate list items (say how many were merged in a "_merged" count).
Return JSON only, with the same top-level shape.

{obj}"""


def _compress(obj):
    """The summary arm: every returned result is compressed on the spot (§9.4). A compressor failure returns
    the uncompressed result marked as such — never a lost call."""
    from ..extract import prompts as PR
    from ..llm.client import call_local
    from ..llm.jsonparse import parse_json_response
    try:
        raw = call_local(_SUMMARY_PROMPT.format(obj=json.dumps(obj, ensure_ascii=False, default=str)[:24000]),
                         model=_CFG.summary_model or PR.MODEL, max_tokens=4000, temperature=0.0,
                         enable_thinking=False, item="tool:summary")
        parsed = parse_json_response(raw or "")
        return parsed if parsed is not None else {"compression_failed": True, "result": obj}
    except Exception as e:                        # noqa: BLE001 — the arm must not kill the call
        return {"compression_failed": str(e)[:200], "result": obj}


def tool(fn):
    """The one wrapper every tool goes through: the arm post-processing and the budget ledger."""
    @wraps(fn)
    def wrapped(*a, **k):
        t0 = time.monotonic()
        out = fn(*a, **k)
        if _CFG.arm == "summary":
            out = _compress(out)
        _ledger(fn.__name__, _args_repr(a, k), out, t0)
        return out
    wrapped.__is_tool__ = True
    return wrapped


def _resolve_local(view: AsOf, title: str | None = None, doi: str | None = None,
                   arxiv: str | None = None) -> str | None:
    """An external hit -> a local paper_id: identifier point lookups first, then an exact title_key match."""
    from ..core import ids as IDS
    for scheme, val in (("doi", (doi or "").lower() or None), ("arxiv", arxiv or None)):
        if val:
            r = view.reg.execute("SELECT paper_id FROM identifiers WHERE scheme=? AND value=?",
                                 (scheme, val)).fetchone()
            if r:
                return r[0]
    if title:
        tk = IDS.title_key(title)
        if tk:
            r = view.reg.execute("SELECT paper_id FROM records WHERE title_key=?", (tk,)).fetchone()
            if r:
                return r[0]
    return None


def _paper_head(view: AsOf, pid: str) -> dict:
    p = view.paper(pid) or {}
    return {"id": pid, "title": p.get("title"), "date": p.get("date"), "ids": p.get("ids") or {}}


def _brief(view: AsOf, pid: str) -> dict:
    pr = P.profile(view, pid)
    p = pr["paper"] or {}
    own = next((s for s in pr["self"] if s["facet"] == "contribution"), None)
    out = {"id": pid, "title": p.get("title"), "date": p.get("date"),
           "self": {"display": _t(own["text"], 220) if own else _t(p.get("abstract"), 220),
                    "quote": own["quote"] if own else None}}
    if not _no_reception():
        r = pr["reception"]
        out["n_cites"] = r["n_cites"]
        out["used_as"] = r["function_share"]
        out["categories"] = [c for c, _ in r["categories"][:3]]
        out["field_says"] = [{"by": d["citing"], "date": d["date"], "display": _t(d["text"], 160),
                              "quote": d["quote"]} for d in r["descriptions"][-2:]]
    return out


def _stmt_items(rows: list[dict], n=TXT) -> list[dict]:
    return [{"by": s["speaker"], "about": s["about"], "date": s["date"], "kind": s["kind"],
             "facet": s["facet"], "role": s["role"], "epistemic": s["epistemic"],
             **({"condition": s["condition"]} if s.get("condition") else {}),
             "display": _t(s["text"], n), "quote": s["quote"]} for s in rows]


# ---------------------------------------------------------------- find
@tool
def search_papers(query: str, as_of: str, k: int = MAX_ITEMS, tier: str | None = None) -> list[dict]:
    """Paper-level hybrid search (BM25 + dense, RRF, per-paper cap after fusion). tier: 't1' has an abstract,
    't2' has a parsed full text, 'ft' has a careful/Sciverse text."""
    view = _view(as_of)
    return [_brief(view, i) for i in _idx().papers(query, as_of, k, tier=tier)]


@tool
def citations_of(paper: str, as_of: str, k: int = 20) -> list[dict]:
    view = _view(as_of)
    pid = view.canonical(paper)
    rows: dict[str, str] = {}
    for citing, date, _sid in view.cited_by(pid):
        if citing not in rows or date > rows[citing]:
            rows[citing] = date
    out = []
    for c, d in sorted(rows.items(), key=lambda kv: kv[1], reverse=True)[:k]:
        p = view.paper(c) or {}
        out.append({"id": c, "title": p.get("title"), "date": d, "visible": bool(p)})
    return out


@tool
def references_of(paper: str, as_of: str) -> list[dict]:
    view = _view(as_of)
    out = []
    for r in view.references(view.canonical(paper)):
        if r.startswith("stub:"):
            out.append({"id": r, "title": r[5:], "stub": True})
        else:
            p = view.paper(r) or {}
            out.append({"id": r, "title": p.get("title"), "date": p.get("date")})
    return out


@tool
def what_is_missing(topic: str, as_of: str) -> dict:
    """Known unknowns around a topic: works its papers cite that the corpus has no text for (ranked by how many
    of them cite it), plus unresolved references (stubs)."""
    view = _view(as_of)
    _, have = _documents()
    cnt: dict[str, int] = {}
    stubs: dict[str, int] = {}
    for o in _idx().papers(topic, as_of, 30):
        for r in view.references(o):
            if r.startswith("stub:"):
                stubs[r] = stubs.get(r, 0) + 1
            elif r not in have and view.paper(r):
                cnt[r] = cnt.get(r, 0) + 1
    unread = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_ITEMS]
    return {"topic": topic, "as_of": view.T,
            "known_unread": [{**_paper_head(view, o), "cited_by_n": n} for o, n in unread],
            "unresolved_cited": [{"id": s, "title": s[5:], "cited_by_n": n}
                                 for s, n in sorted(stubs.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_ITEMS]]}


@tool
def search_external(query: str, as_of: str, k: int = MAX_ITEMS) -> dict:
    """Sciverse semantic search, leak-safe: the year filter is as_of's year; a same-year hit is merged/shown
    only after its exact date resolved (a local registry record, or an arXiv month) — an unresolvable same-year
    hit is excluded and counted. Hits that resolve to a local paper_id merge into `local` (never written back);
    the rest stay `external` with their verbatim chunk as the quote."""
    if not _CFG.external:
        return {"external_disabled": True, "local": [], "external": []}
    view = _view(as_of)
    from ..sources.sciverse import SciverseClient
    t_year = int(view.T[:4])
    hits = SciverseClient().semantic_search(query, limit=max(k * 3, 12), year_lte=t_year)
    local, external = [], []
    excluded_same_year = excluded_undated = 0
    for h in hits:
        if h.status != "ready" or not h.title:
            continue
        chunk = (h.raw or {}).get("chunk")
        # 1) a local resolution decides by the registry's own visibility (first_hi <= T, alias-followed)
        pid = _resolve_local(view, title=h.title, doi=h.normalized_doi)
        p = view.paper(pid) if pid else None
        if p is not None:
            local.append({**_paper_head(view, pid), "resolved_from": "sciverse", "score": h.candidate_score,
                          "chunk": {"display": _t(chunk, TXT), "quote": chunk}})
        elif h.year is not None and h.year < t_year:
            # 2) older than the cutoff year: visible at year precision (its hi, Dec 31, is <= T)
            external.append({"external": True, "title": h.title, "year": h.year, "venue": h.venue,
                             "doc_id": h.source_record_id, "score": h.candidate_score,
                             "display": _t(chunk, TXT), "quote": chunk})
        elif h.year == t_year:
            # 3) same-year: only an exact date can decide (§2.2 — resolve, else exclude; never guess)
            from ..sources.refgraph import arxiv_lookup
            a = arxiv_lookup(h.title)
            if a and a.get("year") and a.get("month"):
                from ..core.asof import AsOf as _CoreAsOf, Date
                date = Date.month(a["year"], a["month"])
                if not _CoreAsOf(view.T).visible(date):
                    continue                       # resolved and published after T: invisible
                external.append({"external": True, "title": h.title, "year": h.year, "venue": h.venue,
                                 "date": date.hi_iso, "doc_id": h.source_record_id, "score": h.candidate_score,
                                 "display": _t(chunk, TXT), "quote": chunk})
            else:
                excluded_same_year += 1            # same-year and the exact date is unresolvable
                continue
        else:
            excluded_undated += 1                  # no year, or a year after T: invisible / undecidable
            continue
        if len(local) + len(external) >= k:
            break
    errors = [h.error_summary for h in hits if h.status != "ready" and h.error_summary][:2]
    return {"query": query, "as_of": view.T, "local": local, "external": external,
            "excluded_same_year_undated": excluded_same_year, "excluded_undated": excluded_undated,
            **({"errors": errors} if errors and not local and not external else {}),
            "note": "external hits are read-only during evaluation; nothing is written back"}


@tool
def expand_citations(seeds, as_of: str, k: int = MAX_ITEMS) -> dict:
    """Expand a seed set through citations: locally the seeds' references plus their co-citation partners
    (materialised cocite, ranked by co-citation count); externally refgraph's reference lists for up to three
    seeds, merged into local where a hit resolves. Read-only."""
    view = _view(as_of)
    if isinstance(seeds, str):
        seeds = [seeds]
    seeds = [view.canonical(s) for s in list(seeds)[:5]]
    score: dict[str, int] = {}
    via: dict[str, set] = {}
    for s in seeds:
        for r in view.references(s):
            if r in seeds:
                continue
            via.setdefault(r, set()).add("references")
            score[r] = score.get(r, 0)
        if view.cog is not None:
            for other, n in view.cog.execute(
                    "SELECT b, sum(n) FROM cocite WHERE a=? AND day<=? GROUP BY b", (s, view.T)):
                if other not in seeds:
                    score[other] = score.get(other, 0) + n
                    via.setdefault(other, set()).add("cocite")
            for other, n in view.cog.execute(
                    "SELECT a, sum(n) FROM cocite WHERE b=? AND day<=? GROUP BY a", (s, view.T)):
                if other not in seeds:
                    score[other] = score.get(other, 0) + n
                    via.setdefault(other, set()).add("cocite")
    ext_rows, ext_seen = [], {}
    if _CFG.external:
        from ..sources import refgraph
        for s in seeds[:3]:
            p = view.paper(s)
            if not p or not p.get("title"):
                continue
            refs, src = refgraph.references(p["title"], deadline=time.time() + 45)   # refgraph's clock is time.time
            for r in refs:
                t = (r.get("title") or "").strip()
                if not t:
                    continue
                pid = _resolve_local(view, title=t, doi=r.get("doi"), arxiv=r.get("arxiv"))
                if pid and (view.paper(pid) or pid in score):
                    score[pid] = score.get(pid, 0) + 1
                    via.setdefault(pid, set()).add(f"refgraph:{src}")
                else:
                    key = t.lower()
                    e = ext_seen.setdefault(key, {"external": True, "title": t, "year": r.get("year"),
                                                  "n": 0, "via": set()})
                    e["n"] += 1
                    e["via"].add(f"refgraph:{src}")
        ext_rows = sorted(ext_seen.values(), key=lambda e: (-e["n"], e["title"]))[:k]
        for e in ext_rows:
            e["via"] = sorted(e["via"])

    out_local = []
    for pid, sc in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:k]:
        if pid.startswith("stub:"):
            out_local.append({"id": pid, "title": pid[5:], "stub": True, "n_cocite": sc,
                              "via": sorted(via.get(pid, ()))})
        elif view.paper(pid):                       # invisible objects are never shown
            out_local.append({**_paper_head(view, pid), "n_cocite": sc, "via": sorted(via.get(pid, ()))})
    return {"seeds": seeds, "as_of": view.T, "local": out_local, "external": ext_rows,
            "note": "ranked by co-citation count; external hits are read-only during evaluation"}


# ---------------------------------------------------------------- read
@tool
def paper_card(paper: str, as_of: str) -> dict:
    """One paper's own account (self statements only): contributions, proposals, method points, findings, its
    own result-table rows (values valid inside this paper only), configuration, setting, limitations."""
    view = _view(as_of)
    pid = view.canonical(paper)
    p = view.paper(pid)
    if p is None and not pid.startswith("stub:"):
        return {"id": pid, "as_of": view.T, "error": "not visible at as_of"}
    selfs = view.statements(about=pid, kind="self")

    def by(facet, role=None):
        return _stmt_items([s for s in selfs if s["facet"] == facet and (role is None or s["role"] == role)])

    results = [s for s in selfs if s["pass"] == "results"]
    configs = [s for s in selfs if s["facet"] == "config"]
    setting = [s for s in selfs if s["facet"] == "setting"]
    return {"id": pid, "as_of": view.T, "title": (p or {}).get("title"), "date": (p or {}).get("date"),
            "ids": (p or {}).get("ids") or {},
            "contributions": by("contribution", "proposes")[:MAX_ITEMS],
            "proposes": [{"name": (s.get("meta") or {}).get("name"),
                          "artefact": (s.get("meta") or {}).get("artefact"), "quote": s["quote"]}
                         for s in selfs if (s.get("meta") or {}).get("name")][:MAX_ITEMS],
            "method": by("method")[:MAX_ITEMS],
            "findings": _stmt_items([s for s in selfs if s["facet"] == "result"
                                     and s["pass"] != "results"])[:MAX_ITEMS],
            "result_units": [{"object": (s.get("meta") or {}).get("object"),
                              "metric": (s.get("meta") or {}).get("metric"),
                              "value": (s.get("meta") or {}).get("value"),
                              "unit": (s.get("meta") or {}).get("unit"),
                              "direction": (s.get("meta") or {}).get("direction"),
                              "own": (s.get("meta") or {}).get("own"),
                              "condition": s["condition"], "date": s["date"], "quote": s["quote"]}
                             for s in results[:40]],
            "result_units_scope": "values from this paper's own tables only; not comparable across papers",
            "config": [{"item": ((s.get("meta") or {}).get("config") or {}).get("item"),
                        "value": ((s.get("meta") or {}).get("config") or {}).get("value"),
                        "unit": ((s.get("meta") or {}).get("config") or {}).get("unit"),
                        "applies_to": ((s.get("meta") or {}).get("config") or {}).get("applies_to"),
                        "quote": s["quote"]} for s in configs[:20]],
            "setting": _stmt_items(setting)[:MAX_ITEMS],
            "limitations": by("limitation")[:MAX_ITEMS],
            "n_references": len(view.references(pid))}


@tool
def read(paper: str, as_of: str, section: str | None = None, query: str | None = None, k: int = 6) -> list[dict]:
    """A paper's text: the passages most relevant to a query, or a section browse. Only sentences whose text
    version is dated <= as_of (v1 content at the v1 date, later-version additions at their own date)."""
    view = _view(as_of)
    pid = view.canonical(paper)
    if not view.visible(pid):
        return []
    if query:
        return [{"uid": h["uid"], "paper": h["paper"], "date": h["date"], "section": h["section"],
                 "display": _t(h["text"], 600), "quote": h["text"]}
                for h in _idx().passages(query, as_of, k, paper=pid)]
    from ..extract import reading as RD
    D, _ = _documents()
    full = RD.full_text(D, pid)
    if not full:
        return []
    sec = {u["uid"]: (u.get("section") or "") for u in full["units"]}
    out = []
    for s in full["sentences"]:
        if not s.get("date") or s["date"] > view.T:
            continue
        section_name = sec.get(s.get("unit"), "")
        if section and section.lower() not in section_name.lower():
            continue
        out.append({"uid": s["sid"], "paper": pid, "date": s["date"], "section": section_name,
                    "display": _t(s["text"], 600), "quote": s["text"]})
        if len(out) >= k:
            break
    return out


@tool
def deep_read(paper: str, as_of: str) -> dict:
    """Runtime deep extraction for ONE paper (v2.4: the same per-paper code the offline build runs — T1 + T2 +
    the results pass with the unified final check). Takes minutes; nothing is written back (evaluation
    discipline). Statements whose sentence date is after as_of are filtered out."""
    view = _view(as_of)
    pid = view.canonical(paper)
    if not view.visible(pid):
        return {"id": pid, "as_of": view.T, "error": "not visible at as_of"}
    from ..extract import passes as PS
    D, _ = _documents()
    r = PS.deep_read(D, view.reg, pid)
    items = []
    for s in r["statements"]:
        if not s.date or s.date > view.T:
            continue
        items.append({"pass": s.pass_name, "facet": s.facet, "role": s.role, "epistemic": s.epistemic,
                      "date": s.date, "condition": s.condition, "display": _t(s.text), "quote": s.quote,
                      **({"meta": s.meta} if s.facet in ("config", "result") and s.meta else {})})
    return {"id": pid, "as_of": view.T, "source": r["source"], "own_methods": r["own_methods"],
            "stats": r["stats"], "statements": items, "written_back": False}


# ---------------------------------------------------------------- evidence
@tool
def find_evidence(claim: str, as_of: str, k: int = MAX_ITEMS) -> list[dict]:
    """Verbatim evidence for a claim: dated statements (self + how other papers describe it) and raw passages,
    hybrid-ranked. Every item's `quote` is the verbatim source sentence."""
    view = _view(as_of)
    kind = "self" if _abl("self_only") else None
    ids = _idx().statements(claim, as_of, k, kind=kind)
    rows = {r["id"]: r for r in view.statements_by_id(ids)}
    out = []
    for i in ids:
        s = rows.get(i)
        if s is None:
            continue
        if _no_reception() and s["kind"] == "other":
            continue
        out.append({"kind": s["kind"], "by": s["speaker"], "about": s["about"], "date": s["date"],
                    "facet": s["facet"], "role": s["role"], "epistemic": s["epistemic"],
                    "display": _t(s["text"], 200), "quote": s["quote"]})
        if len(out) >= k:
            break
    for h in _idx().passages(claim, as_of, max(0, k - len(out))):
        out.append({"kind": "passage", "by": h["paper"], "date": h["date"], "section": h["section"],
                    "display": _t(h["text"], 200), "quote": h["text"]})
    return out[:k]


# ---------------------------------------------------------------- field
def _scope(view: AsOf, topic: str) -> tuple[list[str], set[str]]:
    seeds = _idx().papers(topic, view.T, 40)
    scope = set(seeds)
    for sd in seeds:
        scope |= {r for r in view.references(sd) if view.visible(r)}
    return seeds, scope


@tool
def field_map(topic: str, as_of: str) -> dict:
    """The families of approaches around a topic at T (materialised Leiden snapshots + LLM names), their
    family-level facts with independence counts and status timelines, and the boundary (cited-but-unread and
    unresolved works). A T before the snapshot grid says so instead of clamping."""
    if _abl("self_only"):
        return _ablated("self_only")
    view = _view(as_of)
    seeds, scope = _scope(view, topic)
    fams = F.families(view, scope=scope)
    out = []
    for f in fams["families"][:MAX_ITEMS]:
        members = [{"id": m, "title": (view.paper(m) or {}).get("title"), "date": (view.paper(m) or {}).get("date")}
                   for m in f["members"][:6]]
        fx = FA.facts(view, family_id=f["family_id"], k=4)
        out.append({"family_id": f["family_id"], "name": f["name"], "named_by": f["named_by"],
                    "n_members": f["n_members"], "members": members,
                    "facts": [{"facet": x["facet"], "display": _t(x["text"], 200), "quote": x["quote"],
                               "status": x["status"],
                               **({} if _abl("-time") else {"contested_since": x["contested_since"],
                                                            "first_seen": x["first_seen"]}),
                               "n_independent": x["n_independent"]} for x in fx]})
    stubs: dict[str, int] = {}
    for sd in seeds:
        for r in view.references(sd):
            if r.startswith("stub:"):
                stubs[r] = stubs.get(r, 0) + 1
    return {"topic": topic, "as_of": view.T, "snapshot": fams["snapshot"],
            "before_grid": fams["before_grid"], "families": out,
            "boundary": {"cited_not_resolved": len(stubs),
                         "most_cited_unresolved": [{"id": s, "title": s[5:], "cited_by_n": n}
                                                   for s, n in sorted(stubs.items(),
                                                                      key=lambda kv: (-kv[1], kv[0]))[:5]]}}


@tool
def paper_profile(paper: str, as_of: str) -> dict:
    """How the field sees one paper at T: its own account vs. its reception (counts from the materialised daily
    table, descriptions from statements), approved reception shifts, lineage with §2.4 effective dates, family
    membership, family-level facts about it, and the names it is known as."""
    view = _view(as_of)
    pid = view.canonical(paper)
    if not view.visible(pid) and not pid.startswith("stub:"):
        return {"id": pid, "as_of": view.T, "error": "not visible at as_of"}
    pr = P.profile(view, pid)
    ident = Identity(view)
    lin = L.lineage_of(view, pid)

    def edge_rows(edges, n=MAX_ITEMS):
        out = []
        for e in edges[:n]:
            qs = L.quotes(view, e, k=1)
            out.append({"id": e["parent"] if e["child"] == pid else e["child"], "relation": e["relation"],
                        "valid_from": e["valid_from"], "n_self": sum(a["kind"] == "self" for a in e["assertions"]),
                        "n_third": sum(a["kind"] == "third" for a in e["assertions"]),
                        **({"quote": qs[0]["quote"]} if qs else {})})
        return out

    out = {"id": pid, "as_of": view.T, "title": (pr["paper"] or {}).get("title"),
           "date": (pr["paper"] or {}).get("date"), "ids": (pr["paper"] or {}).get("ids") or {},
           "self": [{"facet": s["facet"], "display": _t(s["text"], 200), "quote": s["quote"],
                     **({"name": s["name"]} if s.get("name") else {})}
                    for s in pr["self"] if s["facet"] in ("contribution", "limitation")][:MAX_ITEMS],
           "lineage": {"parents": edge_rows(lin["parents"]), "children": edge_rows(lin["children"]),
                       "combines": [{"members": h["members"], "relation": h["relation"], "date": h["date"]}
                                    for h in lin["combines"][:4]]},
           "family": (lambda f: None if not f else {"name": f["name"], "family_id": f["family_id"],
                                                    "n_members": len(f["members"])})(F.family_of(view, pid)),
           "facts": [{"facet": x["facet"], "display": _t(x["text"], 200), "quote": x["quote"],
                      "status": x["status"], "n_independent": x["n_independent"],
                      **({} if _abl("-time") else {"contested_since": x["contested_since"]})}
                     for x in FA.facts(view, members=[pid], k=6)],
           "known_as": (ident.surface_names(pid) or ident.aliases(pid))[:6],
           "evidence": pr["evidence"]}
    if not _no_reception():
        r = pr["reception"]
        out["reception"] = {"n_cites": r["n_cites"], "n_citing": r["n_citing"], "first": r["first"],
                            "last": r["last"], "used_as": r["function_share"],
                            "relation": r["relation_share"], "categories": r["categories"],
                            "limitations": [{"display": _t(x["text"], 200), "n": x["n"], "citing": x["citing"]}
                                            for x in r["limitations"]],
                            "descriptions": [{"date": d["date"], "by": d["citing"],
                                              "display": _t(d["text"], 200), "quote": d["quote"]}
                                             for d in r["descriptions"]]}
        if not _abl("-time"):
            out["reception"]["monthly"] = r["monthly"]
    if not _abl("-time") and not _no_reception():
        out["shifts"] = [{"type": e["type"], "date": e["date"], "window": e["window"],
                          "direction": e["direction"], "evidence": e["evidence"][:2]} for e in pr["shifts"]]
    return out


@tool
def closest_prior(idea: str, as_of: str, k: int = MAX_ITEMS) -> list[dict]:
    view = _view(as_of)
    return [_brief(view, i) for i in _idx().papers(idea, as_of, k)]


@tool
def baselines_for(problem: str, as_of: str, k: int = MAX_ITEMS) -> list[dict]:
    if _no_reception():
        return _ablated("-reception")
    view = _view(as_of)
    seeds = _idx().papers(problem, as_of, 30)
    out = []
    for obj, n, ps in C.baselines_in(view, seeds)[:k]:
        if obj.startswith("stub:"):
            out.append({"id": obj, "title": obj[5:], "stub": True, "used_as_baseline_by": n,
                        "in_papers": sorted(ps)[:5]})
        else:
            out.append({**_paper_head(view, obj), "used_as_baseline_by": n, "in_papers": sorted(ps)[:5]})
    return out


@tool
def open_issues(topic: str, as_of: str) -> list[dict]:
    """The limitations the field states about works on a topic, as materialised family-level facts: support,
    independence, status, and (unless -time) when each formed and whether it is contested."""
    if _abl("self_only"):
        return _ablated("self_only")
    view = _view(as_of)
    _, scope = _scope(view, topic)
    fx = FA.facts(view, members=sorted(scope)[:200], facets=("limitation",), k=MAX_ITEMS)
    return [{"issue": _t(x["text"], 200), "quote": x["quote"], "status": x["status"],
             "n_independent": x["n_independent"], "about": x["members"][:5],
             **({} if _abl("-time") else {"first_seen": x["first_seen"], "contested_since": x["contested_since"]})}
            for x in fx]


@tool
def frontier(topic: str, as_of: str) -> dict:
    """The newest works on a topic and the diachronic signals around them: works becoming standard components
    or baselines, and superseded works (approved shift events with closed windows at T)."""
    if _abl("-time"):
        return _ablated("-time")
    view = _view(as_of)
    cands = _idx().papers(topic, as_of, 40)
    cand_set = set(cands)
    briefs = [_brief(view, o) for o in cands]
    turning, superseded = [], []
    for e in SH.events(view):
        if e["subject"] not in cand_set:
            continue
        if e["type"] in ("became_component", "became_baseline"):
            turning.append({"id": e["subject"], "event": e["type"], "since": e["date"],
                            "direction": e["direction"]})
        elif e["type"] == "superseded":
            by = next((x["child"] for x in L.edges(view, e["subject"])
                       if x["parent"] == e["subject"] and x["relation"] == "replaces"), None)
            superseded.append({"id": e["subject"], "by": by, "since": e["date"]})
    newest = sorted(briefs, key=lambda b: b["date"] or "", reverse=True)
    return {"topic": topic, "as_of": view.T, "newest": newest[:MAX_ITEMS],
            "becoming_standard": turning[:MAX_ITEMS], "superseded": superseded[:MAX_ITEMS]}


# ---------------------------------------------------------------- compare
@tool
def compared_with(paper: str, as_of: str) -> dict:
    """Who compared against a paper and the sentence's own qualitative outcome (user ruling 10-05: no
    cross-paper numbers — values stay in each paper's card)."""
    if _no_reception():
        return _ablated("-reception")
    view = _view(as_of)
    pid = view.canonical(paper)
    g = C.compared_with(view, pid)
    return {"id": pid, "as_of": view.T, "n_compared_by": g["n_compared_by"], "outcomes": g["outcomes"],
            "first": g["first"],
            "rows": [{"by": r["by"], "date": r["date"], "function": r["function"], "outcome": r["outcome"],
                      "display": _t(r["quote"], 240), "quote": r["quote"]} for r in g["rows"][-MAX_ITEMS:]],
            "note": "qualitative outcomes only; numbers stay inside each paper's card"}


# ---------------------------------------------------------------- flat arm
@tool
def search_flat(query: str, as_of: str, k: int = 16) -> dict:
    """The flat arm's only tool (§9.4): same-budget flat retrieval — raw passages, no structure, no field
    layer."""
    return {"as_of": as_of,
            "passages": [{"paper": h["paper"], "date": h["date"], "section": h["section"],
                          "display": _t(h["text"], 400), "quote": h["text"]}
                         for h in _idx().passages(query, as_of, k, per_paper=3)]}


# ---------------------------------------------------------------- arm wiring
_FLAT = ("search_flat",)
_FULL = ("search_papers", "citations_of", "references_of", "what_is_missing", "search_external",
         "expand_citations", "paper_card", "read", "deep_read", "find_evidence", "field_map", "paper_profile",
         "closest_prior", "baselines_for", "open_issues", "frontier", "compared_with")
_BY_NAME = {f.__name__: f for f in (search_papers, citations_of, references_of, what_is_missing,
                                    search_external, expand_citations, paper_card, read, deep_read,
                                    find_evidence, field_map, paper_profile, closest_prior, baselines_for,
                                    open_issues, frontier, compared_with, search_flat)}


def active_tools() -> dict:
    """name -> callable, for the arm the server was configured with (none: nothing; flat: the flat search only;
    summary/full: everything — the summary arm compresses inside the tool wrapper)."""
    names = {"none": (), "flat": _FLAT, "summary": _FULL, "full": _FULL}[_CFG.arm]
    return {n: _BY_NAME[n] for n in names}
