"""Gap-driven search primitive (2026-09-25, user-directed design).

The differentiation argument (user: "外部检索工具和引用链别人都做过，没
创新性"): generic agentic search walks citations and refines keywords blind.
Our knowledge model knows WHAT IS MISSING — 1,939 empirical absence records
("SAMMR cannot compensate sample-induced dispersion", with verbatim quotes)
and 2,014 derived matrix holes. gap_search turns those coordinates into
TARGETED queries whose terms do not exist in the user's question, reaching
filler papers that blind retrieval cannot express.

Flow:
  1. gap matching: embed the query against absence/derived-gap docs
     (kb-side injects a prebuilt index; fallback = token overlap)
  2. query synthesis: matched gap -> 2-3 directed queries built from
     [gap subject + missing-content terms], optionally refined by the
     entity's lineage successors (has the gap already been filled by a
     method the corpus knows?)
  3. execution: queries run through the SAME tiered engine as blind
     search (Sciverse semantic leads, keyword fallbacks, circuit, rerank)
  4. fill annotation: returned candidates are matched back against the
     gap coordinate (entity name + missing terms) — hit = a candidate
     that plausibly fills the gap

Engine-agnostic: the caller injects `search_fn(query, limit)` (defaults to
SearchService.search) and `gap_docs` (list of {doc, subject, missing,
entity, kind}). This module only contributes the STRATEGY — the layer the
paper claims.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class GapHit:
    gap_doc: dict          # the matched absence/derived record
    score: float
    queries: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GapSearchResult:
    query: str
    gaps: list[GapHit] = field(default_factory=list)
    candidates: list[Any] = field(default_factory=list)
    fill_annotations: list[dict] = field(default_factory=list)
    latency: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _tokens(t: str) -> set:
    return {w for w in _norm(t).split() if len(w) > 3}


def match_gaps(query: str, gap_docs: list[dict], k: int = 3,
               embed: Callable[[list[str]], list[list[float]]] | None = None
               ) -> list[GapHit]:
    """Query -> top-k matching gap coordinates. Embedding when the caller
    provides it (same-model discipline), token-overlap fallback otherwise
    (deterministic, zero-API)."""
    if not gap_docs:
        return []
    if embed is not None:
        try:
            import math
            texts = [query] + [g["doc"] for g in gap_docs]
            vecs = embed(texts)
            qv = vecs[0]
            def _cos(a, b):
                dot = sum(x * y for x, y in zip(a, b))
                na = math.sqrt(sum(x * x for x in a)) or 1.0
                nb = math.sqrt(sum(x * x for x in b)) or 1.0
                return dot / (na * nb)
            scored = sorted(
                ((_cos(qv, v), g) for v, g in zip(vecs[1:], gap_docs)),
                key=lambda kv: -kv[0])
            return [GapHit(gap_doc=g, score=round(s, 4))
                    for s, g in scored[:k] if s > 0.3]
        except Exception:
            pass  # fall through to token overlap
    qtoks = _tokens(query)
    scored = []
    for g in gap_docs:
        gtoks = _tokens(g["doc"])
        ov = len(qtoks & gtoks)
        if ov:
            scored.append((ov / max(1, len(qtoks | gtoks)), g))
    scored.sort(key=lambda kv: -kv[0])
    return [GapHit(gap_doc=g, score=round(s, 4))
            for s, g in scored[:k] if s > 0.15]


def synthesize_queries(gap: dict, question: str = "") -> list[str]:
    """Gap coordinate -> directed retrieval queries. The terms here are
    things the QUESTION does not contain — that is the whole point: the
    knowledge model contributes vocabulary blind search cannot invent."""
    subject = (gap.get("subject") or gap.get("entity") or "").strip()
    missing = (gap.get("missing") or "").strip()
    queries = []
    if subject and missing:
        # direct filler search: who addresses exactly this missing piece
        queries.append(f"{subject} {missing}")
        # inverse framing: solutions often describe the fix, not the hole
        queries.append(f"{missing.replace('compensation for ', 'compensation ').replace('applicability to ', '')} method")
    elif subject:
        queries.append(f"{subject} limitations solutions alternative")
    # question-context variant: anchor the filler to the question's domain
    if question:
        qhead = _norm(question)[:120]
        if subject and subject.lower() not in qhead:
            queries.append(f"{subject} {qhead.split()[0]} {' '.join(qhead.split()[:4])}")
    return [q for q in queries if len(_tokens(q)) >= 2][:3]


def _fills_gap(candidate: Any, gap: dict) -> bool:
    """Fill annotation: does this candidate plausibly address the gap?
    Cheap lexical test — candidate title/abstract carries the gap subject
    AND at least one missing-content token."""
    subject = _norm(gap.get("subject") or gap.get("entity") or "")
    missing_toks = _tokens(gap.get("missing") or "")
    text = _norm(str(getattr(candidate, "title", "") or ""))
    raw = getattr(candidate, "raw", None)
    if isinstance(raw, dict):
        text += " " + _norm(str(raw.get("abstract") or "")[:800])
    if subject and subject not in text:
        subj_toks = subject.split()
        if not (len(subj_toks) >= 1 and all(t in text for t in subj_toks[:2])):
            return False
    return bool(missing_toks & set(text.split()))


def gap_search(
    question: str,
    *,
    gap_docs: list[dict],
    search_fn: Callable[[str, int], Any] | None = None,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
    limit: int = 10,
    max_gaps: int = 2,
) -> GapSearchResult:
    """Gap-driven search entry. search_fn(query, limit) defaults to the
    shared SearchService (injected by the broker with ledger/semaphore)."""
    import time
    lat: dict[str, float] = {}
    notes: list[str] = []

    t0 = time.time()
    hits = match_gaps(question, gap_docs, k=max_gaps, embed=embed)
    lat["gap_match_ms"] = round((time.time() - t0) * 1000, 1)
    if not hits:
        notes.append("no matching gap coordinate — caller should fall back to blind search")
        return GapSearchResult(query=question, latency=lat, notes=notes)

    for h in hits:
        h.queries.extend(synthesize_queries(h.gap_doc, question))

    if search_fn is None:
        from retrieval.search_service import SearchService
        svc = SearchService(limit=limit)
        search_fn = lambda q, k: svc.search(q, mode="full", limit=k)  # noqa: E731

    t0 = time.time()
    candidates: list[Any] = []
    fill_notes: list[dict] = []
    _fill_seen: set[tuple[str, str]] = set()
    for h in hits[:max_gaps]:
        for q in h.queries:
            res = search_fn(q, limit)
            cands = getattr(res, "candidates", None) or (res or [])
            for c in cands:
                candidates.append(c)
                ctitle = str(getattr(c, "title", ""))[:100]
                gsubj = h.gap_doc.get("subject") or h.gap_doc.get("entity")
                if _fills_gap(c, h.gap_doc) and (ctitle, gsubj) not in _fill_seen:
                    _fill_seen.add((ctitle, gsubj))
                    fill_notes.append({
                        "candidate_title": ctitle,
                        "gap_subject": gsubj,
                        "gap_missing": h.gap_doc.get("missing") or "",
                        "gap_kind": h.gap_doc.get("kind", "absence"),
                    })
    lat["search_ms"] = round((time.time() - t0) * 1000, 1)

    # dedupe candidates (title-normalized; engine-level dedupe already ran
    # per query, this merges across gap queries)
    seen: set[str] = set()
    uniq: list[Any] = []
    for c in candidates:
        key = _norm(str(getattr(c, "title", "") or ""))[:80]
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        uniq.append(c)
    return GapSearchResult(query=question, gaps=hits,
                           candidates=uniq[:limit],
                           fill_annotations=fill_notes[:20],
                           latency=lat, notes=notes)


def build_gap_docs(views: dict) -> list[dict]:
    """CompileScholar adapter: views.json -> gap_docs. Empirical absences
    (subject/missing/quote) + derived matrix holes (entity/subject_family)."""
    cov = (views or {}).get("coverage", {}) or {}
    docs = []
    for a in cov.get("absences_extracted") or []:
        # P0-2（FIX-PLAN v2）：缺口生灭链——resolved_by（LLM 核验过的
        # in-corpus 解决者）随 gap doc 暴露：消费方（gap_search 调用者/
        # agent）能先查库内解决者的记录再决定外部检索。未核验候选链
        # （resolution_candidates）不进 doc——结构候选不是语义断言。
        rb = a.get("resolved_by") or []
        doc = {
            "doc": f"{a.get('subject')} lacks {a.get('missing')}",
            "subject": a.get("subject"), "missing": a.get("missing"),
            "kind": "empirical_absence", "paper_id": a.get("paper_id"),
            "quote": a.get("quote", "")[:200],
        }
        if rb:
            doc["resolved_by"] = [
                {"entity": r.get("entity"), "paper_id": r.get("paper_id"),
                 "year": r.get("year"), "relation": r.get("relation")}
                for r in rb[:3]]
        docs.append(doc)
    for a in cov.get("absences_derived") or []:
        docs.append({
            "doc": f"no corpus results for {a.get('entity')} on "
                   f"{a.get('subject_family')}",
            "subject": a.get("subject_family"),
            "entity": a.get("entity"),
            "missing": f"results on {a.get('subject_family')}",
            "kind": "derived_hole",
        })
    return docs
