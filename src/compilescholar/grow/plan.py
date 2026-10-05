# -*- coding: utf-8 -*-
"""L7 growth: diagnose what the store is missing and plan typed acquisitions (DESIGN-LITERATURE-LAYER §8, N10).

Offline and question-blind: gaps are computed from corpus statistics only (citations, scope, documents), never from a
benchmark or a user question. Acquisitions go back through the same pipeline (sources -> documents -> citations ->
extract -> index); nothing is written to the store from here. During an evaluation no growth runs (no per-question
store changes; the old rule "never add papers to the KB per question" is kept).

Gap types and the action each one maps to (the old growth loop used one action — query = gap text — and resolved
1/24 gaps; the typed mapping is the fix):
  unread_member   a paper cited by >= MIN_CITERS papers in scope, no full text held      -> fetch its arXiv HTML
  shallow         a paper in scope with reception (>= MIN_RECEPTION citing papers) but only T1 extraction
                                                                                         -> deep-extract it (T2)
  frontier        scope papers whose newest citing paper is older than STALE_MONTHS before `now`
                                                                                         -> harvest OAI-PMH since then,
                                                                                            fetch new citers
  unresolved      bibliography entries cited >= MIN_CITERS times that resolve to no arXiv id (stubs)
                                                                                         -> record only (no fetch path)
Each item carries the evidence for its priority (counts) so a run can be audited and capped by budget."""
from __future__ import annotations

from collections import Counter

from ..dfc import store
from ..documents.build import Documents

MIN_CITERS = 3
MIN_RECEPTION = 5
STALE_MONTHS = 6


def _months_between(a: str, b: str) -> int:
    return (int(b[:4]) - int(a[:4])) * 12 + int(b[5:7]) - int(a[5:7])


def diagnose(now: str, limit: int = 500) -> dict:
    store.require_fresh("papers", "documents", "citations", "extract")
    cit = store.connect("citations", readonly=True)
    ext = store.connect("extract", readonly=True)
    have = set(Documents().ids())
    scope = {r[0] for r in ext.execute("SELECT arxiv_id FROM scope")}
    tiers = dict(ext.execute("SELECT arxiv_id, tier FROM tiers").fetchall())
    citers = Counter(dict(cit.execute(
        "SELECT cited, count(DISTINCT citing) FROM cites GROUP BY cited").fetchall()))
    unread = [(o[6:], n) for o, n in citers.most_common() if o.startswith("paper:") and o[6:] not in have
              and n >= MIN_CITERS]
    shallow = [(a, citers.get(f"paper:{a}", 0)) for a in scope if tiers.get(a) == "T1" and a in have
               and citers.get(f"paper:{a}", 0) >= MIN_RECEPTION]
    newest = dict(cit.execute("SELECT cited, max(date) FROM cites GROUP BY cited").fetchall())
    frontier = sorted({d for a in scope if (d := newest.get(f"paper:{a}")) and _months_between(d, now) > STALE_MONTHS})
    unresolved = [(o[5:], n) for o, n in citers.most_common() if o.startswith("stub:") and n >= MIN_CITERS]
    return {"now": now,
            "unread_member": [{"arxiv_id": a, "cited_by": n, "action": "fetch_html"} for a, n in unread[:limit]],
            "shallow": [{"arxiv_id": a, "cited_by": n, "action": "deep_extract"}
                        for a, n in sorted(shallow, key=lambda x: -x[1])[:limit]],
            "frontier": {"stale_since": frontier[0] if frontier else None, "n_scope_papers_stale":
                         sum(1 for a in scope if (d := newest.get(f"paper:{a}")) and _months_between(d, now) > STALE_MONTHS),
                         "action": "harvest_oai_since_then"},
            "unresolved": [{"title": t, "cited_by": n, "action": "record_only"} for t, n in unresolved[:limit]],
            "counts": {"unread_member": len(unread), "shallow": len(shallow), "unresolved": len(unresolved)}}


def acquire(plan: dict, max_fetch: int = 50, log=print) -> dict:
    """Execute the fetch actions of a plan within a budget (arXiv HTML at the 15 s pace). Returns what was fetched;
    the caller then rebuilds documents -> citations -> extract -> index (stale stages refuse to run otherwise)."""
    from ..sources import arxiv_html
    got, none = [], []
    for item in plan["unread_member"][:max_fetch]:
        p = arxiv_html.fetch(item["arxiv_id"])
        (got if p else none).append(item["arxiv_id"])
        log(f"[grow] {item['arxiv_id']}: {'ok' if p else 'no html'}")
    return {"fetched": got, "no_html": none}
