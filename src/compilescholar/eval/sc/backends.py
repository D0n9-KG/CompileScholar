# -*- coding: utf-8 -*-
"""Retrieval backends for the SC runner.

SystemSearch is the closed-pool adapter over our own L6 tools (tools.api): every call carries the query's
as_of (end of the source-paper month, protocol.as_of_for), and every hit is mapped registry paper_id -> SC
corpus doc_id. Hits outside the closed pool are dropped (the protocol is a closed-pool one: only corpus docs
may be submitted), the query's own paper is withheld from every result, and external retrieval is switched off
server-side (ToolConfig(external=False)) — the SC protocol disallows web search.

The tools are NOT re-implemented here: search goes through tools.api.search_papers / find_evidence /
expand_citations / paper_card exactly as an MCP consumer would call them. A tool whose upstream stage is not
built yet (e.g. the statements index before the overnight rebuild finishes) degrades to an empty result and is
recorded in the per-query trace — the shallow pilot form expects a near-empty cognition layer and must not
crash on it.

Overfetch: the real papers index spans the whole registry (1.19M papers), so pool-only hits can be diluted by
non-pool papers; searches ask for `pool_factor * k` briefs and cut back to k after the pool mapping.
"""
from __future__ import annotations

from . import protocol
from .protocol import Corpus, IdMap


class SystemSearch:
    """One instance per run; stateless per query except the degradation trace reset by `begin()`."""

    POOL_FACTOR = 3

    def __init__(self, idmap: IdMap, corpus: Corpus, tools=None):
        self.idmap = idmap
        self.corpus = corpus
        self.tools = tools
        self.degraded: list[str] = []

    def begin(self) -> None:
        self.degraded = []

    def _tools(self):
        if self.tools is None:
            from ...tools import api
            self.tools = api
        return self.tools

    def _to_docs(self, paper_ids, withheld: set[str]) -> list[str]:
        out = []
        for pid in paper_ids:
            if not pid or pid.startswith("stub:"):
                continue
            for d in self.idmap.docs_of(pid):
                if d not in withheld and d not in out:
                    out.append(d)
        return out

    def search(self, query: str, as_of: str, k: int, withheld: set[str]) -> list[str]:
        """Pool doc_ids ranked by our hybrid paper search (BM25 + dense RRF, as-of filtered)."""
        t = self._tools()
        try:
            hits = t.search_papers(query, as_of, min(k * self.POOL_FACTOR, 150))
        except Exception as e:                          # noqa: BLE001 — a degraded channel is recorded, not fatal
            self.degraded.append(f"search_papers: {type(e).__name__}: {str(e)[:100]}")
            return []
        return self._to_docs([h.get("id") for h in hits if isinstance(h, dict)], withheld)[:k]

    def evidence(self, claim: str, as_of: str, k: int, withheld: set[str]) -> list[str]:
        """Pool doc_ids from sentence-level evidence (statements about/by papers + passages)."""
        t = self._tools()
        try:
            rows = t.find_evidence(claim, as_of, min(k * self.POOL_FACTOR, 60))
        except Exception as e:                          # noqa: BLE001
            self.degraded.append(f"find_evidence: {type(e).__name__}: {str(e)[:100]}")
            return []
        pids = []
        for r in rows if isinstance(rows, list) else []:
            if not isinstance(r, dict):
                continue
            if r.get("kind") == "other" and r.get("about"):
                pids.append(r["about"])                 # a statement describing work X is evidence about X
            pids.append(r.get("about") or r.get("by") or "")
        return self._to_docs(pids, withheld)[:k]

    def expand(self, seed_docs: list[str], as_of: str, k: int, withheld: set[str]) -> list[str]:
        """Pool doc_ids from local citation expansion (co-citation + references; external leg is off)."""
        t = self._tools()
        seeds = [p for d in seed_docs[:5] if (p := self.idmap.paper_of(d))]
        if not seeds:
            return []
        try:
            got = t.expand_citations(seeds, as_of, min(k * self.POOL_FACTOR, 60))
        except Exception as e:                          # noqa: BLE001
            self.degraded.append(f"expand_citations: {type(e).__name__}: {str(e)[:100]}")
            return []
        rows = (got or {}).get("local") or []
        return self._to_docs([r.get("id") for r in rows if isinstance(r, dict) and not r.get("stub")],
                             withheld)[:k]

    def state_expand(self, seed_docs: list[str], as_of: str, k: int, withheld: set[str]) -> list[str]:
        """Pool doc_ids via the compiled state's graph channels: co-members of the seeds' tight families at T
        (family_neighbors; giant Leiden catch-alls are skipped inside the tool), then co-citation partners of
        the same seeds (expand_citations in two 5-seed chunks — the tool caps seeds at 5 per call).
        Pure state reads (no LLM call); as-of enforced twice — the readers pick the snapshot / cocite rows
        <= T, and the corpus-date guard below drops anything published after the query's as-of month.
        Reachability audit 10-10: 46% of never-seen pilot50 positives are one family/cocite hop from the
        agent's seen set (the operational ceiling is lower — giant families carry no ranking signal)."""
        t = self._tools()
        seeds = [p for d in seed_docs[:20] if (p := self.idmap.paper_of(d))]
        if not seeds:
            return []
        pids: list[str] = []
        for chunk in (seeds[:10], seeds[10:20]):        # the tool caps seeds at 10 per call
            if not chunk:
                continue
            try:
                got = t.family_neighbors(chunk, as_of, min(k * self.POOL_FACTOR, 60))
                pids += [r.get("id") for r in (got or {}).get("local") or [] if isinstance(r, dict)]
            except Exception as e:                      # noqa: BLE001 — a degraded channel is recorded, not fatal
                self.degraded.append(f"family_neighbors: {type(e).__name__}: {str(e)[:100]}")
        for chunk in (seeds[:5], seeds[5:10], seeds[10:15], seeds[15:20]):
            if not chunk:
                continue
            try:
                got = t.expand_citations(chunk, as_of, min(k * self.POOL_FACTOR, 60))
                pids += [r.get("id") for r in (got or {}).get("local") or []
                         if isinstance(r, dict) and not r.get("stub")]
            except Exception as e:                      # noqa: BLE001
                self.degraded.append(f"expand_citations(state): {type(e).__name__}: {str(e)[:100]}")
        if not pids:
            return []
        docs = self._to_docs(pids, withheld)
        return [d for d in docs if (self.corpus.dates.get(d) or "")[:7] <= as_of[:7]][:k]

    def card(self, doc_id: str, as_of: str) -> dict | None:
        """The paper_card of one pool paper (self statements); None when the card errors or is empty."""
        pid = self.idmap.paper_of(doc_id)
        if not pid:
            return None
        t = self._tools()
        try:
            c = t.paper_card(pid, as_of)
        except Exception as e:                          # noqa: BLE001
            self.degraded.append(f"paper_card({doc_id}): {type(e).__name__}: {str(e)[:80]}")
            return None
        if not isinstance(c, dict) or c.get("error"):
            return None
        if not any(c.get(f) for f in ("contributions", "method", "findings", "proposes")):
            return None
        return c

    # ---- candidate rendering (canonical pool text, not registry text — the agent reads what the pool holds)
    def candidate(self, doc_id: str, snippet_chars: int = 280) -> dict:
        return {"doc_id": doc_id, "title": self.corpus.titles.get(doc_id, ""),
                "date": self.corpus.dates.get(doc_id, ""),
                "snippet": self.corpus.snippet(doc_id, snippet_chars)}


def backfill_docs(bench_dir, query_type: str) -> dict[str, list[str]]:
    return protocol.load_backfill(bench_dir, query_type)
