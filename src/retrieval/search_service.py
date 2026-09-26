"""Latency-tiered search service (P4-A A5, 2026-09-24).

Facade composing A1-A4 into the two paths the real-scenario latency
budget requires (RETRIEVAL-MODULE-DESIGN.md):

  fast  (<2s target): query-cache hit, else ONE tier (domain-preferred
        source, 5s cap, no LLM understanding, no rerank)
  full  (<15s target): A3 understand -> per-subquery discover_tiered
        (degradation + circuit) -> merge/dedupe -> A4 embed rerank ->
        caches filled for the next fast hit

  mode="auto": fast first; a thin/empty fast result escalates to full.

This is the entry the CompileScholar SQA2 broker calls; inject
kb_infra-backed chat/embed callables there so calls land in the shared
ledger and semaphore. All latencies recorded per stage in the result for
the A6 latency dashboard.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from retrieval.acquisition import (
    TIER_PRESETS,
    SourceCandidate,
    discover_tiered,
)
from retrieval.circuit import SourceCircuit
from retrieval.query_understanding import (
    UnderstoodQuery,
    understand_query,
)
from retrieval.rerank import (
    DoiMetaCache,
    QueryResultCache,
    rerank_candidates,
)


@dataclass(frozen=True)
class SearchResult:
    query: str
    mode_used: str                    # fast | full | fast_cache
    candidates: list[SourceCandidate] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    understood: UnderstoodQuery | None = None
    latency: dict[str, float] = field(default_factory=dict)  # stage -> ms
    tiers: list[dict[str, Any]] = field(default_factory=list)
    reranked: bool = False
    pool: list[SourceCandidate] = field(default_factory=list)
    # pool = full pre-rerank merged candidate list (diagnostic: separates
    # "retrieval found it but ranked it low" from "never retrieved")

    @property
    def total_ms(self) -> float:
        return sum(self.latency.values())


def _dedupe(cands: list[SourceCandidate]) -> list[SourceCandidate]:
    seen: set[str] = set()
    out = []
    for c in cands:
        key = c.normalized_doi or (c.source_record_id
                                   or f"{c.title}@{c.source_name}")
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


class SearchService:
    def __init__(
        self,
        *,
        chat: Callable[[str], str] | None = None,
        embed: Callable[[list[str]], list[list[float]]] | None = None,
        circuit: SourceCircuit | None = None,
        doi_cache_path: str | None = None,
        clients: dict[str, Any] | None = None,
        limit: int = 10,
    ):
        self.chat = chat
        self.embed = embed
        self.circuit = circuit or SourceCircuit()
        self.clients = clients
        self.limit = limit
        self.query_cache = QueryResultCache(max_entries=256, ttl_s=3600.0)
        self.doi_cache = DoiMetaCache(doi_cache_path) if doi_cache_path else None

    # ---- fast path -------------------------------------------------------

    def _fast(self, query: str, limit: int) -> SearchResult:
        t0 = time.time()
        r = discover_tiered(
            query=query,
            tiers=TIER_PRESETS["default"],
            limit=limit,
            clients=self.clients,
            circuit=self.circuit,
            tier_timeout_s=5.0,
        )
        lat = {"fast_discovery_ms": round((time.time() - t0) * 1000, 1)}
        return SearchResult(query=query, mode_used="fast",
                            candidates=r.candidates, tiers=r.tiers,
                            latency=lat)

    # ---- full path -------------------------------------------------------

    def _full(self, query: str, limit: int) -> SearchResult:
        lat: dict[str, float] = {}
        t0 = time.time()
        u = understand_query(query, chat=self.chat)
        lat["understand_ms"] = round((time.time() - t0) * 1000, 1)

        t0 = time.time()
        tiers_trace: list[dict[str, Any]] = []
        # P15: subqueries run CONCURRENTLY (each is an independent
        # discover_tiered with its own tier chain; serial for-loop was the
        # dominant wall-clock term — 3-4 subqueries x 3-12s each). Results
        # are merged in subquery order so the trace stays deterministic.
        from concurrent.futures import ThreadPoolExecutor

        def _run_sq(sq):
            # semantic channel ALWAYS leads (A6 verdict: the only channel
            # that crosses the vocabulary gap); keyword prefs follow it
            order = ["sciverse-semantic"] + (
                sq.get("prefer") or TIER_PRESETS.get(
                    u.domain, TIER_PRESETS["default"]))
            return discover_tiered(
                query=sq["q"], tiers=order, limit=limit,
                clients=self.clients, circuit=self.circuit,
                tier_timeout_s=12.0,
            )
        with ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(_run_sq, u.subqueries))
        merged: list[SourceCandidate] = []
        for sq, r in zip(u.subqueries, results):
            merged.extend(r.candidates)
            tiers_trace.extend([{"subquery": sq["q"], **t} for t in r.tiers])
        merged = _dedupe(merged)
        # no pre-cut before rerank: with 3-4 subqueries x limit candidates
        # the union is ~30-40; cutting to limit*2 pre-rerank can drop a gold
        # paper ranked 21st by source order that the embed would have lifted
        merged = merged[: max(64, limit * 4)]
        lat["discovery_ms"] = round((time.time() - t0) * 1000, 1)

        t0 = time.time()
        rr = rerank_candidates(query, merged, embed=self.embed)
        lat["rerank_ms"] = round((time.time() - t0) * 1000, 1)
        return SearchResult(query=query, mode_used="full",
                            candidates=rr.candidates[:limit],
                            scores=rr.scores[:limit], understood=u,
                            latency=lat, tiers=tiers_trace,
                            reranked=rr.embedded, pool=merged)

    # ---- entry -----------------------------------------------------------

    def _agent(self, query: str, limit: int) -> SearchResult:
        """Agent-loop path (P14, 2026-09-25). The caller is a ReAct model
        whose query is ALREADY refined keyword vocabulary — running the A3
        understand layer on it again is pure waste (measured: every
        search_papers call in the behavior probe burned 5s re-decomposing
        an already-decomposed query, 30-60s/call total, 15min/question).
        Path: semantic channel single-shot first; only if the result is
        THIN does one keyword tier run. No A3, no multi-subquery fan-out —
        the loop itself iterates (that's its job)."""
        from retrieval.acquisition import discover_tiered
        t0 = time.time()
        # parallel fan-out (P15): all three tiers fire at t=0, so the
        # timeout is a CEILING not an additive cost — 15s costs nothing
        # over 8s when the semantic layer answers in 5s (early stop on
        # quota), and rescues the evening-congestion window where Sciverse
        # alone takes 10-13s (measured 2026-09-26)
        r = discover_tiered(
            query=query,
            tiers=["sciverse-semantic", "openalex", "crossref"],
            limit=limit,
            clients=self.clients,
            circuit=self.circuit,
            tier_timeout_s=15.0,
        )
        lat = {"agent_discovery_ms": round((time.time() - t0) * 1000, 1)}
        return SearchResult(query=query, mode_used="agent",
                            candidates=r.candidates, tiers=r.tiers,
                            latency=lat)

    def search(self, query: str, mode: str = "auto",
               limit: int | None = None) -> SearchResult:
        limit = limit or self.limit
        q = (query or "").strip()
        if not q:
            return SearchResult(query="", mode_used="fast")

        if mode in ("fast", "auto"):
            cached = self.query_cache.get(q)
            if cached is not None:
                cached = SearchResult(
                    query=q, mode_used="fast_cache",
                    candidates=cached["candidates"],
                    scores=cached.get("scores", []),
                    latency={"cache_hit_ms": 0.5},
                )
                return cached

        if mode == "fast":
            res = self._fast(q, limit)
        elif mode == "full":
            res = self._full(q, limit)
        elif mode == "agent":
            res = self._agent(q, limit)
        else:  # auto: fast first, escalate on thin results
            res = self._fast(q, limit)
            thin = len(res.candidates) < min(3, limit)
            if thin or not res.candidates:
                full = self._full(q, limit)
                full.latency["fast_attempt_ms"] = res.latency.get(
                    "fast_discovery_ms", 0.0)
                res = full
        # fill caches
        try:
            if res.candidates:
                self.query_cache.put(q, {
                    "candidates": res.candidates[:limit],
                    "scores": res.scores[:limit] if res.scores else [],
                })
                if self.doi_cache:
                    for c in res.candidates:
                        self.doi_cache.put(c)
        except Exception:
            pass
        return res

    def close(self) -> None:
        if self.doi_cache:
            self.doi_cache.flush()

    def stats(self) -> dict:
        out = {"query_cache": self.query_cache.stats(),
               "circuit": self.circuit.snapshot()}
        return out
