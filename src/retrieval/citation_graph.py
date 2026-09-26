"""Citation graph service (P4-A G4, 2026-09-24).

The design note assumed the corpus manifest carried parsed reference
lists — it does not (checked: 430 entries, zero refs fields). The
in-corpus citation structure is the compiled genealogy (971 lineage
edges), already served by the existing lineage tool. What is actually
missing is the EXTERNAL walk: "this paper cites X / is cited by Y /
which of those are in our corpus" at query time.

This module provides it over S2 (both directions; OpenAlex only does
outgoing) with an OpenAlex fallback for references, DOI-keyed on-disk
caching, and corpus annotation: every edge endpoint is matched against
a caller-supplied manifest (paper_id, doi, title) so the in-corpus
subset is visible without another lookup.

Latency shape: one S2 id-resolve + one paginated edge call per
direction — seconds, cache-backed on repeat. Part of the SQA2 broker's
citation_graph tool.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from retrieval.sources import (
    CrossrefClient,
    OpenAlexClient,
    SemanticScholarClient,
    SourceCandidate,
    normalize_doi,
)


def _norm_title(t: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


@dataclass(frozen=True)
class CitationEdge:
    direction: str          # "out" (this paper cites X) | "in" (X cites this paper)
    doi: str | None
    title: str | None
    year: int | None
    external_id: str | None          # S2 paperId
    in_corpus_paper_id: str | None   # manifest match, when corpus given
    is_influential: bool | None = None  # S2's own citation-weight signal


@dataclass(frozen=True)
class CitationGraphResult:
    anchor: dict[str, Any]           # {doi, title, year, s2_paper_id?}
    edges: list[CitationEdge] = field(default_factory=list)
    n_in_corpus: int = 0
    latency_ms: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class CitationGraphService:
    def __init__(
        self,
        *,
        s2_client: SemanticScholarClient | None = None,
        openalex_client: OpenAlexClient | None = None,
        crossref_client: CrossrefClient | None = None,
        cache_path: str | Path | None = None,
        corpus_manifest: list[dict[str, Any]] | None = None,
    ):
        self.s2 = s2_client or SemanticScholarClient()
        self.oa = openalex_client or OpenAlexClient()
        self.cr = crossref_client or CrossrefClient()
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(
                    self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}
        # manifest lookup tables
        self._by_doi: dict[str, dict] = {}
        self._by_title: dict[str, dict] = {}
        for row in corpus_manifest or []:
            if row.get("doi"):
                self._by_doi[normalize_doi(str(row["doi"]))] = row
            if row.get("title"):
                self._by_title[_norm_title(str(row["title"]))] = row

    # ---- corpus annotation ------------------------------------------------

    def _corpus_pid(self, doi: str | None, title: str | None) -> str | None:
        if doi:
            row = self._by_doi.get(normalize_doi(doi))
            if row:
                return row.get("paper_id")
        if title:
            row = self._by_title.get(_norm_title(title))
            if row:
                return row.get("paper_id")
        return None

    # ---- cache ------------------------------------------------------------

    def _cache_key(self, doi: str, direction: str, max_total: int) -> str:
        return f"{normalize_doi(doi)}|{direction}|{max_total}"

    def _load(self, key: str) -> dict | None:
        hit = self._cache.get(key)
        if not hit:
            return None
        if time.time() - hit.get("ts", 0) > 7 * 86400:   # 7-day TTL
            return None
        return hit

    def _store(self, key: str, value: dict) -> None:
        self._cache[key] = {**value, "ts": time.time()}
        if self.cache_path:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.cache_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self._cache, ensure_ascii=False),
                               encoding="utf-8")
                os.replace(tmp, self.cache_path)
            except Exception:
                pass

    # ---- anchor resolution -------------------------------------------------

    def _resolve_anchor(self, doi: str | None, title: str | None) -> dict:
        """DOI -> metadata via OpenAlex; title -> DOI via Crossref/OpenAlex
        search. Returns the anchor dict (never raises)."""
        if doi:
            d = normalize_doi(doi)
            cand = self.oa.fetch_by_doi(d)
            if cand.status == "ready":
                return {"doi": d, "title": cand.title, "year": cand.year,
                        "openalex_id": cand.source_record_id}
            return {"doi": d, "title": title, "year": None,
                    "openalex_id": None}
        if title:
            for client in (self.oa, self.cr):
                rows = client.search_title(title, limit=3)
                for r in rows:
                    if r.status == "ready" and r.normalized_doi:
                        return {"doi": r.normalized_doi, "title": r.title,
                                "year": r.year,
                                "openalex_id": r.source_record_id
                                if r.source_name == "openalex" else None}
        return {"doi": None, "title": title, "year": None,
                "openalex_id": None}

    # ---- main entry ----------------------------------------------------------

    def graph(self, *, doi: str | None = None, title: str | None = None,
              direction: str = "both", max_total: int = 50) -> CitationGraphResult:
        lat: dict[str, float] = {}
        errors: list[str] = []
        t0 = time.time()
        anchor = self._resolve_anchor(doi, title)
        lat["resolve_ms"] = round((time.time() - t0) * 1000, 1)
        if not anchor.get("doi"):
            errors.append("anchor unresolved (no DOI found)")
            return CitationGraphResult(anchor=anchor, errors=errors,
                                       latency_ms=lat)
        edges: list[CitationEdge] = []
        dirs = ["out", "in"] if direction == "both" else [direction]
        for d in dirs:
            key = self._cache_key(anchor["doi"], d, max_total)
            cached = self._load(key)
            if cached is not None:
                rows = cached["edges"]
                lat[f"{d}_cache_ms"] = 0.5
            else:
                t0 = time.time()
                rows = self._fetch_direction(anchor["doi"], d, max_total)
                lat[f"{d}_ms"] = round((time.time() - t0) * 1000, 1)
                self._store(key, {"edges": rows})
            edges.extend(CitationEdge(
                direction=d, doi=r.get("doi"), title=r.get("title"),
                year=r.get("year"), external_id=r.get("external_id"),
                in_corpus_paper_id=self._corpus_pid(r.get("doi"),
                                                    r.get("title")),
                is_influential=r.get("is_influential"),
            ) for r in rows)
        n_in = sum(1 for e in edges if e.in_corpus_paper_id)
        return CitationGraphResult(anchor=anchor, edges=edges,
                                   n_in_corpus=n_in, latency_ms=lat,
                                   errors=errors)

    def _fetch_direction(self, doi: str, direction: str,
                         max_total: int) -> list[dict]:
        """S2 primary (both directions); OpenAlex fallback for outgoing."""
        try:
            # S2 accepts DOI: prefix directly as the paper id
            if direction == "out":
                rows = self.s2.fetch_references(f"DOI:{doi}",
                                                max_total=max_total)
                return [{"doi": r.get("ref_doi"), "title": r.get("ref_title"),
                         "year": r.get("ref_year"),
                         "external_id": r.get("source_ref_id"),
                         "is_influential":
                             (r.get("raw") or {}).get("isInfluential")}
                        for r in rows]
            rows = self.s2.fetch_citations(f"DOI:{doi}", max_total=max_total)
            return [{"doi": r.get("citing_doi"),
                     "title": r.get("citing_title"),
                     "year": r.get("citing_year"),
                     "external_id": r.get("source_ref_id"),
                     "is_influential":
                         (r.get("raw") or {}).get("isInfluential")}
                    for r in rows]
        except Exception as exc:
            if direction == "out":  # OpenAlex fallback (outgoing only)
                try:
                    rows = self.oa.fetch_work_references(doi,
                                                         resolve_metadata=False)
                    return [{"doi": r.get("ref_doi"),
                             "title": r.get("ref_title"),
                             "year": r.get("ref_year"),
                             "external_id": r.get("ref_openalex_id"),
                             "is_influential": None} for r in rows]
                except Exception as exc2:
                    return [{"error": f"s2: {exc}; openalex: {exc2}"}]
            return [{"error": f"s2: {exc}"}]
