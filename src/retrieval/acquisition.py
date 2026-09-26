from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from retrieval.registry import (
    CrossPaperDedupeConflict,
    LibraryRegistry,
    normalize_arxiv_id,
    normalize_doi,
    parse_http_status,
)
from retrieval.sources import (
    ArxivClient,
    CrossrefClient,
    LocalDoiArchive,
    LocalDoiArchiveHit,
    OpenAlexClient,
    SciverseClient,
    SemanticScholarClient,
    SourceAdapterError,
    SourceCandidate,
)
from retrieval._see_upstream import download_bytes


@dataclass(frozen=True)
class AcquisitionResult:
    status: str
    paper: dict[str, Any]
    pdf_asset: dict[str, Any] | None
    source_candidates: list[SourceCandidate] = field(default_factory=list)
    dedupe: str = "none"
    message: str | None = None
    conflict: dict[str, Any] | None = None
    blocking_reason: str | None = None
    recommended_next_action: str | None = None
    stage_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SourceDiscoveryResult:
    query_kind: str
    candidates: list[SourceCandidate] = field(default_factory=list)
    local_archive_hit: LocalDoiArchiveHit | None = None


@dataclass(frozen=True)
class TieredDiscoveryResult:
    """discover_tiered output: candidates + the degradation trace (which
    tier served, which were circuit-skipped or degraded) for the latency
    report of the A5 layer."""
    query: str
    candidates: list[SourceCandidate]
    tiers: list[dict[str, Any]] = field(default_factory=list)


# Domain-routing presets. 2026-09-25 restructure (S2-slot verdict): the
# semantic channel leads everywhere — Sciverse agentic-search is the only
# channel that crosses the A6 vocabulary gap (gold papers linked to the
# question by specific findings, not by shared title keywords); keyword
# sources follow as complements and fallbacks so a Sciverse outage degrades
# to the measured keyword floor instead of zero. S2 keeps no search slot
# (anonymous-pool 429s burned minutes in tier timeouts); it remains the
# citation-graph provider in library/citation_graph.py.
TIER_PRESETS: dict[str, list[str]] = {
    "cs": ["sciverse-semantic", "arxiv", "openalex", "crossref"],
    "default": ["sciverse-semantic", "openalex", "crossref"],
    "bio": ["sciverse-semantic", "openalex", "crossref", "sciverse"],
}


def discover_tiered(
    *,
    query: str,
    tiers: list[str] | None = None,
    domain: str = "default",
    limit: int = 10,
    clients: dict[str, Any] | None = None,
    circuit: Any = None,
    stop_after_hits: int | None = None,
    tier_timeout_s: float = 20.0,
) -> TieredDiscoveryResult:
    """Tiered discovery with circuit breaking and degradation (A1).

    Semantics per tier, in order:
      - circuit-open source -> skipped (recorded in the trace)
      - request errors (failed/blocked candidates) OR tier_timeout_s
        exceeded -> circuit failure + degrade to the next tier
        (measured: S2's anonymous pool can burn minutes in 429
        Retry-After loops — an unbounded tier defeats the <15s path)
      - honest empty result -> degrade WITHOUT tripping the circuit
      - >= stop_after_hits (default: limit) ready candidates -> stop

    Deterministic, no LLM. The existing fan-out discover_by_query stays
    untouched for batch scenes; this is the single-query/interactive path.
    """
    from concurrent.futures import ThreadPoolExecutor

    from retrieval.circuit import DEFAULT_CIRCUIT

    query_text = (query or "").strip()
    if not query_text:
        raise AcquisitionError("discovery query 不能为空。")
    circ = circuit or DEFAULT_CIRCUIT
    cl = clients or {}
    _sciverse = cl.get("sciverse") or SciverseClient()
    cl = {
        "arxiv": cl.get("arxiv") or ArxivClient(),
        "s2": cl.get("s2") or SemanticScholarClient(),
        "openalex": cl.get("openalex") or OpenAlexClient(),
        "crossref": cl.get("crossref") or CrossrefClient(),
        "sciverse": _sciverse,
        # semantic channel shares the SciverseClient (same token/transport,
        # distinct endpoint); injectable for tests
        "sciverse-semantic": cl.get("sciverse-semantic") or _sciverse,
    }
    order = tiers or TIER_PRESETS.get(domain, TIER_PRESETS["default"])
    target = stop_after_hits if stop_after_hits is not None else limit

    candidates: list[SourceCandidate] = []
    trace: list[dict[str, Any]] = []
    # P15 parallel fan-out (2026-09-26): tiers launch CONCURRENTLY — the
    # serial chain (semantic -> only-if-thin -> keyword) wasted the keyword
    # layers' entire latency whenever the semantic layer was slow or
    # circuit-open. Launch discipline: all circuit-allowed tiers are
    # submitted up front; results are collected IN TIER ORDER (semantic
    # hits still win the dedup-first position), stopping as soon as the
    # target is met — later futures are simply never collected. Timed-out
    # or uncollected calls are LEFT RUNNING: executor.shutdown(wait=False)
    # never blocks the caller on a stuck source.
    # Sciverse rate-limit evidence: 8 back-to-back calls zero failures;
    # A6 run2 tripped one throttle only after hundreds of sequential
    # calls — 3-4 concurrent sources is well inside budget.
    ex = ThreadPoolExecutor(max_workers=6)
    try:
        active: list[tuple[str, Any, float, Any]] = []  # (source, client, t0, fut)
        for source in order:
            source = source.lower()
            client = cl.get(source)
            if client is None:
                trace.append({"source": source, "outcome": "unknown_source"})
                continue
            if not circ.allow(source):
                trace.append({"source": source, "outcome": "circuit_skipped",
                              "circuit_state": circ.state(source)})
                continue

            def _call(client=client, source=source):
                if source == "arxiv":
                    return client.search(query_text, limit=limit)
                if source == "sciverse-semantic":
                    return client.semantic_search(query_text, limit=limit)
                return client.search_title(query_text, limit=limit)

            active.append((source, client, time.time(), ex.submit(_call)))
        for source, client, t0, fut in active:
            try:
                # deadline counts from THIS tier's submit time — a tier
                # sitting behind a slow earlier tier must not inherit its
                # wait (measured: openalex fut.result waited 8s of the
                # semantic tier + its own 8s = 16s "timeout")
                remaining = max(0.05, tier_timeout_s - (time.time() - t0))
                rows = fut.result(timeout=remaining)
            except TimeoutError:
                circ.record(source, ok=False)
                trace.append({"source": source, "outcome": "timeout",
                              "latency_s": round(time.time() - t0, 3)})
                continue  # degrade; the stuck call keeps running detached
            except Exception as exc:  # clients normally convert errors; belt
                rows = [SourceCandidate(
                    source_name=source, query_kind="search",
                    status="failed",
                    error_summary=f"{type(exc).__name__}: {exc}")]
            dt = round(time.time() - t0, 3)
            ready = [r for r in rows if r.status == "ready"]
            errored = [r for r in rows if r.status in ("failed", "blocked")]
            if errored:
                circ.record(source, ok=False)
                trace.append({"source": source, "outcome": "error",
                              "error": errored[0].error_summary,
                              "latency_s": dt})
                continue  # degrade
            circ.record(source, ok=True)
            if not ready:
                trace.append({"source": source, "outcome": "empty",
                              "latency_s": dt})
                continue  # degrade without tripping
            candidates.extend(ready)
            trace.append({"source": source, "outcome": "hit",
                          "n": len(ready), "latency_s": dt})
            if len(candidates) >= target:
                break
    finally:
        ex.shutdown(wait=False)
    # Dedup across tiers (first occurrence wins). DOI when present;
    # title-normalized otherwise — the semantic channel carries no DOI, and
    # a doc_id key would never collide with a DOI key for the same paper,
    # so the title fallback also merges semantic hits against keyword tiers.
    seen: set[str] = set()
    unique: list[SourceCandidate] = []
    for c in candidates:
        if c.normalized_doi:
            key = f"doi:{c.normalized_doi}"
        else:
            key = _norm_key(str(c.title or "")) or \
                (c.source_record_id or f"{c.title}@{c.source_name}")
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return TieredDiscoveryResult(query=query_text, candidates=unique[:limit],
                                 tiers=trace)


def _norm_title_key(t: str) -> str:
    import re as _re
    return _re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def _norm_key(title: str) -> str:
    k = _norm_title_key(title)
    return f"title:{k}" if k else ""


def dedupe_conflict_result(
    *,
    paper: dict[str, Any],
    candidates: list[SourceCandidate],
    diagnostic: dict[str, Any],
    message: str | None = None,
) -> AcquisitionResult:
    return AcquisitionResult(
        status="dedupe_cross_paper_conflict",
        paper=paper,
        pdf_asset=None,
        source_candidates=candidates,
        dedupe="cross_paper_conflict",
        message=message or "PDF/evidence 已存在于另一条 paper；请先执行 merge dry-run 并确认 canonical 合并。",
        conflict=diagnostic,
        blocking_reason=diagnostic.get("reason") or "dedupe_asset_on_other_paper",
        recommended_next_action="merge_or_rebind_asset",
        stage_results=[diagnostic],
    )


def discover_by_query(
    *,
    query: str,
    sources: list[str] | None = None,
    mode: str = "keyword",
    filters: dict[str, Any] | None = None,
    openalex_client: OpenAlexClient | None = None,
    crossref_client: CrossrefClient | None = None,
    sciverse_client: SciverseClient | None = None,
    limit: int = 10,
) -> SourceDiscoveryResult:
    query_text = query.strip()
    if not query_text:
        raise AcquisitionError("discovery query 不能为空。")
    if limit < 1:
        raise AcquisitionError("discovery limit 必须大于 0。")

    requested_sources = {source.lower() for source in (sources or ["openalex", "crossref", "sciverse"])}
    candidates: list[SourceCandidate] = []
    if "openalex" in requested_sources:
        candidates.extend((openalex_client or OpenAlexClient()).search_title(query_text, limit=limit))
    if "crossref" in requested_sources:
        candidates.extend((crossref_client or CrossrefClient()).search_title(query_text, limit=limit))
    if "sciverse" in requested_sources:
        candidates.extend((sciverse_client or SciverseClient()).search_title(query_text, limit=limit))

    candidates = apply_discovery_filters(candidates, filters or {})
    if not candidates:
        candidates.append(
            SourceCandidate(
                source_name="local_query",
                query_kind="discovery",
                status="not_found",
                title=query_text,
                error_summary="未发现来源候选。",
            )
        )
    return SourceDiscoveryResult(query_kind=mode or "keyword", candidates=candidates)


def acquire_by_doi(
    *,
    doi: str,
    registry: LibraryRegistry,
    local_doi_index: Path | None = None,
    local_doi_archive_root: Path | None = None,
    sources: list[str] | None = None,
    metadata_candidates: list[SourceCandidate] | None = None,
    openalex_client: OpenAlexClient | None = None,
    crossref_client: CrossrefClient | None = None,
    sciverse_client: SciverseClient | None = None,
    download_pdf: Callable[[str, int], bytes] | None = None,
    download_timeout_seconds: int = 60,
) -> AcquisitionResult:
    normalized = normalize_doi(doi)
    if not normalized:
        raise AcquisitionError(f"DOI 无效：{doi}")

    existing = registry.find_paper_by_doi(normalized)
    if existing:
        assets = registry.list_pdf_assets(existing["paper_id"])
        if assets:
            return AcquisitionResult(
                status="ready",
                paper=existing,
                pdf_asset=assets[0],
                source_candidates=registry_candidates_to_source_candidates(
                    registry.list_source_candidates(existing["paper_id"])
                ),
                dedupe="existing",
                message="已复用本地 registry 中的 PDF asset。",
            )
        dedupe_diagnostic = registry.find_duplicate_asset_diagnostics(existing["paper_id"])
        if dedupe_diagnostic:
            return dedupe_conflict_result(
                paper=existing,
                candidates=metadata_candidates or [],
                diagnostic=dedupe_diagnostic,
            )

    discovery = discover_by_doi(
        doi=normalized,
        local_doi_index=local_doi_index,
        local_doi_archive_root=local_doi_archive_root,
        sources=sources,
        metadata_candidates=metadata_candidates,
        openalex_client=openalex_client,
        crossref_client=crossref_client,
        sciverse_client=sciverse_client,
    )
    candidates = discovery.candidates
    best_metadata = first_ready_candidate(candidates)
    identifiers = source_candidate_identifiers(candidates)
    existing_paper_ids = {
        str(item["paper_id"]) for item in registry.list_papers(include_archived=True)
    }
    paper = registry.upsert_paper(
        doi=normalized,
        title=best_metadata.title if best_metadata else None,
        year=best_metadata.year if best_metadata else None,
        venue=best_metadata.venue if best_metadata else None,
        identity_confidence=best_metadata.candidate_score if best_metadata else None,
        identifiers=identifiers,
    )
    reused_identity = str(paper["paper_id"]) in existing_paper_ids
    register_source_candidates(registry, paper["paper_id"], candidates)

    if local_doi_index and local_doi_archive_root and discovery.local_archive_hit:
        archive = LocalDoiArchive(
            index_path=local_doi_index,
            archive_root=local_doi_archive_root,
        )
        try:
            with tempfile.TemporaryDirectory(prefix="scievo_doi_") as temp_dir:
                temp_pdf = Path(temp_dir) / "paper.pdf"
                archive.extract_pdf(discovery.local_archive_hit, temp_pdf)
                try:
                    asset = registry.store_pdf_asset(
                        paper_id=paper["paper_id"],
                        source_path=temp_pdf,
                        source_kind="local_doi_archive",
                        source_uri=discovery.local_archive_hit.source_uri,
                        license="unknown",
                        open_access_status="unknown",
                    )
                except CrossPaperDedupeConflict as exc:
                    conflict = {
                        **exc.to_dict(),
                        "reason": "dedupe_asset_on_other_paper",
                        "owner_readiness": {
                            "pdf_asset_count": len(registry.list_pdf_assets(exc.asset_owner_paper_id)),
                            "processing_job_count": len(registry.list_processing_jobs(exc.asset_owner_paper_id)),
                            "evidence_count": registry.evidence_coverage(exc.asset_owner_paper_id)["total"],
                        },
                    }
                    return dedupe_conflict_result(
                        paper=paper,
                        candidates=candidates,
                        diagnostic=conflict,
                        message=str(exc),
                    )
            registry.record_provenance_event(
                action="acquire_pdf",
                source="local_doi_archive",
                inputs={"doi": normalized, "source_uri": discovery.local_archive_hit.source_uri},
                outputs={"paper_id": paper["paper_id"], "asset_id": asset["asset_id"]},
            )
            return AcquisitionResult(
                status="ready",
                paper=paper,
                pdf_asset=asset,
                source_candidates=candidates,
                dedupe="existing_identity" if reused_identity else "created",
                message="已从本地 DOI archive 获取 PDF。",
            )
        except SourceAdapterError as exc:
            failed = SourceCandidate(
                source_name="local_doi_archive",
                query_kind="doi",
                status="failed",
                normalized_doi=normalized,
                title=paper.get("title"),
                error_summary=str(exc),
            )
            candidates.append(failed)
            register_source_candidates(registry, paper["paper_id"], [failed])

    dedupe_diagnostic = registry.find_duplicate_asset_diagnostics(paper["paper_id"])
    if dedupe_diagnostic:
        return dedupe_conflict_result(
            paper=paper,
            candidates=candidates,
            diagnostic=dedupe_diagnostic,
        )

    if download_pdf:
        remote_candidates = safe_remote_pdf_candidates(registry.list_source_candidates(paper["paper_id"]))
        if remote_candidates:
            return acquire_remote_pdf_candidates(
                registry=registry,
                paper=paper,
                candidates=remote_candidates,
                download_pdf=download_pdf,
                download_timeout_seconds=download_timeout_seconds,
                dedupe="existing_identity" if reused_identity else "created",
            )

    return AcquisitionResult(
        status="metadata_only",
        paper=paper,
        pdf_asset=None,
        source_candidates=candidates,
        dedupe="existing_identity" if reused_identity else "none",
        message="已登记 metadata，但未找到可用 PDF。",
        blocking_reason="missing_pdf",
        recommended_next_action="find_pdf_or_sciverse",
        stage_results=[
            pdf_acquire_trace(
                registry=registry,
                paper_id=paper["paper_id"],
                status="metadata_only",
                blocking_reason="missing_pdf",
                recommended_next_action="find_pdf_or_sciverse",
            )
        ],
    )


def acquire_by_title(
    *,
    title: str,
    registry: LibraryRegistry,
    sources: list[str] | None = None,
    metadata_candidates: list[SourceCandidate] | None = None,
    openalex_client: OpenAlexClient | None = None,
    crossref_client: CrossrefClient | None = None,
    sciverse_client: SciverseClient | None = None,
) -> AcquisitionResult:
    query = title.strip()
    if not query:
        raise AcquisitionError("题名不能为空。")

    discovery = discover_by_title(
        title=query,
        sources=sources,
        metadata_candidates=metadata_candidates,
        openalex_client=openalex_client,
        crossref_client=crossref_client,
        sciverse_client=sciverse_client,
    )
    candidates = discovery.candidates
    best_metadata = first_ready_candidate(candidates)
    identifiers = source_candidate_identifiers(candidates)
    existing_paper_ids = {
        str(item["paper_id"]) for item in registry.list_papers(include_archived=True)
    }
    paper = registry.upsert_paper(
        doi=best_metadata.normalized_doi if best_metadata else None,
        title=best_metadata.title if best_metadata and best_metadata.title else query,
        year=best_metadata.year if best_metadata else None,
        venue=best_metadata.venue if best_metadata else None,
        identity_confidence=best_metadata.candidate_score if best_metadata else None,
        identifiers=identifiers,
    )
    reused_identity = str(paper["paper_id"]) in existing_paper_ids
    register_source_candidates(registry, paper["paper_id"], candidates)
    return AcquisitionResult(
        status="candidate_review_required",
        paper=paper,
        pdf_asset=None,
        source_candidates=candidates,
        dedupe="existing_identity" if reused_identity else "none",
        message="已登记题名候选，需要人工确认后再获取 PDF。",
    )


def confirm_source_candidate(
    *,
    paper_id: str,
    candidate_id: str,
    registry: LibraryRegistry,
    local_doi_index: Path | None = None,
    local_doi_archive_root: Path | None = None,
    openalex_client: OpenAlexClient | None = None,
    crossref_client: CrossrefClient | None = None,
    sciverse_client: SciverseClient | None = None,
    download_pdf: Callable[[str, int], bytes] | None = None,
    download_timeout_seconds: int = 60,
) -> AcquisitionResult:
    paper = registry.get_paper(paper_id)
    if not paper:
        raise AcquisitionError(f"paper not found: {paper_id}")

    candidate = registry.get_source_candidate(candidate_id)
    if not candidate:
        raise AcquisitionError(f"source candidate not found: {candidate_id}")
    if candidate.get("paper_id") != paper_id:
        raise AcquisitionError("source candidate does not belong to this paper.")

    normalized = normalize_doi(candidate.get("normalized_doi"))
    if not normalized:
        raise AcquisitionError("候选没有 DOI，暂不能自动确认获取 PDF。")

    dedupe_diagnostic = registry.find_duplicate_asset_diagnostics(paper_id)
    if dedupe_diagnostic:
        return dedupe_conflict_result(
            paper=paper,
        candidates=[
                source_candidate_from_registry_row(candidate, paper=paper)
            ],
            diagnostic=dedupe_diagnostic,
        )

    candidate_rows = registry.list_source_candidates(paper_id)
    selected_row = next(
        (row for row in candidate_rows if row.get("candidate_id") == candidate_id),
        candidate,
    )
    safe_rows = safe_remote_pdf_candidates(candidate_rows)
    selected_is_safe_remote = any(row.get("candidate_id") == candidate_id for row in safe_rows)
    needs_arxiv_synthesis = bool(arxiv_candidate_from_doi(normalized)) and not any(
        is_arxiv_pdf_candidate(row) for row in safe_rows
    )
    should_try_local_archive_first = bool(local_doi_index and local_doi_archive_root)
    if selected_is_safe_remote and not needs_arxiv_synthesis and not should_try_local_archive_first:
        return acquire_remote_pdf_candidates(
            registry=registry,
            paper=paper,
            candidates=safe_rows,
            download_pdf=download_pdf or download_bytes,
            download_timeout_seconds=download_timeout_seconds,
            dedupe="existing_identity",
        )

    return acquire_by_doi(
        doi=normalized,
        registry=registry,
        local_doi_index=local_doi_index,
        local_doi_archive_root=local_doi_archive_root,
        metadata_candidates=[
            source_candidate_from_registry_row(candidate, paper=paper)
        ],
        openalex_client=openalex_client,
        crossref_client=crossref_client,
        sciverse_client=sciverse_client,
        download_pdf=download_pdf or download_bytes,
        download_timeout_seconds=download_timeout_seconds,
    )


def acquire_remote_pdf_candidate(
    *,
    registry: LibraryRegistry,
    paper: dict[str, Any],
    candidate: dict[str, Any],
    fallback_candidates: list[dict[str, Any]] | None = None,
    download_pdf: Callable[[str, int], bytes],
    download_timeout_seconds: int = 60,
    dedupe: str = "none",
) -> AcquisitionResult:
    candidates = [candidate]
    seen = {str(candidate.get("candidate_id"))}
    for fallback in fallback_candidates or []:
        fallback_id = str(fallback.get("candidate_id"))
        if fallback_id not in seen:
            candidates.append(fallback)
            seen.add(fallback_id)
    if len(candidates) > 1:
        return acquire_remote_pdf_candidates(
            registry=registry,
            paper=paper,
            candidates=candidates,
            download_pdf=download_pdf,
            download_timeout_seconds=download_timeout_seconds,
            dedupe=dedupe,
            preserve_order=True,
        )
    return _acquire_single_remote_pdf_candidate(
        registry=registry,
        paper=paper,
        candidate=candidate,
        download_pdf=download_pdf,
        download_timeout_seconds=download_timeout_seconds,
        dedupe=dedupe,
    )


def acquire_remote_pdf_candidates(
    *,
    registry: LibraryRegistry,
    paper: dict[str, Any],
    candidates: list[dict[str, Any]],
    download_pdf: Callable[[str, int], bytes],
    download_timeout_seconds: int = 60,
    dedupe: str = "none",
    preserve_order: bool = False,
) -> AcquisitionResult:
    ordered_candidates = candidates if preserve_order else safe_remote_pdf_candidates(candidates)
    if not ordered_candidates:
        return AcquisitionResult(
            status="metadata_only",
            paper=paper,
            pdf_asset=None,
            source_candidates=[],
            dedupe=dedupe,
            message="没有可自动下载的安全远程 PDF candidate。",
            blocking_reason="missing_pdf",
            recommended_next_action="find_pdf_or_sciverse",
            stage_results=[],
        )

    stage_results: list[dict[str, Any]] = []
    last_result: AcquisitionResult | None = None
    for candidate in ordered_candidates:
        result = _acquire_single_remote_pdf_candidate(
            registry=registry,
            paper=paper,
            candidate=candidate,
            download_pdf=download_pdf,
            download_timeout_seconds=download_timeout_seconds,
            dedupe=dedupe,
        )
        last_result = result
        stage_results.extend(result.stage_results)
        if result.status in {"ready", "dedupe_cross_paper_conflict"}:
            return AcquisitionResult(
                status=result.status,
                paper=result.paper,
                pdf_asset=result.pdf_asset,
                source_candidates=result.source_candidates,
                dedupe=result.dedupe,
                message=result.message,
                conflict=result.conflict,
                blocking_reason=result.blocking_reason,
                recommended_next_action=result.recommended_next_action,
                stage_results=stage_results,
            )

    assert last_result is not None
    return AcquisitionResult(
        status=last_result.status,
        paper=last_result.paper,
        pdf_asset=None,
        source_candidates=[
            source_candidate_from_registry_row(candidate, paper=paper)
            for candidate in ordered_candidates
        ],
        dedupe=last_result.dedupe,
        message=last_result.message,
        conflict=last_result.conflict,
        blocking_reason=last_result.blocking_reason,
        recommended_next_action=last_result.recommended_next_action,
        stage_results=stage_results,
    )


def _acquire_single_remote_pdf_candidate(
    *,
    registry: LibraryRegistry,
    paper: dict[str, Any],
    candidate: dict[str, Any],
    download_pdf: Callable[[str, int], bytes],
    download_timeout_seconds: int = 60,
    dedupe: str = "none",
) -> AcquisitionResult:
    trace = remote_pdf_acquire_trace(candidate)
    pdf_url = candidate.get("candidate_pdf_url") or candidate.get("pdf_url")
    if not pdf_url:
        trace.update(
            {
                "status": "metadata_only",
                "download_status": "skipped",
                "asset_write_status": "skipped",
                "blocking_reason": "missing_pdf",
                "recommended_next_action": "find_pdf_or_sciverse",
            }
        )
        return AcquisitionResult(
            status="metadata_only",
            paper=paper,
            pdf_asset=None,
            source_candidates=[source_candidate_from_registry_row(candidate, paper=paper)],
            dedupe=dedupe,
            message="候选没有 PDF URL，无法落库。",
            blocking_reason="missing_pdf",
            recommended_next_action="find_pdf_or_sciverse",
            stage_results=[trace],
        )

    if candidate.get("asset_policy") not in {"oa_auto_allowed", "arxiv_auto_allowed"}:
        trace.update(
            {
                "status": "metadata_only",
                "download_status": "blocked_by_policy",
                "asset_write_status": "skipped",
                "blocking_reason": "pdf_policy_requires_review",
                "recommended_next_action": "register_candidate_only",
            }
        )
        return AcquisitionResult(
            status="metadata_only",
            paper=paper,
            pdf_asset=None,
            source_candidates=[source_candidate_from_registry_row(candidate, paper=paper)],
            dedupe=dedupe,
            message="候选 PDF 策略需要人工确认，未自动下载。",
            blocking_reason="pdf_policy_requires_review",
            recommended_next_action="register_candidate_only",
            stage_results=[trace],
        )

    try:
        pdf_bytes = download_pdf(str(pdf_url), download_timeout_seconds)
    except Exception as exc:  # noqa: BLE001 - trace external provider failure.
        error_summary = str(exc)
        next_action = remote_pdf_failure_next_action(error_summary)
        failure_audit = remote_pdf_failure_audit(
            error_summary=error_summary,
            pdf_url=str(pdf_url),
            candidate=candidate,
        )
        trace.update(
            {
                "status": "failed",
                "download_status": "failed",
                "asset_write_status": "skipped",
                "blocking_reason": "pdf_download_failed",
                "recommended_next_action": next_action,
                "failure_audit": failure_audit,
            }
        )
        alternate_routes = remote_pdf_failure_alternate_routes(error_summary)
        if alternate_routes:
            trace["alternate_routes"] = alternate_routes
        return AcquisitionResult(
            status="pdf_acquire_failed",
            paper=paper,
            pdf_asset=None,
            source_candidates=[source_candidate_from_registry_row(candidate, paper=paper)],
            dedupe=dedupe,
            message=error_summary,
            blocking_reason="pdf_download_failed",
            recommended_next_action=next_action,
            stage_results=[trace],
        )

    try:
        with tempfile.TemporaryDirectory(prefix="scievo_remote_pdf_") as temp_dir:
            temp_pdf = Path(temp_dir) / "paper.pdf"
            temp_pdf.write_bytes(pdf_bytes)
            asset = registry.store_pdf_asset(
                paper_id=paper["paper_id"],
                source_path=temp_pdf,
                source_kind=str(candidate.get("source_name") or "remote_pdf"),
                source_uri=str(pdf_url),
                license=candidate.get("license"),
                open_access_status=candidate.get("open_access_status"),
            )
    except CrossPaperDedupeConflict as exc:
        conflict = {
            **exc.to_dict(),
            "stage": "pdf_acquire",
            "status": "dedupe_cross_paper_conflict",
            "source_candidate_id": candidate.get("candidate_id"),
            "candidate_pdf_url": str(pdf_url),
            "asset_policy": candidate.get("asset_policy"),
            "download_status": "ready",
            "asset_write_status": "dedupe_cross_paper_conflict",
            "reason": "dedupe_asset_on_other_paper",
            "owner_readiness": {
                "pdf_asset_count": len(registry.list_pdf_assets(exc.asset_owner_paper_id)),
                "processing_job_count": len(registry.list_processing_jobs(exc.asset_owner_paper_id)),
                "evidence_count": registry.evidence_coverage(exc.asset_owner_paper_id)["total"],
            },
        }
        return dedupe_conflict_result(
            paper=paper,
            candidates=[source_candidate_from_registry_row(candidate, paper=paper)],
            diagnostic=conflict,
            message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - include storage/PDF validation in trace.
        error_summary = str(exc)
        next_action = remote_pdf_failure_next_action(error_summary)
        failure_audit = remote_pdf_failure_audit(
            error_summary=error_summary,
            pdf_url=str(pdf_url),
            candidate=candidate,
        )
        trace.update(
            {
                "status": "failed",
                "download_status": "ready",
                "asset_write_status": "failed",
                "blocking_reason": "pdf_asset_write_failed",
                "recommended_next_action": next_action,
                "failure_audit": failure_audit,
            }
        )
        alternate_routes = remote_pdf_failure_alternate_routes(error_summary)
        if alternate_routes:
            trace["alternate_routes"] = alternate_routes
        return AcquisitionResult(
            status="pdf_acquire_failed",
            paper=paper,
            pdf_asset=None,
            source_candidates=[source_candidate_from_registry_row(candidate, paper=paper)],
            dedupe=dedupe,
            message=error_summary,
            blocking_reason="pdf_asset_write_failed",
            recommended_next_action=next_action,
            stage_results=[trace],
        )

    registry.record_provenance_event(
        action="acquire_pdf",
        source=str(candidate.get("source_name") or "remote_pdf"),
        inputs={
            "paper_id": paper["paper_id"],
            "candidate_id": candidate.get("candidate_id"),
            "pdf_url": str(pdf_url),
            "asset_policy": candidate.get("asset_policy"),
        },
        outputs={"paper_id": paper["paper_id"], "asset_id": asset["asset_id"]},
    )
    trace.update(
        {
            "status": "ready",
            "download_status": "ready",
            "asset_write_status": "ready",
            "pdf_asset_id": asset["asset_id"],
            "blocking_reason": None,
            "recommended_next_action": "process_with_mineru",
        }
    )
    return AcquisitionResult(
        status="ready",
        paper=paper,
        pdf_asset=asset,
        source_candidates=[source_candidate_from_registry_row(candidate, paper=paper)],
        dedupe=dedupe,
        message="已从安全远程 PDF candidate 获取 PDF。",
        stage_results=[trace],
    )


def discover_by_doi(
    *,
    doi: str,
    local_doi_index: Path | None = None,
    local_doi_archive_root: Path | None = None,
    sources: list[str] | None = None,
    metadata_candidates: list[SourceCandidate] | None = None,
    openalex_client: OpenAlexClient | None = None,
    crossref_client: CrossrefClient | None = None,
    sciverse_client: SciverseClient | None = None,
) -> SourceDiscoveryResult:
    normalized = normalize_doi(doi)
    if not normalized:
        raise AcquisitionError(f"DOI 无效：{doi}")

    candidates = list(metadata_candidates or [])
    requested_sources = requested_discovery_sources(
        sources,
        default=["openalex", "crossref", "sciverse", "arxiv", "local_doi_archive"],
    )
    if metadata_candidates is None:
        if "openalex" in requested_sources:
            candidates.append((openalex_client or OpenAlexClient()).fetch_by_doi(normalized))
        if "crossref" in requested_sources:
            candidates.append((crossref_client or CrossrefClient()).fetch_by_doi(normalized))
        if "sciverse" in requested_sources:
            candidates.append((sciverse_client or SciverseClient()).fetch_by_doi(normalized))
    if "arxiv" in requested_sources and not any(candidate.source_name == "arxiv" for candidate in candidates):
        arxiv_candidate = arxiv_candidate_from_doi(normalized)
        if arxiv_candidate:
            candidates.append(arxiv_candidate)

    local_archive_hit: LocalDoiArchiveHit | None = None
    if "local_doi_archive" in requested_sources and local_doi_index and local_doi_archive_root:
        archive = LocalDoiArchive(index_path=local_doi_index, archive_root=local_doi_archive_root)
        try:
            local_archive_hit = archive.lookup(normalized)
            candidates.append(
                SourceCandidate(
                    source_name="local_doi_archive",
                    query_kind="doi",
                    status="ready" if local_archive_hit else "not_found",
                    source_record_id=local_archive_hit.source_uri if local_archive_hit else None,
                    normalized_doi=normalized,
                    candidate_score=1.0 if local_archive_hit else 0.0,
                    raw={"source_uri": local_archive_hit.source_uri} if local_archive_hit else {},
                    error_summary=None if local_archive_hit else "本地 DOI archive 未命中。",
                )
            )
        except SourceAdapterError as exc:
            candidates.append(
                SourceCandidate(
                    source_name="local_doi_archive",
                    query_kind="doi",
                    status="failed",
                    normalized_doi=normalized,
                    error_summary=str(exc),
                )
            )

    return SourceDiscoveryResult(
        query_kind="doi",
        candidates=candidates,
        local_archive_hit=local_archive_hit,
    )


def discover_by_title(
    *,
    title: str,
    sources: list[str] | None = None,
    metadata_candidates: list[SourceCandidate] | None = None,
    openalex_client: OpenAlexClient | None = None,
    crossref_client: CrossrefClient | None = None,
    sciverse_client: SciverseClient | None = None,
    limit: int = 5,
) -> SourceDiscoveryResult:
    query = title.strip()
    if not query:
        raise AcquisitionError("题名不能为空。")
    candidates = list(metadata_candidates or [])
    requested_sources = requested_discovery_sources(
        sources,
        default=["openalex", "crossref", "sciverse"],
    )
    if metadata_candidates is None:
        if "openalex" in requested_sources:
            candidates.extend((openalex_client or OpenAlexClient()).search_title(query, limit=limit))
        if "crossref" in requested_sources:
            candidates.extend((crossref_client or CrossrefClient()).search_title(query, limit=limit))
        if "sciverse" in requested_sources:
            candidates.extend((sciverse_client or SciverseClient()).search_title(query, limit=limit))
        if "local_doi_archive" in requested_sources:
            candidates.append(
                SourceCandidate(
                    source_name="local_doi_archive",
                    query_kind="title",
                    status="blocked",
                    title=query,
                    error_summary="本地 DOI archive 需要 DOI，无法按题名检索。",
                )
            )
    if not candidates:
        candidates.append(
            SourceCandidate(
                source_name="local_query",
                query_kind="title",
                status="not_found",
                title=query,
                error_summary="未发现来源候选。",
            )
        )
    return SourceDiscoveryResult(query_kind="title", candidates=candidates)


def requested_discovery_sources(
    sources: list[str] | None,
    *,
    default: list[str],
) -> set[str]:
    if sources is None:
        return {source.lower() for source in default}
    return {source.lower() for source in sources}


def arxiv_candidate_from_doi(doi: str) -> SourceCandidate | None:
    normalized = normalize_doi(doi)
    if not normalized:
        return None
    prefix = "10.48550/arxiv."
    if not normalized.startswith(prefix):
        return None
    arxiv_id = normalize_arxiv_id(normalized.removeprefix(prefix))
    if not arxiv_id:
        return None
    return SourceCandidate(
        source_name="arxiv",
        query_kind="doi",
        status="ready",
        source_record_id=arxiv_id,
        normalized_doi=normalized,
        candidate_score=1.0,
        pdf_candidates=[
            {
                "kind": "arxiv_pdf",
                "url": f"https://arxiv.org/pdf/{arxiv_id}",
                "content_type": "application/pdf",
                "source": "arxiv",
                "is_oa": True,
                "license": "arxiv",
            }
        ],
        license="arxiv",
        open_access_status="green",
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
        raw={"arxiv_id": arxiv_id, "doi_fallback": True},
    )


def discovery_candidates_to_rows(
    candidates: list[SourceCandidate],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        first_pdf = first_pdf_candidate(candidate.pdf_candidates)
        has_pdf_url = first_pdf is not None
        pdf_source = pdf_url_source(first_pdf) if first_pdf else None
        license_value = candidate.license or (
            str(first_pdf.get("license")) if first_pdf and first_pdf.get("license") else None
        )
        is_oa = source_candidate_is_oa(candidate, first_pdf)
        asset_policy = discovery_asset_policy(candidate, first_pdf, is_oa=is_oa)
        risk_reasons = discovery_candidate_risk_reasons(candidate, has_pdf_url=has_pdf_url)
        blocking_reason = discovery_blocking_reason(candidate, risk_reasons, has_pdf_url=has_pdf_url)
        recommended_action = discovery_recommended_action(
            candidate,
            risk_reasons,
            has_pdf_url=has_pdf_url,
            is_oa=is_oa,
        )
        raw = candidate.raw if isinstance(candidate.raw, dict) else {}
        rows.append(
            {
                "source_name": candidate.source_name,
                "query_kind": candidate.query_kind,
                "status": candidate.status,
                "source_record_id": candidate.source_record_id,
                "normalized_doi": candidate.normalized_doi,
                "title": candidate.title,
                "year": candidate.year,
                "published_year": candidate.year,
                "published_date": raw.get("publication_date") or raw.get("published_date"),
                "year_source": candidate.source_name if candidate.year else None,
                "year_confidence": 0.95 if candidate.year else None,
                "year_needs_review": candidate.year is None,
                "venue": candidate.venue,
                "candidate_score": candidate.candidate_score,
                # authority signal (2026-08-28): crossref is-referenced-by-count /
                # openalex cited_by_count, for authority-aware reranking downstream
                "citation_count": (raw.get("is-referenced-by-count")
                                   if isinstance(raw, dict) else None)
                or (raw.get("cited_by_count") if isinstance(raw, dict) else None),
                "source_url": candidate.source_url,
                "license": license_value,
                "open_access_status": candidate.open_access_status,
                "is_oa": is_oa,
                "asset_policy": asset_policy,
                "export_policy": discovery_export_policy(asset_policy),
                "has_pdf_url": has_pdf_url,
                "pdf_url": str(first_pdf.get("url")) if first_pdf and first_pdf.get("url") else None,
                "pdf_source": pdf_source,
                "pdf_url_source": pdf_source,
                "pdf_candidates": candidate.pdf_candidates,
                "can_acquire_pdf_now": bool(is_oa or candidate.source_name == "local_doi_archive"),
                "has_local_pdf": candidate.source_name == "local_doi_archive" and candidate.status == "ready",
                "local_pdf_usable": candidate.source_name == "local_doi_archive" and candidate.status == "ready",
                "local_pdf_source": "local_doi_archive" if candidate.source_name == "local_doi_archive" else None,
                "sciverse_configured": None,
                "has_doc_id": bool(raw.get("doc_id") or raw.get("unique_id")),
                "content_accessible": raw.get("is_content_accessible"),
                "can_sciverse_ingest_now": candidate.source_name == "sciverse"
                and candidate.status == "ready"
                and bool(raw.get("doc_id") or raw.get("unique_id")),
                "requires_confirmation": candidate.query_kind in {"title", "discovery"},
                "can_register": candidate.status == "ready",
                "can_process_now": candidate.source_name == "local_doi_archive" and candidate.status == "ready",
                "match_scores": {
                    "doi": 1.0 if candidate.normalized_doi else None,
                    "title": candidate.candidate_score
                    if candidate.query_kind in {"title", "discovery"}
                    else None,
                    "year": None,
                    "overall": candidate.candidate_score,
                },
                "risk_reasons": risk_reasons,
                "blocking_reason": blocking_reason,
                "recommended_action": recommended_action,
                "recommended_next_action": recommended_action,
                "error_summary": candidate.error_summary,
            }
        )
    return rows


def apply_discovery_filters(
    candidates: list[SourceCandidate],
    filters: dict[str, Any],
) -> list[SourceCandidate]:
    year_from = safe_int(filters.get("year_from") or filters.get("from_year") or filters.get("min_year"))
    year_to = safe_int(filters.get("year_to") or filters.get("to_year") or filters.get("max_year"))
    only_oa = filters.get("is_oa")
    if isinstance(only_oa, str):
        only_oa = only_oa.lower() == "true"
    filtered: list[SourceCandidate] = []
    for candidate in candidates:
        if year_from is not None and candidate.year is not None and candidate.year < year_from:
            continue
        if year_to is not None and candidate.year is not None and candidate.year > year_to:
            continue
        if only_oa is True and not source_candidate_is_oa(candidate, first_pdf_candidate(candidate.pdf_candidates)):
            continue
        filtered.append(candidate)
    return filtered


def discovery_candidate_risk_reasons(
    candidate: SourceCandidate,
    *,
    has_pdf_url: bool,
) -> list[str]:
    risks: list[str] = []
    if candidate.status not in {"ready", "candidate_review_required"}:
        risks.append("source_status_not_ready")
    if candidate.query_kind in {"title", "discovery"} and (candidate.candidate_score or 0) < 0.75:
        risks.append("low_title_similarity")
    if not has_pdf_url and candidate.source_name != "local_doi_archive":
        risks.append("no_pdf_url")
    if has_pdf_url and candidate.source_name not in {"local_doi_archive"}:
        risks.append("requires_explicit_pdf_acquire")
    if not candidate.normalized_doi:
        risks.append("missing_doi")
    if candidate.year is None:
        risks.append("missing_year")
    return risks


def discovery_blocking_reason(
    candidate: SourceCandidate,
    risk_reasons: list[str],
    *,
    has_pdf_url: bool,
) -> str | None:
    if candidate.status != "ready":
        return "source_not_ready"
    if "low_title_similarity" in risk_reasons:
        return "candidate_needs_review"
    if "missing_year" in risk_reasons:
        return "missing_year"
    if not has_pdf_url and candidate.source_name != "sciverse":
        return "missing_pdf"
    return None


def discovery_recommended_action(
    candidate: SourceCandidate,
    risk_reasons: list[str],
    *,
    has_pdf_url: bool,
    is_oa: bool,
) -> str:
    if candidate.status != "ready":
        return "review_source_error"
    if "low_title_similarity" in risk_reasons:
        return "review_candidate"
    if candidate.source_name == "sciverse" and candidate.raw.get("doc_id"):
        return "sciverse_ingest"
    if candidate.source_name == "local_doi_archive":
        return "register_and_process_local_pdf"
    if has_pdf_url and is_oa:
        return "register_then_pdf_acquire"
    if has_pdf_url:
        return "register_candidate_only"
    return "register_candidate_or_find_pdf"


def discovery_asset_policy(
    candidate: SourceCandidate,
    first_pdf: dict[str, Any] | None,
    *,
    is_oa: bool,
) -> str:
    if candidate.source_name == "local_doi_archive":
        return "local_available_pdf" if candidate.status == "ready" else "local_pdf_missing"
    if first_pdf and candidate.source_name == "arxiv":
        return "arxiv_auto_allowed"
    if first_pdf and is_oa:
        return "oa_auto_allowed"
    if first_pdf:
        return "unknown_public_pdf"
    return "no_pdf_candidate"


def discovery_export_policy(asset_policy: str) -> str:
    if asset_policy == "local_available_pdf":
        return "local_pdf_allowed"
    if asset_policy in {"oa_auto_allowed", "arxiv_auto_allowed"}:
        return "remote_pdf_allowed_after_commit"
    if asset_policy == "unknown_public_pdf":
        return "candidate_only_review_required"
    return "no_pdf_available"


def source_candidate_is_oa(candidate: SourceCandidate, first_pdf: dict[str, Any] | None) -> bool:
    if first_pdf and first_pdf.get("is_oa") is not None:
        return bool(first_pdf.get("is_oa"))
    if candidate.open_access_status:
        return candidate.open_access_status.lower() not in {"closed", "unknown"}
    if candidate.license:
        return True
    return False


def first_pdf_candidate(pdf_candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    for candidate in pdf_candidates:
        if candidate.get("url") and str(candidate.get("content_type", "application/pdf")).lower().endswith("pdf"):
            return candidate
    for candidate in pdf_candidates:
        if candidate.get("url"):
            return candidate
    return None


def pdf_url_source(candidate: dict[str, Any]) -> str | None:
    value = candidate.get("source") or candidate.get("kind")
    return str(value) if value else None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def register_source_candidates(
    registry: LibraryRegistry,
    paper_id: str,
    candidates: list[SourceCandidate],
) -> None:
    for candidate in candidates:
        registry.add_source_candidate(
            paper_id=paper_id,
            source_name=candidate.source_name,
            query_kind=candidate.query_kind,
            source_record_id=candidate.source_record_id,
            doi=candidate.normalized_doi,
            title=candidate.title,
            year=candidate.year,
            venue=candidate.venue,
            candidate_score=candidate.candidate_score,
            pdf_candidates=candidate.pdf_candidates,
            license=candidate.license,
            open_access_status=candidate.open_access_status,
            source_url=candidate.source_url,
            status=candidate.status,
            error_summary=candidate.error_summary,
        )


def source_candidate_from_registry_row(
    row: dict[str, Any],
    *,
    paper: dict[str, Any] | None = None,
) -> SourceCandidate:
    return SourceCandidate(
        source_name=str(row["source_name"]),
        query_kind=str(row.get("query_kind") or "candidate_confirmation"),
        status=str(row.get("status") or "ready"),
        source_record_id=row.get("source_record_id"),
        normalized_doi=normalize_doi(row.get("normalized_doi")),
        title=row.get("title") or (paper or {}).get("title"),
        year=row.get("year"),
        venue=row.get("venue"),
        candidate_score=row.get("candidate_score"),
        pdf_candidates=row.get("pdf_candidates") if isinstance(row.get("pdf_candidates"), list) else [],
        license=row.get("license"),
        open_access_status=row.get("open_access_status"),
        source_url=row.get("source_url"),
        raw={"source_candidate_id": row.get("candidate_id")},
        error_summary=row.get("error_summary"),
    )


def pdf_acquire_trace(
    *,
    registry: LibraryRegistry,
    paper_id: str,
    status: str,
    blocking_reason: str | None,
    recommended_next_action: str | None,
) -> dict[str, Any]:
    rows = registry.list_source_candidates(paper_id)
    return {
        "stage": "pdf_acquire",
        "status": status,
        "blocking_reason": blocking_reason,
        "recommended_next_action": recommended_next_action,
        "candidate_count": len(rows),
        "candidates": [source_candidate_trace(row) for row in rows],
        "risk_reasons": sorted(
            {
                risk
                for row in rows
                for risk in row.get("risk_reasons", [])
            }
        ),
    }


def safe_remote_pdf_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed_rows = [
        (index, row)
        for index, row in enumerate(rows)
        if row.get("asset_policy") in {"oa_auto_allowed", "arxiv_auto_allowed"}
        and (row.get("candidate_pdf_url") or row.get("pdf_url"))
    ]
    return [
        row
        for _, row in sorted(
            indexed_rows,
            key=lambda item: (safe_remote_pdf_candidate_rank(item[1]), item[0]),
        )
    ]


def first_safe_remote_pdf_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = safe_remote_pdf_candidates(rows)
    return candidates[0] if candidates else None


def safe_remote_pdf_candidate_rank(row: dict[str, Any]) -> int:
    if is_arxiv_pdf_candidate(row):
        return 0
    if row.get("asset_policy") == "arxiv_auto_allowed":
        return 1
    if is_openalex_content_candidate(row):
        return 3
    return 2


def is_arxiv_pdf_candidate(row: dict[str, Any]) -> bool:
    pdf_url = str(row.get("candidate_pdf_url") or row.get("pdf_url") or "").lower()
    pdf_source = str(row.get("pdf_source") or row.get("pdf_url_source") or "").lower()
    source_name = str(row.get("source_name") or "").lower()
    return (
        row.get("asset_policy") == "arxiv_auto_allowed"
        or source_name == "arxiv"
        or pdf_source == "arxiv"
        or "arxiv.org/pdf/" in pdf_url
    )


def is_openalex_content_candidate(row: dict[str, Any]) -> bool:
    pdf_url = str(row.get("candidate_pdf_url") or row.get("pdf_url") or "").lower()
    pdf_source = str(row.get("pdf_source") or row.get("pdf_url_source") or "").lower()
    return pdf_source == "openalex_content" or "api.openalex.org" in pdf_url


def remote_pdf_failure_next_action(error_summary: str) -> str:
    if parse_http_status(error_summary) in {401, 403}:
        return "try_sciverse_or_manual_upload_or_register_metadata"
    return "try_alternate_pdf_source_or_upload"


def remote_pdf_failure_alternate_routes(error_summary: str) -> list[str]:
    if parse_http_status(error_summary) in {401, 403}:
        return ["sciverse_ingest", "manual_upload", "register_metadata_wait_for_upload"]
    if remote_pdf_failure_retryable(error_summary):
        return ["retry_same_url", "try_alternate_pdf_source", "manual_upload"]
    return []


def remote_pdf_failure_retryable(error_summary: str) -> bool:
    http_status = parse_http_status(error_summary)
    lowered = error_summary.lower()
    if http_status in {401, 403, 404, 410}:
        return False
    if http_status is not None:
        return http_status >= 500
    return any(
        token in lowered
        for token in ("timeout", "timed out", "超时", "temporarily", "reset", "connection", "network", "连接")
    )


def remote_pdf_failure_kind(error_summary: str) -> str:
    http_status = parse_http_status(error_summary)
    if http_status is not None:
        return f"http_{http_status}"
    lowered = error_summary.lower()
    if "timeout" in lowered or "timed out" in lowered or "超时" in lowered:
        return "timeout"
    if any(token in lowered for token in ("connection", "network", "reset", "dns", "连接")):
        return "network"
    return "unknown"


def remote_pdf_failure_audit(
    *,
    error_summary: str,
    pdf_url: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    alternate_routes = remote_pdf_failure_alternate_routes(error_summary)
    retryable = remote_pdf_failure_retryable(error_summary)
    return {
        "error_summary": error_summary,
        "http_status": parse_http_status(error_summary),
        "failure_kind": remote_pdf_failure_kind(error_summary),
        "candidate_id": candidate.get("candidate_id"),
        "candidate_pdf_url": pdf_url,
        "final_url": pdf_url,
        "source_name": candidate.get("source_name"),
        "pdf_source": candidate.get("pdf_source") or candidate.get("pdf_url_source"),
        "asset_policy": candidate.get("asset_policy"),
        "license": candidate.get("license"),
        "open_access_status": candidate.get("open_access_status"),
        "retryable": retryable,
        "can_retry_same_url": retryable,
        "fallback_source": alternate_routes[0] if alternate_routes else None,
        "alternate_routes": alternate_routes,
        "recommended_next_action": remote_pdf_failure_next_action(error_summary),
    }


def remote_pdf_acquire_trace(row: dict[str, Any]) -> dict[str, Any]:
    failure_audit = row.get("failure_audit") if isinstance(row.get("failure_audit"), dict) else {}
    return {
        "stage": "pdf_acquire",
        "status": "attempting",
        "source_candidate_id": row.get("candidate_id"),
        "source_name": row.get("source_name"),
        "candidate_pdf_url": row.get("candidate_pdf_url") or row.get("pdf_url"),
        "pdf_source": row.get("pdf_source") or row.get("pdf_url_source"),
        "asset_policy": row.get("asset_policy"),
        "export_policy": row.get("export_policy"),
        "license": row.get("license"),
        "open_access_status": row.get("open_access_status"),
        "risk_reasons": row.get("risk_reasons", []),
        "failure_audit": failure_audit,
    }


def source_candidate_trace(row: dict[str, Any]) -> dict[str, Any]:
    failure_audit = row.get("failure_audit") if isinstance(row.get("failure_audit"), dict) else {}
    return {
        "source_candidate_id": row.get("candidate_id"),
        "source_name": row.get("source_name"),
        "status": row.get("status"),
        "normalized_doi": row.get("normalized_doi"),
        "title": row.get("title"),
        "year": row.get("year"),
        "candidate_pdf_url": row.get("candidate_pdf_url") or row.get("pdf_url"),
        "pdf_source": row.get("pdf_source") or row.get("pdf_url_source"),
        "asset_policy": row.get("asset_policy"),
        "export_policy": row.get("export_policy"),
        "license": row.get("license"),
        "open_access_status": row.get("open_access_status"),
        "has_pdf_url": row.get("has_pdf_url"),
        "can_acquire_pdf_now": row.get("can_acquire_pdf_now"),
        "risk_reasons": row.get("risk_reasons", []),
        "blocking_reason": row.get("blocking_reason"),
        "recommended_action": row.get("recommended_action"),
        "failure_audit": failure_audit,
    }


def source_candidate_identifiers(candidates: list[SourceCandidate]) -> list[dict[str, str]]:
    identifiers: list[dict[str, str]] = []
    for candidate in candidates:
        if candidate.status not in {"ready", "candidate_review_required"}:
            continue
        if candidate.normalized_doi:
            identifiers.append(
                {
                    "scheme": "doi",
                    "value": candidate.normalized_doi,
                    "source": candidate.source_name,
                }
            )
        if candidate.source_record_id:
            identifiers.append(
                {
                    "scheme": candidate.source_name,
                    "value": candidate.source_record_id,
                    "source": candidate.source_name,
                }
            )
    return identifiers


def registry_candidates_to_source_candidates(rows: list[dict[str, Any]]) -> list[SourceCandidate]:
    return [
        SourceCandidate(
            source_name=str(row["source_name"]),
            query_kind=str(row["query_kind"]),
            status=str(row["status"]),
            source_record_id=row.get("source_record_id"),
            normalized_doi=row.get("normalized_doi"),
            title=row.get("title"),
            year=row.get("year"),
            venue=row.get("venue"),
            candidate_score=row.get("candidate_score"),
            pdf_candidates=row.get("pdf_candidates", []),
            license=row.get("license"),
            open_access_status=row.get("open_access_status"),
            source_url=row.get("source_url"),
            error_summary=row.get("error_summary"),
        )
        for row in rows
    ]


def first_ready_candidate(candidates: list[SourceCandidate]) -> SourceCandidate | None:
    ready = [candidate for candidate in candidates if candidate.status == "ready"]
    if not ready:
        return None
    return sorted(ready, key=lambda item: item.candidate_score or 0.0, reverse=True)[0]


class AcquisitionError(RuntimeError):
    pass
