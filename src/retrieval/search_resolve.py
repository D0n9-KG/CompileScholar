from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from retrieval.acquisition import (
    discover_by_doi,
    discover_by_query,
    discover_by_title,
    discovery_candidates_to_rows,
)
from retrieval.pdf_resolver import (
    find_registered_paper,
    resolve_pdf_attempts,
    select_pdf_attempt,
)
from retrieval.registry import LibraryRegistry, normalize_doi
from retrieval.sources import CrossrefClient, OpenAlexClient, SciverseClient, SourceCandidate


class SearchResolveError(RuntimeError):
    pass


def search_resolve(
    *,
    registry: LibraryRegistry,
    query: str,
    query_type: str = "auto",
    metadata_sources: list[str] | None = None,
    pdf_sources: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    limit: int = 10,
    dry_run: bool = True,
    write_policy: str = "dry_run",
    local_doi_index: Path | None = None,
    local_doi_archive_root: Path | None = None,
    openalex_client: OpenAlexClient | None = None,
    crossref_client: CrossrefClient | None = None,
    sciverse_client: SciverseClient | None = None,
) -> dict[str, Any]:
    if not dry_run or write_policy != "dry_run":
        raise SearchResolveError("search-resolve 当前只支持 dry-run no-write；请用显式 action 接口注册、下载或处理。")
    query_text = query.strip()
    if not query_text:
        raise SearchResolveError("query 不能为空。")
    resolved_query_type = infer_query_type(query_text, query_type)
    if limit < 1:
        raise SearchResolveError("limit 必须大于 0。")

    if resolved_query_type == "doi":
        discovery = discover_by_doi(
            doi=query_text,
            sources=metadata_sources,
            local_doi_index=local_doi_index,
            local_doi_archive_root=local_doi_archive_root,
            openalex_client=openalex_client,
            crossref_client=crossref_client,
            sciverse_client=sciverse_client,
        )
    elif resolved_query_type == "title":
        discovery = discover_by_title(
            title=query_text,
            sources=metadata_sources,
            openalex_client=openalex_client,
            crossref_client=crossref_client,
            sciverse_client=sciverse_client,
            limit=limit,
        )
    elif resolved_query_type in {"topic", "query"}:
        discovery = discover_by_query(
            query=query_text,
            sources=metadata_sources,
            filters=filters or {},
            openalex_client=openalex_client,
            crossref_client=crossref_client,
            sciverse_client=sciverse_client,
            limit=limit,
        )
    else:
        raise SearchResolveError(f"query_type 不支持：{query_type}")

    groups = group_candidates(discovery.candidates)
    results = [
        build_result_row(
            registry=registry,
            candidates=candidates,
            fallback_query=query_text,
            pdf_sources=pdf_sources,
            local_doi_index=local_doi_index,
            local_doi_archive_root=local_doi_archive_root,
        )
        for candidates in groups
    ]
    results = [row for row in results if row["source_candidates"]]
    return {
        "status": "ready" if results else "empty",
        "query": query_text,
        "query_type": resolved_query_type,
        "dry_run": True,
        "write_policy": "dry_run",
        "summary": {
            "result_count": len(results),
            "pdf_found_count": sum(1 for row in results if row.get("pdf_found")),
            "registered_count": sum(1 for row in results if row.get("registered_already")),
        },
        "results": results,
        "next": {
            "register_candidate": "/api/library/acquisitions/batch",
            "acquire_pdf": "/api/library/acquisitions/batch",
            "process_with_mineru": "/api/library/papers/process-batch",
            "export": "/api/library/exports",
        },
    }


def infer_query_type(query: str, query_type: str) -> str:
    requested = query_type.strip().lower()
    if requested != "auto":
        return requested
    return "doi" if normalize_doi(query) else "title"


def group_candidates(candidates: list[SourceCandidate]) -> list[list[SourceCandidate]]:
    grouped: dict[str, list[SourceCandidate]] = defaultdict(list)
    order: list[str] = []
    for candidate in candidates:
        key = candidate_group_key(candidate)
        if key not in grouped:
            order.append(key)
        grouped[key].append(candidate)
    return [grouped[key] for key in order]


def candidate_group_key(candidate: SourceCandidate) -> str:
    normalized = normalize_doi(candidate.normalized_doi)
    if normalized:
        return f"doi:{normalized}"
    if candidate.title:
        return f"title:{' '.join(candidate.title.lower().split())}"
    return f"source:{candidate.source_name}:{candidate.source_record_id or id(candidate)}"


def build_result_row(
    *,
    registry: LibraryRegistry,
    candidates: list[SourceCandidate],
    fallback_query: str,
    pdf_sources: list[str] | None,
    local_doi_index: Path | None,
    local_doi_archive_root: Path | None,
) -> dict[str, Any]:
    rows = discovery_candidates_to_rows(candidates)
    primary = best_candidate(candidates)
    doi = first_value(candidate.normalized_doi for candidate in candidates)
    title = first_value(candidate.title for candidate in candidates) or fallback_query
    year = first_value(candidate.year for candidate in candidates)
    paper = find_registered_paper(registry, doi=doi, title=title)
    attempts = resolve_pdf_attempts(
        registry=registry,
        doi=doi,
        title=title,
        candidates=candidates,
        pdf_sources=pdf_sources,
        local_doi_index=local_doi_index,
        local_doi_archive_root=local_doi_archive_root,
    )
    selected = select_pdf_attempt(attempts)
    process_state = result_process_state(registry=registry, paper=paper, selected=selected)
    risk_reasons = result_risk_reasons(selected, attempts)
    allowed_actions = result_allowed_actions(selected, attempts)
    disabled_actions = result_disabled_actions(selected)
    action_payloads = result_action_payloads(
        paper=paper,
        doi=doi,
        title=title,
        selected=selected,
        attempts=attempts,
        candidates=candidates,
    )
    action_explanations = result_action_explanations(selected, attempts)
    return {
        "doi": doi,
        "title": title,
        "published_year": year,
        "year": year,
        "venue": primary.venue if primary else None,
        "registered_already": paper is not None,
        "paper_id": paper.get("paper_id") if paper else None,
        "source_candidates": rows,
        "pdf_attempts": attempts,
        "pdf_found": selected is not None,
        "selected_pdf_source": selected.get("source") if selected else None,
        "asset_policy": selected.get("asset_policy") if selected else "no_pdf_available",
        "can_process_with_mineru": bool(selected and selected.get("can_process_with_mineru")),
        "existing_mineru_ready": process_state["mineru_ready"],
        "existing_evidence_ready": process_state["evidence_ready"],
        "mineru_ready": process_state["mineru_ready"],
        "evidence_ready": process_state["evidence_ready"],
        "evidence_count": process_state["evidence_count"],
        "blocking_reason": process_state["blocking_reason"],
        "process_state": process_state,
        "risk_reasons": risk_reasons,
        "allowed_actions": allowed_actions,
        "disabled_actions": disabled_actions,
        "action_payloads": action_payloads,
        "action_explanations": action_explanations,
        "recommended_action": result_recommended_action(selected, attempts, process_state),
        "recommended_next_action": result_recommended_action(selected, attempts, process_state),
    }


def best_candidate(candidates: list[SourceCandidate]) -> SourceCandidate | None:
    ready = [candidate for candidate in candidates if candidate.status == "ready"]
    pool = ready or candidates
    if not pool:
        return None
    return sorted(pool, key=lambda item: item.candidate_score or 0.0, reverse=True)[0]


def first_value(values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def result_risk_reasons(
    selected: dict[str, Any] | None,
    attempts: list[dict[str, Any]],
) -> list[str]:
    risks: list[str] = []
    if not selected:
        risks.append("pdf_not_found")
    elif selected.get("source") not in {"existing_asset", "local_doi_archive"}:
        risks.extend(selected.get("risk_reasons") or ["requires_explicit_pdf_acquire"])
    if any(attempt.get("source") == "sciverse_ai_ready" and attempt.get("status") == "hit" for attempt in attempts):
        risks.append("sciverse_remote_asset_available")
    return list(dict.fromkeys(risks))


def result_allowed_actions(selected: dict[str, Any] | None, attempts: list[dict[str, Any]]) -> list[str]:
    actions = ["register_candidate"]
    if selected and selected.get("source") in {"existing_asset", "local_doi_archive"}:
        actions.append("process_with_mineru")
    elif selected and selected.get("asset_policy") in {"oa_auto_allowed", "arxiv_auto_allowed"}:
        actions.append("acquire_pdf")
    if not selected and any(
        attempt.get("source") == "sciverse_ai_ready" and attempt.get("status") == "hit"
        for attempt in attempts
    ):
        actions.append("use_sciverse_ai_ready")
    return actions


def result_disabled_actions(selected: dict[str, Any] | None) -> list[str]:
    disabled: list[str] = []
    if not selected or selected.get("source") not in {"existing_asset", "local_doi_archive"}:
        disabled.append("process_with_mineru")
    if selected and selected.get("source") in {"existing_asset", "local_doi_archive"}:
        disabled.append("acquire_pdf")
    if selected and selected.get("asset_policy") == "unknown_public_pdf":
        disabled.append("acquire_pdf")
    return disabled


def result_recommended_action(
    selected: dict[str, Any] | None,
    attempts: list[dict[str, Any]],
    process_state: dict[str, Any] | None = None,
) -> str:
    if process_state and process_state.get("evidence_ready"):
        return "ready_for_experiment"
    if process_state and process_state.get("failure_report"):
        return "inspect_processing_job"
    if not selected:
        if any(attempt.get("source") == "sciverse_ai_ready" and attempt.get("status") == "hit" for attempt in attempts):
            return "ingest_sciverse_ai_ready"
        return "review_or_upload_pdf"
    if selected.get("source") == "existing_asset":
        return "process_existing_pdf"
    if selected.get("source") == "local_doi_archive":
        return "register_and_process_local_pdf"
    if selected.get("asset_policy") == "unknown_public_pdf":
        return "review_pdf_candidate"
    return "register_then_pdf_acquire"


def result_action_payloads(
    *,
    paper: dict[str, Any] | None,
    doi: str | None,
    title: str | None,
    selected: dict[str, Any] | None,
    attempts: list[dict[str, Any]],
    candidates: list[SourceCandidate],
) -> dict[str, Any]:
    metadata_sources = sorted(
        {
            candidate.source_name
            for candidate in candidates
            if candidate.source_name not in {"local_query", "local_doi_archive"}
        }
    )
    item: dict[str, Any] = {}
    if doi:
        item["doi"] = doi
    if title:
        item["title"] = title
    if metadata_sources:
        item["sources"] = metadata_sources

    payloads: dict[str, Any] = {
        "register_candidate": {
            "endpoint": "/api/library/acquisitions/batch",
            "method": "POST",
            "body": {
                "items": [item],
                "dry_run": True,
                "write_policy": "dry_run",
                "process": False,
            },
        }
    }

    acquire_enabled = bool(selected and selected.get("asset_policy") in {"oa_auto_allowed", "arxiv_auto_allowed"})
    payloads["acquire_pdf"] = {
        "endpoint": "/api/library/acquisitions/batch",
        "method": "POST",
        "enabled": acquire_enabled,
        "reason": None if acquire_enabled else acquire_disabled_reason(selected),
        "body": {
            "items": [item],
            "dry_run": not acquire_enabled,
            "write_policy": "commit" if acquire_enabled else "dry_run",
            "process": False,
        },
    }

    paper_id = paper.get("paper_id") if paper else None
    process_enabled = bool(paper_id and selected and selected.get("source") in {"existing_asset", "local_doi_archive"})
    payloads["process_with_mineru"] = {
        "endpoint": "/api/library/papers/process-batch",
        "method": "POST",
        "enabled": process_enabled,
        "reason": None if process_enabled else "missing_registered_pdf_asset",
        "body": {
            "items": [{"paper_id": paper_id}] if paper_id else [],
            "stages": ["mineru", "evidence_index"],
            "dry_run": True,
            "write_policy": "dry_run",
            "preferred_source": [str(selected.get("source"))] if selected else [],
        },
    }

    sciverse_hit = next(
        (
            attempt
            for attempt in attempts
            if attempt.get("source") == "sciverse_ai_ready" and attempt.get("status") == "hit"
        ),
        None,
    )
    payloads["use_sciverse_ai_ready"] = {
        "endpoint": "/api/library/papers/sciverse-ingest-batch",
        "method": "POST",
        "enabled": paper_id is not None and sciverse_hit is not None,
        "reason": None if paper_id else "register_candidate_first",
        "body": {
            "items": [{"paper_id": paper_id, "query": title or doi}] if paper_id else [],
            "dry_run": True,
            "write_policy": "dry_run",
        },
    }
    return payloads


def result_action_explanations(
    selected: dict[str, Any] | None,
    attempts: list[dict[str, Any]],
) -> dict[str, str]:
    explanations = {
        "register_candidate": "只登记候选 metadata；不会下载 PDF，也不会运行 MinerU。",
        "acquire_pdf": "显式 commit 后才会尝试获取允许来源的 PDF。",
        "process_with_mineru": "先显式获取或上传 PDF，才能运行 MinerU。",
        "use_sciverse_ai_ready": "Sciverse AI-ready 是远端解析资产，不会被当成本地 PDF。",
    }
    if selected and selected.get("source") in {"existing_asset", "local_doi_archive"}:
        explanations["process_with_mineru"] = "已找到本地 PDF，可 dry-run 或 commit 运行 MinerU。"
        explanations["acquire_pdf"] = "已有本地 PDF asset，不需要再获取 PDF。"
    elif selected and selected.get("asset_policy") == "unknown_public_pdf":
        explanations["acquire_pdf"] = "PDF URL 来源或授权不明确，需要人工审核后再处理。"
    elif any(attempt.get("source") == "sciverse_ai_ready" and attempt.get("status") == "hit" for attempt in attempts):
        explanations["use_sciverse_ai_ready"] = "可登记论文后摄取 Sciverse AI-ready parsed content；这不是 PDF 下载。"
    return explanations


def acquire_disabled_reason(selected: dict[str, Any] | None) -> str:
    if selected is None:
        return "pdf_not_found"
    if selected.get("source") in {"existing_asset", "local_doi_archive"}:
        return "already_has_local_pdf"
    if selected.get("asset_policy") == "unknown_public_pdf":
        return "unknown_public_pdf"
    return "policy_blocked"


def result_process_state(
    *,
    registry: LibraryRegistry,
    paper: dict[str, Any] | None,
    selected: dict[str, Any] | None,
) -> dict[str, Any]:
    if not paper:
        return {
            "registered": False,
            "paper_id": None,
            "preferred_pdf_asset_id": selected.get("asset_id") if selected else None,
            "mineru_ready": False,
            "evidence_ready": False,
            "evidence_count": 0,
            "artifact_manifest_status": "not_registered",
            "blocking_reason": "not_registered",
            "recommended_next_action": "register_candidate",
            "latest_mineru_job": None,
            "failure_report": [],
        }

    paper_id = str(paper["paper_id"])
    readiness = registry.paper_readiness(paper_id)
    manifest = registry.paper_artifact_manifest(paper_id)
    coverage = registry.evidence_coverage(paper_id)
    mineru_jobs = [
        job for job in registry.list_processing_jobs(paper_id) if job.get("tool") == "mineru"
    ]
    latest_mineru_job = mineru_jobs[-1] if mineru_jobs else None
    failure_report = [
        {
            "job_id": job["job_id"],
            "status": job["status"],
            "error": job.get("error"),
        }
        for job in mineru_jobs
        if job.get("status") in {"failed", "blocked"}
    ]
    preferred_pdf_asset_id = None
    if selected and selected.get("asset_id"):
        preferred_pdf_asset_id = selected.get("asset_id")
    else:
        pdf_assets = registry.list_pdf_assets(paper_id)
        preferred_pdf_asset_id = pdf_assets[0]["asset_id"] if pdf_assets else None
    return {
        "registered": True,
        "paper_id": paper_id,
        "preferred_pdf_asset_id": preferred_pdf_asset_id,
        "mineru_ready": bool(readiness["local_mineru_ready"]),
        "evidence_ready": bool(readiness["evidence_ready"]),
        "evidence_count": int(coverage.get("total") or 0),
        "artifact_manifest_status": manifest["status"],
        "blocking_reason": readiness["blocking_reason"],
        "recommended_next_action": readiness["recommended_next_action"],
        "latest_mineru_job": {
            "job_id": latest_mineru_job["job_id"],
            "status": latest_mineru_job["status"],
            "error": latest_mineru_job.get("error"),
        }
        if latest_mineru_job
        else None,
        "failure_report": failure_report,
    }
