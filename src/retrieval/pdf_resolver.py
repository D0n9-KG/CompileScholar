from __future__ import annotations

from pathlib import Path
from typing import Any

from retrieval.acquisition import first_pdf_candidate
from retrieval.registry import LibraryRegistry, normalize_doi
from retrieval.sources import LocalDoiArchive, SourceAdapterError, SourceCandidate


DEFAULT_PDF_SOURCES = [
    "existing_asset",
    "local_doi_archive",
    "local_upload",
    "openalex_oa_pdf",
    "arxiv",
    "crossref_pdf",
    "unpaywall_pdf",
    "sciverse_ai_ready",
]

PDF_SELECTION_PRIORITY = {
    "existing_asset": 10,
    "local_doi_archive": 20,
    "openalex_oa_pdf": 30,
    "arxiv": 40,
    "crossref_pdf": 50,
    "unpaywall_pdf": 60,
}


def resolve_pdf_attempts(
    *,
    registry: LibraryRegistry,
    doi: str | None,
    title: str | None,
    candidates: list[SourceCandidate],
    pdf_sources: list[str] | None = None,
    local_doi_index: Path | None = None,
    local_doi_archive_root: Path | None = None,
) -> list[dict[str, Any]]:
    requested_sources = normalize_sources(pdf_sources)
    attempts: list[dict[str, Any]] = []
    for source in requested_sources:
        if source == "existing_asset":
            attempts.append(existing_asset_attempt(registry=registry, doi=doi, title=title))
        elif source == "local_doi_archive":
            attempts.append(
                local_doi_archive_attempt(
                    doi=doi,
                    local_doi_index=local_doi_index,
                    local_doi_archive_root=local_doi_archive_root,
                )
            )
        elif source == "local_upload":
            attempts.append({"source": source, "status": "miss", "reason": "requires_user_upload"})
        elif source == "openalex_oa_pdf":
            attempts.append(candidate_pdf_attempt(source=source, candidates=candidates, candidate_source="openalex"))
        elif source == "crossref_pdf":
            attempts.append(candidate_pdf_attempt(source=source, candidates=candidates, candidate_source="crossref"))
        elif source == "unpaywall_pdf":
            attempts.append(candidate_pdf_attempt(source=source, candidates=candidates, candidate_source="unpaywall"))
        elif source == "arxiv":
            attempts.append(arxiv_pdf_attempt(candidates))
        elif source == "sciverse_ai_ready":
            attempts.append(sciverse_ai_ready_attempt(candidates))
        else:
            attempts.append({"source": source, "status": "blocked", "reason": "unknown_pdf_source"})
    return attempts


def select_pdf_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    hits = [
        attempt
        for attempt in attempts
        if attempt.get("status") == "hit" and attempt.get("source") in PDF_SELECTION_PRIORITY
    ]
    if not hits:
        return None
    return sorted(hits, key=lambda item: PDF_SELECTION_PRIORITY[str(item["source"])])[0]


def normalize_sources(sources: list[str] | None) -> list[str]:
    raw_sources = sources or DEFAULT_PDF_SOURCES
    normalized: list[str] = []
    for source in raw_sources:
        cleaned = source.strip().lower()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def existing_asset_attempt(
    *,
    registry: LibraryRegistry,
    doi: str | None,
    title: str | None,
) -> dict[str, Any]:
    paper = find_registered_paper(registry, doi=doi, title=title)
    if not paper:
        return {"source": "existing_asset", "status": "miss", "reason": "paper_not_registered"}
    pdf_assets = registry.list_pdf_assets(str(paper["paper_id"]))
    if not pdf_assets:
        return {
            "source": "existing_asset",
            "status": "miss",
            "paper_id": paper["paper_id"],
            "reason": "registered_without_pdf_asset",
        }
    asset = pdf_assets[0]
    return {
        "source": "existing_asset",
        "status": "hit",
        "paper_id": paper["paper_id"],
        "asset_id": asset.get("asset_id"),
        "source_kind": asset.get("source_kind"),
        "source_uri": asset.get("source_uri"),
        "asset_policy": "existing_local_pdf",
        "can_process_with_mineru": True,
    }


def local_doi_archive_attempt(
    *,
    doi: str | None,
    local_doi_index: Path | None,
    local_doi_archive_root: Path | None,
) -> dict[str, Any]:
    normalized = normalize_doi(doi)
    if not normalized:
        return {"source": "local_doi_archive", "status": "miss", "reason": "missing_doi"}
    if not local_doi_index or not local_doi_archive_root:
        return {"source": "local_doi_archive", "status": "miss", "reason": "not_configured"}
    try:
        hit = LocalDoiArchive(index_path=local_doi_index, archive_root=local_doi_archive_root).lookup(normalized)
    except SourceAdapterError as exc:
        return {"source": "local_doi_archive", "status": "blocked", "reason": str(exc)}
    if not hit:
        return {"source": "local_doi_archive", "status": "miss", "reason": "doi_not_found"}
    return {
        "source": "local_doi_archive",
        "status": "hit",
        "source_uri": hit.source_uri,
        "archive_path": str(hit.zip_path),
        "inner_path": hit.inner_path,
        "asset_policy": "local_available_pdf",
        "can_process_with_mineru": True,
    }


def candidate_pdf_attempt(
    *,
    source: str,
    candidates: list[SourceCandidate],
    candidate_source: str,
) -> dict[str, Any]:
    for candidate in candidates:
        if candidate.source_name != candidate_source or candidate.status != "ready":
            continue
        first_pdf = first_pdf_candidate(candidate.pdf_candidates)
        if not first_pdf:
            continue
        return {
            "source": source,
            "status": "hit",
            "url": first_pdf.get("url"),
            "pdf_source": first_pdf.get("source") or first_pdf.get("kind"),
            "license": candidate.license or first_pdf.get("license"),
            "open_access_status": candidate.open_access_status,
            "is_oa": bool(first_pdf.get("is_oa")) if first_pdf.get("is_oa") is not None else bool(candidate.license),
            "candidate_source_name": candidate.source_name,
            "candidate_score": candidate.candidate_score,
            "asset_policy": remote_pdf_asset_policy(candidate, first_pdf),
            "can_process_with_mineru": False,
            "risk_reasons": remote_pdf_risk_reasons(candidate, first_pdf),
        }
    return {"source": source, "status": "miss", "reason": "no_pdf_candidate"}


def arxiv_pdf_attempt(candidates: list[SourceCandidate]) -> dict[str, Any]:
    for candidate in candidates:
        first_pdf = first_pdf_candidate(candidate.pdf_candidates)
        if candidate.source_name == "arxiv" and first_pdf:
            return {
                "source": "arxiv",
                "status": "hit",
                "url": first_pdf.get("url"),
                "asset_policy": "arxiv_auto_allowed",
                "can_process_with_mineru": False,
                "risk_reasons": ["requires_explicit_pdf_acquire"],
            }
    return {"source": "arxiv", "status": "miss", "reason": "no_arxiv_pdf_candidate"}


def sciverse_ai_ready_attempt(candidates: list[SourceCandidate]) -> dict[str, Any]:
    for candidate in candidates:
        if candidate.source_name != "sciverse" or candidate.status != "ready":
            continue
        raw = candidate.raw if isinstance(candidate.raw, dict) else {}
        doc_id = raw.get("doc_id") or raw.get("unique_id") or candidate.source_record_id
        if not doc_id:
            continue
        content_accessible = raw.get("is_content_accessible")
        if content_accessible is False:
            return {
                "source": "sciverse_ai_ready",
                "status": "blocked",
                "doc_id": doc_id,
                "reason": "content_not_accessible",
            }
        return {
            "source": "sciverse_ai_ready",
            "status": "hit",
            "remote_asset_kind": "sciverse_ai_ready",
            "doc_id": doc_id,
            "content_status": "accessible" if content_accessible else "unknown",
            "asset_policy": "remote_parsed_asset_available_after_ingest",
            "can_process_with_mineru": False,
        }
    return {"source": "sciverse_ai_ready", "status": "miss", "reason": "no_sciverse_doc_id"}


def find_registered_paper(
    registry: LibraryRegistry,
    *,
    doi: str | None,
    title: str | None,
) -> dict[str, Any] | None:
    normalized = normalize_doi(doi)
    if normalized:
        paper = registry.find_paper_by_identifier("doi", normalized)
        if paper:
            return paper
        for item in registry.list_papers(include_archived=True):
            if normalize_doi(item.get("normalized_doi")) == normalized:
                return item
    title_key = normalize_title(title)
    if title_key:
        for item in registry.list_papers(include_archived=True):
            if normalize_title(item.get("title")) == title_key:
                return item
    return None


def normalize_title(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().split())


def remote_pdf_asset_policy(candidate: SourceCandidate, first_pdf: dict[str, Any]) -> str:
    if first_pdf.get("is_oa") is True:
        return "oa_auto_allowed"
    if candidate.open_access_status and candidate.open_access_status.lower() not in {"closed", "unknown"}:
        return "oa_auto_allowed"
    if candidate.license or first_pdf.get("license"):
        return "oa_auto_allowed"
    return "unknown_public_pdf"


def remote_pdf_risk_reasons(candidate: SourceCandidate, first_pdf: dict[str, Any]) -> list[str]:
    if remote_pdf_asset_policy(candidate, first_pdf) == "unknown_public_pdf":
        return ["requires_manual_pdf_review", "unknown_public_pdf"]
    return ["requires_explicit_pdf_acquire"]
