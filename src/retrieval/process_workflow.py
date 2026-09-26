from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from retrieval.acquisition import (
    acquire_remote_pdf_candidates,
    safe_remote_pdf_candidates,
    source_candidate_trace,
)
from retrieval.evidence_units import ensure_paper_evidence
from retrieval.processing import process_pdf_with_mineru
from retrieval.registry import LibraryRegistry, LibraryRegistryError
from retrieval._see_upstream import UpstreamConfig, download_bytes, run_mineru_smoke


DEFAULT_STAGES = ["pdf_acquire", "mineru", "evidence_index"]


@dataclass(frozen=True)
class ProcessWorkflowResult:
    paper_id: str
    status: str
    dry_run: bool
    stage_results: list[dict[str, Any]] = field(default_factory=list)
    blocking_reason: str | None = None
    recommended_next_action: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "status": self.status,
            "dry_run": self.dry_run,
            "stage_results": self.stage_results,
            "blocking_reason": self.blocking_reason,
            "recommended_next_action": self.recommended_next_action,
        }


def process_paper(
    *,
    registry: LibraryRegistry,
    paper_id: str,
    stages: list[str] | None = None,
    dry_run: bool = True,
    force: bool = False,
    preferred_source: list[str] | None = None,
    config: UpstreamConfig | None = None,
    run_mineru: Callable[..., Any] = run_mineru_smoke,
    download_pdf: Callable[[str, int], bytes] | None = None,
    download_timeout_seconds: int = 60,
) -> ProcessWorkflowResult:
    if not registry.get_paper(paper_id):
        raise ProcessWorkflowError(f"paper not found: {paper_id}")

    selected_stages = stages or DEFAULT_STAGES
    config = config or UpstreamConfig.from_env()
    stage_results: list[dict[str, Any]] = []
    pdf_assets = registry.list_pdf_assets(paper_id)
    pdf_asset = pdf_assets[0] if pdf_assets else None

    if "pdf_acquire" in selected_stages:
        pdf_stage = plan_pdf_acquire_stage(
            registry=registry,
            paper_id=paper_id,
            pdf_asset=pdf_asset,
            preferred_source=preferred_source or [],
        )
        stage_results.append(pdf_stage)
        if pdf_stage["status"] == "dedupe_cross_paper_conflict":
            return blocked_result(
                paper_id=paper_id,
                dry_run=dry_run,
                stage_results=stage_results,
                blocking_reason="dedupe_asset_on_other_paper",
                recommended_next_action="merge_or_rebind_asset",
            )
        if pdf_stage["status"] == "blocked":
            return blocked_result(
                paper_id=paper_id,
                dry_run=dry_run,
                stage_results=stage_results,
                blocking_reason=pdf_stage.get("blocking_reason") or "missing_pdf",
                recommended_next_action=pdf_stage.get("recommended_next_action")
                or "upload_pdf_or_confirm_local_candidate",
            )
        if pdf_stage["status"] == "needs_explicit_commit" and not dry_run:
            paper = registry.get_paper(paper_id)
            remote_candidates = safe_remote_pdf_candidates(registry.list_source_candidates(paper_id))
            if not paper or not remote_candidates:
                stage_results.append(
                    {
                        "stage": "pdf_acquire",
                        "status": "failed",
                        "blocking_reason": "missing_pdf_candidate",
                        "recommended_next_action": "refresh_source_candidates",
                    }
                )
                return blocked_result(
                    paper_id=paper_id,
                    dry_run=dry_run,
                    stage_results=stage_results,
                    blocking_reason="missing_pdf_candidate",
                    recommended_next_action="refresh_source_candidates",
                )
            remote_result = acquire_remote_pdf_candidates(
                registry=registry,
                paper=paper,
                candidates=remote_candidates,
                download_pdf=download_pdf or download_bytes,
                download_timeout_seconds=download_timeout_seconds,
                dedupe="existing_identity",
            )
            stage_results = stage_results[:-1] + remote_result.stage_results
            if remote_result.status == "dedupe_cross_paper_conflict":
                return blocked_result(
                    paper_id=paper_id,
                    dry_run=dry_run,
                    stage_results=stage_results,
                    blocking_reason="dedupe_asset_on_other_paper",
                    recommended_next_action="merge_or_rebind_asset",
                )
            if not remote_result.pdf_asset:
                return blocked_result(
                    paper_id=paper_id,
                    dry_run=dry_run,
                    stage_results=stage_results,
                    blocking_reason=remote_result.blocking_reason or "pdf_acquire_failed",
                    recommended_next_action=remote_result.recommended_next_action
                    or "try_alternate_pdf_source_or_upload",
                )
            pdf_asset = remote_result.pdf_asset

    if "mineru" in selected_stages:
        if pdf_asset is None:
            stage_results.append(
                {
                    "stage": "mineru",
                    "status": "blocked",
                    "blocking_reason": "missing_pdf",
                    "recommended_next_action": "upload_pdf_or_confirm_local_candidate",
                }
            )
            return blocked_result(
                paper_id=paper_id,
                dry_run=dry_run,
                stage_results=stage_results,
                blocking_reason="missing_pdf",
                recommended_next_action="upload_pdf_or_confirm_local_candidate",
            )
        if not config.mineru_server_url:
            stage_results.append(
                {
                    "stage": "mineru",
                    "status": "blocked",
                    "pdf_asset_id": pdf_asset["asset_id"],
                    "blocking_reason": "missing_mineru_config",
                    "recommended_next_action": "configure_mineru",
                }
            )
            return blocked_result(
                paper_id=paper_id,
                dry_run=dry_run,
                stage_results=stage_results,
                blocking_reason="missing_mineru_config",
                recommended_next_action="configure_mineru",
            )
        if dry_run:
            stage_results.append(
                {
                    "stage": "mineru",
                    "status": "would_process",
                    "pdf_asset_id": pdf_asset["asset_id"],
                    "force": force,
                }
            )
        else:
            processing_result = process_pdf_with_mineru(
                registry=registry,
                paper_id=paper_id,
                pdf_asset_id=pdf_asset["asset_id"],
                config=config,
                force=force,
                run_mineru=run_mineru,
            )
            stage_results.append(
                {
                    "stage": "mineru",
                    "status": processing_result.status,
                    "pdf_asset_id": pdf_asset["asset_id"],
                    "job_id": processing_result.job["job_id"],
                    "artifacts": len(processing_result.artifacts),
                    "reused": processing_result.reused,
                    "message": processing_result.message,
                }
            )
            if processing_result.status != "ready":
                return blocked_result(
                    paper_id=paper_id,
                    dry_run=dry_run,
                    stage_results=stage_results,
                    blocking_reason="mineru_not_ready",
                    recommended_next_action="inspect_processing_job",
                )

    if "evidence_index" in selected_stages:
        existing_evidence = registry.list_evidence_units(paper_id)
        if existing_evidence and not force:
            stage_results.append(
                {
                    "stage": "evidence_index",
                    "status": "already_ready",
                    "evidence_units": len(existing_evidence),
                }
            )
        elif dry_run:
            readiness = registry.paper_readiness(paper_id)
            if not readiness["local_mineru_ready"] and readiness["sciverse_ai_ready"] == "missing":
                stage_results.append(
                    {
                        "stage": "evidence_index",
                        "status": "blocked",
                        "blocking_reason": "missing_readable_body",
                        "recommended_next_action": "process_with_mineru_or_use_sciverse",
                    }
                )
                return blocked_result(
                    paper_id=paper_id,
                    dry_run=dry_run,
                    stage_results=stage_results,
                    blocking_reason="missing_readable_body",
                    recommended_next_action="process_with_mineru_or_use_sciverse",
                )
            stage_results.append({"stage": "evidence_index", "status": "would_index"})
        else:
            evidence = ensure_paper_evidence(registry=registry, paper_id=paper_id, force=force)
            stage_results.append(
                {
                    "stage": "evidence_index",
                    "status": "ready" if evidence.coverage["total"] else "missing",
                    "evidence_units": evidence.coverage["total"],
                }
            )
            if not evidence.coverage["total"]:
                return blocked_result(
                    paper_id=paper_id,
                    dry_run=dry_run,
                    stage_results=stage_results,
                    blocking_reason="missing_evidence",
                    recommended_next_action="inspect_source_artifacts",
                )

    return ProcessWorkflowResult(
        paper_id=paper_id,
        status="ready",
        dry_run=dry_run,
        stage_results=stage_results,
        blocking_reason=None,
        recommended_next_action="ready_for_experiment",
    )


def plan_pdf_acquire_stage(
    *,
    registry: LibraryRegistry,
    paper_id: str,
    pdf_asset: dict[str, Any] | None,
    preferred_source: list[str],
) -> dict[str, Any]:
    if pdf_asset is not None:
        return {
            "stage": "pdf_acquire",
            "status": "already_ready",
            "pdf_asset_id": pdf_asset["asset_id"],
            "asset_policy": "local_available_pdf",
            "export_policy": "local_pdf_allowed",
            "source_kind": pdf_asset.get("source_kind"),
            "recommended_next_action": "process_with_mineru",
        }
    dedupe_diagnostic = registry.find_duplicate_asset_diagnostics(paper_id)
    if dedupe_diagnostic:
        return {
            "stage": "pdf_acquire",
            "status": "dedupe_cross_paper_conflict",
            "blocking_reason": "dedupe_asset_on_other_paper",
            "recommended_next_action": "merge_or_rebind_asset",
            "conflict": dedupe_diagnostic,
        }
    candidates = registry.list_source_candidates(paper_id)
    local_ready = [
        candidate
        for candidate in candidates
        if candidate.get("source_name") == "local_doi_archive" and candidate.get("status") == "ready"
    ]
    if local_ready:
        return {
            "stage": "pdf_acquire",
            "status": "needs_candidate_confirmation",
            "candidate_ids": [candidate["candidate_id"] for candidate in local_ready],
            "asset_policy": "local_available_pdf",
            "export_policy": "local_pdf_allowed",
            "recommended_next_action": "confirm_local_candidate",
        }
    remote_allowed = safe_remote_pdf_candidates(candidates)
    if remote_allowed:
        return {
            "stage": "pdf_acquire",
            "status": "needs_explicit_commit",
            "candidate_ids": [candidate["candidate_id"] for candidate in remote_allowed],
            "candidates": [source_candidate_trace(candidate) for candidate in remote_allowed],
            "asset_policy": remote_allowed[0].get("asset_policy"),
            "export_policy": "remote_pdf_allowed_after_commit",
            "recommended_next_action": "commit_pdf_acquire",
        }
    alternate_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("candidate_pdf_url") or candidate.get("pdf_url") or candidate.get("source_name") == "sciverse"
    ]
    if alternate_candidates:
        return {
            "stage": "pdf_acquire",
            "status": "blocked",
            "blocking_reason": "missing_pdf",
            "recommended_next_action": "confirm_alternate_candidate_or_upload_pdf",
            "candidate_ids": [candidate["candidate_id"] for candidate in alternate_candidates],
            "candidates": [source_candidate_trace(candidate) for candidate in alternate_candidates],
            "risk_reasons": sorted(
                {
                    risk
                    for candidate in alternate_candidates
                    for risk in candidate.get("risk_reasons", [])
                }
            ),
            "next_actions": ["manual_upload", "register_candidate_only", "try_sciverse"],
        }
    risk_reasons = ["missing_pdf"]
    if any(source in {"openalex_oa_pdf", "crossref_pdf", "arxiv"} for source in preferred_source):
        risk_reasons.append("public_pdf_download_not_enabled")
    return {
        "stage": "pdf_acquire",
        "status": "blocked",
        "blocking_reason": "missing_pdf",
        "recommended_next_action": "upload_pdf_or_confirm_local_candidate",
        "risk_reasons": risk_reasons,
    }


def blocked_result(
    *,
    paper_id: str,
    dry_run: bool,
    stage_results: list[dict[str, Any]],
    blocking_reason: str,
    recommended_next_action: str,
) -> ProcessWorkflowResult:
    return ProcessWorkflowResult(
        paper_id=paper_id,
        status="blocked",
        dry_run=dry_run,
        stage_results=stage_results,
        blocking_reason=blocking_reason,
        recommended_next_action=recommended_next_action,
    )


class ProcessWorkflowError(RuntimeError):
    pass
