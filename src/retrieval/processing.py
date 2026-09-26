from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from retrieval.registry import LibraryRegistry
from retrieval._see_upstream import MINERU_FORM_FIELDS, UpstreamConfig, run_mineru_smoke


@dataclass(frozen=True)
class ProcessingResult:
    status: str
    job: dict[str, Any]
    artifacts: list[dict[str, Any]]
    reused: bool = False
    message: str | None = None


def process_pdf_with_mineru(
    *,
    registry: LibraryRegistry,
    paper_id: str,
    pdf_asset_id: str,
    config: UpstreamConfig,
    timeout_seconds: int = 600,
    force: bool = False,
    run_mineru: Callable[..., Any] = run_mineru_smoke,
) -> ProcessingResult:
    assets = registry.list_pdf_assets(paper_id)
    asset = next((item for item in assets if item["asset_id"] == pdf_asset_id), None)
    if asset is None:
        raise ProcessingError(f"pdf asset not found for paper: {pdf_asset_id}")

    tool_config = {
        "mineru_form_fields": MINERU_FORM_FIELDS,
        "mineru_server_url": config.mineru_server_url or "",
    }
    existing = registry.find_ready_processing_job(
        pdf_asset_id=pdf_asset_id,
        tool="mineru",
        tool_config=tool_config,
    )
    if existing and not force:
        return ProcessingResult(
            status="ready",
            job=existing,
            artifacts=registry.list_mineru_artifacts(existing["job_id"]),
            reused=True,
            message="已复用 existing ready MinerU processing job。",
        )

    pdf_path = registry.resolve_local_path(asset["local_path"])
    if not config.mineru_server_url:
        job = registry.create_processing_job(
            paper_id=paper_id,
            pdf_asset_id=pdf_asset_id,
            tool="mineru",
            tool_config=tool_config,
            status="blocked",
            server_url_redacted=None,
            error="缺少配置：SCIEVO_MINERU_SERVER_URL",
        )
        return ProcessingResult(
            status="blocked",
            job=job,
            artifacts=[],
            message="缺少配置：SCIEVO_MINERU_SERVER_URL",
        )

    output_dir = (
        registry.library_root
        / "papers"
        / paper_id
        / "mineru"
        / f"mineru_{pdf_asset_id.lower()}"
    )
    manifest = run_mineru(
        pdf_path=pdf_path,
        output_dir=output_dir,
        config=config,
        run_id=f"mineru_{pdf_asset_id.lower()}",
        timeout_seconds=timeout_seconds,
    )
    status = manifest.stages["mineru"]
    error = status.detail if status.state.value in {"failed", "blocked"} else None
    job = registry.create_processing_job(
        paper_id=paper_id,
        pdf_asset_id=pdf_asset_id,
        tool="mineru",
        tool_config=tool_config,
        status=status.state.value,
        server_url_redacted=redact_url(config.mineru_server_url),
        error=error,
    )

    artifacts: list[dict[str, Any]] = []
    if status.state.value == "ready":
        for kind, local_path in sorted(manifest.artifacts.items()):
            path = Path(local_path)
            if path.is_file() or path.is_dir():
                artifacts.append(
                    registry.register_mineru_artifact(
                        job_id=job["job_id"],
                        kind=kind,
                        path=path,
                        status="ready",
                    )
                )
    registry.record_provenance_event(
        action="process_pdf",
        source="mineru",
        inputs={"paper_id": paper_id, "pdf_asset_id": pdf_asset_id},
        outputs={"job_id": job["job_id"], "status": status.state.value},
        error_summary=error,
    )
    return ProcessingResult(
        status=status.state.value,
        job=job,
        artifacts=artifacts,
        message=status.detail,
    )


def redact_url(value: str | None) -> str | None:
    if not value:
        return None
    return value.rstrip("/")


class ProcessingError(RuntimeError):
    pass
