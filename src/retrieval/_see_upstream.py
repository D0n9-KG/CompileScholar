from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import zipfile
from datetime import UTC, datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


SECRET_TOKENS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASS")
MINERU_FORM_FIELDS = {
    "output_dir": "./output",
    "lang_list": "en",
    "backend": "vlm-vllm-async-engine",
    "parse_method": "auto",
    "formula_enable": "true",
    "table_enable": "true",
    "return_md": "true",
    "return_middle_json": "false",
    "return_model_output": "false",
    "return_content_list": "true",
    "return_images": "true",
    "response_format_zip": "true",
    "start_page_id": "0",
    "end_page_id": "99999",
}
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MANIFEST_FILENAME = "upstream_run_manifest.json"
SCIMKG_MATERIALIZE_MANIFEST_FILENAME = "scimkg_materialize_manifest.json"
USER_AGENT = "sci-evo-extract/0.1"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class UpstreamError(RuntimeError):
    pass


class StageState(str, Enum):
    pending = "pending"
    ready = "ready"
    blocked = "blocked"
    failed = "failed"


class StageStatus(BaseModel):
    state: StageState = StageState.pending
    reason: str | None = None
    detail: str | None = None


class UpstreamConfig(BaseModel):
    scimkg_base_url: str | None = None
    postgres_uri: str | None = None
    postgres_host: str | None = None
    postgres_port: str | None = None
    postgres_db: str | None = None
    postgres_user: str | None = None
    postgres_password: str | None = None
    minio_endpoint: str | None = None
    minio_access_key: str | None = None
    minio_secret_key: str | None = None
    minio_bucket: str | None = None
    minio_image_bucket: str | None = None
    mineru_server_url: str | None = None

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "UpstreamConfig":
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
        return cls(
            postgres_uri=_env("SCIEVO_UPSTREAM_POSTGRES_URI"),
            postgres_host=_env("SCIEVO_UPSTREAM_POSTGRES_HOST"),
            postgres_port=_env("SCIEVO_UPSTREAM_POSTGRES_PORT"),
            postgres_db=_env("SCIEVO_UPSTREAM_POSTGRES_DB"),
            postgres_user=_env("SCIEVO_UPSTREAM_POSTGRES_USER"),
            postgres_password=_env("SCIEVO_UPSTREAM_POSTGRES_PASSWORD"),
            minio_endpoint=_env("SCIEVO_UPSTREAM_MINIO_ENDPOINT"),
            minio_access_key=_env("SCIEVO_UPSTREAM_MINIO_ACCESS_KEY"),
            minio_secret_key=_env("SCIEVO_UPSTREAM_MINIO_SECRET_KEY"),
            minio_bucket=_env("SCIEVO_UPSTREAM_MINIO_BUCKET"),
            minio_image_bucket=_env("SCIEVO_UPSTREAM_MINIO_IMAGE_BUCKET"),
            mineru_server_url=_env("SCIEVO_MINERU_SERVER_URL"),
            scimkg_base_url=_env("SCIEVO_SCIMKG_BASE_URL"),
        )

    def redacted_summary(self) -> dict[str, Any]:
        return {
            key: _redact(key, value)
            for key, value in self.model_dump().items()
            if value is not None
        }

    def missing_readiness_keys(self) -> list[str]:
        missing: list[str] = []
        if not self.postgres_uri and not all(
            [
                self.postgres_host,
                self.postgres_port,
                self.postgres_db,
                self.postgres_user,
                self.postgres_password,
            ]
        ):
            missing.append(
                "SCIEVO_UPSTREAM_POSTGRES_URI 或完整 SCIEVO_UPSTREAM_POSTGRES_*"
            )
        if not self.minio_endpoint:
            missing.append("SCIEVO_UPSTREAM_MINIO_ENDPOINT")
        if not self.minio_access_key:
            missing.append("SCIEVO_UPSTREAM_MINIO_ACCESS_KEY")
        if not self.minio_secret_key:
            missing.append("SCIEVO_UPSTREAM_MINIO_SECRET_KEY")
        if not self.minio_bucket:
            missing.append("SCIEVO_UPSTREAM_MINIO_BUCKET")
        if not self.minio_image_bucket:
            missing.append("SCIEVO_UPSTREAM_MINIO_IMAGE_BUCKET")
        return missing


class UpstreamReadiness(BaseModel):
    config: dict[str, Any]
    postgres: StageStatus
    minio: StageStatus
    mineru: StageStatus


class ScimkgMaterializeSummary(BaseModel):
    project_id: int
    project_name: str | None = None
    status: str | None = None
    source_count: int = 0
    materialized_papers: int = 0
    ready_content_lists: int = 0
    ready_markdown: int = 0
    ready_pdfs: int = 0
    ready_kg_nodes: int = 0
    ready_kg_edges: int = 0
    warnings: list[str] = Field(default_factory=list)


class ScimkgMaterializeManifest(BaseModel):
    run_id: str
    scimkg_base_url: str
    project_id: int
    output_dir: Path
    project: dict[str, Any] = Field(default_factory=dict)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    summary: ScimkgMaterializeSummary
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)


def readiness_lines(readiness: UpstreamReadiness) -> list[str]:
    lines = ["远端上游 readiness 检查"]
    for label, status in (
        ("PostgreSQL", readiness.postgres),
        ("MinIO", readiness.minio),
        ("MinerU", readiness.mineru),
    ):
        detail = f" - {status.detail}" if status.detail else ""
        reason = f" ({status.reason})" if status.reason else ""
        lines.append(f"- {label}: {status.state.value}{reason}{detail}")
    if readiness.config:
        lines.append("配置摘要：")
        for key in sorted(readiness.config):
            lines.append(f"  - {key}: {readiness.config[key]}")
    else:
        lines.append("配置摘要：未发现远端上游配置。")
    return lines


def materialize_scimkg_project(
    *,
    base_url: str,
    project_id: int,
    output_dir: Path,
    run_id: str | None = None,
    max_nodes: int = 500,
    timeout_seconds: int = 180,
    download_pdfs: bool = False,
    download_timeout_seconds: int = 180,
    request_json_fn: Callable[[str, int], Any] | None = None,
    download_bytes_fn: Callable[[str, int], bytes] | None = None,
) -> ScimkgMaterializeManifest:
    """Materialize an already-built Sci-MKG project into local artifacts.

    This is read-only against Sci-MKG/MinIO. It consumes signed URLs returned by
    Sci-MKG and writes only to the local output directory.
    """
    base = _validated_base_url(base_url, "SCIEVO_SCIMKG_BASE_URL")
    request_json_call = request_json_fn or request_json
    download_bytes_call = download_bytes_fn or download_bytes
    output_dir.mkdir(parents=True, exist_ok=True)
    project_dir = output_dir / f"project_{project_id}"
    project_dir.mkdir(parents=True, exist_ok=True)

    project = _expect_object(
        request_json_call(f"{base}/api/projects/{project_id}", timeout_seconds),
        f"Sci-MKG project {project_id}",
    )
    sources_payload = request_json_call(
        f"{base}/api/projects/{project_id}/sources",
        timeout_seconds,
    )
    if not isinstance(sources_payload, list):
        raise UpstreamError("Sci-MKG sources response must be a JSON array.")
    sources = [source for source in sources_payload if isinstance(source, dict)]
    completed_sources = [
        source for source in sources if source.get("insert_status") == "completed"
    ]

    graph = try_request_json(
        request_json_call,
        f"{base}/api/projects/{project_id}/graph/full?max_nodes={max_nodes}",
        timeout_seconds,
    )
    graph_nodes, graph_edges, graph_warning = normalize_scimkg_graph(graph)

    now = _utc_now()
    materialized_run_id = run_id or f"scimkg_project_{project_id}_{_utc_stamp()}"
    papers_dir = project_dir / "papers"
    graph_dir = project_dir / "graph"
    paper_rows: list[dict[str, Any]] = []
    mineru_registry_rows: list[dict[str, Any]] = []
    kg_registry_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    ready_markdown = 0
    ready_pdfs = 0
    ready_content_lists = 0

    for source in completed_sources:
        file_id = _int_value(source.get("file_id"), "file_id")
        file_name = str(source.get("file_name") or f"file_{file_id}.pdf")
        paper_id = stable_id("PPR_", "Sci-MKG", str(project_id), str(file_id), file_name)
        paper_dir = papers_dir / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)

        preview_payload = try_request_json(
            request_json_call,
            f"{base}/api/files/{file_id}/preview",
            timeout_seconds,
        )
        download_payload = try_request_json(
            request_json_call,
            f"{base}/api/files/{file_id}/download",
            timeout_seconds,
        )
        preview_url = signed_url(preview_payload)
        pdf_url = signed_url(download_payload)

        markdown_path: Path | None = None
        content_list_path: Path | None = None
        pdf_path: Path | None = None
        title = title_from_filename(file_name)

        if preview_url:
            markdown_path = paper_dir / "document.md"
            try:
                markdown_bytes = download_bytes_call(preview_url, download_timeout_seconds)
                markdown_path.write_bytes(markdown_bytes)
                markdown_text = markdown_bytes.decode("utf-8", errors="ignore")
                ready_markdown += 1
                title = title_from_markdown(markdown_text, title)
                content_list = markdown_to_content_list(markdown_text)
                if content_list:
                    content_list_path = paper_dir / "content_list.json"
                    write_json(content_list_path, content_list)
                    ready_content_lists += 1
            except UpstreamError as exc:
                issue_rows.append(
                    issue_row(
                        component="scimkg_preview",
                        file_id=file_id,
                        paper_id=paper_id,
                        summary="Sci-MKG markdown preview download failed.",
                        detail=str(exc),
                    )
                )
        else:
            issue_rows.append(
                issue_row(
                    component="scimkg_preview",
                    file_id=file_id,
                    paper_id=paper_id,
                    summary="Sci-MKG preview endpoint did not return a signed URL.",
                )
            )

        if download_pdfs and pdf_url:
            pdf_path = paper_dir / safe_filename(file_name, f"file_{file_id}.pdf")
            try:
                pdf_path.write_bytes(
                    download_bytes_call(pdf_url, download_timeout_seconds)
                )
                ready_pdfs += 1
            except UpstreamError as exc:
                issue_rows.append(
                    issue_row(
                        component="scimkg_pdf",
                        file_id=file_id,
                        paper_id=paper_id,
                        summary="Sci-MKG PDF download failed.",
                        detail=str(exc),
                    )
                )

        paper_rows.append(
            {
                "paper_id": paper_id,
                "source_file_id": file_id,
                "source_project_id": project_id,
                "title": title,
                "file_name": file_name,
                "content_list_path": str(content_list_path) if content_list_path else None,
                "markdown_path": str(markdown_path) if markdown_path else None,
                "pdf_path": str(pdf_path) if pdf_path and pdf_path.exists() else None,
                "metadata": {
                    "source": "Sci-MKG",
                    "source_record": source,
                    "preview_url_available": bool(preview_url),
                    "pdf_url_available": bool(pdf_url),
                },
            }
        )
        mineru_registry_rows.append(
            {
                "artifact_id": stable_id("GAF_", "Sci-MKG", "content_list", paper_id),
                "paper_id": paper_id,
                "tool_name": "MinerU",
                "artifact_kind": "content_list",
                "uri": content_list_path.resolve().as_uri() if content_list_path else None,
                "status": "ready" if content_list_path else "failed",
                "created_at": now,
                "notes": (
                    "Generated local content_list from Sci-MKG markdown preview."
                    if content_list_path
                    else "No local content_list could be materialized from Sci-MKG."
                ),
            }
        )

    nodes_path = graph_dir / "kg_nodes.jsonl"
    edges_path = graph_dir / "kg_edges.jsonl"
    if graph_nodes:
        write_jsonl_dicts(nodes_path, graph_nodes)
    if graph_edges:
        write_jsonl_dicts(edges_path, graph_edges)
    if graph_warning:
        issue_rows.append(
            issue_row(
                component="scimkg_graph",
                file_id=None,
                paper_id=None,
                summary=graph_warning,
            )
        )

    kg_status = "ready" if graph_nodes or graph_edges else "failed"
    kg_registry_rows.extend(
        [
            {
                "artifact_id": stable_id("GAF_", "Sci-MKG", "kg_nodes", str(project_id)),
                "paper_id": None,
                "tool_name": "RAGAnything",
                "artifact_kind": "multimodal_kg_nodes",
                "uri": nodes_path.resolve().as_uri() if graph_nodes else None,
                "status": kg_status,
                "created_at": now,
                "notes": "Exported from Sci-MKG graph/full endpoint.",
            },
            {
                "artifact_id": stable_id("GAF_", "Sci-MKG", "kg_edges", str(project_id)),
                "paper_id": None,
                "tool_name": "RAGAnything",
                "artifact_kind": "multimodal_kg_edges",
                "uri": edges_path.resolve().as_uri() if graph_edges else None,
                "status": kg_status,
                "created_at": now,
                "notes": "Exported from Sci-MKG graph/full endpoint.",
            },
        ]
    )

    paths = {
        "paper_manifest": project_dir / "paper_manifest.jsonl",
        "mineru_registry": project_dir / "mineru_registry.jsonl",
        "kg_registry": project_dir / "kg_registry.jsonl",
        "upstream_issue_report": project_dir / "upstream_issue_report.json",
        "scimkg_manifest": project_dir / SCIMKG_MATERIALIZE_MANIFEST_FILENAME,
    }
    write_jsonl_dicts(paths["paper_manifest"], paper_rows)
    write_jsonl_dicts(paths["mineru_registry"], mineru_registry_rows)
    write_jsonl_dicts(paths["kg_registry"], kg_registry_rows)
    write_json(
        paths["upstream_issue_report"],
        {
            "issues": issue_rows,
            "summary": {
                "issue_count": len(issue_rows),
                "warnings": len(issue_rows),
            },
        },
    )

    warnings = [row["summary"] for row in issue_rows]
    summary = ScimkgMaterializeSummary(
        project_id=project_id,
        project_name=project.get("name"),
        status=project.get("status"),
        source_count=len(completed_sources),
        materialized_papers=len(paper_rows),
        ready_content_lists=ready_content_lists,
        ready_markdown=ready_markdown,
        ready_pdfs=ready_pdfs,
        ready_kg_nodes=len(graph_nodes),
        ready_kg_edges=len(graph_edges),
        warnings=warnings,
    )
    manifest = ScimkgMaterializeManifest(
        run_id=materialized_run_id,
        scimkg_base_url=base,
        project_id=project_id,
        output_dir=project_dir,
        project=project,
        sources=completed_sources,
        artifacts={key: str(value) for key, value in paths.items()},
        summary=summary,
    )
    write_scimkg_materialize_manifest(manifest, paths["scimkg_manifest"])
    return manifest


class UpstreamRunManifest(BaseModel):
    run_id: str
    pdf_path: Path
    output_dir: Path
    pdf_sha256: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    stages: dict[str, StageStatus] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)

    @field_validator("run_id")
    @classmethod
    def run_id_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("run_id must not be blank")
        return value


def build_readiness(config: UpstreamConfig) -> UpstreamReadiness:
    missing = config.missing_readiness_keys()
    postgres = StageStatus(state=StageState.ready)
    minio = StageStatus(state=StageState.ready)
    if missing:
        postgres_missing = [item for item in missing if "POSTGRES" in item]
        minio_missing = [item for item in missing if "MINIO" in item]
        if postgres_missing:
            postgres = StageStatus(
                state=StageState.blocked,
                reason="missing_config",
                detail="缺少配置：" + "、".join(postgres_missing),
            )
        if minio_missing:
            minio = StageStatus(
                state=StageState.blocked,
                reason="missing_config",
                detail="缺少配置：" + "、".join(minio_missing),
            )

    mineru = (
        StageStatus(state=StageState.ready)
        if config.mineru_server_url
        else StageStatus(
            state=StageState.blocked,
            reason="missing_config",
            detail="缺少配置：SCIEVO_MINERU_SERVER_URL",
        )
    )
    return UpstreamReadiness(
        config=config.redacted_summary(),
        postgres=postgres,
        minio=minio,
        mineru=mineru,
    )


def validate_pdf_input(path: Path) -> str:
    if not path.exists():
        raise UpstreamError(f"PDF 文件不存在：{path}")
    if not path.is_file():
        raise UpstreamError(f"PDF 路径不是文件：{path}")
    try:
        with path.open("rb") as handle:
            header = handle.read(5)
    except OSError as exc:
        raise UpstreamError(f"PDF 文件不可读：{path}") from exc
    if header != b"%PDF-":
        raise UpstreamError(f"输入文件不是有效 PDF：{path}")
    return sha256_file(path)


def create_run_manifest(
    *,
    pdf_path: Path,
    output_dir: Path,
    config: UpstreamConfig,
    run_id: str | None = None,
) -> UpstreamRunManifest:
    checksum = validate_pdf_input(pdf_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    readiness = build_readiness(config)
    manifest = UpstreamRunManifest(
        run_id=run_id or stable_run_id(pdf_path, checksum),
        pdf_path=pdf_path,
        output_dir=output_dir,
        pdf_sha256=checksum,
        config=config.redacted_summary(),
        stages={
            "readiness": aggregate_readiness_stage(readiness),
            "mineru": StageStatus(state=StageState.pending),
        },
    )
    write_run_manifest(manifest, output_dir / MANIFEST_FILENAME)
    return manifest


def run_mineru_smoke(
    *,
    pdf_path: Path,
    output_dir: Path,
    config: UpstreamConfig,
    run_id: str | None = None,
    timeout_seconds: int = 600,
    post_file_parse: Callable[[Path, str, int], tuple[bytes, str | None]]
    | None = None,
) -> UpstreamRunManifest:
    manifest = create_run_manifest(
        pdf_path=pdf_path,
        output_dir=output_dir,
        config=config,
        run_id=run_id,
    )
    manifest_path = output_dir / MANIFEST_FILENAME
    if not config.mineru_server_url:
        manifest.stages["mineru"] = StageStatus(
            state=StageState.blocked,
            reason="missing_config",
            detail="缺少配置：SCIEVO_MINERU_SERVER_URL",
        )
        write_run_manifest(manifest, manifest_path)
        return manifest

    post = post_file_parse or post_mineru_file_parse
    try:
        zip_bytes, content_type = post(
            pdf_path,
            config.mineru_server_url,
            timeout_seconds,
        )
        materialized = materialize_mineru_response(
            response_bytes=zip_bytes,
            output_dir=output_dir,
            content_type=content_type,
        )
    except UpstreamError as exc:
        manifest.stages["mineru"] = StageStatus(
            state=StageState.failed,
            reason="mineru_error",
            detail=str(exc),
        )
        write_run_manifest(manifest, manifest_path)
        return manifest

    manifest.artifacts.update(materialized)
    if "mineru_content_list" not in materialized:
        manifest.stages["mineru"] = StageStatus(
            state=StageState.failed,
            reason="missing_artifact",
            detail="MinerU 响应未包含或无法生成 content_list JSON。",
        )
    else:
        manifest.stages["mineru"] = StageStatus(
            state=StageState.ready,
            detail="MinerU 响应已保存并物化 markdown/content_list。",
        )
    write_run_manifest(manifest, manifest_path)
    return manifest


def post_mineru_file_parse(
    pdf_path: Path,
    server_url: str,
    timeout_seconds: int,
) -> tuple[bytes, str | None]:
    parsed = urlsplit(server_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise UpstreamError(
            "SCIEVO_MINERU_SERVER_URL 必须是 http(s) URL，例如 http://host:port"
        )
    api_url = f"{server_url.rstrip('/')}/file_parse"
    body, content_type = build_multipart_payload(pdf_path, MINERU_FORM_FIELDS)
    request = Request(
        api_url,
        data=body,
        headers={
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_content_type = response.headers.get("Content-Type")
            return response.read(), response_content_type
    except HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise UpstreamError(f"MinerU HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise UpstreamError(f"MinerU 请求失败：{exc.reason}") from exc
    except TimeoutError as exc:
        raise UpstreamError(f"MinerU 请求超时：{timeout_seconds}s") from exc
    except OSError as exc:
        raise UpstreamError(f"MinerU 请求失败：{exc}") from exc


def build_multipart_payload(
    pdf_path: Path,
    fields: dict[str, str],
) -> tuple[bytes, str]:
    boundary = f"----scievo-{os.urandom(8).hex()}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(
                    "utf-8"
                ),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    filename = pdf_path.name.replace('"', "")
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                'Content-Disposition: form-data; name="files"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8"),
            b"Content-Type: application/pdf\r\n\r\n",
            pdf_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def materialize_mineru_response(
    *,
    response_bytes: bytes,
    output_dir: Path,
    content_type: str | None,
) -> dict[str, str]:
    if zipfile.is_zipfile(BytesIO(response_bytes)):
        return materialize_mineru_zip(
            zip_bytes=response_bytes,
            output_dir=output_dir,
            content_type=content_type,
        )
    if content_type and "zip" in content_type.lower():
        return materialize_mineru_zip(
            zip_bytes=response_bytes,
            output_dir=output_dir,
            content_type=content_type,
        )
    return materialize_mineru_json(
        response_bytes=response_bytes,
        output_dir=output_dir,
        content_type=content_type,
    )


def materialize_mineru_zip(
    *,
    zip_bytes: bytes,
    output_dir: Path,
    content_type: str | None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "mineru_response.zip"
    zip_path.write_bytes(zip_bytes)
    if content_type and "zip" not in content_type.lower():
        if not zipfile.is_zipfile(zip_path):
            raise UpstreamError(f"MinerU 返回不是 zip：content-type={content_type}")
    if not zipfile.is_zipfile(zip_path):
        raise UpstreamError("MinerU 返回内容不是有效 zip。")

    extract_dir = output_dir / "mineru_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    safe_extract_zip(zip_path, extract_dir)

    artifacts = {
        "mineru_zip": str(zip_path),
        "mineru_extract_dir": str(extract_dir),
    }
    markdown = find_markdown(extract_dir)
    content_list = find_content_list(extract_dir)
    images_dir = find_images_dir(extract_dir)
    middle_json = find_named_json(extract_dir, "middle")
    model_json = find_named_json(extract_dir, "model")
    layout_json = find_named_json(extract_dir, "layout")
    tables_dir = find_named_dir(extract_dir, {"table", "tables"})
    equations_dir = find_named_dir(extract_dir, {"equation", "equations", "formula", "formulas"})
    if markdown:
        artifacts["mineru_markdown"] = str(markdown)
    if content_list:
        artifacts["mineru_content_list"] = str(content_list)
    if images_dir:
        artifacts["mineru_images_dir"] = str(images_dir)
    if middle_json:
        artifacts["mineru_middle_json"] = str(middle_json)
    if model_json:
        artifacts["mineru_model_json"] = str(model_json)
    if layout_json:
        artifacts["mineru_layout_json"] = str(layout_json)
    if tables_dir:
        artifacts["mineru_tables_dir"] = str(tables_dir)
    if equations_dir:
        artifacts["mineru_equations_dir"] = str(equations_dir)
    return artifacts


def materialize_mineru_json(
    *,
    response_bytes: bytes,
    output_dir: Path,
    content_type: str | None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mineru_response.json"
    try:
        payload = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UpstreamError(
            f"MinerU 返回不是 zip，也不是有效 JSON：content-type={content_type}"
        ) from exc

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not isinstance(payload, dict):
        raise UpstreamError("MinerU JSON 响应不是对象。")

    status = str(payload.get("status") or "").lower()
    if status in {"failed", "failure", "error"}:
        error = payload.get("error") or payload.get("detail") or payload.get("message")
        raise UpstreamError(f"MinerU JSON status={status}: {error}")

    result_name, result = pick_mineru_json_result(payload)
    result_dir = output_dir / "mineru_json_extract" / safe_path_name(result_name)
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "mineru_response_json": str(json_path),
        "mineru_extract_dir": str(result_dir),
    }

    markdown = result.get("md_content") or result.get("markdown") or result.get("md")
    if isinstance(markdown, str) and markdown.strip():
        markdown_path = result_dir / "document.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        artifacts["mineru_markdown"] = str(markdown_path)

    content_list = extract_mineru_content_list(result)
    if content_list is None and isinstance(markdown, str) and markdown.strip():
        content_list = [
            {
                "type": "text",
                "text": markdown,
                "source": "mineru_router_json.md_content",
            }
        ]
    if content_list is not None:
        content_list_path = result_dir / "content_list.json"
        content_list_path.write_text(
            json.dumps(content_list, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifacts["mineru_content_list"] = str(content_list_path)

    for source_key, artifact_key, filename in (
        ("middle_json", "mineru_middle_json", "middle.json"),
        ("middle", "mineru_middle_json", "middle.json"),
        ("model_json", "mineru_model_json", "model.json"),
        ("model", "mineru_model_json", "model.json"),
        ("layout_json", "mineru_layout_json", "layout.json"),
        ("layout", "mineru_layout_json", "layout.json"),
    ):
        if artifact_key in artifacts or source_key not in result:
            continue
        value = result[source_key]
        if isinstance(value, (dict, list)):
            structured_path = result_dir / filename
            structured_path.write_text(
                json.dumps(value, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            artifacts[artifact_key] = str(structured_path)

    result_manifest = {
        "task_id": payload.get("task_id"),
        "status": payload.get("status"),
        "backend": payload.get("backend"),
        "version": payload.get("version"),
        "file_names": payload.get("file_names"),
        "selected_result": result_name,
        "result_keys": sorted(result.keys()),
        "content_list_synthesized_from_markdown": (
            "mineru_content_list" in artifacts
            and isinstance(markdown, str)
            and "content_list" not in result
        ),
    }
    result_manifest_path = result_dir / "result_manifest.json"
    result_manifest_path.write_text(
        json.dumps(result_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifacts["mineru_result_manifest"] = str(result_manifest_path)
    return artifacts


def pick_mineru_json_result(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    results = payload.get("results")
    if isinstance(results, dict):
        for name, result in results.items():
            if isinstance(result, dict):
                return str(name), result
    if isinstance(results, list):
        for index, result in enumerate(results):
            if isinstance(result, dict):
                return f"result_{index}", result
    if any(key in payload for key in ("md_content", "markdown", "md", "content_list")):
        return "result", payload
    raise UpstreamError("MinerU JSON 未包含可物化的 results/md_content。")


def extract_mineru_content_list(result: dict[str, Any]) -> object | None:
    for key in ("content_list", "content_list_json", "content"):
        value = result.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return [{"type": "text", "text": value, "source": f"mineru_json.{key}"}]
        return value
    return None


def safe_path_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned[:120] or "result"


def safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_root = extract_dir.resolve()
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for info in archive.infolist():
                target = (extract_dir / info.filename).resolve()
                if target != extract_root and not target.is_relative_to(extract_root):
                    raise UpstreamError(f"MinerU zip 包含不安全路径：{info.filename}")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
    except zipfile.BadZipFile as exc:
        raise UpstreamError("MinerU 返回内容不是有效 zip。") from exc


def find_markdown(root: Path) -> Path | None:
    paths = sorted(root.rglob("*.md"))
    return paths[0] if paths else None


def find_content_list(root: Path) -> Path | None:
    paths = sorted(root.rglob("*content_list*.json"))
    return paths[0] if paths else None


def find_images_dir(root: Path) -> Path | None:
    named_dirs = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_dir() and path.name.lower() in {"image", "images", "imgs"}
    ]
    if named_dirs:
        return named_dirs[0]
    image_files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return image_files[0].parent if image_files else None


def find_named_json(root: Path, stem: str) -> Path | None:
    names = {f"{stem}.json", f"{stem}_json.json", f"{stem}.middle.json"}
    matches = [
        path
        for path in sorted(root.rglob("*.json"))
        if path.name.lower() in names or path.stem.lower() == stem
    ]
    return matches[0] if matches else None


def find_named_dir(root: Path, names: set[str]) -> Path | None:
    matches = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_dir() and path.name.lower() in names
    ]
    return matches[0] if matches else None


def aggregate_readiness_stage(readiness: UpstreamReadiness) -> StageStatus:
    blocked = [
        status
        for status in (readiness.postgres, readiness.minio, readiness.mineru)
        if status.state == StageState.blocked
    ]
    if blocked:
        details = [status.detail for status in blocked if status.detail]
        return StageStatus(
            state=StageState.blocked,
            reason="missing_config",
            detail="；".join(details) if details else "远端上游配置不完整",
        )
    return StageStatus(state=StageState.ready)


def write_run_manifest(manifest: UpstreamRunManifest, path: Path) -> None:
    manifest.updated_at = _utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def write_scimkg_materialize_manifest(
    manifest: ScimkgMaterializeManifest,
    path: Path,
) -> None:
    manifest.updated_at = _utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def request_json(url: str, timeout_seconds: int) -> Any:
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise UpstreamError(f"Sci-MKG HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise UpstreamError(f"Sci-MKG 请求失败：{exc.reason}") from exc
    except TimeoutError as exc:
        raise UpstreamError(f"Sci-MKG 请求超时：{timeout_seconds}s") from exc
    except OSError as exc:
        raise UpstreamError(f"Sci-MKG 请求失败：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise UpstreamError(f"Sci-MKG 返回不是有效 JSON：{url}") from exc


def download_bytes(url: str, timeout_seconds: int) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.read()
    except HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise UpstreamError(f"下载 HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise UpstreamError(f"下载失败：{exc.reason}") from exc
    except TimeoutError as exc:
        raise UpstreamError(f"下载超时：{timeout_seconds}s") from exc
    except OSError as exc:
        raise UpstreamError(f"下载失败：{exc}") from exc


def try_request_json(
    request_json_fn: Callable[[str, int], Any],
    url: str,
    timeout_seconds: int,
) -> Any | None:
    try:
        return request_json_fn(url, timeout_seconds)
    except UpstreamError:
        return None


def normalize_scimkg_graph(graph: Any | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    if graph is None:
        return [], [], "Sci-MKG graph/full endpoint was not available."
    if not isinstance(graph, dict):
        return [], [], "Sci-MKG graph/full response was not a JSON object."

    raw_nodes = graph.get("nodes") or graph.get("vertices") or []
    raw_edges = graph.get("edges") or graph.get("links") or []
    nodes = [normalize_scimkg_node(node) for node in raw_nodes if isinstance(node, dict)]
    edges = [normalize_scimkg_edge(edge) for edge in raw_edges if isinstance(edge, dict)]
    warning = None
    if not nodes and not edges:
        warning = "Sci-MKG graph/full response contained no nodes or edges."
    return nodes, edges, warning


def normalize_scimkg_node(node: dict[str, Any]) -> dict[str, Any]:
    raw_id = node.get("id") or node.get("node_id") or node.get("name") or node.get("label")
    node_id = stable_id("KGN_", "Sci-MKG", str(raw_id), json.dumps(node, sort_keys=True, default=str))
    return {
        "node_id": str(raw_id) if raw_id else node_id,
        "node_kind": node.get("type") or node.get("entity_type") or node.get("label"),
        "canonical_name": node.get("name") or node.get("label") or str(raw_id or ""),
        "description": node.get("description") or node.get("content"),
        "source_refs": [],
        "raw": node,
    }


def normalize_scimkg_edge(edge: dict[str, Any]) -> dict[str, Any]:
    source = edge.get("source") or edge.get("src") or edge.get("from") or edge.get("source_id")
    target = edge.get("target") or edge.get("dst") or edge.get("to") or edge.get("target_id")
    edge_id = edge.get("id") or edge.get("edge_id") or stable_id(
        "KGE_",
        "Sci-MKG",
        str(source),
        str(target),
        json.dumps(edge, sort_keys=True, default=str),
    )
    return {
        "edge_id": str(edge_id),
        "source_node_id": str(source or ""),
        "target_node_id": str(target or ""),
        "edge_kind": edge.get("type") or edge.get("relation") or edge.get("label"),
        "description": edge.get("description") or edge.get("content"),
        "status": edge.get("status"),
        "raw": edge,
    }


def signed_url(payload: Any | None) -> str | None:
    if isinstance(payload, dict) and isinstance(payload.get("url"), str):
        return payload["url"]
    return None


def markdown_to_content_list(markdown_text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    paragraphs = re.split(r"\n\s*\n", markdown_text)
    for index, paragraph in enumerate(paragraphs, start=1):
        text = clean_text(paragraph)
        if not text:
            continue
        blocks.append(
            {
                "type": "text",
                "text": text,
                "page_idx": None,
                "block_id": f"md_block_{index:04d}",
                "source": "scimkg_preview_markdown",
            }
        )
    return blocks


def title_from_markdown(markdown_text: str, default: str) -> str:
    for line in markdown_text.splitlines():
        text = clean_text(line.lstrip("#").strip())
        if 8 <= len(text) <= 300:
            return text
    return default


def title_from_filename(file_name: str) -> str:
    stem = Path(file_name).stem
    stem = re.sub(r"[_-]+", " ", stem)
    return clean_text(stem) or file_name


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl_dicts(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def issue_row(
    *,
    component: str,
    file_id: int | None,
    paper_id: str | None,
    summary: str,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "severity": "warning",
        "component": component,
        "file_id": file_id,
        "paper_id": paper_id,
        "summary": summary,
        "detail": detail,
        "created_at": _utc_now(),
    }


def safe_filename(name: str, fallback: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return cleaned or fallback


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return prefix + digest[:12].upper()


def _validated_base_url(value: str, label: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise UpstreamError(f"{label} 必须是 http(s) URL，例如 http://host:port")
    return value.rstrip("/")


def _expect_object(payload: Any, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise UpstreamError(f"{label} response must be a JSON object.")
    return payload


def _int_value(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise UpstreamError(f"Sci-MKG source missing integer {label}: {value}") from exc


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def stable_run_id(path: Path, checksum: str) -> str:
    digest = hashlib.sha256(f"{path.name}:{checksum}".encode("utf-8")).hexdigest()
    return "upstream_" + digest[:12]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _redact(key: str, value: Any) -> Any:
    if value is None:
        return None
    upper = key.upper()
    if any(token in upper for token in SECRET_TOKENS):
        return "<redacted>"
    if isinstance(value, str) and ("URI" in upper or "URL" in upper):
        return _redact_url_userinfo(value)
    return value


def _redact_url_userinfo(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if not parsed.netloc or "@" not in parsed.netloc:
        return value
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    return urlunsplit(
        (parsed.scheme, f"<redacted>@{host}{port}", parsed.path, parsed.query, parsed.fragment)
    )
