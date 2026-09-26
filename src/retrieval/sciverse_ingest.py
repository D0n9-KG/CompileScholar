from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from retrieval.registry import LibraryRegistry, stable_id, utc_now
from retrieval.sources import (
    SciverseClient,
    SourceAdapterError,
    assess_sciverse_ai_ready,
)
from retrieval._see_models import EvidenceUnit


SCIVERSE_ASSET_KIND = "sciverse_live_ingest"
SCIVERSE_AI_READY_ASSET_KIND = "sciverse_ai_ready"
SCIVERSE_SOURCE_TYPE = "sciverse_chunk"


@dataclass(frozen=True)
class SciverseContentFailure:
    source_id: str
    doc_id: str | None
    chunk_id: str | None
    error: str


@dataclass(frozen=True)
class SciverseIngestResult:
    paper_id: str
    status: str
    query: str
    evidence_units: list[EvidenceUnit] = field(default_factory=list)
    ingested_count: int = 0
    skipped_count: int = 0
    content_successes: int = 0
    content_failures: list[SciverseContentFailure] = field(default_factory=list)
    raw_response_path: str | None = None
    resource_status: dict[str, str] = field(default_factory=dict)
    remote_asset: dict[str, Any] | None = None
    message: str | None = None


def ingest_sciverse_evidence_for_paper(
    *,
    registry: LibraryRegistry,
    paper_id: str,
    sciverse_client: SciverseClient,
    query: str | None = None,
    limit: int = 5,
    force: bool = False,
) -> SciverseIngestResult:
    paper = registry.get_paper(paper_id)
    if not paper:
        raise SciverseIngestError(f"paper not found: {paper_id}")
    resolved_query = (query or build_sciverse_query(paper)).strip()
    if not resolved_query:
        raise SciverseIngestError("Sciverse ingest 需要 paper title、DOI 或显式 query。")

    resource_status = resource_status_for(sciverse_client)
    raw_path = sciverse_raw_path(registry, paper_id, resolved_query)
    raw_relative_path = str(registry.relative_local_path(raw_path))
    asset_id = sciverse_asset_id(paper_id)

    try:
        hits = sciverse_client.agentic_search(resolved_query, limit=limit)
    except SourceAdapterError as exc:
        status = "blocked" if "SCIVERSE_API_TOKEN" in str(exc) else "failed"
        registry.record_provenance_event(
            action="ingest_sciverse_evidence",
            source="sciverse",
            inputs={"paper_id": paper_id, "query": resolved_query, "limit": limit},
            outputs={"status": status, "evidence_units": 0},
            error_summary=str(exc),
        )
        return SciverseIngestResult(
            paper_id=paper_id,
            status=status,
            query=resolved_query,
            resource_status=resource_status,
            message=str(exc),
        )

    existing_keys = set() if force else evidence_source_keys(registry.list_evidence_units(paper_id))
    raw_rows: list[dict[str, Any]] = []
    registered: list[EvidenceUnit] = []
    skipped_count = 0
    content_successes = 0
    content_failures: list[SciverseContentFailure] = []
    best_ai_ready: dict[str, Any] | None = None
    best_locator: dict[str, Any] = {}
    best_provenance: dict[str, Any] = {}
    best_source_record_id: str | None = None

    for index, hit in enumerate(hits, start=1):
        normalized = normalize_sciverse_hit(hit, index=index)
        if not normalized["text"]:
            continue
        source_key = (SCIVERSE_SOURCE_TYPE, normalized["source_id"])
        if source_key in existing_keys:
            skipped_count += 1
            continue

        content_status = "not_requested"
        content_text = None
        content_response: dict[str, Any] | None = None
        content_error = None
        if normalized["doc_id"]:
            try:
                content_response = sciverse_client.read_content(
                    doc_id=normalized["doc_id"],
                    chunk_id=normalized["chunk_id"] or None,
                    offset=normalized["offset"],
                    limit=None,
                )
                content_text = first_text_value(content_response)
                content_status = "ready" if content_text else "empty"
                if content_status == "ready":
                    content_successes += 1
            except SourceAdapterError as exc:
                content_status = "failed"
                content_error = str(exc)
                content_failures.append(
                    SciverseContentFailure(
                        source_id=normalized["source_id"],
                        doc_id=normalized["doc_id"],
                        chunk_id=normalized["chunk_id"],
                        error=str(exc),
                    )
                )

        ai_ready = assess_sciverse_ai_ready(
            doc_id=normalized["doc_id"] or None,
            chunk_id=normalized["chunk_id"] or None,
            content_accessible=True,
            content_status=content_status,
            content_text=content_text or normalized["text"],
            content_error=content_error,
            locator=normalized["locator"],
            provenance={
                **normalized["provenance"],
                "raw_response_path": raw_relative_path,
                "content_endpoint": "/content" if normalized["doc_id"] else None,
            },
        )
        if is_better_sciverse_ai_ready(ai_ready, best_ai_ready):
            best_ai_ready = ai_ready
            best_locator = normalized["locator"]
            best_provenance = {
                **normalized["provenance"],
                "source_system": "sciverse",
                "source_artifact_kind": SCIVERSE_AI_READY_ASSET_KIND,
                "raw_response_path": raw_relative_path,
                "content_endpoint": "/content" if normalized["doc_id"] else None,
            }
            best_source_record_id = normalized["doc_id"] or normalized["source_id"]
        raw_rows.append(
            {
                "source_id": normalized["source_id"],
                "hit": normalized["raw"],
                "content_status": content_status,
                "content": content_response,
                "content_error": content_error,
                "retrieved_at": utc_now(),
            }
        )
        stored = register_sciverse_evidence_unit(
            registry=registry,
            paper_id=paper_id,
            asset_id=asset_id,
            source_id=normalized["source_id"],
            text=normalized["text"],
            page_idx=normalized["page_idx"],
            locator=normalized["locator"],
            provenance={
                **normalized["provenance"],
                "source_system": "sciverse",
                "source_artifact_kind": SCIVERSE_ASSET_KIND,
                "raw_response_path": raw_relative_path,
                "content_endpoint": "/content" if normalized["doc_id"] else None,
                "content_status": content_status,
                "retrieved_at": utc_now(),
            },
            metadata={
                "content_status": content_status,
                "content_text": content_text,
                "content_error": content_error,
                "sciverse_ai_ready": ai_ready,
                "resource_status": resource_status,
                "raw_response_path": raw_relative_path,
            },
        )
        registered.append(EvidenceUnit.model_validate(stored))
        existing_keys.add(source_key)

    write_jsonl(raw_path, raw_rows)
    status = "ready" if registered else "empty"
    existing_remote_assets = registry.list_remote_parsed_assets(
        paper_id,
        asset_kind=SCIVERSE_AI_READY_ASSET_KIND,
        source_name="sciverse",
    )
    existing_best_asset = best_existing_sciverse_asset(existing_remote_assets)
    if not registered and existing_best_asset:
        remote_asset = existing_best_asset
    else:
        remote_asset = register_sciverse_remote_asset(
            registry=registry,
            paper_id=paper_id,
            asset_id=asset_id,
            status="ready" if registered else "empty",
            source_record_id=best_source_record_id,
            locator=best_locator,
            provenance=best_provenance
            or {
                "source_system": "sciverse",
                "source_artifact_kind": SCIVERSE_AI_READY_ASSET_KIND,
                "raw_response_path": raw_relative_path,
            },
            quality=best_ai_ready
            or {
                "asset_kind": SCIVERSE_AI_READY_ASSET_KIND,
                "quality_tier": "local_mineru",
                "readiness_status": "missing",
                "label": "建议本地 MinerU",
                "next_action": "run_local_mineru",
                "quality_issues": ["no_new_sciverse_chunks"],
            },
            metadata={
                "query": resolved_query,
                "raw_response_path": raw_relative_path,
                "evidence_units": len(registered),
                "skipped": skipped_count,
                "content_successes": content_successes,
                "content_failures": len(content_failures),
                "resource_status": resource_status,
                "reused_existing": bool(existing_remote_assets and not registered),
            },
        )
    registry.record_provenance_event(
        action="ingest_sciverse_evidence",
        source="sciverse",
        inputs={"paper_id": paper_id, "query": resolved_query, "limit": limit, "force": force},
        outputs={
            "status": status,
            "evidence_units": len(registered),
            "skipped": skipped_count,
            "content_successes": content_successes,
            "content_failures": len(content_failures),
            "raw_response_path": raw_relative_path,
            "remote_asset_id": remote_asset.get("asset_id"),
        },
    )
    return SciverseIngestResult(
        paper_id=paper_id,
        status=status,
        query=resolved_query,
        evidence_units=registered,
        ingested_count=len(registered),
        skipped_count=skipped_count,
        content_successes=content_successes,
        content_failures=content_failures,
        raw_response_path=raw_relative_path,
        resource_status=resource_status,
        remote_asset=remote_asset,
    )


def register_sciverse_remote_asset(
    *,
    registry: LibraryRegistry,
    paper_id: str,
    asset_id: str,
    status: str,
    source_record_id: str | None,
    locator: dict[str, Any],
    provenance: dict[str, Any],
    quality: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return registry.register_remote_parsed_asset(
        asset_id=asset_id,
        paper_id=paper_id,
        source_name="sciverse",
        asset_kind=SCIVERSE_AI_READY_ASSET_KIND,
        status=status,
        source_record_id=source_record_id,
        locator=drop_none(locator),
        provenance=drop_none(provenance),
        quality=drop_none(quality),
        metadata=drop_none(metadata),
    )


def is_better_sciverse_ai_ready(
    candidate: dict[str, Any],
    current: dict[str, Any] | None,
) -> bool:
    if current is None:
        return True
    rank = {"recommended": 3, "try": 2, "local_mineru": 1}
    candidate_rank = rank.get(str(candidate.get("quality_tier")), 0)
    current_rank = rank.get(str(current.get("quality_tier")), 0)
    if candidate_rank != current_rank:
        return candidate_rank > current_rank
    return int(candidate.get("content_length") or 0) > int(current.get("content_length") or 0)


def best_existing_sciverse_asset(assets: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not assets:
        return None
    rank = {"recommended": 3, "try": 2, "local_mineru": 1}
    return max(
        assets,
        key=lambda asset: (
            1 if asset.get("status") == "ready" else 0,
            rank.get(str(asset.get("quality", {}).get("quality_tier")), 0),
            str(asset.get("updated_at") or asset.get("created_at") or ""),
        ),
    )


def register_sciverse_evidence_unit(
    *,
    registry: LibraryRegistry,
    paper_id: str,
    asset_id: str,
    source_id: str,
    text: str,
    page_idx: int | None,
    locator: dict[str, Any],
    provenance: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    evidence_id = stable_id("EV_", paper_id, asset_id, SCIVERSE_SOURCE_TYPE, source_id)
    return registry.register_evidence_unit(
        evidence_id=evidence_id,
        paper_id=paper_id,
        asset_id=asset_id,
        source_type=SCIVERSE_SOURCE_TYPE,
        source_id=source_id,
        text=text,
        content_hash=content_hash,
        modality="text",
        status="candidate",
        page_idx=page_idx,
        locator=locator,
        provenance=drop_none(provenance),
        quality={
            "status": "usable",
            "issues": [],
            "sciverse_ai_ready": metadata.get("sciverse_ai_ready", {}),
        },
        metadata=drop_none(metadata),
    )


def build_sciverse_query(paper: dict[str, Any]) -> str:
    title = str(paper.get("title") or "").strip()
    doi = str(paper.get("normalized_doi") or "").strip()
    if title and doi:
        return f"{title} DOI {doi}"
    return title or doi


def sciverse_asset_id(paper_id: str) -> str:
    return stable_id("SCIV_", paper_id, SCIVERSE_ASSET_KIND)


def sciverse_raw_path(registry: LibraryRegistry, paper_id: str, query: str) -> Path:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
    return registry.library_root / "papers" / paper_id / "sciverse" / f"agentic_search_{digest}.jsonl"


def normalize_sciverse_hit(hit: dict[str, Any], *, index: int) -> dict[str, Any]:
    locator = dict(hit.get("locator") if isinstance(hit.get("locator"), dict) else {})
    provenance = dict(hit.get("provenance") if isinstance(hit.get("provenance"), dict) else {})
    raw = hit.get("raw") if isinstance(hit.get("raw"), dict) else hit
    doc_id = str(hit.get("doc_id") or locator.get("doc_id") or raw.get("doc_id") or "")
    chunk_id = str(hit.get("chunk_id") or locator.get("chunk_id") or raw.get("chunk_id") or "")
    source_id = str(hit.get("source_id") or chunk_id or raw.get("id") or f"sciverse_{index:04d}")
    text = str(hit.get("text") or hit.get("chunk") or raw.get("chunk") or raw.get("text") or "").strip()
    page_idx = int_or_none(locator.get("page_idx") if "page_idx" in locator else locator.get("page_no"))
    offset = int_or_none(locator.get("offset") if "offset" in locator else raw.get("offset"))
    page_no = int_or_none(locator.get("page_no") if "page_no" in locator else raw.get("page_no"))
    return {
        "source_id": source_id,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "text": text,
        "offset": offset,
        "page_idx": page_idx or page_no,
        "locator": drop_none(
            {
                **locator,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "offset": offset,
                "page_no": page_no,
                "row_index": index,
            }
        ),
        "provenance": provenance,
        "raw": raw,
    }


def evidence_source_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in rows:
        source = row.get("source")
        if not isinstance(source, dict):
            continue
        source_type = source.get("source_type")
        source_id = source.get("source_id")
        if isinstance(source_type, str) and isinstance(source_id, str):
            keys.add((source_type, source_id))
    return keys


def resource_status_for(sciverse_client: SciverseClient) -> dict[str, str]:
    resource_status = getattr(sciverse_client, "resource_status", None)
    if callable(resource_status):
        return resource_status()
    return {
        "status": "unsupported",
        "reason": "Sciverse /resource 需要真实 file_name + type=image/file，当前未确认文件名来源，暂不默认拉取。",
    }


def first_text_value(payload: object) -> str | None:
    if isinstance(payload, str):
        return payload.strip() or None
    if isinstance(payload, dict):
        for key in ("text", "content", "markdown", "chunk", "abstract"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("data", "chunks", "items", "results"):
            value = first_text_value(payload.get(key))
            if value:
                return value
    if isinstance(payload, list):
        for item in payload:
            value = first_text_value(item)
            if value:
                return value
    return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


class SciverseIngestError(RuntimeError):
    pass
