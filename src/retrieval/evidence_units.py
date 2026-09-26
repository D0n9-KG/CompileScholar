from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retrieval.registry import LibraryRegistry, LibraryRegistryError, stable_id
from retrieval._see_models import EvidenceUnit, SourceRef


@dataclass(frozen=True)
class EvidenceRegistrationResult:
    paper_id: str
    artifact_id: str
    evidence_units: list[EvidenceUnit]
    coverage: dict[str, Any]


def ensure_paper_evidence(
    *,
    registry: LibraryRegistry,
    paper_id: str,
    force: bool = False,
) -> EvidenceRegistrationResult:
    existing = registry.list_evidence_units(paper_id)
    existing_keys = set() if force else evidence_source_keys(existing)

    registered: list[EvidenceUnit] = []
    registered.extend(
        _register_mineru_content_list_units(
            registry=registry,
            paper_id=paper_id,
            existing_keys=existing_keys,
        )
    )
    registered.extend(_register_kg_units(registry=registry, paper_id=paper_id, existing_keys=existing_keys))
    registered.extend(
        _register_sciverse_units(registry=registry, paper_id=paper_id, existing_keys=existing_keys)
    )

    if registered:
        registry.record_provenance_event(
            action="register_evidence_units",
            source="multisource_evidence",
            inputs={"paper_id": paper_id, "force": force},
            outputs={"evidence_units": len(registered)},
        )
    return EvidenceRegistrationResult(
        paper_id=paper_id,
        artifact_id="",
        evidence_units=[
            EvidenceUnit.model_validate(item) for item in registry.list_evidence_units(paper_id)
        ],
        coverage=registry.evidence_coverage(paper_id),
    )


def ensure_mineru_content_evidence(
    *,
    registry: LibraryRegistry,
    paper_id: str,
    force: bool = False,
) -> EvidenceRegistrationResult:
    artifact = latest_ready_content_list(registry, paper_id)
    if artifact is None:
        coverage = registry.evidence_coverage(paper_id)
        return EvidenceRegistrationResult(
            paper_id=paper_id,
            artifact_id="",
            evidence_units=[
                EvidenceUnit.model_validate(item) for item in registry.list_evidence_units(paper_id)
            ],
            coverage=coverage,
        )

    existing = registry.list_evidence_units(paper_id)
    if existing and not force:
        return EvidenceRegistrationResult(
            paper_id=paper_id,
            artifact_id=artifact["artifact_id"],
            evidence_units=[EvidenceUnit.model_validate(item) for item in existing],
            coverage=registry.evidence_coverage(paper_id),
        )

    registered = _register_mineru_content_list_units(
        registry=registry,
        paper_id=paper_id,
        artifact=artifact,
        include_visual_units=False,
    )

    registry.record_provenance_event(
        action="register_evidence_units",
        source="mineru_content_list",
        inputs={"paper_id": paper_id, "artifact_id": artifact["artifact_id"]},
        outputs={"evidence_units": len(registered)},
    )
    return EvidenceRegistrationResult(
        paper_id=paper_id,
        artifact_id=artifact["artifact_id"],
        evidence_units=registered,
        coverage=registry.evidence_coverage(paper_id),
    )


def latest_ready_content_list(
    registry: LibraryRegistry,
    paper_id: str,
) -> dict[str, Any] | None:
    artifacts = [
        item
        for item in registry.list_paper_artifacts(paper_id)
        if item.get("kind") == "mineru_content_list" and item.get("status") == "ready"
    ]
    return artifacts[-1] if artifacts else None


def _register_mineru_content_list_units(
    *,
    registry: LibraryRegistry,
    paper_id: str,
    artifact: dict[str, Any] | None = None,
    include_visual_units: bool = True,
    existing_keys: set[tuple[str, str]] | None = None,
) -> list[EvidenceUnit]:
    artifact = artifact or latest_ready_content_list(registry, paper_id)
    if artifact is None:
        return []

    content_path = _artifact_path_inside_library(registry, artifact, "MinerU content_list")
    rows = load_content_list(content_path)
    registered: list[EvidenceUnit] = []
    for index, block in enumerate(normalize_content_blocks(rows), start=1):
        if ("mineru_content_block", block["block_id"]) in (existing_keys or set()):
            continue
        registered.append(
            _register_unit(
                registry=registry,
                paper_id=paper_id,
                artifact=artifact,
                source_type="mineru_content_block",
                source_id=block["block_id"],
                text=block["text"],
                modality="text",
                page_idx=_int_or_none(block.get("page_idx")),
                locator={
                    "page_idx": block.get("page_idx"),
                    "block_index": index,
                    "block_id": block["block_id"],
                },
                provenance={"source_system": "mineru"},
                metadata={"block_type": block.get("type") or block.get("block_type") or "text"},
            )
        )

    if include_visual_units:
        for index, block in enumerate(normalize_visual_blocks(rows), start=1):
            source_type = "table" if block["kind"] == "table" else "figure"
            if (source_type, block["block_id"]) in (existing_keys or set()):
                continue
            registered.append(
                _register_unit(
                    registry=registry,
                    paper_id=paper_id,
                    artifact=artifact,
                    source_type=source_type,
                    source_id=block["block_id"],
                    text=block["text"],
                    modality=source_type,
                    page_idx=_int_or_none(block.get("page_idx")),
                    locator={
                        "page_idx": block.get("page_idx"),
                        "block_index": index,
                        "block_id": block["block_id"],
                        "path": block.get("path"),
                    },
                    provenance={"source_system": "mineru"},
                    metadata={"block_type": block.get("type"), "kind": block["kind"]},
                )
            )
    return registered


def _register_kg_units(
    *,
    registry: LibraryRegistry,
    paper_id: str,
    existing_keys: set[tuple[str, str]] | None = None,
) -> list[EvidenceUnit]:
    registered: list[EvidenceUnit] = []
    for artifact in latest_ready_artifacts(registry, paper_id, {"kg_nodes", "kg_node", "kg_nodes_jsonl"}):
        for index, row in enumerate(load_json_rows(_artifact_path_inside_library(registry, artifact, "KG nodes")), start=1):
            text = kg_node_text(row)
            if not text:
                continue
            source_id = str(row.get("node_id") or row.get("id") or f"kg_node_{index:04d}")
            if ("kg_node", source_id) in (existing_keys or set()):
                continue
            registered.append(
                _register_unit(
                    registry=registry,
                    paper_id=paper_id,
                    artifact=artifact,
                    source_type="kg_node",
                    source_id=source_id,
                    text=text,
                    modality="kg",
                    locator={"row_index": index, "node_id": source_id},
                    provenance={"source_system": "kg"},
                    metadata={
                        "node_kind": row.get("node_kind") or row.get("type"),
                        "canonical_name": row.get("canonical_name") or row.get("name"),
                        "source_refs": row.get("source_refs") if isinstance(row.get("source_refs"), list) else [],
                    },
                )
            )
    for artifact in latest_ready_artifacts(registry, paper_id, {"kg_edges", "kg_edge", "kg_edges_jsonl"}):
        for index, row in enumerate(load_json_rows(_artifact_path_inside_library(registry, artifact, "KG edges")), start=1):
            text = kg_edge_text(row)
            if not text:
                continue
            source_id = str(row.get("edge_id") or row.get("id") or f"kg_edge_{index:04d}")
            if ("kg_edge", source_id) in (existing_keys or set()):
                continue
            registered.append(
                _register_unit(
                    registry=registry,
                    paper_id=paper_id,
                    artifact=artifact,
                    source_type="kg_edge",
                    source_id=source_id,
                    text=text,
                    modality="kg",
                    locator={
                        "row_index": index,
                        "edge_id": source_id,
                        "source_node_id": row.get("source_node_id"),
                        "target_node_id": row.get("target_node_id"),
                    },
                    provenance={"source_system": "kg"},
                    metadata={
                        "edge_kind": row.get("edge_kind") or row.get("type"),
                        "status": row.get("status"),
                    },
                )
            )
    return registered


def _register_sciverse_units(
    *,
    registry: LibraryRegistry,
    paper_id: str,
    existing_keys: set[tuple[str, str]] | None = None,
) -> list[EvidenceUnit]:
    registered: list[EvidenceUnit] = []
    artifact_kinds = {"sciverse_chunks", "sciverse_chunk", "sciverse_jsonl", "sciverse_resource"}
    for artifact in latest_ready_artifacts(registry, paper_id, artifact_kinds):
        for index, row in enumerate(load_json_rows(_artifact_path_inside_library(registry, artifact, "Sciverse chunks")), start=1):
            text = first_text(row, "text", "content", "markdown", "abstract", "description")
            if not text:
                continue
            source_id = str(row.get("chunk_id") or row.get("id") or row.get("resource_id") or f"sciverse_{index:04d}")
            if ("sciverse_chunk", source_id) in (existing_keys or set()):
                continue
            page_idx = _int_or_none(row.get("page_idx") if "page_idx" in row else row.get("page"))
            registered.append(
                _register_unit(
                    registry=registry,
                    paper_id=paper_id,
                    artifact=artifact,
                    source_type="sciverse_chunk",
                    source_id=source_id,
                    text=text,
                    modality="text",
                    page_idx=page_idx,
                    locator={
                        "row_index": index,
                        "chunk_id": source_id,
                        "section": row.get("section"),
                        "page_idx": page_idx,
                    },
                    provenance={"source_system": str(row.get("source") or "sciverse")},
                    metadata={"section": row.get("section"), "resource_type": row.get("resource_type")},
                )
            )
    return registered


def latest_ready_artifacts(
    registry: LibraryRegistry,
    paper_id: str,
    kinds: set[str],
) -> list[dict[str, Any]]:
    return [
        item
        for item in registry.list_paper_artifacts(paper_id)
        if item.get("kind") in kinds and item.get("status") == "ready"
    ]


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


def load_content_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceUnitError(f"MinerU content_list file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvidenceUnitError(f"MinerU content_list is not valid JSON: {path}") from exc
    if not isinstance(payload, list) or not payload:
        raise EvidenceUnitError(f"MinerU content_list must be a non-empty JSON array: {path}")
    rows = [item for item in payload if isinstance(item, dict)]
    if not rows:
        raise EvidenceUnitError(f"MinerU content_list has no JSON object rows: {path}")
    return rows


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise EvidenceUnitError(f"Evidence source file not found: {path}") from exc
    if not text.strip():
        return []
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise EvidenceUnitError(f"Evidence JSONL row {line_number} is invalid: {path}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
        return rows
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvidenceUnitError(f"Evidence JSON file is invalid: {path}") from exc
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("chunks", "items", "records", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def normalize_content_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, block in enumerate(blocks, start=1):
        text = str(block.get("text") or block.get("content") or "").strip()
        if not text:
            continue
        normalized.append(
            {
                **block,
                "block_id": str(block.get("block_id") or block.get("id") or f"content_{index:04d}"),
                "text": text,
                "page_idx": block.get("page_idx") if isinstance(block.get("page_idx"), int) else block.get("page"),
            }
        )
    return normalized


def normalize_visual_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, block in enumerate(blocks, start=1):
        block_type = str(block.get("type") or block.get("block_type") or "").lower()
        kind = "table" if "table" in block_type else "figure" if block_type in {"image", "figure", "fig"} else ""
        if not kind:
            continue
        caption = first_text(block, "caption", "title", "text", "html", "table_body")
        if not caption:
            continue
        path = block.get("img_path") or block.get("image_path") or block.get("path")
        normalized.append(
            {
                **block,
                "block_id": str(block.get("block_id") or block.get("id") or f"{kind}_{index:04d}"),
                "kind": kind,
                "path": path,
                "text": caption,
                "page_idx": block.get("page_idx") if isinstance(block.get("page_idx"), int) else block.get("page"),
            }
        )
    return normalized


def kg_node_text(row: dict[str, Any]) -> str:
    name = first_text(row, "canonical_name", "name", "label")
    description = first_text(row, "description", "text", "summary")
    if name and description:
        return f"{name}: {description}"
    return name or description


def kg_edge_text(row: dict[str, Any]) -> str:
    description = first_text(row, "description", "text", "summary")
    source = first_text(row, "source_node_id", "source", "head")
    target = first_text(row, "target_node_id", "target", "tail")
    kind = first_text(row, "edge_kind", "relation", "type")
    if description:
        return description
    if source and target:
        relation = f" --{kind}--> " if kind else " -> "
        return f"{source}{relation}{target}"
    return ""


def first_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return ""


def _register_unit(
    *,
    registry: LibraryRegistry,
    paper_id: str,
    artifact: dict[str, Any],
    source_type: str,
    source_id: str,
    text: str,
    modality: str,
    page_idx: int | None = None,
    locator: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvidenceUnit:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    evidence_id = stable_id(
        "EV_",
        paper_id,
        artifact["artifact_id"],
        source_type,
        source_id,
        content_hash,
    )
    merged_provenance = {
        **(provenance or {}),
        "source_artifact_id": artifact["artifact_id"],
        "source_artifact_kind": artifact.get("kind"),
        "source_checksum": artifact.get("checksum"),
    }
    stored = registry.register_evidence_unit(
        evidence_id=evidence_id,
        paper_id=paper_id,
        asset_id=artifact["artifact_id"],
        source_type=source_type,
        source_id=source_id,
        text=text,
        content_hash=content_hash,
        modality=modality,
        status="candidate",
        page_idx=page_idx,
        locator=locator or {},
        provenance=merged_provenance,
        quality={"status": "usable", "issues": []},
        metadata=metadata or {},
    )
    return EvidenceUnit.model_validate(stored)


def _artifact_path_inside_library(
    registry: LibraryRegistry,
    artifact: dict[str, Any],
    label: str,
) -> Path:
    path = registry.resolve_local_path(artifact["local_path"])
    try:
        registry.relative_local_path(path)
    except LibraryRegistryError as exc:
        raise EvidenceUnitError(f"{label} artifact path is outside library root.") from exc
    return path


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


class EvidenceUnitError(RuntimeError):
    pass
