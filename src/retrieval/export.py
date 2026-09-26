from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from retrieval.registry import LibraryRegistry, utc_now


DEFAULT_STREAMS = [
    "documents",
    "artifacts",
    "evidence_units",
    "readiness",
    "source_candidates",
    "provenance_events",
    "manifest",
]

DEFAULT_EXCLUDE_TAGS = ["exclude_smoke", "exclude_from_research_corpus"]


def build_snapshot_export(
    registry: LibraryRegistry,
    *,
    collection_id: str | None = None,
    tag: str | None = None,
    readiness: dict[str, Any] | None = None,
    exclude_tags: list[str] | None = None,
    streams: list[str] | None = None,
    destination: str = "inline",
    output_dir: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    selected_streams = streams or DEFAULT_STREAMS
    if destination not in {"inline", "file"}:
        raise ValueError("destination must be 'inline' or 'file'")
    readiness_filters = readiness or {}
    evidence_min_count = readiness_filters.get("evidence_min_count")
    if evidence_min_count is not None:
        try:
            evidence_min_count = int(evidence_min_count)
        except (TypeError, ValueError) as exc:
            raise ValueError("readiness.evidence_min_count must be an integer") from exc
        if evidence_min_count < 0:
            raise ValueError("readiness.evidence_min_count must be >= 0")
    papers = registry.list_papers(
        tag=tag,
        collection_id=collection_id,
        include_archived=False,
        metadata_ready=readiness_filters.get("metadata_ready"),
        pdf_ready=readiness_filters.get("pdf_ready"),
        local_mineru_ready=readiness_filters.get("local_mineru_ready"),
        evidence_ready=readiness_filters.get("evidence_ready"),
        sciverse_ai_ready=readiness_filters.get("sciverse_ai_ready"),
        extraction_ready=readiness_filters.get("extraction_ready"),
        year_ready=readiness_filters.get("year_ready"),
    )
    effective_exclude_tags = DEFAULT_EXCLUDE_TAGS if exclude_tags is None else exclude_tags
    excluded = set(effective_exclude_tags)
    if excluded:
        papers = [
            paper
            for paper in papers
            if not excluded.intersection(registry.list_paper_tags(str(paper["paper_id"])))
        ]

    evidence_by_paper_id = {
        str(paper["paper_id"]): registry.list_evidence_units(str(paper["paper_id"]))
        for paper in papers
    }
    if evidence_min_count is not None:
        papers = [
            paper
            for paper in papers
            if len(evidence_by_paper_id[str(paper["paper_id"])]) >= evidence_min_count
        ]

    paper_ids = [str(paper["paper_id"]) for paper in papers]
    readiness_rows = [registry.paper_readiness(paper_id) for paper_id in paper_ids]
    evidence_rows = [
        enrich_evidence_unit(row, registry.get_paper(paper_id) or {})
        for paper_id in paper_ids
        for row in evidence_by_paper_id.get(paper_id, [])
    ]
    source_candidate_rows = [
        row for paper_id in paper_ids for row in registry.list_source_candidates(paper_id)
    ]
    artifact_rows = [row for paper_id in paper_ids for row in registry.list_paper_artifacts(paper_id)]
    provenance_rows = registry.list_provenance_events(paper_ids=paper_ids)

    manifest = {
        "exported_at": utc_now(),
        "service_version": "0.1.0",
        "destination": destination,
        "filters": {
            "collection_id": collection_id,
            "tag": tag,
            "readiness": readiness_filters,
            "exclude_tags": effective_exclude_tags,
            "exclude_tags_default_applied": exclude_tags is None,
        },
        "streams": selected_streams,
        "paper_count": len(papers),
        "evidence_count": len(evidence_rows),
        "artifact_count": len(artifact_rows),
        "source_candidate_count": len(source_candidate_rows),
        "provenance_event_count": len(provenance_rows),
        "failures": [],
    }

    output: dict[str, Any] = {"manifest": manifest}
    stream_rows = {
        "documents": papers,
        "artifacts": artifact_rows,
        "evidence_units": evidence_rows,
        "readiness": readiness_rows,
        "source_candidates": source_candidate_rows,
        "provenance_events": provenance_rows,
    }
    payloads: dict[str, str] = {}
    for stream in selected_streams:
        if stream == "manifest":
            continue
        rows = stream_rows.get(stream)
        if rows is not None:
            payloads[stream] = to_jsonl(rows)
    if destination == "file":
        files = write_snapshot_files(
            registry=registry,
            name=name,
            output_dir=output_dir,
            payloads=payloads,
            manifest=manifest,
            include_manifest="manifest" in selected_streams,
        )
        output["files"] = files
    else:
        for stream, payload in payloads.items():
            output[f"{stream}_jsonl"] = payload
    return output


def enrich_evidence_unit(row: dict[str, Any], paper: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["paper_title"] = paper.get("title")
    item["published_year"] = paper.get("published_year")
    item["published_date"] = paper.get("published_date")
    item["section_title"] = first_present(
        item.get("locator"),
        item.get("metadata"),
        keys=("section_title", "section", "heading"),
    )
    item["block_type"] = first_present(
        item.get("locator"),
        item.get("metadata"),
        keys=("block_type", "type", "kind"),
    ) or item.get("source_type") or (item.get("source") or {}).get("source_type")
    item["source_artifact_kind"] = first_present(
        item.get("provenance"),
        item.get("metadata"),
        keys=("source_artifact_kind", "artifact_kind", "asset_kind"),
    )
    item["stable_sort_key"] = stable_sort_key(item)
    return item


def to_jsonl(rows: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)


def write_snapshot_files(
    *,
    registry: LibraryRegistry,
    name: str | None,
    output_dir: str | None,
    payloads: dict[str, str],
    manifest: dict[str, Any],
    include_manifest: bool,
) -> dict[str, dict[str, Any]]:
    root = resolve_snapshot_dir(registry=registry, output_dir=output_dir, name=name)
    root.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for stream, payload in payloads.items():
        files[stream] = write_text_file(root / f"{stream}.jsonl", payload)
    manifest["snapshot_dir"] = str(root)
    manifest["files"] = files
    if include_manifest:
        files["manifest"] = write_text_file(
            root / "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    manifest["files"] = files
    return files


def resolve_snapshot_dir(
    *,
    registry: LibraryRegistry,
    output_dir: str | None,
    name: str | None,
) -> Path:
    base = registry.library_root / "exports"
    if output_dir:
        requested = Path(output_dir)
        if requested.is_absolute():
            base = requested
        else:
            base = registry.library_root / requested
    snapshot_name = safe_snapshot_name(name or f"snapshot-{utc_now()}")
    root = (base / snapshot_name).resolve()
    library_root = registry.library_root.resolve()
    if not root.is_relative_to(library_root):
        raise ValueError(
            "snapshot output_dir must stay inside the library root; pass a relative output_dir "
            "such as 'exports/my-run' or omit output_dir and use manifest.snapshot_dir from the response."
        )
    return root


def safe_snapshot_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip(".-")
    return cleaned or "snapshot"


def write_text_file(path: Path, payload: str) -> dict[str, Any]:
    encoded = payload.encode("utf-8")
    path.write_bytes(encoded)
    record_count = 0 if not payload else len(payload.splitlines())
    return {
        "path": str(path),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "byte_size": len(encoded),
        "record_count": record_count,
    }


def first_present(*containers: dict[str, Any] | None, keys: tuple[str, ...]) -> Any:
    for container in containers:
        if not container:
            continue
        for key in keys:
            value = container.get(key)
            if value not in (None, ""):
                return value
    return None


def stable_sort_key(item: dict[str, Any]) -> str:
    source = item.get("source") or {}
    page = item.get("page_idx") if item.get("page_idx") is not None else source.get("page_idx")
    locator = item.get("locator") or {}
    offset = first_present(locator, item.get("metadata"), keys=("char_start", "offset", "block_index"))
    parts = [
        str(item.get("paper_id") or ""),
        str(page if page is not None else ""),
        str(offset if offset is not None else ""),
        str(source.get("source_type") or item.get("source_type") or ""),
        str(source.get("source_id") or item.get("source_id") or ""),
        str(item.get("evidence_id") or ""),
    ]
    return "|".join(parts)
