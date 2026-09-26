from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1"


def normalize_doi(value: str | None) -> str | None:
    if value is None:
        return None
    doi = value.strip()
    if not doi:
        return None
    lower = doi.lower()
    for prefix in (
        "doi:",
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ):
        if lower.startswith(prefix):
            doi = doi[len(prefix) :]
            break
    return doi.strip().lower() or None


def normalize_title(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", normalized) or None


GENERIC_SHORT_TITLES = {
    "ai",
    "llm",
    "ml",
    "nlp",
    "rag",
}


def title_dedupe_match_reason(requested_title: str | None, owner_title: str | None) -> str | None:
    requested_key = normalize_title(requested_title)
    owner_key = normalize_title(owner_title)
    if not requested_key or not owner_key:
        return None
    if requested_key == owner_key:
        return "exact_title"

    short_key, long_key = (
        (owner_key, requested_key)
        if len(owner_key) <= len(requested_key)
        else (requested_key, owner_key)
    )
    short_tokens = short_key.split()
    short_compact = short_key.replace(" ", "")
    if (
        len(short_tokens) <= 2
        and len(short_compact) >= 5
        and short_tokens[0] not in GENERIC_SHORT_TITLES
        and (
            long_key.startswith(f"{short_key} ")
            or f" {short_key} " in f" {long_key} "
        )
    ):
        return "short_title_token"
    return None


def normalize_identifier(identifier: dict[str, str]) -> dict[str, str] | None:
    scheme = str(identifier.get("scheme") or "").strip().lower()
    value = str(identifier.get("value") or "").strip()
    source = str(identifier.get("source") or "source_adapter").strip() or "source_adapter"
    if not scheme or not value:
        return None
    if scheme == "doi":
        value = normalize_doi(value) or ""
    elif scheme == "openalex":
        value = value.rstrip("/").split("/")[-1].upper()
    elif scheme == "arxiv":
        value = normalize_arxiv_id(value) or ""
    else:
        value = value.strip()
    if not value:
        return None
    return {"scheme": scheme, "value": value, "source": source}


def normalize_identifiers(identifiers: Iterable[dict[str, str]] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for identifier in identifiers or []:
        item = normalize_identifier(identifier)
        if not item:
            continue
        key = (item["scheme"], item["value"])
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
    return normalized


def normalize_arxiv_id(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    cleaned = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.removesuffix(".pdf")
    cleaned = re.sub(r"v\d+$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.lower() or None


class LibraryRegistry:
    def __init__(self, db_path: Path, library_root: Path):
        self.db_path = db_path
        self.library_root = library_root

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.library_root.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA_SQL)
            ensure_schema_migrations(connection)
            connection.execute(
                """
                INSERT INTO schema_info(key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (SCHEMA_VERSION,),
            )

    def upsert_paper(
        self,
        *,
        doi: str | None = None,
        title: str | None = None,
        year: int | None = None,
        published_year: int | None = None,
        published_date: str | None = None,
        year_source: str | None = None,
        year_confidence: float | None = None,
        year_needs_review: bool | None = None,
        venue: str | None = None,
        identity_confidence: float | None = None,
        identifiers: Iterable[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        normalized_doi = normalize_doi(doi)
        normalized_identifiers = normalize_identifiers(identifiers)
        now = utc_now()
        paper_id = stable_id("PPR_", normalized_doi or title or uuid.uuid4().hex)
        effective_published_year = published_year if published_year is not None else year
        effective_year_source = year_source
        if effective_published_year is not None and effective_year_source is None:
            effective_year_source = "metadata"
        effective_year_needs_review = year_needs_review
        if effective_year_needs_review is None and effective_published_year is not None:
            effective_year_needs_review = False
        with self._connect() as connection:
            if normalized_doi:
                existing = connection.execute(
                    "SELECT * FROM papers WHERE normalized_doi = ?",
                    (normalized_doi,),
                ).fetchone()
                if existing:
                    connection.execute(
                        """
                        UPDATE papers
                        SET
                            published_year = COALESCE(published_year, ?),
                            published_date = COALESCE(published_date, ?),
                            year_source = COALESCE(year_source, ?),
                            year_confidence = COALESCE(year_confidence, ?),
                            year_needs_review = CASE
                                WHEN published_year IS NULL AND ? IS NOT NULL THEN ?
                                ELSE year_needs_review
                            END,
                            updated_at = ?
                        WHERE paper_id = ?
                        """,
                        (
                            effective_published_year,
                            published_date,
                            effective_year_source,
                            year_confidence,
                            effective_published_year,
                            int(bool(effective_year_needs_review)),
                            now,
                            existing["paper_id"],
                        ),
                    )
                    self._insert_identifier(
                        connection,
                        existing["paper_id"],
                        "doi",
                        normalized_doi,
                        "user_input",
                    )
                    self._insert_identifiers(connection, existing["paper_id"], normalized_identifiers)
                    return dict(
                        connection.execute(
                            "SELECT * FROM papers WHERE paper_id = ?",
                            (existing["paper_id"],),
                        ).fetchone()
                    )

            existing = self._find_paper_by_identifiers(connection, normalized_identifiers)
            if not existing:
                existing = self._find_paper_by_title_year(
                    connection,
                    title=title,
                    year=effective_published_year,
                    identity_confidence=identity_confidence,
                    normalized_doi=normalized_doi,
                )
            if existing:
                connection.execute(
                    """
                    UPDATE papers
                    SET
                        normalized_doi = COALESCE(normalized_doi, ?),
                        published_year = COALESCE(published_year, ?),
                        published_date = COALESCE(published_date, ?),
                        year_source = COALESCE(year_source, ?),
                        year_confidence = COALESCE(year_confidence, ?),
                        year_needs_review = CASE
                            WHEN published_year IS NULL AND ? IS NOT NULL THEN ?
                            ELSE year_needs_review
                        END,
                        updated_at = ?
                    WHERE paper_id = ?
                    """,
                    (
                        normalized_doi,
                        effective_published_year,
                        published_date,
                        effective_year_source,
                        year_confidence,
                        effective_published_year,
                        int(bool(effective_year_needs_review)),
                        now,
                        existing["paper_id"],
                    ),
                )
                if normalized_doi:
                    self._insert_identifier(
                        connection,
                        existing["paper_id"],
                        "doi",
                        normalized_doi,
                        "user_input",
                    )
                self._insert_identifiers(connection, existing["paper_id"], normalized_identifiers)
                return dict(
                    connection.execute(
                        "SELECT * FROM papers WHERE paper_id = ?",
                        (existing["paper_id"],),
                    ).fetchone()
                )

            connection.execute(
                """
                INSERT INTO papers(
                    paper_id, normalized_doi, title, year, venue,
                    identity_confidence, published_year, published_date, year_source,
                    year_confidence, year_needs_review, archived, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    normalized_doi = COALESCE(papers.normalized_doi, excluded.normalized_doi),
                    title = COALESCE(papers.title, excluded.title),
                    year = COALESCE(papers.year, excluded.year),
                    published_year = COALESCE(papers.published_year, excluded.published_year),
                    published_date = COALESCE(papers.published_date, excluded.published_date),
                    year_source = COALESCE(papers.year_source, excluded.year_source),
                    year_confidence = COALESCE(papers.year_confidence, excluded.year_confidence),
                    year_needs_review = CASE
                        WHEN papers.published_year IS NULL AND excluded.published_year IS NOT NULL
                            THEN excluded.year_needs_review
                        ELSE papers.year_needs_review
                    END,
                    venue = COALESCE(papers.venue, excluded.venue),
                    identity_confidence = COALESCE(
                        papers.identity_confidence,
                        excluded.identity_confidence
                    ),
                    updated_at = excluded.updated_at
                """,
                (
                    paper_id,
                    normalized_doi,
                    title,
                    year,
                    venue,
                    identity_confidence,
                    effective_published_year,
                    published_date,
                    effective_year_source,
                    year_confidence,
                    int(bool(effective_year_needs_review)),
                    0,
                    now,
                    now,
                ),
            )
            if normalized_doi:
                self._insert_identifier(
                    connection,
                    paper_id,
                    "doi",
                    normalized_doi,
                    "user_input",
                )
            self._insert_identifiers(connection, paper_id, normalized_identifiers)
            row = connection.execute(
                "SELECT * FROM papers WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
        return dict(row)

    def list_identifiers(self, paper_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT scheme, value, source
                FROM paper_identifiers
                WHERE paper_id = ?
                ORDER BY scheme, value, source
                """,
                (paper_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def find_paper_by_identifier(self, scheme: str, value: str) -> dict[str, Any] | None:
        identifier = normalize_identifier({"scheme": scheme, "value": value, "source": "lookup"})
        if not identifier:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT papers.*
                FROM paper_identifiers
                JOIN papers ON papers.paper_id = paper_identifiers.paper_id
                WHERE paper_identifiers.scheme = ?
                  AND paper_identifiers.value = ?
                """,
                (identifier["scheme"], identifier["value"]),
            ).fetchone()
        return dict(row) if row else None

    def store_pdf_asset(
        self,
        *,
        paper_id: str,
        source_path: Path,
        source_kind: str,
        source_uri: str | None = None,
        license: str | None = None,
        open_access_status: str | None = None,
        status: str = "ready",
    ) -> dict[str, Any]:
        self._ensure_pdf(source_path)
        checksum = sha256_file(source_path)
        size_bytes = source_path.stat().st_size
        now = utc_now()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM pdf_assets WHERE sha256 = ?",
                (checksum,),
            ).fetchone()
            if existing:
                if existing["paper_id"] != paper_id:
                    raise CrossPaperDedupeConflict(
                        requested_paper_id=paper_id,
                        asset_owner_paper_id=str(existing["paper_id"]),
                        pdf_asset_id=str(existing["asset_id"]),
                        sha256=str(existing["sha256"]),
                    )
                return dict(existing)

            self._require_paper(connection, paper_id)
            asset_id = "PDF_" + checksum[:12].upper()
            relative_path = Path("papers") / paper_id / "pdf_assets" / f"{asset_id}.pdf"
            destination = self.library_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source_path.resolve() != destination.resolve():
                shutil.copyfile(source_path, destination)
            connection.execute(
                """
                INSERT INTO pdf_assets(
                    asset_id, paper_id, sha256, size_bytes, source_kind, source_uri,
                    license, open_access_status, local_path, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_id,
                    paper_id,
                    checksum,
                    size_bytes,
                    source_kind,
                    source_uri,
                    license,
                    open_access_status,
                    str(relative_path),
                    status,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM pdf_assets WHERE asset_id = ?",
                (asset_id,),
            ).fetchone()
        return dict(row)

    def find_pdf_asset_by_sha256(self, sha256: str) -> dict[str, Any] | None:
        checksum = sha256.strip().lower()
        if not checksum:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM pdf_assets WHERE sha256 = ?",
                (checksum,),
            ).fetchone()
        return dict(row) if row else None

    def list_pdf_assets(self, paper_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM pdf_assets WHERE paper_id = ? ORDER BY created_at",
                (paper_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def find_duplicate_asset_diagnostics(self, paper_id: str) -> dict[str, Any] | None:
        paper = self.get_paper(paper_id)
        if not paper:
            raise LibraryRegistryError(f"paper not found: {paper_id}")
        if self.list_pdf_assets(paper_id):
            return None

        identifiers = self.list_identifiers(paper_id)
        title_key = normalize_title(paper.get("title"))
        published_year = paper.get("published_year") or paper.get("year")
        candidates: list[dict[str, Any]] = []
        with self._connect() as connection:
            doi = normalize_doi(paper.get("normalized_doi"))
            if doi:
                rows = connection.execute(
                    """
                    SELECT pdf_assets.*, papers.title AS owner_title, papers.normalized_doi AS owner_doi
                    FROM pdf_assets
                    JOIN papers ON papers.paper_id = pdf_assets.paper_id
                    WHERE papers.normalized_doi = ?
                      AND pdf_assets.paper_id != ?
                    ORDER BY pdf_assets.created_at
                    """,
                    (doi, paper_id),
                ).fetchall()
                candidates.extend(dict(row) for row in rows)
            for identifier in identifiers:
                rows = connection.execute(
                    """
                    SELECT pdf_assets.*, papers.title AS owner_title, papers.normalized_doi AS owner_doi
                    FROM paper_identifiers
                    JOIN pdf_assets ON pdf_assets.paper_id = paper_identifiers.paper_id
                    JOIN papers ON papers.paper_id = pdf_assets.paper_id
                    WHERE paper_identifiers.scheme = ?
                      AND paper_identifiers.value = ?
                      AND paper_identifiers.paper_id != ?
                    ORDER BY pdf_assets.created_at
                    """,
                    (identifier["scheme"], identifier["value"], paper_id),
                ).fetchall()
                candidates.extend(dict(row) for row in rows)
            if title_key and published_year:
                rows = connection.execute(
                    """
                    SELECT pdf_assets.*, papers.title AS owner_title, papers.normalized_doi AS owner_doi
                    FROM pdf_assets
                    JOIN papers ON papers.paper_id = pdf_assets.paper_id
                    WHERE COALESCE(papers.published_year, papers.year) = ?
                      AND pdf_assets.paper_id != ?
                    ORDER BY pdf_assets.created_at
                    """,
                    (published_year, paper_id),
                ).fetchall()
                for row in rows:
                    item = dict(row)
                    match_reason = title_dedupe_match_reason(paper.get("title"), item.get("owner_title"))
                    if match_reason == "exact_title":
                        item["title_match_reason"] = match_reason
                        candidates.append(item)
            if title_key:
                rows = connection.execute(
                    """
                    SELECT pdf_assets.*, papers.title AS owner_title, papers.normalized_doi AS owner_doi
                    FROM pdf_assets
                    JOIN papers ON papers.paper_id = pdf_assets.paper_id
                    WHERE pdf_assets.paper_id != ?
                    ORDER BY pdf_assets.created_at
                    """,
                    (paper_id,),
                ).fetchall()
                for row in rows:
                    item = dict(row)
                    match_reason = title_dedupe_match_reason(paper.get("title"), item.get("owner_title"))
                    if match_reason:
                        item["title_match_reason"] = match_reason
                        candidates.append(item)

        deduped: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            deduped[str(candidate["asset_id"])] = candidate
        if not deduped:
            return None
        best_asset = next(iter(deduped.values()))
        owner_id = str(best_asset["paper_id"])
        return {
            "reason": "dedupe_asset_on_other_paper",
            "requested_paper_id": paper_id,
            "asset_owner_paper_id": owner_id,
            "pdf_asset_id": best_asset["asset_id"],
            "sha256": best_asset["sha256"],
            "source_kind": best_asset.get("source_kind"),
            "source_uri": best_asset.get("source_uri"),
            "owner_title": best_asset.get("owner_title"),
            "owner_doi": best_asset.get("owner_doi"),
            "match_reason": best_asset.get("title_match_reason") or "identity_or_exact_title",
            "owner_readiness": {
                "pdf_asset_count": len(self.list_pdf_assets(owner_id)),
                "processing_job_count": len(self.list_processing_jobs(owner_id)),
                "evidence_count": self.evidence_coverage(owner_id)["total"],
            },
            "recommended_next_action": "merge_or_rebind_asset",
            "merge_endpoint": f"/api/library/papers/{paper_id}/merge",
            "merge_dedupe_endpoint": f"/api/library/papers/{paper_id}/merge-dedupe",
            "merge_request_body_dry_run": {
                "source_paper_id": owner_id,
                "dry_run": True,
                "archive_source": True,
            },
            "merge_request_body_commit": {
                "source_paper_id": owner_id,
                "dry_run": False,
                "archive_source": True,
            },
        }

    def merge_paper_assets(
        self,
        *,
        target_paper_id: str,
        source_paper_id: str,
        dry_run: bool = True,
        archive_source: bool = True,
    ) -> dict[str, Any]:
        if target_paper_id == source_paper_id:
            raise LibraryRegistryError("source and target paper must be different")
        if not self.get_paper(target_paper_id):
            raise LibraryRegistryError(f"paper not found: {target_paper_id}")
        if not self.get_paper(source_paper_id):
            raise LibraryRegistryError(f"paper not found: {source_paper_id}")

        now = utc_now()
        with self._connect() as connection:
            counts = {
                "pdf_assets": connection.execute("SELECT COUNT(*) AS count FROM pdf_assets WHERE paper_id = ?", (source_paper_id,)).fetchone()["count"],
                "processing_jobs": connection.execute("SELECT COUNT(*) AS count FROM processing_jobs WHERE paper_id = ?", (source_paper_id,)).fetchone()["count"],
                "evidence_units": connection.execute("SELECT COUNT(*) AS count FROM evidence_units WHERE paper_id = ?", (source_paper_id,)).fetchone()["count"],
                "source_candidates": connection.execute("SELECT COUNT(*) AS count FROM source_candidates WHERE paper_id = ?", (source_paper_id,)).fetchone()["count"],
                "remote_parsed_assets": connection.execute("SELECT COUNT(*) AS count FROM remote_parsed_assets WHERE paper_id = ?", (source_paper_id,)).fetchone()["count"],
                "api_jobs": connection.execute("SELECT COUNT(*) AS count FROM api_jobs WHERE paper_id = ?", (source_paper_id,)).fetchone()["count"],
                "upstream_asset_links": connection.execute("SELECT COUNT(*) AS count FROM upstream_asset_links WHERE paper_id = ?", (source_paper_id,)).fetchone()["count"],
                "identifiers": connection.execute("SELECT COUNT(*) AS count FROM paper_identifiers WHERE paper_id = ?", (source_paper_id,)).fetchone()["count"],
                "tags": connection.execute("SELECT COUNT(*) AS count FROM paper_tags WHERE paper_id = ?", (source_paper_id,)).fetchone()["count"],
                "collections": connection.execute("SELECT COUNT(*) AS count FROM paper_collection_members WHERE paper_id = ?", (source_paper_id,)).fetchone()["count"],
            }
            result = {
                "target_paper_id": target_paper_id,
                "source_paper_id": source_paper_id,
                "dry_run": dry_run,
                "archive_source": archive_source,
                "status": "would_merge" if dry_run else "merged",
                "counts": counts,
                "actions": [
                    "rebind_pdf_assets",
                    "rebind_processing_jobs",
                    "rebind_remote_parsed_assets",
                    "rebind_evidence_units",
                    "merge_source_candidates",
                    "merge_tags",
                    "merge_collections",
                    "merge_identifiers",
                    "rebind_api_jobs",
                    "rebind_upstream_asset_links",
                    "archive_source_paper" if archive_source else "leave_source_visible",
                ],
                "note": "文件路径保留原位置，只迁移 registry 指针并记录 provenance。",
            }
            if dry_run:
                return result

            self._require_paper(connection, target_paper_id)
            self._require_paper(connection, source_paper_id)
            connection.execute("UPDATE pdf_assets SET paper_id = ? WHERE paper_id = ?", (target_paper_id, source_paper_id))
            connection.execute("UPDATE processing_jobs SET paper_id = ? WHERE paper_id = ?", (target_paper_id, source_paper_id))
            connection.execute("UPDATE remote_parsed_assets SET paper_id = ? WHERE paper_id = ?", (target_paper_id, source_paper_id))
            connection.execute("UPDATE evidence_units SET paper_id = ? WHERE paper_id = ?", (target_paper_id, source_paper_id))
            connection.execute("UPDATE source_candidates SET paper_id = ? WHERE paper_id = ?", (target_paper_id, source_paper_id))
            connection.execute("UPDATE api_jobs SET paper_id = ? WHERE paper_id = ?", (target_paper_id, source_paper_id))
            connection.execute("UPDATE upstream_asset_links SET paper_id = ? WHERE paper_id = ?", (target_paper_id, source_paper_id))
            source_doi = connection.execute(
                "SELECT normalized_doi FROM papers WHERE paper_id = ?",
                (source_paper_id,),
            ).fetchone()["normalized_doi"]
            target_doi = connection.execute(
                "SELECT normalized_doi FROM papers WHERE paper_id = ?",
                (target_paper_id,),
            ).fetchone()["normalized_doi"]
            connection.execute("UPDATE papers SET normalized_doi = NULL WHERE paper_id = ?", (source_paper_id,))
            if source_doi and not target_doi:
                connection.execute(
                    "UPDATE papers SET normalized_doi = ? WHERE paper_id = ?",
                    (source_doi, target_paper_id),
                )
            connection.execute(
                """
                UPDATE paper_identifiers
                SET paper_id = ?
                WHERE paper_id = ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM paper_identifiers AS target_identifiers
                      WHERE target_identifiers.paper_id = ?
                        AND target_identifiers.scheme = paper_identifiers.scheme
                        AND target_identifiers.value = paper_identifiers.value
                  )
                """,
                (target_paper_id, source_paper_id, target_paper_id),
            )
            connection.execute("DELETE FROM paper_identifiers WHERE paper_id = ?", (source_paper_id,))
            connection.execute(
                """
                INSERT OR IGNORE INTO paper_tags(paper_id, tag, created_at)
                SELECT ?, tag, ?
                FROM paper_tags
                WHERE paper_id = ?
                """,
                (target_paper_id, now, source_paper_id),
            )
            connection.execute("DELETE FROM paper_tags WHERE paper_id = ?", (source_paper_id,))
            connection.execute(
                """
                INSERT OR IGNORE INTO paper_collection_members(collection_id, paper_id, created_at)
                SELECT collection_id, ?, ?
                FROM paper_collection_members
                WHERE paper_id = ?
                """,
                (target_paper_id, now, source_paper_id),
            )
            connection.execute("DELETE FROM paper_collection_members WHERE paper_id = ?", (source_paper_id,))
            if archive_source:
                connection.execute("UPDATE papers SET archived = 1, updated_at = ? WHERE paper_id = ?", (now, source_paper_id))
            connection.execute(
                "UPDATE papers SET updated_at = ? WHERE paper_id IN (?, ?)",
                (now, target_paper_id, source_paper_id),
            )

        self.record_provenance_event(
            action="merge_paper_assets",
            source="registry",
            inputs={"target_paper_id": target_paper_id, "source_paper_id": source_paper_id, "archive_source": archive_source},
            outputs=result,
        )
        return result

    def register_remote_parsed_asset(
        self,
        *,
        asset_id: str,
        paper_id: str,
        source_name: str,
        asset_kind: str,
        status: str,
        source_record_id: str | None = None,
        locator: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        quality: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            connection.execute(
                """
                INSERT INTO remote_parsed_assets(
                    asset_id, paper_id, source_name, asset_kind, source_record_id,
                    status, locator_json, provenance_json, quality_json, metadata_json,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id) DO UPDATE SET
                    paper_id = excluded.paper_id,
                    source_name = excluded.source_name,
                    asset_kind = excluded.asset_kind,
                    source_record_id = excluded.source_record_id,
                    status = excluded.status,
                    locator_json = excluded.locator_json,
                    provenance_json = excluded.provenance_json,
                    quality_json = excluded.quality_json,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    asset_id,
                    paper_id,
                    source_name,
                    asset_kind,
                    source_record_id,
                    status,
                    json.dumps(locator or {}, ensure_ascii=False, sort_keys=True),
                    json.dumps(provenance or {}, ensure_ascii=False, sort_keys=True),
                    json.dumps(quality or {}, ensure_ascii=False, sort_keys=True),
                    json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM remote_parsed_assets WHERE asset_id = ?",
                (asset_id,),
            ).fetchone()
        return decode_remote_parsed_asset_row(row)

    def list_remote_parsed_assets(
        self,
        paper_id: str,
        *,
        asset_kind: str | None = None,
        source_name: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["paper_id = ?"]
        params: list[Any] = [paper_id]
        if asset_kind:
            clauses.append("asset_kind = ?")
            params.append(asset_kind)
        if source_name:
            clauses.append("source_name = ?")
            params.append(source_name)
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            rows = connection.execute(
                f"""
                SELECT *
                FROM remote_parsed_assets
                WHERE {' AND '.join(clauses)}
                ORDER BY source_name, asset_kind, created_at
                """,
                tuple(params),
            ).fetchall()
        return [decode_remote_parsed_asset_row(row) for row in rows]

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM papers WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
        return dict(row) if row else None

    def find_paper_by_doi(self, doi: str) -> dict[str, Any] | None:
        normalized_doi = normalize_doi(doi)
        if not normalized_doi:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM papers WHERE normalized_doi = ?",
                (normalized_doi,),
            ).fetchone()
            if not row:
                row = connection.execute(
                    """
                    SELECT papers.*
                    FROM paper_identifiers
                    JOIN papers ON papers.paper_id = paper_identifiers.paper_id
                    WHERE paper_identifiers.scheme = 'doi'
                      AND paper_identifiers.value = ?
                    """,
                    (normalized_doi,),
                ).fetchone()
        return dict(row) if row else None

    def add_source_candidate(
        self,
        *,
        source_name: str,
        query_kind: str,
        status: str,
        paper_id: str | None = None,
        source_record_id: str | None = None,
        doi: str | None = None,
        title: str | None = None,
        year: int | None = None,
        venue: str | None = None,
        candidate_score: float | None = None,
        pdf_candidates: list[dict[str, Any]] | None = None,
        license: str | None = None,
        open_access_status: str | None = None,
        source_url: str | None = None,
        raw_metadata_path: Path | None = None,
        error_summary: str | None = None,
    ) -> dict[str, Any]:
        normalized_doi = normalize_doi(doi)
        candidate_id = stable_id(
            "SRC_",
            source_name,
            query_kind,
            normalized_doi or source_record_id or title or uuid.uuid4().hex,
        )
        with self._connect() as connection:
            if paper_id:
                self._require_paper(connection, paper_id)
            connection.execute(
                """
                INSERT INTO source_candidates(
                    candidate_id, paper_id, source_name, query_kind,
                    source_record_id, normalized_doi, title, year, venue, candidate_score,
                    pdf_candidates_json, license, open_access_status, source_url,
                    status, raw_metadata_path, error_summary, retrieved_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    paper_id = COALESCE(excluded.paper_id, source_candidates.paper_id),
                    source_record_id = COALESCE(excluded.source_record_id, source_candidates.source_record_id),
                    normalized_doi = COALESCE(excluded.normalized_doi, source_candidates.normalized_doi),
                    title = COALESCE(excluded.title, source_candidates.title),
                    year = COALESCE(excluded.year, source_candidates.year),
                    venue = COALESCE(excluded.venue, source_candidates.venue),
                    candidate_score = COALESCE(excluded.candidate_score, source_candidates.candidate_score),
                    pdf_candidates_json = CASE
                        WHEN excluded.pdf_candidates_json != '[]' THEN excluded.pdf_candidates_json
                        ELSE source_candidates.pdf_candidates_json
                    END,
                    license = COALESCE(excluded.license, source_candidates.license),
                    open_access_status = COALESCE(
                        excluded.open_access_status,
                        source_candidates.open_access_status
                    ),
                    source_url = COALESCE(excluded.source_url, source_candidates.source_url),
                    status = excluded.status,
                    raw_metadata_path = COALESCE(excluded.raw_metadata_path, source_candidates.raw_metadata_path),
                    error_summary = excluded.error_summary,
                    retrieved_at = excluded.retrieved_at
                """,
                (
                    candidate_id,
                    paper_id,
                    source_name,
                    query_kind,
                    source_record_id,
                    normalized_doi,
                    title,
                    year,
                    venue,
                    candidate_score,
                    json.dumps(pdf_candidates or [], ensure_ascii=False, sort_keys=True),
                    license,
                    open_access_status,
                    source_url,
                    status,
                    str(raw_metadata_path) if raw_metadata_path else None,
                    error_summary,
                    utc_now(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM source_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return dict(row)

    def list_source_candidates(self, paper_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM source_candidates"
        params: tuple[Any, ...] = ()
        if paper_id:
            query += " WHERE paper_id = ?"
            params = (paper_id,)
        query += " ORDER BY retrieved_at, source_name"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
            paper_ids = {
                str(row["paper_id"])
                for row in rows
                if row["paper_id"]
            }
            papers = {
                row["paper_id"]: dict(row)
                for row in connection.execute(
                    f"SELECT * FROM papers WHERE paper_id IN ({','.join('?' for _ in paper_ids)})",
                    tuple(paper_ids),
                ).fetchall()
            } if paper_ids else {}
        return [decode_source_candidate_row(row, papers.get(row["paper_id"])) for row in rows]

    def get_source_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM source_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_source_candidate_status(
        self,
        candidate_id: str,
        *,
        status: str,
        error_summary: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self._connect() as connection:
            if not connection.execute(
                "SELECT 1 FROM source_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone():
                raise LibraryRegistryError(f"source candidate not found: {candidate_id}")
            connection.execute(
                """
                UPDATE source_candidates
                SET status = ?, error_summary = ?, retrieved_at = ?
                WHERE candidate_id = ?
                """,
                (status, error_summary, now, candidate_id),
            )
            row = connection.execute(
                "SELECT * FROM source_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return dict(row)

    def list_papers(
        self,
        *,
        query: str | None = None,
        tag: str | None = None,
        collection_id: str | None = None,
        include_archived: bool = True,
        metadata_ready: bool | None = None,
        pdf_ready: bool | None = None,
        local_mineru_ready: bool | None = None,
        evidence_ready: bool | None = None,
        sciverse_ai_ready: str | None = None,
        extraction_ready: bool | None = None,
        year_ready: bool | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT DISTINCT papers.* FROM papers"
        joins: list[str] = []
        clauses: list[str] = []
        params: list[Any] = []
        if tag:
            joins.append("JOIN paper_tags ON paper_tags.paper_id = papers.paper_id")
            clauses.append("paper_tags.tag = ?")
            params.append(tag)
        if collection_id:
            joins.append(
                "JOIN paper_collection_members pcm ON pcm.paper_id = papers.paper_id"
            )
            clauses.append("pcm.collection_id = ?")
            params.append(collection_id)
        if query:
            clauses.append(
                "(lower(COALESCE(papers.title, '')) LIKE ? OR lower(COALESCE(papers.normalized_doi, '')) LIKE ? OR lower(COALESCE(papers.venue, '')) LIKE ?)"
            )
            like = f"%{query.lower()}%"
            params.extend([like, like, like])
        if not include_archived:
            clauses.append("papers.archived = 0")
        if joins:
            sql += " " + " ".join(joins)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY papers.updated_at DESC, papers.created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(sql, tuple(params)).fetchall()
        papers = [dict(row) for row in rows]
        readiness_filters = {
            "metadata_ready": metadata_ready,
            "pdf_ready": pdf_ready,
            "local_mineru_ready": local_mineru_ready,
            "evidence_ready": evidence_ready,
            "extraction_ready": extraction_ready,
            "year_ready": year_ready,
        }
        if any(value is not None for value in readiness_filters.values()) or sciverse_ai_ready is not None:
            filtered: list[dict[str, Any]] = []
            for paper in papers:
                readiness = self.paper_readiness(paper["paper_id"])
                if any(
                    expected is not None and bool(readiness[key]) is not expected
                    for key, expected in readiness_filters.items()
                ):
                    continue
                if sciverse_ai_ready is not None and readiness["sciverse_ai_ready"] != sciverse_ai_ready:
                    continue
                filtered.append(paper)
            return filtered
        return papers

    def set_paper_tags(self, paper_id: str, tags: list[str]) -> list[str]:
        normalized = sorted({tag.strip() for tag in tags if tag and tag.strip()})
        now = utc_now()
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            connection.execute("DELETE FROM paper_tags WHERE paper_id = ?", (paper_id,))
            connection.executemany(
                """
                INSERT INTO paper_tags(paper_id, tag, created_at)
                VALUES (?, ?, ?)
                """,
                [(paper_id, tag, now) for tag in normalized],
            )
            connection.execute(
                "UPDATE papers SET updated_at = ? WHERE paper_id = ?",
                (now, paper_id),
            )
        return normalized

    def list_paper_tags(self, paper_id: str) -> list[str]:
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            rows = connection.execute(
                "SELECT tag FROM paper_tags WHERE paper_id = ? ORDER BY tag",
                (paper_id,),
            ).fetchall()
        return [str(row["tag"]) for row in rows]

    def upsert_collection(self, *, name: str, description: str | None = None) -> dict[str, Any]:
        cleaned = name.strip()
        if not cleaned:
            raise LibraryRegistryError("collection name cannot be empty")
        now = utc_now()
        collection_id = stable_id("COLL_", cleaned)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO paper_collections(collection_id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description = COALESCE(excluded.description, paper_collections.description),
                    updated_at = excluded.updated_at
                """,
                (collection_id, cleaned, description, now, now),
            )
            row = connection.execute(
                "SELECT * FROM paper_collections WHERE name = ?",
                (cleaned,),
            ).fetchone()
        return dict(row)

    def add_paper_to_collection(self, paper_id: str, collection_id: str) -> dict[str, Any]:
        now = utc_now()
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            if not connection.execute(
                "SELECT 1 FROM paper_collections WHERE collection_id = ?",
                (collection_id,),
            ).fetchone():
                raise LibraryRegistryError(f"collection not found: {collection_id}")
            connection.execute(
                """
                INSERT INTO paper_collection_members(collection_id, paper_id, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(collection_id, paper_id) DO NOTHING
                """,
                (collection_id, paper_id, now),
            )
            connection.execute(
                "UPDATE papers SET updated_at = ? WHERE paper_id = ?",
                (now, paper_id),
            )
            row = connection.execute(
                """
                SELECT paper_collection_members.*, paper_collections.name
                FROM paper_collection_members
                JOIN paper_collections ON paper_collections.collection_id = paper_collection_members.collection_id
                WHERE paper_collection_members.collection_id = ?
                  AND paper_collection_members.paper_id = ?
                """,
                (collection_id, paper_id),
            ).fetchone()
        return dict(row)

    def archive_paper(self, paper_id: str, *, archived: bool = True) -> dict[str, Any]:
        now = utc_now()
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            connection.execute(
                "UPDATE papers SET archived = ?, updated_at = ? WHERE paper_id = ?",
                (1 if archived else 0, now, paper_id),
            )
            row = connection.execute(
                "SELECT * FROM papers WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
        return dict(row)

    def backfill_paper_year_from_candidates(
        self,
        paper_id: str,
        *,
        force: bool = False,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        paper = self.get_paper(paper_id)
        if not paper:
            raise LibraryRegistryError(f"paper not found: {paper_id}")
        if paper.get("published_year") and not paper.get("year_needs_review") and not force:
            return {
                "paper_id": paper_id,
                "status": "already_ready",
                "dry_run": dry_run,
                "published_year": paper.get("published_year"),
                "published_date": paper.get("published_date"),
                "year_source": paper.get("year_source"),
                "year_confidence": paper.get("year_confidence"),
                "year_needs_review": bool(paper.get("year_needs_review")),
                "recommended_next_action": "ready_for_cutoff_experiment",
            }

        candidate = best_year_candidate(self.list_source_candidates(paper_id), paper)
        if candidate is None:
            return {
                "paper_id": paper_id,
                "status": "blocked",
                "dry_run": dry_run,
                "blocking_reason": "missing_year",
                "recommended_next_action": "review_or_enrich_metadata",
            }

        confidence = year_confidence_for_candidate(candidate, paper)
        needs_review = confidence < 0.8
        published_year = int(candidate["year"])
        published_date = candidate.get("published_date")
        year_source = str(candidate.get("source_name") or "source_candidate")
        result = {
            "paper_id": paper_id,
            "status": "would_update" if dry_run else "updated",
            "dry_run": dry_run,
            "published_year": published_year,
            "published_date": published_date,
            "year_source": year_source,
            "year_confidence": confidence,
            "year_needs_review": needs_review,
            "source_candidate_id": candidate.get("candidate_id"),
            "recommended_next_action": "review_year_metadata" if needs_review else "ready_for_cutoff_experiment",
        }
        if dry_run:
            return result

        now = utc_now()
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            connection.execute(
                """
                UPDATE papers
                SET year = COALESCE(year, ?),
                    published_year = ?,
                    published_date = COALESCE(?, published_date),
                    year_source = ?,
                    year_confidence = ?,
                    year_needs_review = ?,
                    updated_at = ?
                WHERE paper_id = ?
                """,
                (
                    published_year,
                    published_year,
                    published_date,
                    year_source,
                    confidence,
                    int(needs_review),
                    now,
                    paper_id,
                ),
            )
        self.record_provenance_event(
            action="backfill_paper_year",
            source="registry",
            inputs={"paper_id": paper_id, "force": force, "source_candidate_id": candidate.get("candidate_id")},
            outputs={
                "published_year": published_year,
                "year_source": year_source,
                "year_confidence": confidence,
                "year_needs_review": needs_review,
            },
        )
        return result

    def delete_paper(
        self,
        paper_id: str,
        *,
        confirm: bool = False,
        delete_assets: bool = False,
    ) -> dict[str, Any]:
        if not confirm:
            raise LibraryRegistryError("硬删除需要二次确认：请传 confirm=true。")
        paper = self.get_paper(paper_id)
        if not paper:
            raise LibraryRegistryError(f"paper not found: {paper_id}")
        asset_paths = [self.resolve_local_path(asset["local_path"]) for asset in self.list_pdf_assets(paper_id)]
        paper_dir = self.library_root / "papers" / paper_id
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            connection.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))
        if delete_assets:
            for path in asset_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    raise LibraryRegistryError(f"删除资产失败：{path}") from exc
            if paper_dir.exists():
                shutil.rmtree(paper_dir, ignore_errors=True)
        return {"paper_id": paper_id, "deleted": True, "assets_deleted": bool(delete_assets)}

    def create_processing_job(
        self,
        *,
        paper_id: str,
        pdf_asset_id: str,
        tool: str,
        tool_config: dict[str, Any],
        status: str,
        server_url_redacted: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        tool_config_hash = hash_json(tool_config)
        now = utc_now()
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            self._require_pdf_asset(connection, pdf_asset_id)
            if status == "ready":
                existing = connection.execute(
                    """
                    SELECT * FROM processing_jobs
                    WHERE pdf_asset_id = ? AND tool = ? AND tool_config_hash = ?
                    AND status = 'ready'
                    """,
                    (pdf_asset_id, tool, tool_config_hash),
                ).fetchone()
                if existing:
                    return dict(existing)

            job_id = stable_id("JOB_", pdf_asset_id, tool, tool_config_hash, uuid.uuid4().hex)
            connection.execute(
                """
                INSERT INTO processing_jobs(
                    job_id, paper_id, pdf_asset_id, tool, tool_config_hash,
                    tool_config_json, status, server_url_redacted, started_at,
                    finished_at, error, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    paper_id,
                    pdf_asset_id,
                    tool,
                    tool_config_hash,
                    json.dumps(tool_config, ensure_ascii=False, sort_keys=True),
                    status,
                    server_url_redacted,
                    now,
                    now if status in {"ready", "failed", "blocked"} else None,
                    error,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM processing_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return dict(row)

    def find_ready_processing_job(
        self,
        *,
        pdf_asset_id: str,
        tool: str,
        tool_config: dict[str, Any],
    ) -> dict[str, Any] | None:
        tool_config_hash = hash_json(tool_config)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT processing_jobs.*
                FROM processing_jobs
                WHERE pdf_asset_id = ? AND tool = ? AND tool_config_hash = ?
                AND status = 'ready'
                AND EXISTS (
                    SELECT 1
                    FROM mineru_artifacts
                    WHERE mineru_artifacts.job_id = processing_jobs.job_id
                    AND mineru_artifacts.status = 'ready'
                )
                """,
                (pdf_asset_id, tool, tool_config_hash),
            ).fetchone()
        return dict(row) if row else None

    def register_mineru_artifact(
        self,
        *,
        job_id: str,
        kind: str,
        path: Path,
        status: str,
    ) -> dict[str, Any]:
        checksum = sha256_path(path)
        manifest = path_manifest(path)
        relative_path = self.relative_local_path(path)
        now = utc_now()
        artifact_id = stable_id("ART_", job_id, kind, str(relative_path))
        with self._connect() as connection:
            self._require_processing_job(connection, job_id)
            connection.execute(
                """
                INSERT INTO mineru_artifacts(
                    artifact_id, job_id, kind, local_path, checksum, status,
                    manifest_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    checksum = excluded.checksum,
                    status = excluded.status,
                    manifest_json = excluded.manifest_json
                """,
                (
                    artifact_id,
                    job_id,
                    kind,
                    str(relative_path),
                    checksum,
                    status,
                    json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM mineru_artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        return decode_artifact_row(row)

    def list_mineru_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM mineru_artifacts WHERE job_id = ? ORDER BY kind",
                (job_id,),
            ).fetchall()
        return [decode_artifact_row(row) for row in rows]

    def store_external_artifact(
        self,
        *,
        paper_id: str,
        kind: str,
        path: Path,
        source: str,
        status: str = "ready",
    ) -> dict[str, Any]:
        """Store a non-mineru artifact (e.g. LogicKG hypergraph JSON) for a paper.

        Idempotent on (paper_id, kind, source). Computes sha256 + manifest,
        validates the path is inside library_root.
        """
        self.get_paper(paper_id)  # raises if missing
        checksum = sha256_path(path)
        manifest = path_manifest(path)
        relative_path = self.relative_local_path(path)
        now = utc_now()
        artifact_id = stable_id("ART_", paper_id, kind, source, str(relative_path))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO external_artifacts(
                    artifact_id, paper_id, kind, source, local_path,
                    checksum, status, manifest_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    local_path = excluded.local_path,
                    checksum = excluded.checksum,
                    status = excluded.status,
                    manifest_json = excluded.manifest_json
                """,
                (
                    artifact_id, paper_id, kind, source, str(relative_path),
                    checksum, status, json.dumps(manifest, ensure_ascii=False, sort_keys=True), now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM external_artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        return decode_artifact_row(row)

    def list_external_artifacts(
        self, paper_id: str, kind: str | None = None
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if kind:
                rows = connection.execute(
                    "SELECT * FROM external_artifacts WHERE paper_id = ? AND kind = ? ORDER BY created_at",
                    (paper_id, kind),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM external_artifacts WHERE paper_id = ? ORDER BY kind, created_at",
                    (paper_id,),
                ).fetchall()
        return [decode_artifact_row(row) for row in rows]

    def list_processing_jobs(self, paper_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM processing_jobs WHERE paper_id = ? ORDER BY created_at",
                (paper_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def paper_artifact_manifest(self, paper_id: str) -> dict[str, Any]:
        paper = self.get_paper(paper_id)
        if not paper:
            raise LibraryRegistryError(f"paper not found: {paper_id}")
        pdf_assets = self.list_pdf_assets(paper_id)
        jobs = self.list_processing_jobs(paper_id)
        artifacts = self.list_paper_artifacts(paper_id)
        sciverse_ai_ready_assets = [
            artifact
            for artifact in artifacts
            if artifact.get("kind") == "sciverse_ai_ready" and artifact.get("status") == "ready"
        ]
        ready_kinds = {
            str(artifact["kind"]): artifact
            for artifact in artifacts
            if artifact.get("status") == "ready"
        }
        ready_mineru_jobs = [
            job
            for job in jobs
            if job.get("tool") == "mineru" and job.get("status") == "ready"
        ]
        failed_or_blocked_jobs = [
            job
            for job in jobs
            if job.get("tool") == "mineru" and job.get("status") in {"failed", "blocked"}
        ]

        required = {
            "original_pdf": manifest_item(
                "ready" if pdf_assets else "missing",
                count=len(pdf_assets),
                artifact_id=pdf_assets[0]["asset_id"] if pdf_assets else None,
                next_action=None if pdf_assets else "upload_pdf",
            ),
            "mineru_job": manifest_item(
                "ready" if ready_mineru_jobs else "missing",
                count=len(ready_mineru_jobs),
                job_id=ready_mineru_jobs[-1]["job_id"] if ready_mineru_jobs else None,
                next_action="process_with_mineru" if pdf_assets and not ready_mineru_jobs else None,
            ),
            "mineru_content_list": artifact_manifest_item(
                ready_kinds.get("mineru_content_list") or ready_kinds.get("content_list"),
                next_action="process_with_mineru" if pdf_assets else "upload_pdf",
            ),
        }
        required["readable_body"] = manifest_item(
            "ready"
            if required["mineru_content_list"]["status"] == "ready"
            or "mineru_markdown" in ready_kinds
            or sciverse_ai_ready_assets
            else "missing",
            artifact_id=(
                required["mineru_content_list"].get("artifact_id")
                or ready_kinds.get("mineru_markdown", {}).get("artifact_id")
                or (
                    sciverse_ai_ready_assets[0]["artifact_id"]
                    if sciverse_ai_ready_assets
                    else None
                )
            ),
            next_action="process_with_mineru" if pdf_assets else "upload_pdf",
        )
        optional = {
            "sciverse_ai_ready": remote_artifact_manifest_item(
                sciverse_ai_ready_assets[0] if sciverse_ai_ready_assets else None,
                next_action="discover_sciverse",
            ),
            "mineru_archive": artifact_manifest_item(ready_kinds.get("mineru_zip")),
            "mineru_raw_bundle": artifact_manifest_item(ready_kinds.get("mineru_raw_bundle")),
            "mineru_response_json": artifact_manifest_item(ready_kinds.get("mineru_response_json")),
            "mineru_result_manifest": artifact_manifest_item(ready_kinds.get("mineru_result_manifest")),
            "mineru_markdown": artifact_manifest_item(ready_kinds.get("mineru_markdown")),
            "mineru_middle_json": artifact_manifest_item(
                ready_kinds.get("mineru_middle_json") or ready_kinds.get("mineru_middle")
            ),
            "mineru_model_json": artifact_manifest_item(
                ready_kinds.get("mineru_model_json") or ready_kinds.get("mineru_model")
            ),
            "mineru_layout_json": artifact_manifest_item(
                ready_kinds.get("mineru_layout_json") or ready_kinds.get("mineru_layout")
            ),
            "mineru_images_dir": artifact_manifest_item(ready_kinds.get("mineru_images_dir")),
            "mineru_tables_dir": artifact_manifest_item(ready_kinds.get("mineru_tables_dir")),
            "mineru_equations_dir": artifact_manifest_item(ready_kinds.get("mineru_equations_dir")),
        }
        required_ready = all(item["status"] == "ready" for item in required.values())
        status = "ready" if required_ready else "blocked"
        errors = [
            {
                "job_id": job["job_id"],
                "status": job["status"],
                "error": job.get("error"),
            }
            for job in failed_or_blocked_jobs
        ]
        return {
            "paper_id": paper_id,
            "status": status,
            "required": required,
            "optional": optional,
            "counts": {
                "pdf_assets": len(pdf_assets),
                "processing_jobs": len(jobs),
                "artifacts": len(artifacts),
            },
            "errors": errors,
        }

    def paper_readiness(self, paper_id: str) -> dict[str, Any]:
        paper = self.get_paper(paper_id)
        if not paper:
            raise LibraryRegistryError(f"paper not found: {paper_id}")
        identifiers = self.list_identifiers(paper_id)
        source_candidates = self.list_source_candidates(paper_id)
        pdf_assets = self.list_pdf_assets(paper_id)
        jobs = self.list_processing_jobs(paper_id)
        evidence = self.evidence_coverage(paper_id)
        manifest = self.paper_artifact_manifest(paper_id)
        sciverse_ai_ready_assets = self.list_remote_parsed_assets(
            paper_id,
            asset_kind="sciverse_ai_ready",
            source_name="sciverse",
        )
        recovered_sciverse_asset = recovered_sciverse_ai_ready_asset(
            paper_id,
            self.list_evidence_units(paper_id, source_type="sciverse_chunk"),
        )
        all_sciverse_ai_ready_assets = [*sciverse_ai_ready_assets]
        if recovered_sciverse_asset and not any(
            asset.get("asset_id") == recovered_sciverse_asset["asset_id"]
            and asset.get("status") == "ready"
            for asset in all_sciverse_ai_ready_assets
        ):
            all_sciverse_ai_ready_assets.append(recovered_sciverse_asset)
        ready_sciverse_ai_ready = [
            asset for asset in all_sciverse_ai_ready_assets if asset.get("status") == "ready"
        ]
        best_sciverse_asset = best_remote_parsed_asset(ready_sciverse_ai_ready)
        best_sciverse_quality = (
            best_sciverse_asset.get("quality", {}) if best_sciverse_asset else {}
        )

        metadata_status = "ready" if paper.get("normalized_doi") or paper.get("title") or identifiers else "missing"
        pdf_status = "ready" if pdf_assets else "missing"
        mineru_jobs = [job for job in jobs if job.get("tool") == "mineru"]
        if pdf_status != "ready":
            mineru_status = "missing"
        elif manifest["status"] == "ready":
            mineru_status = "ready"
        elif any(job.get("status") in {"failed", "blocked"} for job in mineru_jobs):
            mineru_status = "blocked"
        else:
            mineru_status = "missing"
        sciverse_candidates = [
            candidate
            for candidate in source_candidates
            if candidate.get("source_name") == "sciverse"
        ]
        evidence_status = "ready" if evidence["total"] else "missing"
        published_year = paper.get("published_year") or paper.get("year")
        published_date = paper.get("published_date")
        year_needs_review = bool(paper.get("year_needs_review"))
        year_ready = bool(published_year) and not year_needs_review
        job_errors = [
            {
                "job_id": job["job_id"],
                "kind": job.get("tool") or "api",
                "status": job["status"],
                "error": job.get("error"),
            }
            for job in jobs
            if job.get("status") in {"failed", "blocked"}
        ]
        next_actions: list[str] = []
        if metadata_status != "ready":
            next_actions.append("add_metadata")
        if pdf_status != "ready" and not best_sciverse_asset:
            next_actions.append("upload_pdf")
        if pdf_status == "ready" and mineru_status != "ready" and not best_sciverse_asset:
            next_actions.append("process_with_mineru")
        if mineru_status == "ready" and evidence_status != "ready":
            next_actions.append("build_evidence_units")
        if not sciverse_candidates and not best_sciverse_asset:
            next_actions.append("discover_sciverse")
        if best_sciverse_asset and mineru_status != "ready":
            next_actions.append("run_local_mineru")

        dedupe_diagnostic = self.find_duplicate_asset_diagnostics(paper_id) if not pdf_assets else None
        if dedupe_diagnostic:
            next_actions.append("merge_or_rebind_asset")

        has_readable_body = (
            mineru_status == "ready"
            or str(best_sciverse_quality.get("readiness_status") or "") == "ready"
            or bool(best_sciverse_asset)
        )
        local_mineru_ready = mineru_status == "ready"
        evidence_ready = evidence_status == "ready"
        extraction_ready = has_readable_body and evidence_ready
        sciverse_ai_ready_status = (
            str(best_sciverse_quality.get("readiness_status"))
            if best_sciverse_quality.get("readiness_status")
            else ("ready" if best_sciverse_asset else "missing")
        )
        identity_chain = self.paper_identity_chain(paper_id)
        blocking_reason = readiness_blocking_reason(
            metadata_ready=metadata_status == "ready",
            pdf_ready=pdf_status == "ready",
            local_mineru_ready=local_mineru_ready,
            evidence_ready=evidence_ready,
            sciverse_ai_ready=sciverse_ai_ready_status,
            year_ready=year_ready,
            published_year=published_year,
            year_needs_review=year_needs_review,
            dedupe_diagnostic=dedupe_diagnostic,
        )
        recommended_next_action = recommended_next_action_for_blocking_reason(blocking_reason, next_actions)
        status = "ready" if metadata_status == "ready" and has_readable_body else "blocked"
        return {
            "paper_id": paper_id,
            "status": status,
            "metadata_ready": metadata_status == "ready",
            "pdf_ready": pdf_status == "ready",
            "local_mineru_ready": local_mineru_ready,
            "evidence_ready": evidence_ready,
            "sciverse_ai_ready": sciverse_ai_ready_status,
            "extraction_ready": extraction_ready,
            "year_ready": year_ready,
            "published_year": published_year,
            "published_date": published_date,
            "year_source": paper.get("year_source"),
            "year_confidence": paper.get("year_confidence"),
            "year_needs_review": year_needs_review,
            "blocking_reason": blocking_reason,
            "recommended_next_action": recommended_next_action,
            "dedupe_diagnostic": dedupe_diagnostic,
            "identity_chain": identity_chain,
            "sections": {
                "metadata": {
                    "status": metadata_status,
                    "identifiers": len(identifiers),
                    "has_title": bool(paper.get("title")),
                    "has_doi": bool(paper.get("normalized_doi")),
                    "published_year": published_year,
                    "published_date": published_date,
                    "year_source": paper.get("year_source"),
                    "year_confidence": paper.get("year_confidence"),
                    "year_needs_review": year_needs_review,
                },
                "year": {
                    "status": "ready" if year_ready else "needs_review" if published_year else "missing",
                    "published_year": published_year,
                    "published_date": published_date,
                    "year_source": paper.get("year_source"),
                    "year_confidence": paper.get("year_confidence"),
                    "year_needs_review": year_needs_review,
                },
                "pdf": {
                    "status": pdf_status,
                    "count": len(pdf_assets),
                    "assets": pdf_assets,
                    "dedupe_diagnostic": dedupe_diagnostic,
                },
                "mineru": {
                    "status": mineru_status,
                    "required": manifest["required"],
                    "optional": manifest["optional"],
                    "jobs": jobs,
                    "errors": manifest["errors"],
                },
                "sciverse": {
                    "status": "ready" if sciverse_candidates else "missing",
                    "candidate_count": len(sciverse_candidates),
                },
                "sciverse_ai_ready": {
                    "status": sciverse_ai_ready_status,
                    "asset_count": len(all_sciverse_ai_ready_assets),
                    "quality_tier": best_sciverse_quality.get("quality_tier"),
                    "label": best_sciverse_quality.get("label"),
                    "next_action": best_sciverse_quality.get("next_action"),
                    "quality_issues": best_sciverse_quality.get("quality_issues", []),
                    "assets": ready_sciverse_ai_ready,
                },
                "evidence": {
                    "status": evidence_status,
                    "ready_units": evidence["by_status"].get("candidate", 0)
                    + evidence["by_status"].get("ready", 0)
                    + evidence["by_status"].get("accepted", 0),
                    "total_units": evidence["total"],
                    "by_source_type": evidence["by_source_type"],
                    "by_modality": evidence["by_modality"],
                    "by_status": evidence["by_status"],
                    "coverage": evidence,
                },
                "tasks": {
                    "status": "blocked" if job_errors else "ready",
                    "errors": job_errors,
                },
            },
            "next_actions": next_actions,
        }

    def paper_identity_chain(self, paper_id: str) -> dict[str, Any]:
        paper = self.get_paper(paper_id)
        if not paper:
            raise LibraryRegistryError(f"paper not found: {paper_id}")
        pdf_assets = self.list_pdf_assets(paper_id)
        jobs = self.list_processing_jobs(paper_id)
        with self._connect() as connection:
            evidence_asset_rows = connection.execute(
                """
                SELECT asset_id, COUNT(*) AS count
                FROM evidence_units
                WHERE paper_id = ?
                GROUP BY asset_id
                ORDER BY asset_id
                """,
                (paper_id,),
            ).fetchall()
            evidence_total = connection.execute(
                "SELECT COUNT(*) AS count FROM evidence_units WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()["count"]
        pdf_owner_links = [
            {"asset_id": asset["asset_id"], "paper_id": asset["paper_id"]}
            for asset in pdf_assets
        ]
        job_owner_links = [
            {
                "job_id": job["job_id"],
                "paper_id": job["paper_id"],
                "pdf_asset_id": job["pdf_asset_id"],
            }
            for job in jobs
        ]
        evidence_asset_ids = [str(row["asset_id"]) for row in evidence_asset_rows if row["asset_id"]]
        mismatches: list[dict[str, Any]] = []
        for asset in pdf_owner_links:
            if asset["paper_id"] != paper_id:
                mismatches.append({"kind": "pdf_asset", **asset})
        for job in job_owner_links:
            if job["paper_id"] != paper_id:
                mismatches.append({"kind": "processing_job", **job})
        return {
            "paper_id": paper_id,
            "consistent": not mismatches,
            "pdf_assets": pdf_owner_links,
            "processing_jobs": job_owner_links,
            "evidence": {
                "paper_id": paper_id,
                "total_units": evidence_total,
                "asset_ids": evidence_asset_ids,
            },
            "mismatches": mismatches,
        }

    def paper_sciverse_ai_ready(self, paper_id: str) -> dict[str, Any]:
        paper = self.get_paper(paper_id)
        if not paper:
            raise LibraryRegistryError(f"paper not found: {paper_id}")

        remote_assets = self.list_remote_parsed_assets(
            paper_id,
            asset_kind="sciverse_ai_ready",
            source_name="sciverse",
        )
        recovered_asset = recovered_sciverse_ai_ready_asset(
            paper_id,
            self.list_evidence_units(paper_id, source_type="sciverse_chunk"),
        )
        all_assets = [*remote_assets]
        if recovered_asset and not any(
            asset.get("asset_id") == recovered_asset["asset_id"] and asset.get("status") == "ready"
            for asset in all_assets
        ):
            all_assets.append(recovered_asset)
        ready_assets = [asset for asset in all_assets if asset.get("status") == "ready"]
        best_asset = best_remote_parsed_asset(ready_assets) or best_remote_parsed_asset(all_assets)
        quality = best_asset.get("quality", {}) if best_asset else {}
        status = (
            str(quality.get("readiness_status"))
            if quality.get("readiness_status")
            else ("ready" if best_asset and best_asset.get("status") == "ready" else "missing")
        )
        quality_tier = quality.get("quality_tier")
        next_action = quality.get("next_action")

        source_candidates = [
            candidate
            for candidate in self.list_source_candidates(paper_id)
            if candidate.get("source_name") == "sciverse"
        ]
        pdf_assets = self.list_pdf_assets(paper_id)
        if not next_action:
            next_action = "discover_sciverse" if not best_asset else "run_local_mineru"

        next_links = {
            "self": f"/api/library/papers/{paper_id}/sciverse-ai-ready",
            "paper": f"/api/library/papers/{paper_id}",
            "artifacts": f"/api/library/papers/{paper_id}/artifacts",
            "evidence": f"/api/library/papers/{paper_id}/evidence?source_type=sciverse_chunk",
            "sciverse_ingest": f"/api/library/papers/{paper_id}/sciverse-ingest",
        }
        return {
            "paper_id": paper_id,
            "status": status,
            "quality_tier": quality_tier,
            "label": quality.get("label"),
            "next_action": next_action,
            "quality_issues": quality.get("quality_issues", []),
            "remote_asset": best_asset,
            "assets": ready_assets,
            "asset_count": len(all_assets),
            "candidate_count": len(source_candidates),
            "source_candidates": source_candidates,
            "actions": sciverse_ai_ready_actions(
                paper_id=paper_id,
                status=status,
                quality_tier=str(quality_tier or ""),
                next_action=str(next_action or ""),
                has_remote_asset=bool(best_asset and best_asset.get("status") == "ready"),
                has_pdf=bool(pdf_assets),
            ),
            "next": next_links,
        }

    def list_paper_artifacts(self, paper_id: str) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []
        for pdf_asset in self.list_pdf_assets(paper_id):
            artifacts.append(
                {
                    "artifact_id": pdf_asset["asset_id"],
                    "paper_id": paper_id,
                    "kind": "pdf",
                    "local_path": pdf_asset["local_path"],
                    "checksum": pdf_asset["sha256"],
                    "status": pdf_asset["status"],
                    "source": pdf_asset["source_kind"],
                }
            )
        for remote_asset in self.list_remote_parsed_assets(paper_id):
            artifacts.append(
                {
                    "artifact_id": remote_asset["asset_id"],
                    "paper_id": paper_id,
                    "kind": remote_asset["asset_kind"],
                    "local_path": None,
                    "checksum": None,
                    "status": remote_asset["status"],
                    "source": remote_asset["source_name"],
                    "remote": True,
                    "locator": remote_asset["locator"],
                    "provenance": remote_asset["provenance"],
                    "quality": remote_asset["quality"],
                    "metadata": remote_asset["metadata"],
                }
            )
        recovered_sciverse_asset = recovered_sciverse_ai_ready_asset(
            paper_id,
            self.list_evidence_units(paper_id, source_type="sciverse_chunk"),
        )
        if recovered_sciverse_asset and not any(
            artifact.get("kind") == "sciverse_ai_ready" and artifact.get("status") == "ready"
            for artifact in artifacts
        ):
            artifacts.append(
                {
                    "artifact_id": recovered_sciverse_asset["asset_id"],
                    "paper_id": paper_id,
                    "kind": recovered_sciverse_asset["asset_kind"],
                    "local_path": None,
                    "checksum": None,
                    "status": recovered_sciverse_asset["status"],
                    "source": recovered_sciverse_asset["source_name"],
                    "remote": True,
                    "locator": recovered_sciverse_asset["locator"],
                    "provenance": recovered_sciverse_asset["provenance"],
                    "quality": recovered_sciverse_asset["quality"],
                    "metadata": recovered_sciverse_asset["metadata"],
                }
            )
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT ma.*, pj.paper_id
                FROM mineru_artifacts ma
                JOIN processing_jobs pj ON pj.job_id = ma.job_id
                WHERE pj.paper_id = ?
                ORDER BY ma.kind, ma.created_at
                """,
                (paper_id,),
            ).fetchall()
        artifacts.extend(decode_artifact_row(row) for row in rows)
        # external (non-mineru) artifacts: LogicKG hypergraph, future downstream products
        artifacts.extend(self.list_external_artifacts(paper_id))
        return artifacts

    def register_evidence_unit(
        self,
        *,
        evidence_id: str,
        paper_id: str,
        asset_id: str,
        source_type: str,
        source_id: str,
        text: str,
        content_hash: str,
        modality: str = "text",
        status: str = "candidate",
        page_idx: int | None = None,
        locator: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        quality: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            connection.execute(
                """
                INSERT INTO evidence_units(
                    evidence_id, paper_id, asset_id, source_type, source_id,
                    modality, text, content_hash, status, page_idx,
                    locator_json, provenance_json, quality_json, metadata_json,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evidence_id) DO UPDATE SET
                    text = excluded.text,
                    content_hash = excluded.content_hash,
                    status = excluded.status,
                    page_idx = excluded.page_idx,
                    locator_json = excluded.locator_json,
                    provenance_json = excluded.provenance_json,
                    quality_json = excluded.quality_json,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    evidence_id,
                    paper_id,
                    asset_id,
                    source_type,
                    source_id,
                    modality,
                    text,
                    content_hash,
                    status,
                    page_idx,
                    json.dumps(locator or {}, ensure_ascii=False, sort_keys=True),
                    json.dumps(provenance or {}, ensure_ascii=False, sort_keys=True),
                    json.dumps(quality or {}, ensure_ascii=False, sort_keys=True),
                    json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM evidence_units WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()
        return self._decode_evidence_row(row)

    def list_evidence_units(
        self,
        paper_id: str,
        *,
        source_type: str | None = None,
        modality: str | None = None,
        page_idx: int | None = None,
    ) -> list[dict[str, Any]]:
        where, params = evidence_filters(
            paper_id=paper_id,
            source_type=source_type,
            modality=modality,
            page_idx=page_idx,
        )
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            rows = connection.execute(
                f"""
                SELECT *
                FROM evidence_units
                WHERE {where}
                ORDER BY source_type, source_id, created_at
                """,
                params,
            ).fetchall()
        return [self._decode_evidence_row(row) for row in rows]

    def register_references(
        self,
        *,
        paper_id: str,
        references: list[dict[str, Any]],
        source: str,
    ) -> int:
        """Store a paper's outgoing references (papers it cites).

        Multi-source: each source ('openalex'/'crossref'/...) upserts its own
        rows keyed by (paper_id, source, source_ref_id). references items:
        {ref_doi?, ref_title?, ref_year?, ref_openalex_id?, source_ref_id,
         raw?}. Returns count written. Existing rows for (paper_id, source)
        are replaced (delete+insert) so re-fetching refreshes the set.
        """
        now = utc_now()
        if not references:
            return 0
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            connection.execute(
                "DELETE FROM paper_references WHERE paper_id = ? AND source = ?",
                (paper_id, source),
            )
            written = 0
            for ref in references:
                src_ref_id = ref.get("source_ref_id") or ref.get("ref_openalex_id") or ref.get("ref_doi") or ""
                if not src_ref_id:
                    continue
                connection.execute(
                    """
                    INSERT INTO paper_references(
                        paper_id, ref_doi, ref_title, ref_year, ref_openalex_id,
                        source, source_ref_id, raw_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(paper_id, source, source_ref_id) DO UPDATE SET
                        ref_doi = excluded.ref_doi,
                        ref_title = excluded.ref_title,
                        ref_year = excluded.ref_year,
                        ref_openalex_id = excluded.ref_openalex_id,
                        raw_json = excluded.raw_json
                    """,
                    (
                        paper_id,
                        ref.get("ref_doi"),
                        ref.get("ref_title"),
                        ref.get("ref_year"),
                        ref.get("ref_openalex_id"),
                        source,
                        src_ref_id,
                        json.dumps(ref.get("raw") or {}, ensure_ascii=False),
                        now,
                    ),
                )
                written += 1
            return written

    def list_references(self, paper_id: str, *, source: str | None = None) -> list[dict[str, Any]]:
        """Outgoing references (papers this paper cites)."""
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            if source:
                rows = connection.execute(
                    "SELECT * FROM paper_references WHERE paper_id = ? AND source = ? ORDER BY created_at",
                    (paper_id, source),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM paper_references WHERE paper_id = ? ORDER BY source, created_at",
                    (paper_id,),
                ).fetchall()
        return [dict(row) for row in rows]

    def register_citations(
        self,
        *,
        paper_id: str,
        citations: list[dict[str, Any]],
        source: str,
    ) -> int:
        """Store a paper's incoming citations (papers citing it)."""
        now = utc_now()
        if not citations:
            return 0
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            connection.execute(
                "DELETE FROM paper_citations WHERE paper_id = ? AND source = ?",
                (paper_id, source),
            )
            written = 0
            for cit in citations:
                src_ref_id = cit.get("source_ref_id") or cit.get("citing_openalex_id") or cit.get("citing_doi") or ""
                if not src_ref_id:
                    continue
                connection.execute(
                    """
                    INSERT INTO paper_citations(
                        paper_id, citing_doi, citing_title, citing_year, citing_openalex_id,
                        source, source_ref_id, raw_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(paper_id, source, source_ref_id) DO UPDATE SET
                        citing_doi = excluded.citing_doi,
                        citing_title = excluded.citing_title,
                        citing_year = excluded.citing_year,
                        citing_openalex_id = excluded.citing_openalex_id,
                        raw_json = excluded.raw_json
                    """,
                    (
                        paper_id,
                        cit.get("citing_doi"),
                        cit.get("citing_title"),
                        cit.get("citing_year"),
                        cit.get("citing_openalex_id"),
                        source,
                        src_ref_id,
                        json.dumps(cit.get("raw") or {}, ensure_ascii=False),
                        now,
                    ),
                )
                written += 1
            return written

    def list_citations(self, paper_id: str, *, source: str | None = None) -> list[dict[str, Any]]:
        """Incoming citations (papers citing this paper)."""
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            if source:
                rows = connection.execute(
                    "SELECT * FROM paper_citations WHERE paper_id = ? AND source = ? ORDER BY created_at",
                    (paper_id, source),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM paper_citations WHERE paper_id = ? ORDER BY source, created_at",
                    (paper_id,),
                ).fetchall()
        return [dict(row) for row in rows]

    # ---- compile-level tracking (CompileScholar broker, 2026-09-20) ----

    COMPILE_LEVELS = ("none", "shallow", "deep")

    def get_compile_state(self, paper_id: str, *, kb_name: str = "default") -> dict[str, Any] | None:
        """Current compile level for a paper in a knowledge base. Absent row
        means the paper was never touched by the compiler."""
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            row = connection.execute(
                "SELECT * FROM paper_compile_states WHERE paper_id = ? AND kb_name = ?",
                (paper_id, kb_name),
            ).fetchone()
        return dict(row) if row else None

    def set_compile_state(
        self,
        *,
        paper_id: str,
        level: str,
        kb_name: str = "default",
        reason: str | None = None,
        detail: dict[str, Any] | None = None,
        artifact_ref: str | None = None,
        status: str = "ok",
    ) -> dict[str, Any]:
        """Upsert a paper's compile level and append the transition to
        paper_compile_log. Every change of level is logged with its reason
        (e.g. 'target:gap#123', 'hub_paper', 'manual') so the shallow->deep
        upgrade trail is never lost. Level must be one of COMPILE_LEVELS."""
        if level not in self.COMPILE_LEVELS:
            raise ValueError(f"level 必须是 {self.COMPILE_LEVELS} 之一，收到：{level}")
        now = utc_now()
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            row = connection.execute(
                "SELECT level FROM paper_compile_states WHERE paper_id = ? AND kb_name = ?",
                (paper_id, kb_name),
            ).fetchone()
            from_level = row["level"] if row else None
            connection.execute(
                """
                INSERT INTO paper_compile_states(
                    paper_id, kb_name, level, status, artifact_ref, notes_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id, kb_name) DO UPDATE SET
                    level = excluded.level,
                    status = excluded.status,
                    artifact_ref = COALESCE(excluded.artifact_ref, paper_compile_states.artifact_ref),
                    notes_json = COALESCE(excluded.notes_json, paper_compile_states.notes_json),
                    updated_at = excluded.updated_at
                """,
                (
                    paper_id,
                    kb_name,
                    level,
                    status,
                    artifact_ref,
                    json.dumps(detail, ensure_ascii=False) if detail else None,
                    now,
                    now,
                ),
            )
            if from_level != level:
                connection.execute(
                    """
                    INSERT INTO paper_compile_log(
                        paper_id, kb_name, from_level, to_level, reason, detail_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper_id,
                        kb_name,
                        from_level,
                        level,
                        reason,
                        json.dumps(detail, ensure_ascii=False) if detail else None,
                        now,
                    ),
                )
        return self.get_compile_state(paper_id, kb_name=kb_name) or {}

    def list_compile_log(self, paper_id: str, *, kb_name: str = "default", limit: int = 50) -> list[dict[str, Any]]:
        """Transition history for one paper (newest first)."""
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            rows = connection.execute(
                """
                SELECT * FROM paper_compile_log
                WHERE paper_id = ? AND kb_name = ?
                ORDER BY log_id DESC LIMIT ?
                """,
                (paper_id, kb_name, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_papers_by_compile_level(self, level: str, *, kb_name: str = "default", limit: int = 200) -> list[dict[str, Any]]:
        """All papers at a compile level in a KB (broker's 'what is already
        shallow' dedup check, and the 'who is due for upgrade' pool)."""
        if level not in self.COMPILE_LEVELS:
            raise ValueError(f"level 必须是 {self.COMPILE_LEVELS} 之一，收到：{level}")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.*, p.title, p.year, p.normalized_doi
                FROM paper_compile_states s
                JOIN papers p ON p.paper_id = s.paper_id
                WHERE s.level = ? AND s.kb_name = ?
                ORDER BY s.updated_at DESC LIMIT ?
                """,
                (level, kb_name, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def compile_state_summary(self, *, kb_name: str = "default") -> dict[str, Any]:
        """Counts per level (papers never touched are implicitly 'none')."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT level, COUNT(*) AS n FROM paper_compile_states
                WHERE kb_name = ? GROUP BY level
                """,
                (kb_name,),
            ).fetchall()
            total_papers = connection.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"]
        counts = {row["level"]: row["n"] for row in rows}
        return {
            "kb_name": kb_name,
            "counts": counts,
            "untouched": max(0, total_papers - sum(counts.values())),
        }

    def evidence_coverage(
        self,
        paper_id: str,
        *,
        source_type: str | None = None,
        modality: str | None = None,
        page_idx: int | None = None,
    ) -> dict[str, Any]:
        where, params = evidence_filters(
            paper_id=paper_id,
            source_type=source_type,
            modality=modality,
            page_idx=page_idx,
        )
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            total = connection.execute(
                f"SELECT COUNT(*) AS count FROM evidence_units WHERE {where}",
                params,
            ).fetchone()["count"]
            by_source = connection.execute(
                f"""
                SELECT source_type, COUNT(*) AS count
                FROM evidence_units
                WHERE {where}
                GROUP BY source_type
                ORDER BY source_type
                """,
                params,
            ).fetchall()
            by_modality = connection.execute(
                f"""
                SELECT modality, COUNT(*) AS count
                FROM evidence_units
                WHERE {where}
                GROUP BY modality
                ORDER BY modality
                """,
                params,
            ).fetchall()
            by_status = connection.execute(
                f"""
                SELECT status, COUNT(*) AS count
                FROM evidence_units
                WHERE {where}
                GROUP BY status
                ORDER BY status
                """,
                params,
            ).fetchall()
        return {
            "paper_id": paper_id,
            "total": total,
            "by_source_type": {row["source_type"]: row["count"] for row in by_source},
            "by_modality": {row["modality"]: row["count"] for row in by_modality},
            "by_status": {row["status"]: row["count"] for row in by_status},
        }

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            pdf = connection.execute(
                """
                SELECT asset_id AS artifact_id, paper_id, 'pdf' AS kind, local_path,
                       sha256 AS checksum, status, source_kind AS source
                FROM pdf_assets
                WHERE asset_id = ?
                """,
                (artifact_id,),
            ).fetchone()
            if pdf:
                return dict(pdf)
            artifact = connection.execute(
                """
                SELECT ma.*, pj.paper_id
                FROM mineru_artifacts ma
                JOIN processing_jobs pj ON pj.job_id = ma.job_id
                WHERE ma.artifact_id = ?
                """,
                (artifact_id,),
            ).fetchone()
            if not artifact:
                artifact = connection.execute(
                    "SELECT * FROM external_artifacts WHERE artifact_id = ?",
                    (artifact_id,),
                ).fetchone()
        return decode_artifact_row(artifact) if artifact else None

    def record_provenance_event(
        self,
        *,
        action: str,
        source: str,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        error_summary: str | None = None,
    ) -> dict[str, Any]:
        event_id = stable_id("EVT_", action, source, utc_now(), uuid.uuid4().hex)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO provenance_events(
                    event_id, action, source, inputs_json, outputs_json,
                    error_summary, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    action,
                    source,
                    json.dumps(inputs or {}, ensure_ascii=False, sort_keys=True),
                    json.dumps(outputs or {}, ensure_ascii=False, sort_keys=True),
                    error_summary,
                    utc_now(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM provenance_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return dict(row)

    def list_provenance_events(self, *, paper_ids: list[str] | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM provenance_events
                ORDER BY created_at, event_id
                """
            ).fetchall()
        events = [decode_provenance_event_row(row) for row in rows]
        if not paper_ids:
            return events
        paper_id_set = set(paper_ids)
        filtered: list[dict[str, Any]] = []
        for event in events:
            event_text = json.dumps(
                {"inputs": event["inputs"], "outputs": event["outputs"]},
                ensure_ascii=False,
                sort_keys=True,
            )
            if any(paper_id in event_text for paper_id in paper_id_set):
                filtered.append(event)
        return filtered

    def create_api_job(
        self,
        *,
        kind: str,
        status: str,
        paper_id: str | None = None,
        pdf_asset_id: str | None = None,
        current_stage: str | None = None,
        error: str | None = None,
        progress: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        job_id = stable_id("APIJOB_", kind, paper_id or "", pdf_asset_id or "", now)
        with self._connect() as connection:
            if paper_id:
                self._require_paper(connection, paper_id)
            if pdf_asset_id:
                self._require_pdf_asset(connection, pdf_asset_id)
            connection.execute(
                """
                INSERT INTO api_jobs(
                    job_id, kind, paper_id, pdf_asset_id, status,
                    current_stage, error, progress_json, result_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    kind,
                    paper_id,
                    pdf_asset_id,
                    status,
                    current_stage,
                    error,
                    json.dumps(progress or {}, ensure_ascii=False, sort_keys=True),
                    json.dumps(result or {}, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM api_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return dict(row)

    def update_api_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        paper_id: str | None = None,
        pdf_asset_id: str | None = None,
        current_stage: str | None = None,
        error: str | None = None,
        progress: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM api_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if not row:
                raise LibraryRegistryError(f"api job not found: {job_id}")
            connection.execute(
                """
                UPDATE api_jobs
                SET status = COALESCE(?, status),
                    paper_id = COALESCE(?, paper_id),
                    pdf_asset_id = COALESCE(?, pdf_asset_id),
                    current_stage = COALESCE(?, current_stage),
                    error = ?,
                    progress_json = COALESCE(?, progress_json),
                    result_json = COALESCE(?, result_json),
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    paper_id,
                    pdf_asset_id,
                    current_stage,
                    error,
                    json.dumps(progress, ensure_ascii=False, sort_keys=True) if progress is not None else None,
                    json.dumps(result, ensure_ascii=False, sort_keys=True) if result is not None else None,
                    now,
                    job_id,
                ),
            )
            updated = connection.execute("SELECT * FROM api_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(updated)

    def get_api_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM api_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_api_jobs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM api_jobs
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_extraction_runs(self, paper_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            rows = connection.execute(
                """
                SELECT * FROM extraction_runs
                WHERE paper_id = ?
                ORDER BY updated_at DESC, created_at DESC
                """,
                (paper_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_extraction_run_for_paper(self, paper_id: str) -> dict[str, Any] | None:
        runs = self.list_extraction_runs(paper_id)
        return runs[0] if runs else None

    def register_extraction_run(
        self,
        *,
        run_id: str,
        paper_id: str,
        status: str,
        input_artifact_ids: list[str],
        output_artifacts: dict[str, Path],
        llm_config_hash: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        relative_outputs = {
            key: str(self.relative_local_path(path)) for key, path in output_artifacts.items()
        }
        with self._connect() as connection:
            self._require_paper(connection, paper_id)
            connection.execute(
                """
                INSERT INTO extraction_runs(
                    run_id, paper_id, status, input_artifact_ids_json,
                    output_artifacts_json, llm_config_hash, error, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status = excluded.status,
                    input_artifact_ids_json = excluded.input_artifact_ids_json,
                    output_artifacts_json = excluded.output_artifacts_json,
                    llm_config_hash = excluded.llm_config_hash,
                    error = excluded.error,
                    updated_at = excluded.updated_at
                """,
                (
                    run_id,
                    paper_id,
                    status,
                    json.dumps(input_artifact_ids, ensure_ascii=False),
                    json.dumps(relative_outputs, ensure_ascii=False, sort_keys=True),
                    llm_config_hash,
                    error,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM extraction_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return dict(row)

    def get_extraction_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM extraction_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return dict(row) if row else None

    def resolve_local_path(self, local_path: str | Path) -> Path:
        path = Path(local_path)
        if path.is_absolute():
            return path
        return self.library_root / path

    def relative_local_path(self, path: Path) -> Path:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.library_root.resolve())
        except ValueError as exc:
            raise LibraryRegistryError(
                f"artifact path must be inside library root: {path}"
            ) from exc

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _insert_identifiers(
        self,
        connection: sqlite3.Connection,
        paper_id: str,
        identifiers: Iterable[dict[str, str]] | None,
    ) -> None:
        for identifier in identifiers or []:
            self._insert_identifier(
                connection,
                paper_id,
                identifier["scheme"],
                identifier["value"],
                identifier.get("source", "source_adapter"),
            )

    def _insert_identifier(
        self,
        connection: sqlite3.Connection,
        paper_id: str,
        scheme: str,
        value: str,
        source: str,
    ) -> None:
        identifier = normalize_identifier({"scheme": scheme, "value": value, "source": source})
        if not identifier:
            return
        connection.execute(
            """
            INSERT INTO paper_identifiers(
                identifier_id, paper_id, scheme, value, source, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(scheme, value) DO NOTHING
            """,
            (
                stable_id("PID_", identifier["scheme"], identifier["value"]),
                paper_id,
                identifier["scheme"],
                identifier["value"],
                identifier["source"],
                utc_now(),
            ),
        )

    def _find_paper_by_identifiers(
        self,
        connection: sqlite3.Connection,
        identifiers: Iterable[dict[str, str]],
    ) -> sqlite3.Row | None:
        for identifier in identifiers:
            row = connection.execute(
                """
                SELECT papers.*
                FROM paper_identifiers
                JOIN papers ON papers.paper_id = paper_identifiers.paper_id
                WHERE paper_identifiers.scheme = ?
                  AND paper_identifiers.value = ?
                """,
                (identifier["scheme"], identifier["value"]),
            ).fetchone()
            if row:
                return row
        return None

    def _find_paper_by_title_year(
        self,
        connection: sqlite3.Connection,
        *,
        title: str | None,
        year: int | None,
        identity_confidence: float | None,
        normalized_doi: str | None,
    ) -> sqlite3.Row | None:
        normalized_title = normalize_title(title)
        if not normalized_title or year is None or (identity_confidence or 0.0) < 0.92:
            return None
        rows = connection.execute(
            """
            SELECT *
            FROM papers
            WHERE COALESCE(published_year, year) = ?
            """,
            (year,),
        ).fetchall()
        for row in rows:
            existing_doi = normalize_doi(row["normalized_doi"])
            if existing_doi and normalized_doi and existing_doi != normalized_doi:
                continue
            if normalize_title(row["title"]) == normalized_title:
                return row
        return None

    def _require_paper(self, connection: sqlite3.Connection, paper_id: str) -> None:
        if not connection.execute(
            "SELECT 1 FROM papers WHERE paper_id = ?",
            (paper_id,),
        ).fetchone():
            raise LibraryRegistryError(f"paper not found: {paper_id}")

    def _require_pdf_asset(self, connection: sqlite3.Connection, asset_id: str) -> None:
        if not connection.execute(
            "SELECT 1 FROM pdf_assets WHERE asset_id = ?",
            (asset_id,),
        ).fetchone():
            raise LibraryRegistryError(f"pdf asset not found: {asset_id}")

    def _require_processing_job(self, connection: sqlite3.Connection, job_id: str) -> None:
        if not connection.execute(
            "SELECT 1 FROM processing_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone():
            raise LibraryRegistryError(f"processing job not found: {job_id}")

    def _ensure_pdf(self, path: Path) -> None:
        if not path.exists():
            raise LibraryRegistryError(f"PDF 文件不存在：{path}")
        if not path.is_file():
            raise LibraryRegistryError(f"PDF 路径不是文件：{path}")
        with path.open("rb") as handle:
            magic = handle.read(5)
        if magic != b"%PDF-":
            raise LibraryRegistryError(f"不是有效 PDF：{path}")

    def _decode_evidence_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key in ("locator_json", "provenance_json", "quality_json", "metadata_json"):
            output_key = key.removesuffix("_json")
            value = item.pop(key)
            item[output_key] = json.loads(value) if value else {}
        item["source"] = {
            "source_type": item["source_type"],
            "source_id": item["source_id"],
            "page_idx": item["page_idx"],
        }
        return item


class LibraryRegistryError(RuntimeError):
    pass


class CrossPaperDedupeConflict(LibraryRegistryError):
    def __init__(
        self,
        *,
        requested_paper_id: str,
        asset_owner_paper_id: str,
        pdf_asset_id: str,
        sha256: str,
    ) -> None:
        self.requested_paper_id = requested_paper_id
        self.asset_owner_paper_id = asset_owner_paper_id
        self.pdf_asset_id = pdf_asset_id
        self.sha256 = sha256
        super().__init__(
            "dedupe_cross_paper_conflict: PDF sha256 已属于另一个 paper，"
            "请使用受控 merge/rebind 后再继续。"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": "dedupe_cross_paper_conflict",
            "requested_paper_id": self.requested_paper_id,
            "asset_owner_paper_id": self.asset_owner_paper_id,
            "pdf_asset_id": self.pdf_asset_id,
            "sha256": self.sha256,
            "recommended_next_action": "merge_or_rebind_asset",
            "merge_endpoint": f"/api/library/papers/{self.requested_paper_id}/merge",
        }


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return prefix + digest[:12].upper()


def best_year_candidate(
    candidates: list[dict[str, Any]],
    paper: dict[str, Any],
) -> dict[str, Any] | None:
    usable = [
        candidate
        for candidate in candidates
        if candidate.get("year") is not None
        and candidate.get("status") in {"ready", "candidate_review_required"}
    ]
    if not usable:
        return None
    source_priority = {"openalex": 4, "sciverse": 3, "crossref": 2, "local_doi_archive": 1}

    def sort_key(candidate: dict[str, Any]) -> tuple[float, float, float]:
        confidence = year_confidence_for_candidate(candidate, paper)
        source_score = float(source_priority.get(str(candidate.get("source_name")), 0))
        candidate_score = float(candidate.get("candidate_score") or 0.0)
        return (confidence, source_score, candidate_score)

    return sorted(usable, key=sort_key, reverse=True)[0]


def year_confidence_for_candidate(candidate: dict[str, Any], paper: dict[str, Any]) -> float:
    candidate_year = candidate.get("year")
    if candidate_year is None:
        return 0.0
    paper_year = paper.get("published_year") or paper.get("year")
    if paper_year is not None:
        try:
            if int(candidate_year) != int(paper_year):
                return 0.45
        except (TypeError, ValueError):
            return 0.4
    score = float(candidate.get("candidate_score") or 0.0)
    doi_score = None
    match_scores = candidate.get("match_scores")
    if isinstance(match_scores, dict):
        doi_score = match_scores.get("doi")
    if doi_score == 1.0:
        return max(0.95, score)
    if candidate.get("normalized_doi") and paper.get("normalized_doi"):
        return max(0.9, score)
    if candidate.get("query_kind") == "title":
        return min(max(score, 0.4), 0.85)
    return max(score, 0.75)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str | None:
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            relative = child.relative_to(path).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            with child.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        return digest.hexdigest()
    return None


def path_manifest(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "type": "file",
            "file_count": 1,
            "size_bytes": path.stat().st_size,
            "files": [
                {
                    "relative_path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            ],
        }
    if path.is_dir():
        files = [
            {
                "relative_path": child.relative_to(path).as_posix(),
                "size_bytes": child.stat().st_size,
                "sha256": sha256_file(child),
            }
            for child in sorted(item for item in path.rglob("*") if item.is_file())
        ]
        return {"type": "directory", "file_count": len(files), "files": files}
    return {"type": "missing", "file_count": 0, "files": []}


def hash_json(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def decode_artifact_row(row: sqlite3.Row | None) -> dict[str, Any]:
    if row is None:
        return {}
    item = dict(row)
    manifest_json = item.pop("manifest_json", None)
    item["manifest"] = json.loads(manifest_json) if manifest_json else {}
    return item


def decode_remote_parsed_asset_row(row: sqlite3.Row | None) -> dict[str, Any]:
    if row is None:
        return {}
    item = dict(row)
    for key in ("locator_json", "provenance_json", "quality_json", "metadata_json"):
        output_key = key.removesuffix("_json")
        value = item.pop(key)
        item[output_key] = json.loads(value) if value else {}
    return item


def decode_provenance_event_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    for key in ("inputs_json", "outputs_json"):
        output_key = key.removesuffix("_json")
        value = item.pop(key)
        item[output_key] = json.loads(value) if value else {}
    return item


def decode_source_candidate_row(row: sqlite3.Row, paper: dict[str, Any] | None = None) -> dict[str, Any]:
    item = dict(row)
    pdf_candidates_json = item.pop("pdf_candidates_json", "[]")
    try:
        pdf_candidates = json.loads(pdf_candidates_json) if pdf_candidates_json else []
    except json.JSONDecodeError:
        pdf_candidates = []
    if not isinstance(pdf_candidates, list):
        pdf_candidates = []
    item["pdf_candidates"] = [candidate for candidate in pdf_candidates if isinstance(candidate, dict)]
    first_pdf = first_pdf_candidate(item["pdf_candidates"])
    item["has_pdf_url"] = first_pdf is not None
    item["pdf_url"] = str(first_pdf.get("url")) if first_pdf and first_pdf.get("url") else None
    item["pdf_url_source"] = pdf_url_source(first_pdf) if first_pdf else None
    item["pdf_source"] = item["pdf_url_source"]
    item["candidate_pdf_url"] = item["pdf_url"]
    item["asset_policy"] = source_candidate_asset_policy(item, first_pdf)
    item["export_policy"] = source_candidate_export_policy(item["asset_policy"])
    item["is_oa"] = source_candidate_is_oa(item, first_pdf)
    item["failure_audit"] = source_candidate_failure_audit(item, first_pdf)
    item["can_acquire_pdf_now"] = source_candidate_can_acquire_pdf_now(item)
    item["requires_confirmation"] = source_candidate_requires_confirmation(item, paper)
    item["can_process"] = source_candidate_can_process(item)
    item["can_process_now"] = item["can_process"]
    item["match_scores"] = source_candidate_match_scores(item, paper)
    item["risk_reasons"] = source_candidate_risk_reasons(item, paper)
    item["blocking_reason"] = source_candidate_blocking_reason(item)
    item["recommended_action"] = source_candidate_recommended_action(item)
    item["recommended_next_action"] = item["recommended_action"]
    return item


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


def source_candidate_is_oa(item: dict[str, Any], first_pdf: dict[str, Any] | None) -> bool:
    if first_pdf and first_pdf.get("is_oa") is not None:
        return bool(first_pdf.get("is_oa"))
    if item.get("open_access_status"):
        return str(item["open_access_status"]).lower() not in {"closed", "unknown"}
    return bool(item.get("license"))


def source_candidate_asset_policy(item: dict[str, Any], first_pdf: dict[str, Any] | None) -> str:
    if item.get("source_name") == "local_doi_archive":
        return "local_available_pdf" if item.get("status") == "ready" else "local_pdf_missing"
    if first_pdf and item.get("source_name") == "arxiv":
        return "arxiv_auto_allowed"
    if first_pdf and source_candidate_is_oa(item, first_pdf):
        return "oa_auto_allowed"
    if first_pdf:
        return "unknown_public_pdf"
    return "no_pdf_candidate"


def source_candidate_can_acquire_pdf_now(item: dict[str, Any]) -> bool:
    return item.get("asset_policy") in {
        "local_available_pdf",
        "oa_auto_allowed",
        "arxiv_auto_allowed",
    }


def source_candidate_export_policy(asset_policy: str) -> str:
    if asset_policy == "local_available_pdf":
        return "local_pdf_allowed"
    if asset_policy in {"oa_auto_allowed", "arxiv_auto_allowed"}:
        return "remote_pdf_allowed_after_commit"
    if asset_policy == "unknown_public_pdf":
        return "candidate_only_review_required"
    return "no_pdf_available"


def source_candidate_requires_confirmation(
    item: dict[str, Any],
    paper: dict[str, Any] | None,
) -> bool:
    if item.get("status") != "ready":
        return True
    if item.get("query_kind") == "title":
        return True
    identity_risks = {
        "doi_mismatch",
        "year_mismatch",
        "low_title_similarity",
        "missing_doi",
    }
    return any(
        risk in identity_risks
        for risk in source_candidate_risk_reasons(item, paper, include_process_risks=False)
    )


def source_candidate_can_process(item: dict[str, Any]) -> bool:
    return item.get("source_name") == "local_doi_archive" and item.get("status") == "ready"


def source_candidate_match_scores(
    item: dict[str, Any],
    paper: dict[str, Any] | None,
) -> dict[str, float | None]:
    candidate_doi = normalize_doi(item.get("normalized_doi"))
    paper_doi = normalize_doi(paper.get("normalized_doi") if paper else None)
    doi_score: float | None = None
    if candidate_doi and paper_doi:
        doi_score = 1.0 if candidate_doi == paper_doi else 0.0
    title_score = float(item["candidate_score"]) if item.get("query_kind") == "title" and item.get("candidate_score") is not None else None
    candidate_year = item.get("year")
    paper_year = (paper.get("published_year") or paper.get("year")) if paper else None
    year_score: float | None = None
    if candidate_year and paper_year:
        year_score = 1.0 if int(candidate_year) == int(paper_year) else 0.0
    return {
        "doi": doi_score,
        "title": title_score,
        "year": year_score,
        "overall": item.get("candidate_score"),
    }


def source_candidate_risk_reasons(
    item: dict[str, Any],
    paper: dict[str, Any] | None,
    *,
    include_process_risks: bool = True,
) -> list[str]:
    risks: list[str] = []
    if item.get("status") not in {"ready", "candidate_review_required"}:
        risks.append("source_status_not_ready")
    if item.get("query_kind") == "title" and (item.get("candidate_score") or 0) < 0.75:
        risks.append("low_title_similarity")
    match_scores = source_candidate_match_scores(item, paper)
    if match_scores["year"] == 0.0:
        risks.append("year_mismatch")
    if match_scores["doi"] == 0.0:
        risks.append("doi_mismatch")
    if not item.get("has_pdf_url") and item.get("source_name") != "local_doi_archive":
        risks.append("no_pdf_url")
    if (
        include_process_risks
        and item.get("has_pdf_url")
        and item.get("source_name") != "local_doi_archive"
        and item.get("asset_policy") not in {"oa_auto_allowed", "arxiv_auto_allowed"}
    ):
        risks.append("public_pdf_download_not_enabled")
    if not item.get("normalized_doi"):
        risks.append("missing_doi")
    http_status = (item.get("failure_audit") or {}).get("http_status")
    if http_status:
        risks.append(f"http_{http_status}")
    return risks


def source_candidate_failure_audit(
    item: dict[str, Any],
    first_pdf: dict[str, Any] | None,
) -> dict[str, Any]:
    http_status = parse_http_status(item.get("error_summary"))
    return {
        "source_candidate_id": item.get("candidate_id"),
        "source_name": item.get("source_name"),
        "status": item.get("status"),
        "http_status": http_status,
        "error_summary": item.get("error_summary"),
        "candidate_pdf_url": str(first_pdf.get("url")) if first_pdf and first_pdf.get("url") else None,
        "pdf_source": pdf_url_source(first_pdf) if first_pdf else None,
        "fallback_url": item.get("source_url"),
        "license": item.get("license") or (first_pdf or {}).get("license"),
        "open_access_status": item.get("open_access_status"),
        "is_oa": item.get("is_oa"),
    }


def parse_http_status(value: Any) -> int | None:
    if not value:
        return None
    match = re.search(r"\bHTTP(?:\s+Error)?\s+(\d{3})\b", str(value), flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def source_candidate_blocking_reason(item: dict[str, Any]) -> str | None:
    if item.get("status") != "ready":
        return "source_not_ready"
    if item.get("asset_policy") == "unknown_public_pdf":
        return "pdf_policy_requires_review"
    if not item.get("has_pdf_url") and item.get("source_name") != "local_doi_archive":
        return "missing_pdf"
    return None


def source_candidate_recommended_action(item: dict[str, Any]) -> str:
    if item.get("status") != "ready":
        return "review_source_error"
    if item.get("asset_policy") == "local_available_pdf":
        return "confirm_local_candidate"
    if item.get("asset_policy") in {"oa_auto_allowed", "arxiv_auto_allowed"}:
        return "register_then_pdf_acquire"
    if item.get("asset_policy") == "unknown_public_pdf":
        return "register_candidate_only"
    return "find_pdf_or_sciverse"


def manifest_item(
    status: str,
    *,
    count: int | None = None,
    artifact_id: str | None = None,
    job_id: str | None = None,
    next_action: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {"status": status}
    if count is not None:
        item["count"] = count
    if artifact_id:
        item["artifact_id"] = artifact_id
    if job_id:
        item["job_id"] = job_id
    if next_action and status != "ready":
        item["next_action"] = next_action
    return item


def artifact_manifest_item(
    artifact: dict[str, Any] | None,
    *,
    next_action: str | None = None,
) -> dict[str, Any]:
    if artifact is None:
        return manifest_item("missing", next_action=next_action)
    return {
        "status": artifact.get("status") or "ready",
        "artifact_id": artifact.get("artifact_id"),
        "kind": artifact.get("kind"),
        "local_path": artifact.get("local_path"),
        "checksum": artifact.get("checksum"),
    }


def remote_artifact_manifest_item(
    artifact: dict[str, Any] | None,
    *,
    next_action: str | None = None,
) -> dict[str, Any]:
    if artifact is None:
        return manifest_item("missing", next_action=next_action)
    return {
        "status": artifact.get("status") or "ready",
        "artifact_id": artifact.get("artifact_id"),
        "kind": artifact.get("kind"),
        "source": artifact.get("source"),
        "remote": True,
        "locator": artifact.get("locator", {}),
        "quality": artifact.get("quality", {}),
    }


def best_remote_parsed_asset(assets: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not assets:
        return None
    rank = {"recommended": 3, "try": 2, "local_mineru": 1}
    return max(
        assets,
        key=lambda asset: (
            rank.get(str(asset.get("quality", {}).get("quality_tier")), 0),
            str(asset.get("updated_at") or asset.get("created_at") or ""),
        ),
    )


def recovered_sciverse_ai_ready_asset(
    paper_id: str,
    evidence_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    ranked_rows: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    tier_rank = {"recommended": 3, "try": 2, "local_mineru": 1}
    readiness_rank = {"ready": 3, "partial": 2}
    for row in evidence_rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        ai_ready = metadata.get("sciverse_ai_ready")
        if not isinstance(ai_ready, dict):
            ai_ready = quality.get("sciverse_ai_ready")
        if not isinstance(ai_ready, dict):
            continue
        tier = str(ai_ready.get("quality_tier") or "")
        readiness = str(ai_ready.get("readiness_status") or "")
        if not readiness and tier == "recommended":
            readiness = "ready"
        elif not readiness and tier == "try":
            readiness = "partial"
        if readiness not in {"ready", "partial"} and tier not in {"recommended", "try"}:
            continue
        ranked_rows.append(
            (
                readiness_rank.get(readiness, 0),
                tier_rank.get(tier, 0),
                row,
                {**ai_ready, "readiness_status": readiness or ai_ready.get("readiness_status")},
            )
        )
    if not ranked_rows:
        return None
    _readiness_score, _tier_score, best_row, best_quality = max(
        ranked_rows,
        key=lambda item: (
            item[0],
            item[1],
            len(str(item[2].get("text") or "")),
            str(item[2].get("updated_at") or item[2].get("created_at") or ""),
        ),
    )
    recovered_rows = [item for item in ranked_rows if item[2].get("asset_id") == best_row.get("asset_id")]
    return {
        "asset_id": str(best_row.get("asset_id") or stable_id("SCIV_", paper_id, "sciverse_live_ingest")),
        "paper_id": paper_id,
        "source_name": "sciverse",
        "asset_kind": "sciverse_ai_ready",
        "source_record_id": best_quality.get("doc_id") or best_row.get("source", {}).get("source_id"),
        "status": "ready",
        "locator": best_row.get("locator") or {},
        "provenance": {
            **(best_row.get("provenance") if isinstance(best_row.get("provenance"), dict) else {}),
            "source_system": "sciverse",
            "source_artifact_kind": "sciverse_ai_ready",
            "recovered_from": "sciverse_chunk_evidence",
        },
        "quality": {
            **best_quality,
            "asset_kind": "sciverse_ai_ready",
            "next_action": best_quality.get("next_action") or "try_sciverse_ai_ready",
            "label": best_quality.get("label") or "可尝试 Sciverse AI-ready",
            "quality_issues": best_quality.get("quality_issues") or ["recovered_from_existing_evidence"],
        },
        "metadata": {
            "evidence_units": len(recovered_rows),
            "recovered": True,
            "recovered_from": "sciverse_chunk_evidence",
        },
        "created_at": best_row.get("created_at"),
        "updated_at": best_row.get("updated_at"),
    }


def sciverse_ai_ready_actions(
    *,
    paper_id: str,
    status: str,
    quality_tier: str,
    next_action: str,
    has_remote_asset: bool,
    has_pdf: bool,
) -> list[dict[str, Any]]:
    can_use = has_remote_asset and status == "ready" and quality_tier == "recommended"
    can_try = has_remote_asset and status in {"ready", "partial"}
    return [
        {
            "action": "use_sciverse_ai_ready",
            "label": "直接使用 Sciverse AI-ready",
            "enabled": can_use,
            "recommended": next_action == "use_sciverse_ai_ready",
            "href": f"/api/library/papers/{paper_id}/evidence?source_type=sciverse_chunk",
            "reason": None if can_use else "需要 recommended 且 ready 的 Sciverse AI-ready 资产",
        },
        {
            "action": "try_sciverse_ai_ready",
            "label": "尝试使用 Sciverse AI-ready",
            "enabled": can_try,
            "recommended": next_action == "try_sciverse_ai_ready",
            "href": f"/api/library/papers/{paper_id}/evidence?source_type=sciverse_chunk",
            "reason": None if can_try else "需要可用的 Sciverse AI-ready 远端解析资产",
        },
        {
            "action": "run_local_mineru",
            "label": "用本地 MinerU 重新处理",
            "enabled": has_pdf,
            "recommended": next_action == "run_local_mineru",
            "href": f"/api/library/papers/{paper_id}",
            "reason": None if has_pdf else "需要先上传或确认 PDF",
        },
        {
            "action": "discover_sciverse",
            "label": "检索 Sciverse AI-ready",
            "enabled": True,
            "recommended": next_action == "discover_sciverse",
            "href": f"/api/library/papers/{paper_id}/sciverse-ingest",
            "reason": None,
        },
    ]


def readiness_blocking_reason(
    *,
    metadata_ready: bool,
    pdf_ready: bool,
    local_mineru_ready: bool,
    evidence_ready: bool,
    sciverse_ai_ready: str,
    year_ready: bool,
    published_year: int | None,
    year_needs_review: bool,
    dedupe_diagnostic: dict[str, Any] | None = None,
) -> str | None:
    if not metadata_ready:
        return "missing_metadata"
    if not published_year:
        return "missing_year"
    if year_needs_review or not year_ready:
        return "year_needs_review"
    if evidence_ready:
        return None
    if local_mineru_ready or sciverse_ai_ready in {"ready", "partial"}:
        return "missing_evidence_index"
    if not pdf_ready and dedupe_diagnostic:
        return "dedupe_asset_on_other_paper"
    if not pdf_ready and sciverse_ai_ready == "missing":
        return "missing_pdf_or_remote_body"
    if pdf_ready and not local_mineru_ready:
        return "missing_local_mineru"
    return "not_evidence_ready"


def recommended_next_action_for_blocking_reason(
    blocking_reason: str | None,
    next_actions: list[str],
) -> str:
    if blocking_reason is None:
        return "ready_for_experiment"
    if blocking_reason in {"missing_year", "year_needs_review"}:
        return "review_year_metadata"
    if blocking_reason == "missing_metadata":
        return "add_metadata"
    if blocking_reason == "missing_evidence_index":
        return "build_evidence_units"
    if blocking_reason == "missing_pdf_or_remote_body":
        return "upload_pdf_or_discover_sciverse"
    if blocking_reason == "dedupe_asset_on_other_paper":
        return "merge_or_rebind_asset"
    if blocking_reason == "missing_local_mineru":
        return "process_with_mineru"
    return next_actions[0] if next_actions else "inspect_paper"


def evidence_filters(
    *,
    paper_id: str,
    source_type: str | None = None,
    modality: str | None = None,
    page_idx: int | None = None,
) -> tuple[str, tuple[Any, ...]]:
    clauses = ["paper_id = ?"]
    params: list[Any] = [paper_id]
    if source_type:
        clauses.append("source_type = ?")
        params.append(source_type)
    if modality:
        clauses.append("modality = ?")
        params.append(modality)
    if page_idx is not None:
        clauses.append("page_idx = ?")
        params.append(page_idx)
    return " AND ".join(clauses), tuple(params)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    normalized_doi TEXT UNIQUE,
    title TEXT,
    year INTEGER,
    published_year INTEGER,
    published_date TEXT,
    year_source TEXT,
    year_confidence REAL,
    year_needs_review INTEGER NOT NULL DEFAULT 1,
    venue TEXT,
    identity_confidence REAL,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_identifiers (
    identifier_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    scheme TEXT NOT NULL,
    value TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(scheme, value)
);

CREATE TABLE IF NOT EXISTS source_candidates (
    candidate_id TEXT PRIMARY KEY,
    paper_id TEXT REFERENCES papers(paper_id) ON DELETE SET NULL,
    source_name TEXT NOT NULL,
    query_kind TEXT NOT NULL,
    source_record_id TEXT,
    normalized_doi TEXT,
    title TEXT,
    year INTEGER,
    venue TEXT,
    candidate_score REAL,
    pdf_candidates_json TEXT NOT NULL DEFAULT '[]',
    license TEXT,
    open_access_status TEXT,
    source_url TEXT,
    status TEXT NOT NULL,
    raw_metadata_path TEXT,
    error_summary TEXT,
    retrieved_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pdf_assets (
    asset_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    sha256 TEXT NOT NULL UNIQUE,
    size_bytes INTEGER NOT NULL,
    source_kind TEXT NOT NULL,
    source_uri TEXT,
    license TEXT,
    open_access_status TEXT,
    local_path TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS remote_parsed_assets (
    asset_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    source_name TEXT NOT NULL,
    asset_kind TEXT NOT NULL,
    source_record_id TEXT,
    status TEXT NOT NULL,
    locator_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    quality_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_remote_parsed_assets_paper
ON remote_parsed_assets(paper_id, source_name, asset_kind, status);

CREATE TABLE IF NOT EXISTS processing_jobs (
    job_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    pdf_asset_id TEXT NOT NULL REFERENCES pdf_assets(asset_id) ON DELETE CASCADE,
    tool TEXT NOT NULL,
    tool_config_hash TEXT NOT NULL,
    tool_config_json TEXT NOT NULL,
    status TEXT NOT NULL,
    server_url_redacted TEXT,
    started_at TEXT,
    finished_at TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_processing_jobs_ready_reuse
ON processing_jobs(pdf_asset_id, tool, tool_config_hash)
WHERE status = 'ready';

CREATE TABLE IF NOT EXISTS mineru_artifacts (
    artifact_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES processing_jobs(job_id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    local_path TEXT NOT NULL,
    checksum TEXT,
    status TEXT NOT NULL,
    manifest_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_units (
    evidence_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    asset_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    modality TEXT NOT NULL,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    page_idx INTEGER,
    locator_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    quality_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(paper_id, asset_id, source_type, source_id)
);

CREATE INDEX IF NOT EXISTS idx_evidence_units_paper
ON evidence_units(paper_id, source_type, modality, status);

CREATE TABLE IF NOT EXISTS provenance_events (
    event_id TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    source TEXT NOT NULL,
    inputs_json TEXT NOT NULL,
    outputs_json TEXT NOT NULL,
    error_summary TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_jobs (
    job_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    paper_id TEXT REFERENCES papers(paper_id) ON DELETE SET NULL,
    pdf_asset_id TEXT REFERENCES pdf_assets(asset_id) ON DELETE SET NULL,
    status TEXT NOT NULL,
    current_stage TEXT,
    error TEXT,
    progress_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extraction_runs (
    run_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    input_artifact_ids_json TEXT NOT NULL,
    output_artifacts_json TEXT NOT NULL,
    llm_config_hash TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS upstream_asset_links (
    link_id TEXT PRIMARY KEY,
    paper_id TEXT REFERENCES papers(paper_id) ON DELETE SET NULL,
    asset_id TEXT REFERENCES pdf_assets(asset_id) ON DELETE SET NULL,
    upstream_source TEXT NOT NULL,
    upstream_project_id TEXT,
    upstream_file_id TEXT,
    minio_object TEXT,
    signed_url_redacted TEXT,
    materialized_local_path TEXT,
    checksum TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_tags (
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(paper_id, tag)
);

CREATE TABLE IF NOT EXISTS paper_collections (
    collection_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_collection_members (
    collection_id TEXT NOT NULL REFERENCES paper_collections(collection_id) ON DELETE CASCADE,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    PRIMARY KEY(collection_id, paper_id)
);

CREATE TABLE IF NOT EXISTS paper_references (
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    ref_doi TEXT,
    ref_title TEXT,
    ref_year INTEGER,
    ref_openalex_id TEXT,
    source TEXT NOT NULL,
    source_ref_id TEXT,
    raw_json TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY(paper_id, source, source_ref_id)
);

CREATE TABLE IF NOT EXISTS paper_citations (
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    citing_doi TEXT,
    citing_title TEXT,
    citing_year INTEGER,
    citing_openalex_id TEXT,
    source TEXT NOT NULL,
    source_ref_id TEXT,
    raw_json TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY(paper_id, source, source_ref_id)
);

CREATE TABLE IF NOT EXISTS paper_compile_states (
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    kb_name TEXT NOT NULL,
    level TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ok',
    artifact_ref TEXT,
    notes_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(paper_id, kb_name)
);

CREATE TABLE IF NOT EXISTS paper_compile_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    kb_name TEXT NOT NULL,
    from_level TEXT,
    to_level TEXT NOT NULL,
    reason TEXT,
    detail_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS external_artifacts (
    artifact_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    source TEXT NOT NULL,
    local_path TEXT NOT NULL,
    checksum TEXT,
    status TEXT NOT NULL DEFAULT 'ready',
    manifest_json TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(paper_id, kind, source)
);
"""


def ensure_schema_migrations(connection: sqlite3.Connection) -> None:
    paper_columns = {row["name"] for row in connection.execute("PRAGMA table_info(papers)")}
    if "archived" not in paper_columns:
        connection.execute("ALTER TABLE papers ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
    if "published_year" not in paper_columns:
        connection.execute("ALTER TABLE papers ADD COLUMN published_year INTEGER")
        connection.execute("UPDATE papers SET published_year = year WHERE published_year IS NULL AND year IS NOT NULL")
    if "published_date" not in paper_columns:
        connection.execute("ALTER TABLE papers ADD COLUMN published_date TEXT")
    if "year_source" not in paper_columns:
        connection.execute("ALTER TABLE papers ADD COLUMN year_source TEXT")
        connection.execute("UPDATE papers SET year_source = 'legacy_year' WHERE year IS NOT NULL AND year_source IS NULL")
    if "year_confidence" not in paper_columns:
        connection.execute("ALTER TABLE papers ADD COLUMN year_confidence REAL")
    if "year_needs_review" not in paper_columns:
        connection.execute("ALTER TABLE papers ADD COLUMN year_needs_review INTEGER NOT NULL DEFAULT 1")
        connection.execute("UPDATE papers SET year_needs_review = 0 WHERE published_year IS NOT NULL")
    source_candidate_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(source_candidates)")
    }
    if "year" not in source_candidate_columns:
        connection.execute("ALTER TABLE source_candidates ADD COLUMN year INTEGER")
    if "venue" not in source_candidate_columns:
        connection.execute("ALTER TABLE source_candidates ADD COLUMN venue TEXT")
    if "pdf_candidates_json" not in source_candidate_columns:
        connection.execute("ALTER TABLE source_candidates ADD COLUMN pdf_candidates_json TEXT NOT NULL DEFAULT '[]'")
    if "license" not in source_candidate_columns:
        connection.execute("ALTER TABLE source_candidates ADD COLUMN license TEXT")
    if "open_access_status" not in source_candidate_columns:
        connection.execute("ALTER TABLE source_candidates ADD COLUMN open_access_status TEXT")
    if "source_url" not in source_candidate_columns:
        connection.execute("ALTER TABLE source_candidates ADD COLUMN source_url TEXT")
    mineru_artifact_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(mineru_artifacts)")
    }
    if "manifest_json" not in mineru_artifact_columns:
        connection.execute(
            "ALTER TABLE mineru_artifacts ADD COLUMN manifest_json TEXT NOT NULL DEFAULT '{}'"
        )
    api_job_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(api_jobs)")
    }
    if "progress_json" not in api_job_columns:
        connection.execute("ALTER TABLE api_jobs ADD COLUMN progress_json TEXT NOT NULL DEFAULT '{}'")
    if "result_json" not in api_job_columns:
        connection.execute("ALTER TABLE api_jobs ADD COLUMN result_json TEXT NOT NULL DEFAULT '{}'")
    # compile-level tracking (2026-09-20, CompileScholar broker): new tables
    # only — CREATE IF NOT EXISTS is idempotent for fresh and old databases.
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_compile_states (
            paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
            kb_name TEXT NOT NULL,
            level TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ok',
            artifact_ref TEXT,
            notes_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(paper_id, kb_name)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_compile_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
            kb_name TEXT NOT NULL,
            from_level TEXT,
            to_level TEXT NOT NULL,
            reason TEXT,
            detail_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
