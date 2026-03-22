from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Callable

from app.crossref.client import CrossrefClient
from app.graph.neo4j_client import paper_id_for_md_path
from app.ingest.models import DocumentIR

DOI_STRATEGIES = {"extract_only", "title_crossref"}
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
_CROSSREF_CONFIDENCE_THRESHOLD = 0.25


def normalize_doi_strategy(value: str | None) -> str:
    strategy = str(value or "").strip().lower()
    if strategy in DOI_STRATEGIES:
        return strategy
    return "title_crossref"


def _normalize_doi(value: str | None) -> str | None:
    doi = str(value or "").strip().lower()
    if not doi or not _DOI_RE.match(doi):
        return None
    return doi


@dataclass(frozen=True)
class PaperIdentity:
    doi: str | None
    paper_id: str
    doi_source: str
    existing_paper_id: str | None = None
    existing_ingested: bool = False


def resolve_document_identity(
    doc: DocumentIR,
    *,
    doi_override: str | None = None,
    doi_strategy: str | None = None,
    crossref: CrossrefClient | None = None,
    existing_lookup: Callable[[str], dict[str, Any]] | None = None,
) -> PaperIdentity:
    normalized_override = _normalize_doi(doi_override)
    normalized_doc_doi = _normalize_doi(doc.paper.doi)
    resolved_doi = normalized_override or normalized_doc_doi
    doi_source = "override" if normalized_override else ("document" if normalized_doc_doi else "none")

    strategy = normalize_doi_strategy(doi_strategy)
    if not resolved_doi and strategy == "title_crossref":
        query = str(doc.paper.title or doc.paper.title_alt or "").strip()
        if query:
            client = crossref or CrossrefClient()
            try:
                result = client.resolve_reference(query)
            except Exception:
                result = None
            selected = result.selected if result else None
            confidence = float(result.confidence) if result else 0.0
            selected_doi = _normalize_doi(selected.doi if selected else None)
            if selected_doi and confidence >= _CROSSREF_CONFIDENCE_THRESHOLD:
                resolved_doi = selected_doi
                doi_source = "title_crossref"

    paper_id = paper_id_for_md_path(doc.paper.md_path, doi=resolved_doi)
    existing_paper_id = None
    existing_ingested = False
    if resolved_doi and existing_lookup is not None:
        candidate_paper_id = f"doi:{resolved_doi}"
        try:
            existing = existing_lookup(candidate_paper_id)
        except KeyError:
            existing = None
        if isinstance(existing, dict):
            existing_paper_id = candidate_paper_id
            existing_ingested = bool(existing.get("ingested"))

    return PaperIdentity(
        doi=resolved_doi,
        paper_id=paper_id,
        doi_source=doi_source,
        existing_paper_id=existing_paper_id,
        existing_ingested=existing_ingested,
    )


def apply_identity_to_document(doc: DocumentIR, identity: PaperIdentity) -> DocumentIR:
    if _normalize_doi(doc.paper.doi) == identity.doi:
        return doc
    return replace(doc, paper=replace(doc.paper, doi=identity.doi))


def dedupe_documents_by_identity(
    docs: list[DocumentIR],
    identities: list[PaperIdentity],
) -> tuple[list[DocumentIR], list[PaperIdentity], list[dict[str, str]]]:
    unique_docs: list[DocumentIR] = []
    unique_identities: list[PaperIdentity] = []
    duplicates: list[dict[str, str]] = []
    seen_paper_ids: dict[str, str] = {}

    for doc, identity in zip(docs, identities):
        first_source = seen_paper_ids.get(identity.paper_id)
        if first_source is not None:
            duplicates.append(
                {
                    "paper_source": str(doc.paper.paper_source or ""),
                    "paper_id": identity.paper_id,
                    "duplicate_of": first_source,
                    "doi": str(identity.doi or ""),
                }
            )
            continue
        seen_paper_ids[identity.paper_id] = str(doc.paper.paper_source or "")
        unique_docs.append(doc)
        unique_identities.append(identity)

    return unique_docs, unique_identities, duplicates
