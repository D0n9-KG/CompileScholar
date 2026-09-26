from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from retrieval._see_models import EvidenceUnit, SourceRef


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class RetrievalCandidate:
    source_type: str
    source_id: str
    text: str
    page_idx: int | None = None
    score: float = 0.0


def build_content_candidates(content_blocks: Iterable[dict]) -> list[RetrievalCandidate]:
    candidates: list[RetrievalCandidate] = []
    for index, block in enumerate(content_blocks):
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        block_id = str(block.get("block_id") or f"content_{index:04d}")
        page_idx = block.get("page_idx")
        candidates.append(
            RetrievalCandidate(
                source_type="content",
                source_id=block_id,
                text=text,
                page_idx=page_idx if isinstance(page_idx, int) else None,
            )
        )
    return candidates


def build_kg_node_candidates(kg_nodes: Iterable[dict]) -> list[RetrievalCandidate]:
    candidates: list[RetrievalCandidate] = []
    for node in kg_nodes:
        node_id = node.get("node_id")
        if not isinstance(node_id, str):
            continue
        attributes = node.get("attributes") if isinstance(node.get("attributes"), dict) else {}
        text = " ".join(
            part
            for part in [
                str(node.get("canonical_name") or ""),
                str(attributes.get("description") or ""),
            ]
            if part.strip()
        ).strip()
        if text:
            candidates.append(
                RetrievalCandidate(
                    source_type="kg_node",
                    source_id=node_id,
                    text=text,
                )
            )
    return candidates


def build_kg_edge_candidates(kg_edges: Iterable[dict]) -> list[RetrievalCandidate]:
    candidates: list[RetrievalCandidate] = []
    for edge in kg_edges:
        edge_id = edge.get("edge_id")
        if not isinstance(edge_id, str):
            continue
        attributes = edge.get("attributes") if isinstance(edge.get("attributes"), dict) else {}
        text = " ".join(
            part
            for part in [
                str(edge.get("edge_kind") or ""),
                str(attributes.get("description") or ""),
            ]
            if part.strip()
        ).strip()
        if text:
            candidates.append(
                RetrievalCandidate(
                    source_type="kg_edge",
                    source_id=edge_id,
                    text=text,
                )
            )
    return candidates


def lexical_retrieve(
    query: str,
    candidates: Iterable[RetrievalCandidate],
    *,
    explicit_content_refs: set[str] | None = None,
    limit: int = 8,
) -> list[RetrievalCandidate]:
    query_terms = _tokenize(query)
    pinned = explicit_content_refs or set()
    scored: list[RetrievalCandidate] = []
    for candidate in candidates:
        terms = _tokenize(candidate.text)
        overlap = len(query_terms & terms)
        score = float(overlap)
        if candidate.source_type == "content" and candidate.source_id in pinned:
            score += 10_000
        if score > 0:
            scored.append(_replace_score(candidate, score))

    scored.sort(key=lambda item: (-item.score, item.source_type, item.source_id))
    return scored[:limit]


def candidates_to_evidence(
    candidates: Iterable[RetrievalCandidate],
    *,
    prefix: str = "ev",
) -> list[EvidenceUnit]:
    evidence: list[EvidenceUnit] = []
    for index, candidate in enumerate(candidates, start=1):
        content_hash = hashlib.sha256(candidate.text.encode("utf-8")).hexdigest()
        evidence.append(
            EvidenceUnit(
                evidence_id=f"{prefix}_{index:04d}",
                source=SourceRef(
                    source_type=candidate.source_type,  # type: ignore[arg-type]
                    source_id=candidate.source_id,
                    page_idx=candidate.page_idx,
                ),
                text=candidate.text,
                content_hash=content_hash,
                metadata={"score": candidate.score},
            )
        )
    return evidence


def _tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_RE.finditer(text)}


def _replace_score(candidate: RetrievalCandidate, score: float) -> RetrievalCandidate:
    return RetrievalCandidate(
        source_type=candidate.source_type,
        source_id=candidate.source_id,
        text=candidate.text,
        page_idx=candidate.page_idx,
        score=score,
    )
