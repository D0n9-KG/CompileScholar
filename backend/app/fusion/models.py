from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class FusionEdgeEvidence:
    source_chunk_id: str
    evidence_quote: str
    confidence: float
    reason: str | None = None


@dataclass(slots=True)
class FusionKeywordModel:
    keyword_id: str
    keyword: str
    rank: int = 0
    weight: float = 0.0


@dataclass(slots=True)
class FusionCommunityModel:
    community_id: str
    title: str
    confidence: float = 0.0
    evidence: list[FusionEdgeEvidence] = field(default_factory=list)
