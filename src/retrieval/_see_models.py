from typing import Any, Literal

from pydantic import BaseModel, Field


class ContentBlock(BaseModel):
    block_id: str
    block_type: str
    text: str | None = None
    page_idx: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class KgNode(BaseModel):
    node_id: str
    node_kind: str | None = None
    canonical_name: str | None = None
    description: str | None = None
    source_refs: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class KgEdge(BaseModel):
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_kind: str | None = None
    description: str | None = None
    status: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class SourceRef(BaseModel):
    source_type: Literal[
        "content",
        "mineru_content_block",
        "mineru_markdown_chunk",
        "figure",
        "table",
        "kg_node",
        "kg_edge",
        "sciverse_chunk",
        "metadata_record",
        "citation_context",
    ]
    source_id: str
    page_idx: int | None = None


class EvidenceUnit(BaseModel):
    evidence_id: str
    paper_id: str | None = None
    asset_id: str | None = None
    source: SourceRef
    text: str
    modality: Literal["text", "image", "table", "figure", "kg", "metadata", "citation", "mixed"] = "text"
    status: Literal["candidate", "accepted", "rejected"] = "candidate"
    content_hash: str
    locator: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvolutionCase(BaseModel):
    case_id: str
    title: str
    summary: str
    evidence_ids: list[str]
    status: Literal["candidate", "final", "rejected"] = "candidate"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    query_id: str
    query: str
    matched_evidence_ids: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunSummary(BaseModel):
    run_id: str
    paper_id: str
    content_blocks: int
    kg_nodes: int
    kg_edges: int
    evidence_units: int = 0
    final_cases: int = 0
    failures: list[str] = Field(default_factory=list)
