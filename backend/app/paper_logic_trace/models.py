from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


PaperType = Literal['empirical', 'theoretical', 'review', 'software', 'benchmark', 'case_study', 'unknown']
AnchorModality = Literal['text', 'figure', 'table', 'citation']
SupportType = Literal['direct', 'contextual', 'indirect']
ExtractionMode = Literal['direct', 'normalized', 'inferred']
SupportStrength = Literal['exact', 'strong', 'weak']
MoveRole = Literal['problem', 'background', 'hypothesis', 'method', 'experiment', 'result', 'interpretation', 'limitation', 'future_work']
MoveActType = Literal[
    'identify_gap',
    'define_task',
    'formulate_hypothesis',
    'propose_method',
    'adapt_method',
    'build_resource',
    'set_condition',
    'run_experiment',
    'measure_outcome',
    'compare_baseline',
    'report_effect',
    'explain_mechanism',
    'diagnose_failure',
    'state_limitation',
    'suggest_extension',
]
EffectDirection = Literal['increase', 'decrease', 'improve', 'worsen', 'mixed', 'none', 'unknown']
AuditState = Literal['hot_path', 'audited']
MoveRelationType = Literal['motivates', 'addresses', 'implements', 'evaluates', 'yields', 'explains', 'limits', 'extends']


class PaperMetadata(BaseModel):
    paper_id: str
    canonical_doi: str | None = None
    title: str
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    paper_type: PaperType = 'unknown'
    source_refs: list[str] = Field(default_factory=list)


class EvidenceAnchor(BaseModel):
    anchor_id: str
    paper_id: str
    source_ref: str
    modality: AnchorModality
    section_path: list[str] = Field(default_factory=list)
    locator: dict[str, Any] = Field(default_factory=dict)
    quote: str
    citation_ids: list[str] = Field(default_factory=list)
    support_type: SupportType
    weak: bool = False


class MentionValue(BaseModel):
    surface: str
    normalized: str | None = None
    type: str | None = None
    anchor_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None
    inferred: bool = False


class EffectValue(BaseModel):
    direction: EffectDirection
    magnitude_text: str | None = None
    magnitude_numeric: float | None = None
    unit: str | None = None
    comparator_surface: str | None = None
    anchor_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None


class SlotProvenance(BaseModel):
    field: str
    value_index: int | None = None
    anchor_ids: list[str] = Field(default_factory=list)
    extraction_mode: ExtractionMode
    support_strength: SupportStrength
    confidence: float | None = None
    notes: str | None = None


class ResearchMove(BaseModel):
    move_id: str
    sequence_no: int
    role: MoveRole
    act_type: MoveActType
    summary: str
    research_objects: list[MentionValue] = Field(default_factory=list)
    methods: list[MentionValue] = Field(default_factory=list)
    observed_variables: list[MentionValue] = Field(default_factory=list)
    metrics: list[MentionValue] = Field(default_factory=list)
    comparators: list[MentionValue] = Field(default_factory=list)
    conditions: list[MentionValue] = Field(default_factory=list)
    effects: list[EffectValue] = Field(default_factory=list)
    limitation_types: list[MentionValue] = Field(default_factory=list)
    resource_mentions: list[MentionValue] = Field(default_factory=list)
    anchor_ids: list[str] = Field(default_factory=list)
    slot_provenance: list[SlotProvenance] = Field(default_factory=list)
    confidence: float | None = None
    audit_state: AuditState = 'hot_path'


class MoveRelation(BaseModel):
    relation_id: str
    source_move_id: str
    target_move_id: str
    relation_type: MoveRelationType
    anchor_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None


class CitationAct(BaseModel):
    citation_act_id: str
    source_move_id: str | None = None
    target_paper_id: str | None = None
    purpose: str | None = None
    polarity: str | None = None
    semantic_signal: str | None = None
    target_scope: str | None = None
    anchor_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None


class FigureRef(BaseModel):
    figure_id: str
    caption: str | None = None
    anchor_ids: list[str] = Field(default_factory=list)


class TableRef(BaseModel):
    table_id: str
    caption: str | None = None
    anchor_ids: list[str] = Field(default_factory=list)


class CanonicalCore(BaseModel):
    evidence_anchors: list[EvidenceAnchor] = Field(default_factory=list)
    moves: list[ResearchMove] = Field(default_factory=list)
    move_relations: list[MoveRelation] = Field(default_factory=list)
    citation_acts: list[CitationAct] = Field(default_factory=list)
    figure_refs: list[FigureRef] = Field(default_factory=list)
    table_refs: list[TableRef] = Field(default_factory=list)


class PaperLogicTrace(BaseModel):
    trace_id: str
    schema_version: str = 'v2'
    built_at: str
    paper_metadata: PaperMetadata
    canonical_core: CanonicalCore
    derived_views: dict[str, Any] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)
