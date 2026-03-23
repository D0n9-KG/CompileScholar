"""Pydantic v2 response models for active LLM output validation."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ResearchMoveMentionItem(BaseModel):
    model_config = ConfigDict(extra='allow')
    surface: str = ''
    normalized: str = ''
    type: str = ''
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ResearchMoveEffectItem(BaseModel):
    model_config = ConfigDict(extra='allow')
    direction: str = 'unknown'
    magnitude_text: str = ''
    comparator_surface: str = ''
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ResearchMoveWindowItem(BaseModel):
    model_config = ConfigDict(extra='allow')
    role: str = ''
    act_type: str = ''
    summary: str = ''
    anchor_chunk_ids: list[str] = Field(default_factory=list)
    research_objects: list[ResearchMoveMentionItem] = Field(default_factory=list)
    methods: list[ResearchMoveMentionItem] = Field(default_factory=list)
    observed_variables: list[ResearchMoveMentionItem] = Field(default_factory=list)
    metrics: list[ResearchMoveMentionItem] = Field(default_factory=list)
    comparators: list[ResearchMoveMentionItem] = Field(default_factory=list)
    conditions: list[ResearchMoveMentionItem] = Field(default_factory=list)
    limitation_types: list[ResearchMoveMentionItem] = Field(default_factory=list)
    resource_mentions: list[ResearchMoveMentionItem] = Field(default_factory=list)
    effects: list[ResearchMoveEffectItem] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ResearchMoveWindowResponse(BaseModel):
    """Response from PaperLogicTrace ResearchMove extraction on one semantic window."""

    model_config = ConfigDict(extra='allow')
    moves: list[ResearchMoveWindowItem] = Field(default_factory=list)


class CitationPurposeItem(BaseModel):
    model_config = ConfigDict(extra='allow')
    label: str = ''
    score: float = Field(default=0.5, ge=0.0, le=1.0)


class CitationPurposeResponse(BaseModel):
    """Response from single citation purpose classification."""

    model_config = ConfigDict(extra='allow')
    purposes: list[CitationPurposeItem] = Field(default_factory=list)


class BatchCitationItem(BaseModel):
    model_config = ConfigDict(extra='allow')
    ref_id: str = ''
    purposes: list[CitationPurposeItem] = Field(default_factory=list)


class BatchCitationPurposeResponse(BaseModel):
    """Response from batch citation purpose classification."""

    model_config = ConfigDict(extra='allow')
    citations: list[BatchCitationItem] = Field(default_factory=list)

