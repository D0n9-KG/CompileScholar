"""Fusion graph domain primitives."""

from app.fusion.builder import build_fusion_projection
from app.fusion.community import detect_fusion_communities
from app.fusion.keywords import extract_fusion_keywords
from app.fusion.linking import generate_explains_links
from app.fusion.models import (
    FusionCommunityModel,
    FusionEdgeEvidence,
    FusionKeywordModel,
)
from app.fusion.schema import FUSION_SCHEMA_STATEMENTS, ensure_fusion_schema
from app.fusion.schema_evolution import (
    AcceptedSchemaPatch,
    SchemaCandidate,
    SchemaEvolutionEngine,
    SchemaPatchProposal,
)

__all__ = [
    "FUSION_SCHEMA_STATEMENTS",
    "FusionCommunityModel",
    "FusionEdgeEvidence",
    "FusionKeywordModel",
    "ensure_fusion_schema",
    "build_fusion_projection",
    "detect_fusion_communities",
    "extract_fusion_keywords",
    "generate_explains_links",
    "AcceptedSchemaPatch",
    "SchemaCandidate",
    "SchemaEvolutionEngine",
    "SchemaPatchProposal",
]
