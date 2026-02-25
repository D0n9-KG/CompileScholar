from __future__ import annotations

from collections.abc import Callable


FUSION_SCHEMA_STATEMENTS: tuple[str, ...] = (
    "CREATE CONSTRAINT fusion_community_id_unique IF NOT EXISTS FOR (fc:FusionCommunity) REQUIRE fc.community_id IS UNIQUE",
    "CREATE CONSTRAINT fusion_keyword_id_unique IF NOT EXISTS FOR (fk:FusionKeyword) REQUIRE fk.keyword_id IS UNIQUE",
    "CREATE INDEX fusion_community_title IF NOT EXISTS FOR (fc:FusionCommunity) ON (fc.title)",
    "CREATE INDEX fusion_keyword_text IF NOT EXISTS FOR (fk:FusionKeyword) ON (fk.keyword)",
    "CREATE INDEX fusion_explains_score IF NOT EXISTS FOR ()-[r:EXPLAINS]-() ON (r.score)",
    "CREATE INDEX fusion_in_community_weight IF NOT EXISTS FOR ()-[r:IN_COMMUNITY]-() ON (r.weight)",
    "CREATE INDEX fusion_has_keyword_rank IF NOT EXISTS FOR ()-[r:HAS_KEYWORD]-() ON (r.rank)",
    "CREATE INDEX fusion_semantically_related_similarity IF NOT EXISTS FOR ()-[r:SEMANTICALLY_RELATED]-() ON (r.similarity)",
)


def ensure_fusion_schema(run_statement: Callable[[str], None]) -> None:
    for stmt in FUSION_SCHEMA_STATEMENTS:
        run_statement(stmt)
