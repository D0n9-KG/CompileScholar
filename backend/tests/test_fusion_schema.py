from app.fusion.schema import FUSION_SCHEMA_STATEMENTS


def test_fusion_schema_contains_required_constraints_and_indexes() -> None:
    joined = "\n".join(FUSION_SCHEMA_STATEMENTS)

    assert "fusion_community_id_unique" in joined
    assert "fusion_keyword_id_unique" in joined

    assert "FOR ()-[r:EXPLAINS]-() ON (r.score)" in joined
    assert "FOR ()-[r:IN_COMMUNITY]-() ON (r.weight)" in joined
    assert "FOR ()-[r:HAS_KEYWORD]-() ON (r.rank)" in joined
    assert "FOR ()-[r:SEMANTICALLY_RELATED]-() ON (r.similarity)" in joined
