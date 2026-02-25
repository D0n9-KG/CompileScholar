from pathlib import Path

from app.fusion.schema_evolution import (
    SchemaCandidate,
    SchemaEvolutionEngine,
)


def test_schema_evolution_clusters_near_duplicates(tmp_path: Path) -> None:
    engine = SchemaEvolutionEngine(storage_dir=tmp_path)
    proposals = engine.build_patch_proposals(
        [
            SchemaCandidate(
                candidate_type="entity",
                raw_label="Laplace Transform",
                confidence=0.91,
                evidence_chunk_id="ch:001",
                evidence_quote="Laplace Transform is defined for causal systems.",
            ),
            SchemaCandidate(
                candidate_type="entity",
                raw_label="laplace-transform",
                confidence=0.84,
                evidence_chunk_id="ch:002",
                evidence_quote="The laplace transform maps time domain to s-domain.",
            ),
        ]
    )

    assert len(proposals) == 1
    p = proposals[0]
    assert p.normalized_label == "laplace transform"
    assert p.evidence_count == 2
    assert set(p.evidence_chunk_ids) == {"ch:001", "ch:002"}


def test_schema_evolution_threshold_routing(tmp_path: Path) -> None:
    engine = SchemaEvolutionEngine(storage_dir=tmp_path, t_high=0.85, t_mid=0.60)
    proposals = engine.build_patch_proposals(
        [
            SchemaCandidate("entity", "HighType", 0.92, "c1", "high"),
            SchemaCandidate("entity", "MidType", 0.70, "c2", "mid"),
            SchemaCandidate("entity", "LowType", 0.45, "c3", "low"),
        ]
    )
    by_label = {p.normalized_label: p for p in proposals}

    assert engine.route_proposal(by_label["hightype"]) == "auto_accept"
    assert engine.route_proposal(by_label["midtype"]) == "pending_review"
    assert engine.route_proposal(by_label["lowtype"]) == "rejected"


def test_schema_evolution_acceptance_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    engine = SchemaEvolutionEngine(storage_dir=tmp_path, t_high=0.80, t_mid=0.60)
    proposal = engine.build_patch_proposals(
        [SchemaCandidate("relation", "DerivesFrom", 0.93, "ch:009", "A derives from B")]
    )[0]

    v1 = engine.accept_patch(proposal)
    v2 = engine.accept_patch(proposal)

    assert v1.version_id == v2.version_id
    assert (tmp_path / "versions" / f"{v1.version_id}.json").is_file()


def test_schema_evolution_replay_scope_only_includes_affected_chunks(tmp_path: Path) -> None:
    engine = SchemaEvolutionEngine(storage_dir=tmp_path)
    proposal = engine.build_patch_proposals(
        [
            SchemaCandidate("entity", "Helmholtz", 0.89, "ch:010", "Helmholtz equation"),
            SchemaCandidate("entity", "helmholtz ", 0.82, "ch:011", "Helmholtz equation in acoustics"),
        ]
    )[0]

    replay_chunks = engine.compute_incremental_replay_chunks(proposal)
    assert replay_chunks == ["ch:010", "ch:011"]
