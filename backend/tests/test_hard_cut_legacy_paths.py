from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

import app.extraction as extraction_module
from app.extraction import orchestrator as extraction_orchestrator
from app.graph.neo4j_client import Neo4jClient
from app.llm import schemas as llm_schemas
from app.similarity import service as similarity_service


def test_legacy_paper_edits_router_module_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('app.api.routers.paper_edits')


def test_legacy_global_community_projection_module_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('app.community.projection')


def test_neo4j_client_no_longer_exposes_legacy_logic_claim_write_helpers() -> None:
    assert not hasattr(Neo4jClient, 'upsert_logic_steps_and_claims')
    assert not hasattr(Neo4jClient, 'list_logic_steps_for_fusion')
    assert not hasattr(Neo4jClient, 'list_claims_for_fusion')


def test_neo4j_schema_no_longer_mentions_legacy_logic_claim_nodes() -> None:
    source = inspect.getsource(Neo4jClient.ensure_schema)
    assert 'LogicStep' not in source
    assert 'Claim' not in source
    assert 'logic_step_id_unique' not in source
    assert 'claim_id_unique' not in source


def test_similarity_service_no_longer_exposes_legacy_claim_logic_entrypoints() -> None:
    source = inspect.getsource(similarity_service)
    assert 'list_claim_similarity_rows' not in source
    assert 'list_logic_step_similarity_rows' not in source
    assert 'replace_similar_claim_edges_batch' not in source
    assert 'replace_similar_logic_edges_batch' not in source
    assert 'SIMILAR_CLAIM' not in source
    assert 'SIMILAR_LOGIC' not in source


def test_extraction_package_exports_only_trace_entrypoint() -> None:
    assert getattr(extraction_module, '__all__', []) == ['run_phase1_paper_logic_trace']
    assert not hasattr(extraction_module, 'run_phase1_extraction')


def test_extraction_orchestrator_no_longer_contains_legacy_phase1_claim_pipeline() -> None:
    source = inspect.getsource(extraction_orchestrator)
    assert 'def run_phase1_extraction' not in source
    assert 'claims_merged' not in source
    assert 'logic_steps.json' not in source
    assert 'claim_candidates.json' not in source
    assert 'extract_logic_and_claims_v2' not in source
    assert 'ChunkClaimsResponse' not in source
    assert 'ChunkClaimsBatchResponse' not in source


def test_legacy_claim_modules_are_removed() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / 'app' / 'llm' / 'logic_claims_v2.py').exists()
    assert not (repo_root / 'app' / 'extraction' / 'noise_filters.py').exists()
    assert not (repo_root / 'app' / 'paper_logic_trace' / 'normalization.py').exists()


def test_llm_schemas_no_longer_define_legacy_logic_claim_models() -> None:
    source = inspect.getsource(llm_schemas)
    assert 'LogicStepItem' not in source
    assert 'LogicClaimItem' not in source
    assert 'LogicClaimsResponse' not in source
    assert 'ChunkClaimItem' not in source
    assert 'ChunkClaimsResponse' not in source
    assert 'ChunkClaimsBatchResponse' not in source


def test_schema_store_source_has_no_legacy_step_or_claim_configuration() -> None:
    import app.schema_store as schema_store

    source = inspect.getsource(schema_store)
    assert 'claim_kinds' not in source
    assert 'phase1_claim_' not in source
    assert 'phase2_gate_logic_steps' not in source


def test_neo4j_client_no_longer_exposes_legacy_claim_logic_read_write_helpers() -> None:
    for name in (
        'get_paper_detail',
        'get_paper_logic_trace_inputs',
        'set_logic_step_evidence',
        'list_paper_ids_for_claims',
        'upsert_human_only_claim_node',
        'set_claim_evidence',
        'apply_human_claim_evidence_overrides',
        'list_claim_structured_rows',
        'list_gap_like_claims',
    ):
        assert not hasattr(Neo4jClient, name)
