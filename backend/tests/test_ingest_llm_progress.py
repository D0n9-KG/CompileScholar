from __future__ import annotations

import time
from pathlib import Path

from app.graph.neo4j_client import paper_id_for_md_path
from app.ingest import pipeline
from app.ingest.models import DocumentIR, PaperDraft
from app.paper_logic_trace.models import CanonicalCore, EvidenceAnchor, PaperLogicTrace, PaperMetadata, ResearchMove


class _FakeNeo4jClient:
    def __init__(self):
        self.trace_writes: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False

    def ensure_schema(self) -> None:
        pass

    def get_paper_basic(self, paper_id: str) -> dict:
        raise KeyError(paper_id)

    def delete_paper_subgraph(self, paper_id: str) -> None:  # noqa: ARG002
        raise AssertionError("delete_paper_subgraph should not be called for new papers")

    def upsert_paper_and_chunks(self, doc: DocumentIR) -> None:  # noqa: ARG002
        pass

    def update_paper_props(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, ARG002
        pass

    def upsert_figures(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, ARG002
        pass

    def upsert_references_and_citations(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, ARG002
        pass

    def upsert_paper_logic_trace(self, paper_id: str, trace_payload: dict) -> None:
        self.trace_writes.append((paper_id, trace_payload))

    def update_cites_purposes(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, ARG002
        pass

    def backfill_missing_citation_purposes(self, *args, **kwargs) -> int:  # noqa: ANN002, ANN003, ARG002
        return 0

    def list_global_community_rows(self, *args, **kwargs) -> list[dict]:  # noqa: ANN002, ANN003, ARG002
        return []

    def list_global_community_members(self, *args, **kwargs) -> list[dict]:  # noqa: ANN002, ANN003, ARG002
        return []

    def list_research_moves(self, *args, **kwargs) -> list[dict]:  # noqa: ANN002, ANN003, ARG002
        return []

    def list_evidence_anchors(self, *args, **kwargs) -> list[dict]:  # noqa: ANN002, ANN003, ARG002
        return []


def _mock_document(index: int) -> DocumentIR:
    return DocumentIR(
        paper=PaperDraft(
            paper_source=f"test_paper_{index}",
            md_path=f"C:/tmp/test_paper_{index}.md",
            title=f"Test Paper {index}",
            title_alt=None,
            authors=[],
            doi=f"10.1000/TEST{index}",
            year=2024,
        ),
        chunks=[],
        references=[],
        citations=[],
    )


def _mock_paper_logic_trace(paper_id: str, *, quality_tier: str = 'yellow', audit_status: str = 'eligible') -> PaperLogicTrace:
    trace = PaperLogicTrace(
        trace_id=f'{paper_id}:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-22T12:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id=paper_id,
            title='Demo',
            source_refs=['chunk:1'],
        ),
        canonical_core=CanonicalCore(
            evidence_anchors=[
                EvidenceAnchor(
                    anchor_id='a-1',
                    paper_id=paper_id,
                    source_ref='chunk:1',
                    modality='text',
                    section_path=['Method'],
                    locator={'chunk_id': 'chunk:1'},
                    quote='We propose a graph encoder.',
                    citation_ids=[],
                    support_type='direct',
                    weak=False,
                )
            ],
            moves=[
                ResearchMove(
                    move_id='m-1',
                    sequence_no=1,
                    role='method',
                    act_type='propose_method',
                    summary='We propose a graph encoder.',
                    anchor_ids=['a-1'],
                    confidence=0.9,
                )
            ],
            move_relations=[],
            citation_acts=[],
            figure_refs=[],
            table_refs=[],
        ),
        derived_views={'community_signatures': [{'move_id': 'm-1'}]},
        quality={
            'quality_tier': quality_tier,
            'hot_path_gate_report': {'passed': True, 'move_count': 1, 'anchor_count': 1, 'sparse_trace': quality_tier != 'green'},
            'audit_status': audit_status,
        },
    )
    return trace


def _fake_phase1_trace_output(paper_id: str, *, quality_tier: str = 'yellow', audit_status: str = 'eligible') -> dict:
    return {
        'quality_report': {
            'gate_passed': True,
            'quality_tier': 'green',
            'quality_tier_score': 0.92,
        },
        'paper_logic_trace': _mock_paper_logic_trace(
            paper_id,
            quality_tier=quality_tier,
            audit_status=audit_status,
        ),
    }


def test_ingest_llm_progress_reports_active_and_queued_counts(monkeypatch):  # noqa: ANN001, ANN201
    docs = [_mock_document(index) for index in range(3)]
    docs_iter = iter(docs)
    progress_messages: list[str] = []
    fake_client = _FakeNeo4jClient()

    monkeypatch.setattr(pipeline, "Neo4jClient", lambda *args, **kwargs: fake_client)  # noqa: ARG005
    monkeypatch.setattr(pipeline, "parse_mineru_markdown", lambda _md: next(docs_iter))
    monkeypatch.setattr(
        pipeline,
        "recover_references_with_agent",
        lambda doc, **kwargs: (doc, {"before_refs": 0, "after_refs": 0}),  # noqa: ARG005
    )
    monkeypatch.setattr(
        pipeline,
        "recover_citation_events_from_references",
        lambda doc, **kwargs: (doc, {"before_events": 0, "after_events": 0}),  # noqa: ARG005
    )
    monkeypatch.setattr(pipeline, "CrossrefClient", lambda: object())
    monkeypatch.setattr(
        pipeline,
        "build_reference_and_cite_records",
        lambda doc, **kwargs: {  # noqa: ARG005
            "paper_id": paper_id_for_md_path(doc.paper.md_path, doi=doc.paper.doi),
            "refs": [],
            "cited_papers": [],
            "cites_resolved": [],
            "cites_unresolved": [],
        },
    )
    monkeypatch.setattr(pipeline, "load_canonical_meta", lambda _path: {"paper_type": "research"})
    monkeypatch.setattr(
        pipeline,
        "load_active",
        lambda _paper_type: {"version": 1, "paper_type": "research", "rules": {}, "prompts": {}},
    )
    monkeypatch.setattr(pipeline, "extract_figures_from_markdown", lambda **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(pipeline, "build_faiss_for_chunks", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(pipeline, "build_faiss_for_rows", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(pipeline, "build_community_corpus_rows", lambda *args, **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(pipeline, "_write_document_ir", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(Path, "write_text", lambda self, *args, **kwargs: 0)  # noqa: ARG005
    monkeypatch.setattr(
        pipeline,
        "merge_runtime_config",
        lambda _overrides: {
            "ingest_pre_llm_max_workers": 1,
            "ingest_llm_max_workers": 2,
            "llm_global_max_concurrent": 12,
        },
    )
    monkeypatch.setattr(pipeline.settings, "ingest_llm_heartbeat_seconds", 5)

    def fake_phase1_trace(**kwargs):  # noqa: ANN003
        time.sleep(5.2)
        rec = kwargs['cite_rec']
        return _fake_phase1_trace_output(rec['paper_id'])

    monkeypatch.setattr(pipeline, "run_phase1_paper_logic_trace", fake_phase1_trace)
    monkeypatch.setattr(
        pipeline,
        "classify_citation_purposes_batch",
        lambda **kwargs: {"by_id": {}},  # noqa: ARG005
    )

    pipeline.ingest_markdowns(
        ["dummy-1.md", "dummy-2.md", "dummy-3.md"],
        progress=lambda stage, _p, msg=None: progress_messages.append(str(msg or "")) if stage == "ingest:llm" else None,
    )

    assert len(fake_client.trace_writes) == 3
    assert any("active=2" in message and "queued=1" in message for message in progress_messages)
    assert all("running=3" not in message for message in progress_messages)
    assert any("completed=3/3" in message and "queued=0" in message for message in progress_messages)


def test_ingest_result_phase1_quality_exposes_trace_metrics_not_legacy_claim_metrics(monkeypatch):  # noqa: ANN001, ANN201
    docs = [_mock_document(0)]
    docs_iter = iter(docs)
    fake_client = _FakeNeo4jClient()

    monkeypatch.setattr(pipeline, "Neo4jClient", lambda *args, **kwargs: fake_client)  # noqa: ARG005
    monkeypatch.setattr(pipeline, "parse_mineru_markdown", lambda _md: next(docs_iter))
    monkeypatch.setattr(pipeline, "recover_references_with_agent", lambda doc, **kwargs: (doc, {"before_refs": 0, "after_refs": 0}))  # noqa: ARG005,E501
    monkeypatch.setattr(pipeline, "recover_citation_events_from_references", lambda doc, **kwargs: (doc, {"before_events": 0, "after_events": 0}))  # noqa: ARG005,E501
    monkeypatch.setattr(pipeline, "CrossrefClient", lambda: object())
    monkeypatch.setattr(
        pipeline,
        "build_reference_and_cite_records",
        lambda doc, **kwargs: {  # noqa: ARG005
            "paper_id": paper_id_for_md_path(doc.paper.md_path, doi=doc.paper.doi),
            "refs": [],
            "cited_papers": [],
            "cites_resolved": [],
            "cites_unresolved": [],
        },
    )
    monkeypatch.setattr(pipeline, "load_canonical_meta", lambda _path: {"paper_type": "research"})
    monkeypatch.setattr(pipeline, "load_active", lambda _paper_type: {"version": 1, "paper_type": "research", "rules": {}, "prompts": {}})  # noqa: E501
    monkeypatch.setattr(pipeline, "extract_figures_from_markdown", lambda **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(pipeline, "build_faiss_for_chunks", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(pipeline, "build_faiss_for_rows", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(pipeline, "build_community_corpus_rows", lambda *args, **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(pipeline, "_write_document_ir", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(Path, "write_text", lambda self, *args, **kwargs: 0)  # noqa: ARG005
    monkeypatch.setattr(
        pipeline,
        "merge_runtime_config",
        lambda _overrides: {
            "ingest_pre_llm_max_workers": 1,
            "ingest_llm_max_workers": 1,
            "llm_global_max_concurrent": 4,
        },
    )
    monkeypatch.setattr(
        pipeline,
        "run_phase1_paper_logic_trace",
        lambda **kwargs: _fake_phase1_trace_output(kwargs["cite_rec"]["paper_id"], quality_tier="green", audit_status="not_needed"),
    )
    monkeypatch.setattr(pipeline, "classify_citation_purposes_batch", lambda **kwargs: {"by_id": {}})  # noqa: ARG005

    result = pipeline.ingest_markdowns(["dummy-1.md"])

    phase1_quality = result["phase1_quality"][0]
    assert phase1_quality["hot_path_gate_passed"] is True
    assert phase1_quality["move_count"] == 1
    assert phase1_quality["anchor_count"] == 1
    assert "supported_claim_ratio" not in phase1_quality
    assert "step_coverage_ratio" not in phase1_quality


def test_ingest_llm_requests_are_not_pinned_to_a_single_bound_worker(monkeypatch):  # noqa: ANN001, ANN201
    from app.llm import client as llm_client

    docs = [_mock_document(index) for index in range(2)]
    docs_iter = iter(docs)
    seen_bound_ids: list[str | None] = []
    fake_client = _FakeNeo4jClient()

    monkeypatch.setattr(pipeline, "Neo4jClient", lambda *args, **kwargs: fake_client)  # noqa: ARG005
    monkeypatch.setattr(pipeline, "parse_mineru_markdown", lambda _md: next(docs_iter))
    monkeypatch.setattr(
        pipeline,
        "recover_references_with_agent",
        lambda doc, **kwargs: (doc, {"before_refs": 0, "after_refs": 0}),  # noqa: ARG005
    )
    monkeypatch.setattr(
        pipeline,
        "recover_citation_events_from_references",
        lambda doc, **kwargs: (doc, {"before_events": 0, "after_events": 0}),  # noqa: ARG005
    )
    monkeypatch.setattr(pipeline, "CrossrefClient", lambda: object())
    monkeypatch.setattr(
        pipeline,
        "build_reference_and_cite_records",
        lambda doc, **kwargs: {  # noqa: ARG005
            "paper_id": paper_id_for_md_path(doc.paper.md_path, doi=doc.paper.doi),
            "refs": [],
            "cited_papers": [],
            "cites_resolved": [],
            "cites_unresolved": [],
        },
    )
    monkeypatch.setattr(pipeline, "load_canonical_meta", lambda _path: {"paper_type": "research"})
    monkeypatch.setattr(
        pipeline,
        "load_active",
        lambda _paper_type: {"version": 1, "paper_type": "research", "rules": {}, "prompts": {}},
    )
    monkeypatch.setattr(pipeline, "extract_figures_from_markdown", lambda **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(pipeline, "build_faiss_for_chunks", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(pipeline, "build_faiss_for_rows", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(pipeline, "build_community_corpus_rows", lambda *args, **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(pipeline, "_write_document_ir", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(Path, "write_text", lambda self, *args, **kwargs: 0)  # noqa: ARG005
    monkeypatch.setattr(
        pipeline,
        "merge_runtime_config",
        lambda _overrides: {
            "ingest_pre_llm_max_workers": 1,
            "ingest_llm_max_workers": 2,
            "llm_global_max_concurrent": 12,
        },
    )

    def fake_phase1_trace(**kwargs):  # noqa: ANN003
        seen_bound_ids.append(llm_client.get_bound_llm_worker_id())
        rec = kwargs['cite_rec']
        return _fake_phase1_trace_output(rec['paper_id'])

    monkeypatch.setattr(pipeline, "run_phase1_paper_logic_trace", fake_phase1_trace)
    monkeypatch.setattr(
        pipeline,
        "classify_citation_purposes_batch",
        lambda **kwargs: {"by_id": {}},  # noqa: ARG005
    )

    pipeline.ingest_markdowns(["dummy-1.md", "dummy-2.md"])

    assert seen_bound_ids == [None, None]


def test_ingest_llm_propagates_active_paper_count_for_single_paper_bursting(monkeypatch):  # noqa: ANN001, ANN201
    from app.llm import client as llm_client

    docs = [_mock_document(index) for index in range(2)]
    docs_iter = iter(docs)
    seen_active_papers: list[int | None] = []

    monkeypatch.setattr(pipeline, "Neo4jClient", lambda *args, **kwargs: _FakeNeo4jClient())  # noqa: ARG005
    monkeypatch.setattr(pipeline, "parse_mineru_markdown", lambda _md: next(docs_iter))
    monkeypatch.setattr(
        pipeline,
        "recover_references_with_agent",
        lambda doc, **kwargs: (doc, {"before_refs": 0, "after_refs": 0}),  # noqa: ARG005
    )
    monkeypatch.setattr(
        pipeline,
        "recover_citation_events_from_references",
        lambda doc, **kwargs: (doc, {"before_events": 0, "after_events": 0}),  # noqa: ARG005
    )
    monkeypatch.setattr(pipeline, "CrossrefClient", lambda: object())
    monkeypatch.setattr(
        pipeline,
        "build_reference_and_cite_records",
        lambda doc, **kwargs: {  # noqa: ARG005
            "paper_id": paper_id_for_md_path(doc.paper.md_path, doi=doc.paper.doi),
            "refs": [],
            "cited_papers": [],
            "cites_resolved": [],
            "cites_unresolved": [],
        },
    )
    monkeypatch.setattr(pipeline, "load_canonical_meta", lambda _path: {"paper_type": "research"})
    monkeypatch.setattr(
        pipeline,
        "load_active",
        lambda _paper_type: {"version": 1, "paper_type": "research", "rules": {}, "prompts": {}},
    )
    monkeypatch.setattr(pipeline, "extract_figures_from_markdown", lambda **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(pipeline, "build_faiss_for_chunks", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(pipeline, "build_faiss_for_rows", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(pipeline, "build_community_corpus_rows", lambda *args, **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(pipeline, "_write_document_ir", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(Path, "write_text", lambda self, *args, **kwargs: 0)  # noqa: ARG005
    monkeypatch.setattr(
        pipeline,
        "merge_runtime_config",
        lambda _overrides: {
            "ingest_pre_llm_max_workers": 1,
            "ingest_llm_max_workers": 2,
            "llm_global_max_concurrent": 12,
        },
    )

    def fake_phase1_trace(**kwargs):  # noqa: ANN003
        seen_active_papers.append(llm_client.get_active_llm_paper_count())
        rec = kwargs['cite_rec']
        return _fake_phase1_trace_output(rec['paper_id'])

    monkeypatch.setattr(pipeline, "run_phase1_paper_logic_trace", fake_phase1_trace)
    monkeypatch.setattr(
        pipeline,
        "classify_citation_purposes_batch",
        lambda **kwargs: {"by_id": {}},  # noqa: ARG005
    )

    pipeline.ingest_markdowns(["dummy-1.md", "dummy-2.md"])

    assert seen_active_papers == [2, 2]


def test_ingest_llm_can_defer_citation_purpose_enrichment(monkeypatch):  # noqa: ANN001, ANN201
    docs = [_mock_document(index) for index in range(1)]
    docs_iter = iter(docs)

    monkeypatch.setattr(pipeline, "Neo4jClient", lambda *args, **kwargs: _FakeNeo4jClient())  # noqa: ARG005
    monkeypatch.setattr(pipeline, "parse_mineru_markdown", lambda _md: next(docs_iter))
    monkeypatch.setattr(
        pipeline,
        "recover_references_with_agent",
        lambda doc, **kwargs: (doc, {"before_refs": 0, "after_refs": 0}),  # noqa: ARG005
    )
    monkeypatch.setattr(
        pipeline,
        "recover_citation_events_from_references",
        lambda doc, **kwargs: (doc, {"before_events": 0, "after_events": 0}),  # noqa: ARG005
    )
    monkeypatch.setattr(pipeline, "CrossrefClient", lambda: object())
    monkeypatch.setattr(
        pipeline,
        "build_reference_and_cite_records",
        lambda doc, **kwargs: {  # noqa: ARG005
            "paper_id": paper_id_for_md_path(doc.paper.md_path, doi=doc.paper.doi),
            "refs": [],
            "cited_papers": [],
            "cites_resolved": [],
            "cites_unresolved": [],
        },
    )
    monkeypatch.setattr(pipeline, "load_canonical_meta", lambda _path: {"paper_type": "research"})
    monkeypatch.setattr(
        pipeline,
        "load_active",
        lambda _paper_type: {"version": 1, "paper_type": "research", "rules": {}, "prompts": {}},
    )
    monkeypatch.setattr(pipeline, "extract_figures_from_markdown", lambda **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(pipeline, "build_faiss_for_chunks", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(pipeline, "build_faiss_for_rows", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(pipeline, "build_community_corpus_rows", lambda *args, **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(pipeline, "_write_document_ir", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(Path, "write_text", lambda self, *args, **kwargs: 0)  # noqa: ARG005
    monkeypatch.setattr(
        pipeline,
        "merge_runtime_config",
        lambda _overrides: {
            "ingest_pre_llm_max_workers": 1,
            "ingest_llm_max_workers": 1,
            "llm_global_max_concurrent": 12,
            "ingest_defer_citation_purposes": True,
        },
    )
    monkeypatch.setattr(
        pipeline,
        "run_phase1_paper_logic_trace",
        lambda **kwargs: _fake_phase1_trace_output(kwargs["cite_rec"]["paper_id"]),  # noqa: ARG005
    )

    def _should_not_run(**kwargs):  # noqa: ANN003
        raise AssertionError("citation purpose enrichment should be deferred off the ingest hot path")

    monkeypatch.setattr(pipeline, "classify_citation_purposes_batch", _should_not_run)

    out = pipeline.ingest_markdowns(["dummy-1.md"])

    assert out["llm_built"] is True
    assert out["llm_error"] is None


def test_ingest_completes_when_valid_paper_logic_trace_exists_without_audit(monkeypatch):  # noqa: ANN001, ANN201
    docs = [_mock_document(index) for index in range(1)]
    docs_iter = iter(docs)

    monkeypatch.setattr(pipeline, "Neo4jClient", lambda *args, **kwargs: _FakeNeo4jClient())  # noqa: ARG005
    monkeypatch.setattr(pipeline, "parse_mineru_markdown", lambda _md: next(docs_iter))
    monkeypatch.setattr(
        pipeline,
        "recover_references_with_agent",
        lambda doc, **kwargs: (doc, {"before_refs": 0, "after_refs": 0}),  # noqa: ARG005
    )
    monkeypatch.setattr(
        pipeline,
        "recover_citation_events_from_references",
        lambda doc, **kwargs: (doc, {"before_events": 0, "after_events": 0}),  # noqa: ARG005
    )
    monkeypatch.setattr(pipeline, "CrossrefClient", lambda: object())
    monkeypatch.setattr(
        pipeline,
        "build_reference_and_cite_records",
        lambda doc, **kwargs: {  # noqa: ARG005
            "paper_id": paper_id_for_md_path(doc.paper.md_path, doi=doc.paper.doi),
            "refs": [],
            "cited_papers": [],
            "cites_resolved": [],
            "cites_unresolved": [],
        },
    )
    monkeypatch.setattr(pipeline, "load_canonical_meta", lambda _path: {"paper_type": "research"})
    monkeypatch.setattr(
        pipeline,
        "load_active",
        lambda _paper_type: {"version": 1, "paper_type": "research", "rules": {}, "prompts": {}},
    )
    monkeypatch.setattr(pipeline, "extract_figures_from_markdown", lambda **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(pipeline, "build_faiss_for_chunks", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(pipeline, "build_faiss_for_rows", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(pipeline, "build_community_corpus_rows", lambda *args, **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(pipeline, "_write_document_ir", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(Path, "write_text", lambda self, *args, **kwargs: 0)  # noqa: ARG005
    monkeypatch.setattr(
        pipeline,
        "merge_runtime_config",
        lambda _overrides: {
            "ingest_pre_llm_max_workers": 1,
            "ingest_llm_max_workers": 1,
            "llm_global_max_concurrent": 12,
            "ingest_defer_citation_purposes": True,
        },
    )
    monkeypatch.setattr(
        pipeline,
        "run_phase1_paper_logic_trace",
        lambda **kwargs: _fake_phase1_trace_output(  # noqa: ARG005
            kwargs["cite_rec"]["paper_id"],
            quality_tier="yellow",
            audit_status="eligible",
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "classify_citation_purposes_batch",
        lambda **kwargs: {"by_id": {}},  # noqa: ARG005
    )

    out = pipeline.ingest_markdowns(["dummy-1.md"])

    assert out["llm_built"] is True
    assert out["paper_logic_traces"][0]["quality_tier"] == "yellow"
    assert out["paper_logic_traces"][0]["audit_status"] == "eligible"


def test_ingest_markdowns_avoids_legacy_artifacts_and_faiss_corpora(monkeypatch):  # noqa: ANN001, ANN201
    docs = [_mock_document(0)]
    docs_iter = iter(docs)
    fake_client = _FakeNeo4jClient()
    written_files: list[str] = []
    faiss_targets: list[str] = []

    monkeypatch.setattr(pipeline, "Neo4jClient", lambda *args, **kwargs: fake_client)  # noqa: ARG005
    monkeypatch.setattr(pipeline, "parse_mineru_markdown", lambda _md: next(docs_iter))
    monkeypatch.setattr(
        pipeline,
        "recover_references_with_agent",
        lambda doc, **kwargs: (doc, {"before_refs": 0, "after_refs": 0}),  # noqa: ARG005
    )
    monkeypatch.setattr(
        pipeline,
        "recover_citation_events_from_references",
        lambda doc, **kwargs: (doc, {"before_events": 0, "after_events": 0}),  # noqa: ARG005
    )
    monkeypatch.setattr(pipeline, "CrossrefClient", lambda: object())
    monkeypatch.setattr(
        pipeline,
        "build_reference_and_cite_records",
        lambda doc, **kwargs: {  # noqa: ARG005
            "paper_id": paper_id_for_md_path(doc.paper.md_path, doi=doc.paper.doi),
            "refs": [],
            "cited_papers": [],
            "cites_resolved": [],
            "cites_unresolved": [],
        },
    )
    monkeypatch.setattr(pipeline, "load_canonical_meta", lambda _path: {"paper_type": "research"})
    monkeypatch.setattr(
        pipeline,
        "load_active",
        lambda _paper_type: {"version": 1, "paper_type": "research", "rules": {}, "prompts": {}},
    )
    monkeypatch.setattr(pipeline, "extract_figures_from_markdown", lambda **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(
        pipeline,
        "build_faiss_for_chunks",
        lambda *args, **kwargs: {"dir": str(kwargs.get("out_dir") or args[1])},  # noqa: ARG005
    )
    monkeypatch.setattr(
        pipeline,
        "build_faiss_for_rows",
        lambda rows, out_dir, **kwargs: faiss_targets.append(str(out_dir)) or {  # noqa: ARG005
            "dir": str(out_dir),
            "rows_indexed": len(rows),
            "segments_indexed": len(rows),
        },
    )
    monkeypatch.setattr(pipeline, "build_community_corpus_rows", lambda *args, **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(
        pipeline,
        "run_phase1_paper_logic_trace",
        lambda **kwargs: _fake_phase1_trace_output(kwargs["cite_rec"]["paper_id"]),  # noqa: ARG005
    )
    monkeypatch.setattr(
        pipeline,
        "classify_citation_purposes_batch",
        lambda **kwargs: {"by_id": {}},  # noqa: ARG005
    )
    monkeypatch.setattr(pipeline, "_write_document_ir", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(Path, "write_text", lambda self, *args, **kwargs: written_files.append(self.name) or 0)  # noqa: ARG005,E501
    monkeypatch.setattr(
        pipeline,
        "merge_runtime_config",
        lambda _overrides: {
            "ingest_pre_llm_max_workers": 1,
            "ingest_llm_max_workers": 1,
            "llm_global_max_concurrent": 4,
        },
    )
    monkeypatch.setattr(
        fake_client,
        "list_research_moves",
        lambda *args, **kwargs: [  # noqa: ARG005
            {
                "kind": "research_move",
                "source_id": "m-1",
                "text": "We propose a graph encoder.",
                "paper_id": "doi:10.1000/test0",
                "paper_source": "test_paper_0",
                "role": "method",
                "act_type": "propose_method",
            }
        ],
    )
    monkeypatch.setattr(
        fake_client,
        "list_evidence_anchors",
        lambda *args, **kwargs: [  # noqa: ARG005
            {
                "kind": "evidence_anchor",
                "source_id": "a-1",
                "text": "We propose a graph encoder.",
                "paper_id": "doi:10.1000/test0",
                "paper_source": "test_paper_0",
                "move_id": "m-1",
                "quote": "We propose a graph encoder.",
            }
        ],
    )
    pipeline.ingest_markdowns(["dummy-1.md"])

    assert "test_paper_0.paper_logic_trace.json" in written_files
    assert "test_paper_0.llm_imrad.json" not in written_files
    corpus_dirs = {Path(path).name for path in faiss_targets}
    assert "research_moves" in corpus_dirs
    assert "evidence_anchors" in corpus_dirs
    assert "logic_steps" not in corpus_dirs
    assert "claims" not in corpus_dirs
