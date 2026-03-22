from __future__ import annotations

from app.paper_logic_trace.models import (
    CanonicalCore,
    EvidenceAnchor,
    PaperLogicTrace,
    PaperMetadata,
    ResearchMove,
)


def test_paper_logic_trace_uses_canonical_core() -> None:
    trace = PaperLogicTrace(
        trace_id='trace-1',
        schema_version='v2',
        built_at='2026-03-22T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-1',
            title='Demo',
            source_refs=['chunk:1'],
        ),
        canonical_core=CanonicalCore(
            evidence_anchors=[
                EvidenceAnchor(
                    anchor_id='a-1',
                    paper_id='paper-1',
                    source_ref='chunk:1',
                    modality='text',
                    section_path=[],
                    locator={},
                    quote='demo',
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
                    summary='Uses graph encoding',
                    anchor_ids=['a-1'],
                    confidence=0.8,
                )
            ],
            move_relations=[],
            citation_acts=[],
            figure_refs=[],
            table_refs=[],
        ),
        derived_views={},
        quality={'quality_tier': 'green', 'hot_path_gate_report': {}},
    )

    assert trace.canonical_core.moves[0].role == 'method'
