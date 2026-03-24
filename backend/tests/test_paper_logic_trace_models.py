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


def test_derived_views_can_fallback_summary_tokens_for_sparse_move() -> None:
    from app.paper_logic_trace.derived_views import build_community_signatures

    move = ResearchMove(
        move_id='m-1',
        sequence_no=1,
        role='background',
        act_type='define_task',
        summary='Particle packing behavior under high stress conditions is studied.',
        anchor_ids=['a-1'],
    )

    signatures = build_community_signatures('paper-1', [move])

    assert signatures[0]['object_tokens'] or signatures[0]['condition_tokens']


def test_compile_trace_normalizes_research_paper_type_to_empirical() -> None:
    from app.paper_logic_trace.compiler import compile_paper_logic_trace

    trace = compile_paper_logic_trace(
        paper_metadata={
            'paper_id': 'paper-1',
            'title': 'Demo',
            'paper_type': 'research',
            'source_refs': ['chunk:1'],
        },
        evidence_rows=[
            {
                'anchor_id': 'a-1',
                'paper_id': 'paper-1',
                'source_ref': 'chunk:1',
                'modality': 'text',
                'section_path': ['1. Introduction'],
                'locator': {'chunk_id': 'chunk:1'},
                'quote': 'We investigate particle crushing prediction.',
                'citation_ids': [],
                'support_type': 'direct',
                'weak': False,
                'move_id': 'm-1',
                'sequence_no': 1,
                'role_hint': 'problem',
                'act_hint': 'define_task',
                'summary': 'We investigate particle crushing prediction.',
                'research_objects': [{'surface': 'particle crushing prediction'}],
                'methods': [],
                'observed_variables': [],
                'metrics': [],
                'comparators': [],
                'conditions': [],
                'effects': [],
                'limitation_types': [],
                'resource_mentions': [],
                'slot_provenance': [],
                'confidence': 0.8,
            }
        ],
        figure_rows=[],
        table_rows=[],
        citation_rows=[],
        move_relation_rows=[],
    )

    assert trace.paper_metadata.paper_type == 'empirical'
