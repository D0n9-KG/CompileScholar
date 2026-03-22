from __future__ import annotations

from app.paper_logic_trace.compiler import compile_paper_logic_trace


def test_compile_paper_logic_trace_emits_research_moves() -> None:
    trace = compile_paper_logic_trace(
        paper_metadata={
            'paper_id': 'paper-1',
            'title': 'Demo',
            'source_refs': ['chunk:1'],
        },
        evidence_rows=[
            {
                'anchor_id': 'a-1',
                'paper_id': 'paper-1',
                'source_ref': 'chunk:1',
                'modality': 'text',
                'section_path': ['Method'],
                'locator': {'chunk_id': 'chunk:1'},
                'quote': 'We propose a graph encoder for retrieval.',
                'citation_ids': [],
                'support_type': 'direct',
                'weak': False,
                'move_id': 'm-1',
                'sequence_no': 1,
                'role_hint': 'method',
                'act_hint': 'propose_method',
                'summary': 'We propose a graph encoder for retrieval.',
                'methods': [{'surface': 'graph encoder', 'normalized': 'graph encoder', 'anchor_ids': ['a-1']}],
                'research_objects': [{'surface': 'retrieval', 'normalized': 'retrieval', 'anchor_ids': ['a-1']}],
            }
        ],
        figure_rows=[],
        table_rows=[],
        citation_rows=[],
        built_at='2026-03-22T12:00:00Z',
    )

    assert trace.paper_metadata.paper_id == 'paper-1'
    assert trace.canonical_core.moves[0].act_type == 'propose_method'
    assert trace.canonical_core.moves[0].anchor_ids == ['a-1']
    assert trace.derived_views['community_signatures'][0]['move_id'] == 'm-1'
