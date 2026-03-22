from __future__ import annotations

from app.paper_logic_trace.exporter import export_paper_logic_trace


class _FakeClient:
    def get_paper_logic_trace_inputs(self, paper_id: str) -> dict:
        return {
            'paper_metadata': {
                'paper_id': paper_id,
                'title': 'Demo Paper',
                'source_refs': ['chunk:1'],
            },
            'evidence_rows': [
                {
                    'anchor_id': 'a-1',
                    'paper_id': paper_id,
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
            'figure_rows': [],
            'table_rows': [],
            'citation_rows': [],
            'move_relation_rows': [],
            'built_at': '2026-03-22T12:00:00Z',
        }


def test_export_paper_logic_trace_returns_canonical_payload() -> None:
    trace = export_paper_logic_trace(_FakeClient(), 'paper-1')

    assert trace.canonical_core.moves[0].move_id == 'm-1'
    assert trace.canonical_core.move_relations == []
    assert trace.derived_views['community_signatures'][0]['move_id'] == 'm-1'
    assert trace.quality['audit_status'] == 'eligible'
