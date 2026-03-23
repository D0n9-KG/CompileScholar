from __future__ import annotations

from app.paper_logic_trace.exporter import export_paper_logic_trace


class _FakeClient:
    def get_paper_logic_trace(self, paper_id: str) -> dict:
        return {
            'trace_id': f'{paper_id}:paper_logic_trace',
            'schema_version': 'v2',
            'built_at': '2026-03-22T12:00:00Z',
            'paper_metadata': {
                'paper_id': paper_id,
                'title': 'Demo Paper',
                'source_refs': ['chunk:1'],
            },
            'canonical_core': {
                'evidence_anchors': [
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
                    }
                ],
                'moves': [
                    {
                        'move_id': 'm-1',
                        'sequence_no': 1,
                        'role': 'method',
                        'act_type': 'propose_method',
                        'summary': 'We propose a graph encoder for retrieval.',
                        'methods': [{'surface': 'graph encoder', 'normalized': 'graph encoder', 'anchor_ids': ['a-1']}],
                        'research_objects': [{'surface': 'retrieval', 'normalized': 'retrieval', 'anchor_ids': ['a-1']}],
                        'anchor_ids': ['a-1'],
                    }
                ],
                'move_relations': [],
                'citation_acts': [],
                'figure_refs': [],
                'table_refs': [],
            },
            'derived_views': {'community_signatures': [{'move_id': 'm-1'}]},
            'quality': {'audit_status': 'eligible', 'quality_tier': 'yellow'},
        }


def test_export_paper_logic_trace_returns_canonical_payload() -> None:
    trace = export_paper_logic_trace(_FakeClient(), 'paper-1')

    assert trace.canonical_core.moves[0].move_id == 'm-1'
    assert trace.canonical_core.move_relations == []
    assert trace.derived_views['community_signatures'][0]['move_id'] == 'm-1'
    assert trace.quality['audit_status'] == 'eligible'
