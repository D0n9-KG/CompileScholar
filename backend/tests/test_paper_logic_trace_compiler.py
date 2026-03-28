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


def test_compile_paper_logic_trace_preserves_metadata_audit_fields() -> None:
    trace = compile_paper_logic_trace(
        paper_metadata={
            'paper_id': 'paper-1',
            'title': '城镇污水处理厂污染物排放标准 浅释',
            'title_alt': '1.2《污水综合排放标准》不适应污水处理厂建设管理需求',
            'metadata_enrichment': {
                'mode': 'skipped_unreliable_title_match',
                'used_crossref': False,
                'local_fallback_used': True,
                'local_fallback_changed_fields': ['authors', 'title', 'title_alt'],
            },
            'source_refs': ['chunk:1'],
        },
        evidence_rows=[
            {
                'anchor_id': 'a-1',
                'paper_id': 'paper-1',
                'source_ref': 'chunk:1',
                'modality': 'text',
                'section_path': ['Abstract'],
                'locator': {'chunk_id': 'chunk:1'},
                'quote': '本文介绍了城镇污水处理厂污染物排放标准制定思路。',
                'citation_ids': [],
                'support_type': 'direct',
                'weak': False,
                'move_id': 'm-1',
                'sequence_no': 1,
                'role_hint': 'problem',
                'act_hint': 'define_task',
                'summary': '本文介绍了城镇污水处理厂污染物排放标准制定思路。',
                'research_objects': [{'surface': '城镇污水处理厂污染物排放标准', 'anchor_ids': ['a-1']}],
            }
        ],
        figure_rows=[],
        table_rows=[],
        citation_rows=[],
        built_at='2026-03-28T12:00:00Z',
    )

    assert trace.paper_metadata.title_alt == '1.2《污水综合排放标准》不适应污水处理厂建设管理需求'
    assert trace.paper_metadata.metadata_enrichment['local_fallback_used'] is True
    assert trace.paper_metadata.metadata_enrichment['mode'] == 'skipped_unreliable_title_match'


def test_compile_paper_logic_trace_downgrades_metadata_summary_mismatch() -> None:
    trace = compile_paper_logic_trace(
        paper_metadata={
            'paper_id': 'paper-79',
            'title': '高功率光纤激光热光效应及模式不稳定阈值特性研究',
            'authors': ['李学文', '于春雷'],
            'paper_type': 'empirical',
            'source_refs': ['chunk:1'],
        },
        evidence_rows=[
            {
                'anchor_id': 'a-1',
                'paper_id': 'paper-79',
                'source_ref': 'chunk:1',
                'modality': 'text',
                'section_path': ['Abstract'],
                'locator': {'chunk_id': 'chunk:1'},
                'quote': '本文对卧式双轴圆盘反应器的功率特性进行了比较详细的研究。',
                'citation_ids': [],
                'support_type': 'direct',
                'weak': False,
                'move_id': 'm-1',
                'sequence_no': 1,
                'role_hint': 'problem',
                'act_hint': 'define_task',
                'summary': '本文对卧式双轴圆盘反应器的功率特性进行了比较详细的研究。',
                'research_objects': [{'surface': '卧式双轴圆盘反应器', 'normalized': '卧式双轴圆盘反应器', 'anchor_ids': ['a-1']}],
            },
            {
                'anchor_id': 'a-2',
                'paper_id': 'paper-79',
                'source_ref': 'chunk:2',
                'modality': 'text',
                'section_path': ['Method'],
                'locator': {'chunk_id': 'chunk:2'},
                'quote': '通过扭矩法测量卧式双轴圆盘反应器的搅拌功率。',
                'citation_ids': [],
                'support_type': 'direct',
                'weak': False,
                'move_id': 'm-2',
                'sequence_no': 2,
                'role_hint': 'method',
                'act_hint': 'propose_method',
                'summary': '通过扭矩法测量卧式双轴圆盘反应器的搅拌功率。',
                'methods': [{'surface': '扭矩法', 'normalized': '扭矩法', 'anchor_ids': ['a-2']}],
                'research_objects': [{'surface': '卧式双轴圆盘反应器', 'normalized': '卧式双轴圆盘反应器', 'anchor_ids': ['a-2']}],
            },
            {
                'anchor_id': 'a-3',
                'paper_id': 'paper-79',
                'source_ref': 'chunk:3',
                'modality': 'text',
                'section_path': ['Result'],
                'locator': {'chunk_id': 'chunk:3'},
                'quote': '得到了统一的功率关联式。',
                'citation_ids': [],
                'support_type': 'direct',
                'weak': False,
                'move_id': 'm-3',
                'sequence_no': 3,
                'role_hint': 'result',
                'act_hint': 'report_effect',
                'summary': '得到了统一的功率关联式。',
                'metrics': [{'surface': '功率关联式', 'normalized': '功率关联式', 'anchor_ids': ['a-3']}],
                'effects': [{'direction': 'improve', 'anchor_ids': ['a-3']}],
            },
        ],
        figure_rows=[],
        table_rows=[],
        citation_rows=[],
        built_at='2026-03-27T12:00:00Z',
    )

    assert trace.quality['quality_tier'] == 'yellow'
    assert 'metadata_summary_mismatch' in trace.quality['quality_flags']
