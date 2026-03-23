from __future__ import annotations

from app.ingest.models import Chunk, DocumentIR, MdSpan, PaperDraft
from app.paper_logic_trace.compiler import compile_paper_logic_trace
from app.paper_logic_trace.direct_extraction import (
    _build_move_relation_rows,
    _build_research_move_prompt,
    build_paper_logic_trace_inputs,
)


def _doc_with_chunks(*chunks: Chunk, title: str = 'Demo Paper') -> DocumentIR:
    return DocumentIR(
        paper=PaperDraft(
            paper_source='demo-paper',
            md_path='C:/tmp/demo.md',
            title=title,
            title_alt=None,
            authors=['Alice'],
            doi='10.1000/demo',
            year=2024,
            paper_type='empirical',
        ),
        chunks=list(chunks),
        references=[],
        citations=[],
    )


def _chunk(chunk_id: str, section: str | None, text: str, *, line: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_source='demo-paper',
        md_path='C:/tmp/demo.md',
        span=MdSpan(start_line=line, end_line=line),
        section=section,
        kind='block',
        text=text,
    )


def test_intro_task_chunk_is_promoted_to_problem_role(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', '1. Introduction', 'This paper investigates particle crushing prediction under high stress conditions.', line=1),
        _chunk('c-2', '2. Method', 'We propose a discrete element simulation workflow.', line=2),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    roles = {row['role_hint'] for row in payload['evidence_rows']}

    assert 'problem' in roles


def test_pre_section_task_chunk_under_title_is_promoted_to_problem_role(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', 'Demo Paper', 'In this paper, we investigate particle recirculation in granular avalanches.', line=1),
        _chunk('c-2', '2. Method', 'We propose a travelling-wave analysis for breaking segregation shocks.', line=2),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    roles = {row['role_hint'] for row in payload['evidence_rows']}

    assert 'problem' in roles


def test_discussion_result_chunk_is_promoted_to_result_role(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', '4. Discussion', 'Results show that particle recirculation decreases once the travelling wave structure stabilizes.', line=1),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    roles = {row['role_hint'] for row in payload['evidence_rows']}

    assert 'result' in roles


def test_act_type_promotes_problem_role_and_relation_uses_addresses(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', '1. Introduction', 'We investigate how travelling segregation waves break.', line=1),
        _chunk('c-2', '2. Method', 'We propose a travelling-wave construction for the breaking zone.', line=2),
    )

    def _fake_extract_window_moves_llm(**kwargs):
        window = kwargs['window']
        role_hint = window.get('role_hint')
        if role_hint == 'problem':
            return [
                {
                    'role': 'background',
                    'act_type': 'define_task',
                    'summary': 'We investigate how travelling segregation waves break.',
                    'anchor_chunk_ids': ['c-1'],
                    'confidence': 0.6,
                }
            ]
        if role_hint == 'method':
            return [
                {
                    'role': 'background',
                    'act_type': 'propose_method',
                    'summary': 'We propose a travelling-wave construction for the breaking zone.',
                    'anchor_chunk_ids': ['c-2'],
                    'confidence': 0.6,
                }
            ]
        return []

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        _fake_extract_window_moves_llm,
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    roles = [row['role_hint'] for row in payload['evidence_rows']]
    relation_types = [row['relation_type'] for row in payload['move_relation_rows']]

    assert roles == ['problem', 'method']
    assert relation_types == ['addresses']


def test_title_and_image_noise_are_filtered_before_move_construction(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', 'Title', '# Discrete element modeling of crushable sands', line=1),
        _chunk('c-2', None, '![](images/demo.png)', line=2),
        _chunk('c-3', '2. Method', 'We propose a discrete element simulation workflow for crushable sands.', line=3),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    summaries = [str(row['summary']) for row in payload['evidence_rows']]
    joined = ' '.join(summaries).lower()

    assert 'images/demo.png' not in joined
    assert '# discrete element modeling of crushable sands' not in joined


def test_title_front_matter_chunks_are_filtered_before_move_construction(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', 'Demo Paper', 'A. R. THORNTON AND J. M. N. T. GRAY', line=1),
        _chunk('c-2', 'Demo Paper', 'School of Mathematics, University of Manchester, UK', line=2),
        _chunk('c-3', 'Demo Paper', '(Received 27 April 2007 and in revised form 5 October 2007)', line=3),
        _chunk('c-4', 'Demo Paper', 'In this paper, we investigate particle recirculation in granular avalanches.', line=4),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    source_refs = {str(row['source_ref']) for row in payload['evidence_rows']}
    summaries = [str(row['summary']) for row in payload['evidence_rows']]
    joined = ' '.join(summaries).lower()

    assert 'c-1' not in source_refs
    assert 'c-2' not in source_refs
    assert 'c-3' not in source_refs
    assert 'c-4' in source_refs
    assert 'we investigate particle recirculation' in joined


def test_mixed_case_author_list_under_title_is_filtered_before_move_construction(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', 'Demo Paper', 'Ru Fu a, Xinli Hu a, Bo Zhou b,*', line=1),
        _chunk('c-2', 'Demo Paper', '$^{a}$ Faculty of Engineering, China University of Geosciences, Wuhan, China', line=2),
        _chunk('c-3', 'ABSTRACT', 'This study proposed a novel approach for generating crushable agglomerates with realistic particle shapes.', line=3),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    source_refs = {str(row['source_ref']) for row in payload['evidence_rows']}
    joined = ' '.join(str(row['summary']) for row in payload['evidence_rows']).lower()

    assert 'c-1' not in source_refs
    assert 'c-2' not in source_refs
    assert 'c-3' in source_refs
    assert 'ru fu' not in joined
    assert 'xinli hu' not in joined


def test_heading_only_chunks_are_filtered_before_move_construction(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', '3. Results', '# 3. Results', line=1),
        _chunk('c-2', '2.1. Systems studied', '# 2.1. Systems studied', line=2),
        _chunk('c-3', '2. Method', 'We propose a discrete element simulation workflow for crushable sands.', line=3),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    source_refs = {str(row['source_ref']) for row in payload['evidence_rows']}
    joined = ' '.join(str(row['summary']) for row in payload['evidence_rows']).lower()

    assert 'c-1' not in source_refs
    assert 'c-2' not in source_refs
    assert 'c-3' in source_refs
    assert '# 3. results' not in joined
    assert '# 2.1. systems studied' not in joined


def test_article_metadata_chunks_are_filtered_before_move_construction(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', 'ARTICLE INFO', '# ARTICLE INFO', line=1),
        _chunk('c-2', 'Demo Paper', 'Available online 29 June 2006', line=2),
        _chunk('c-3', 'Demo Paper', 'Keywords: DEM, segregation, die filling', line=3),
        _chunk('c-4', 'ABSTRACT', 'This paper investigates segregation during die filling using DEM and RSM.', line=4),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    source_refs = {str(row['source_ref']) for row in payload['evidence_rows']}
    joined = ' '.join(str(row['summary']) for row in payload['evidence_rows']).lower()

    assert 'c-1' not in source_refs
    assert 'c-2' not in source_refs
    assert 'c-3' not in source_refs
    assert 'c-4' in source_refs
    assert 'available online' not in joined
    assert 'keywords:' not in joined


def test_noise_like_llm_summary_is_dropped_after_window_extraction(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', 'ABSTRACT', 'This paper investigates granular crushing in DEM simulations.', line=1),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'define_task',
                'summary': '# Abstract',
                'anchor_chunk_ids': ['c-1'],
                'confidence': 0.5,
            },
            {
                'role': 'problem',
                'act_type': 'define_task',
                'summary': 'This paper investigates granular crushing in DEM simulations.',
                'anchor_chunk_ids': ['c-1'],
                'confidence': 0.6,
            },
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    summaries = [str(row['summary']) for row in payload['evidence_rows']]

    assert summaries == ['This paper investigates granular crushing in DEM simulations.']


def test_research_move_prompt_includes_positive_and_negative_examples() -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', '1. Introduction', 'In this paper, we investigate particle recirculation in granular avalanches.', line=1),
    )
    window = {
        'role_hint': 'problem',
        'section_path': ['1. Introduction'],
        'chunks': [doc.chunks[0]],
    }

    system, user = _build_research_move_prompt(doc=doc, schema={'rules': {}}, window=window)

    lowered = f'{system}\n{user}'.lower()

    assert 'positive example' in lowered
    assert 'negative example' in lowered
    assert 'author line' in lowered
    assert 'received date' in lowered
    assert 'this paper investigates' in lowered


def test_rich_move_without_model_confidence_gets_non_zero_confidence(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', '2. Method', 'We propose a discrete element simulation workflow for crushable sands.', line=1),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': 'We propose a discrete element simulation workflow for crushable sands.',
                'anchor_chunk_ids': ['c-1'],
                'methods': [{'surface': 'discrete element simulation'}],
                'research_objects': [{'surface': 'crushable sands'}],
                'metrics': [{'surface': 'prediction accuracy'}],
                'conditions': [{'surface': 'high stress'}],
                'comparators': [{'surface': 'baseline'}],
                'effects': [{'direction': 'improve'}],
                'confidence': 0.0,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )
    trace = compile_paper_logic_trace(**{k: payload[k] for k in ['paper_metadata', 'evidence_rows', 'figure_rows', 'table_rows', 'citation_rows', 'move_relation_rows']})

    assert trace.canonical_core.moves[0].confidence
    assert trace.canonical_core.moves[0].confidence > 0.0


def test_move_relation_builder_uses_semantic_types_beyond_generic_motivates() -> None:
    rows = _build_move_relation_rows(
        [
            {'move_id': 'm-1', 'sequence_no': 1, 'role': 'problem', 'act_type': 'define_task', 'anchor_chunk_ids': ['c-1']},
            {'move_id': 'm-2', 'sequence_no': 2, 'role': 'method', 'act_type': 'propose_method', 'anchor_chunk_ids': ['c-2']},
            {'move_id': 'm-3', 'sequence_no': 3, 'role': 'result', 'act_type': 'report_effect', 'anchor_chunk_ids': ['c-3']},
            {'move_id': 'm-4', 'sequence_no': 4, 'role': 'limitation', 'act_type': 'state_limitation', 'anchor_chunk_ids': ['c-4']},
        ]
    )

    assert [row['relation_type'] for row in rows[:3]] == ['addresses', 'yields', 'limits']


def test_move_relation_builder_marks_method_sequences_as_implements() -> None:
    rows = _build_move_relation_rows(
        [
            {'move_id': 'm-1', 'sequence_no': 1, 'role': 'method', 'act_type': 'propose_method', 'anchor_chunk_ids': ['c-1']},
            {'move_id': 'm-2', 'sequence_no': 2, 'role': 'method', 'act_type': 'adapt_method', 'anchor_chunk_ids': ['c-2']},
        ]
    )

    assert rows[0]['relation_type'] == 'implements'


def test_move_relation_builder_adds_forward_link_to_next_non_motivates_target() -> None:
    rows = _build_move_relation_rows(
        [
            {'move_id': 'm-1', 'sequence_no': 1, 'role': 'problem', 'act_type': 'define_task', 'anchor_chunk_ids': ['c-1']},
            {'move_id': 'm-2', 'sequence_no': 2, 'role': 'interpretation', 'act_type': 'explain_mechanism', 'anchor_chunk_ids': ['c-2']},
            {'move_id': 'm-3', 'sequence_no': 3, 'role': 'method', 'act_type': 'propose_method', 'anchor_chunk_ids': ['c-3']},
            {'move_id': 'm-4', 'sequence_no': 4, 'role': 'result', 'act_type': 'report_effect', 'anchor_chunk_ids': ['c-4']},
        ]
    )

    relation_pairs = {
        (row['source_move_id'], row['target_move_id']): row['relation_type']
        for row in rows
    }

    assert relation_pairs[('m-1', 'm-2')] == 'motivates'
    assert relation_pairs[('m-1', 'm-3')] == 'addresses'
    assert relation_pairs[('m-3', 'm-4')] == 'yields'
