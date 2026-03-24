from __future__ import annotations

from app.ingest.models import Chunk, DocumentIR, MdSpan, PaperDraft
from app.paper_logic_trace.compiler import compile_paper_logic_trace
from app.paper_logic_trace.direct_extraction import (
    _build_move_relation_rows,
    _build_research_move_prompt,
    _role_for_chunk,
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


def test_intro_in_this_study_chunk_is_promoted_to_problem_role(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', '1. Introduction', 'In this study, we combine X-ray microtomography and DEM to investigate particle packing under compression.', line=1),
        _chunk('c-2', '2. Method', 'A coupled microtomography-DEM workflow is constructed for the compression stages.', line=2),
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


def test_intro_work_presented_aims_to_chunk_is_promoted_to_problem_role(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', 'ABSTRACT', 'The work presented here aims to quantify how irregular particle shape affects yielding and normal compression.', line=1),
        _chunk('c-2', '2. Method', 'Irregular particles are introduced into the DEM packing workflow.', line=2),
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


def test_intro_outlines_technique_to_investigate_chunk_is_promoted_to_problem_role(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            'This paper outlines a novel technique, based on X-ray microtomography and DEM, to investigate randomly packed particles during powder compaction.',
            line=1,
        ),
        _chunk('c-2', '2. Method', 'The coupled workflow reconstructs packed particle systems from XMT data.', line=2),
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


def test_intro_work_presented_utilises_model_and_aims_to_chunk_is_promoted_to_problem_role(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            'The work presented here utilises the same crushing model, but aims to make the next step by introducing irregular particle shape.',
            line=1,
        ),
        _chunk('c-2', '2. Method', 'Irregular particles are represented as clumps in the DEM model.', line=2),
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


def test_role_for_chunk_promotes_outlines_technique_to_investigate_intro_to_problem() -> None:
    role = _role_for_chunk(
        _chunk(
            'c-1',
            '1. Introduction',
            'This paper outlines a novel technique, based on X-ray microtomography and DEM, to investigate randomly packed particles during powder compaction.',
            line=1,
        ),
        paper_title='Demo Paper',
    )

    assert role == 'problem'


def test_role_for_chunk_promotes_work_presented_aims_to_intro_to_problem() -> None:
    role = _role_for_chunk(
        _chunk(
            'c-1',
            '1. Introduction',
            'The work presented here utilises the same crushing model, but aims to make the next step by introducing irregular particle shape.',
            line=1,
        ),
        paper_title='Demo Paper',
    )

    assert role == 'problem'


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
    assert 'this paper outlines' in lowered
    assert 'the work presented here' in lowered
    assert 'named citations' in lowered


def test_research_move_prompt_includes_slot_extraction_examples() -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', 'ABSTRACT', 'Results show that the mixing degree is higher than the dry mixture baseline using X-ray microtomography.', line=1),
    )
    window = {
        'role_hint': 'result',
        'section_path': ['ABSTRACT'],
        'chunks': [doc.chunks[0]],
    }

    system, user = _build_research_move_prompt(doc=doc, schema={'rules': {}}, window=window)

    lowered = f'{system}\n{user}'.lower()

    assert 'metric' in lowered
    assert 'comparator' in lowered
    assert 'resource_mentions' in lowered
    assert 'limitation_types' in lowered
    assert 'mixing degree' in lowered
    assert 'x-ray microtomography' in lowered


def test_sparse_llm_move_is_augmented_with_metric_comparator_limitation_and_resource_slots(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            'ABSTRACT',
            'Results show that the mixing degree is higher than the dry mixture baseline using X-ray microtomography and high-speed camera observations, but advanced tracking techniques remain expensive because of computational cost.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'The mixing degree is higher than the dry mixture baseline using X-ray microtomography and high-speed camera observations, but advanced tracking techniques remain expensive because of computational cost.',
                'anchor_chunk_ids': ['c-1'],
                'metrics': [],
                'comparators': [],
                'limitation_types': [],
                'resource_mentions': [],
                'confidence': 0.6,
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
    move = trace.canonical_core.moves[0]

    assert any('mixing degree' in (item.normalized or item.surface).lower() for item in move.metrics)
    assert any('dry mixture' in (item.normalized or item.surface).lower() for item in move.comparators)
    assert any('x-ray microtomography' in (item.normalized or item.surface).lower() for item in move.resource_mentions)
    assert any('computational cost' in (item.normalized or item.surface).lower() or 'expensive' in (item.normalized or item.surface).lower() for item in move.limitation_types)


def test_problem_summary_does_not_gain_noisy_metric_or_limitation_slots(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            'The packing behavior of particles is of great interest in many practical applications, but a deeper understanding is still needed for this challenging problem.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'identify_gap',
                'summary': 'The packing behavior of particles is of great interest in many practical applications, but a deeper understanding is still needed for this challenging problem.',
                'anchor_chunk_ids': ['c-1'],
                'metrics': [],
                'comparators': [],
                'limitation_types': [],
                'resource_mentions': [],
                'confidence': 0.6,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )
    trace = compile_paper_logic_trace(**{k: payload[k] for k in payload if k in ['paper_metadata', 'evidence_rows', 'figure_rows', 'table_rows', 'citation_rows', 'move_relation_rows']})
    move = trace.canonical_core.moves[0]

    assert move.metrics == []
    assert move.limitation_types == []


def test_result_summary_does_not_turn_generic_challenging_problem_into_limitation(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            'Results show that packing fraction remains a challenging problem in powder compaction, while bulk density increases under stronger confinement.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'Packing fraction remains a challenging problem in powder compaction, while bulk density increases under stronger confinement.',
                'anchor_chunk_ids': ['c-1'],
                'metrics': [],
                'comparators': [],
                'limitation_types': [],
                'resource_mentions': [],
                'confidence': 0.6,
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
    move = trace.canonical_core.moves[0]

    assert move.limitation_types == []
    assert any('packing fraction' in (item.normalized or item.surface).lower() for item in move.metrics)


def test_limitation_summary_refines_generic_assumption_to_specific_phrase(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '5. Limitations',
            'Analytical micromechanical models assume a homogeneous strain field, which becomes invalid at high relative density.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'limitation',
                'act_type': 'state_limitation',
                'summary': 'Analytical micromechanical models assume a homogeneous strain field, which becomes invalid at high relative density.',
                'anchor_chunk_ids': ['c-1'],
                'limitation_types': [{'surface': 'assumption'}],
                'confidence': 0.6,
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
    move = trace.canonical_core.moves[0]
    limitations = {(item.normalized or item.surface).lower() for item in move.limitation_types}

    assert 'assumption' not in limitations
    assert any('homogeneous strain field' in item for item in limitations)


def test_result_summary_extracts_clean_resource_mentions_without_metric_prefix(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            'Bulk density X-ray microtomography measurements agree with the optical microscopy observations during the compaction test.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'Bulk density X-ray microtomography measurements agree with the optical microscopy observations during the compaction test.',
                'anchor_chunk_ids': ['c-1'],
                'metrics': [],
                'comparators': [],
                'limitation_types': [],
                'resource_mentions': [],
                'confidence': 0.6,
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
    move = trace.canonical_core.moves[0]
    resources = {(item.normalized or item.surface).lower() for item in move.resource_mentions}

    assert any('microtomography' in item for item in resources)
    assert 'optical microscopy' in resources
    assert not any('density' in item for item in resources)


def test_result_summary_filters_citation_theory_and_application_resource_noise(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            'X-ray microtomography reproduces the packing structure, while prior cone penetrometer testing and Weibull statistics are only discussed as background context.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'X-ray microtomography reproduces the packing structure, while prior cone penetrometer testing and Weibull statistics are only discussed as background context.',
                'anchor_chunk_ids': ['c-1'],
                'resource_mentions': [
                    {'surface': 'X-ray microtomography', 'type': 'instrument'},
                    {'surface': 'McDowell and Bolton [7]', 'type': 'citation'},
                    {'surface': "Weibull's statistical distribution", 'type': 'theory'},
                    {'surface': 'cone penetrometer testing', 'type': 'application'},
                ],
                'confidence': 0.6,
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
    move = trace.canonical_core.moves[0]
    resources = {(item.normalized or item.surface).lower() for item in move.resource_mentions}

    assert 'x-ray microtomography' in resources
    assert not any('mcdowell' in item for item in resources)
    assert not any('weibull' in item for item in resources)
    assert not any('cone penetrometer' in item for item in resources)


def test_method_summary_filters_author_year_and_generic_class_resource_noise(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            'The YADE-Open DEM software can be run from the command line, while prior Gray & Thornton (2005) references and internal BodyState classes are only explanatory context.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': 'The YADE-Open DEM software can be run from the command line, while prior Gray & Thornton (2005) references and internal BodyState classes are only explanatory context.',
                'anchor_chunk_ids': ['c-1'],
                'resource_mentions': [
                    {'surface': 'YADE-Open DEM software', 'type': 'software'},
                    {'surface': 'Gray & Thornton (2005)'},
                    {'surface': 'BodyState'},
                    {'surface': 'http://yade.wikia.com'},
                ],
                'confidence': 0.6,
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
    move = trace.canonical_core.moves[0]
    resources = {(item.normalized or item.surface).lower() for item in move.resource_mentions}

    assert 'yade-open dem software' in resources
    assert not any('gray & thornton' in item for item in resources)
    assert 'bodystate' not in resources
    assert not any('http://' in item or 'https://' in item for item in resources)


def test_affiliation_summary_with_email_is_filtered_after_llm_extraction(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            'Preface',
            'LMGC UMR CNRS 5508, Universite Montpellier 2, Place Eugene Bataillon 34095 Montpellier Cedex 5, France richefeu@example.edu',
            line=1,
        ),
        _chunk(
            'c-2',
            '1. Introduction',
            'This paper investigates the micromechanics of wet granular materials.',
            line=2,
        ),
    )

    def _fake_extract(**kwargs):
        first_chunk_id = kwargs['window']['chunks'][0].chunk_id
        if first_chunk_id == 'c-1':
            return [
                {
                    'role': 'interpretation',
                    'act_type': 'explain_mechanism',
                    'summary': 'LMGC UMR CNRS 5508, Universite Montpellier 2, Place Eugene Bataillon 34095 Montpellier Cedex 5, France richefeu@example.edu',
                    'anchor_chunk_ids': ['c-1'],
                    'confidence': 0.6,
                }
            ]
        return [
            {
                'role': 'problem',
                'act_type': 'define_task',
                'summary': 'This paper investigates the micromechanics of wet granular materials.',
                'anchor_chunk_ids': ['c-2'],
                'research_objects': [{'surface': 'wet granular materials'}],
                'confidence': 0.7,
            }
        ]

    monkeypatch.setattr('app.paper_logic_trace.direct_extraction._extract_window_moves_llm', _fake_extract)

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )
    trace = compile_paper_logic_trace(**{k: payload[k] for k in ['paper_metadata', 'evidence_rows', 'figure_rows', 'table_rows', 'citation_rows', 'move_relation_rows']})
    summaries = [move.summary for move in trace.canonical_core.moves]

    assert len(summaries) == 1
    assert 'Montpellier' not in summaries[0]


def test_result_summary_extracts_comparators_from_agreement_between_sources(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            'Experiments and numerical simulations are in good agreement for the unsaturated soil response.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'Experiments and numerical simulations are in good agreement for the unsaturated soil response.',
                'anchor_chunk_ids': ['c-1'],
                'comparators': [],
                'confidence': 0.6,
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
    move = trace.canonical_core.moves[0]
    comparators = {(item.normalized or item.surface).lower() for item in move.comparators}

    assert 'experiments' in comparators
    assert 'numerical simulations' in comparators


def test_result_summary_extracts_comparator_from_similar_to_phrase(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            'The medium and dense samples have average particle stresses similar to the loose sample during loading.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'The medium and dense samples have average particle stresses similar to the loose sample during loading.',
                'anchor_chunk_ids': ['c-1'],
                'comparators': [],
                'confidence': 0.6,
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
    move = trace.canonical_core.moves[0]
    comparators = {(item.normalized or item.surface).lower() for item in move.comparators}

    assert 'the loose sample' in comparators or 'loose sample' in comparators


def test_result_summary_extracts_comparator_from_compared_to_that_for_phrase(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            'The normal compression lines for the clumps at different initial densities are examined and compared to that for spheres.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'The normal compression lines for the clumps at different initial densities are examined and compared to that for spheres.',
                'anchor_chunk_ids': ['c-1'],
                'comparators': [],
                'confidence': 0.6,
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
    move = trace.canonical_core.moves[0]
    comparators = {(item.normalized or item.surface).lower() for item in move.comparators}

    assert 'spheres' in comparators or 'for spheres' in comparators
    assert 'that' not in comparators


def test_result_summary_extracts_comparator_from_compare_with_phrase(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            'The workflow makes it possible to compare experimental results with numerical simulations using discrete elements.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'The workflow makes it possible to compare experimental results with numerical simulations using discrete elements.',
                'anchor_chunk_ids': ['c-1'],
                'comparators': [],
                'confidence': 0.6,
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
    move = trace.canonical_core.moves[0]
    comparators = {(item.normalized or item.surface).lower() for item in move.comparators}

    assert 'experimental results' in comparators
    assert 'numerical simulations' in comparators


def test_result_summary_extracts_comparator_from_agreement_with_phrase(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            'The compression law demonstrates agreement with experimental results for granular soil.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'The compression law demonstrates agreement with experimental results for granular soil.',
                'anchor_chunk_ids': ['c-1'],
                'comparators': [],
                'confidence': 0.6,
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
    move = trace.canonical_core.moves[0]
    comparators = {(item.normalized or item.surface).lower() for item in move.comparators}

    assert 'experimental results' in comparators


def test_result_summary_filters_author_only_comparator_noise(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            "This is also in agreement with the authors' compression law for granular soil.",
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': "This is also in agreement with the authors' compression law for granular soil.",
                'anchor_chunk_ids': ['c-1'],
                'comparators': [{'surface': 'authors'}],
                'confidence': 0.6,
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
    move = trace.canonical_core.moves[0]
    comparators = {(item.normalized or item.surface).lower() for item in move.comparators}

    assert 'authors' not in comparators


def test_result_summary_prefers_specific_comparator_phrase_over_single_word_noise(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            'Good agreement is found between experiments and numerical simulations.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'Good agreement is found between experiments and numerical simulations.',
                'anchor_chunk_ids': ['c-1'],
                'comparators': [{'surface': 'numerical'}],
                'confidence': 0.6,
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
    move = trace.canonical_core.moves[0]
    comparators = {(item.normalized or item.surface).lower() for item in move.comparators}

    assert 'numerical simulations' in comparators
    assert 'numerical' not in comparators


def test_result_summary_filters_noisy_model_comparator_when_heuristic_target_exists(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            'Particles with fewer contacts experience much higher maximum stresses and stress variability compared to those with more contacts.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'Particles with fewer contacts experience much higher maximum stresses and stress variability compared to those with more contacts.',
                'anchor_chunk_ids': ['c-1'],
                'comparators': [{'surface': 'experienced by particles'}],
                'confidence': 0.6,
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
    move = trace.canonical_core.moves[0]
    comparators = {(item.normalized or item.surface).lower() for item in move.comparators}

    assert 'those with more contacts' in comparators
    assert 'experienced by particles' not in comparators


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
