from __future__ import annotations

from dataclasses import replace

from app.ingest.models import Chunk, DocumentIR, MdSpan, PaperDraft
from app.paper_logic_trace.compiler import compile_paper_logic_trace
from app.paper_logic_trace.direct_extraction import (
    _build_move_relation_rows,
    _build_research_move_prompt,
    _clean_summary_candidate,
    _method_mentions_from_text,
    _refine_limitation_rows,
    _role_for_chunk,
    _summary_from_role_text,
    _trim_to_first_method_cue,
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


def test_chinese_method_summary_is_promoted_from_problem_to_method_role(monkeypatch) -> None:
    summary = '本文采用ANSYS Fluent对不同项目的各类池体进行了CFD模拟，考察了池型结构、柱网尺寸以及搅拌设备布置对流场的影响。'
    doc = _doc_with_chunks(
        _chunk('c-1', '1 控制方程与数学模型', summary, line=1),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'define_task',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    trace = compile_paper_logic_trace(
        **{k: payload[k] for k in ['paper_metadata', 'evidence_rows', 'figure_rows', 'table_rows', 'citation_rows', 'move_relation_rows']}
    )
    move = trace.canonical_core.moves[0]

    assert move.role == 'method'
    assert move.act_type == 'propose_method'


def test_build_paper_logic_trace_inputs_preserves_metadata_audit_fields(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', 'Demo Paper', 'This paper investigates powder compaction standards.', line=1),
    )
    doc = replace(
        doc,
        paper=replace(
            doc.paper,
            title='城镇污水处理厂污染物排放标准 浅释',
            title_alt='1.2《污水综合排放标准》不适应污水处理厂建设管理需求',
            metadata_enrichment={
                'mode': 'skipped_unreliable_title_match',
                'used_crossref': False,
                'local_fallback_used': True,
                'local_fallback_changed_fields': ['authors', 'title', 'title_alt'],
            },
        ),
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

    assert payload['paper_metadata']['title_alt'] == '1.2《污水综合排放标准》不适应污水处理厂建设管理需求'
    assert payload['paper_metadata']['metadata_enrichment']['mode'] == 'skipped_unreliable_title_match'


def test_refine_limitation_rows_rewrites_indicating_limitation_to_specific_scope() -> None:
    rows = [{'surface': 'indicating limitation', 'normalized': 'indicating limitation', 'inferred': True}]

    refined = _refine_limitation_rows(
        rows,
        text=(
            'Unfortunately, this rotating disk could not accommodate such highly viscous fluid. '
            'The dragged film was so thick that the open windows were jammed, indicating a limitation of the classical setup.'
        ),
    )

    assert [row['normalized'] for row in refined] == ['limitation of classical setup']


def test_refine_limitation_rows_rewrites_bare_difficult_to_difficult_for_clause() -> None:
    rows = [{'surface': 'difficult', 'normalized': 'difficult', 'inferred': True}]

    refined = _refine_limitation_rows(
        rows,
        text=(
            'If there is no wall-bounded film on the spoke, it is difficult for the free film to stabilize '
            'within the following window.'
        ),
    )

    assert [row['normalized'] for row in refined] == ['difficult for free film to stabilize']


def test_build_paper_logic_trace_inputs_repairs_suspicious_local_title_before_export(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            'This work extends a data-driven strategy to nonlinear inelasticity with internal variables.',
            line=1,
        ),
    )
    doc = replace(
        doc,
        paper=replace(
            doc.paper,
            title='2.1. Non-isothermal elasto-visco-plastic behavior',
            title_alt='Data-Driven Computational Plasticity',
            metadata_enrichment={},
        ),
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

    assert payload['paper_metadata']['title'] == 'Data-Driven Computational Plasticity'
    assert payload['paper_metadata']['title_alt'] is None
    assert payload['paper_metadata']['metadata_enrichment']['local_fallback_used'] is True
    assert sorted(payload['paper_metadata']['metadata_enrichment']['local_fallback_changed_fields']) == ['title', 'title_alt']


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


def test_build_inputs_uses_schema_paper_type_when_doc_type_missing(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', '1. Introduction', 'We investigate particle crushing prediction under high stress conditions.', line=1),
        _chunk('c-2', '2. Method', 'We propose a discrete element simulation workflow.', line=2),
    )
    doc = DocumentIR(
        paper=PaperDraft(
            paper_source=doc.paper.paper_source,
            md_path=doc.paper.md_path,
            title=doc.paper.title,
            title_alt=doc.paper.title_alt,
            authors=doc.paper.authors,
            doi=doc.paper.doi,
            year=doc.paper.year,
            paper_type=None,
        ),
        chunks=doc.chunks,
        references=doc.references,
        citations=doc.citations,
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'paper_type': 'research', 'rules': {}},
    )

    assert payload['paper_metadata']['paper_type'] == 'empirical'


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


def test_fallback_background_window_stabilizes_to_method_using_later_support_sentence(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            'Background',
            (
                'Research on blade-gap effects in mixer flow fields remains limited. '
                'The governing mechanism is still unclear. '
                'We then use the finite element method to solve the representative flow problem.'
            ),
            line=1,
        ),
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

    move = payload['evidence_rows'][0]
    normalized_methods = {str(item.get('normalized') or '').lower() for item in move['methods']}

    assert move['role_hint'] == 'method'
    assert move['act_hint'] == 'propose_method'
    assert 'finite element method' in normalized_methods
    assert 'finite element method' in move['summary'].lower()


def test_fallback_chinese_conclusion_section_promotes_to_result_role(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4 结论',
            '随着叶片间隙的增加，有效搅拌比先增加后减小，当Tip = 6 mm时有效搅拌比最大。',
            line=1,
        ),
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

    move = payload['evidence_rows'][0]

    assert move['role_hint'] == 'result'
    assert move['act_hint'] == 'report_effect'


def test_fallback_explicit_challenge_sentence_stays_problem_despite_method_like_terms(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            'Background',
            'The biggest challenge is whether simulation can proceed directly from data without a constitutive model.',
            line=1,
        ),
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

    move = payload['evidence_rows'][0]

    assert move['role_hint'] == 'problem'
    assert move['act_hint'] in {'identify_gap', 'define_task'}


def test_fallback_method_summary_strips_leading_keyword_prefix(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            'Background',
            'Key words: mixer; blade gap; CFX; effective mixing ratio We use the finite element method to analyze blade-gap flow.',
            line=1,
        ),
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

    move = payload['evidence_rows'][0]

    assert move['role_hint'] == 'method'
    assert 'finite element method' in move['summary'].lower()
    assert 'key words:' not in move['summary'].lower()


def test_summary_from_role_text_background_skips_leading_front_matter() -> None:
    text = (
        'John Doe, Jane Smith. '
        'Department of Engineering, Demo University, City 12345. '
        'Abstract: Particle mixing under variable blade gaps is studied.'
    )

    summary = _summary_from_role_text(text, role='background')

    assert 'particle mixing under variable blade gaps is studied' in summary.lower()
    assert 'john doe' not in summary.lower()
    assert 'demo university' not in summary.lower()


def test_summary_from_role_text_method_prefers_explicit_software_use_sentence() -> None:
    text = (
        'With the development of CFD technology, numerical simulation has become an important tool. '
        'We use CFX software to simulate blade-gap flow.'
    )

    summary = _summary_from_role_text(text, role='method')

    assert 'use cfx software' in summary.lower()
    assert 'cfd technology' not in summary.lower()
    assert 'numerical simulation has become an important tool' not in summary.lower()


def test_summary_from_role_text_method_prefers_named_method_sentence_over_generic_descriptor() -> None:
    text = (
        'The less intrusive approach from the point of view of implementation in standard simulation software is considered. '
        'Another approach based on the direct use of data was successfully proposed in the LaTIn method.'
    )

    summary = _summary_from_role_text(text, role='method')

    assert 'latin method' in summary.lower()
    assert 'less intrusive approach' not in summary.lower()


def test_clean_summary_candidate_removes_embedded_chunk_markers() -> None:
    text = (
        'The method is evaluated. '
        '[demo-paper:137-138:10df73f44ca346df] Simulations end up in jammed states.'
    )

    cleaned = _clean_summary_candidate(text)

    assert '[' not in cleaned
    assert '10df73f44ca346df' not in cleaned


def test_clean_summary_candidate_drops_leading_text_before_abstract_marker() -> None:
    text = (
        'John Doe, Jane Smith (Demo University, City 12345) '
        'Abstract: Particle mixing under variable blade gaps is studied.'
    )

    cleaned = _clean_summary_candidate(text)

    assert cleaned.lower().startswith('particle mixing under variable blade gaps is studied')
    assert 'john doe' not in cleaned.lower()


def test_clean_summary_candidate_drops_original_paper_prefix_before_content_sentence() -> None:
    text = (
        'ORIGINAL PAPER John Doe and Jane Smith Springer-Verlag 2019 '
        'The phenomenon of shear-induced jamming is a factor in dense suspensions.'
    )

    cleaned = _clean_summary_candidate(text)

    assert cleaned.lower().startswith('the phenomenon of shear-induced jamming')
    assert 'original paper' not in cleaned.lower()


def test_clean_summary_candidate_drops_leading_article_number_front_matter() -> None:
    text = (
        '文章编号：1000-4750(2012)09-0359-04 文章编号：1000-4750(2012)09-0359-04 '
        '利用静态松弛试验确定松弛模量的方法存在一定的问题。'
    )

    cleaned = _clean_summary_candidate(text)

    assert '文章编号' not in cleaned
    assert cleaned.startswith('利用静态松弛试验确定松弛模量的方法存在一定的问题')


def test_summary_from_role_text_method_prefers_explicit_chinese_method_sentence_over_broad_context() -> None:
    text = (
        '文章编号：1000-4750(2012)09-0359-04 建立准确的粘弹性本构方程至关重要。 '
        '利用静态松弛试验确定松弛模量的方法存在一定的问题。 '
        '针对这一问题，在Joonas Sorvari所做工作的基础上，提出一种改进方法，称之为改进型Sorvari法。'
    )

    summary = _summary_from_role_text(text, role='method')

    assert '改进型Sorvari法' in summary
    assert '文章编号' not in summary
    assert '建立准确的粘弹性本构方程' not in summary


def test_trim_to_first_method_cue_does_not_match_inside_causes_word() -> None:
    text = (
        'Deformability of particles also causes a similar stress dependence; '
        'contact deformation can enhance tangential constraints.'
    )

    trimmed = _trim_to_first_method_cue(text)

    assert trimmed == text
    assert not trimmed.startswith('uses a similar stress dependence')


def test_summary_from_role_text_method_prefers_current_work_sentence_over_prior_using_clause() -> None:
    text = (
        'Gadala-Maria and Acrivos performed shear reversal tests using a rate-controlled setup. '
        'Here, we simulate a stress-controlled shear reversal test.'
    )

    summary = _summary_from_role_text(text, role='method')

    assert 'we simulate a stress-controlled shear reversal test' in summary.lower()
    assert 'using a rate-controlled setup' not in summary.lower()


def test_summary_from_role_text_method_penalizes_prior_work_author_sentence_with_citation() -> None:
    text = (
        'To understand the roles of shear-induced structure, Gadala-Maria and Acrivos [31] '
        'performed shear reversal tests using a rate-controlled setup. '
        'Here, we simulate a stress-controlled shear reversal test.'
    )

    summary = _summary_from_role_text(text, role='method')

    assert 'we simulate a stress-controlled shear reversal test' in summary.lower()
    assert 'using a rate-controlled setup' not in summary.lower()


def test_method_mentions_from_text_ignores_inline_chunk_markers() -> None:
    text = (
        '[1607_Shear_jamming_and_fragility_in_dense_suspensions:137-138:10df73f44ca346df] '
        'simulations for phi >= 0.77 indeed end up in jammed states.'
    )

    assert _method_mentions_from_text(text, limit=3) == []


def test_theoretical_protocol_section_promotes_formalism_into_method_layer(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            'Glass preparation protocol',
            (
                "The 'state following' formalism is designed to describe this regime in which "
                'a typical equilibrium configuration selects a long-lived glass basin, which is then '
                'adiabatically followed upon increasing the density and applying a shear strain.'
            ),
            line=1,
        ),
        _chunk(
            'c-2',
            'Results',
            'We find that the glass generically yields at a finite shear strain and jams at higher densities.',
            line=2,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'background',
                'act_type': 'build_resource',
                'summary': (
                    "The 'state following' formalism is designed to describe this regime in which "
                    'a typical equilibrium configuration selects a long-lived glass basin, which is then '
                    'adiabatically followed upon increasing the density and applying a shear strain.'
                ),
                'resource_mentions': [
                    {
                        'surface': 'state following formalism',
                        'normalized': 'state following formalism',
                        'type': 'theoretical_framework',
                    }
                ],
                'anchor_chunk_ids': ['c-1'],
                'confidence': 0.9,
            },
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'We find that the glass generically yields at a finite shear strain and jams at higher densities.',
                'anchor_chunk_ids': ['c-2'],
                'confidence': 0.9,
            },
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'paper_type': 'theoretical', 'rules': {}},
    )

    trace = compile_paper_logic_trace(
        **{
            k: payload[k]
            for k in ['paper_metadata', 'evidence_rows', 'figure_rows', 'table_rows', 'citation_rows', 'move_relation_rows']
        }
    )

    assert 'state following formalism' in trace.derived_views['route_state_seed']['dominant_method_candidates']
    assert any(move.role == 'method' for move in trace.canonical_core.moves)


def test_chinese_validation_section_recovers_key_findings_after_calculation_section(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2.1 混合比计算',
            '本文提出了一种改进算法，用于计算推进剂加注混合比。',
            line=1,
        ),
        _chunk(
            'c-2',
            '3 改进算法验证与比较',
            (
                '计算结果表明，按照改进算法得到的混合比分配推进剂，会剩余燃烧剂0.96kg，'
                '按照原算法得到的混合比分配推进剂会剩余氧化剂2.83kg，因此改进算法相对原算法节约了推进剂共1.87kg。'
            ),
            line=2,
        ),
    )

    def _fake_extract_window_moves_llm(*, window, **kwargs):
        section_path = ' > '.join(window.get('section_path') or [])
        if '混合比计算' in section_path:
            return [
                {
                    'role': 'method',
                    'act_type': 'propose_method',
                    'summary': '本文提出了一种改进算法，用于计算推进剂加注混合比。',
                    'anchor_chunk_ids': ['c-1'],
                    'methods': [{'surface': '改进算法'}],
                    'confidence': 0.8,
                }
            ]
        if '验证与比较' in section_path:
            return [
                {
                    'role': 'result',
                    'act_type': 'report_effect',
                    'summary': (
                        '计算结果表明，按照改进算法得到的混合比分配推进剂，会剩余燃烧剂0.96kg，'
                        '按照原算法得到的混合比分配推进剂会剩余氧化剂2.83kg，因此改进算法相对原算法节约了推进剂共1.87kg。'
                    ),
                    'anchor_chunk_ids': ['c-2'],
                    'confidence': 0.8,
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

    trace = compile_paper_logic_trace(
        **{
            k: payload[k]
            for k in ['paper_metadata', 'evidence_rows', 'figure_rows', 'table_rows', 'citation_rows', 'move_relation_rows']
        }
    )

    assert any(move.role == 'result' for move in trace.canonical_core.moves)
    assert trace.derived_views['paper_content_profile']['key_findings']


def test_fallback_background_results_window_with_generic_simulations_text_stays_non_method(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '3.2 Features of shear-jammed states',
            'Fig. 3 The larger is phi, the smaller is the average strain to reach a jammed state. '
            'Only jammed results of ten simulations are plotted.',
            line=1,
        ),
        _chunk(
            'c-2',
            '3.2 Features of shear-jammed states',
            'These jammed states indeed end up in jammed states, but require larger strain for lower phi in the simulations.',
            line=2,
        ),
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

    assert all(row['role_hint'] != 'method' for row in payload['evidence_rows'])


def test_fallback_background_prior_model_using_clause_stays_non_method(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            'Background',
            'Many common complex fluids flow only when the applied stress exceeds a critical value.',
            line=1,
        ),
        _chunk(
            'c-2',
            'Background',
            'It is common to model the response of such materials to external stress using continuous functions '
            'of shear rate, such as appear in Bingham and Herschel-Bulkley models [1].',
            line=2,
        ),
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

    assert all(row['role_hint'] != 'method' for row in payload['evidence_rows'])


def test_bom_only_chunk_does_not_produce_fallback_move(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', None, '\ufeff', line=1),
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

    assert payload['evidence_rows'] == []


def test_fallback_method_summary_reselection_ignores_chunk_markers_and_prior_using_clause(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            'Gadala-Maria and Acrivos performed shear reversal tests using a rate-controlled setup.',
            line=1,
        ),
        _chunk(
            'c-2',
            '2. Method',
            'Here, we simulate a stress-controlled shear reversal test.',
            line=2,
        ),
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

    method_summaries = {
        row['summary'].lower()
        for row in payload['evidence_rows']
        if row['role_hint'] == 'method'
    }

    assert any('we simulate a stress-controlled shear reversal test' in summary for summary in method_summaries)
    assert all('using a rate-controlled setup' not in summary for summary in method_summaries)


def test_fallback_method_summary_can_use_later_explicit_method_chunk(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', 'Background', 'Key words: mixer; blade gap; CFX.', line=1),
        _chunk(
            'c-2',
            'Background',
            'With the development of CFD technology, numerical simulation has become an important tool.',
            line=2,
        ),
        _chunk(
            'c-3',
            'Background',
            'We use CFX software to simulate blade-gap flow and compare different blade gaps.',
            line=3,
        ),
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

    move = payload['evidence_rows'][0]

    assert move['role_hint'] == 'method'
    assert 'use cfx software' in move['summary'].lower()
    assert 'cfd technology' not in move['summary'].lower()


def test_fallback_background_window_with_late_method_context_stays_problem(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            'Background',
            'Research on fragile configurations remains limited.',
            line=1,
        ),
        _chunk(
            'c-2',
            'Background',
            'The governing mechanism is still unclear.',
            line=2,
        ),
        _chunk(
            'c-3',
            'Background',
            'It is necessary to develop continuum models based on particle-scale physics.',
            line=3,
        ),
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

    move = payload['evidence_rows'][0]

    assert move['role_hint'] == 'problem'
    assert move['act_hint'] in {'identify_gap', 'define_task'}


def test_fallback_interpretation_clause_consistent_with_model_stays_non_method(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Discussion',
            'It is consistent with a Herschel-Bulkley model of viscosity.',
            line=1,
        ),
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

    move = payload['evidence_rows'][0]

    assert move['role_hint'] != 'method'
    assert move['act_hint'] != 'propose_method'
    assert not move['methods']


def test_fallback_theory_clause_using_symbol_stays_non_method(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '3. Theory',
            'uses $\\Phi$ to decline.',
            line=1,
        ),
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

    move = payload['evidence_rows'][0]

    assert move['role_hint'] != 'method'
    assert move['act_hint'] != 'propose_method'
    assert not move['methods']


def test_fallback_theory_clause_with_causes_word_stays_non_method(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '3. Theory',
            'The temperature determined by the solution to Eq. (2.2) varies as quenching causes $\\Phi$ to decline.',
            line=1,
        ),
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

    move = payload['evidence_rows'][0]

    assert move['role_hint'] != 'method'
    assert move['act_hint'] != 'propose_method'
    assert not move['methods']


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


def test_problem_move_without_research_objects_is_backfilled_from_summary(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            'Granular avalanches are dense shallow flows of grains down an incline, and particle-size segregation remains a central challenge.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'identify_gap',
                'summary': 'Granular avalanches are dense shallow flows of grains down an incline, and particle-size segregation remains a central challenge.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    first = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in first['research_objects']}

    assert normalized
    assert 'granular avalanches' in normalized or 'particle-size segregation' in normalized


def test_problem_research_object_backfill_prefers_constitutive_model_over_goal_clause(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            'The paper identifies the challenge of whether simulation can proceed directly from data without needing a mathematical constitutive model.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'identify_gap',
                'summary': 'The paper identifies the challenge of whether simulation can proceed directly from data without needing a mathematical constitutive model.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    first = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in first['research_objects']}

    assert 'mathematical constitutive model' in normalized or 'constitutive model' in normalized
    assert 'identifies the challenge of whether simulation' not in normalized


def test_problem_research_object_backfill_handles_circumventing_constitutive_expression_phrase(monkeypatch) -> None:
    summary = (
        'The biggest challenge could then be formulated as follows: can simulation proceed directly from data by '
        'circumventing the necessity of establishing a mathematical expression of the constitutive model?'
    )
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'identify_gap',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    first = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in first['research_objects']}

    assert 'mathematical constitutive model' in normalized or 'constitutive model' in normalized
    assert all('challenge could then be formulated' not in item for item in normalized)


def test_problem_research_object_scope_phrase_is_promoted_to_trusted_normalized_signal(monkeypatch) -> None:
    summary = (
        'The biggest challenge could then be formulated as follows: can simulation proceed directly from data by '
        'circumventing the necessity of establishing a mathematical expression of the constitutive model?'
    )
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'identify_gap',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    first = payload['evidence_rows'][0]
    research_objects = first['research_objects']
    constitutive_index = next(
        index
        for index, item in enumerate(research_objects)
        if 'constitutive model' in str(item.get('normalized') or '').lower()
    )
    constitutive_object = research_objects[constitutive_index]
    constitutive_provenance = next(
        row
        for row in first['slot_provenance']
        if row['field'] == 'research_objects' and row['value_index'] == constitutive_index
    )

    assert constitutive_object['inferred'] is False
    assert constitutive_provenance['extraction_mode'] == 'normalized'
    assert constitutive_provenance['support_strength'] == 'strong'


def test_backfilled_research_object_provenance_is_marked_inferred_and_weak(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            'Granular avalanches are dense shallow flows of grains down an incline, and particle-size segregation remains a central challenge.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'identify_gap',
                'summary': 'Granular avalanches are dense shallow flows of grains down an incline, and particle-size segregation remains a central challenge.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    first = payload['evidence_rows'][0]
    research_objects = first['research_objects']
    provenance_rows = [row for row in first['slot_provenance'] if row['field'] == 'research_objects']

    assert research_objects
    assert all(item['inferred'] is True for item in research_objects)
    assert provenance_rows
    assert all(row['extraction_mode'] == 'inferred' for row in provenance_rows)
    assert all(row['support_strength'] == 'weak' for row in provenance_rows)


def test_result_move_without_research_objects_is_backfilled_from_conclusion_scope(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '3. Conclusions',
            'This work extends a data-driven strategy from nonlinear elasticity to more complex scenarios involving internal variables.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'This work extends a data-driven strategy from nonlinear elasticity to more complex scenarios involving internal variables.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'methods': [{'surface': 'data-driven strategy'}],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    first = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in first['research_objects']}

    assert 'nonlinear elasticity' in normalized or 'internal variables' in normalized
    assert 'data-driven strategy' not in normalized


def test_result_scope_backfill_promotes_explicit_domain_objects_to_trusted_normalized_signals(monkeypatch) -> None:
    summary = 'This work extends a data-driven strategy from nonlinear elasticity to more complex scenarios involving internal variables.'
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '3. Conclusions',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'methods': [{'surface': 'data-driven strategy'}],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    first = payload['evidence_rows'][0]
    research_objects = first['research_objects']

    for expected in ('nonlinear elasticity', 'internal variables'):
        value_index = next(
            index
            for index, item in enumerate(research_objects)
            if str(item.get('normalized') or '').lower() == expected
        )
        item = research_objects[value_index]
        provenance = next(
            row
            for row in first['slot_provenance']
            if row['field'] == 'research_objects' and row['value_index'] == value_index
        )

        assert item['inferred'] is False
        assert provenance['extraction_mode'] == 'normalized'
        assert provenance['support_strength'] == 'strong'


def test_result_scope_backfill_uses_stabilized_role_for_trusted_promotion(monkeypatch) -> None:
    summary = 'This work extends a data-driven strategy from nonlinear elasticity to more complex scenarios involving internal variables.'
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '3. Conclusions',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'adapt_method',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'methods': [{'surface': 'data-driven strategy'}],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    first = payload['evidence_rows'][0]
    research_objects = first['research_objects']

    assert first['role_hint'] == 'result'
    for expected in ('nonlinear elasticity', 'internal variables'):
        value_index = next(
            index
            for index, item in enumerate(research_objects)
            if str(item.get('normalized') or '').lower() == expected
        )
        item = research_objects[value_index]
        provenance = next(
            row
            for row in first['slot_provenance']
            if row['field'] == 'research_objects' and row['value_index'] == value_index
        )

        assert item['inferred'] is False
        assert provenance['extraction_mode'] == 'normalized'
        assert provenance['support_strength'] == 'strong'


def test_result_move_merges_partial_research_objects_with_conclusion_scope_backfill(monkeypatch) -> None:
    summary = 'This work extends a data-driven strategy from nonlinear elasticity to more complex scenarios involving internal variables.'
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '3. Conclusions',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [{'surface': 'internal variables'}],
                'methods': [{'surface': 'data-driven strategy'}],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    first = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in first['research_objects']}

    assert 'internal variables' in normalized
    assert 'nonlinear elasticity' in normalized
    assert 'data-driven strategy' not in normalized


def test_method_scope_backfill_keeps_broad_context_objects_as_inferred(monkeypatch) -> None:
    summary = (
        'Standard simulation in classical mechanics uses constitutive equations calibrated from data, but complexity '
        'is increasing due to finer models and engineered materials. Data-driven simulation offers a potential '
        'change of paradigm.'
    )
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            'Abstract',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'methods': [{'surface': 'data-driven simulation'}],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    first = payload['evidence_rows'][0]
    research_objects = first['research_objects']
    engineered_index = next(
        index
        for index, item in enumerate(research_objects)
        if str(item.get('normalized') or '').lower() == 'engineered materials'
    )
    engineered_object = research_objects[engineered_index]
    engineered_provenance = next(
        row
        for row in first['slot_provenance']
        if row['field'] == 'research_objects' and row['value_index'] == engineered_index
    )

    assert engineered_object['inferred'] is True
    assert engineered_provenance['extraction_mode'] == 'inferred'
    assert engineered_provenance['support_strength'] == 'weak'


def test_method_prediction_target_is_promoted_to_trusted_normalized_signal(monkeypatch) -> None:
    summary = (
        'Image-based ML approaches, including CNNs, cGANs, and GNNs, have been used to predict mechanical field '
        'distributions in composite microstructures, achieving significant speedups.'
    )
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'adapt_method',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'methods': [
                    {'surface': 'convolutional neural networks'},
                    {'surface': 'graph neural networks'},
                ],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    first = payload['evidence_rows'][0]
    research_objects = first['research_objects']
    target_index = next(
        index
        for index, item in enumerate(research_objects)
        if str(item.get('normalized') or '').lower() == 'mechanical field distributions in composite microstructures'
    )
    target_object = research_objects[target_index]
    target_provenance = next(
        row
        for row in first['slot_provenance']
        if row['field'] == 'research_objects' and row['value_index'] == target_index
    )

    assert target_object['inferred'] is False
    assert target_provenance['extraction_mode'] == 'normalized'
    assert target_provenance['support_strength'] == 'strong'


def test_problem_prediction_target_in_composites_is_promoted_to_trusted_normalized_signal(monkeypatch) -> None:
    summary = (
        'Existing ML approaches for predicting mechanical fields in composites have significant computational costs '
        'for generating large training datasets via FEA.'
    )
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'identify_gap',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    first = payload['evidence_rows'][0]
    research_objects = first['research_objects']
    target_index = next(
        index
        for index, item in enumerate(research_objects)
        if str(item.get('normalized') or '').lower() == 'mechanical fields in composites'
    )
    target_object = research_objects[target_index]
    target_provenance = next(
        row
        for row in first['slot_provenance']
        if row['field'] == 'research_objects' and row['value_index'] == target_index
    )

    assert target_object['inferred'] is False
    assert target_provenance['extraction_mode'] == 'normalized'
    assert target_provenance['support_strength'] == 'strong'


def test_result_move_backfill_handles_addressing_domain_phrase(monkeypatch) -> None:
    summary = (
        'This work succeeded to extend the data-driven strategy proposed in our former works for addressing '
        'nonlinear elasticity to more complex scenarios involving internal variables.'
    )
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '3. Conclusions',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [{'surface': 'internal variables'}],
                'methods': [{'surface': 'data-driven strategy'}],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    first = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in first['research_objects']}

    assert 'internal variables' in normalized
    assert 'nonlinear elasticity' in normalized


def test_research_object_filter_drops_generic_solution_and_promise_phrases(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            'The proposed solution will allow direct feedback from authors and encourage the scientific community.',
            line=1,
        ),
        _chunk(
            'c-2',
            '2. Method',
            'The YADE framework provides a stable environment for implementing DEM algorithms.',
            line=2,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': 'The proposed solution will allow direct feedback from authors and encourage the scientific community. The YADE framework provides a stable environment for implementing DEM algorithms.',
                'anchor_chunk_ids': ['c-1', 'c-2'],
                'research_objects': [
                    {'surface': 'proposed solution'},
                    {'surface': 'will allow direct feedback from authors and encourage the scientific community'},
                    {'surface': 'YADE framework'},
                ],
                'methods': [{'surface': 'discrete element method'}],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in move['research_objects']}

    assert 'yade framework' in normalized
    assert 'proposed solution' not in normalized
    assert 'will allow direct feedback from authors and encourage the scientific community' not in normalized


def test_research_object_filter_drops_introduced_into_clause_when_subject_object_is_available(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            'The constitutive model is introduced into the weak form to formulate the problem in terms of displacement, which is then discretized using the finite element method.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'adapt_method',
                'summary': 'The constitutive model is introduced into the weak form to formulate the problem in terms of displacement, which is then discretized using the finite element method.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in move['research_objects']}

    assert all('introduced into the weak form' not in item for item in normalized)


def test_method_move_without_methods_backfills_particle_dynamics_simulation(monkeypatch) -> None:
    summary = 'This article examines the fragility of shear jammed states using particle dynamics simulations with idealized conditions.'
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'methods': [],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized_methods = {str(item.get('normalized') or '').lower() for item in move['methods']}

    assert 'particle dynamics simulations' in normalized_methods


def test_method_move_without_methods_backfills_finite_element_method(monkeypatch) -> None:
    summary = 'The constitutive model is introduced into the weak form to formulate the problem in terms of displacement, which is then discretized using the finite element method.'
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'adapt_method',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'methods': [],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized_methods = {str(item.get('normalized') or '').lower() for item in move['methods']}

    assert 'finite element method' in normalized_methods


def test_set_condition_move_does_not_backfill_incidental_algorithm_reference_as_method(monkeypatch) -> None:
    summary = (
        'The penalty parameters are selected to keep particle overlap below 2% of the particle radius, '
        'which is fundamentally different from hard-sphere algorithms.'
    )
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'set_condition',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'methods': [],
                'conditions': [],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]

    assert move['act_hint'] == 'set_condition'
    assert move['methods'] == []


def test_promoted_method_move_backfills_method_mentions_after_role_stabilization(monkeypatch) -> None:
    summary = 'This article examines the fragility of shear jammed states using particle dynamics simulations with idealized conditions.'
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'define_task',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'methods': [],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized_methods = {str(item.get('normalized') or '').lower() for item in move['methods']}

    assert move['role_hint'] == 'method'
    assert move['act_hint'] == 'propose_method'
    assert 'particle dynamics simulations' in normalized_methods


def test_method_backfill_filters_generic_efficient_method_fragment(monkeypatch) -> None:
    summary = (
        'We also omit Brownian motions, which are relevant for smaller particles. '
        'Stokesian Dynamics (SD) is an efficient method to reproduce particle dynamics in this Stokes regime. '
        'Recently, the SD approach was extended to be coupled with frictional contact mechanics.'
    )
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'methods': [],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized_methods = {str(item.get('normalized') or '').lower() for item in move['methods']}

    assert 'stokesian dynamics' in normalized_methods
    assert 'sd approach' in normalized_methods
    assert 'efficient method' not in normalized_methods
    assert 'smaller particles stokesian dynamics' not in normalized_methods
    assert 'reproduce particle dynamics' not in normalized_methods


def test_method_backfill_filters_focus_on_simple_model_fragment(monkeypatch) -> None:
    summary = (
        'In this article, we focus on a simple model and employ a soft-constraint approach '
        'to represent contact constraints between particles.'
    )
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'methods': [],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized_methods = {str(item.get('normalized') or '').lower() for item in move['methods']}

    assert 'soft-constraint approach' in normalized_methods
    assert 'focus on simple model' not in normalized_methods
    assert 'simple model' not in normalized_methods


def test_method_backfill_prefers_named_method_over_reporting_wrapper(monkeypatch) -> None:
    summary = (
        'Another approach based on the direct use of data was successfully proposed in the LaTIn method.'
    )
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'methods': [],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized_methods = {str(item.get('normalized') or '').lower() for item in move['methods']}

    assert 'latin method' in normalized_methods
    assert 'another approach' not in normalized_methods
    assert 'was successfully proposed in the latin method' not in normalized_methods


def test_method_backfill_filters_equation_and_law_fragments(monkeypatch) -> None:
    summary = (
        'The constitutive equation is required to close the problem, and the hardening law is illustrated in Fig. 1.'
    )
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'methods': [],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized_methods = {str(item.get('normalized') or '').lower() for item in move['methods']}

    assert 'constitutive equation' not in normalized_methods
    assert 'hardening law' not in normalized_methods


def test_method_backfill_filters_pronoun_led_generic_model_fragment(monkeypatch) -> None:
    summary = "Hooke's law is more than a law, it is simply a model."
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'methods': [],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized_methods = {str(item.get('normalized') or '').lower() for item in move['methods']}

    assert 'it is simply model' not in normalized_methods
    assert 'simply model' not in normalized_methods


def test_method_backfill_filters_reporting_simulation_count_fragments(monkeypatch) -> None:
    summary = 'Only jammed results of ten simulations are plotted.'
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'methods': [],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized_methods = {str(item.get('normalized') or '').lower() for item in move['methods']}

    assert 'only jammed results of ten simulations' not in normalized_methods
    assert 'ten simulations' not in normalized_methods


def test_method_backfill_filters_intrusive_descriptor_when_named_method_exists(monkeypatch) -> None:
    summary = (
        'The less intrusive approach from the point of view of implementation in standard simulation software '
        'is the LaTIn method.'
    )
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'methods': [],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized_methods = {str(item.get('normalized') or '').lower() for item in move['methods']}

    assert 'latin method' in normalized_methods
    assert 'intrusive' not in normalized_methods


def test_method_backfill_filters_reporting_models_clause_fragment(monkeypatch) -> None:
    summary = (
        'These simulations did not include viscous dissipation in their models [24,25], '
        'but they do report shear localization in which the shear rate is continuous.'
    )
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            summary,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'methods': [],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized_methods = {str(item.get('normalized') or '').lower() for item in move['methods']}

    assert 'these simulations' not in normalized_methods
    assert 'did not include viscous dissipation in their models' not in normalized_methods


def test_research_object_filter_trims_leading_establish_verb_from_domain_object(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            'The paper identifies the challenge of whether simulation can proceed directly from data without needing to establish mathematical constitutive models.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'identify_gap',
                'summary': 'The paper identifies the challenge of whether simulation can proceed directly from data without needing to establish mathematical constitutive models.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [{'surface': 'establish mathematical constitutive models'}],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in move['research_objects']}

    assert 'mathematical constitutive models' in normalized or 'constitutive models' in normalized
    assert 'establish mathematical constitutive models' not in normalized


def test_research_object_filter_drops_generic_parameter_singleton(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            'Advanced clustering techniques and machine learning help extract manifold structures from data to identify uncorrelated parameters and predict solutions via interpolation schemes.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'adapt_method',
                'summary': 'Advanced clustering techniques and machine learning help extract manifold structures from data to identify uncorrelated parameters and predict solutions via interpolation schemes.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [{'surface': 'parameters'}],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in move['research_objects']}

    assert 'parameters' not in normalized


def test_research_object_filter_drops_descriptive_clause_phrases(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            'The YADE framework is capable of describing the mechanical behavior of assemblies of discrete elements.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': 'The YADE framework is capable of describing the mechanical behavior of assemblies of discrete elements.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [
                    {'surface': 'YADE framework'},
                    {'surface': 'capable of describing the mechanical behavior of assemblies of discrete elements'},
                ],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in move['research_objects']}

    assert 'yade framework' in normalized
    assert 'capable of describing the mechanical behavior of assemblies of discrete elements' not in normalized


def test_research_object_filter_drops_verb_led_goal_clause(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Discussion',
            'Particle rotation behavior is studied, and the model aims to produce more realistic particle rotation behavior as exhibited in experiments.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'interpretation',
                'act_type': 'explain_mechanism',
                'summary': 'Particle rotation behavior is studied, and the model aims to produce more realistic particle rotation behavior as exhibited in experiments.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [
                    {'surface': 'particle rotation behavior'},
                    {'surface': 'produce more realistic particle rotation behavior as exhibited in experiments'},
                ],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in move['research_objects']}

    assert 'particle rotation behavior' in normalized
    assert 'produce more realistic particle rotation behavior as exhibited in experiments' not in normalized


def test_research_object_filter_drops_generic_process_and_clause_fragment(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            'The validated model was employed to evaluate the segregation of binary particle mixtures and RSM was used for analysing the DEM results of the segregation.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'The validated model was employed to evaluate the segregation of binary particle mixtures and RSM was used for analysing the DEM results of the segregation.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [
                    {'surface': 'process'},
                    {'surface': 'was employed to evaluate the segregation of binary particle mixtures and rsm was'},
                    {'surface': 'binary particle mixtures'},
                ],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in move['research_objects']}

    assert 'binary particle mixtures' in normalized
    assert 'process' not in normalized
    assert 'was employed to evaluate the segregation of binary particle mixtures and rsm was' not in normalized


def test_research_object_filter_drops_temporal_and_installed_clause_noise(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            'Particle rotation affects granular response, but prior DEM work often used rolling resistance models on spherical particles.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'define_task',
                'summary': 'Particle rotation affects granular response, but prior DEM work often used rolling resistance models on spherical particles.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [
                    {'surface': 'over the last two decades'},
                    {'surface': 'installed typically on spherical particles within the dem community'},
                    {'surface': 'simulate the behavior of granular materials'},
                    {'surface': 'micromechanical mechanisms of granular soil behavior'},
                ],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in move['research_objects']}

    assert 'micromechanical mechanisms of granular soil behavior' in normalized
    assert 'over the last two decades' not in normalized
    assert 'installed typically on spherical particles within the dem community' not in normalized
    assert 'simulate the behavior of granular materials' not in normalized


def test_research_object_filter_drops_generic_challenge_scheme_and_simple_phrases(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            'Large deformation problems arise in geotechnical structures, including landslides and debris flow.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'define_task',
                'summary': 'Large deformation problems arise in geotechnical structures, including landslides and debris flow.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [
                    {'surface': 'geotechnical structures'},
                    {'surface': 'landslides and debris flow'},
                    {'surface': 'second challenge'},
                    {'surface': 'new scheme'},
                    {'surface': 'very simple'},
                    {'surface': 'both of the fundamental challenges described above is presented'},
                    {'surface': 'verifies the conditions at the critical state'},
                ],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in move['research_objects']}

    assert 'geotechnical structures' in normalized
    assert 'landslides and debris flow' in normalized
    assert 'second challenge' not in normalized
    assert 'new scheme' not in normalized
    assert 'very simple' not in normalized
    assert 'both of the fundamental challenges described above is presented' not in normalized
    assert 'verifies the conditions at the critical state' not in normalized


def test_research_object_filter_drops_scheme_reporting_and_description_fragments(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            'A new scheme is proposed for large deformation problems and several machine learning algorithms are considered.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': 'A new scheme is proposed for large deformation problems and several machine learning algorithms are considered.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [
                    {'surface': 'large deformation problems'},
                    {'surface': 'machine learning algorithms'},
                    {'surface': 'new scheme applicable to general large deformation problems'},
                    {'surface': 'noted to be simple and appropriate'},
                    {'surface': 'results show the mass loss of the scheme'},
                    {'surface': 'describes several machine learning algorithms considered'},
                ],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in move['research_objects']}

    assert 'large deformation problems' in normalized
    assert 'machine learning algorithms' in normalized
    assert 'new scheme applicable to general large deformation problems' not in normalized
    assert 'noted to be simple and appropriate' not in normalized
    assert 'results show the mass loss of the scheme' not in normalized
    assert 'describes several machine learning algorithms considered' not in normalized


def test_research_object_filter_drops_method_like_phrases_when_methods_slot_already_captures_them(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            'The granular element method and a numerical simulation method are used to study particle dissolution in a stirred tank reactor.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': 'The granular element method and a numerical simulation method are used to study particle dissolution in a stirred tank reactor.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [
                    {'surface': 'granular element method'},
                    {'surface': 'numerical simulation method'},
                    {'surface': 'particle dissolution in a stirred tank reactor'},
                ],
                'methods': [
                    {'surface': 'granular element method'},
                    {'surface': 'numerical simulation method'},
                ],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in move['research_objects']}

    assert 'particle dissolution in a stirred tank reactor' in normalized
    assert 'granular element method' not in normalized
    assert 'numerical simulation method' not in normalized


def test_research_object_filter_drops_method_like_phrases_in_method_role_even_without_methods_slot(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            'A numerical simulation method is used to study particle dissolution in a stirred tank reactor.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': 'A numerical simulation method is used to study particle dissolution in a stirred tank reactor.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [
                    {'surface': 'numerical simulation method'},
                    {'surface': 'particle dissolution in a stirred tank reactor'},
                ],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in move['research_objects']}

    assert 'particle dissolution in a stirred tank reactor' in normalized
    assert 'numerical simulation method' not in normalized


def test_research_object_filter_drops_inferred_algorithm_fragments_in_method_role(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            'A multi-step algorithm performs surface-surface intersections, loop centroid approximation, curve-surface intersection, and overlap calculation.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': 'Describes a multi-step algorithm for contact detection and force calculation between granular elements.',
                'anchor_chunk_ids': ['c-1'],
                'methods': [
                    {'surface': 'surface-surface intersection'},
                    {'surface': 'loop centroid approximation'},
                    {'surface': 'curve-surface intersection'},
                    {'surface': 'overlap calculation'},
                ],
                'confidence': 0.7,
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
    normalized = {str(item.normalized or item.surface or '').lower() for item in move.research_objects}

    assert 'multi-step algorithm' not in normalized


def test_research_object_filter_drops_inferred_equivalence_clause_fragments(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '6. Discussion',
            'The T-Spline model is geometrically equivalent to the NURBS model, but with about half as many control points. Therefore, for complicated grain geometries, T-Splines may offer an advantage over NURBS.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'limitation',
                'act_type': 'diagnose_failure',
                'summary': 'For complicated grain geometries, NURBS surfaces can generate many superfluous control points, though T-Splines may offer an advantage by reducing them.',
                'anchor_chunk_ids': ['c-1'],
                'methods': [
                    {'surface': 'NURBS surfaces'},
                    {'surface': 'T-Splines'},
                ],
                'limitation_types': [
                    {'surface': 'superfluous control points'},
                ],
                'confidence': 0.7,
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
    normalized = {str(item.normalized or item.surface or '').lower() for item in move.research_objects}

    assert 'geometrically equivalent to the nurbs model' not in normalized


def test_sparse_slot_backfill_uses_move_local_anchor_text(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '6. Discussion',
            'Particle recirculation in granular avalanches occurs under strong shear.',
            line=1,
        ),
        _chunk(
            'c-2',
            '6. Discussion',
            'Computational cost limits high-resolution simulation for large domains.',
            line=2,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'Particle recirculation in granular avalanches occurs under strong shear.',
                'anchor_chunk_ids': ['c-1'],
                'confidence': 0.7,
            },
            {
                'role': 'limitation',
                'act_type': 'state_limitation',
                'summary': 'Computational cost limits high-resolution simulation for large domains.',
                'anchor_chunk_ids': ['c-2'],
                'limitation_types': [{'surface': 'computational cost'}],
                'confidence': 0.7,
            },
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    trace = compile_paper_logic_trace(**{k: payload[k] for k in ['paper_metadata', 'evidence_rows', 'figure_rows', 'table_rows', 'citation_rows', 'move_relation_rows']})
    limitation_move = next(move for move in trace.canonical_core.moves if move.role == 'limitation')
    normalized = {str(item.normalized or item.surface or '').lower() for item in limitation_move.research_objects}

    assert 'particle recirculation in granular avalanches' not in normalized


def test_research_object_filter_drops_reporting_verb_fragments(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            'Prior work has been conducted to quantify segregation, while newer studies investigate its role in shear localization.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'define_task',
                'summary': 'Prior work has been conducted to quantify segregation, while newer studies investigate its role in shear localization.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [
                    {'surface': 'have been conducted to quantify the segregation'},
                    {'surface': 'investigate its role in shear localization'},
                    {'surface': 'shear localization'},
                ],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in move['research_objects']}

    assert 'shear localization' in normalized
    assert 'have been conducted to quantify the segregation' not in normalized
    assert 'investigate its role in shear localization' not in normalized


def test_problem_research_object_filter_drops_data_collection_process_fragments(monkeypatch) -> None:
    summary = (
        'Very little has been done with big-data in scientific computing despite promising first attempts. '
        'The main drawback of data-driven approaches is the huge amount of data required for running simulations.'
    )
    quote = (
        'This unprecedented possibility of directly determine knowledge from data or, in other words, to extract models '
        'from experiments in a automated way, is being followed with great interest in many fields of science and '
        'engineering. In [8] authors followed a similar rationale extending the data-driven framework to nonlinear '
        'elasticity and inelasticity, where model-based simulations where replaced by data-driven simulations operating '
        'on a new kind of constitutive models defined directly from data. Its main drawback is the huge amount of data '
        'required for running simulations. In the present work we will assume that all the needed data is available. '
        'We will not address the way of collecting data from adequate experiments and the use of eventual inverse '
        'techniques to enrich the behavior description, issues that will be reported in incoming works.'
    )
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            quote,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'identify_gap',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    first = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in first['research_objects']}

    assert 'experiments in automated way' not in normalized
    assert 'defined directly from data' not in normalized
    assert all('way of collecting data from adequate experiments' not in item for item in normalized)


def test_problem_summary_with_explicit_main_drawback_gains_trusted_bottleneck_signal(monkeypatch) -> None:
    summary = (
        'Very little has been done with big-data in scientific computing despite promising first attempts. '
        'The main drawback of data-driven approaches is the huge amount of data required for running simulations.'
    )
    quote = (
        'This unprecedented possibility of directly determine knowledge from data or, in other words, to extract models '
        'from experiments in a automated way, is being followed with great interest in many fields of science and '
        'engineering. In [8] authors followed a similar rationale extending the data-driven framework to nonlinear '
        'elasticity and inelasticity, where model-based simulations where replaced by data-driven simulations operating '
        'on a new kind of constitutive models defined directly from data. Its main drawback is the huge amount of data '
        'required for running simulations. In the present work we will assume that all the needed data is available.'
    )
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            quote,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'identify_gap',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'limitation_types': [],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )
    trace = compile_paper_logic_trace(
        **{k: payload[k] for k in ['paper_metadata', 'evidence_rows', 'figure_rows', 'table_rows', 'citation_rows', 'move_relation_rows']}
    )
    move = trace.canonical_core.moves[0]
    bottlenecks = [str(item).lower() for item in ((trace.derived_views.get('route_state_seed') or {}).get('known_bottleneck_candidates') or [])]

    limitations = {str(item.normalized or item.surface or '').lower() for item in move.limitation_types}

    assert any('huge amount of data required' in item for item in limitations)
    assert any('huge amount of data required' in item for item in bottlenecks)


def test_problem_move_recovers_explicit_drawback_from_later_intro_chunk(monkeypatch) -> None:
    summary = (
        'Very little has been done with big-data in scientific computing despite promising first attempts.'
    )
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            (
                'This unprecedented possibility of directly determining knowledge from data is followed with great '
                'interest in many fields of science and engineering.'
            ),
            line=1,
        ),
        _chunk(
            'c-2',
            '1. Introduction',
            (
                'The main drawback of data-driven approaches is the huge amount of data required for running '
                'simulations. In the present work we will assume that all the needed data is available.'
            ),
            line=4,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'identify_gap',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'limitation_types': [],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )
    trace = compile_paper_logic_trace(
        **{k: payload[k] for k in ['paper_metadata', 'evidence_rows', 'figure_rows', 'table_rows', 'citation_rows', 'move_relation_rows']}
    )
    move = trace.canonical_core.moves[0]
    bottlenecks = [str(item).lower() for item in ((trace.derived_views.get('route_state_seed') or {}).get('known_bottleneck_candidates') or [])]

    limitations = {str(item.normalized or item.surface or '').lower() for item in move.limitation_types}
    move_anchor_sources = {
        str(row.get('source_ref') or '').lower()
        for row in payload['evidence_rows']
        if str(row.get('move_id') or '').strip() == move.move_id
    }

    assert any('huge amount of data required' in item for item in limitations)
    assert any('huge amount of data required' in item for item in bottlenecks)
    assert 'c-2' in move_anchor_sources


def test_problem_drawback_signal_creates_grounded_limitation_move(monkeypatch) -> None:
    summary = 'Very little has been done with big-data in scientific computing despite promising first attempts.'
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            (
                'This unprecedented possibility of directly determining knowledge from data is followed with great '
                'interest in many fields of science and engineering.'
            ),
            line=1,
        ),
        _chunk(
            'c-2',
            '1. Introduction',
            (
                'The main drawback of data-driven approaches is the huge amount of data required for running '
                'simulations. In the present work we will assume that all the needed data is available.'
            ),
            line=4,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'identify_gap',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'limitation_types': [],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )
    trace = compile_paper_logic_trace(
        **{k: payload[k] for k in ['paper_metadata', 'evidence_rows', 'figure_rows', 'table_rows', 'citation_rows', 'move_relation_rows']}
    )

    limitation_moves = [move for move in trace.canonical_core.moves if move.role == 'limitation']
    grounded_constraint_move_count = int(((trace.quality or {}).get('l2_completeness_audit') or {}).get('grounded_constraint_move_count') or 0)

    assert any(
        any('huge amount of data required' in str(item.normalized or item.surface or '').lower() for item in move.limitation_types)
        for move in limitation_moves
    )
    assert grounded_constraint_move_count >= 1


def test_problem_drawback_signal_creates_limitation_move_even_when_support_text_truncates(monkeypatch) -> None:
    summary = 'Very little has been done with big-data in scientific computing despite promising first attempts.'
    long_intro = ' '.join(['Data-driven simulation is attracting broad interest across mechanics and scientific computing.'] * 60)
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            long_intro,
            line=1,
        ),
        _chunk(
            'c-2',
            '1. Introduction',
            (
                'The main drawback of data-driven approaches is the huge amount of data required for running '
                'simulations. In the present work we will assume that all the needed data is available.'
            ),
            line=80,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'identify_gap',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'limitation_types': [],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )
    trace = compile_paper_logic_trace(
        **{k: payload[k] for k in ['paper_metadata', 'evidence_rows', 'figure_rows', 'table_rows', 'citation_rows', 'move_relation_rows']}
    )

    limitation_moves = [move for move in trace.canonical_core.moves if move.role == 'limitation']

    assert any(
        any('huge amount of data required' in str(item.normalized or item.surface or '').lower() for item in move.limitation_types)
        for move in limitation_moves
    )


def test_same_window_explicit_drawback_emits_only_one_companion_limitation_move(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            (
                'Data-driven simulation is attracting broad interest across mechanics and scientific computing. '
                'The main drawback of data-driven approaches is the huge amount of data required for running simulations.'
            ),
            line=1,
        ),
        _chunk(
            'c-2',
            '1. Introduction',
            'In the present work we will assume that all the needed data is available.',
            line=4,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'problem',
                'act_type': 'identify_gap',
                'summary': 'Data-driven simulation is attracting broad interest across mechanics and scientific computing.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'limitation_types': [],
                'confidence': 0.8,
            },
            {
                'role': 'problem',
                'act_type': 'identify_gap',
                'summary': 'Very little has been done in applying big-data to scientific computing despite promising first attempts.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'limitation_types': [],
                'confidence': 0.8,
            },
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )
    trace = compile_paper_logic_trace(
        **{k: payload[k] for k in ['paper_metadata', 'evidence_rows', 'figure_rows', 'table_rows', 'citation_rows', 'move_relation_rows']}
    )

    limitation_moves = [move for move in trace.canonical_core.moves if move.role == 'limitation']

    assert len(limitation_moves) == 1
    assert all(
        any('huge amount of data required' in str(item.normalized or item.surface or '').lower() for item in move.limitation_types)
        for move in limitation_moves
    )


def test_adjacent_redundant_method_moves_merge_into_one_richer_move(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '中文摘要',
            '针对推进剂性能预测建立广义回归神经网络与遗传算法反向传播神经网络模型，并结合配方变量进行建模与对照分析。',
            line=1,
        ),
        _chunk(
            'c-2',
            'Abstract',
            'General regression neural network and genetic algorithm back propagation neural network models are established to predict propellant mechanical performance and burning rate from formulation variables and particle-size settings.',
            line=4,
        ),
    )

    def _fake_extract_window_moves_llm(**kwargs):
        chunk_ids = [str(chunk.chunk_id) for chunk in (kwargs.get('window', {}).get('chunks') or [])]
        if chunk_ids == ['c-1']:
            return [
                {
                    'role': 'method',
                    'act_type': 'propose_method',
                    'summary': 'Two neural network models, Generalized Regression Neural Network (GRNN) and Genetic Algorithm Backpropagation (GABP) neural network, are established to predict propellant properties.',
                    'anchor_chunk_ids': ['c-1'],
                    'research_objects': [{'surface': 'propellant', 'normalized': 'propellant'}],
                    'methods': [
                        {'surface': 'Generalized Regression Neural Network (GRNN)', 'normalized': 'generalized regression neural network (grnn)'},
                        {'surface': 'Genetic Algorithm Backpropagation (GABP) neural network', 'normalized': 'genetic algorithm backpropagation (gabp) neural network'},
                    ],
                    'confidence': 0.78,
                }
            ]
        if chunk_ids == ['c-2']:
            return [
                {
                    'role': 'method',
                    'act_type': 'propose_method',
                    'summary': 'Establishes general regression neural network (GRNN) and genetic algorithm back propagation (GABP) neural network models to predict propellant mechanical performance and burning rate based on fine ammonium perchlorate content.',
                    'anchor_chunk_ids': ['c-2'],
                    'research_objects': [
                        {'surface': 'composite solid propellant', 'normalized': 'composite solid propellant'},
                        {'surface': 'mechanical performance', 'normalized': 'mechanical performance'},
                        {'surface': 'burning rate', 'normalized': 'burning rate'},
                    ],
                    'methods': [
                        {'surface': 'general regression neural network', 'normalized': 'general regression neural network'},
                        {'surface': 'genetic algorithm back propagation neural network', 'normalized': 'genetic algorithm back propagation neural network'},
                    ],
                    'confidence': 0.84,
                }
            ]
        return []

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        _fake_extract_window_moves_llm,
    )
    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._semantic_windows',
        lambda *_args, **_kwargs: [
            {
                'window_id': 'w-1',
                'role_hint': 'method',
                'act_hint': 'propose_method',
                'section_path': ['中文摘要'],
                'chunks': [doc.chunks[0]],
                'char_count': len(doc.chunks[0].text),
            },
            {
                'window_id': 'w-2',
                'role_hint': 'method',
                'act_hint': 'propose_method',
                'section_path': ['Abstract'],
                'chunks': [doc.chunks[1]],
                'char_count': len(doc.chunks[1].text),
            },
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )
    trace = compile_paper_logic_trace(
        **{k: payload[k] for k in ['paper_metadata', 'evidence_rows', 'figure_rows', 'table_rows', 'citation_rows', 'move_relation_rows']}
    )

    method_moves = [move for move in trace.canonical_core.moves if move.role == 'method']

    assert len(method_moves) == 1
    assert 'mechanical performance and burning rate' in method_moves[0].summary.lower()
    assert len(method_moves[0].anchor_ids) == 2
    normalized_methods = [str(item.normalized or item.surface or '').lower() for item in method_moves[0].methods]
    assert any('regression' in item and 'network' in item for item in normalized_methods)
    assert any('genetic' in item and 'algorithm' in item for item in normalized_methods)


def test_adjacent_method_moves_with_distinct_method_signatures_do_not_merge(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Simulation model',
            'We employ an algorithm to mimic stress-controlled rheology while extending Stokesian Dynamics under quasistatic conditions.',
            line=1,
        ),
        _chunk(
            'c-2',
            '2. Simulation model (continued)',
            'The SD approach is extended with frictional contact mechanics to reproduce discontinuous shear thickening and analyze viscosity divergence.',
            line=4,
        ),
    )

    def _fake_extract_window_moves_llm(**kwargs):
        chunk_ids = [str(chunk.chunk_id) for chunk in (kwargs.get('window', {}).get('chunks') or [])]
        if chunk_ids == ['c-1']:
            return [
                {
                    'role': 'method',
                    'act_type': 'propose_method',
                    'summary': 'The authors employ an algorithm to mimic stress-controlled rheology, extending Stokesian Dynamics by coupling it with frictional contact mechanics to reproduce discontinuous shear thickening.',
                    'anchor_chunk_ids': ['c-1'],
                    'methods': [
                        {'surface': 'stress-controlled rheology algorithm', 'normalized': 'stress-controlled rheology algorithm'},
                        {'surface': 'Stokesian Dynamics', 'normalized': 'stokesian dynamics'},
                    ],
                    'confidence': 0.8,
                }
            ]
        if chunk_ids == ['c-2']:
            return [
                {
                    'role': 'method',
                    'act_type': 'propose_method',
                    'summary': 'The SD approach is extended by coupling it with frictional contact mechanics to reproduce discontinuous shear thickening, noting that hydrodynamic contributions vanish at viscosity divergence.',
                    'anchor_chunk_ids': ['c-2'],
                    'methods': [
                        {
                            'surface': 'Stokesian Dynamics extended with frictional contact mechanics',
                            'normalized': 'stokesian dynamics extended with frictional contact mechanics',
                        }
                    ],
                    'confidence': 0.81,
                }
            ]
        return []

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        _fake_extract_window_moves_llm,
    )
    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._semantic_windows',
        lambda *_args, **_kwargs: [
            {
                'window_id': 'w-1',
                'role_hint': 'method',
                'act_hint': 'propose_method',
                'section_path': ['2. Simulation model'],
                'chunks': [doc.chunks[0]],
                'char_count': len(doc.chunks[0].text),
            },
            {
                'window_id': 'w-2',
                'role_hint': 'method',
                'act_hint': 'propose_method',
                'section_path': ['2. Simulation model (continued)'],
                'chunks': [doc.chunks[1]],
                'char_count': len(doc.chunks[1].text),
            },
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )
    trace = compile_paper_logic_trace(
        **{k: payload[k] for k in ['paper_metadata', 'evidence_rows', 'figure_rows', 'table_rows', 'citation_rows', 'move_relation_rows']}
    )

    method_moves = [move for move in trace.canonical_core.moves if move.role == 'method']

    assert len(method_moves) == 2


def test_nearby_redundant_method_moves_merge_across_interleaved_result_move(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '中文摘要',
            '建立广义回归神经网络与遗传算法反向传播神经网络模型，用于推进剂性能预测。',
            line=1,
        ),
        _chunk(
            'c-2',
            'Abstract',
            'The predicted values agree well with the measured burning rate and mechanical performance.',
            line=4,
        ),
        _chunk(
            'c-3',
            'Abstract',
            'General regression neural network and genetic algorithm back propagation neural network models are established to predict propellant burning rate and mechanical performance from formulation variables.',
            line=7,
        ),
    )

    def _fake_extract_window_moves_llm(**kwargs):
        chunk_ids = [str(chunk.chunk_id) for chunk in (kwargs.get('window', {}).get('chunks') or [])]
        if chunk_ids == ['c-1']:
            return [
                {
                    'role': 'method',
                    'act_type': 'propose_method',
                    'summary': 'Establishes general regression neural network (GRNN) and genetic algorithm back propagation (GABP) neural network models to predict propellant burning rate and mechanical performance.',
                    'anchor_chunk_ids': ['c-1'],
                    'research_objects': [{'surface': 'propellant', 'normalized': 'propellant'}],
                    'methods': [
                        {'surface': 'GRNN', 'normalized': 'general regression neural network'},
                        {'surface': 'GABP neural network', 'normalized': 'genetic algorithm back propagation neural network'},
                    ],
                    'confidence': 0.78,
                }
            ]
        if chunk_ids == ['c-2']:
            return [
                {
                    'role': 'result',
                    'act_type': 'report_effect',
                    'summary': 'The predictions agree well with measured burning rate and mechanical performance.',
                    'anchor_chunk_ids': ['c-2'],
                    'research_objects': [{'surface': 'burning rate', 'normalized': 'burning rate'}],
                    'observed_variables': [{'surface': 'mechanical performance', 'normalized': 'mechanical performance'}],
                    'confidence': 0.76,
                }
            ]
        if chunk_ids == ['c-3']:
            return [
                {
                    'role': 'method',
                    'act_type': 'propose_method',
                    'summary': 'Established general regression neural network and genetic algorithm back propagation neural network models to predict propellant burning rate and mechanical performance from formulation variables.',
                    'anchor_chunk_ids': ['c-3'],
                    'research_objects': [
                        {'surface': 'propellant', 'normalized': 'propellant'},
                        {'surface': 'burning rate', 'normalized': 'burning rate'},
                        {'surface': 'mechanical performance', 'normalized': 'mechanical performance'},
                    ],
                    'methods': [
                        {'surface': 'general regression neural network', 'normalized': 'general regression neural network'},
                        {'surface': 'genetic algorithm back propagation neural network', 'normalized': 'genetic algorithm back propagation neural network'},
                    ],
                    'confidence': 0.84,
                }
            ]
        return []

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        _fake_extract_window_moves_llm,
    )
    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._semantic_windows',
        lambda *_args, **_kwargs: [
            {
                'window_id': 'w-1',
                'role_hint': 'method',
                'act_hint': 'propose_method',
                'section_path': ['中文摘要'],
                'chunks': [doc.chunks[0]],
                'char_count': len(doc.chunks[0].text),
            },
            {
                'window_id': 'w-2',
                'role_hint': 'result',
                'act_hint': 'report_effect',
                'section_path': ['Abstract'],
                'chunks': [doc.chunks[1]],
                'char_count': len(doc.chunks[1].text),
            },
            {
                'window_id': 'w-3',
                'role_hint': 'method',
                'act_hint': 'propose_method',
                'section_path': ['Abstract'],
                'chunks': [doc.chunks[2]],
                'char_count': len(doc.chunks[2].text),
            },
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )
    trace = compile_paper_logic_trace(
        **{k: payload[k] for k in ['paper_metadata', 'evidence_rows', 'figure_rows', 'table_rows', 'citation_rows', 'move_relation_rows']}
    )

    method_moves = [move for move in trace.canonical_core.moves if move.role == 'method']
    result_moves = [move for move in trace.canonical_core.moves if move.role == 'result']

    assert len(method_moves) == 1
    assert len(result_moves) == 1
    assert 'burning rate and mechanical performance' in method_moves[0].summary.lower()
    assert len(method_moves[0].anchor_ids) == 2


def test_bilingual_nearby_redundant_method_moves_merge_when_method_signature_matches(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '中文摘要',
            '针对推进剂中细高氯酸铵含量及粒度对推进剂力学性能、燃速的影响，分别建立广义回归神经网络与遗传算法反向传播神经网络模型。',
            line=1,
        ),
        _chunk(
            'c-2',
            'Abstract',
            'The predicted values agree well with the measured burning rate and mechanical performance.',
            line=4,
        ),
        _chunk(
            'c-3',
            'Abstract',
            'General regression neural network and genetic algorithm back propagation neural network models are established to predict propellant burning rate and mechanical performance from formulation variables.',
            line=7,
        ),
    )

    def _fake_extract_window_moves_llm(**kwargs):
        chunk_ids = [str(chunk.chunk_id) for chunk in (kwargs.get('window', {}).get('chunks') or [])]
        if chunk_ids == ['c-1']:
            return [
                {
                    'role': 'method',
                    'act_type': 'propose_method',
                    'summary': '针对推进剂力学性能与燃速预测，建立广义回归神经网络(GRNN)和遗传算法反向传播(GABP)神经网络模型。',
                    'anchor_chunk_ids': ['c-1'],
                    'research_objects': [
                        {'surface': '推进剂力学性能', 'normalized': 'propellant mechanical performance'},
                        {'surface': '燃速', 'normalized': 'burning rate'},
                    ],
                    'methods': [
                        {'surface': '广义回归神经网络', 'normalized': 'general regression neural network'},
                        {'surface': '遗传算法反向传播神经网络', 'normalized': 'genetic algorithm back propagation neural network'},
                    ],
                    'confidence': 0.78,
                }
            ]
        if chunk_ids == ['c-2']:
            return [
                {
                    'role': 'result',
                    'act_type': 'report_effect',
                    'summary': 'The predictions agree well with measured burning rate and mechanical performance.',
                    'anchor_chunk_ids': ['c-2'],
                    'research_objects': [{'surface': 'burning rate', 'normalized': 'burning rate'}],
                    'observed_variables': [{'surface': 'mechanical performance', 'normalized': 'mechanical performance'}],
                    'confidence': 0.76,
                }
            ]
        if chunk_ids == ['c-3']:
            return [
                {
                    'role': 'method',
                    'act_type': 'propose_method',
                    'summary': 'Established general regression neural network and genetic algorithm back propagation neural network models to predict propellant burning rate and mechanical performance from formulation variables.',
                    'anchor_chunk_ids': ['c-3'],
                    'research_objects': [
                        {'surface': 'propellant mechanical performance', 'normalized': 'propellant mechanical performance'},
                        {'surface': 'burning rate', 'normalized': 'burning rate'},
                    ],
                    'methods': [
                        {'surface': 'general regression neural network', 'normalized': 'general regression neural network'},
                        {'surface': 'genetic algorithm back propagation neural network', 'normalized': 'genetic algorithm back propagation neural network'},
                    ],
                    'confidence': 0.84,
                }
            ]
        return []

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        _fake_extract_window_moves_llm,
    )
    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._semantic_windows',
        lambda *_args, **_kwargs: [
            {
                'window_id': 'w-1',
                'role_hint': 'method',
                'act_hint': 'propose_method',
                'section_path': ['中文摘要'],
                'chunks': [doc.chunks[0]],
                'char_count': len(doc.chunks[0].text),
            },
            {
                'window_id': 'w-2',
                'role_hint': 'result',
                'act_hint': 'report_effect',
                'section_path': ['Abstract'],
                'chunks': [doc.chunks[1]],
                'char_count': len(doc.chunks[1].text),
            },
            {
                'window_id': 'w-3',
                'role_hint': 'method',
                'act_hint': 'propose_method',
                'section_path': ['Abstract'],
                'chunks': [doc.chunks[2]],
                'char_count': len(doc.chunks[2].text),
            },
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )
    trace = compile_paper_logic_trace(
        **{k: payload[k] for k in ['paper_metadata', 'evidence_rows', 'figure_rows', 'table_rows', 'citation_rows', 'move_relation_rows']}
    )

    method_moves = [move for move in trace.canonical_core.moves if move.role == 'method']
    result_moves = [move for move in trace.canonical_core.moves if move.role == 'result']

    assert len(method_moves) == 1
    assert len(result_moves) == 1
    assert len(method_moves[0].anchor_ids) == 2
    normalized_methods = [str(item.normalized or item.surface or '').lower() for item in method_moves[0].methods]
    assert any('general' in item and 'regression' in item for item in normalized_methods)
    assert any('genetic' in item and 'algorithm' in item for item in normalized_methods)


def test_method_research_object_filter_drops_free_energy_process_fragment(monkeypatch) -> None:
    summary = 'A continuum-thermodynamics framework defines state variables, free energy, and dissipation potentials.'
    quote = 'A simple elastoviscoplastic model consists of choosing as free energy and dissipation potentials.'
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            quote,
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'adapt_method',
                'summary': summary,
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [],
                'methods': [{'surface': 'continuum-thermodynamics framework'}],
                'confidence': 0.8,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    first = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in first['research_objects']}

    assert 'consists of choosing as free energy and dissipation potentials' not in normalized


def test_research_object_filter_drops_auxiliary_clause_fragments(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            'The study aimed to establish fundamental understanding of particle flow during die filling. '
            'The influence of die velocity on filling behaviour and segregation was assessed, and binary particle mixtures and RSM was used for analysis.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'The study aimed to establish fundamental understanding of particle flow during die filling. '
                'The influence of die velocity on filling behaviour and segregation was assessed, and binary particle mixtures and RSM was used for analysis.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [
                    {'surface': 'aimed to establish fundamental understanding of particle flow'},
                    {'surface': 'particle flow during die filling'},
                    {'surface': 'die velocity on filling behaviour and segregation was assessed'},
                    {'surface': 'binary particle mixtures and rsm was used'},
                    {'surface': 'binary particle mixtures'},
                ],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in move['research_objects']}

    assert 'particle flow during die filling' in normalized
    assert 'binary particle mixtures' in normalized
    assert 'aimed to establish fundamental understanding of particle flow' not in normalized
    assert 'die velocity on filling behaviour and segregation was assessed' not in normalized
    assert 'binary particle mixtures and rsm was used' not in normalized


def test_resource_filter_drops_verb_led_noise_phrase(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '2. Method',
            'The YADE framework provides a stable environment for DEM algorithms.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': 'The YADE framework provides a stable environment for DEM algorithms.',
                'anchor_chunk_ids': ['c-1'],
                'resource_mentions': [
                    {'surface': 'find framework'},
                    {'surface': 'yade software'},
                ],
                'confidence': 0.7,
            }
        ],
    )

    payload = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id='doi:10.1000/demo',
        cite_rec=None,
        schema={'rules': {}},
    )

    move = payload['evidence_rows'][0]
    normalized = {str(item.get('normalized') or '').lower() for item in move['resource_mentions']}

    assert 'yade software' in normalized
    assert 'find framework' not in normalized


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

def test_leading_article_number_line_without_section_is_filtered_before_move_construction(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', None, '文章编号：1000-4750(2012)09-0359-04', line=1),
        _chunk('c-2', 'Demo Paper', '# Demo Paper', line=2),
        _chunk('c-3', 'ABSTRACT', '本文提出一种改进方法来获取松弛模量参数。', line=3),
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
    joined = ' '.join(str(row['summary']) for row in payload['evidence_rows'])

    assert 'c-1' not in source_refs
    assert '文章编号' not in joined


def test_title_block_bibliographic_chunks_with_bullets_and_received_lines_are_filtered(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk('c-1', 'Demo Paper', 'Bo Zhou · Runqiu Huang · Huabin Wang · Jianfeng Wang', line=1),
        _chunk('c-2', 'Demo Paper', 'Received: 5 January 2013 / Published online: 23 March 2013', line=2),
        _chunk('c-3', 'Demo Paper', 'Keywords Anti-rotation · Energy dissipation · DEM', line=3),
        _chunk('c-4', 'Demo Paper', 'This paper aims to compare irregular-shaped particles with disc particles installed with rolling resistance.', line=4),
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
    assert 'bo zhou' not in joined
    assert 'published online' not in joined
    assert 'keywords anti-rotation' not in joined


def test_title_block_long_author_list_with_affiliation_markers_is_filtered(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            'Demo Paper',
            r'Ryoichi Furukawa a, b,\*, Yuki Shiosaka b, Kazunori Kadota c, Keisuke Takagaki a, Tetsurou Noguchi a, Atsuko Shimosaka b, Yoshiyuki Shirakawa b,\*\*',
            line=1,
        ),
        _chunk(
            'c-2',
            'ABSTRACT',
            'This paper investigates segregation behaviour during pharmaceutical die filling using DEM and response surface methodology.',
            line=2,
        ),
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
    assert 'c-2' in source_refs
    assert 'ryoichi furukawa' not in joined


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


def test_experiment_summary_does_not_turn_generic_performance_phrase_into_metric(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Experiments',
            'This reveals details on the performance of the technique for different grain shapes and insight into the differences in the grain-scale mechanisms occurring in these two sands as they exhibit strain localisation under triaxial loading.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'experiment',
                'act_type': 'run_experiment',
                'summary': 'This reveals details on the performance of the technique for different grain shapes and insight into the differences in the grain-scale mechanisms occurring in these two sands as they exhibit strain localisation under triaxial loading.',
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
    metrics = {(item.normalized or item.surface).lower() for item in move.metrics}

    assert metrics == set()


def test_result_summary_preserves_standard_deviation_metric(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            'The measurement of rotations is relatively accurate, with distributions centered on the imposed 4.7 degrees, but not precise, as shown by large standard deviations.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'The measurement of rotations is relatively accurate, with distributions centered on the imposed 4.7 degrees, but not precise, as shown by large standard deviations.',
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
    metrics = {(item.normalized or item.surface).lower() for item in move.metrics}

    assert any('standard deviation' in item for item in metrics)


def test_limitation_summary_does_not_treat_measurement_of_quantities_as_metrics(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '5. Limitations',
            'The technique measures rotations less precisely and reliably than Discrete DIC over a smaller range, and the measurement of displacements and rotations should be interpreted carefully.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'limitation',
                'act_type': 'state_limitation',
                'summary': 'The technique measures rotations less precisely and reliably than Discrete DIC over a smaller range, and the measurement of displacements and rotations should be interpreted carefully.',
                'anchor_chunk_ids': ['c-1'],
                'metrics': [{'surface': 'measurement of displacements'}, {'surface': 'measurement of rotations'}],
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
    metrics = {(item.normalized or item.surface).lower() for item in move.metrics}

    assert 'measurement of displacements' not in metrics
    assert 'measurement of rotations' not in metrics


def test_experiment_summary_prefers_observed_variables_for_physical_quantities(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Experiments',
            'The experiment tracks grain displacements and rotations during strain localisation under triaxial loading.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'experiment',
                'act_type': 'run_experiment',
                'summary': 'The experiment tracks grain displacements and rotations during strain localisation under triaxial loading.',
                'anchor_chunk_ids': ['c-1'],
                'observed_variables': [],
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
    observed = {(item.normalized or item.surface).lower() for item in move.observed_variables}
    metrics = {(item.normalized or item.surface).lower() for item in move.metrics}

    assert any('displacements' in item or 'rotations' in item or 'strain localisation' in item for item in observed)
    assert metrics == set()


def test_experiment_summary_filters_pronoun_clause_observed_variable_noise(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Experiments',
            'This reveals insight into the differences in the two sands as they exhibit strain localisation under triaxial loading.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'experiment',
                'act_type': 'run_experiment',
                'summary': 'This reveals insight into the differences in the two sands as they exhibit strain localisation under triaxial loading.',
                'anchor_chunk_ids': ['c-1'],
                'observed_variables': [],
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
    observed = {(item.normalized or item.surface).lower() for item in move.observed_variables}

    assert 'they exhibit strain' not in observed


def test_method_summary_filters_observed_variable_clause_noise(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '3. Method',
            'The method weights candidate matches by how far the displacement is from the neighborhood median.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': 'The method weights candidate matches by how far the displacement is from the neighborhood median.',
                'anchor_chunk_ids': ['c-1'],
                'observed_variables': [{'surface': 'how far the displacement'}],
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
    observed = {(item.normalized or item.surface).lower() for item in move.observed_variables}

    assert 'how far the displacement' not in observed


def test_result_summary_moves_physical_quantity_from_metric_to_observed_variable(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            'The results show increased rotation variability together with a large standard deviation.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'The results show increased rotation variability together with a large standard deviation.',
                'anchor_chunk_ids': ['c-1'],
                'observed_variables': [],
                'metrics': [{'surface': 'rotation'}, {'surface': 'standard deviation'}],
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
    observed = {(item.normalized or item.surface).lower() for item in move.observed_variables}
    metrics = {(item.normalized or item.surface).lower() for item in move.metrics}

    assert 'rotation' in observed
    assert 'rotation' not in metrics
    assert any('standard deviation' in item for item in metrics)


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


def test_result_summary_filters_bare_generic_comparator_nouns(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            'The filling ratio shows good agreement with experimental results, while additional simulations were also reported.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'The filling ratio shows good agreement with experimental results, while additional simulations were also reported.',
                'anchor_chunk_ids': ['c-1'],
                'comparators': [
                    {'surface': 'experimental results'},
                    {'surface': 'simulations'},
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
    comparators = {(item.normalized or item.surface).lower() for item in move.comparators}

    assert 'experimental results' in comparators
    assert 'simulations' not in comparators


def test_result_summary_filters_validation_and_one_suffix_comparator_fragments(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            'The predicted filling ratio agrees with experimental results and the simulated profile, which was used to validate the proposed model.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'The predicted filling ratio agrees with experimental results and the simulated profile, which was used to validate the proposed model.',
                'anchor_chunk_ids': ['c-1'],
                'comparators': [
                    {'surface': 'experimental results'},
                    {'surface': 'simulated one'},
                    {'surface': 'validate the proposed model'},
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
    comparators = {(item.normalized or item.surface).lower() for item in move.comparators}

    assert 'experimental results' in comparators
    assert 'simulated one' not in comparators
    assert 'validate the proposed model' not in comparators


def test_result_summary_filters_inferred_variable_like_comparator_fragments(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            'The filling ratio differs across die velocity conditions, and comparisons to the dry mixture baseline are reported.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'The filling ratio differs across die velocity conditions, and comparisons to the dry mixture baseline are reported.',
                'anchor_chunk_ids': ['c-1'],
                'comparators': [
                    {'surface': 'die velocity', 'inferred': True},
                    {'surface': 'particle size and position', 'inferred': True},
                    {'surface': 'experimental high-speed camera observations and mea', 'inferred': True},
                    {'surface': 'dry mixture baseline', 'inferred': True},
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
    comparators = {(item.normalized or item.surface).lower() for item in move.comparators}

    assert 'dry mixture baseline' in comparators
    assert 'die velocity' not in comparators
    assert 'particle size and position' not in comparators
    assert 'experimental high-speed camera observations and mea' not in comparators


def test_result_summary_trims_comparator_clause_suffixes(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Results',
            'The disc sample cannot reach the level of the triangle clump sample, and a peak stress ratio similar to the square clump sample is attained by setting a high rolling resistance.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'The disc sample cannot reach the level of the triangle clump sample, and a peak stress ratio similar to the square clump sample is attained by setting a high rolling resistance.',
                'anchor_chunk_ids': ['c-1'],
                'comparators': [
                    {'surface': 'triangle clump sample'},
                    {'surface': 'square clump sample by setting'},
                    {'surface': 'disc samples and their implications'},
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
    comparators = {(item.normalized or item.surface).lower() for item in move.comparators}

    assert 'triangle clump sample' in comparators
    assert 'square clump sample' in comparators
    assert 'square clump sample by setting' not in comparators
    assert 'disc samples' in comparators
    assert 'disc samples and their implications' not in comparators


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


def test_problem_like_summary_stabilizes_interpretation_move_to_problem(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '1. Introduction',
            'This paper aims to quantify segregation during die filling of binary particle mixtures.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'interpretation',
                'act_type': 'explain_mechanism',
                'summary': 'This paper aims to quantify segregation during die filling of binary particle mixtures.',
                'anchor_chunk_ids': ['c-1'],
                'research_objects': [{'surface': 'binary particle mixtures'}],
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

    move = payload['evidence_rows'][0]

    assert move['role_hint'] == 'problem'
    assert move['act_hint'] == 'define_task'


def test_result_like_summary_stabilizes_interpretation_move_to_result(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '4. Discussion',
            'Simulation results show that the calibrated DEM workflow improves segregation prediction accuracy under high die velocity.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'interpretation',
                'act_type': 'explain_mechanism',
                'summary': 'Simulation results show that the calibrated DEM workflow improves segregation prediction accuracy under high die velocity.',
                'anchor_chunk_ids': ['c-1'],
                'metrics': [{'surface': 'prediction accuracy'}],
                'effects': [{'direction': 'improve'}],
                'conditions': [{'surface': 'high die velocity'}],
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

    move = payload['evidence_rows'][0]

    assert move['role_hint'] == 'result'
    assert move['act_hint'] == 'report_effect'


def test_conclusion_achievement_summary_stabilizes_method_move_to_result(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '3. Conclusions',
            'This work succeeded to extend the data-driven strategy from nonlinear elasticity to more complex scenarios involving internal variables.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'method',
                'act_type': 'propose_method',
                'summary': 'This work extends a data-driven strategy from nonlinear elasticity to more complex scenarios involving internal variables.',
                'anchor_chunk_ids': ['c-1'],
                'methods': [{'surface': 'data-driven strategy'}],
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

    move = payload['evidence_rows'][0]

    assert move['role_hint'] == 'result'
    assert move['act_hint'] == 'report_effect'


def test_limitation_like_summary_stabilizes_result_move_to_limitation(monkeypatch) -> None:
    doc = _doc_with_chunks(
        _chunk(
            'c-1',
            '5. Discussion',
            'The conventional rolling resistance model cannot reproduce particle rotation behavior, which is a limitation of the current approach.',
            line=1,
        ),
    )

    monkeypatch.setattr(
        'app.paper_logic_trace.direct_extraction._extract_window_moves_llm',
        lambda **kwargs: [
            {
                'role': 'result',
                'act_type': 'report_effect',
                'summary': 'The conventional rolling resistance model cannot reproduce particle rotation behavior, which is a limitation of the current approach.',
                'anchor_chunk_ids': ['c-1'],
                'limitation_types': [{'surface': 'cannot reproduce particle rotation behavior'}],
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

    move = payload['evidence_rows'][0]

    assert move['role_hint'] == 'limitation'
    assert move['act_hint'] == 'state_limitation'


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


def test_move_relation_builder_attaches_anchor_ids_for_semantic_relations() -> None:
    rows = _build_move_relation_rows(
        [
            {'move_id': 'm-1', 'sequence_no': 1, 'role': 'problem', 'act_type': 'define_task', 'anchor_chunk_ids': ['c-1'], 'anchor_ids': ['a-1']},
            {'move_id': 'm-2', 'sequence_no': 2, 'role': 'method', 'act_type': 'propose_method', 'anchor_chunk_ids': ['c-2'], 'anchor_ids': ['a-2']},
            {'move_id': 'm-3', 'sequence_no': 3, 'role': 'result', 'act_type': 'report_effect', 'anchor_chunk_ids': ['c-3'], 'anchor_ids': ['a-3']},
        ]
    )

    relation_pairs = {
        (row['source_move_id'], row['target_move_id']): row
        for row in rows
    }

    assert relation_pairs[('m-1', 'm-2')]['relation_type'] == 'addresses'
    assert relation_pairs[('m-1', 'm-2')]['anchor_ids'] == ['a-1', 'a-2']
    assert relation_pairs[('m-2', 'm-3')]['relation_type'] == 'yields'
    assert relation_pairs[('m-2', 'm-3')]['anchor_ids'] == ['a-2', 'a-3']


def test_move_relation_builder_keeps_motivates_unanchored() -> None:
    rows = _build_move_relation_rows(
        [
            {'move_id': 'm-1', 'sequence_no': 1, 'role': 'problem', 'act_type': 'define_task', 'anchor_chunk_ids': ['c-1'], 'anchor_ids': ['a-1']},
            {'move_id': 'm-2', 'sequence_no': 2, 'role': 'interpretation', 'act_type': 'explain_mechanism', 'anchor_chunk_ids': ['c-2'], 'anchor_ids': ['a-2']},
        ]
    )

    assert rows[0]['relation_type'] == 'motivates'
    assert rows[0]['anchor_ids'] == []


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
