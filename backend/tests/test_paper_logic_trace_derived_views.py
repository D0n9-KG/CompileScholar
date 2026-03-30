from __future__ import annotations

from app.paper_logic_trace.derived_views import (
    build_derived_views,
    build_community_signatures,
    build_future_work_signals,
    build_l1_bridge_hints,
    build_l2_5_slot_inventory,
    build_paper_content_audit,
    build_paper_content_profile,
    build_paper_summaries,
    build_route_compiler_contract,
    build_route_feature_candidates,
    build_route_state_seed,
)
from app.paper_logic_trace.models import CanonicalCore, CitationAct, EffectValue, EvidenceAnchor, FigureRef, MentionValue, PaperLogicTrace, PaperMetadata, ResearchMove, SlotProvenance, TableRef


def _build_problem_move() -> ResearchMove:
    return ResearchMove(
        move_id='m-0',
        sequence_no=0,
        role='problem',
        act_type='define_task',
        summary='The paper targets robust retrieval over entity relation graphs with sparse supervision.',
        research_objects=[
            MentionValue(
                surface='entity relation graphs',
                normalized='entity relation graph',
                anchor_ids=['a-0'],
            )
        ],
        conditions=[
            MentionValue(
                surface='under sparse supervision',
                normalized='under sparse supervision',
                anchor_ids=['a-0b'],
            )
        ],
        anchor_ids=['a-0', 'a-0b'],
        slot_provenance=[
            SlotProvenance(field='research_objects', value_index=0, anchor_ids=['a-0'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='conditions', value_index=0, anchor_ids=['a-0b'], extraction_mode='direct', support_strength='strong'),
        ],
        confidence=0.87,
    )


def _build_move() -> ResearchMove:
    return ResearchMove(
        move_id='m-1',
        sequence_no=1,
        role='method',
        act_type='propose_method',
        summary='Uses graph neural network modeling for retrieval.',
        research_objects=[
            MentionValue(
                surface='entity relation graph',
                normalized='entity relation graph',
                anchor_ids=['a-1'],
            )
        ],
        methods=[
            MentionValue(
                surface='graph neural network',
                normalized='graph neural network',
                anchor_ids=['a-1'],
            )
        ],
        metrics=[
            MentionValue(
                surface='MRR',
                normalized='mean reciprocal rank',
                anchor_ids=['a-2'],
            )
        ],
        conditions=[
            MentionValue(
                surface='low-resource setting',
                normalized='low-resource setting',
                anchor_ids=['a-3'],
            )
        ],
        comparators=[
            MentionValue(
                surface='baseline retriever',
                normalized='baseline retriever',
                anchor_ids=['a-4'],
            )
        ],
        limitation_types=[
            MentionValue(
                surface='computational cost',
                normalized='computational cost',
                anchor_ids=['a-6'],
            )
        ],
        resource_mentions=[
            MentionValue(
                surface='WN18RR',
                normalized='wn18rr',
                type='benchmark',
                anchor_ids=['a-5'],
            )
        ],
        anchor_ids=['a-1', 'a-2', 'a-3', 'a-4', 'a-5'],
        slot_provenance=[
            SlotProvenance(
                field='resource_mentions',
                value_index=0,
                anchor_ids=['a-5'],
                extraction_mode='direct',
                support_strength='exact',
                confidence=0.95,
            )
        ],
        confidence=0.9,
    )


def _build_result_move() -> ResearchMove:
    return ResearchMove(
        move_id='m-2',
        sequence_no=2,
        role='result',
        act_type='report_effect',
        summary='The graph neural network improves MRR compared to the baseline retriever under low-resource setting.',
        research_objects=[
            MentionValue(
                surface='entity relation graph',
                normalized='entity relation graph',
                anchor_ids=['a-10'],
            )
        ],
        methods=[
            MentionValue(
                surface='graph neural network',
                normalized='graph neural network',
                anchor_ids=['a-10'],
            )
        ],
        metrics=[
            MentionValue(
                surface='MRR',
                normalized='mean reciprocal rank',
                anchor_ids=['a-11'],
            )
        ],
        comparators=[
            MentionValue(
                surface='baseline retriever',
                normalized='baseline retriever',
                anchor_ids=['a-12'],
            )
        ],
        conditions=[
            MentionValue(
                surface='low-resource setting',
                normalized='low-resource setting',
                anchor_ids=['a-13'],
            )
        ],
        limitation_types=[
            MentionValue(
                surface='computational cost',
                normalized='computational cost',
                anchor_ids=['a-14'],
            )
        ],
        anchor_ids=['a-10', 'a-11', 'a-12', 'a-13', 'a-14'],
        effects=[],
        slot_provenance=[
            SlotProvenance(
                field='research_objects',
                value_index=0,
                anchor_ids=['a-10'],
                extraction_mode='direct',
                support_strength='strong',
                confidence=0.9,
            ),
            SlotProvenance(
                field='methods',
                value_index=0,
                anchor_ids=['a-10'],
                extraction_mode='direct',
                support_strength='strong',
                confidence=0.9,
            ),
            SlotProvenance(
                field='metrics',
                value_index=0,
                anchor_ids=['a-11'],
                extraction_mode='direct',
                support_strength='strong',
                confidence=0.9,
            ),
            SlotProvenance(
                field='comparators',
                value_index=0,
                anchor_ids=['a-12'],
                extraction_mode='direct',
                support_strength='strong',
                confidence=0.9,
            ),
            SlotProvenance(
                field='conditions',
                value_index=0,
                anchor_ids=['a-13'],
                extraction_mode='direct',
                support_strength='strong',
                confidence=0.9,
            ),
            SlotProvenance(
                field='limitation_types',
                value_index=0,
                anchor_ids=['a-14'],
                extraction_mode='direct',
                support_strength='strong',
                confidence=0.9,
            ),
        ],
        confidence=0.92,
    )


def _build_bridge_move() -> ResearchMove:
    return ResearchMove(
        move_id='m-3',
        sequence_no=3,
        role='experiment',
        act_type='set_condition',
        summary='Uses X-ray microtomography and YADE software to measure packing density against the dry baseline under high compression.',
        methods=[
            MentionValue(
                surface='discrete element simulation',
                normalized='discrete element simulation',
                anchor_ids=['a-20'],
            )
        ],
        metrics=[
            MentionValue(
                surface='packing density',
                normalized='packing density',
                anchor_ids=['a-21'],
            )
        ],
        comparators=[
            MentionValue(
                surface='dry baseline',
                normalized='dry baseline',
                anchor_ids=['a-22'],
            )
        ],
        conditions=[
            MentionValue(
                surface='under high compression',
                normalized='under high compression',
                anchor_ids=['a-23'],
            )
        ],
        resource_mentions=[
            MentionValue(
                surface='X-ray microtomography',
                normalized='x-ray microtomography',
                type='instrument',
                anchor_ids=['a-24'],
            ),
            MentionValue(
                surface='YADE',
                normalized='yade',
                type='software',
                anchor_ids=['a-25'],
            ),
        ],
        anchor_ids=['a-20', 'a-21', 'a-22', 'a-23', 'a-24', 'a-25'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-20'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='metrics', value_index=0, anchor_ids=['a-21'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='comparators', value_index=0, anchor_ids=['a-22'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='conditions', value_index=0, anchor_ids=['a-23'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='resource_mentions', value_index=0, anchor_ids=['a-24'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='resource_mentions', value_index=1, anchor_ids=['a-25'], extraction_mode='direct', support_strength='strong'),
        ],
        confidence=0.88,
    )


def _build_future_work_move() -> ResearchMove:
    return ResearchMove(
        move_id='m-4',
        sequence_no=4,
        role='future_work',
        act_type='suggest_extension',
        summary='Future work should extend graph neural network retrieval to temporal knowledge graphs under noisy labels using web-scale corpora.',
        research_objects=[
            MentionValue(
                surface='temporal knowledge graphs',
                normalized='temporal knowledge graph',
                anchor_ids=['a-30'],
            )
        ],
        methods=[
            MentionValue(
                surface='graph neural network retrieval',
                normalized='graph neural network retrieval',
                anchor_ids=['a-31'],
            )
        ],
        conditions=[
            MentionValue(
                surface='under noisy labels',
                normalized='under noisy labels',
                anchor_ids=['a-32'],
            )
        ],
        limitation_types=[
            MentionValue(
                surface='limited labeled data',
                normalized='limited labeled data',
                anchor_ids=['a-33'],
            )
        ],
        resource_mentions=[
            MentionValue(
                surface='web-scale corpora',
                normalized='web-scale corpora',
                type='dataset',
                anchor_ids=['a-34'],
            )
        ],
        anchor_ids=['a-30', 'a-31', 'a-32', 'a-33', 'a-34'],
        slot_provenance=[
            SlotProvenance(field='research_objects', value_index=0, anchor_ids=['a-30'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-31'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='conditions', value_index=0, anchor_ids=['a-32'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='limitation_types', value_index=0, anchor_ids=['a-33'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='resource_mentions', value_index=0, anchor_ids=['a-34'], extraction_mode='direct', support_strength='strong'),
        ],
        confidence=0.81,
    )


def _build_limitation_move() -> ResearchMove:
    return ResearchMove(
        move_id='m-5',
        sequence_no=5,
        role='limitation',
        act_type='state_limitation',
        summary='The current graph neural network retriever remains sensitive to noisy labels and expensive training runs.',
        limitation_types=[
            MentionValue(
                surface='noisy labels',
                normalized='noisy labels',
                anchor_ids=['a-40'],
            ),
            MentionValue(
                surface='expensive training runs',
                normalized='expensive training runs',
                anchor_ids=['a-41'],
            ),
        ],
        anchor_ids=['a-40', 'a-41'],
        slot_provenance=[
            SlotProvenance(field='limitation_types', value_index=0, anchor_ids=['a-40'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='limitation_types', value_index=1, anchor_ids=['a-41'], extraction_mode='direct', support_strength='strong'),
        ],
        confidence=0.83,
    )


def test_build_community_signatures_uses_move_slots_not_raw_summary() -> None:
    signatures = build_community_signatures('paper-1', [_build_move()])

    assert signatures[0]['move_id'] == 'm-1'
    assert 'graph neural network' in signatures[0]['method_tokens']
    assert 'entity relation graph' in signatures[0]['object_tokens']


def test_build_route_feature_candidates_points_back_to_canonical_moves() -> None:
    candidates = build_route_feature_candidates([_build_move()])
    candidate_types = {candidate['candidate_type'] for candidate in candidates}

    assert candidates[0]['move_id'] == 'm-1'
    assert 'method_candidate' in candidate_types
    assert 'metric_candidate' in candidate_types
    assert 'condition_candidate' in candidate_types
    assert 'comparison_candidate' in candidate_types
    assert 'benchmark_candidate' in candidate_types
    assert 'limitation_candidate' in candidate_types
    assert any(
        candidate['candidate_type'] == 'benchmark_candidate' and 'wn18rr' in candidate['tokens']
        for candidate in candidates
    )


def test_build_l1_bridge_hints_preserves_provenance_refs() -> None:
    hints = build_l1_bridge_hints([_build_move()])

    assert hints['resource_candidates'][0]['move_id'] == 'm-1'
    assert hints['resource_candidates'][0]['anchor_ids'] == ['a-5']
    assert hints['resource_candidates'][0]['provenance'][0]['field'] == 'resource_mentions'


def test_build_l2_5_slot_inventory_exposes_doc_aligned_slots() -> None:
    move = ResearchMove(
        move_id='m-l25',
        sequence_no=1,
        role='result',
        act_type='report_effect',
        summary='The tomography-assisted DEM workflow increases packing density by 12% under high compression relative to the dry baseline.',
        research_objects=[
            MentionValue(
                surface='powder packing',
                normalized='powder packing',
                anchor_ids=['a-1'],
            )
        ],
        methods=[
            MentionValue(
                surface='tomography-assisted DEM workflow',
                normalized='tomography-assisted dem workflow',
                anchor_ids=['a-2'],
            )
        ],
        observed_variables=[
            MentionValue(
                surface='packing density',
                normalized='packing density',
                anchor_ids=['a-3'],
            )
        ],
        metrics=[
            MentionValue(
                surface='packing density',
                normalized='packing density',
                anchor_ids=['a-3'],
            )
        ],
        comparators=[
            MentionValue(
                surface='dry baseline',
                normalized='dry baseline',
                anchor_ids=['a-4'],
            )
        ],
        conditions=[
            MentionValue(
                surface='under high compression',
                normalized='under high compression',
                anchor_ids=['a-5'],
            )
        ],
        limitation_types=[
            MentionValue(
                surface='computational cost',
                normalized='computational cost',
                anchor_ids=['a-6'],
            )
        ],
        resource_mentions=[
            MentionValue(
                surface='X-ray microtomography',
                normalized='x-ray microtomography',
                type='instrument',
                anchor_ids=['a-7'],
            )
        ],
        effects=[
            EffectValue(
                direction='increase',
                magnitude_text='12%',
                comparator_surface='dry baseline',
                anchor_ids=['a-8'],
                confidence=0.91,
            )
        ],
        anchor_ids=['a-1', 'a-2', 'a-3', 'a-4', 'a-5', 'a-6', 'a-7', 'a-8'],
        slot_provenance=[
            SlotProvenance(field='research_objects', value_index=0, anchor_ids=['a-1'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-2'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='observed_variables', value_index=0, anchor_ids=['a-3'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='metrics', value_index=0, anchor_ids=['a-3'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='comparators', value_index=0, anchor_ids=['a-4'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='conditions', value_index=0, anchor_ids=['a-5'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='limitation_types', value_index=0, anchor_ids=['a-6'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='resource_mentions', value_index=0, anchor_ids=['a-7'], extraction_mode='direct', support_strength='strong'),
        ],
    )

    inventory = build_l2_5_slot_inventory([move])

    assert inventory['observed_variables'][0]['normalized'] == 'packing density'
    assert inventory['effects'][0]['direction'] == 'increase'
    assert inventory['effect_size'][0]['magnitude_text'] == '12%'
    assert inventory['operation_or_method'][0]['normalized'] == 'tomography-assisted dem workflow'
    assert inventory['comparison_target'][0]['normalized'] == 'dry baseline'
    assert inventory['condition_context'][0]['normalized'] == 'under high compression'
    assert inventory['limitation_type'][0]['normalized'] == 'computational cost'


def test_build_l1_bridge_hints_derives_protocol_and_toolchain_candidates() -> None:
    move = ResearchMove(
        move_id='m-bridge',
        sequence_no=1,
        role='experiment',
        act_type='set_condition',
        summary='Uses X-ray microtomography and YADE software to measure packing density against the dry baseline under high compression.',
        methods=[
            MentionValue(
                surface='discrete element simulation',
                normalized='discrete element simulation',
                anchor_ids=['a-10'],
            )
        ],
        metrics=[
            MentionValue(
                surface='packing density',
                normalized='packing density',
                anchor_ids=['a-11'],
            )
        ],
        comparators=[
            MentionValue(
                surface='dry baseline',
                normalized='dry baseline',
                anchor_ids=['a-12'],
            )
        ],
        conditions=[
            MentionValue(
                surface='under high compression',
                normalized='under high compression',
                anchor_ids=['a-13'],
            )
        ],
        resource_mentions=[
            MentionValue(
                surface='X-ray microtomography',
                normalized='x-ray microtomography',
                type='instrument',
                anchor_ids=['a-14'],
            ),
            MentionValue(
                surface='YADE',
                normalized='yade',
                type='software',
                anchor_ids=['a-15'],
            ),
        ],
        anchor_ids=['a-10', 'a-11', 'a-12', 'a-13', 'a-14', 'a-15'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-10'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='metrics', value_index=0, anchor_ids=['a-11'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='comparators', value_index=0, anchor_ids=['a-12'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='conditions', value_index=0, anchor_ids=['a-13'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='resource_mentions', value_index=0, anchor_ids=['a-14'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='resource_mentions', value_index=1, anchor_ids=['a-15'], extraction_mode='direct', support_strength='strong'),
        ],
    )

    hints = build_l1_bridge_hints([move])

    assert hints['protocol_candidates']
    assert hints['protocol_candidates'][0]['metric_tokens'] == ['packing density']
    assert hints['protocol_candidates'][0]['comparator_tokens'] == ['dry baseline']
    assert hints['protocol_candidates'][0]['condition_tokens'] == ['under high compression']
    assert hints['toolchain_candidates']
    assert hints['toolchain_candidates'][0]['resource_tokens'] == ['x-ray microtomography', 'yade']


def test_build_route_and_l1_bridge_skip_weak_inferred_slot_candidates() -> None:
    move = ResearchMove(
        move_id='m-weak',
        sequence_no=1,
        role='result',
        act_type='report_effect',
        summary='The workflow changes prediction accuracy relative to the baseline under high stress conditions.',
        methods=[
            MentionValue(
                surface='graph neural network',
                normalized='graph neural network',
                anchor_ids=['a-1'],
            )
        ],
        metrics=[
            MentionValue(
                surface='prediction accuracy',
                normalized='prediction accuracy',
                anchor_ids=['a-2'],
                inferred=True,
            )
        ],
        comparators=[
            MentionValue(
                surface='baseline',
                normalized='baseline',
                anchor_ids=['a-2'],
                inferred=True,
            )
        ],
        resource_mentions=[
            MentionValue(
                surface='WN18RR',
                normalized='wn18rr',
                type='benchmark',
                anchor_ids=['a-3'],
                inferred=True,
            )
        ],
        anchor_ids=['a-1', 'a-2', 'a-3'],
        slot_provenance=[
            SlotProvenance(
                field='methods',
                value_index=0,
                anchor_ids=['a-1'],
                extraction_mode='direct',
                support_strength='strong',
                confidence=0.9,
            ),
            SlotProvenance(
                field='metrics',
                value_index=0,
                anchor_ids=['a-2'],
                extraction_mode='inferred',
                support_strength='weak',
                confidence=0.42,
            ),
            SlotProvenance(
                field='comparators',
                value_index=0,
                anchor_ids=['a-2'],
                extraction_mode='inferred',
                support_strength='weak',
                confidence=0.42,
            ),
            SlotProvenance(
                field='resource_mentions',
                value_index=0,
                anchor_ids=['a-3'],
                extraction_mode='inferred',
                support_strength='weak',
                confidence=0.42,
            ),
        ],
        confidence=0.7,
    )

    candidates = build_route_feature_candidates([move])
    candidate_types = {candidate['candidate_type'] for candidate in candidates}
    hints = build_l1_bridge_hints([move])

    assert candidate_types == {'method_candidate'}
    assert hints['resource_candidates'] == []
    assert hints['benchmark_candidates'] == []
    assert hints['metric_candidates'] == []


def test_build_route_compiler_contract_collects_auditable_l3_l4_inputs() -> None:
    contract = build_route_compiler_contract(
        paper_id='paper-1',
        paper_type='empirical',
        moves=[_build_move(), _build_result_move(), _build_future_work_move()],
    )

    assert contract['paper_id'] == 'paper-1'
    assert 'method_proposal' in contract['primary_contributions']
    assert 'outcome_evidence' in contract['primary_contributions']
    assert 'comparison_evidence' in contract['primary_contributions']
    assert 'future_direction' in contract['primary_contributions']
    assert any(entry['normalized'] == 'entity relation graph' for entry in contract['topic_signals']['objects'])
    assert any(entry['normalized'] == 'graph neural network' for entry in contract['topic_signals']['methods'])
    assert any(entry['metric_tokens'] == ['mean reciprocal rank'] for entry in contract['outcome_signals'])
    assert any(entry['comparator_tokens'] == ['baseline retriever'] for entry in contract['comparison_signals'])
    assert any(entry['normalized'] == 'computational cost' for entry in contract['constraint_signals']['limitations'])
    assert contract['future_direction_signals'][0]['target_object_tokens'] == ['temporal knowledge graph']
    assert contract['future_direction_signals'][0]['resource_tokens'] == ['web-scale corpora']
    assert contract['signal_counts']['future_work_entries'] == 1


def test_build_future_work_signals_and_derived_views_export_structured_extension_targets() -> None:
    future_work_signals = build_future_work_signals([_build_future_work_move()])

    assert future_work_signals[0]['move_id'] == 'm-4'
    assert future_work_signals[0]['target_object_tokens'] == ['temporal knowledge graph']
    assert future_work_signals[0]['method_tokens'] == ['graph neural network retrieval']
    assert future_work_signals[0]['condition_tokens'] == ['under noisy labels']
    assert future_work_signals[0]['resource_tokens'] == ['web-scale corpora']
    assert future_work_signals[0]['limitation_tokens'] == ['limited labeled data']

    trace = PaperLogicTrace(
        trace_id='paper-1:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-28T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-1',
            title='Demo',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[_build_move(), _build_result_move(), _build_future_work_move()],
        ),
        quality={},
    )

    derived = build_derived_views(trace)

    assert derived['future_work_signals'][0]['target_object_tokens'] == ['temporal knowledge graph']
    assert derived['route_compiler_contract']['future_direction_signals'][0]['resource_tokens'] == ['web-scale corpora']


def test_build_paper_content_profile_exports_single_paper_story_surfaces() -> None:
    trace = PaperLogicTrace(
        trace_id='paper-1:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-28T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-1',
            title='Demo',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[
                _build_problem_move(),
                _build_move(),
                _build_result_move(),
                _build_limitation_move(),
                _build_future_work_move(),
            ],
            citation_acts=[
                CitationAct(
                    citation_act_id='c-1',
                    source_move_id='m-2',
                    target_paper_id='paper-prev',
                    purpose='compare',
                    polarity='neutral',
                    semantic_signal='baseline comparison',
                    target_scope='method',
                    anchor_ids=['a-12'],
                    confidence=0.74,
                )
            ],
            figure_refs=[
                FigureRef(
                    figure_id='fig-1',
                    caption='Retrieval performance across sparse-supervision regimes.',
                    anchor_ids=['a-11'],
                )
            ],
            table_refs=[
                TableRef(
                    table_id='tab-1',
                    caption='Benchmark comparison on WN18RR.',
                    anchor_ids=['a-12'],
                )
            ],
        ),
        quality={},
    )

    profile = build_paper_content_profile(trace)
    derived = build_derived_views(trace)

    assert profile['paper_id'] == 'paper-1'
    assert profile['problem_statements'] == ['The paper targets robust retrieval over entity relation graphs with sparse supervision.']
    assert profile['method_statements'] == ['Uses graph neural network modeling for retrieval.']
    assert profile['key_findings'] == ['The graph neural network improves MRR compared to the baseline retriever under low-resource setting.']
    assert profile['limitation_statements'][0]['limitation_tokens'] == ['noisy labels', 'expensive training runs']
    assert profile['future_work_statements'][0]['target_object_tokens'] == ['temporal knowledge graph']
    assert profile['citation_contexts'][0]['semantic_signal'] == 'baseline comparison'
    assert profile['figure_refs'][0]['figure_id'] == 'fig-1'
    assert profile['table_refs'][0]['table_id'] == 'tab-1'
    assert profile['coverage']['future_work_count'] == 1
    assert profile['coverage']['citation_context_count'] == 1
    assert derived['paper_content_profile']['coverage']['figure_count'] == 1


def test_build_paper_content_profile_prioritizes_result_summaries_over_earlier_interpretation() -> None:
    interpretation = ResearchMove(
        move_id='m-int-1',
        sequence_no=1,
        role='interpretation',
        act_type='explain_mechanism',
        summary='Continuum thermodynamics assumes the system state is defined by observable and internal state variables.',
        anchor_ids=['a-1'],
    )
    result_one = ResearchMove(
        move_id='m-result-1',
        sequence_no=2,
        role='result',
        act_type='report_effect',
        summary='This work extends the data-driven strategy to inelastic scenarios involving internal variables.',
        anchor_ids=['a-2'],
    )
    result_two = ResearchMove(
        move_id='m-result-2',
        sequence_no=3,
        role='result',
        act_type='report_effect',
        summary='The approach includes plastic strain rate and accumulated plastic deformation in the behavior manifold.',
        anchor_ids=['a-3'],
    )
    trace = PaperLogicTrace(
        trace_id='paper-1:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-29T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-1',
            title='Data-Driven Computational Plasticity',
            paper_type='theoretical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[interpretation, result_one, result_two],
        ),
        quality={},
    )

    profile = build_paper_content_profile(trace)

    assert profile['key_findings'][:2] == [result_one.summary, result_two.summary]
    assert profile['coverage']['finding_count'] == 3


def test_build_paper_content_profile_prefers_current_work_method_statements_over_prior_work_review() -> None:
    prior_work_method = ResearchMove(
        move_id='m-method-0',
        sequence_no=1,
        role='method',
        act_type='adapt_method',
        summary='Previous work by Zapas, KelchnerLee, FloryA, and Joonas Sorvari used numerical methods to improve the accuracy of relaxation modulus determination.',
        anchor_ids=['a-1'],
    )
    current_method = ResearchMove(
        move_id='m-method-1',
        sequence_no=2,
        role='method',
        act_type='propose_method',
        summary='The paper proposes an improved Sorvari method to determine the relaxation modulus of modified double-base propellant.',
        anchor_ids=['a-2'],
    )
    current_experiment_method = ResearchMove(
        move_id='m-method-exp-1',
        sequence_no=3,
        role='method',
        act_type='run_experiment',
        summary='进行了改性双基推进剂的拉伸松弛试验，并利用提出的改进型Sorvari法得到改性双基推进剂的松弛模量。',
        anchor_ids=['a-3'],
    )
    current_method_detail = ResearchMove(
        move_id='m-method-2',
        sequence_no=4,
        role='method',
        act_type='propose_method',
        summary='The improved Sorvari method iteratively solves for alpha(t) values and then calculates the relaxation modulus.',
        anchor_ids=['a-4'],
    )
    trace = PaperLogicTrace(
        trace_id='paper-1732:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-30T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-1732',
            title='Determination Way of Relaxation Modulus of Modified DB Propellant',
            title_alt='改性双基推进剂松弛模量的确定方法',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[prior_work_method, current_method, current_experiment_method, current_method_detail],
        ),
        quality={},
    )

    profile = build_paper_content_profile(trace)

    assert prior_work_method.summary not in profile['method_statements']
    assert current_method.summary in profile['method_statements']
    assert current_experiment_method.summary in profile['method_statements']
    assert current_method_detail.summary in profile['method_statements']


def test_build_paper_content_profile_prefers_problem_statements_over_background_context_when_limit_reached() -> None:
    background = ResearchMove(
        move_id='m-bg-1',
        sequence_no=1,
        role='background',
        act_type='build_resource',
        summary='The paper describes the experimental setup and measurement techniques used to study velocity profiles in slowly sheared bubble rafts.',
        anchor_ids=['a-1'],
    )
    problem_one = ResearchMove(
        move_id='m-problem-1',
        sequence_no=2,
        role='problem',
        act_type='identify_gap',
        summary='Quantitative experimental studies of velocity profiles in jammed systems have only recently been carried out.',
        anchor_ids=['a-2'],
    )
    problem_two = ResearchMove(
        move_id='m-problem-2',
        sequence_no=3,
        role='problem',
        act_type='identify_gap',
        summary='For slow shear rates, prior work focused on the high shear-rate continuum limit rather than rearrangement-scale fluctuations.',
        anchor_ids=['a-3'],
    )
    problem_three = ResearchMove(
        move_id='m-problem-3',
        sequence_no=4,
        role='problem',
        act_type='define_task',
        summary='The paper investigates whether the coexistence of flowing and jammed states persists in the low shear-rate limit.',
        anchor_ids=['a-4'],
    )
    trace = PaperLogicTrace(
        trace_id='paper-1505:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-30T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-1505',
            title='Velocity Profiles in Slowly Sheared Bubble Rafts',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[background, problem_one, problem_two, problem_three],
        ),
        quality={},
    )

    profile = build_paper_content_profile(trace)

    assert profile['problem_statements'] == [
        problem_one.summary,
        problem_two.summary,
        problem_three.summary,
    ]
    assert background.summary not in profile['problem_statements']


def test_build_route_state_seed_collects_l3_compiler_inputs_without_overclaiming() -> None:
    trace = PaperLogicTrace(
        trace_id='paper-1:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-27T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-1',
            title='Demo',
            year=2024,
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[_build_move(), _build_result_move(), _build_bridge_move(), _build_future_work_move()],
        ),
        quality={},
    )

    seed = build_route_state_seed(trace)

    assert seed['paper_id'] == 'paper-1'
    assert seed['cutoff_year_hint'] == 2024
    assert 'entity relation graph' in seed['topic_scope_candidates']
    assert 'graph neural network' in seed['dominant_method_candidates']
    assert 'wn18rr' in seed['active_benchmark_candidates']
    assert 'computational cost' in seed['known_bottleneck_candidates']
    assert 'under high compression' in seed['enabling_condition_candidates']
    assert 'graph neural network retrieval -> temporal knowledge graph' in seed['alternative_route_candidates']
    assert seed['measurement_protocol_candidates']
    assert seed['toolchain_candidates']
    assert 'a-14' in seed['challenging_evidence_ids']
    assert 'a-24' in seed['supporting_evidence_ids']


def test_build_route_state_seed_excludes_prior_work_method_signal_when_current_method_evidence_is_sufficient() -> None:
    prior_work_method = ResearchMove(
        move_id='m-method-0',
        sequence_no=1,
        role='method',
        act_type='adapt_method',
        summary='Previous work by Zapas, KelchnerLee, FloryA, and Joonas Sorvari used numerical methods to improve the accuracy of relaxation modulus determination.',
        methods=[
            MentionValue(surface='Sorvari method', normalized='sorvari method', anchor_ids=['a-1']),
        ],
        anchor_ids=['a-1'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-1'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    current_method = ResearchMove(
        move_id='m-method-1',
        sequence_no=2,
        role='method',
        act_type='propose_method',
        summary='The paper proposes an improved Sorvari method to determine the relaxation modulus of modified double-base propellant.',
        methods=[
            MentionValue(surface='improved Sorvari method', normalized='improved sorvari method', anchor_ids=['a-2']),
        ],
        anchor_ids=['a-2'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-2'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    current_experiment_method = ResearchMove(
        move_id='m-method-exp-1',
        sequence_no=3,
        role='method',
        act_type='run_experiment',
        summary='进行了改性双基推进剂的拉伸松弛试验，并利用提出的改进型Sorvari法得到改性双基推进剂的松弛模量。',
        methods=[
            MentionValue(surface='拉伸松弛试验', normalized='tensile relaxation test', anchor_ids=['a-3']),
            MentionValue(surface='改进型Sorvari法', normalized='improved sorvari method', anchor_ids=['a-3']),
        ],
        anchor_ids=['a-3'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-3'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='methods', value_index=1, anchor_ids=['a-3'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    current_method_detail = ResearchMove(
        move_id='m-method-2',
        sequence_no=4,
        role='method',
        act_type='propose_method',
        summary='The improved Sorvari method iteratively solves for alpha(t) values and then calculates the relaxation modulus.',
        methods=[
            MentionValue(surface='numerical iteration', normalized='numerical iteration', anchor_ids=['a-4']),
        ],
        anchor_ids=['a-4'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-4'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    trace = PaperLogicTrace(
        trace_id='paper-1732:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-30T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-1732',
            title='Determination Way of Relaxation Modulus of Modified DB Propellant',
            title_alt='改性双基推进剂松弛模量的确定方法',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[prior_work_method, current_method, current_experiment_method, current_method_detail],
        ),
        quality={},
    )

    contract = build_route_compiler_contract(
        paper_id=trace.paper_metadata.paper_id,
        paper_type=trace.paper_metadata.paper_type,
        moves=list(trace.canonical_core.moves),
    )
    seed = build_route_state_seed(trace)

    method_labels = [entry['normalized'] for entry in contract['topic_signals']['methods']]

    assert 'sorvari method' not in method_labels
    assert 'improved sorvari method' in method_labels
    assert 'tensile relaxation test' in method_labels
    assert 'numerical iteration' in method_labels
    assert 'sorvari method' not in seed['dominant_method_candidates']


def test_build_route_state_seed_prioritizes_operational_methods_over_broad_framework_labels() -> None:
    move_algorithm = ResearchMove(
        move_id='m-method-1',
        sequence_no=1,
        role='method',
        act_type='propose_method',
        summary='The authors employ an algorithm to mimic stress-controlled rheology, building on an extended Stokesian Dynamics approach.',
        methods=[
            MentionValue(surface='stress-controlled rheology algorithm', normalized='stress-controlled rheology algorithm', anchor_ids=['a-1']),
            MentionValue(surface='Stokesian Dynamics', normalized='stokesian dynamics', anchor_ids=['a-1']),
        ],
        anchor_ids=['a-1'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-1'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='methods', value_index=1, anchor_ids=['a-1'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    move_framework = ResearchMove(
        move_id='m-method-2',
        sequence_no=2,
        role='method',
        act_type='adapt_method',
        summary='Describes the governing equations for dense suspensions under stress-controlled quasi-static conditions, adapting hydrodynamic and non-hydrodynamic interaction models.',
        methods=[
            MentionValue(surface='force and torque balance equations', normalized='force and torque balance equations', anchor_ids=['a-2']),
            MentionValue(surface='linear resistance', normalized='linear resistance', anchor_ids=['a-2']),
            MentionValue(surface='pair-wise hydrodynamic lubrication', normalized='pair-wise hydrodynamic lubrication', anchor_ids=['a-2']),
        ],
        anchor_ids=['a-2'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-2'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='methods', value_index=1, anchor_ids=['a-2'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='methods', value_index=2, anchor_ids=['a-2'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    move_solver = ResearchMove(
        move_id='m-method-3',
        sequence_no=3,
        role='method',
        act_type='propose_method',
        summary='Proposes a simulation framework to determine particle velocities and shear rate from a given shear stress by solving linear equations derived from stress decomposition.',
        methods=[
            MentionValue(surface='determine shear rate from given shear stress', normalized='determine shear rate from given shear stress', anchor_ids=['a-3']),
        ],
        anchor_ids=['a-3'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-3'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    move_contacts = ResearchMove(
        move_id='m-method-4',
        sequence_no=4,
        role='method',
        act_type='propose_method',
        summary='The authors model contact forces using a soft-constraint approach with harmonic penalty functions and a Coulomb friction law.',
        methods=[
            MentionValue(surface='soft-constraint approach', normalized='soft-constraint approach', anchor_ids=['a-4']),
        ],
        anchor_ids=['a-4'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-4'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    move_validation = ResearchMove(
        move_id='m-method-5',
        sequence_no=5,
        role='method',
        act_type='adapt_method',
        summary='The authors simulate a stress-controlled shear reversal test to confirm the concept of fragile matter in dense suspensions.',
        methods=[
            MentionValue(surface='stress-controlled shear reversal test', normalized='stress-controlled shear reversal test', anchor_ids=['a-5']),
        ],
        anchor_ids=['a-5'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-5'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    trace = PaperLogicTrace(
        trace_id='paper-1607:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-30T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-1607',
            title='Shear jamming and fragility in dense suspensions',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[move_algorithm, move_framework, move_solver, move_contacts, move_validation],
        ),
        quality={},
    )

    seed = build_route_state_seed(trace)
    top_methods = seed['dominant_method_candidates'][:5]

    assert 'stress-controlled rheology algorithm' in top_methods
    assert 'stokesian dynamics' in top_methods
    assert 'soft-constraint approach' in top_methods
    assert 'stress-controlled shear reversal test' in top_methods
    assert 'force and torque balance equations' not in top_methods
    assert 'linear resistance' not in top_methods


def test_build_route_state_seed_keeps_framework_method_when_it_is_the_only_grounded_method_signal() -> None:
    framework_move = ResearchMove(
        move_id='m-method-1',
        sequence_no=1,
        role='method',
        act_type='adapt_method',
        summary='A continuum-thermodynamics framework is described, assuming state variables, a free energy, and dissipation potentials to define state laws and evolution laws for elastoviscoplastic behavior.',
        methods=[
            MentionValue(surface='continuum-thermodynamics framework', normalized='continuum-thermodynamics framework', anchor_ids=['a-1']),
        ],
        anchor_ids=['a-1'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-1'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    trace = PaperLogicTrace(
        trace_id='paper-1243:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-30T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-1243',
            title='Data-Driven Computational Plasticity',
            paper_type='theoretical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[framework_move],
        ),
        quality={},
    )

    seed = build_route_state_seed(trace)

    assert seed['dominant_method_candidates'] == ['continuum-thermodynamics framework']


def test_build_route_state_seed_deprioritizes_clause_style_method_labels_when_named_methods_exist() -> None:
    move_algorithm = ResearchMove(
        move_id='m-method-1',
        sequence_no=1,
        role='method',
        act_type='propose_method',
        summary='The authors employ an algorithm to mimic stress-controlled rheology, building on an extended Stokesian Dynamics approach.',
        methods=[
            MentionValue(surface='stress-controlled rheology algorithm', normalized='stress-controlled rheology algorithm', anchor_ids=['a-1']),
            MentionValue(surface='Stokesian Dynamics', normalized='stokesian dynamics', anchor_ids=['a-1']),
        ],
        anchor_ids=['a-1'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-1'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='methods', value_index=1, anchor_ids=['a-1'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    move_solver = ResearchMove(
        move_id='m-method-2',
        sequence_no=2,
        role='method',
        act_type='propose_method',
        summary='Proposes a simulation framework to determine particle velocities and shear rate from a given shear stress by solving linear equations derived from stress decomposition.',
        methods=[
            MentionValue(surface='determine shear rate from given shear stress', normalized='determine shear rate from given shear stress', anchor_ids=['a-2']),
            MentionValue(surface='determine particle velocities from shear rate', normalized='determine particle velocities from shear rate', anchor_ids=['a-2']),
        ],
        anchor_ids=['a-2'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-2'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='methods', value_index=1, anchor_ids=['a-2'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    move_contacts = ResearchMove(
        move_id='m-method-3',
        sequence_no=3,
        role='method',
        act_type='propose_method',
        summary='The authors model contact forces using a soft-constraint approach with harmonic penalty functions and a Coulomb friction law.',
        methods=[
            MentionValue(surface='soft-constraint approach', normalized='soft-constraint approach', anchor_ids=['a-3']),
        ],
        anchor_ids=['a-3'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-3'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    move_validation = ResearchMove(
        move_id='m-method-4',
        sequence_no=4,
        role='method',
        act_type='adapt_method',
        summary='The authors simulate a stress-controlled shear reversal test to confirm the concept of fragile matter in dense suspensions.',
        methods=[
            MentionValue(surface='stress-controlled shear reversal test', normalized='stress-controlled shear reversal test', anchor_ids=['a-4']),
        ],
        anchor_ids=['a-4'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-4'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    trace = PaperLogicTrace(
        trace_id='paper-1607:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-30T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-1607',
            title='Shear jamming and fragility in dense suspensions',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[move_algorithm, move_solver, move_contacts, move_validation],
        ),
        quality={},
    )

    seed = build_route_state_seed(trace)
    assert seed['dominant_method_candidates'] == [
        'stress-controlled rheology algorithm',
        'soft-constraint approach',
        'stress-controlled shear reversal test',
        'stokesian dynamics',
    ]


def test_build_route_state_seed_keeps_clause_style_method_label_when_it_is_the_only_grounded_method_signal() -> None:
    move_solver = ResearchMove(
        move_id='m-method-1',
        sequence_no=1,
        role='method',
        act_type='propose_method',
        summary='Proposes a simulation framework to determine particle velocities and shear rate from a given shear stress by solving linear equations derived from stress decomposition.',
        methods=[
            MentionValue(surface='determine shear rate from given shear stress', normalized='determine shear rate from given shear stress', anchor_ids=['a-1']),
        ],
        anchor_ids=['a-1'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-1'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    trace = PaperLogicTrace(
        trace_id='paper-solver:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-30T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-solver',
            title='Solver Demo',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[move_solver],
        ),
        quality={},
    )

    seed = build_route_state_seed(trace)

    assert seed['dominant_method_candidates'] == ['determine shear rate from given shear stress']


def test_build_route_state_seed_omits_setup_context_method_fillers_when_core_methods_exist() -> None:
    move_core_1 = ResearchMove(
        move_id='m-method-1',
        sequence_no=1,
        role='method',
        act_type='propose_method',
        summary='The paper proposes a Prandtl mixing length approach as an alternative description for dense granular flows.',
        methods=[
            MentionValue(surface='Prandtl mixing length approach', normalized='prandtl mixing length approach', anchor_ids=['a-1']),
        ],
        anchor_ids=['a-1'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-1'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    move_core_2 = ResearchMove(
        move_id='m-method-2',
        sequence_no=2,
        role='method',
        act_type='propose_method',
        summary='The approach shows that a local rheology model can be used as a particular case of the mixing length description.',
        methods=[
            MentionValue(surface='local rheology model', normalized='local rheology model', anchor_ids=['a-2']),
        ],
        anchor_ids=['a-2'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-2'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    move_setup = ResearchMove(
        move_id='m-method-3',
        sequence_no=3,
        role='method',
        act_type='propose_method',
        summary='Describes a simplified experimental geometry with controlled flow rate and rough walls to create a uniform flow driven by gravity.',
        methods=[
            MentionValue(surface='controlled flow rate via aperture or moving wall', normalized='controlled flow rate via aperture or moving wall', anchor_ids=['a-3']),
            MentionValue(surface='silo flow', normalized='silo flow', anchor_ids=['a-3']),
            MentionValue(surface='gravity-driven flow', normalized='gravity-driven flow', anchor_ids=['a-3']),
        ],
        anchor_ids=['a-3'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-3'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='methods', value_index=1, anchor_ids=['a-3'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='methods', value_index=2, anchor_ids=['a-3'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    move_analysis = ResearchMove(
        move_id='m-method-4',
        sequence_no=4,
        role='method',
        act_type='propose_method',
        summary='The work proceeds by comparing data from different experiments in the same geometry and by performing a transverse analysis across configurations.',
        methods=[
            MentionValue(surface='cross-experiment comparison within geometry', normalized='cross-experiment comparison within geometry', anchor_ids=['a-4']),
            MentionValue(surface='cross-configuration analysis', normalized='cross-configuration analysis', anchor_ids=['a-4']),
        ],
        anchor_ids=['a-4'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-4'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='methods', value_index=1, anchor_ids=['a-4'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    trace = PaperLogicTrace(
        trace_id='paper-1901:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-30T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-1901',
            title='On dense granular flows',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[move_core_1, move_core_2, move_setup, move_analysis],
        ),
        quality={},
    )

    seed = build_route_state_seed(trace)

    assert seed['dominant_method_candidates'] == [
        'prandtl mixing length approach',
        'local rheology model',
    ]


def test_build_route_state_seed_keeps_setup_context_method_label_when_it_is_the_only_grounded_signal() -> None:
    move_setup = ResearchMove(
        move_id='m-method-1',
        sequence_no=1,
        role='method',
        act_type='propose_method',
        summary='Describes a simplified experimental geometry with controlled flow rate and rough walls to create a uniform flow driven by gravity.',
        methods=[
            MentionValue(surface='controlled flow rate via aperture or moving wall', normalized='controlled flow rate via aperture or moving wall', anchor_ids=['a-1']),
            MentionValue(surface='silo flow', normalized='silo flow', anchor_ids=['a-1']),
        ],
        anchor_ids=['a-1'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-1'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='methods', value_index=1, anchor_ids=['a-1'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    trace = PaperLogicTrace(
        trace_id='paper-setup-only:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-30T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-setup-only',
            title='Setup Demo',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[move_setup],
        ),
        quality={},
    )

    seed = build_route_state_seed(trace)

    assert seed['dominant_method_candidates'] == ['controlled flow rate via aperture or moving wall']


def test_build_route_state_seed_falls_back_to_inferred_topic_objects_when_trusted_objects_are_missing() -> None:
    result_move = ResearchMove(
        move_id='m-result-1',
        sequence_no=1,
        role='result',
        act_type='report_effect',
        summary='This work extends the data-driven strategy from nonlinear elasticity to scenarios involving internal variables.',
        research_objects=[
            MentionValue(surface='nonlinear elasticity', normalized='nonlinear elasticity', inferred=True, anchor_ids=['a-1']),
            MentionValue(surface='internal variables', normalized='internal variables', inferred=True, anchor_ids=['a-1']),
        ],
        methods=[MentionValue(surface='data-driven strategy', normalized='data-driven strategy', anchor_ids=['a-1'])],
        anchor_ids=['a-1'],
        slot_provenance=[
            SlotProvenance(
                field='research_objects',
                value_index=0,
                anchor_ids=['a-1'],
                extraction_mode='inferred',
                support_strength='weak',
            ),
            SlotProvenance(
                field='research_objects',
                value_index=1,
                anchor_ids=['a-1'],
                extraction_mode='inferred',
                support_strength='weak',
            ),
            SlotProvenance(
                field='methods',
                value_index=0,
                anchor_ids=['a-1'],
                extraction_mode='direct',
                support_strength='strong',
            ),
        ],
    )
    trace = PaperLogicTrace(
        trace_id='paper-1:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-29T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-1',
            title='Data-Driven Computational Plasticity',
            paper_type='theoretical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(moves=[result_move]),
        quality={},
    )

    seed = build_route_state_seed(trace)

    assert 'nonlinear elasticity' in seed['topic_scope_candidates']
    assert 'internal variables' in seed['topic_scope_candidates']
    assert 'a-1' in seed['supporting_evidence_ids']


def test_build_route_state_seed_ranks_specific_inferred_topic_objects_above_broad_background_terms() -> None:
    broad_problem_move = ResearchMove(
        move_id='m-problem-0',
        sequence_no=1,
        role='problem',
        act_type='identify_gap',
        summary='Big-data is becoming a key protagonist in scientific computing.',
        research_objects=[
            MentionValue(surface='big-data', normalized='big-data', inferred=True, anchor_ids=['a-0']),
        ],
        anchor_ids=['a-0'],
        slot_provenance=[
            SlotProvenance(
                field='research_objects',
                value_index=0,
                anchor_ids=['a-0'],
                extraction_mode='inferred',
                support_strength='weak',
            ),
        ],
    )
    challenge_move = ResearchMove(
        move_id='m-problem-1',
        sequence_no=2,
        role='problem',
        act_type='identify_gap',
        summary='The challenge is to avoid relying on a constitutive model when simulating from data.',
        research_objects=[
            MentionValue(surface='constitutive model', normalized='constitutive model', inferred=True, anchor_ids=['a-1']),
        ],
        anchor_ids=['a-1'],
        slot_provenance=[
            SlotProvenance(
                field='research_objects',
                value_index=0,
                anchor_ids=['a-1'],
                extraction_mode='inferred',
                support_strength='weak',
            ),
        ],
    )
    result_move = ResearchMove(
        move_id='m-result-2',
        sequence_no=3,
        role='result',
        act_type='report_effect',
        summary='This work extends the data-driven strategy from nonlinear elasticity to scenarios involving internal variables.',
        research_objects=[
            MentionValue(surface='nonlinear elasticity', normalized='nonlinear elasticity', inferred=True, anchor_ids=['a-2']),
            MentionValue(surface='internal variables', normalized='internal variables', inferred=True, anchor_ids=['a-2']),
        ],
        methods=[MentionValue(surface='data-driven strategy', normalized='data-driven strategy', anchor_ids=['a-2'])],
        anchor_ids=['a-2'],
        slot_provenance=[
            SlotProvenance(
                field='research_objects',
                value_index=0,
                anchor_ids=['a-2'],
                extraction_mode='inferred',
                support_strength='weak',
            ),
            SlotProvenance(
                field='research_objects',
                value_index=1,
                anchor_ids=['a-2'],
                extraction_mode='inferred',
                support_strength='weak',
            ),
            SlotProvenance(
                field='methods',
                value_index=0,
                anchor_ids=['a-2'],
                extraction_mode='direct',
                support_strength='strong',
            ),
        ],
    )
    method_like_result_move = ResearchMove(
        move_id='m-result-3',
        sequence_no=4,
        role='result',
        act_type='report_effect',
        summary='The LaTIn technique is suggested for problem discretization.',
        research_objects=[
            MentionValue(surface='latin technique', normalized='latin technique', inferred=True, anchor_ids=['a-3']),
        ],
        methods=[MentionValue(surface='LaTIn', normalized='latin technique', anchor_ids=['a-3'])],
        anchor_ids=['a-3'],
        slot_provenance=[
            SlotProvenance(
                field='research_objects',
                value_index=0,
                anchor_ids=['a-3'],
                extraction_mode='inferred',
                support_strength='weak',
            ),
            SlotProvenance(
                field='methods',
                value_index=0,
                anchor_ids=['a-3'],
                extraction_mode='direct',
                support_strength='strong',
            ),
        ],
    )
    trace = PaperLogicTrace(
        trace_id='paper-1b:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-29T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-1b',
            title='Data-Driven Computational Plasticity',
            paper_type='theoretical',
            source_refs=['a-0', 'a-1', 'a-2', 'a-3'],
        ),
        canonical_core=CanonicalCore(
            moves=[broad_problem_move, challenge_move, result_move, method_like_result_move],
        ),
        quality={},
    )

    seed = build_route_state_seed(trace)

    assert seed['topic_scope_candidates'].index('constitutive model') < seed['topic_scope_candidates'].index('big-data')
    assert seed['topic_scope_candidates'].index('nonlinear elasticity') < seed['topic_scope_candidates'].index('big-data')
    assert seed['topic_scope_candidates'].index('internal variables') < seed['topic_scope_candidates'].index('latin technique')


def test_build_derived_views_keeps_single_paper_summary_surface_while_adding_compiler_contract() -> None:
    trace = PaperLogicTrace(
        trace_id='paper-1:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-27T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-1',
            title='Demo',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[_build_move(), _build_result_move()],
        ),
        quality={},
    )

    derived = build_derived_views(trace)

    assert 'paper_summaries' in derived
    assert derived['paper_summaries']['one_paragraph_summary']
    assert 'route_compiler_contract' in derived
    assert derived['route_compiler_contract']['paper_id'] == 'paper-1'
    assert 'route_state_seed' in derived
    assert derived['route_state_seed']['paper_id'] == 'paper-1'


def test_build_paper_summaries_prefers_content_moves_over_author_line_noise() -> None:
    noisy_background = ResearchMove(
        move_id='m-noise',
        sequence_no=1,
        role='background',
        act_type='build_resource',
        summary='Dr. I.C.',
        anchor_ids=['a-0'],
    )
    useful_method = ResearchMove(
        move_id='m-method',
        sequence_no=2,
        role='method',
        act_type='propose_method',
        summary='The paper proposes a finite-element compaction workflow for powder compaction.',
        methods=[MentionValue(surface='finite-element compaction workflow', anchor_ids=['a-1'])],
        anchor_ids=['a-1'],
        slot_provenance=[
            SlotProvenance(
                field='methods',
                value_index=0,
                anchor_ids=['a-1'],
                extraction_mode='direct',
                support_strength='strong',
            )
        ],
    )
    useful_result = ResearchMove(
        move_id='m-result',
        sequence_no=3,
        role='result',
        act_type='report_effect',
        summary='The model predicts density distributions and die wear trends observed in experiments.',
        metrics=[MentionValue(surface='density distributions', anchor_ids=['a-2'])],
        anchor_ids=['a-2'],
        slot_provenance=[
            SlotProvenance(
                field='metrics',
                value_index=0,
                anchor_ids=['a-2'],
                extraction_mode='direct',
                support_strength='strong',
            )
        ],
    )
    useful_limitation = ResearchMove(
        move_id='m-limit',
        sequence_no=4,
        role='limitation',
        act_type='state_limitation',
        summary='Current isotropic compaction models cannot predict crack initiation reliably.',
        limitation_types=[MentionValue(surface='crack prediction limit', anchor_ids=['a-3'])],
        anchor_ids=['a-3'],
        slot_provenance=[
            SlotProvenance(
                field='limitation_types',
                value_index=0,
                anchor_ids=['a-3'],
                extraction_mode='direct',
                support_strength='strong',
            )
        ],
    )
    trace = PaperLogicTrace(
        trace_id='paper-2:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-27T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-2',
            title='Modelling Powder Compaction',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[noisy_background, useful_method, useful_result, useful_limitation],
        ),
        quality={},
    )

    derived = build_derived_views(trace)

    assert 'Dr. I.C.' not in derived['paper_summaries']['one_paragraph_summary']
    assert derived['paper_summaries']['key_method_summary'] == useful_method.summary
    assert useful_result.summary in derived['paper_summaries']['one_paragraph_summary']


def test_build_paper_summaries_prefers_method_focused_sentence_over_outcome_colored_sentence() -> None:
    noisy_method = ResearchMove(
        move_id='m-method-1',
        sequence_no=1,
        role='method',
        act_type='propose_method',
        summary='The LES workflow including passive scalar transport and particle tracking gave detailed insight into complex dissolution phenomena.',
        methods=[MentionValue(surface='LES workflow', normalized='les workflow', anchor_ids=['a-1'])],
        metrics=[MentionValue(surface='dissolution phenomena', normalized='dissolution phenomena', anchor_ids=['a-1'])],
        anchor_ids=['a-1'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-1'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='metrics', value_index=0, anchor_ids=['a-1'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    focused_method = ResearchMove(
        move_id='m-method-2',
        sequence_no=2,
        role='method',
        act_type='propose_method',
        summary='The simulation uses LES with passive scalar transport and particle tracking to model dissolution in a stirred tank.',
        methods=[MentionValue(surface='large eddy simulation', normalized='large eddy simulation', anchor_ids=['a-2'])],
        anchor_ids=['a-2'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-2'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    trace = PaperLogicTrace(
        trace_id='paper-3:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-28T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-3',
            title='LES Dissolution Demo',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[noisy_method, focused_method],
        ),
        quality={},
    )

    summaries = build_paper_summaries(trace)

    assert summaries['key_method_summary'] == focused_method.summary


def test_build_paper_summaries_prefers_operative_method_over_governing_equation_context() -> None:
    operative_method = ResearchMove(
        move_id='m-method-operative',
        sequence_no=1,
        role='method',
        act_type='propose_method',
        summary='The authors employ an algorithm to mimic stress-controlled rheology, building on an extended Stokesian Dynamics approach.',
        methods=[
            MentionValue(surface='stress-controlled rheology algorithm', normalized='stress-controlled rheology algorithm', anchor_ids=['a-1']),
            MentionValue(surface='Stokesian Dynamics', normalized='stokesian dynamics', anchor_ids=['a-1']),
        ],
        anchor_ids=['a-1'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-1'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='methods', value_index=1, anchor_ids=['a-1'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    governing_context = ResearchMove(
        move_id='m-method-governing',
        sequence_no=2,
        role='method',
        act_type='adapt_method',
        summary='Describes the governing equations for dense suspensions under stress-controlled quasi-static conditions, adapting hydrodynamic and non-hydrodynamic interaction models.',
        methods=[
            MentionValue(surface='force and torque balance equations', normalized='force and torque balance equations', anchor_ids=['a-2']),
            MentionValue(surface='linear resistance', normalized='linear resistance', anchor_ids=['a-2']),
            MentionValue(surface='pair-wise hydrodynamic lubrication', normalized='pair-wise hydrodynamic lubrication', anchor_ids=['a-2']),
            MentionValue(surface='regularization with cutoff length', normalized='regularization with cutoff length', anchor_ids=['a-2']),
        ],
        anchor_ids=['a-2'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-2'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='methods', value_index=1, anchor_ids=['a-2'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='methods', value_index=2, anchor_ids=['a-2'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='methods', value_index=3, anchor_ids=['a-2'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    supporting_method = ResearchMove(
        move_id='m-method-supporting',
        sequence_no=3,
        role='method',
        act_type='propose_method',
        summary='The authors model contact forces using a soft-constraint approach with harmonic penalty functions and a Coulomb friction law.',
        methods=[
            MentionValue(surface='soft-constraint approach', normalized='soft-constraint approach', anchor_ids=['a-3']),
        ],
        anchor_ids=['a-3'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-3'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    trace = PaperLogicTrace(
        trace_id='paper-method-governing:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-30T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-method-governing',
            title='Shear jamming and fragility in dense suspensions',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[operative_method, governing_context, supporting_method],
        ),
        quality={},
    )

    summaries = build_paper_summaries(trace)

    assert summaries['key_method_summary'] == operative_method.summary


def test_build_paper_summaries_prefers_grounded_method_move_over_setup_condition_sentence() -> None:
    setup_condition = ResearchMove(
        move_id='m-method-setup-condition',
        sequence_no=1,
        role='method',
        act_type='set_condition',
        summary='Describes the stress distribution in the experimental geometry under equilibrium and Janssen effect assumptions, stating constant normal stress and linearly varying tangential stress with distance from the walls.',
        research_objects=[
            MentionValue(surface='stress distribution in the experimental geometry', normalized='stress distribution in the experimental geometry', anchor_ids=['a-1']),
        ],
        conditions=[
            MentionValue(surface='high column with janssen effect', normalized='high column with janssen effect', anchor_ids=['a-1']),
        ],
        anchor_ids=['a-1'],
        slot_provenance=[
            SlotProvenance(field='research_objects', value_index=0, anchor_ids=['a-1'], extraction_mode='direct', support_strength='strong'),
            SlotProvenance(field='conditions', value_index=0, anchor_ids=['a-1'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    grounded_method = ResearchMove(
        move_id='m-method-grounded',
        sequence_no=2,
        role='method',
        act_type='propose_method',
        summary='The paper proposes a Prandtl mixing length approach as an alternative description for dense granular flows, generalizing the Bagnold shear stress by introducing a coherence length scale l instead of the grain diameter d.',
        methods=[
            MentionValue(surface='Prandtl mixing length approach', normalized='prandtl mixing length approach', anchor_ids=['a-2']),
        ],
        effects=[
            EffectValue(direction='unknown', object='dense granular flows'),
        ],
        anchor_ids=['a-2'],
        slot_provenance=[
            SlotProvenance(field='methods', value_index=0, anchor_ids=['a-2'], extraction_mode='direct', support_strength='strong'),
        ],
    )
    trace = PaperLogicTrace(
        trace_id='paper-method-grounded:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-30T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-method-grounded',
            title='On dense granular flows',
            paper_type='empirical',
            source_refs=['src-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[setup_condition, grounded_method],
            evidence_anchors=[
                EvidenceAnchor(
                    anchor_id='a-1',
                    paper_id='paper-method-grounded',
                    source_ref='src-1',
                    modality='text',
                    section_path=['Methods'],
                    quote='Stress distribution is characterized in the experimental geometry.',
                    support_type='direct',
                ),
                EvidenceAnchor(
                    anchor_id='a-2',
                    paper_id='paper-method-grounded',
                    source_ref='src-1',
                    modality='text',
                    section_path=['Introduction'],
                    quote='A Prandtl mixing length approach is proposed.',
                    support_type='direct',
                ),
            ],
        ),
        quality={},
    )

    summaries = build_paper_summaries(trace)

    assert summaries['key_method_summary'] == grounded_method.summary


def test_build_paper_summaries_prefers_language_coherent_interpretation_when_scores_are_similar() -> None:
    problem_zh = ResearchMove(
        move_id='m-problem-zh',
        sequence_no=1,
        role='problem',
        act_type='define_task',
        summary='本文针对固体火箭发动机装药结构完整性评估问题，系统梳理现有建模路线、验证方式以及工程应用中的主要分析挑战。',
        anchor_ids=['a-1'],
    )
    method_zh = ResearchMove(
        move_id='m-method-zh',
        sequence_no=2,
        role='method',
        act_type='propose_method',
        summary='文献综述了结构完整性数值仿真中采用的模型、方法与软件平台，并比较了不同路线在工程实现中的适用边界。',
        anchor_ids=['a-2'],
    )
    interpretation_en = ResearchMove(
        move_id='m-interp-en',
        sequence_no=3,
        role='interpretation',
        act_type='explain_mechanism',
        summary='The paper interprets the critical importance of structural integrity analysis for motor grains by linking grain failure to mission failure rates.',
        anchor_ids=['a-3'],
    )
    interpretation_zh = ResearchMove(
        move_id='m-interp-zh',
        sequence_no=4,
        role='interpretation',
        act_type='explain_mechanism',
        summary='文章进一步解释了结构完整性分析为何会直接影响发动机任务可靠性，并将这一点与任务失效率和结构失效风险联系起来。',
        anchor_ids=['a-4'],
    )
    trace = PaperLogicTrace(
        trace_id='paper-3b:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-29T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-3b',
            title='Recent Progress upon Structural Integrity Analysis of Solid Rocket Motor Grain',
            title_alt='固体火箭发动机装药结构完整性研究进展',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[problem_zh, method_zh, interpretation_en, interpretation_zh],
        ),
        quality={},
    )

    summaries = build_paper_summaries(trace)

    assert interpretation_zh.summary in summaries['one_paragraph_summary']
    assert interpretation_en.summary not in summaries['one_paragraph_summary']


def test_build_paper_summaries_prefers_complete_single_language_bundle_over_mixed_bundle() -> None:
    problem_zh = ResearchMove(
        move_id='m-problem-zh-2',
        sequence_no=1,
        role='problem',
        act_type='identify_gap',
        summary='目前，在工艺与结构方案设计过程中，搅拌器布置以及曝气等因素在设计初期没有明确且统一的设计思路。',
        anchor_ids=['a-1'],
    )
    problem_en = ResearchMove(
        move_id='m-problem-en-2',
        sequence_no=2,
        role='problem',
        act_type='identify_gap',
        summary='Current design of submersible agitators relies on empirical power consumption data, which can lead to inefficient layouts and excessive energy consumption.',
        anchor_ids=['a-2'],
    )
    method_en = ResearchMove(
        move_id='m-method-en-2',
        sequence_no=3,
        role='method',
        act_type='propose_method',
        summary='The simulation method uses the momentum source method together with the Multi-Reference Frame method to analyze pool layouts and agitator operating conditions.',
        anchor_ids=['a-3'],
    )
    result_en = ResearchMove(
        move_id='m-result-en-2',
        sequence_no=4,
        role='result',
        act_type='report_effect',
        summary='The results show that cylindrical columns reduce flow resistance and rounded corners improve the overall flow pattern.',
        anchor_ids=['a-4'],
    )
    trace = PaperLogicTrace(
        trace_id='paper-3c:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-29T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-3c',
            title='Application of CFD simulation in submersible agitator layout and optimization of operating conditions',
            title_alt='CFD 模拟在潜水搅拌设备布置及工况优化中的应用',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[problem_zh, problem_en, method_en, result_en],
        ),
        quality={},
    )

    summaries = build_paper_summaries(trace)

    assert problem_en.summary in summaries['one_paragraph_summary']
    assert method_en.summary in summaries['one_paragraph_summary']
    assert result_en.summary in summaries['one_paragraph_summary']
    assert problem_zh.summary not in summaries['one_paragraph_summary']


def test_build_paper_summaries_can_use_later_opening_move_to_preserve_language_bundle() -> None:
    problem_zh = ResearchMove(
        move_id='m-problem-zh-3',
        sequence_no=1,
        role='problem',
        act_type='identify_gap',
        summary='针对固体火箭发动机装药结构完整性的计算与评估问题，总结相关模型与分析挑战。',
        anchor_ids=['a-1'],
    )
    method_zh = ResearchMove(
        move_id='m-method-zh-3',
        sequence_no=2,
        role='method',
        act_type='propose_method',
        summary='文献综述了结构完整性研究中常见的数值仿真方法与软件平台。',
        anchor_ids=['a-2'],
    )
    result_en = ResearchMove(
        move_id='m-result-en-3',
        sequence_no=3,
        role='result',
        act_type='report_effect',
        summary='The paper explains that structural integrity analysis is crucial because grain failure contributes to the vast majority of engine failures.',
        anchor_ids=['a-3'],
    )
    problem_en = ResearchMove(
        move_id='m-problem-en-3',
        sequence_no=5,
        role='problem',
        act_type='identify_gap',
        summary='Composite solid propellants exhibit complex viscoelastic and damage behavior, creating major challenges for structural integrity analysis in engineering practice.',
        anchor_ids=['a-4'],
    )
    method_en = ResearchMove(
        move_id='m-method-en-3',
        sequence_no=6,
        role='method',
        act_type='adapt_method',
        summary='Summarizes numerical simulation methods for structural integrity analysis, including defect analysis, grain optimization, and finite element program development.',
        anchor_ids=['a-5'],
    )
    trace = PaperLogicTrace(
        trace_id='paper-3d:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-29T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-3d',
            title='Recent Progress upon Structural Integrity Analysis of Solid Rocket Motor Grain',
            title_alt='固体火箭发动机装药结构完整性研究进展',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[problem_zh, method_zh, result_en, problem_en, method_en],
        ),
        quality={},
    )

    summaries = build_paper_summaries(trace)

    assert problem_en.summary in summaries['one_paragraph_summary']
    assert method_en.summary in summaries['one_paragraph_summary']
    assert result_en.summary in summaries['one_paragraph_summary']
    assert problem_zh.summary not in summaries['one_paragraph_summary']


def test_build_paper_summaries_prefers_title_aligned_opening_over_generic_background() -> None:
    generic_problem = ResearchMove(
        move_id='m-problem-generic-title',
        sequence_no=1,
        role='problem',
        act_type='identify_gap',
        summary='Chemical absorption remains a central CCUS strategy, but high-viscosity absorbents create practical transport challenges in industrial process design.',
        anchor_ids=['a-1'],
    )
    aligned_problem = ResearchMove(
        move_id='m-problem-aligned-title',
        sequence_no=2,
        role='problem',
        act_type='define_task',
        summary='This study investigates twin-liquid film behavior on a spiked rotating disk reactor under highly viscous fluid conditions.',
        anchor_ids=['a-2'],
    )
    method_move = ResearchMove(
        move_id='m-method-title',
        sequence_no=3,
        role='method',
        act_type='propose_method',
        summary='The paper applies the Volume of Fluid method to simulate laminar incompressible flow and track the gas-liquid interface in the reactor.',
        anchor_ids=['a-3'],
    )
    result_move = ResearchMove(
        move_id='m-result-title',
        sequence_no=4,
        role='result',
        act_type='report_effect',
        summary='The open-window region drastically reduces film thickness and alters velocity fluctuations across the reactor surface.',
        anchor_ids=['a-4'],
    )
    trace = PaperLogicTrace(
        trace_id='paper-title-align-opening:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-29T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-title-align-opening',
            title='Numerical investigation of twin-liquid film on spiked rotating disk reactor with highly viscous fluid',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[generic_problem, aligned_problem, method_move, result_move],
        ),
        quality={},
    )

    summaries = build_paper_summaries(trace)

    assert aligned_problem.summary in summaries['one_paragraph_summary']
    assert generic_problem.summary not in summaries['one_paragraph_summary']


def test_build_paper_summaries_prefers_title_aligned_method_over_generic_method_context() -> None:
    problem_move = ResearchMove(
        move_id='m-problem-method-title',
        sequence_no=1,
        role='problem',
        act_type='identify_gap',
        summary='Analytical models for composite laminates remain limited to specific geometries, loading cases, and boundary conditions.',
        anchor_ids=['a-1'],
    )
    generic_method = ResearchMove(
        move_id='m-method-generic-title',
        sequence_no=2,
        role='method',
        act_type='propose_method',
        summary='Machine learning methods can provide faster surrogate models by learning input-output mappings without repeatedly solving governing equations.',
        anchor_ids=['a-2'],
    )
    aligned_method = ResearchMove(
        move_id='m-method-aligned-title',
        sequence_no=3,
        role='method',
        act_type='propose_method',
        summary='Image-based neural networks are used to predict mechanical field distributions in composite microstructures under applied strain.',
        anchor_ids=['a-3'],
    )
    result_move = ResearchMove(
        move_id='m-result-method-title',
        sequence_no=4,
        role='result',
        act_type='report_effect',
        summary='The integrated model improves stress-field accuracy relative to earlier surrogate baselines.',
        anchor_ids=['a-4'],
    )
    trace = PaperLogicTrace(
        trace_id='paper-title-align-method:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-29T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-title-align-method',
            title='Integrated convolutional and graph neural networks for predicting mechanical fields in composite microstructures',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[problem_move, generic_method, aligned_method, result_move],
        ),
        quality={},
    )

    summaries = build_paper_summaries(trace)

    assert aligned_method.summary in summaries['one_paragraph_summary']
    assert generic_method.summary not in summaries['one_paragraph_summary']


def test_build_paper_content_audit_flags_suspicious_title_alt_and_missing_findings() -> None:
    trace = PaperLogicTrace(
        trace_id='paper-4:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-28T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='paper-4',
            title='Numerical simulation of a dissolution process in a stirred tank reactor',
            title_alt='3. Simulation procedure',
            paper_type='empirical',
            source_refs=['a-1'],
        ),
        canonical_core=CanonicalCore(
            moves=[_build_problem_move(), _build_move()],
        ),
        quality={},
    )

    profile = build_paper_content_profile(trace)
    audit = build_paper_content_audit(trace, paper_content_profile=profile)
    derived = build_derived_views(trace)

    assert 'suspicious_title_alt' in audit['flags']
    assert 'missing_key_findings' in audit['flags']
    assert audit['title_alt_heading_like'] is True
    assert audit['coverage']['method_statement_count'] == 1
    assert derived['paper_content_audit']['flags'] == audit['flags']


def test_build_paper_content_audit_flags_pipe_and_formula_style_title_alt_noise() -> None:
    for title_alt in (
        '2 | REVIEW OF DATA-DRIVEN SCHEMES',
        '$\\mathrm{M = (err./t_test)^{*}100;}\\%$ Relative prediction error',
    ):
        trace = PaperLogicTrace(
            trace_id='paper-5:paper_logic_trace',
            schema_version='v2',
            built_at='2026-03-28T00:00:00Z',
            paper_metadata=PaperMetadata(
                paper_id='paper-5',
                title='Data-driven computing in dynamics',
                title_alt=title_alt,
                paper_type='empirical',
                source_refs=['a-1'],
            ),
            canonical_core=CanonicalCore(
                moves=[_build_problem_move(), _build_move()],
            ),
            quality={},
        )

        audit = build_paper_content_audit(trace)

        assert 'suspicious_title_alt' in audit['flags']
        assert audit['title_alt_heading_like'] is True
