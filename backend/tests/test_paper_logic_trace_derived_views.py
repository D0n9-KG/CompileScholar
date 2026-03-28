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
from app.paper_logic_trace.models import CanonicalCore, CitationAct, EffectValue, FigureRef, MentionValue, PaperLogicTrace, PaperMetadata, ResearchMove, SlotProvenance, TableRef


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
