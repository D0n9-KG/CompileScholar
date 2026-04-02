from __future__ import annotations

import pytest

from app.paper_logic_trace.derived_views import build_derived_views
from app.paper_logic_trace.models import CanonicalCore, MentionValue, PaperLogicTrace, PaperMetadata, ResearchMove, SlotProvenance
from app.research_logic.decision_prior_builder import build_decision_prior_card
from app.research_logic.historical_replay_compiler import HistoricalReplayCompiler, compile_historical_replay
from app.research_logic.models import RouteState
from app.research_logic.route_comparison_builder import build_route_comparison_case


def _mention(surface: str, normalized: str, anchor_id: str, *, mention_type: str | None = None) -> MentionValue:
    return MentionValue(
        surface=surface,
        normalized=normalized,
        type=mention_type,
        anchor_ids=[anchor_id],
    )


def _prov(field: str, anchor_id: str, value_index: int) -> SlotProvenance:
    return SlotProvenance(
        field=field,
        value_index=value_index,
        anchor_ids=[anchor_id],
        extraction_mode='direct',
        support_strength='strong',
    )


def _method_move(prefix: str) -> ResearchMove:
    return ResearchMove(
        move_id=f'{prefix}-m1',
        sequence_no=1,
        role='method',
        act_type='propose_method',
        summary='Uses graph neural network modeling for retrieval.',
        research_objects=[_mention('entity relation graph', 'entity relation graph', f'{prefix}-a1')],
        methods=[_mention('graph neural network', 'graph neural network', f'{prefix}-a2')],
        metrics=[_mention('MRR', 'mean reciprocal rank', f'{prefix}-a3')],
        conditions=[_mention('under high compression', 'under high compression', f'{prefix}-a4')],
        comparators=[_mention('baseline retriever', 'baseline retriever', f'{prefix}-a5')],
        limitation_types=[_mention('computational cost', 'computational cost', f'{prefix}-a6')],
        resource_mentions=[_mention('WN18RR', 'wn18rr', f'{prefix}-a7', mention_type='benchmark')],
        anchor_ids=[f'{prefix}-a1', f'{prefix}-a2', f'{prefix}-a3', f'{prefix}-a4', f'{prefix}-a5', f'{prefix}-a6', f'{prefix}-a7'],
        slot_provenance=[
            _prov('research_objects', f'{prefix}-a1', 0),
            _prov('methods', f'{prefix}-a2', 0),
            _prov('metrics', f'{prefix}-a3', 0),
            _prov('conditions', f'{prefix}-a4', 0),
            _prov('comparators', f'{prefix}-a5', 0),
            _prov('limitation_types', f'{prefix}-a6', 0),
            _prov('resource_mentions', f'{prefix}-a7', 0),
        ],
        confidence=0.9,
    )


def _result_move(prefix: str) -> ResearchMove:
    return ResearchMove(
        move_id=f'{prefix}-m2',
        sequence_no=2,
        role='result',
        act_type='report_effect',
        summary='The graph neural network improves MRR compared to the baseline retriever under high compression.',
        methods=[_mention('graph neural network', 'graph neural network', f'{prefix}-a10')],
        metrics=[_mention('MRR', 'mean reciprocal rank', f'{prefix}-a11')],
        comparators=[_mention('baseline retriever', 'baseline retriever', f'{prefix}-a12')],
        conditions=[_mention('under high compression', 'under high compression', f'{prefix}-a13')],
        limitation_types=[_mention('computational cost', 'computational cost', f'{prefix}-a14')],
        anchor_ids=[f'{prefix}-a10', f'{prefix}-a11', f'{prefix}-a12', f'{prefix}-a13', f'{prefix}-a14'],
        slot_provenance=[
            _prov('methods', f'{prefix}-a10', 0),
            _prov('metrics', f'{prefix}-a11', 0),
            _prov('comparators', f'{prefix}-a12', 0),
            _prov('conditions', f'{prefix}-a13', 0),
            _prov('limitation_types', f'{prefix}-a14', 0),
        ],
        confidence=0.92,
    )


def _bridge_move(prefix: str) -> ResearchMove:
    return ResearchMove(
        move_id=f'{prefix}-m3',
        sequence_no=3,
        role='experiment',
        act_type='set_condition',
        summary='Uses X-ray microtomography and YADE software to measure packing density against the dry baseline under high compression.',
        methods=[_mention('discrete element simulation', 'discrete element simulation', f'{prefix}-a20')],
        metrics=[_mention('packing density', 'packing density', f'{prefix}-a21')],
        comparators=[_mention('dry baseline', 'dry baseline', f'{prefix}-a22')],
        conditions=[_mention('under high compression', 'under high compression', f'{prefix}-a23')],
        resource_mentions=[
            _mention('X-ray microtomography', 'x-ray microtomography', f'{prefix}-a24', mention_type='instrument'),
            _mention('YADE', 'yade', f'{prefix}-a25', mention_type='software'),
        ],
        anchor_ids=[f'{prefix}-a20', f'{prefix}-a21', f'{prefix}-a22', f'{prefix}-a23', f'{prefix}-a24', f'{prefix}-a25'],
        slot_provenance=[
            _prov('methods', f'{prefix}-a20', 0),
            _prov('metrics', f'{prefix}-a21', 0),
            _prov('comparators', f'{prefix}-a22', 0),
            _prov('conditions', f'{prefix}-a23', 0),
            _prov('resource_mentions', f'{prefix}-a24', 0),
            _prov('resource_mentions', f'{prefix}-a25', 1),
        ],
        confidence=0.88,
    )


def _future_work_move(prefix: str) -> ResearchMove:
    return ResearchMove(
        move_id=f'{prefix}-m4',
        sequence_no=4,
        role='future_work',
        act_type='suggest_extension',
        summary='Future work should extend graph neural network retrieval to temporal knowledge graphs under noisy labels using web-scale corpora.',
        research_objects=[_mention('temporal knowledge graphs', 'temporal knowledge graph', f'{prefix}-a30')],
        methods=[_mention('graph neural network retrieval', 'graph neural network retrieval', f'{prefix}-a31')],
        conditions=[_mention('under noisy labels', 'under noisy labels', f'{prefix}-a32')],
        limitation_types=[_mention('limited labeled data', 'limited labeled data', f'{prefix}-a33')],
        resource_mentions=[_mention('web-scale corpora', 'web-scale corpora', f'{prefix}-a34', mention_type='dataset')],
        anchor_ids=[f'{prefix}-a30', f'{prefix}-a31', f'{prefix}-a32', f'{prefix}-a33', f'{prefix}-a34'],
        slot_provenance=[
            _prov('research_objects', f'{prefix}-a30', 0),
            _prov('methods', f'{prefix}-a31', 0),
            _prov('conditions', f'{prefix}-a32', 0),
            _prov('limitation_types', f'{prefix}-a33', 0),
            _prov('resource_mentions', f'{prefix}-a34', 0),
        ],
        confidence=0.81,
    )


def _thin_method_move(prefix: str) -> ResearchMove:
    return ResearchMove(
        move_id=f'{prefix}-m1',
        sequence_no=1,
        role='method',
        act_type='propose_method',
        summary='Uses graph neural network modeling for retrieval.',
        research_objects=[_mention('entity relation graph', 'entity relation graph', f'{prefix}-a1')],
        methods=[_mention('graph neural network', 'graph neural network', f'{prefix}-a2')],
        anchor_ids=[f'{prefix}-a1', f'{prefix}-a2'],
        slot_provenance=[
            _prov('research_objects', f'{prefix}-a1', 0),
            _prov('methods', f'{prefix}-a2', 0),
        ],
        confidence=0.86,
    )


def _trace(paper_id: str, year: int, prefix: str, *, thin: bool = False) -> PaperLogicTrace:
    moves = [_thin_method_move(prefix)] if thin else [_method_move(prefix), _result_move(prefix), _bridge_move(prefix), _future_work_move(prefix)]
    trace = PaperLogicTrace(
        trace_id=f'{paper_id}:paper_logic_trace',
        schema_version='v2',
        built_at='2026-04-02T01:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id=paper_id,
            title=f'Demo {paper_id}',
            year=year,
            paper_type='empirical',
            source_refs=[f'{prefix}-src'],
        ),
        canonical_core=CanonicalCore(moves=moves),
        quality={},
    )
    trace.derived_views = build_derived_views(trace)
    return trace


def _packet(traces: list[PaperLogicTrace], *, cutoff_year: int) -> dict:
    return {
        'packet_id': f'packet-{cutoff_year}',
        'built_at': '2026-04-02T01:05:00Z',
        'topic_scope_candidate': 'large-scale image recognition with deep neural networks',
        'cutoff_year': cutoff_year,
        'packet_status': 'frozen',
        'packet_composition': {
            'target_size': len(traces),
            'actual_size': len(traces),
            'role_counts': {
                'core_method': max(1, len(traces)),
                'resource_or_benchmark': 1,
                'limitation_or_critique': 1,
                'survey_or_review': 1,
                'alternative_route': 1,
            },
            'coverage_ok': True,
            'missing_roles': [],
        },
        'l1_snapshot_ref': {
            'snapshot_id': 'imagenet_2011_snapshot',
            'resource_registry_ref': 'l1:resource-registry',
            'resource_timeline_ref': 'l1:resource-timeline',
            'benchmark_timeline_ref': 'l1:benchmark-timeline',
            'toolchain_timeline_ref': 'l1:toolchain-timeline',
            'protocol_registry_ref': 'l1:protocol-registry',
        },
        'inclusion_rules': {
            'scope_definition': 'Compile a bounded route state from packetized traces.',
            'scope_aliases': ['cnn image recognition'],
            'accepted_year_range': {'min_year': 2005, 'max_year': cutoff_year},
            'hard_exclusion_rules': ['exclude papers after cutoff'],
            'role_assignment_rules': ['one primary role per packet item'],
            'leakage_policy': 'Papers after cutoff are excluded and hindsight labels never enter packet compilation.',
        },
        'included_items': [
            {
                'paper_id': trace.paper_metadata.paper_id,
                'trace_id': trace.trace_id,
                'paper_year': trace.paper_metadata.year,
                'item_role': 'core_method' if index == 0 else 'resource_or_benchmark',
                'inclusion_reason': 'Supports packetized route reconstruction.',
                'source_selector': 'manual',
                'evidence_for_inclusion': [trace.paper_metadata.paper_id],
                'title': trace.paper_metadata.title,
            }
            for index, trace in enumerate(traces)
        ],
        'excluded_items': [
            {
                'paper_id': 'future-paper',
                'paper_year': cutoff_year + 1,
                'exclusion_reason': 'after_cutoff',
                'notes': 'Audit only',
            }
        ],
        'packet_quality': {
            'quality_tier': 'green',
            'ready_for_route_state': True,
            'quality_flags': [],
            'topic_boundary_confidence': 0.82,
            'leakage_risk': 'low',
            'manual_review_status': 'completed',
        },
        'compiler_hints': {
            'preferred_scope_label': 'large-scale image recognition with deep neural networks',
            'preferred_method_labels': ['graph neural network'],
            'preferred_benchmark_labels': ['wn18rr'],
            'preferred_bottleneck_labels': ['computational cost'],
            'expected_alternative_routes': ['graph neural network retrieval -> temporal knowledge graph'],
            'notes_for_route_state_compiler': 'Prefer packet-level aggregation over single-paper restatement.',
        },
    }


def _l1_snapshot(*, cutoff_year: int, snapshot_id: str = 'imagenet_2011_snapshot') -> dict:
    return {
        'snapshot_id': snapshot_id,
        'built_at': '2026-04-02T09:00:00Z',
        'topic_scope': 'large-scale image recognition with deep neural networks',
        'cutoff_year': cutoff_year,
        'grounding_mode': 'paper_grounded_l1_lite',
        'source_packet_id': f'packet-{cutoff_year}',
        'source_trace_ids': ['paper-thin:paper_logic_trace'],
        'source_paper_ids': ['paper-thin'],
        'resource_registry': [
            {
                'label': 'imagenet-1k',
                'status': 'available',
                'confidence': 0.82,
                'source_paper_ids': ['paper-thin'],
                'source_trace_ids': ['paper-thin:paper_logic_trace'],
                'source_move_ids': ['pt-m1'],
                'evidence_ids': ['l1-r1'],
                'resource_types': ['dataset'],
            }
        ],
        'benchmark_timeline': [
            {
                'label': 'wn18rr',
                'status': 'available',
                'confidence': 0.8,
                'source_paper_ids': ['paper-thin'],
                'source_trace_ids': ['paper-thin:paper_logic_trace'],
                'source_move_ids': ['pt-m1'],
                'evidence_ids': ['l1-b1'],
                'benchmark_type': 'benchmark',
                'metric_tokens': ['mrr'],
                'comparator_tokens': ['baseline retriever'],
            }
        ],
        'toolchain_timeline': [
            {
                'label': 'cuda stack',
                'status': 'available',
                'confidence': 0.78,
                'source_paper_ids': ['paper-thin'],
                'source_trace_ids': ['paper-thin:paper_logic_trace'],
                'source_move_ids': ['pt-m1'],
                'evidence_ids': ['l1-t1'],
                'resource_types': ['compute'],
                'method_tokens': ['graph neural network'],
            }
        ],
        'protocol_registry': [
            {
                'label': 'top-1 accuracy',
                'status': 'available',
                'confidence': 0.79,
                'source_paper_ids': ['paper-thin'],
                'source_trace_ids': ['paper-thin:paper_logic_trace'],
                'source_move_ids': ['pt-m1'],
                'evidence_ids': ['l1-p1'],
                'metric_tokens': ['top-1 accuracy'],
                'comparator_tokens': ['baseline retriever'],
                'condition_tokens': ['standard split'],
                'method_tokens': ['evaluation'],
                'resource_tokens': ['imagenet-1k'],
            }
        ],
        'unresolved_questions': ['External benchmark governance remains unmodeled.'],
        'quality': {
            'quality_tier': 'yellow',
            'quality_flags': ['paper_only_snapshot'],
            'audit_status': 'eligible',
            'paper_only_snapshot': True,
            'completeness_score': 0.74,
        },
    }


def _route_state_payload(
    *,
    route_state_id: str,
    method_label: str,
    support_ids: list[str],
    challenge_ids: list[str],
    method_score: float,
    measurement_score: float,
    data_resource_score: float,
    infrastructure_score: float,
    cost_cycle_score: float,
    overall_score: float,
    bottleneck_label: str,
    bottleneck_type: str,
    bottleneck_severity: str,
    positive_signal_confidence: float,
    scope: str = 'large-scale image recognition with deep neural networks',
    cutoff_year: int = 2011,
) -> dict:
    return {
        'route_state_id': route_state_id,
        'built_at': '2026-04-02T01:10:00Z',
        'topic_scope': scope,
        'cutoff_year': cutoff_year,
        'source_packet': {
            'packet_id': f'{route_state_id}:packet',
            'included_trace_ids': [f'{route_state_id}:trace-1', f'{route_state_id}:trace-2'],
            'included_paper_ids': [f'{route_state_id}:paper-1', f'{route_state_id}:paper-2'],
            'packet_role_counts': {
                'core_method': 4,
                'resource_or_benchmark': 2,
                'limitation_or_critique': 1,
                'survey_or_review': 1,
                'alternative_route': 1,
            },
            'l1_snapshot_ref': f'{route_state_id}:snapshot',
        },
        'scope_resolution': {
            'topic_scope_candidates': [scope],
            'accepted_scope_label': scope,
            'rejected_scope_labels': [],
            'resolution_rationale': 'Synthetic route state for replay compiler tests.',
            'resolution_evidence_ids': support_ids[:1],
        },
        'route_landscape': {
            'dominant_methods': [
                {
                    'label': method_label,
                    'family': 'image recognition',
                    'maturity_score': method_score,
                    'adoption_level': 'established',
                    'source_move_ids': [f'{route_state_id}:m1'],
                    'source_paper_ids': [f'{route_state_id}:paper-1', f'{route_state_id}:paper-2'],
                    'evidence_ids': support_ids[:1],
                }
            ],
            'active_benchmarks': [
                {
                    'label': f'{route_state_id}:benchmark',
                    'benchmark_type': 'benchmark',
                    'adoption_level': 'active',
                    'source_paper_ids': [f'{route_state_id}:paper-1'],
                    'evidence_ids': support_ids[:1],
                }
            ],
            'measurement_protocols': [
                {
                    'label': f'{route_state_id}:top-1-accuracy',
                    'protocol_type': 'measurement',
                    'maturity_score': measurement_score,
                    'source_paper_ids': [f'{route_state_id}:paper-1'],
                    'evidence_ids': support_ids[1:2] or support_ids[:1],
                }
            ],
            'toolchains_and_infrastructure': [
                {
                    'label': f'{route_state_id}:gpu-stack',
                    'infra_type': 'compute',
                    'availability_level': 'usable',
                    'source_paper_ids': [f'{route_state_id}:paper-2'],
                    'evidence_ids': support_ids[1:2] or support_ids[:1],
                }
            ],
            'known_capabilities': [
                {
                    'label': f'{route_state_id}:classification-gain',
                    'capability_type': 'prediction',
                    'status': 'repeatable',
                    'metric_signals': ['top-1 accuracy'],
                    'condition_signals': ['benchmark scale'],
                    'source_paper_ids': [f'{route_state_id}:paper-1'],
                    'evidence_ids': support_ids[:1],
                }
            ],
            'known_bottlenecks': [
                {
                    'label': bottleneck_label,
                    'bottleneck_type': bottleneck_type,
                    'severity': bottleneck_severity,
                    'blocking_scope': 'route_level',
                    'source_paper_ids': [f'{route_state_id}:paper-2'],
                    'evidence_ids': challenge_ids[:1],
                    'counterevidence_ids': [],
                }
            ],
            'enabling_conditions': [
                {
                    'label': f'{route_state_id}:benchmark-data-available',
                    'condition_type': 'data',
                    'status': 'met',
                    'source_paper_ids': [f'{route_state_id}:paper-1'],
                    'evidence_ids': support_ids[1:2] or support_ids[:1],
                }
            ],
            'alternative_routes': [
                {
                    'label': 'feature-engineering pipeline',
                    'route_family': 'alternative',
                    'relation_to_main_route': 'competing',
                    'distinguishing_features': ['distinct optimization regime'],
                    'source_paper_ids': [f'{route_state_id}:paper-3'],
                    'evidence_ids': [f'{route_state_id}:alt-evidence'],
                }
            ],
        },
        'readiness_scores': {
            'theory': 0.62,
            'method': method_score,
            'measurement': measurement_score,
            'data_resource': data_resource_score,
            'infrastructure': infrastructure_score,
            'community': 0.6,
            'cost_cycle': cost_cycle_score,
            'overall': overall_score,
            'score_rationale': 'Synthetic payload for replay compiler tests.',
        },
        'why_now_features': {
            'unlocking_factors': [
                {
                    'label': f'{route_state_id}:benchmark-scale-data',
                    'feature_type': 'benchmark_availability',
                    'direction': 'unlock',
                    'source_paper_ids': [f'{route_state_id}:paper-1'],
                    'evidence_ids': support_ids[:1],
                    'l1_refs': [f'{route_state_id}:l1-benchmark'],
                    'confidence': 0.9,
                }
            ],
            'acceleration_factors': [
                {
                    'label': f'{route_state_id}:method-improving',
                    'feature_type': 'method_maturity',
                    'direction': 'accelerate',
                    'source_paper_ids': [f'{route_state_id}:paper-1'],
                    'evidence_ids': support_ids[:1],
                    'l1_refs': [],
                    'confidence': max(method_score, 0.5),
                }
            ],
            'positive_comparison_signals': [
                {
                    'label': f'{route_state_id}:comparative-upside',
                    'feature_type': 'comparative_gain',
                    'direction': 'accelerate',
                    'source_paper_ids': [f'{route_state_id}:paper-3'],
                    'evidence_ids': [f'{route_state_id}:alt-evidence'],
                    'l1_refs': [],
                    'confidence': positive_signal_confidence,
                }
            ],
        },
        'not_now_features': {
            'blocking_factors': [
                {
                    'label': bottleneck_label,
                    'feature_type': 'blocker',
                    'direction': 'block',
                    'source_paper_ids': [f'{route_state_id}:paper-2'],
                    'evidence_ids': challenge_ids[:1],
                    'l1_refs': [f'{route_state_id}:l1-toolchain'],
                    'confidence': 0.78,
                }
            ],
            'fragility_factors': [],
            'missing_prerequisites': [],
        },
        'evidence_bundle': {
            'supporting_evidence_ids': support_ids,
            'challenging_evidence_ids': challenge_ids,
            'representative_move_ids': [f'{route_state_id}:m1', f'{route_state_id}:m2'],
            'representative_paper_ids': [f'{route_state_id}:paper-1', f'{route_state_id}:paper-2'],
            'l1_support_refs': [f'{route_state_id}:l1-benchmark'],
            'l1_constraint_refs': [f'{route_state_id}:l1-toolchain'],
        },
        'uncertainty_points': {
            'open_questions': [],
            'unresolved_conflicts': [],
            'weak_fields': ['community'],
            'low_confidence_clusters': [],
        },
        'compiler_metadata': {
            'compiler_version': 'route_state_synthesizer_v1',
            'packet_builder_version': 'packet_builder_v1',
            'l1_snapshot_version': f'{route_state_id}:snapshot',
            'trace_versions': {
                f'{route_state_id}:trace-1': 'v2',
                f'{route_state_id}:trace-2': 'v2',
            },
            'compile_mode': 'rule_only',
            'llm_usage_notes': None,
        },
        'quality': {
            'quality_tier': 'green',
            'ready_for_why_now': True,
            'ready_for_route_comparison': True,
            'ready_for_prior_selection': True,
            'quality_flags': [],
            'audit_status': 'reviewed',
        },
    }


def _route_state(
    *,
    route_state_id: str,
    method_label: str = 'deep convolutional network',
    support_ids: list[str],
    challenge_ids: list[str],
    method_score: float,
    measurement_score: float,
    data_resource_score: float,
    infrastructure_score: float,
    cost_cycle_score: float,
    overall_score: float,
    bottleneck_label: str = 'GPU training remains costly',
    bottleneck_type: str = 'compute',
    bottleneck_severity: str = 'high',
    positive_signal_confidence: float = 0.8,
) -> RouteState:
    return RouteState(
        **_route_state_payload(
            route_state_id=route_state_id,
            method_label=method_label,
            support_ids=support_ids,
            challenge_ids=challenge_ids,
            method_score=method_score,
            measurement_score=measurement_score,
            data_resource_score=data_resource_score,
            infrastructure_score=infrastructure_score,
            cost_cycle_score=cost_cycle_score,
            overall_score=overall_score,
            bottleneck_label=bottleneck_label,
            bottleneck_type=bottleneck_type,
            bottleneck_severity=bottleneck_severity,
            positive_signal_confidence=positive_signal_confidence,
        )
    )


def test_historical_replay_compiler_compiles_green_replay_bundle() -> None:
    traces = [_trace('paper-a', 2011, 'pa'), _trace('paper-b', 2011, 'pb')]
    support_route_states = [
        _route_state(
            route_state_id='support-1',
            support_ids=['s1-e1', 's1-e2'],
            challenge_ids=['s1-c1'],
            method_score=0.77,
            measurement_score=0.79,
            data_resource_score=0.88,
            infrastructure_score=0.6,
            cost_cycle_score=0.48,
            overall_score=0.69,
        ),
        _route_state(
            route_state_id='support-2',
            support_ids=['s2-e1', 's2-e2'],
            challenge_ids=['s2-c1'],
            method_score=0.75,
            measurement_score=0.78,
            data_resource_score=0.86,
            infrastructure_score=0.59,
            cost_cycle_score=0.47,
            overall_score=0.68,
        ),
    ]
    alternative_route_states = [
        _route_state(
            route_state_id='alt-1',
            method_label='feature-engineering pipeline',
            support_ids=['a1-e1', 'a1-e2'],
            challenge_ids=['a1-c1'],
            method_score=0.62,
            measurement_score=0.61,
            data_resource_score=0.45,
            infrastructure_score=0.84,
            cost_cycle_score=0.8,
            overall_score=0.62,
            bottleneck_label='manual tuning remains brittle',
            bottleneck_type='engineering',
            bottleneck_severity='low',
            positive_signal_confidence=0.25,
        )
    ]
    held_out_route_states = [
        _route_state(
            route_state_id='held-1',
            support_ids=['h1-e1', 'h1-e2'],
            challenge_ids=['h1-c1'],
            method_score=0.76,
            measurement_score=0.79,
            data_resource_score=0.87,
            infrastructure_score=0.61,
            cost_cycle_score=0.49,
            overall_score=0.69,
        )
    ]

    compilation = compile_historical_replay(
        _packet(traces, cutoff_year=2011),
        traces,
        support_route_states=support_route_states,
        alternative_route_states=alternative_route_states,
        held_out_route_states=held_out_route_states,
        reviewer_ids=['expert-1'],
        route_state_ref='route_state_store/primary.json',
        hindsight_outcome={
            'outcome_label': 'success',
            'later_evidence_refs': ['future-ref-1'],
            'retrospective_notes': 'Later history favored the route.',
            'input_visible': False,
        },
        built_at='2026-04-02T01:20:00Z',
    )

    assert compilation.primary_route_state.quality.quality_tier == 'green'
    assert compilation.why_now_case.quality.quality_tier == 'green'
    assert compilation.route_comparison_cases
    assert compilation.selected_comparison_case_id == compilation.route_comparison_cases[0].route_comparison_case_id
    assert compilation.decision_prior_card.quality.quality_tier == 'green'
    assert compilation.decision_episode.quality.quality_tier == 'green'
    assert compilation.decision_episode.decision_output.final_choice == 'primary'
    assert compilation.ready_for_pilot is True
    assert not compilation.quality_flags
    assert compilation.failure_records
    assert all(record.blocking is False for record in compilation.failure_records)
    assert {record.failure_code for record in compilation.failure_records} == {'l2_expected_slot_missing'}


def test_historical_replay_compiler_marks_underconstrained_bundle() -> None:
    thin_trace = _trace('paper-thin', 2011, 'pt', thin=True)

    compilation = HistoricalReplayCompiler().compile(
        _packet([thin_trace], cutoff_year=2011),
        [thin_trace],
        built_at='2026-04-02T01:30:00Z',
    )

    assert compilation.primary_route_state.quality.quality_tier == 'yellow'
    assert compilation.decision_prior_card.quality.quality_tier == 'yellow'
    assert compilation.decision_episode.quality.quality_tier == 'yellow'
    assert 'support_cluster_too_small' in compilation.quality_flags
    assert 'no_alternative_route_states' in compilation.quality_flags
    assert 'held_out_routes_missing' in compilation.quality_flags
    assert 'reviewer_missing' in compilation.quality_flags
    assert compilation.ready_for_pilot is False
    failure_codes = {record.failure_code for record in compilation.failure_records}
    assert 'l2_expected_slot_missing' in failure_codes
    assert 'l2_challenging_evidence_missing' in failure_codes
    assert 'prior_support_cluster_too_small' in failure_codes
    assert 'held_out_routes_missing' in failure_codes
    assert 'reviewer_missing' in failure_codes


def test_historical_replay_compiler_uses_l1_snapshot_to_fill_minimal_attack_path_resources() -> None:
    thin_trace = _trace('paper-thin', 2011, 'pt', thin=True)

    compilation = HistoricalReplayCompiler().compile(
        _packet([thin_trace], cutoff_year=2011),
        [thin_trace],
        l1_snapshot=_l1_snapshot(cutoff_year=2011),
        built_at='2026-04-02T09:30:00Z',
    )

    assert compilation.primary_route_state.compiler_metadata.l1_snapshot_version == 'imagenet_2011_snapshot'
    assert compilation.decision_episode.minimal_attack_path.required_resources == ['wn18rr', 'cuda stack']
    assert compilation.decision_episode.minimal_attack_path.required_measurements == ['top-1 accuracy']


def test_historical_replay_compiler_rejects_l1_snapshot_cutoff_mismatch() -> None:
    thin_trace = _trace('paper-thin', 2011, 'pt', thin=True)

    with pytest.raises(ValueError, match='l1_snapshot cutoff_year must match RoutePacket cutoff_year'):
        HistoricalReplayCompiler().compile(
            _packet([thin_trace], cutoff_year=2011),
            [thin_trace],
            l1_snapshot=_l1_snapshot(cutoff_year=2012),
        )


def test_historical_replay_compiler_rejects_cutoff_mismatch_in_support_cluster() -> None:
    traces = [_trace('paper-a', 2011, 'pa'), _trace('paper-b', 2011, 'pb')]
    mismatched_support = _route_state(
        route_state_id='support-2012',
        support_ids=['s1-e1', 's1-e2'],
        challenge_ids=['s1-c1'],
        method_score=0.77,
        measurement_score=0.79,
        data_resource_score=0.88,
        infrastructure_score=0.6,
        cost_cycle_score=0.48,
        overall_score=0.69,
    )
    mismatched_payload = mismatched_support.model_dump()
    mismatched_payload['cutoff_year'] = 2012
    mismatched_support = RouteState(**mismatched_payload)

    with pytest.raises(ValueError, match='support_route_states must share cutoff_year'):
        HistoricalReplayCompiler().compile(
            _packet(traces, cutoff_year=2011),
            traces,
            support_route_states=[mismatched_support],
        )


def test_historical_replay_compiler_accepts_injected_green_prior_registry() -> None:
    traces = [_trace('paper-a', 2011, 'pa'), _trace('paper-b', 2011, 'pb')]
    support_route_states = [
        _route_state(
            route_state_id='support-1',
            support_ids=['s1-e1', 's1-e2'],
            challenge_ids=['s1-c1'],
            method_score=0.77,
            measurement_score=0.79,
            data_resource_score=0.88,
            infrastructure_score=0.6,
            cost_cycle_score=0.48,
            overall_score=0.69,
        ),
        _route_state(
            route_state_id='support-2',
            support_ids=['s2-e1', 's2-e2'],
            challenge_ids=['s2-c1'],
            method_score=0.75,
            measurement_score=0.78,
            data_resource_score=0.86,
            infrastructure_score=0.59,
            cost_cycle_score=0.47,
            overall_score=0.68,
        ),
    ]
    route_main = _route_state(
        route_state_id='primary-route',
        support_ids=['m-e1', 'm-e2'],
        challenge_ids=['m-c1'],
        method_score=0.78,
        measurement_score=0.81,
        data_resource_score=0.9,
        infrastructure_score=0.61,
        cost_cycle_score=0.49,
        overall_score=0.7,
    )
    accepted_prior = build_decision_prior_card(
        [route_main, *support_route_states],
        comparison_cases=[
            build_route_comparison_case(
                route_main,
                _route_state(
                    route_state_id='route-alt',
                    method_label='feature-engineering pipeline',
                    support_ids=['alt-e1', 'alt-e2'],
                    challenge_ids=['alt-c1'],
                    method_score=0.62,
                    measurement_score=0.61,
                    data_resource_score=0.45,
                    infrastructure_score=0.84,
                    cost_cycle_score=0.8,
                    overall_score=0.62,
                    bottleneck_label='manual tuning remains brittle',
                    bottleneck_type='engineering',
                    bottleneck_severity='low',
                    positive_signal_confidence=0.25,
                ),
            )
        ],
        held_out_route_states=[
            _route_state(
                route_state_id='held-1',
                support_ids=['h1-e1', 'h1-e2'],
                challenge_ids=['h1-c1'],
                method_score=0.76,
                measurement_score=0.79,
                data_resource_score=0.87,
                infrastructure_score=0.61,
                cost_cycle_score=0.49,
                overall_score=0.69,
            )
        ],
        reviewer_ids=['expert-1'],
        built_at='2026-04-02T02:20:00Z',
        prior_id='prior:accepted-cluster',
    )

    compilation = HistoricalReplayCompiler().compile(
        _packet(traces, cutoff_year=2011),
        traces,
        support_route_states=support_route_states,
        alternative_route_states=[],
        held_out_route_states=[],
        reviewer_ids=None,
        prior_cards=[accepted_prior],
        route_state_id='primary-route',
        built_at='2026-04-02T02:30:00Z',
    )

    assert compilation.decision_prior_card.prior_id == 'prior:accepted-cluster'
    assert compilation.decision_episode.relevant_priors.selected_prior_ids == ['prior:accepted-cluster']
    assert 'weak_prior_support' not in compilation.decision_episode.quality.quality_flags


def test_historical_replay_compiler_keeps_weak_prior_support_for_non_green_registry() -> None:
    traces = [_trace('paper-a', 2011, 'pa'), _trace('paper-b', 2011, 'pb')]
    yellow_prior = build_decision_prior_card(
        [
            _route_state(
                route_state_id='primary-route',
                support_ids=['m-e1', 'm-e2'],
                challenge_ids=['m-c1'],
                method_score=0.78,
                measurement_score=0.81,
                data_resource_score=0.9,
                infrastructure_score=0.61,
                cost_cycle_score=0.49,
                overall_score=0.7,
            ),
            _route_state(
                route_state_id='support-1',
                support_ids=['s1-e1', 's1-e2'],
                challenge_ids=['s1-c1'],
                method_score=0.77,
                measurement_score=0.79,
                data_resource_score=0.88,
                infrastructure_score=0.6,
                cost_cycle_score=0.48,
                overall_score=0.69,
            ),
        ],
        built_at='2026-04-02T02:40:00Z',
        prior_id='prior:yellow-cluster',
    )

    compilation = HistoricalReplayCompiler().compile(
        _packet(traces, cutoff_year=2011),
        traces,
        prior_cards=[yellow_prior],
        route_state_id='primary-route',
        built_at='2026-04-02T02:50:00Z',
    )

    assert compilation.decision_prior_card.prior_id == 'prior:yellow-cluster'
    assert compilation.decision_episode.relevant_priors.selected_prior_ids == []
    assert 'weak_prior_support' in compilation.decision_episode.quality.quality_flags
