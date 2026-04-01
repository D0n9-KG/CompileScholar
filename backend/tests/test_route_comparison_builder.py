from __future__ import annotations

import pytest

from app.research_logic.models import RouteState
from app.research_logic.route_comparison_builder import RouteComparisonBuilder, build_route_comparison_case


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
    bottleneck_severity: str,
    alternative_route_label: str,
    positive_signal_confidence: float | None,
    scope: str = 'large-scale image recognition',
    cutoff_year: int = 2011,
) -> dict:
    return {
        'route_state_id': route_state_id,
        'built_at': '2026-04-01T23:00:00Z',
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
            'resolution_rationale': 'Comparison payload for route-level evaluation.',
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
                    'bottleneck_type': 'compute',
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
                    'label': alternative_route_label,
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
            'score_rationale': 'Synthetic payload for route comparison builder tests.',
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
                },
                {
                    'label': f'{route_state_id}:data-pipeline',
                    'feature_type': 'new_resource',
                    'direction': 'unlock',
                    'source_paper_ids': [f'{route_state_id}:paper-2'],
                    'evidence_ids': support_ids[1:2] or support_ids[:1],
                    'l1_refs': [f'{route_state_id}:l1-resource'],
                    'confidence': 0.72,
                },
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
            'positive_comparison_signals': (
                [
                    {
                        'label': f'{route_state_id}:comparative-upside',
                        'feature_type': 'comparative_gain',
                        'direction': 'accelerate',
                        'source_paper_ids': [f'{route_state_id}:paper-3'],
                        'evidence_ids': [f'{route_state_id}:alt-evidence'],
                        'l1_refs': [],
                        'confidence': positive_signal_confidence,
                    }
                ]
                if positive_signal_confidence is not None
                else []
            ),
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
            'weak_fields': [],
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


def test_route_comparison_builder_prefers_route_a_with_grounded_dimensions() -> None:
    route_a = RouteState(
        **_route_state_payload(
            route_state_id='route-a',
            method_label='deep convolutional network',
            support_ids=['a-e1', 'a-e2'],
            challenge_ids=['a-c1'],
            method_score=0.79,
            measurement_score=0.84,
            data_resource_score=0.93,
            infrastructure_score=0.62,
            cost_cycle_score=0.5,
            overall_score=0.71,
            bottleneck_label='GPU training remains costly',
            bottleneck_severity='high',
            alternative_route_label='feature-engineering pipeline',
            positive_signal_confidence=0.82,
        )
    )
    route_b = RouteState(
        **_route_state_payload(
            route_state_id='route-b',
            method_label='feature-engineering pipeline',
            support_ids=['b-e1', 'b-e2'],
            challenge_ids=['b-c1'],
            method_score=0.62,
            measurement_score=0.61,
            data_resource_score=0.45,
            infrastructure_score=0.84,
            cost_cycle_score=0.81,
            overall_score=0.62,
            bottleneck_label='manual tuning remains brittle',
            bottleneck_severity='low',
            alternative_route_label='deep convolutional network',
            positive_signal_confidence=0.25,
        )
    )

    comparison_case = build_route_comparison_case(route_a, route_b, built_at='2026-04-01T23:10:00Z')

    assert comparison_case.preference_label == 'prefer_a'
    assert comparison_case.quality.quality_tier == 'green'
    assert comparison_case.quality.ready_for_training is True
    assert 'data_resource' in comparison_case.why_a_not_b.decisive_dimensions
    assert 'strategic_value' in comparison_case.why_a_not_b.decisive_dimensions
    assert 'bottleneck' in comparison_case.why_b_not_a.decisive_dimensions
    assert 'infrastructure' in comparison_case.why_b_not_a.decisive_dimensions
    assert comparison_case.evidence_chain.cross_route_comparison_ids
    assert len(comparison_case.comparison_dimension_scores) == 9


def test_route_comparison_builder_raises_for_cutoff_mismatch() -> None:
    route_a = RouteState(
        **_route_state_payload(
            route_state_id='route-a',
            method_label='deep convolutional network',
            support_ids=['a-e1', 'a-e2'],
            challenge_ids=['a-c1'],
            method_score=0.79,
            measurement_score=0.84,
            data_resource_score=0.93,
            infrastructure_score=0.62,
            cost_cycle_score=0.5,
            overall_score=0.71,
            bottleneck_label='GPU training remains costly',
            bottleneck_severity='high',
            alternative_route_label='feature-engineering pipeline',
            positive_signal_confidence=0.82,
            cutoff_year=2011,
        )
    )
    route_b = RouteState(
        **_route_state_payload(
            route_state_id='route-b',
            method_label='feature-engineering pipeline',
            support_ids=['b-e1', 'b-e2'],
            challenge_ids=['b-c1'],
            method_score=0.62,
            measurement_score=0.61,
            data_resource_score=0.45,
            infrastructure_score=0.84,
            cost_cycle_score=0.81,
            overall_score=0.62,
            bottleneck_label='manual tuning remains brittle',
            bottleneck_severity='low',
            alternative_route_label='deep convolutional network',
            positive_signal_confidence=0.25,
            cutoff_year=2012,
        )
    )

    with pytest.raises(ValueError, match='same cutoff_year'):
        RouteComparisonBuilder().build(route_a, route_b)


def test_route_comparison_builder_flags_same_route_disguised_as_two() -> None:
    route_a = RouteState(
        **_route_state_payload(
            route_state_id='route-a',
            method_label='deep convolutional network',
            support_ids=['shared-e1', 'shared-e2'],
            challenge_ids=['shared-c1'],
            method_score=0.78,
            measurement_score=0.81,
            data_resource_score=0.88,
            infrastructure_score=0.63,
            cost_cycle_score=0.51,
            overall_score=0.7,
            bottleneck_label='GPU training remains costly',
            bottleneck_severity='high',
            alternative_route_label='feature-engineering pipeline',
            positive_signal_confidence=0.8,
            scope='large-scale image recognition with deep models',
        )
    )
    route_b = RouteState(
        **_route_state_payload(
            route_state_id='route-b',
            method_label='deep convolutional network',
            support_ids=['shared-e1', 'shared-e2'],
            challenge_ids=['shared-c1'],
            method_score=0.77,
            measurement_score=0.8,
            data_resource_score=0.87,
            infrastructure_score=0.64,
            cost_cycle_score=0.5,
            overall_score=0.7,
            bottleneck_label='GPU training remains costly',
            bottleneck_severity='high',
            alternative_route_label='feature-engineering pipeline',
            positive_signal_confidence=0.8,
            scope='large-scale image recognition with deep models',
        )
    )

    comparison_case = build_route_comparison_case(route_a, route_b, built_at='2026-04-01T23:20:00Z')

    assert comparison_case.preference_label == 'tie'
    assert comparison_case.quality.quality_tier == 'red'
    assert 'same_route_disguised_as_two' in comparison_case.quality.quality_flags
    assert 'alternative_route_not_real' in comparison_case.quality.quality_flags
    assert comparison_case.quality.ready_for_training is False
