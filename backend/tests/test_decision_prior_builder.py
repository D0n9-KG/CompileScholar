from __future__ import annotations

import pytest

from app.research_logic.decision_prior_builder import DecisionPriorBuilder, build_decision_prior_card
from app.research_logic.prior_induction import build_prior_candidate_registry
from app.research_logic.models import RouteState
from app.research_logic.route_comparison_builder import build_route_comparison_case
from app.research_logic.why_now_builder import build_why_now_case


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
    scope: str = 'large-scale image recognition',
    cutoff_year: int = 2011,
    route_family_id: str | None = None,
) -> dict:
    return {
        'route_state_id': route_state_id,
        'route_family_id': route_family_id,
        'built_at': '2026-04-01T23:30:00Z',
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
            'resolution_rationale': 'Synthetic route state for decision prior builder tests.',
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
            'score_rationale': 'Synthetic payload for decision prior builder tests.',
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
    route_family_id: str | None = None,
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
            route_family_id=route_family_id,
        )
    )


def test_decision_prior_builder_emits_green_prior_for_reviewed_supported_cluster() -> None:
    support_cluster = [
        _route_state(
            route_state_id='route-s1',
            support_ids=['s1-e1', 's1-e2'],
            challenge_ids=['s1-c1'],
            method_score=0.76,
            measurement_score=0.79,
            data_resource_score=0.88,
            infrastructure_score=0.6,
            cost_cycle_score=0.48,
            overall_score=0.69,
        ),
        _route_state(
            route_state_id='route-s2',
            support_ids=['s2-e1', 's2-e2'],
            challenge_ids=['s2-c1'],
            method_score=0.78,
            measurement_score=0.8,
            data_resource_score=0.9,
            infrastructure_score=0.62,
            cost_cycle_score=0.5,
            overall_score=0.7,
        ),
        _route_state(
            route_state_id='route-s3',
            support_ids=['s3-e1', 's3-e2'],
            challenge_ids=['s3-c1'],
            method_score=0.74,
            measurement_score=0.77,
            data_resource_score=0.85,
            infrastructure_score=0.58,
            cost_cycle_score=0.46,
            overall_score=0.67,
        ),
    ]
    why_now_cases = [build_why_now_case(route_state) for route_state in support_cluster]
    alternative_route = _route_state(
        route_state_id='route-alt',
        method_label='feature-engineering pipeline',
        support_ids=['alt-e1', 'alt-e2'],
        challenge_ids=['alt-c1'],
        method_score=0.62,
        measurement_score=0.6,
        data_resource_score=0.45,
        infrastructure_score=0.84,
        cost_cycle_score=0.8,
        overall_score=0.62,
        bottleneck_label='manual tuning remains brittle',
        bottleneck_type='engineering',
        bottleneck_severity='low',
        positive_signal_confidence=0.25,
    )
    comparison_cases = [
        build_route_comparison_case(support_cluster[0], alternative_route),
        build_route_comparison_case(support_cluster[1], alternative_route),
    ]
    held_out_route_states = [
        _route_state(
            route_state_id='route-h1',
            support_ids=['h1-e1', 'h1-e2'],
            challenge_ids=['h1-c1'],
            method_score=0.75,
            measurement_score=0.78,
            data_resource_score=0.87,
            infrastructure_score=0.61,
            cost_cycle_score=0.49,
            overall_score=0.69,
        )
    ]

    prior_card = build_decision_prior_card(
        support_cluster,
        why_now_cases=why_now_cases,
        comparison_cases=comparison_cases,
        held_out_route_states=held_out_route_states,
        reviewer_ids=['expert-1'],
        built_at='2026-04-01T23:40:00Z',
    )

    assert prior_card.quality.quality_tier == 'green'
    assert prior_card.review.review_status == 'reviewed'
    assert prior_card.held_out_consistency.pass_rate == 1.0
    assert len(prior_card.supporting_route_state_ids) == 3
    assert any(action.action_type == 'compare_routes' for action in prior_card.recommended_actions)
    assert any(action.action_type == 'pursue_question' for action in prior_card.recommended_actions)
    assert 'missing_counterexample_search' not in prior_card.quality.quality_flags


def test_decision_prior_builder_marks_missing_counterexample_search_yellow() -> None:
    support_cluster = [
        _route_state(
            route_state_id='route-s1',
            support_ids=['s1-e1', 's1-e2'],
            challenge_ids=['s1-c1'],
            method_score=0.76,
            measurement_score=0.79,
            data_resource_score=0.88,
            infrastructure_score=0.6,
            cost_cycle_score=0.48,
            overall_score=0.69,
        ),
        _route_state(
            route_state_id='route-s2',
            support_ids=['s2-e1', 's2-e2'],
            challenge_ids=['s2-c1'],
            method_score=0.78,
            measurement_score=0.8,
            data_resource_score=0.9,
            infrastructure_score=0.62,
            cost_cycle_score=0.5,
            overall_score=0.7,
        ),
    ]

    prior_card = DecisionPriorBuilder().build(support_cluster, built_at='2026-04-01T23:50:00Z')

    assert prior_card.quality.quality_tier == 'yellow'
    assert 'weak_support_cluster' in prior_card.quality.quality_flags
    assert 'missing_counterexample_search' in prior_card.quality.quality_flags
    assert prior_card.review.review_status == 'candidate'


def test_decision_prior_builder_raises_on_empty_cluster() -> None:
    with pytest.raises(ValueError, match='at least one supporting RouteState'):
        DecisionPriorBuilder().build([])


def test_prior_candidate_registry_emits_multiple_clusters_and_anti_patterns() -> None:
    support_cluster_a = [
        _route_state(
            route_state_id='route-a1',
            route_family_id='route_family:image-recognition:2011:cluster-a',
            support_ids=['a1-e1', 'a1-e2'],
            challenge_ids=['a1-c1'],
            method_score=0.76,
            measurement_score=0.79,
            data_resource_score=0.88,
            infrastructure_score=0.6,
            cost_cycle_score=0.48,
            overall_score=0.69,
            bottleneck_label='GPU training remains costly',
            bottleneck_type='compute',
            bottleneck_severity='high',
        ),
        _route_state(
            route_state_id='route-a2',
            route_family_id='route_family:image-recognition:2011:cluster-a',
            support_ids=['a2-e1', 'a2-e2'],
            challenge_ids=['a2-c1'],
            method_score=0.77,
            measurement_score=0.8,
            data_resource_score=0.9,
            infrastructure_score=0.61,
            cost_cycle_score=0.49,
            overall_score=0.7,
            bottleneck_label='GPU training remains costly',
            bottleneck_type='compute',
            bottleneck_severity='high',
        ),
    ]
    support_cluster_b = [
        _route_state(
            route_state_id='route-b1',
            route_family_id='route_family:image-recognition:2011:cluster-b',
            method_label='probabilistic graphical model',
            support_ids=['b1-e1', 'b1-e2'],
            challenge_ids=['b1-c1'],
            method_score=0.63,
            measurement_score=0.74,
            data_resource_score=0.67,
            infrastructure_score=0.58,
            cost_cycle_score=0.56,
            overall_score=0.62,
            bottleneck_label='evaluation protocols stay brittle',
            bottleneck_type='evaluation',
            bottleneck_severity='medium',
        ),
        _route_state(
            route_state_id='route-b2',
            route_family_id='route_family:image-recognition:2011:cluster-b',
            method_label='probabilistic graphical model',
            support_ids=['b2-e1', 'b2-e2'],
            challenge_ids=['b2-c1'],
            method_score=0.64,
            measurement_score=0.73,
            data_resource_score=0.68,
            infrastructure_score=0.57,
            cost_cycle_score=0.57,
            overall_score=0.61,
            bottleneck_label='evaluation protocols stay brittle',
            bottleneck_type='evaluation',
            bottleneck_severity='medium',
        ),
    ]
    alternative_route = _route_state(
        route_state_id='route-alt',
        route_family_id='route_family:image-recognition:2011:feature-engineering',
        method_label='feature-engineering pipeline',
        support_ids=['alt-e1', 'alt-e2'],
        challenge_ids=['alt-c1'],
        method_score=0.62,
        measurement_score=0.6,
        data_resource_score=0.45,
        infrastructure_score=0.84,
        cost_cycle_score=0.8,
        overall_score=0.62,
        bottleneck_label='manual tuning remains brittle',
        bottleneck_type='engineering',
        bottleneck_severity='low',
        positive_signal_confidence=0.25,
    )

    registry = build_prior_candidate_registry(
        support_route_states=[*support_cluster_a, *support_cluster_b],
        alternative_route_states=[alternative_route],
        held_out_route_states=[],
        built_at='2026-04-02T02:10:00Z',
    )

    assert len(registry.prior_candidates) == 2
    assert all(len(candidate.supporting_route_state_ids) == 2 for candidate in registry.prior_candidates)
    assert registry.anti_pattern_candidates
    assert registry.anti_pattern_candidates[0].warning_signal_pattern.signals
    assert registry.anti_pattern_candidates[0].failure_examples.route_state_ids
    assert registry.anti_pattern_candidates[0].failure_examples.route_family_ids
