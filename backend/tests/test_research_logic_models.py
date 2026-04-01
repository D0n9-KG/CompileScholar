from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.research_logic.models import (
    AntiPatternCard,
    DecisionEpisode,
    DecisionPriorCard,
    RouteComparisonCase,
    RoutePacket,
    RouteState,
    WhyNowCase,
)


def _packet_payload() -> dict:
    return {
        'packet_id': 'image_recognition_2011_packet_01',
        'built_at': '2026-04-01T20:00:00Z',
        'topic_scope_candidate': 'large-scale image recognition with deep neural networks',
        'cutoff_year': 2011,
        'packet_status': 'frozen',
        'packet_composition': {
            'target_size': 24,
            'actual_size': 10,
            'role_counts': {
                'core_method': 6,
                'resource_or_benchmark': 2,
                'limitation_or_critique': 2,
                'survey_or_review': 1,
                'alternative_route': 2,
            },
            'coverage_ok': True,
            'missing_roles': [],
        },
        'l1_snapshot_ref': {
            'snapshot_id': 'imagenet_2011_snapshot',
            'resource_registry_ref': 'l1:resource-registry',
            'benchmark_timeline_ref': 'l1:benchmark-timeline',
        },
        'inclusion_rules': {
            'scope_definition': 'Focus on historically bounded image recognition routes.',
            'scope_aliases': ['cnn image recognition'],
            'accepted_year_range': {'min_year': 2005, 'max_year': 2011},
            'hard_exclusion_rules': ['exclude after-cutoff papers'],
            'role_assignment_rules': ['benchmark papers count as resource_or_benchmark'],
            'leakage_policy': 'Papers after cutoff are excluded and hindsight labels are never visible.',
        },
        'included_items': [
            {
                'paper_id': 'p1',
                'trace_id': 'p1:trace',
                'paper_year': 2011,
                'item_role': 'core_method',
                'inclusion_reason': 'Defines the main route.',
                'source_selector': 'rule',
                'evidence_for_inclusion': ['e1'],
                'title': 'CNN route paper',
            },
            {
                'paper_id': 'p2',
                'trace_id': 'p2:trace',
                'paper_year': 2010,
                'item_role': 'resource_or_benchmark',
                'inclusion_reason': 'Introduces the benchmark regime.',
                'source_selector': 'manual',
                'evidence_for_inclusion': ['e2'],
                'title': 'ImageNet benchmark paper',
            },
        ],
        'excluded_items': [
            {
                'paper_id': 'p99',
                'paper_year': 2012,
                'exclusion_reason': 'after_cutoff',
                'notes': 'Visible only for audit.',
            }
        ],
        'packet_quality': {
            'quality_tier': 'green',
            'ready_for_route_state': True,
            'quality_flags': [],
            'topic_boundary_confidence': 0.86,
            'leakage_risk': 'low',
            'manual_review_status': 'completed',
        },
        'compiler_hints': {
            'preferred_scope_label': 'large-scale image recognition with deep neural networks',
            'preferred_method_labels': ['cnn'],
            'preferred_benchmark_labels': ['imagenet'],
            'expected_alternative_routes': ['feature engineering'],
        },
    }


def _route_state_payload() -> dict:
    return {
        'route_state_id': 'route_state:image_recognition:2011',
        'built_at': '2026-04-01T20:05:00Z',
        'topic_scope': 'large-scale image recognition with deep neural networks',
        'cutoff_year': 2011,
        'source_packet': {
            'packet_id': 'image_recognition_2011_packet_01',
            'included_trace_ids': ['p1:trace', 'p2:trace'],
            'included_paper_ids': ['p1', 'p2'],
            'packet_role_counts': {
                'core_method': 6,
                'resource_or_benchmark': 2,
                'limitation_or_critique': 2,
                'survey_or_review': 1,
                'alternative_route': 2,
            },
            'l1_snapshot_ref': 'imagenet_2011_snapshot',
        },
        'scope_resolution': {
            'topic_scope_candidates': ['cnn image recognition', 'large-scale image recognition'],
            'accepted_scope_label': 'large-scale image recognition with deep neural networks',
            'rejected_scope_labels': ['general computer vision'],
            'resolution_rationale': 'Packet evidence converges on a single route family.',
            'resolution_evidence_ids': ['e1', 'e2'],
        },
        'route_landscape': {
            'dominant_methods': [
                {
                    'label': 'convolutional neural network',
                    'family': 'deep neural network',
                    'maturity_score': 0.62,
                    'adoption_level': 'workable',
                    'source_move_ids': ['m1'],
                    'source_paper_ids': ['p1'],
                    'evidence_ids': ['e1'],
                }
            ],
            'known_bottlenecks': [
                {
                    'label': 'compute throughput remains costly',
                    'bottleneck_type': 'compute',
                    'severity': 'high',
                    'blocking_scope': 'route_level',
                    'source_paper_ids': ['p1'],
                    'evidence_ids': ['e3'],
                    'counterevidence_ids': [],
                }
            ],
            'enabling_conditions': [
                {
                    'label': 'benchmark-scale labeled data is available',
                    'condition_type': 'data',
                    'status': 'met',
                    'source_paper_ids': ['p2'],
                    'evidence_ids': ['e2'],
                }
            ],
            'alternative_routes': [
                {
                    'label': 'feature-engineering pipeline',
                    'route_family': 'classical vision',
                    'relation_to_main_route': 'competing',
                    'distinguishing_features': ['hand-crafted descriptors'],
                    'source_paper_ids': ['p3'],
                    'evidence_ids': ['e4'],
                }
            ],
        },
        'readiness_scores': {
            'theory': 0.58,
            'method': 0.62,
            'measurement': 0.85,
            'data_resource': 0.9,
            'infrastructure': 0.66,
            'community': 0.55,
            'cost_cycle': 0.48,
            'overall': 0.66,
            'score_rationale': 'Data is strong, method maturity is moderate, and compute is still limiting.',
        },
        'why_now_features': {
            'unlocking_factors': [
                {
                    'label': 'ImageNet-scale supervision',
                    'feature_type': 'benchmark_availability',
                    'direction': 'unlock',
                    'source_paper_ids': ['p2'],
                    'evidence_ids': ['e2'],
                    'l1_refs': ['l1:benchmark-timeline'],
                    'confidence': 0.9,
                }
            ]
        },
        'not_now_features': {
            'blocking_factors': [
                {
                    'label': 'GPU throughput remains constrained',
                    'feature_type': 'blocker',
                    'direction': 'block',
                    'source_paper_ids': ['p1'],
                    'evidence_ids': ['e3'],
                    'l1_refs': ['l1:toolchain-timeline'],
                    'confidence': 0.72,
                }
            ]
        },
        'evidence_bundle': {
            'supporting_evidence_ids': ['e1', 'e2'],
            'challenging_evidence_ids': ['e3'],
            'representative_move_ids': ['m1'],
            'representative_paper_ids': ['p1', 'p2'],
            'l1_support_refs': ['l1:benchmark-timeline'],
            'l1_constraint_refs': ['l1:toolchain-timeline'],
        },
        'uncertainty_points': {
            'open_questions': ['Will gains persist at larger scale?'],
            'unresolved_conflicts': [],
            'weak_fields': ['community'],
            'low_confidence_clusters': [],
        },
        'compiler_metadata': {
            'compiler_version': 'route_state_synthesizer_v1',
            'packet_builder_version': 'packet_builder_v1',
            'l1_snapshot_version': 'imagenet_snapshot_v1',
            'trace_versions': {'p1:trace': 'v2', 'p2:trace': 'v2'},
            'compile_mode': 'rule_plus_llm',
            'llm_usage_notes': 'LLM only phrases rationales.',
        },
        'quality': {
            'quality_tier': 'green',
            'ready_for_why_now': True,
            'ready_for_route_comparison': True,
            'ready_for_prior_selection': True,
            'quality_flags': [],
            'audit_status': 'eligible',
        },
    }


def _why_now_payload() -> dict:
    return {
        'why_now_case_id': 'why-now:route-a',
        'built_at': '2026-04-01T20:10:00Z',
        'route_state_id': 'route_state:image_recognition:2011',
        'why_now_label': 'almost_now',
        'unlocking_factors': [
            {
                'label': 'Benchmark regime now exists',
                'factor_type': 'benchmark_availability',
                'strength': 'high',
                'source_route_fields': ['why_now_features.unlocking_factors'],
                'evidence_ids': ['e2'],
                'l1_refs': ['l1:benchmark-timeline'],
            }
        ],
        'blocking_factors': [
            {
                'label': 'Training costs remain high',
                'factor_type': 'cost_cycle',
                'strength': 'medium',
                'source_route_fields': ['not_now_features.blocking_factors'],
                'evidence_ids': ['e3'],
                'l1_refs': ['l1:toolchain-timeline'],
            }
        ],
        'evidence_chain': {
            'supporting_evidence_ids': ['e1', 'e2'],
            'challenging_evidence_ids': ['e3'],
            'representative_route_fields': ['readiness_scores', 'why_now_features', 'not_now_features'],
        },
        'uncertainty_points': {
            'unresolved_conflicts': [],
            'weak_signals': ['community readiness'],
            'ambiguous_enablers': [],
        },
        'quality': {
            'quality_tier': 'green',
            'ready_for_training': True,
            'quality_flags': [],
        },
    }


def _route_comparison_payload() -> dict:
    return {
        'route_comparison_case_id': 'cmp:route-a-vs-route-b',
        'built_at': '2026-04-01T20:15:00Z',
        'cutoff_year': 2011,
        'route_a_state_id': 'route_state:image_recognition:2011',
        'route_b_state_id': 'route_state:classical_vision:2011',
        'comparison_dimension_scores': [
            {
                'dimension': 'data_resource',
                'route_a_score': 0.9,
                'route_b_score': 0.55,
                'preferred_route': 'a',
                'rationale': 'Route A benefits from the benchmark shift.',
                'evidence_ids': ['e2'],
            }
        ],
        'preference_label': 'prefer_a',
        'why_a_not_b': {
            'summary': 'Route A scales better under the new benchmark regime.',
            'decisive_dimensions': ['data_resource', 'strategic_value'],
            'decisive_evidence_ids': ['e2'],
        },
        'why_b_not_a': {
            'summary': 'Route B is cheaper today but less strategically expandable.',
            'decisive_dimensions': ['infrastructure'],
            'decisive_evidence_ids': ['e3'],
        },
        'evidence_chain': {
            'route_a_support_ids': ['e1', 'e2'],
            'route_b_support_ids': ['b1'],
            'cross_route_comparison_ids': ['e3'],
        },
        'uncertainty_points': {
            'incomparable_dimensions': [],
            'weak_dimensions': ['novelty'],
            'unresolved_conflicts': [],
        },
        'quality': {
            'quality_tier': 'green',
            'ready_for_training': True,
            'quality_flags': [],
        },
    }


def _decision_prior_payload() -> dict:
    return {
        'prior_id': 'prior:benchmark-plus-moderate-method-readiness',
        'built_at': '2026-04-01T20:20:00Z',
        'prior_text': 'When benchmark availability and method readiness co-occur, pursue a narrowly scoped route test.',
        'applies_when': {
            'readiness_pattern': [
                {
                    'label': 'method readiness is moderate',
                    'condition_type': 'readiness',
                    'polarity': 'high',
                    'required': True,
                }
            ],
            'bottleneck_pattern': [
                {
                    'label': 'compute bottleneck is present but not blocking',
                    'condition_type': 'bottleneck',
                    'polarity': 'present',
                    'required': True,
                }
            ],
        },
        'does_not_apply_when': {
            'blocker_pattern': [
                {
                    'label': 'benchmark support is absent',
                    'condition_type': 'environment',
                    'polarity': 'absent',
                    'required': True,
                }
            ]
        },
        'recommended_actions': [
            {
                'action_type': 'pursue_question',
                'action_text': 'Run a bounded route comparison under historically available resources.',
                'target_route_feature': 'benchmark availability',
                'confidence': 0.74,
            }
        ],
        'expected_failure_modes': [
            {
                'label': 'Compute cost dominates training loop',
                'linked_bottleneck_types': ['compute'],
                'warning_signals': ['unstable scaling'],
                'mitigation_hint': 'Narrow model size and protocol.',
            }
        ],
        'supporting_route_state_ids': ['route_state:image_recognition:2011', 'route_state:speech:2012'],
        'counterexample_ids': [],
        'held_out_consistency': {
            'held_out_route_state_ids': ['route_state:vision_transfer:2013'],
            'pass_rate': 0.67,
            'failure_notes': None,
        },
        'review': {
            'review_status': 'reviewed',
            'reviewer_notes': 'Grounded in replayable historical states.',
            'reviewer_ids': ['reviewer-1'],
        },
        'quality': {
            'quality_tier': 'green',
            'quality_flags': [],
            'audit_status': 'reviewed',
        },
    }


def _anti_pattern_payload() -> dict:
    return {
        'anti_pattern_id': 'anti:premature-scale-up',
        'built_at': '2026-04-01T20:25:00Z',
        'anti_pattern_text': 'Do not scale a route before its measurement protocol stabilizes.',
        'warning_signal_pattern': {
            'signals': [
                {
                    'label': 'measurement remains weak',
                    'signal_type': 'weak_measurement',
                    'severity': 'blocking',
                    'source_field': 'readiness_scores.measurement',
                }
            ],
            'trigger_logic': 'all',
        },
        'failure_examples': {
            'route_state_ids': ['route_state:premature-scale:2010'],
            'decision_episode_ids': ['episode:premature-scale'],
            'notes': 'Historical replay shows unstable evaluation.',
        },
        'corrective_checklist': ['stabilize protocol', 'compare against stronger baselines'],
        'counterexamples': {
            'route_state_ids': ['route_state:robust-measurement:2012'],
            'notes': 'Pattern can be survivable if evaluation matures quickly.',
        },
        'review': {
            'review_status': 'reviewed',
            'reviewer_notes': None,
            'reviewer_ids': ['reviewer-1'],
        },
        'quality': {
            'quality_tier': 'yellow',
            'quality_flags': ['counterexample_gap'],
            'audit_status': 'eligible',
        },
    }


def _decision_episode_payload() -> dict:
    return {
        'episode_id': 'episode:image-recognition:2011',
        'built_at': '2026-04-01T20:30:00Z',
        'historical_cutoff_time': '2011-12-31T23:59:59Z',
        'observation_evidence_pack': {
            'route_packet_id': 'image_recognition_2011_packet_01',
            'l1_snapshot_ref': 'imagenet_2011_snapshot',
            'visible_paper_ids': ['p1', 'p2'],
            'visible_trace_ids': ['p1:trace', 'p2:trace'],
            'excluded_after_cutoff_ids': ['p99'],
            'evidence_refs': ['e1', 'e2', 'e3'],
        },
        'paper_logic_traces': {
            'trace_ids': ['p1:trace', 'p2:trace'],
            'representative_trace_ids': ['p1:trace'],
        },
        'route_state': {
            'route_state_id': 'route_state:image_recognition:2011',
            'route_state_ref': 'route_state:image_recognition:2011',
        },
        'relevant_priors': {
            'selected_prior_ids': ['prior:benchmark-plus-moderate-method-readiness'],
            'selected_antipattern_ids': ['anti:premature-scale-up'],
            'prior_selection_rationale': 'Positive prior applies while anti-pattern constrains protocol rigor.',
        },
        'candidate_question': {
            'question_text': 'Can a bounded CNN route outperform classical baselines under 2011-scale resources?',
            'question_type': 'route_choice',
            'target_route_feature': 'benchmark availability',
            'justification_evidence_ids': ['e1', 'e2'],
        },
        'alternative_questions': [
            {
                'question_text': 'Should we defer until compute becomes cheaper?',
                'question_type': 'resource_gap',
                'target_route_feature': 'compute throughput',
                'justification_evidence_ids': ['e3'],
            }
        ],
        'why_this_not_that': {
            'primary_reasoning': 'Benchmark availability now outweighs the remaining compute bottleneck for a narrow test.',
            'comparison_dimensions': [
                {
                    'dimension': 'data_resource',
                    'preferred_candidate': 'primary',
                    'rationale': 'The new benchmark makes the route evaluable.',
                }
            ],
            'evidence_chain': ['e1', 'e2', 'e3'],
        },
        'not_now_cases': [
            {
                'rejected_question_text': 'Scale to very large training runs immediately.',
                'blocker_summary': 'Compute and protocol maturity are still insufficient.',
                'evidence_ids': ['e3'],
            }
        ],
        'minimal_attack_path': {
            'prerequisite_steps': ['verify data pipeline', 'stabilize training protocol'],
            'required_resources': ['ImageNet subset', 'GPU throughput'],
            'required_measurements': ['top-1 accuracy'],
            'expected_checkpoints': ['baseline parity'],
        },
        'decision_output': {
            'final_choice': 'primary',
            'final_decision_text': 'Pursue the narrow CNN route test under historical resource constraints.',
            'confidence': 0.76,
        },
        'hindsight_outcome': {
            'outcome_label': 'success',
            'later_evidence_refs': ['future_ref_1'],
            'retrospective_notes': 'Later work validated the route, but that was not visible at cutoff.',
            'input_visible': False,
        },
        'compiler_metadata': {
            'episode_builder_version': 'decision_episode_builder_v1',
            'route_state_version': 'route_state_synthesizer_v1',
            'prior_layer_version': 'decision_prior_builder_v1',
            'compile_mode': 'rule_plus_llm',
            'notes': 'LLM only phrases rationale text.',
        },
        'quality': {
            'quality_tier': 'green',
            'ready_for_training': True,
            'ready_for_eval': True,
            'quality_flags': [],
            'audit_status': 'eligible',
        },
    }


def test_valid_research_logic_contracts_instantiate() -> None:
    packet = RoutePacket(**_packet_payload())
    route_state = RouteState(**_route_state_payload())
    why_now = WhyNowCase(**_why_now_payload())
    comparison = RouteComparisonCase(**_route_comparison_payload())
    prior = DecisionPriorCard(**_decision_prior_payload())
    anti_pattern = AntiPatternCard(**_anti_pattern_payload())
    episode = DecisionEpisode(**_decision_episode_payload())

    assert packet.packet_quality.ready_for_route_state is True
    assert route_state.quality.ready_for_prior_selection is True
    assert why_now.why_now_label == 'almost_now'
    assert comparison.preference_label == 'prefer_a'
    assert prior.review.review_status == 'reviewed'
    assert anti_pattern.warning_signal_pattern.trigger_logic == 'all'
    assert episode.decision_output.final_choice == 'primary'


def test_green_route_packet_requires_trace_coverage() -> None:
    payload = _packet_payload()
    payload['included_items'][0]['trace_id'] = None

    with pytest.raises(ValidationError, match='trace coverage'):
        RoutePacket(**payload)


def test_green_route_state_requires_supporting_and_challenging_evidence() -> None:
    payload = _route_state_payload()
    payload['evidence_bundle']['challenging_evidence_ids'] = []

    with pytest.raises(ValidationError, match='challenging evidence'):
        RouteState(**payload)


def test_positive_why_now_case_requires_unlocking_factors() -> None:
    payload = _why_now_payload()
    payload['unlocking_factors'] = []

    with pytest.raises(ValidationError, match='unlocking factors'):
        WhyNowCase(**payload)


def test_route_comparison_case_requires_distinct_routes() -> None:
    payload = _route_comparison_payload()
    payload['route_b_state_id'] = payload['route_a_state_id']

    with pytest.raises(ValidationError, match='distinct route states'):
        RouteComparisonCase(**payload)


def test_green_decision_prior_requires_held_out_results() -> None:
    payload = _decision_prior_payload()
    payload['held_out_consistency'] = {
        'held_out_route_state_ids': [],
        'pass_rate': None,
        'failure_notes': None,
    }

    with pytest.raises(ValidationError, match='held-out results'):
        DecisionPriorCard(**payload)


def test_green_decision_prior_requires_completed_review_metadata() -> None:
    payload = _decision_prior_payload()
    payload['review'] = {
        'review_status': 'candidate',
        'reviewer_notes': None,
        'reviewer_ids': [],
    }

    with pytest.raises(ValidationError, match='completed review metadata'):
        DecisionPriorCard(**payload)


def test_decision_episode_rejects_visible_hindsight() -> None:
    payload = _decision_episode_payload()
    payload['hindsight_outcome']['input_visible'] = True

    with pytest.raises(ValidationError, match='must not be visible'):
        DecisionEpisode(**payload)
