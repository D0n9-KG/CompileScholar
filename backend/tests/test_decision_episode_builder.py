from __future__ import annotations

import pytest

from app.research_logic.decision_episode_builder import DecisionEpisodeBuilder, build_decision_episode
from app.research_logic.decision_prior_builder import build_decision_prior_card
from app.research_logic.models import AntiPatternCard, RouteState
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
        'built_at': '2026-04-02T00:00:00Z',
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
            'resolution_rationale': 'Synthetic route state for decision episode builder tests.',
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
            'score_rationale': 'Synthetic payload for decision episode builder tests.',
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


def _route_packet(route_state: RouteState) -> dict:
    return {
        'packet_id': route_state.source_packet.packet_id,
        'built_at': '2026-04-02T00:05:00Z',
        'topic_scope_candidate': route_state.topic_scope,
        'cutoff_year': route_state.cutoff_year,
        'packet_status': 'frozen',
        'packet_composition': {
            'target_size': len(route_state.source_packet.included_trace_ids),
            'actual_size': len(route_state.source_packet.included_trace_ids),
            'role_counts': route_state.source_packet.packet_role_counts.model_dump(),
            'coverage_ok': True,
            'missing_roles': [],
        },
        'l1_snapshot_ref': {
            'snapshot_id': route_state.source_packet.l1_snapshot_ref,
            'resource_registry_ref': 'l1:resource-registry',
            'resource_timeline_ref': 'l1:resource-timeline',
            'benchmark_timeline_ref': 'l1:benchmark-timeline',
            'toolchain_timeline_ref': 'l1:toolchain-timeline',
            'protocol_registry_ref': 'l1:protocol-registry',
        },
        'inclusion_rules': {
            'scope_definition': 'Episode builder replay packet.',
            'scope_aliases': ['cnn image recognition'],
            'accepted_year_range': {'min_year': 2005, 'max_year': route_state.cutoff_year},
            'hard_exclusion_rules': ['exclude papers after cutoff'],
            'role_assignment_rules': ['one primary role per packet item'],
            'leakage_policy': 'After-cutoff papers are excluded from observation.',
        },
        'included_items': [
            {
                'paper_id': paper_id,
                'trace_id': trace_id,
                'paper_year': route_state.cutoff_year,
                'item_role': 'core_method',
                'inclusion_reason': 'Visible at cutoff.',
                'source_selector': 'manual',
                'evidence_for_inclusion': [paper_id],
            }
            for paper_id, trace_id in zip(route_state.source_packet.included_paper_ids, route_state.source_packet.included_trace_ids)
        ],
        'excluded_items': [
            {
                'paper_id': 'future-paper',
                'paper_year': route_state.cutoff_year + 1,
                'exclusion_reason': 'after_cutoff',
                'notes': 'Replay visibility boundary.',
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
            'preferred_scope_label': route_state.topic_scope,
            'preferred_method_labels': [route_state.route_landscape.dominant_methods[0].label],
            'preferred_benchmark_labels': [route_state.route_landscape.active_benchmarks[0].label],
            'preferred_bottleneck_labels': [route_state.route_landscape.known_bottlenecks[0].label],
            'expected_alternative_routes': [route_state.route_landscape.alternative_routes[0].label],
            'notes_for_route_state_compiler': 'Episode test packet.',
        },
    }


def test_decision_episode_builder_emits_green_historical_replay_sample() -> None:
    route_main = _route_state(
        route_state_id='route-main',
        support_ids=['m-e1', 'm-e2'],
        challenge_ids=['m-c1'],
        method_score=0.78,
        measurement_score=0.81,
        data_resource_score=0.9,
        infrastructure_score=0.61,
        cost_cycle_score=0.49,
        overall_score=0.7,
    )
    route_peer_1 = _route_state(
        route_state_id='route-peer-1',
        support_ids=['p1-e1', 'p1-e2'],
        challenge_ids=['p1-c1'],
        method_score=0.76,
        measurement_score=0.8,
        data_resource_score=0.87,
        infrastructure_score=0.6,
        cost_cycle_score=0.48,
        overall_score=0.69,
    )
    route_peer_2 = _route_state(
        route_state_id='route-peer-2',
        support_ids=['p2-e1', 'p2-e2'],
        challenge_ids=['p2-c1'],
        method_score=0.75,
        measurement_score=0.78,
        data_resource_score=0.86,
        infrastructure_score=0.59,
        cost_cycle_score=0.47,
        overall_score=0.68,
    )
    route_alt = _route_state(
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
    )

    why_now_case = build_why_now_case(route_main)
    comparison_case = build_route_comparison_case(route_main, route_alt)
    prior_card = build_decision_prior_card(
        [route_main, route_peer_1, route_peer_2],
        why_now_cases=[why_now_case, build_why_now_case(route_peer_1), build_why_now_case(route_peer_2)],
        comparison_cases=[comparison_case],
        held_out_route_states=[
            _route_state(
                route_state_id='route-held-out',
                support_ids=['h-e1', 'h-e2'],
                challenge_ids=['h-c1'],
                method_score=0.77,
                measurement_score=0.79,
                data_resource_score=0.88,
                infrastructure_score=0.61,
                cost_cycle_score=0.49,
                overall_score=0.69,
            )
        ],
        reviewer_ids=['expert-1'],
        built_at='2026-04-02T00:10:00Z',
    )

    episode = build_decision_episode(
        route_main,
        route_packet=_route_packet(route_main),
        why_now_case=why_now_case,
        comparison_case=comparison_case,
        prior_cards=[prior_card],
        route_state_ref='route_state_store/route-main.json',
        hindsight_outcome={
            'outcome_label': 'success',
            'later_evidence_refs': ['future-ref-1'],
            'retrospective_notes': 'Later history favored the route.',
            'input_visible': False,
        },
        built_at='2026-04-02T00:20:00Z',
    )

    assert episode.quality.quality_tier == 'green'
    assert episode.quality.ready_for_training is True
    assert episode.decision_output.final_choice == 'primary'
    assert episode.candidate_question.question_type == 'route_choice'
    assert episode.alternative_questions
    assert episode.not_now_cases
    assert episode.relevant_priors.selected_prior_ids == [prior_card.prior_id]
    assert episode.observation_evidence_pack.excluded_after_cutoff_ids == ['future-paper']
    assert episode.paper_logic_traces.trace_ids == route_main.source_packet.included_trace_ids


def test_decision_episode_builder_selects_antipatterns_by_route_family_id() -> None:
    route_main = _route_state(
        route_state_id='route-runtime-subset',
        route_family_id='route_family:image-recognition:2011:cnn',
        support_ids=['m-e1', 'm-e2'],
        challenge_ids=['m-c1'],
        method_score=0.78,
        measurement_score=0.81,
        data_resource_score=0.9,
        infrastructure_score=0.61,
        cost_cycle_score=0.49,
        overall_score=0.7,
    )
    matching_anti_pattern = AntiPatternCard(
        anti_pattern_id='anti:route-main:family-match',
        built_at='2026-04-02T00:25:00Z',
        anti_pattern_text='Repeated blocker pressure should trigger caution.',
        warning_signal_pattern={
            'signals': [
                {
                    'label': 'anisotropic packing',
                    'signal_type': 'bottleneck',
                    'severity': 'high',
                }
            ],
            'trigger_logic': 'all',
        },
        failure_examples={
            'route_state_ids': ['route-support-context'],
            'route_family_ids': ['route_family:image-recognition:2011:cnn'],
            'decision_episode_ids': [],
            'notes': 'Synthetic family-matching anti-pattern test.',
        },
        corrective_checklist=['Reduce blocker pressure before reuse.'],
        counterexamples={
            'route_state_ids': ['route-alt'],
            'route_family_ids': ['route_family:image-recognition:2011:feature-engineering'],
            'notes': 'Synthetic counterexample.',
        },
        review={
            'review_status': 'reviewed',
            'reviewer_notes': 'Reviewed for route family selection tests.',
            'reviewer_ids': ['reviewer-1'],
        },
        quality={
            'quality_tier': 'green',
            'quality_flags': [],
            'audit_status': 'reviewed',
        },
    )
    other_anti_pattern = AntiPatternCard(
        anti_pattern_id='anti:route-main:other-family',
        built_at='2026-04-02T00:26:00Z',
        anti_pattern_text='Other route family should not match.',
        warning_signal_pattern={
            'signals': [
                {
                    'label': 'other blocker',
                    'signal_type': 'bottleneck',
                    'severity': 'high',
                }
            ],
            'trigger_logic': 'all',
        },
        failure_examples={
            'route_state_ids': ['route-other'],
            'route_family_ids': ['route_family:image-recognition:2011:other'],
            'decision_episode_ids': [],
            'notes': 'Should not match.',
        },
        corrective_checklist=['Ignore for this route.'],
        counterexamples={'route_state_ids': [], 'route_family_ids': [], 'notes': None},
        review={
            'review_status': 'reviewed',
            'reviewer_notes': 'Reviewed for route family selection tests.',
            'reviewer_ids': ['reviewer-1'],
        },
        quality={
            'quality_tier': 'green',
            'quality_flags': [],
            'audit_status': 'reviewed',
        },
    )

    episode = build_decision_episode(
        route_main,
        route_packet=_route_packet(route_main),
        why_now_case=build_why_now_case(route_main),
        anti_pattern_cards=[matching_anti_pattern, other_anti_pattern],
        built_at='2026-04-02T00:30:00Z',
    )

    assert episode.route_state.route_family_id == 'route_family:image-recognition:2011:cnn'
    assert episode.relevant_priors.selected_antipattern_ids == ['anti:route-main:family-match']


def test_decision_episode_builder_marks_underconstrained_episode_yellow() -> None:
    payload = _route_state_payload(
        route_state_id='route-thin',
        method_label='deep convolutional network',
        support_ids=['t-e1', 't-e2'],
        challenge_ids=['t-c1'],
        method_score=0.72,
        measurement_score=0.76,
        data_resource_score=0.82,
        infrastructure_score=0.64,
        cost_cycle_score=0.58,
        overall_score=0.69,
        bottleneck_label='GPU training remains costly',
        bottleneck_type='compute',
        bottleneck_severity='high',
        positive_signal_confidence=0.8,
    )
    payload['route_landscape']['active_benchmarks'] = []
    payload['route_landscape']['measurement_protocols'] = []
    payload['route_landscape']['toolchains_and_infrastructure'] = []
    payload['not_now_features']['blocking_factors'] = []
    payload['why_now_features']['unlocking_factors'] = []
    payload['why_now_features']['acceleration_factors'] = []
    route_state = RouteState(**payload)

    episode = DecisionEpisodeBuilder().build(route_state, built_at='2026-04-02T00:30:00Z')

    assert episode.quality.quality_tier == 'yellow'
    assert 'weak_prior_support' in episode.quality.quality_flags
    assert 'missing_not_now_case' in episode.quality.quality_flags
    assert 'weak_alternative_set' in episode.quality.quality_flags
    assert 'candidate_question_too_open_ended' in episode.quality.quality_flags
    assert 'minimal_attack_path_missing' in episode.quality.quality_flags
    assert episode.quality.ready_for_training is False


def test_decision_episode_builder_rejects_visible_hindsight() -> None:
    route_main = _route_state(
        route_state_id='route-main',
        support_ids=['m-e1', 'm-e2'],
        challenge_ids=['m-c1'],
        method_score=0.78,
        measurement_score=0.81,
        data_resource_score=0.9,
        infrastructure_score=0.61,
        cost_cycle_score=0.49,
        overall_score=0.7,
    )

    with pytest.raises(ValueError, match='input_visible=true'):
        DecisionEpisodeBuilder().build(
            route_main,
            hindsight_outcome={
                'outcome_label': 'success',
                'later_evidence_refs': ['future-ref-1'],
                'retrospective_notes': 'Leak',
                'input_visible': True,
            },
        )


def test_decision_episode_builder_distinguishes_green_and_non_green_prior_support() -> None:
    route_main = _route_state(
        route_state_id='route-main',
        support_ids=['m-e1', 'm-e2'],
        challenge_ids=['m-c1'],
        method_score=0.78,
        measurement_score=0.81,
        data_resource_score=0.9,
        infrastructure_score=0.61,
        cost_cycle_score=0.49,
        overall_score=0.7,
    )
    route_peer_1 = _route_state(
        route_state_id='route-peer-1',
        support_ids=['p1-e1', 'p1-e2'],
        challenge_ids=['p1-c1'],
        method_score=0.76,
        measurement_score=0.8,
        data_resource_score=0.87,
        infrastructure_score=0.6,
        cost_cycle_score=0.48,
        overall_score=0.69,
    )
    route_peer_2 = _route_state(
        route_state_id='route-peer-2',
        support_ids=['p2-e1', 'p2-e2'],
        challenge_ids=['p2-c1'],
        method_score=0.75,
        measurement_score=0.78,
        data_resource_score=0.86,
        infrastructure_score=0.59,
        cost_cycle_score=0.47,
        overall_score=0.68,
    )

    green_prior = build_decision_prior_card(
        [route_main, route_peer_1, route_peer_2],
        why_now_cases=[build_why_now_case(route_main), build_why_now_case(route_peer_1), build_why_now_case(route_peer_2)],
        held_out_route_states=[
            _route_state(
                route_state_id='route-held-out',
                support_ids=['h-e1', 'h-e2'],
                challenge_ids=['h-c1'],
                method_score=0.77,
                measurement_score=0.79,
                data_resource_score=0.88,
                infrastructure_score=0.61,
                cost_cycle_score=0.49,
                overall_score=0.69,
            )
        ],
        reviewer_ids=['expert-1'],
        built_at='2026-04-02T03:00:00Z',
        prior_id='prior:green-support',
    )
    yellow_prior = build_decision_prior_card(
        [route_main, route_peer_1],
        built_at='2026-04-02T03:05:00Z',
        prior_id='prior:yellow-support',
    )

    green_episode = build_decision_episode(
        route_main,
        route_packet=_route_packet(route_main),
        why_now_case=build_why_now_case(route_main),
        prior_cards=[green_prior],
        built_at='2026-04-02T03:10:00Z',
    )
    yellow_episode = build_decision_episode(
        route_main,
        route_packet=_route_packet(route_main),
        why_now_case=build_why_now_case(route_main),
        prior_cards=[yellow_prior],
        built_at='2026-04-02T03:15:00Z',
    )

    assert green_episode.relevant_priors.selected_prior_ids == ['prior:green-support']
    assert 'weak_prior_support' not in green_episode.quality.quality_flags
    assert yellow_episode.relevant_priors.selected_prior_ids == ['prior:yellow-support']
    assert 'weak_prior_support' in yellow_episode.quality.quality_flags


def test_decision_episode_builder_caps_confidence_without_grounded_route_or_prior_support() -> None:
    route_main = _route_state(
        route_state_id='route-main-thin-support',
        support_ids=['m-e1', 'm-e2'],
        challenge_ids=['m-c1'],
        method_score=0.7,
        measurement_score=0.71,
        data_resource_score=0.72,
        infrastructure_score=0.69,
        cost_cycle_score=0.65,
        overall_score=0.8,
    )
    route_alt = _route_state(
        route_state_id='route-alt-thin-support',
        method_label='feature-engineering pipeline',
        support_ids=['a-e1', 'a-e2'],
        challenge_ids=['a-c1'],
        method_score=0.67,
        measurement_score=0.68,
        data_resource_score=0.69,
        infrastructure_score=0.67,
        cost_cycle_score=0.67,
        overall_score=0.74,
        bottleneck_label='manual tuning remains brittle',
        bottleneck_type='engineering',
        bottleneck_severity='high',
        positive_signal_confidence=0.7,
    )
    comparison_case = build_route_comparison_case(route_main, route_alt)

    episode = build_decision_episode(
        route_main,
        route_packet=_route_packet(route_main),
        why_now_case=build_why_now_case(route_main),
        comparison_case=comparison_case,
        built_at='2026-04-02T03:20:00Z',
    )

    assert comparison_case.preference_label == 'prefer_a'
    assert comparison_case.recommended_route_state_id is None
    assert comparison_case.route_advantage_summary is None
    assert episode.decision_output.final_choice == 'primary'
    assert episode.decision_output.confidence == 0.85
