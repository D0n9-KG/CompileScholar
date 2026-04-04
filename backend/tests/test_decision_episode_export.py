from __future__ import annotations

from app.research_logic import build_decision_episode
from app.research_logic.decision_episode_export import build_decision_episode_audit_export
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
            'resolution_rationale': 'Synthetic route state for export tests.',
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
            'score_rationale': 'Synthetic payload for export tests.',
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
            'scope_definition': 'Episode export replay packet.',
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
            'notes_for_route_state_compiler': 'Episode export test packet.',
        },
    }


def _green_prior(route_main: RouteState, route_peer_1: RouteState, route_peer_2: RouteState) -> object:
    return build_decision_prior_card(
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


def _anti_pattern_card(
    *,
    anti_pattern_id: str,
    route_state_ids: list[str],
    route_family_ids: list[str] | None = None,
    counterexample_route_family_ids: list[str] | None = None,
) -> AntiPatternCard:
    return AntiPatternCard(
        anti_pattern_id=anti_pattern_id,
        built_at='2026-04-02T03:20:00Z',
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
            'route_state_ids': route_state_ids,
            'route_family_ids': route_family_ids or [],
            'decision_episode_ids': [],
            'notes': 'Synthetic export anti-pattern test.',
        },
        corrective_checklist=['Reduce blocker pressure before reuse.'],
        counterexamples={
            'route_state_ids': ['route-alt'],
            'route_family_ids': counterexample_route_family_ids or [],
            'notes': 'Synthetic counterexample.',
        },
        review={
            'review_status': 'reviewed',
            'reviewer_notes': 'Reviewed for export tests.',
            'reviewer_ids': ['reviewer-1'],
        },
        quality={
            'quality_tier': 'green',
            'quality_flags': [],
            'audit_status': 'reviewed',
        },
    )


def _source_refs(bundle_name: str) -> dict[str, object]:
    return {
        'bundle_ref': f'tmp/{bundle_name}',
        'manifest_ref': f'tmp/{bundle_name}/bundle_manifest.json',
        'files': {
            'decision_episode': f'tmp/{bundle_name}/outputs/decision_episode.json',
            'summary': f'tmp/{bundle_name}/summary.json',
        },
    }


def test_build_decision_episode_audit_export_preserves_empty_accepted_prior_truth() -> None:
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
    prior_card = _green_prior(route_main, route_peer_1, route_peer_2)

    live_episode = build_decision_episode(
        route_main,
        route_packet=_route_packet(route_main),
        why_now_case=why_now_case,
        comparison_case=comparison_case,
        prior_cards=[prior_card],
        built_at='2026-04-02T03:10:00Z',
    )
    export = build_decision_episode_audit_export(
        route_packet=_route_packet(route_main),
        route_state=route_main,
        why_now_case=why_now_case,
        comparison_case=comparison_case,
        prior_cards=[prior_card],
        anti_pattern_cards=[],
        accepted_prior_ids=[],
        accepted_anti_pattern_ids=[],
        route_state_ref='route_state_store/route-main.json',
        source_replay_bundle_refs=_source_refs('replay_bundle'),
        source_review_bundle_refs=_source_refs('review_bundle'),
        built_at='2026-04-02T03:15:00Z',
    )

    assert live_episode.relevant_priors.selected_prior_ids == ['prior:green-support']
    assert export.decision_episode.relevant_priors.selected_prior_ids == []
    assert export.prior_selection_note == (
        'Review bundle accepted no prior ids, so the audited export preserved empty selected_prior_ids.'
    )


def test_build_decision_episode_audit_export_records_accepted_but_unselected_prior_exclusions() -> None:
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
    route_peer_3 = _route_state(
        route_state_id='route-peer-3',
        support_ids=['p3-e1', 'p3-e2'],
        challenge_ids=['p3-c1'],
        method_score=0.74,
        measurement_score=0.77,
        data_resource_score=0.85,
        infrastructure_score=0.58,
        cost_cycle_score=0.46,
        overall_score=0.67,
    )
    prior_card = _green_prior(route_peer_1, route_peer_2, route_peer_3)

    export = build_decision_episode_audit_export(
        route_packet=_route_packet(route_main),
        route_state=route_main,
        why_now_case=build_why_now_case(route_main),
        prior_cards=[prior_card],
        anti_pattern_cards=[],
        accepted_prior_ids=[prior_card.prior_id],
        accepted_anti_pattern_ids=[],
        source_replay_bundle_refs=_source_refs('replay_bundle'),
        source_review_bundle_refs=_source_refs('review_bundle'),
        built_at='2026-04-02T03:18:00Z',
    )

    assert export.decision_episode.relevant_priors.selected_prior_ids == []
    assert len(export.accepted_but_unselected_priors) == 1
    assert export.accepted_but_unselected_priors[0].prior_id == prior_card.prior_id
    assert export.accepted_but_unselected_priors[0].exclusion_reason_code == 'route_state_not_supported'
    assert export.accepted_but_unselected_priors[0].supporting_route_state_ids == [
        'route-peer-1',
        'route-peer-2',
        'route-peer-3',
    ]
    assert export.accepted_but_unselected_priors[0].evidence_refs[0] == f'accepted_prior:{prior_card.prior_id}'


def test_build_decision_episode_audit_export_carries_route_family_matching_accepted_antipatterns() -> None:
    route_main = _route_state(
        route_state_id='route-main-runtime-subset',
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
    matching_anti_pattern = _anti_pattern_card(
        anti_pattern_id='anti:route-main:matching',
        route_state_ids=['route-support-context', 'route-peer-1'],
        route_family_ids=['route_family:image-recognition:2011:cnn'],
        counterexample_route_family_ids=['route_family:image-recognition:2011:feature-engineering'],
    )
    other_route_anti_pattern = _anti_pattern_card(
        anti_pattern_id='anti:route-main:other-route',
        route_state_ids=['route-other'],
        route_family_ids=['route_family:image-recognition:2011:other'],
    )
    unaccepted_matching = _anti_pattern_card(
        anti_pattern_id='anti:route-main:unaccepted',
        route_state_ids=['route-unaccepted-support'],
        route_family_ids=['route_family:image-recognition:2011:cnn'],
    )

    export = build_decision_episode_audit_export(
        route_packet=_route_packet(route_main),
        route_state=route_main,
        why_now_case=build_why_now_case(route_main),
        prior_cards=[],
        anti_pattern_cards=[matching_anti_pattern, other_route_anti_pattern, unaccepted_matching],
        accepted_prior_ids=[],
        accepted_anti_pattern_ids=['anti:route-main:matching', 'anti:route-main:other-route'],
        source_replay_bundle_refs=_source_refs('replay_bundle'),
        source_review_bundle_refs=_source_refs('review_bundle'),
        built_at='2026-04-02T03:25:00Z',
    )

    assert export.decision_episode.route_state.route_family_id == 'route_family:image-recognition:2011:cnn'
    assert export.decision_episode.relevant_priors.selected_antipattern_ids == ['anti:route-main:matching']
    assert export.anti_pattern_selection_note == (
        'Carried 1 reviewed accepted anti-pattern id(s) because their failure examples match route_state route-main-runtime-subset or route_family_id route_family:image-recognition:2011:cnn.'
    )


def test_build_decision_episode_audit_export_keeps_hindsight_label_only_and_out_of_visible_inputs() -> None:
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

    export = build_decision_episode_audit_export(
        route_packet=_route_packet(route_main),
        route_state=route_main,
        why_now_case=build_why_now_case(route_main),
        hindsight_outcome={
            'outcome_label': 'success',
            'later_evidence_refs': ['future-ref-1'],
            'retrospective_notes': 'Later history favored the route.',
            'input_visible': False,
        },
        source_replay_bundle_refs=_source_refs('replay_bundle'),
        source_review_bundle_refs=_source_refs('review_bundle'),
        built_at='2026-04-02T03:30:00Z',
    )

    assert export.decision_episode.hindsight_outcome.input_visible is False
    assert export.label_eval_only_refs == ['hindsight_evidence:future-ref-1']
    assert 'hindsight_evidence:future-ref-1' not in export.visible_input_refs
    assert 'after_cutoff_paper:future-paper' in export.audit_only_refs
    assert 'replay_bundle:bundle:tmp/replay_bundle' in export.audit_only_refs


def test_build_decision_episode_audit_export_includes_machine_readable_review_metadata() -> None:
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

    export = build_decision_episode_audit_export(
        route_packet=_route_packet(route_main),
        route_state=route_main,
        why_now_case=build_why_now_case(route_main),
        source_replay_bundle_refs=_source_refs('replay_bundle'),
        source_review_bundle_refs=_source_refs('review_bundle'),
        review_status='reviewed',
        training_acceptance_verdict='needs_revision',
        reviewer_ids=['reviewer-1', 'reviewer-2'],
        reviewed_at='2026-04-02T03:35:00Z',
        rationale='The audited export is review-backed, but the training-facing artifact still needs stronger prior grounding.',
        residual_defects=['weak_prior_support'],
        section_reviews={
            'evidence_pack': {
                'review_status': 'reviewed',
                'training_acceptance_verdict': 'accepted',
                'reviewer_ids': ['reviewer-1'],
                'reviewed_at': '2026-04-02T03:35:00Z',
                'rationale': 'Evidence pack is self-contained and auditable.',
                'residual_defects': [],
            },
            'review_labels': {
                'review_status': 'reviewed',
                'training_acceptance_verdict': 'needs_revision',
                'reviewer_ids': ['reviewer-1', 'reviewer-2'],
                'reviewed_at': '2026-04-02T03:35:00Z',
                'rationale': 'Final acceptance labels are present but still too cautious for release.',
                'residual_defects': ['weak_prior_support'],
            },
        },
        built_at='2026-04-02T03:35:00Z',
    )

    assert export.review_status == 'reviewed'
    assert export.training_acceptance_verdict == 'needs_revision'
    assert export.reviewer_ids == ['reviewer-1', 'reviewer-2']
    assert export.reviewed_at == '2026-04-02T03:35:00Z'
    assert export.residual_defects == ['weak_prior_support']
    assert export.route_state_snapshot.route_state_id == 'route-main'
    assert export.why_now_case is not None
    assert export.section_reviews['evidence_pack'].training_acceptance_verdict == 'accepted'
    assert export.section_reviews['review_labels'].residual_defects == ['weak_prior_support']
    assert export.section_reviews['route_comparison'].review_status == 'not_started'
    assert set(export.section_reviews) == {
        'evidence_pack',
        'route_synthesis',
        'why_now',
        'route_comparison',
        'priors_antipatterns',
        'minimal_attack_path',
        'final_decision',
        'review_labels',
    }
