from __future__ import annotations

from app.research_logic.models import RouteState
from app.research_logic.why_now_builder import WhyNowCaseBuilder, build_why_now_case


def _route_state_payload(*, blockers: bool = True) -> dict:
    payload = {
        'route_state_id': 'route_state:image_recognition:2011',
        'built_at': '2026-04-01T22:00:00Z',
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
                    'maturity_score': 0.7,
                    'adoption_level': 'established',
                    'source_move_ids': ['m1', 'm2'],
                    'source_paper_ids': ['p1', 'p2'],
                    'evidence_ids': ['e1', 'e2'],
                }
            ],
            'active_benchmarks': [
                {
                    'label': 'imagenet',
                    'benchmark_type': 'benchmark',
                    'adoption_level': 'active',
                    'source_paper_ids': ['p2'],
                    'evidence_ids': ['e2'],
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
            'method': 0.72,
            'measurement': 0.85,
            'data_resource': 0.9,
            'infrastructure': 0.66,
            'community': 0.55,
            'cost_cycle': 0.48,
            'overall': 0.68,
            'score_rationale': 'Data is strong, method readiness is moderate, and compute is still limiting.',
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
            ],
            'acceleration_factors': [
                {
                    'label': 'CNN maturity is improving',
                    'feature_type': 'method_maturity',
                    'direction': 'accelerate',
                    'source_paper_ids': ['p1', 'p2'],
                    'evidence_ids': ['e1'],
                    'l1_refs': [],
                    'confidence': 0.75,
                }
            ],
            'positive_comparison_signals': [],
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
            ],
            'fragility_factors': [],
            'missing_prerequisites': [],
        },
        'evidence_bundle': {
            'supporting_evidence_ids': ['e1', 'e2'],
            'challenging_evidence_ids': ['e3'],
            'representative_move_ids': ['m1', 'm2'],
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
            'audit_status': 'reviewed',
        },
    }
    if not blockers:
        payload['not_now_features']['blocking_factors'] = []
        payload['route_landscape']['known_bottlenecks'] = []
        payload['evidence_bundle']['challenging_evidence_ids'] = ['e3']
        payload['readiness_scores']['overall'] = 0.79
    return payload


def test_why_now_builder_emits_almost_now_for_green_route_state() -> None:
    route_state = RouteState(**_route_state_payload())

    why_now_case = build_why_now_case(route_state, built_at='2026-04-01T22:10:00Z')

    assert why_now_case.why_now_label == 'almost_now'
    assert why_now_case.quality.quality_tier == 'green'
    assert why_now_case.quality.ready_for_training is True
    assert why_now_case.because_now is not None
    assert why_now_case.why_not_before is not None
    assert why_now_case.unlocking_factors
    assert why_now_case.blocking_factors
    assert why_now_case.evidence_chain.supporting_evidence_ids == ['e1', 'e2']


def test_why_now_builder_marks_missing_blockers_as_yellow() -> None:
    route_state = RouteState(**_route_state_payload(blockers=False))

    why_now_case = WhyNowCaseBuilder().build(route_state, built_at='2026-04-01T22:20:00Z')

    assert why_now_case.why_now_label == 'now'
    assert why_now_case.quality.quality_tier == 'yellow'
    assert why_now_case.quality.ready_for_training is False
    assert why_now_case.because_now is not None
    assert why_now_case.why_not_before is not None
    assert 'blocking_factors_missing' in why_now_case.quality.quality_flags
