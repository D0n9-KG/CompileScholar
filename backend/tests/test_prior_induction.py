from __future__ import annotations

from app.research_logic.prior_induction import build_prior_candidate_registry

from test_historical_replay_compiler import _route_state


def _singleton_support_route_states():
    return [
        _route_state(
            route_state_id='support-fixed-point',
            method_label='fixed point iteration',
            support_ids=['fp-e1', 'fp-e2'],
            challenge_ids=['fp-c1'],
            method_score=0.71,
            measurement_score=0.78,
            data_resource_score=0.87,
            infrastructure_score=0.61,
            cost_cycle_score=0.49,
            overall_score=0.7,
            bottleneck_label='GPU training remains costly',
            bottleneck_type='compute',
            bottleneck_severity='high',
        ),
        _route_state(
            route_state_id='support-polycrystal',
            method_label='implicit time integration',
            support_ids=['pc-e1', 'pc-e2'],
            challenge_ids=['pc-c1'],
            method_score=0.72,
            measurement_score=0.77,
            data_resource_score=0.86,
            infrastructure_score=0.6,
            cost_cycle_score=0.48,
            overall_score=0.69,
            bottleneck_label='Experimental throughput remains limited',
            bottleneck_type='measurement',
            bottleneck_severity='medium',
        ),
        _route_state(
            route_state_id='support-clustering',
            method_label='k means algorithm',
            support_ids=['km-e1', 'km-e2'],
            challenge_ids=['km-c1'],
            method_score=0.74,
            measurement_score=0.79,
            data_resource_score=0.88,
            infrastructure_score=0.62,
            cost_cycle_score=0.5,
            overall_score=0.71,
            bottleneck_label='Scope transfer remains uncertain',
            bottleneck_type='evaluation',
            bottleneck_severity='medium',
        ),
    ]


def test_prior_induction_preserves_singleton_support_clusters_by_default() -> None:
    registry = build_prior_candidate_registry(
        support_route_states=_singleton_support_route_states(),
        reviewer_ids=['reviewer-1'],
        built_at='2026-04-04T04:50:00Z',
    )

    assert registry.cluster_strategy == 'default'
    assert registry.fallback_reason is None
    assert len(registry.clusters) == 3
    assert all(cluster.support_count == 1 for cluster in registry.clusters)
    assert len(registry.prior_candidates) == 0


def test_prior_induction_can_merge_same_scope_singletons_into_fallback_cluster() -> None:
    registry = build_prior_candidate_registry(
        support_route_states=_singleton_support_route_states(),
        alternative_route_states=[
            _route_state(
                route_state_id='alt-feature-engineering',
                method_label='feature engineering pipeline',
                support_ids=['alt-e1', 'alt-e2'],
                challenge_ids=['alt-c1'],
                method_score=0.62,
                measurement_score=0.61,
                data_resource_score=0.45,
                infrastructure_score=0.84,
                cost_cycle_score=0.8,
                overall_score=0.62,
                bottleneck_label='Manual tuning remains brittle',
                bottleneck_type='engineering',
                bottleneck_severity='low',
                positive_signal_confidence=0.25,
            )
        ],
        held_out_route_states=[
            _route_state(
                route_state_id='held-out-route',
                support_ids=['held-e1', 'held-e2'],
                challenge_ids=['held-c1'],
                method_score=0.76,
                measurement_score=0.79,
                data_resource_score=0.87,
                infrastructure_score=0.61,
                cost_cycle_score=0.49,
                overall_score=0.69,
            )
        ],
        reviewer_ids=['reviewer-1'],
        allow_scope_fallback_merge=True,
        built_at='2026-04-04T04:55:00Z',
    )

    assert registry.cluster_strategy == 'fallback_scope_merge'
    assert registry.fallback_reason == 'singleton_support_clusters'
    assert len(registry.clusters) == 1
    assert registry.clusters[0].cluster_id.endswith('scope_fallback_cluster')
    assert registry.clusters[0].support_count == 3
    assert len(registry.prior_candidates) >= 1
