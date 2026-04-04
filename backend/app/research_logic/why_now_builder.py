from __future__ import annotations

from datetime import datetime, timezone

from .models import (
    RouteFeature,
    RouteState,
    WhyNowCase,
    WhyNowEvidenceChain,
    WhyNowFactor,
    WhyNowUncertaintyPoints,
    TrainingQuality,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _map_factor_type(feature: RouteFeature) -> str:
    mapping = {
        'method_maturity': 'method_maturity',
        'benchmark_availability': 'benchmark_availability',
        'new_resource': 'data_resource',
        'infrastructure': 'infrastructure',
        'protocol': 'protocol',
        'comparative_gain': 'comparative_advantage',
        'blocker': 'bottleneck',
        'fragility': 'fragility',
        'missing_prerequisite': 'resource_gap',
    }
    return mapping.get(feature.feature_type, 'unknown')


def _map_strength(confidence: float | None) -> str:
    if confidence is None:
        return 'unknown'
    if confidence >= 0.8:
        return 'high'
    if confidence >= 0.5:
        return 'medium'
    return 'low'


def _render_factor_labels(labels: list[str]) -> str:
    cleaned = [str(label or '').strip() for label in labels if str(label or '').strip()]
    if not cleaned:
        return 'the currently visible evidence'
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f'{cleaned[0]} and {cleaned[1]}'
    return f'{", ".join(cleaned[:-1])}, and {cleaned[-1]}'


def _because_now(route_state: RouteState, unlocking_factors: list[WhyNowFactor], *, label: str) -> str | None:
    if label not in {'now', 'almost_now'}:
        return None
    unlocking_labels = _render_factor_labels([factor.label for factor in unlocking_factors[:3]])
    return (
        f'Now is plausible because {unlocking_labels} are visible at the cutoff and the route now looks actionable '
        f'for {route_state.topic_scope}.'
    )


def _why_not_before(route_state: RouteState, blocking_factors: list[WhyNowFactor], *, label: str) -> str | None:
    if label not in {'now', 'almost_now'}:
        return None
    blocking_labels = _render_factor_labels([factor.label for factor in blocking_factors[:2]])
    return (
        f'Not before now because {blocking_labels} still constrained the route, so the earlier evidence was not '
        f'strong enough to justify committing to {route_state.topic_scope}.'
    )


def _to_why_now_factor(feature: RouteFeature, *, source_field: str) -> WhyNowFactor:
    return WhyNowFactor(
        label=feature.label,
        factor_type=_map_factor_type(feature),
        strength=_map_strength(feature.confidence),
        source_route_fields=[source_field],
        evidence_ids=list(feature.evidence_ids),
        l1_refs=list(feature.l1_refs),
    )


def _derive_label(route_state: RouteState) -> str:
    unlock_count = len(route_state.why_now_features.unlocking_factors) + len(route_state.why_now_features.acceleration_factors)
    blocking_count = len(route_state.not_now_features.blocking_factors) + len(route_state.not_now_features.missing_prerequisites)
    overall = route_state.readiness_scores.overall or 0.0
    if unlock_count == 0 and blocking_count == 0:
        return 'unclear'
    if unlock_count >= 2 and blocking_count == 0 and overall >= 0.7:
        return 'now'
    if unlock_count >= 1 and overall >= 0.55:
        return 'almost_now'
    if blocking_count > unlock_count:
        return 'not_now'
    return 'unclear'


class WhyNowCaseBuilder:
    def __init__(self, *, builder_version: str = 'why_now_case_builder_v1') -> None:
        self.builder_version = builder_version

    def build(
        self,
        route_state: RouteState,
        *,
        built_at: str | None = None,
        why_now_case_id: str | None = None,
    ) -> WhyNowCase:
        unlocking_factors = [
            _to_why_now_factor(feature, source_field='why_now_features.unlocking_factors')
            for feature in route_state.why_now_features.unlocking_factors
        ]
        unlocking_factors.extend(
            _to_why_now_factor(feature, source_field='why_now_features.acceleration_factors')
            for feature in route_state.why_now_features.acceleration_factors
        )
        blocking_factors = [
            _to_why_now_factor(feature, source_field='not_now_features.blocking_factors')
            for feature in route_state.not_now_features.blocking_factors
        ]
        blocking_factors.extend(
            _to_why_now_factor(feature, source_field='not_now_features.missing_prerequisites')
            for feature in route_state.not_now_features.missing_prerequisites
        )
        label = _derive_label(route_state)
        quality_flags: list[str] = []
        if label in {'now', 'almost_now'} and not unlocking_factors:
            quality_flags.append('weak_unlocking_factors')
        if not blocking_factors:
            quality_flags.append('blocking_factors_missing')
        if not route_state.evidence_bundle.supporting_evidence_ids or not route_state.evidence_bundle.challenging_evidence_ids:
            quality_flags.append('evidence_chain_thin')
        representative_route_fields = []
        if route_state.why_now_features.unlocking_factors:
            representative_route_fields.append('why_now_features.unlocking_factors')
        if route_state.not_now_features.blocking_factors:
            representative_route_fields.append('not_now_features.blocking_factors')
        if route_state.readiness_scores.overall is not None:
            representative_route_fields.append('readiness_scores')
        if label in {'now', 'almost_now'} and not representative_route_fields:
            quality_flags.append('label_not_grounded')
        because_now = _because_now(route_state, unlocking_factors, label=label)
        why_not_before = _why_not_before(route_state, blocking_factors, label=label)

        quality_tier = 'green' if route_state.quality.quality_tier == 'green' and not quality_flags else 'yellow' if route_state.quality.quality_tier != 'red' else 'red'

        return WhyNowCase(
            why_now_case_id=why_now_case_id or f'{route_state.route_state_id}:why_now',
            built_at=built_at or _utc_now_iso(),
            route_state_id=route_state.route_state_id,
            why_now_label=label,
            because_now=because_now,
            why_not_before=why_not_before,
            unlocking_factors=unlocking_factors,
            blocking_factors=blocking_factors,
            evidence_chain=WhyNowEvidenceChain(
                supporting_evidence_ids=list(route_state.evidence_bundle.supporting_evidence_ids),
                challenging_evidence_ids=list(route_state.evidence_bundle.challenging_evidence_ids),
                representative_route_fields=representative_route_fields,
            ),
            uncertainty_points=WhyNowUncertaintyPoints(
                unresolved_conflicts=list(route_state.uncertainty_points.unresolved_conflicts),
                weak_signals=list(route_state.uncertainty_points.weak_fields),
                ambiguous_enablers=[],
            ),
            quality=TrainingQuality(
                quality_tier=quality_tier,
                ready_for_training=quality_tier == 'green',
                quality_flags=quality_flags,
            ),
        )


def build_why_now_case(
    route_state: RouteState,
    *,
    built_at: str | None = None,
    why_now_case_id: str | None = None,
    builder_version: str = 'why_now_case_builder_v1',
) -> WhyNowCase:
    builder = WhyNowCaseBuilder(builder_version=builder_version)
    return builder.build(
        route_state,
        built_at=built_at,
        why_now_case_id=why_now_case_id,
    )


__all__ = ['WhyNowCaseBuilder', 'build_why_now_case']
