from __future__ import annotations

from datetime import datetime, timezone

from .models import (
    ComparisonDimensionScore,
    RouteComparisonCase,
    RouteComparisonEvidenceChain,
    RouteComparisonUncertaintyPoints,
    RoutePreferenceExplanation,
    RouteState,
    TrainingQuality,
)


_DIMENSION_WEIGHTS = {
    'method_maturity': 1.0,
    'measurement': 1.0,
    'data_resource': 1.15,
    'infrastructure': 0.95,
    'bottleneck': 1.1,
    'feasibility': 1.2,
    'strategic_value': 1.4,
    'novelty': 0.7,
    'cost_cycle': 0.9,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value or '').strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _mean(values: list[float | None]) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return round(sum(numeric_values) / len(numeric_values), 2)


def _normalize(value: str) -> str:
    return str(value or '').strip().lower()


def _severity_burden(severity: str) -> float:
    mapping = {
        'low': 0.2,
        'medium': 0.45,
        'high': 0.75,
        'blocking': 0.95,
        'unknown': 0.5,
    }
    return mapping.get(str(severity or '').strip().lower(), 0.5)


def _feature_density(count: int, *, max_count: int = 3) -> float | None:
    if count <= 0:
        return None
    return round(min(float(count) / float(max_count), 1.0), 2)


def _feature_confidence_mean(features: list) -> float | None:
    return _mean([feature.confidence for feature in features])


def _bottleneck_relief_score(route_state: RouteState) -> float | None:
    bottlenecks = route_state.route_landscape.known_bottlenecks
    if bottlenecks:
        burden = _mean([_severity_burden(bottleneck.severity) for bottleneck in bottlenecks])
        if burden is not None:
            return round(max(0.0, 1.0 - burden), 2)
    return route_state.readiness_scores.cost_cycle


def _feasibility_score(route_state: RouteState) -> float | None:
    return _mean(
        [
            route_state.readiness_scores.method,
            route_state.readiness_scores.measurement,
            route_state.readiness_scores.infrastructure,
            route_state.readiness_scores.cost_cycle,
        ]
    )


def _strategic_value_score(route_state: RouteState) -> float | None:
    positive_signals = list(route_state.why_now_features.positive_comparison_signals)
    alternative_density = _feature_density(len(route_state.route_landscape.alternative_routes))
    positive_density = _feature_confidence_mean(positive_signals)
    if positive_density is None:
        positive_density = _feature_density(len(positive_signals))
    return _mean(
        [
            route_state.readiness_scores.data_resource,
            _feature_density(
                len(route_state.why_now_features.unlocking_factors) + len(route_state.why_now_features.acceleration_factors)
            ),
            positive_density or alternative_density,
        ]
    )


def _novelty_score(route_state: RouteState) -> float | None:
    positive_signals = list(route_state.why_now_features.positive_comparison_signals)
    if not positive_signals:
        return None
    return _feature_confidence_mean(positive_signals) or _feature_density(len(positive_signals))


def _dimension_value(route_state: RouteState, dimension: str) -> float | None:
    if dimension == 'method_maturity':
        return route_state.readiness_scores.method
    if dimension == 'measurement':
        return route_state.readiness_scores.measurement
    if dimension == 'data_resource':
        return route_state.readiness_scores.data_resource
    if dimension == 'infrastructure':
        return route_state.readiness_scores.infrastructure
    if dimension == 'bottleneck':
        return _bottleneck_relief_score(route_state)
    if dimension == 'feasibility':
        return _feasibility_score(route_state)
    if dimension == 'strategic_value':
        return _strategic_value_score(route_state)
    if dimension == 'novelty':
        return _novelty_score(route_state)
    if dimension == 'cost_cycle':
        return route_state.readiness_scores.cost_cycle
    return None


def _dimension_evidence_ids(route_state: RouteState, dimension: str) -> list[str]:
    if dimension == 'method_maturity':
        return _unique(
            [
                evidence_id
                for method in route_state.route_landscape.dominant_methods
                for evidence_id in method.evidence_ids
            ]
            + [
                evidence_id
                for feature in route_state.why_now_features.acceleration_factors
                for evidence_id in feature.evidence_ids
            ]
        )
    if dimension == 'measurement':
        return _unique(
            [
                evidence_id
                for protocol in route_state.route_landscape.measurement_protocols
                for evidence_id in protocol.evidence_ids
            ]
            + [
                evidence_id
                for capability in route_state.route_landscape.known_capabilities
                for evidence_id in capability.evidence_ids
            ]
        )
    if dimension == 'data_resource':
        return _unique(
            [
                evidence_id
                for benchmark in route_state.route_landscape.active_benchmarks
                for evidence_id in benchmark.evidence_ids
            ]
            + [
                evidence_id
                for condition in route_state.route_landscape.enabling_conditions
                for evidence_id in condition.evidence_ids
            ]
            + [
                evidence_id
                for feature in route_state.why_now_features.unlocking_factors
                if feature.feature_type in {'benchmark_availability', 'new_resource'}
                for evidence_id in feature.evidence_ids
            ]
        )
    if dimension == 'infrastructure':
        return _unique(
            [
                evidence_id
                for infra in route_state.route_landscape.toolchains_and_infrastructure
                for evidence_id in infra.evidence_ids
            ]
            + [
                evidence_id
                for feature in route_state.why_now_features.acceleration_factors
                if feature.feature_type == 'infrastructure'
                for evidence_id in feature.evidence_ids
            ]
        )
    if dimension == 'bottleneck':
        return _unique(
            [
                evidence_id
                for bottleneck in route_state.route_landscape.known_bottlenecks
                for evidence_id in bottleneck.evidence_ids
            ]
            + [
                evidence_id
                for feature in route_state.not_now_features.blocking_factors
                for evidence_id in feature.evidence_ids
            ]
            + list(route_state.evidence_bundle.challenging_evidence_ids)
        )
    if dimension == 'feasibility':
        return _unique(
            _dimension_evidence_ids(route_state, 'method_maturity')
            + _dimension_evidence_ids(route_state, 'measurement')
            + _dimension_evidence_ids(route_state, 'infrastructure')
            + _dimension_evidence_ids(route_state, 'cost_cycle')
        )
    if dimension == 'strategic_value':
        return _unique(
            [
                evidence_id
                for feature in route_state.why_now_features.unlocking_factors
                for evidence_id in feature.evidence_ids
            ]
            + [
                evidence_id
                for feature in route_state.why_now_features.positive_comparison_signals
                for evidence_id in feature.evidence_ids
            ]
            + [
                evidence_id
                for alternative in route_state.route_landscape.alternative_routes
                for evidence_id in alternative.evidence_ids
            ]
            + _dimension_evidence_ids(route_state, 'data_resource')
        )
    if dimension == 'novelty':
        return _unique(
            [
                evidence_id
                for feature in route_state.why_now_features.positive_comparison_signals
                for evidence_id in feature.evidence_ids
            ]
            + [
                evidence_id
                for alternative in route_state.route_landscape.alternative_routes
                for evidence_id in alternative.evidence_ids
            ]
        )
    if dimension == 'cost_cycle':
        return _unique(
            [
                evidence_id
                for feature in route_state.not_now_features.blocking_factors
                for evidence_id in feature.evidence_ids
            ]
            + list(route_state.evidence_bundle.challenging_evidence_ids)
        )
    return []


def _preferred_route(route_a_score: float | None, route_b_score: float | None, *, tie_threshold: float) -> str:
    if route_a_score is None or route_b_score is None:
        return 'unknown'
    gap = route_a_score - route_b_score
    if abs(gap) <= tie_threshold:
        return 'tie'
    return 'a' if gap > 0 else 'b'


def _dimension_rationale(
    dimension: str,
    route_a_score: float | None,
    route_b_score: float | None,
    preferred_route: str,
) -> str | None:
    if route_a_score is None or route_b_score is None:
        return f'The {dimension} evidence is incomplete for at least one route.'
    if preferred_route == 'tie':
        return f'The two routes are effectively tied on {dimension} ({route_a_score:.2f} vs {route_b_score:.2f}).'
    route_label = 'Route A' if preferred_route == 'a' else 'Route B'
    return f'{route_label} leads on {dimension} ({route_a_score:.2f} vs {route_b_score:.2f}).'


def _method_overlap(route_a: RouteState, route_b: RouteState) -> float:
    methods_a = {_normalize(method.label) for method in route_a.route_landscape.dominant_methods if _normalize(method.label)}
    methods_b = {_normalize(method.label) for method in route_b.route_landscape.dominant_methods if _normalize(method.label)}
    if not methods_a and not methods_b:
        return 0.0
    union = methods_a | methods_b
    if not union:
        return 0.0
    return len(methods_a & methods_b) / len(union)


def _support_overlap(route_a: RouteState, route_b: RouteState) -> float:
    support_a = {_normalize(value) for value in route_a.evidence_bundle.supporting_evidence_ids if _normalize(value)}
    support_b = {_normalize(value) for value in route_b.evidence_bundle.supporting_evidence_ids if _normalize(value)}
    if not support_a and not support_b:
        return 0.0
    union = support_a | support_b
    if not union:
        return 0.0
    return len(support_a & support_b) / len(union)


class RouteComparisonBuilder:
    def __init__(
        self,
        *,
        builder_version: str = 'route_comparison_builder_v1',
        tie_threshold: float = 0.05,
        decisive_gap: float = 0.12,
    ) -> None:
        self.builder_version = builder_version
        self.tie_threshold = tie_threshold
        self.decisive_gap = decisive_gap

    def build(
        self,
        route_a: RouteState,
        route_b: RouteState,
        *,
        built_at: str | None = None,
        route_comparison_case_id: str | None = None,
    ) -> RouteComparisonCase:
        if route_a.cutoff_year != route_b.cutoff_year:
            raise ValueError('RouteComparisonBuilder requires route states with the same cutoff_year')

        dimensions = [
            'method_maturity',
            'measurement',
            'data_resource',
            'infrastructure',
            'bottleneck',
            'feasibility',
            'strategic_value',
            'novelty',
            'cost_cycle',
        ]
        comparison_scores: list[ComparisonDimensionScore] = []
        weighted_margin = 0.0
        comparable_weight = 0.0
        for dimension in dimensions:
            route_a_score = _dimension_value(route_a, dimension)
            route_b_score = _dimension_value(route_b, dimension)
            preferred_route = _preferred_route(route_a_score, route_b_score, tie_threshold=self.tie_threshold)
            evidence_ids = _unique(_dimension_evidence_ids(route_a, dimension) + _dimension_evidence_ids(route_b, dimension))
            comparison_scores.append(
                ComparisonDimensionScore(
                    dimension=dimension,
                    route_a_score=route_a_score,
                    route_b_score=route_b_score,
                    preferred_route=preferred_route,
                    rationale=_dimension_rationale(dimension, route_a_score, route_b_score, preferred_route),
                    evidence_ids=evidence_ids,
                )
            )
            if route_a_score is not None and route_b_score is not None:
                weight = _DIMENSION_WEIGHTS[dimension]
                weighted_margin += weight * (route_a_score - route_b_score)
                comparable_weight += weight

        average_margin = (weighted_margin / comparable_weight) if comparable_weight else 0.0
        overall_tie_threshold = self.tie_threshold / 2.0
        if comparable_weight == 0.0:
            preference_label = 'unclear'
        elif abs(average_margin) <= overall_tie_threshold:
            preference_label = 'tie'
        else:
            preference_label = 'prefer_a' if average_margin > 0 else 'prefer_b'

        decisive_a = [
            score
            for score in comparison_scores
            if score.preferred_route == 'a'
            and score.route_a_score is not None
            and score.route_b_score is not None
            and abs(score.route_a_score - score.route_b_score) >= self.decisive_gap
        ]
        decisive_b = [
            score
            for score in comparison_scores
            if score.preferred_route == 'b'
            and score.route_a_score is not None
            and score.route_b_score is not None
            and abs(score.route_a_score - score.route_b_score) >= self.decisive_gap
        ]
        decisive_a.sort(key=lambda score: abs((score.route_a_score or 0.0) - (score.route_b_score or 0.0)), reverse=True)
        decisive_b.sort(key=lambda score: abs((score.route_a_score or 0.0) - (score.route_b_score or 0.0)), reverse=True)

        why_a_not_b = RoutePreferenceExplanation(
            summary=(
                f'Route A is strategically preferable because it leads on {", ".join(score.dimension for score in decisive_a[:3])}.'
                if preference_label == 'prefer_a' and decisive_a
                else 'Route A does not clearly dominate Route B under the currently comparable dimensions.'
            ),
            decisive_dimensions=[score.dimension for score in decisive_a[:3]],
            decisive_evidence_ids=_unique(
                [
                    evidence_id
                    for score in decisive_a[:3]
                    for evidence_id in score.evidence_ids
                ]
            ),
        )
        why_b_not_a = RoutePreferenceExplanation(
            summary=(
                f'Route B is strategically preferable because it leads on {", ".join(score.dimension for score in decisive_b[:3])}.'
                if preference_label == 'prefer_b' and decisive_b
                else 'Route B does not clearly dominate Route A under the currently comparable dimensions.'
            ),
            decisive_dimensions=[score.dimension for score in decisive_b[:3]],
            decisive_evidence_ids=_unique(
                [
                    evidence_id
                    for score in decisive_b[:3]
                    for evidence_id in score.evidence_ids
                ]
            ),
        )

        comparable_dimensions = [
            score.dimension
            for score in comparison_scores
            if score.route_a_score is not None and score.route_b_score is not None
        ]
        weak_dimensions = [
            score.dimension
            for score in comparison_scores
            if (
                score.route_a_score is not None
                and score.route_b_score is not None
                and abs(score.route_a_score - score.route_b_score) <= self.decisive_gap
            )
            or not score.evidence_ids
        ]
        quality_flags: list[str] = []
        if len(comparable_dimensions) < 4:
            quality_flags.append('comparison_dimensions_too_sparse')
        if preference_label == 'prefer_a' and (not why_a_not_b.decisive_dimensions or not why_a_not_b.decisive_evidence_ids):
            quality_flags.append('preference_not_grounded')
        if preference_label == 'prefer_b' and (not why_b_not_a.decisive_dimensions or not why_b_not_a.decisive_evidence_ids):
            quality_flags.append('preference_not_grounded')

        same_scope = _normalize(route_a.topic_scope) == _normalize(route_b.topic_scope)
        method_overlap = _method_overlap(route_a, route_b)
        support_overlap = _support_overlap(route_a, route_b)
        if same_scope and method_overlap >= 0.8 and support_overlap >= 0.5:
            quality_flags.append('same_route_disguised_as_two')
        if same_scope and method_overlap >= 0.8 and preference_label in {'tie', 'unclear'}:
            quality_flags.append('alternative_route_not_real')

        if 'same_route_disguised_as_two' in quality_flags:
            quality_tier = 'red'
        elif (
            route_a.quality.quality_tier == 'green'
            and route_b.quality.quality_tier == 'green'
            and not quality_flags
        ):
            quality_tier = 'green'
        elif comparable_dimensions and route_a.quality.quality_tier != 'red' and route_b.quality.quality_tier != 'red':
            quality_tier = 'yellow'
        else:
            quality_tier = 'red'

        return RouteComparisonCase(
            route_comparison_case_id=route_comparison_case_id or f'{route_a.route_state_id}__vs__{route_b.route_state_id}:route_comparison',
            built_at=built_at or _utc_now_iso(),
            cutoff_year=route_a.cutoff_year,
            route_a_state_id=route_a.route_state_id,
            route_b_state_id=route_b.route_state_id,
            comparison_dimension_scores=comparison_scores,
            preference_label=preference_label,
            why_a_not_b=why_a_not_b,
            why_b_not_a=why_b_not_a,
            evidence_chain=RouteComparisonEvidenceChain(
                route_a_support_ids=list(route_a.evidence_bundle.supporting_evidence_ids),
                route_b_support_ids=list(route_b.evidence_bundle.supporting_evidence_ids),
                cross_route_comparison_ids=_unique(
                    why_a_not_b.decisive_evidence_ids + why_b_not_a.decisive_evidence_ids
                ),
            ),
            uncertainty_points=RouteComparisonUncertaintyPoints(
                incomparable_dimensions=[
                    score.dimension
                    for score in comparison_scores
                    if score.route_a_score is None or score.route_b_score is None
                ],
                weak_dimensions=_unique(weak_dimensions),
                unresolved_conflicts=_unique(
                    list(route_a.uncertainty_points.unresolved_conflicts)
                    + list(route_b.uncertainty_points.unresolved_conflicts)
                ),
            ),
            quality=TrainingQuality(
                quality_tier=quality_tier,
                ready_for_training=quality_tier == 'green',
                quality_flags=_unique(quality_flags),
            ),
        )


def build_route_comparison_case(
    route_a: RouteState,
    route_b: RouteState,
    *,
    built_at: str | None = None,
    route_comparison_case_id: str | None = None,
    builder_version: str = 'route_comparison_builder_v1',
) -> RouteComparisonCase:
    builder = RouteComparisonBuilder(builder_version=builder_version)
    return builder.build(
        route_a,
        route_b,
        built_at=built_at,
        route_comparison_case_id=route_comparison_case_id,
    )


__all__ = ['RouteComparisonBuilder', 'build_route_comparison_case']
