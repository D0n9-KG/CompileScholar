from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import re

from .models import (
    CardQuality,
    DecisionPriorCard,
    ExpectedFailureMode,
    HeldOutConsistency,
    PriorAppliesWhen,
    PriorCondition,
    PriorDoesNotApplyWhen,
    PriorThreshold,
    RecommendedAction,
    ReviewMetadata,
    RouteComparisonCase,
    RouteState,
    WhyNowCase,
)


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


def _slug(value: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', str(value or '').strip().lower())
    return slug.strip('_') or 'prior'


def _normalize(value: str) -> str:
    return str(value or '').strip().lower()


def _label_counts(values: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for value in values:
        normalized = str(value or '').strip()
        if normalized:
            counter[normalized] += 1
    return counter


def _top_labels(values: list[str], *, limit: int = 3) -> list[str]:
    counts = _label_counts(values)
    return [label for label, _ in counts.most_common(limit)]


def _why_now_lookup(why_now_cases: list[WhyNowCase] | None) -> dict[str, WhyNowCase]:
    if not why_now_cases:
        return {}
    return {case.route_state_id: case for case in why_now_cases}


def _average_readiness(route_states: list[RouteState], field: str) -> float | None:
    return _mean([getattr(route_state.readiness_scores, field) for route_state in route_states])


def _positive_why_now_ratio(route_states: list[RouteState], why_now_by_route_id: dict[str, WhyNowCase]) -> float:
    if not route_states:
        return 0.0
    positive_count = 0
    for route_state in route_states:
        why_now_case = why_now_by_route_id.get(route_state.route_state_id)
        if why_now_case and why_now_case.why_now_label in {'now', 'almost_now'}:
            positive_count += 1
    return round(positive_count / len(route_states), 2)


def _common_bottleneck_labels(route_states: list[RouteState]) -> list[str]:
    return _top_labels(
        [
            bottleneck.label
            for route_state in route_states
            for bottleneck in route_state.route_landscape.known_bottlenecks
        ]
    )


def _common_bottleneck_types(route_states: list[RouteState]) -> list[str]:
    return _top_labels(
        [
            bottleneck.bottleneck_type
            for route_state in route_states
            for bottleneck in route_state.route_landscape.known_bottlenecks
        ]
    )


def _common_weak_fields(route_states: list[RouteState]) -> list[str]:
    return _top_labels(
        [
            weak_field
            for route_state in route_states
            for weak_field in route_state.uncertainty_points.weak_fields
        ]
    )


def _counterexample_ids(
    support_route_ids: set[str],
    comparison_cases: list[RouteComparisonCase] | None,
) -> list[str]:
    if not comparison_cases:
        return []
    counterexamples: list[str] = []
    for comparison_case in comparison_cases:
        if comparison_case.route_a_state_id in support_route_ids and comparison_case.route_b_state_id not in support_route_ids:
            if comparison_case.preference_label == 'prefer_b':
                counterexamples.append(comparison_case.route_b_state_id)
        if comparison_case.route_b_state_id in support_route_ids and comparison_case.route_a_state_id not in support_route_ids:
            if comparison_case.preference_label == 'prefer_a':
                counterexamples.append(comparison_case.route_a_state_id)
    return _unique(counterexamples)


def _cluster_matches_prior(
    route_state: RouteState,
    *,
    method_threshold: float | None,
    data_threshold: float | None,
    measurement_threshold: float | None,
    overall_threshold: float | None,
    required_bottleneck_types: list[str],
) -> bool:
    readiness_checks = [
        route_state.readiness_scores.method is not None and method_threshold is not None and route_state.readiness_scores.method >= method_threshold,
        route_state.readiness_scores.data_resource is not None
        and data_threshold is not None
        and route_state.readiness_scores.data_resource >= data_threshold,
        route_state.readiness_scores.measurement is not None
        and measurement_threshold is not None
        and route_state.readiness_scores.measurement >= measurement_threshold,
        route_state.readiness_scores.overall is not None
        and overall_threshold is not None
        and route_state.readiness_scores.overall >= overall_threshold,
    ]
    readiness_hits = sum(1 for passed in readiness_checks if passed)
    bottleneck_types = {
        _normalize(bottleneck.bottleneck_type)
        for bottleneck in route_state.route_landscape.known_bottlenecks
        if _normalize(bottleneck.bottleneck_type)
    }
    bottleneck_match = not required_bottleneck_types or bool(bottleneck_types & {_normalize(value) for value in required_bottleneck_types})
    return readiness_hits >= 3 and bottleneck_match


class DecisionPriorBuilder:
    def __init__(self, *, builder_version: str = 'decision_prior_builder_v1') -> None:
        self.builder_version = builder_version

    def build(
        self,
        route_states: list[RouteState],
        *,
        why_now_cases: list[WhyNowCase] | None = None,
        comparison_cases: list[RouteComparisonCase] | None = None,
        held_out_route_states: list[RouteState] | None = None,
        reviewer_ids: list[str] | None = None,
        built_at: str | None = None,
        prior_id: str | None = None,
    ) -> DecisionPriorCard:
        if not route_states:
            raise ValueError('DecisionPriorBuilder requires at least one supporting RouteState')

        why_now_by_route_id = _why_now_lookup(why_now_cases)
        support_route_ids = [route_state.route_state_id for route_state in route_states]
        support_route_id_set = set(support_route_ids)
        held_out_route_states = held_out_route_states or []
        reviewer_ids = _unique(list(reviewer_ids or []))

        avg_method = _average_readiness(route_states, 'method')
        avg_measurement = _average_readiness(route_states, 'measurement')
        avg_data = _average_readiness(route_states, 'data_resource')
        avg_infrastructure = _average_readiness(route_states, 'infrastructure')
        avg_cost = _average_readiness(route_states, 'cost_cycle')
        avg_overall = _average_readiness(route_states, 'overall')
        positive_why_now_ratio = _positive_why_now_ratio(route_states, why_now_by_route_id)
        dominant_bottleneck_labels = _common_bottleneck_labels(route_states)
        dominant_bottleneck_types = _common_bottleneck_types(route_states)
        weak_fields = _common_weak_fields(route_states)

        if (avg_method or 0.0) >= 0.65 and (avg_data or 0.0) >= 0.7 and dominant_bottleneck_labels:
            prior_text = (
                'When method, measurement, and data readiness are strong but explicit bottlenecks remain visible, '
                'pursue the route through scoped validation rather than broad expansion.'
            )
        elif (avg_measurement or 0.0) < 0.6:
            prior_text = 'When route promise appears before measurement maturity, improve evaluation before committing to the route.'
        else:
            prior_text = 'When route readiness is directionally positive but still uneven, narrow scope and test the core assumption first.'

        readiness_pattern: list[PriorCondition] = []
        if avg_method is not None and avg_method >= 0.65:
            readiness_pattern.append(
                PriorCondition(
                    label='method readiness is high',
                    condition_type='readiness',
                    polarity='high',
                    required=True,
                )
            )
        if avg_data is not None and avg_data >= 0.65:
            readiness_pattern.append(
                PriorCondition(
                    label='data or benchmark support is high',
                    condition_type='readiness',
                    polarity='high',
                    required=True,
                )
            )
        if avg_measurement is not None and avg_measurement >= 0.6:
            readiness_pattern.append(
                PriorCondition(
                    label='measurement support is high enough to compare progress',
                    condition_type='readiness',
                    polarity='high',
                    required=True,
                )
            )

        bottleneck_pattern = [
            PriorCondition(
                label=f'{label} remains visible',
                condition_type='bottleneck',
                polarity='present',
                required=True,
            )
            for label in dominant_bottleneck_labels[:2]
        ]

        route_pattern: list[PriorCondition] = []
        if positive_why_now_ratio >= 0.5:
            route_pattern.append(
                PriorCondition(
                    label='why-now evidence is improving across the support cluster',
                    condition_type='route_shape',
                    polarity='improving',
                    required=False,
                )
            )
        if comparison_cases:
            preferred_cluster_comparisons = 0
            for comparison_case in comparison_cases:
                if comparison_case.route_a_state_id in support_route_id_set and comparison_case.preference_label == 'prefer_a':
                    preferred_cluster_comparisons += 1
                if comparison_case.route_b_state_id in support_route_id_set and comparison_case.preference_label == 'prefer_b':
                    preferred_cluster_comparisons += 1
            if preferred_cluster_comparisons:
                route_pattern.append(
                    PriorCondition(
                        label='comparison evidence still favors the route over nearby alternatives',
                        condition_type='comparison',
                        polarity='present',
                        required=False,
                    )
                )

        evidence_thresholds = []
        if avg_overall is not None:
            evidence_thresholds.append(
                PriorThreshold(
                    field='readiness_scores.overall',
                    operator='gte',
                    value=max(round(avg_overall - 0.05, 2), 0.0),
                )
            )
        evidence_thresholds.append(
            PriorThreshold(
                field='evidence_bundle.supporting_evidence_ids_count',
                operator='gte',
                value=2,
            )
        )

        blocker_pattern: list[PriorCondition] = []
        if avg_cost is not None and avg_cost < 0.45:
            blocker_pattern.append(
                PriorCondition(
                    label='cost or cycle burden is too high',
                    condition_type='bottleneck',
                    polarity='high',
                    required=True,
                )
            )
        if any(route_state.not_now_features.missing_prerequisites for route_state in route_states):
            blocker_pattern.append(
                PriorCondition(
                    label='a critical prerequisite is still absent',
                    condition_type='environment',
                    polarity='absent',
                    required=True,
                )
            )
        if not blocker_pattern:
            blocker_pattern.append(
                PriorCondition(
                    label='the dominant bottleneck becomes unresolved and route-shaping',
                    condition_type='bottleneck',
                    polarity='high',
                    required=True,
                )
            )

        fragility_pattern: list[PriorCondition] = []
        if weak_fields:
            fragility_pattern.extend(
                PriorCondition(
                    label=f'{field} remains a weak field',
                    condition_type='route_shape',
                    polarity='present',
                    required=False,
                )
                for field in weak_fields[:2]
            )

        mismatch_pattern: list[PriorCondition] = []
        if comparison_cases:
            mismatch_pattern.append(
                PriorCondition(
                    label='an alternative route becomes clearly preferred under the same cutoff',
                    condition_type='comparison',
                    polarity='present',
                    required=True,
                )
            )
        else:
            mismatch_pattern.append(
                PriorCondition(
                    label='comparison support against nearby alternatives is missing',
                    condition_type='comparison',
                    polarity='absent',
                    required=False,
                )
            )

        recommended_actions: list[RecommendedAction] = []
        if avg_measurement is not None and avg_measurement < 0.72:
            recommended_actions.append(
                RecommendedAction(
                    action_type='improve_measurement',
                    action_text='Improve measurement quality before increasing route commitment.',
                    target_route_feature='measurement',
                    confidence=0.78,
                )
            )
        if avg_data is not None and avg_data < 0.65:
            recommended_actions.append(
                RecommendedAction(
                    action_type='gather_resource',
                    action_text='Gather the missing resource or benchmark support before scaling the route.',
                    target_route_feature='data_resource',
                    confidence=0.76,
                )
            )
        if comparison_cases:
            recommended_actions.append(
                RecommendedAction(
                    action_type='compare_routes',
                    action_text='Keep the route paired with explicit comparison against nearby alternatives.',
                    target_route_feature='strategic_value',
                    confidence=0.81,
                )
            )
        if avg_overall is not None and avg_overall >= 0.65 and positive_why_now_ratio >= 0.5:
            recommended_actions.append(
                RecommendedAction(
                    action_type='pursue_question',
                    action_text='Pursue the route with a bounded, historically feasible question rather than a full commitment jump.',
                    target_route_feature='overall',
                    confidence=0.84,
                )
            )
        if dominant_bottleneck_labels:
            recommended_actions.append(
                RecommendedAction(
                    action_type='narrow_scope',
                    action_text='Narrow scope to the slice that avoids the dominant bottleneck becoming the whole program.',
                    target_route_feature=dominant_bottleneck_labels[0],
                    confidence=0.73,
                )
            )
        if not recommended_actions:
            recommended_actions.append(
                RecommendedAction(
                    action_type='test_assumption',
                    action_text='Test the core route assumption before escalating investment.',
                    target_route_feature=None,
                    confidence=0.65,
                )
            )

        expected_failure_modes = [
            ExpectedFailureMode(
                label='Dominant bottlenecks erase route-level gains before they compound.',
                linked_bottleneck_types=dominant_bottleneck_types[:3],
                warning_signals=_unique(dominant_bottleneck_labels[:2] + weak_fields[:2]),
                mitigation_hint='Keep the route scoped and pair progress claims with explicit bottleneck checks.',
            )
        ]

        counterexample_ids = _counterexample_ids(support_route_id_set, comparison_cases)
        counterexample_search_performed = bool(comparison_cases or held_out_route_states)

        held_out_route_ids = [route_state.route_state_id for route_state in held_out_route_states]
        held_out_results = [
            _cluster_matches_prior(
                route_state,
                method_threshold=max(round((avg_method or 0.0) - 0.1, 2), 0.0) if avg_method is not None else None,
                data_threshold=max(round((avg_data or 0.0) - 0.1, 2), 0.0) if avg_data is not None else None,
                measurement_threshold=max(round((avg_measurement or 0.0) - 0.1, 2), 0.0) if avg_measurement is not None else None,
                overall_threshold=max(round((avg_overall or 0.0) - 0.1, 2), 0.0) if avg_overall is not None else None,
                required_bottleneck_types=dominant_bottleneck_types,
            )
            for route_state in held_out_route_states
        ]
        held_out_pass_rate = (
            round(sum(1 for result in held_out_results if result) / len(held_out_results), 2)
            if held_out_results
            else None
        )

        quality_flags: list[str] = []
        if len(route_states) < 3:
            quality_flags.append('weak_support_cluster')
        if not counterexample_search_performed:
            quality_flags.append('missing_counterexample_search')
        if held_out_results and held_out_pass_rate is not None and held_out_pass_rate < 0.67:
            quality_flags.append('held_out_failure')

        review_status = 'reviewed' if reviewer_ids else 'candidate'
        if len(route_states) >= 3 and not quality_flags and held_out_pass_rate is not None and reviewer_ids:
            quality_tier = 'green'
        elif route_states and any(route_state.quality.quality_tier != 'red' for route_state in route_states):
            quality_tier = 'yellow'
        else:
            quality_tier = 'red'

        audit_status = 'reviewed' if quality_tier == 'green' else 'eligible' if quality_tier == 'yellow' else 'hot_path'
        topic_scope = route_states[0].topic_scope
        built_at_value = built_at or _utc_now_iso()

        return DecisionPriorCard(
            prior_id=prior_id or f'prior:{_slug(topic_scope)}:{route_states[0].cutoff_year}',
            built_at=built_at_value,
            prior_text=prior_text,
            applies_when=PriorAppliesWhen(
                readiness_pattern=readiness_pattern,
                bottleneck_pattern=bottleneck_pattern,
                route_pattern=route_pattern,
                evidence_thresholds=evidence_thresholds,
            ),
            does_not_apply_when=PriorDoesNotApplyWhen(
                blocker_pattern=blocker_pattern,
                fragility_pattern=fragility_pattern,
                mismatch_pattern=mismatch_pattern,
            ),
            recommended_actions=recommended_actions,
            expected_failure_modes=expected_failure_modes,
            supporting_route_state_ids=support_route_ids,
            counterexample_ids=counterexample_ids,
            held_out_consistency=HeldOutConsistency(
                held_out_route_state_ids=held_out_route_ids,
                pass_rate=held_out_pass_rate,
                failure_notes='Held-out support is inconsistent with the cluster prior.' if held_out_results and held_out_pass_rate is not None and held_out_pass_rate < 0.67 else None,
            ),
            review=ReviewMetadata(
                review_status=review_status,
                reviewer_notes='Rule-built decision prior reviewed for training eligibility.' if reviewer_ids else 'Awaiting expert review.',
                reviewer_ids=reviewer_ids,
            ),
            quality=CardQuality(
                quality_tier=quality_tier,
                quality_flags=_unique(quality_flags),
                audit_status=audit_status,
            ),
        )


def build_decision_prior_card(
    route_states: list[RouteState],
    *,
    why_now_cases: list[WhyNowCase] | None = None,
    comparison_cases: list[RouteComparisonCase] | None = None,
    held_out_route_states: list[RouteState] | None = None,
    reviewer_ids: list[str] | None = None,
    built_at: str | None = None,
    prior_id: str | None = None,
    builder_version: str = 'decision_prior_builder_v1',
) -> DecisionPriorCard:
    builder = DecisionPriorBuilder(builder_version=builder_version)
    return builder.build(
        route_states,
        why_now_cases=why_now_cases,
        comparison_cases=comparison_cases,
        held_out_route_states=held_out_route_states,
        reviewer_ids=reviewer_ids,
        built_at=built_at,
        prior_id=prior_id,
    )


__all__ = ['DecisionPriorBuilder', 'build_decision_prior_card']
