from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import re

from .models import (
    AntiPatternCard,
    CardQuality,
    CounterexampleSet,
    FailureExamples,
    ReviewMetadata,
    RouteComparisonCase,
    RouteState,
    WarningSignal,
    WarningSignalPattern,
)
from .route_comparison_builder import build_route_comparison_case


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _slug(value: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', str(value or '').strip().lower())
    return slug.strip('_') or 'anti_pattern'


def _normalize(value: str | None) -> str:
    return str(value or '').strip().lower()


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


def _signal_type_from_field(field: str, label: str) -> str:
    normalized_field = _normalize(field)
    normalized_label = _normalize(label)
    if normalized_field == 'weak_fields' and 'measurement' in normalized_label:
        return 'weak_measurement'
    if normalized_field == 'blocking_factors' and 'cost' in normalized_label:
        return 'cost_spike'
    if normalized_field == 'missing_prerequisites':
        return 'resource_gap'
    if normalized_field == 'comparison':
        return 'contradiction'
    return 'route_fragility'


def _signal_severity(field: str) -> str:
    if field == 'comparison':
        return 'high'
    if field == 'missing_prerequisites':
        return 'blocking'
    return 'medium'


def _candidate_text(scope_label: str, repeated_label: str, source_field: str) -> str:
    if source_field == 'comparison':
        return (
            f'Do not keep scaling {scope_label} when nearby alternative routes repeatedly outperform the support cluster '
            f'under the same historical cutoff.'
        )
    if source_field == 'missing_prerequisites':
        return (
            f'Do not treat {scope_label} as reusable prior knowledge while the same prerequisite gap keeps recurring: '
            f'{repeated_label}.'
        )
    if source_field == 'weak_fields':
        return (
            f'Do not convert {scope_label} into a reusable prior when the same weak field keeps recurring across support '
            f'routes: {repeated_label}.'
        )
    return f'Do not scale {scope_label} while the repeated blocker "{repeated_label}" still shapes multiple support routes.'


def _corrective_checklist(source_field: str) -> list[str]:
    if source_field == 'comparison':
        return [
            'Re-run the route comparison against the stronger alternative before promoting the pattern.',
            'Record why the alternative wins and which bottleneck or readiness field drives the loss.',
            'Keep the cluster in candidate status until the contradiction is resolved.',
        ]
    if source_field == 'missing_prerequisites':
        return [
            'Resolve the missing prerequisite in at least one bounded replay slice.',
            'Verify the support cluster no longer repeats the same prerequisite gap.',
            'Re-check held-out consistency after the prerequisite is satisfied.',
        ]
    if source_field == 'weak_fields':
        return [
            'Strengthen the weak field with replay-visible evidence before reuse.',
            'Confirm the same weakness no longer appears across the support cluster.',
            'Keep the anti-pattern as a review checkpoint until the field stabilizes.',
        ]
    return [
        'Reduce the blocker pressure in at least one support route before reuse.',
        'Re-run comparison and held-out checks after the blocker changes.',
        'Keep the route scoped until the blocker stops recurring across the cluster.',
    ]


def _review_status(reviewer_ids: list[str]) -> str:
    return 'reviewed' if reviewer_ids else 'candidate'


def _quality_tier(
    *,
    reviewer_ids: list[str],
    failure_route_ids: list[str],
    counterexample_route_ids: list[str],
) -> str:
    if reviewer_ids and len(failure_route_ids) >= 2 and counterexample_route_ids:
        return 'green'
    if len(failure_route_ids) >= 2:
        return 'yellow'
    return 'red'


def _quality_flags(
    *,
    failure_route_ids: list[str],
    counterexample_route_ids: list[str],
    reviewer_ids: list[str],
    comparison_linked: bool,
) -> list[str]:
    flags: list[str] = []
    if len(failure_route_ids) < 2:
        flags.append('single_failure_example')
    if not counterexample_route_ids:
        flags.append('missing_counterexamples')
    if not reviewer_ids:
        flags.append('reviewer_missing')
    if not comparison_linked:
        flags.append('comparison_gap')
    return flags


class AntiPatternBuilder:
    def __init__(self, *, builder_version: str = 'anti_pattern_builder_v1') -> None:
        self.builder_version = builder_version

    def build_candidates(
        self,
        support_route_states: list[RouteState],
        *,
        alternative_route_states: list[RouteState] | None = None,
        comparison_cases: list[RouteComparisonCase] | None = None,
        reviewer_ids: list[str] | None = None,
        built_at: str | None = None,
        anti_pattern_prefix: str | None = None,
    ) -> list[AntiPatternCard]:
        if not support_route_states:
            return []

        alternative_route_states = list(alternative_route_states or [])
        reviewer_ids = _unique(list(reviewer_ids or []))
        support_route_ids = {route_state.route_state_id for route_state in support_route_states}
        scope_label = (
            support_route_states[0].scope_resolution.accepted_scope_label
            or support_route_states[0].topic_scope
        )

        comparison_cases = list(comparison_cases or [])
        if not comparison_cases and alternative_route_states:
            comparison_cases = [
                build_route_comparison_case(route_state, alternative_route_state, built_at=built_at)
                for route_state in support_route_states
                for alternative_route_state in alternative_route_states
            ]

        repeated_signals: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        counterexample_route_ids: dict[tuple[str, str], list[str]] = defaultdict(list)

        for route_state in support_route_states:
            for feature in route_state.not_now_features.blocking_factors:
                label = str(feature.label or '').strip()
                if label:
                    repeated_signals[('blocking_factors', label)].append((route_state.route_state_id, feature.source_field if hasattr(feature, 'source_field') else ''))
            for feature in route_state.not_now_features.missing_prerequisites:
                label = str(feature.label or '').strip()
                if label:
                    repeated_signals[('missing_prerequisites', label)].append((route_state.route_state_id, ''))
            for weak_field in route_state.uncertainty_points.weak_fields:
                label = str(weak_field or '').strip()
                if label:
                    repeated_signals[('weak_fields', label)].append((route_state.route_state_id, ''))

        for comparison_case in comparison_cases:
            if comparison_case.route_a_state_id in support_route_ids and comparison_case.preference_label == 'prefer_b':
                repeated_signals[('comparison', comparison_case.why_b_not_a.summary)].append((comparison_case.route_a_state_id, ''))
                counterexample_route_ids[('comparison', comparison_case.why_b_not_a.summary)].append(comparison_case.route_b_state_id)
            if comparison_case.route_b_state_id in support_route_ids and comparison_case.preference_label == 'prefer_a':
                repeated_signals[('comparison', comparison_case.why_a_not_b.summary)].append((comparison_case.route_b_state_id, ''))
                counterexample_route_ids[('comparison', comparison_case.why_a_not_b.summary)].append(comparison_case.route_a_state_id)

        for alternative_route_state in alternative_route_states:
            blocking_labels = {_normalize(feature.label) for feature in alternative_route_state.not_now_features.blocking_factors}
            missing_labels = {_normalize(feature.label) for feature in alternative_route_state.not_now_features.missing_prerequisites}
            weak_fields = {_normalize(field) for field in alternative_route_state.uncertainty_points.weak_fields}
            for key in list(repeated_signals):
                source_field, label = key
                normalized_label = _normalize(label)
                if (
                    (source_field == 'blocking_factors' and normalized_label not in blocking_labels)
                    or (source_field == 'missing_prerequisites' and normalized_label not in missing_labels)
                    or (source_field == 'weak_fields' and normalized_label not in weak_fields)
                ):
                    counterexample_route_ids[key].append(alternative_route_state.route_state_id)

        built_at_value = built_at or _utc_now_iso()
        candidates: list[AntiPatternCard] = []
        for index, ((source_field, label), route_pairs) in enumerate(sorted(repeated_signals.items()), start=1):
            failure_route_ids = _unique([route_state_id for route_state_id, _ in route_pairs])
            if len(failure_route_ids) < 2 and source_field != 'comparison':
                continue

            candidate_counterexamples = _unique(counterexample_route_ids.get((source_field, label), []))
            comparison_linked = source_field == 'comparison' or bool(comparison_cases)
            quality_tier = _quality_tier(
                reviewer_ids=reviewer_ids,
                failure_route_ids=failure_route_ids,
                counterexample_route_ids=candidate_counterexamples,
            )
            quality_flags = _quality_flags(
                failure_route_ids=failure_route_ids,
                counterexample_route_ids=candidate_counterexamples,
                reviewer_ids=reviewer_ids,
                comparison_linked=comparison_linked,
            )

            anti_pattern_id = anti_pattern_prefix or f'anti:{_slug(scope_label)}:{index:02d}:{_slug(label)}'
            candidates.append(
                AntiPatternCard(
                    anti_pattern_id=anti_pattern_id,
                    built_at=built_at_value,
                    anti_pattern_text=_candidate_text(scope_label, label, source_field),
                    warning_signal_pattern=WarningSignalPattern(
                        signals=[
                            WarningSignal(
                                label=label,
                                signal_type=_signal_type_from_field(source_field, label),
                                severity=_signal_severity(source_field),
                                source_field=source_field,
                            )
                        ],
                        trigger_logic='all' if len(failure_route_ids) >= 2 else 'any',
                    ),
                    failure_examples=FailureExamples(
                        route_state_ids=failure_route_ids,
                        decision_episode_ids=[],
                        notes=f'Repeated {source_field.replace("_", " ")} signal across the support cluster.',
                    ),
                    corrective_checklist=_corrective_checklist(source_field),
                    counterexamples=CounterexampleSet(
                        route_state_ids=candidate_counterexamples,
                        notes=(
                            'Routes without the recurring signal can remain counterexamples for review.'
                            if candidate_counterexamples
                            else 'No clean counterexample route is recorded yet.'
                        ),
                    ),
                    review=ReviewMetadata(
                        review_status=_review_status(reviewer_ids),
                        reviewer_notes='Candidate anti-pattern extracted from repeated replay-visible failure signals.',
                        reviewer_ids=reviewer_ids,
                    ),
                    quality=CardQuality(
                        quality_tier=quality_tier,
                        quality_flags=_unique(quality_flags),
                        audit_status='reviewed' if quality_tier == 'green' else 'eligible' if quality_tier == 'yellow' else 'hot_path',
                    ),
                )
            )

        return candidates


def build_anti_pattern_candidates(
    support_route_states: list[RouteState],
    *,
    alternative_route_states: list[RouteState] | None = None,
    comparison_cases: list[RouteComparisonCase] | None = None,
    reviewer_ids: list[str] | None = None,
    built_at: str | None = None,
    anti_pattern_prefix: str | None = None,
    builder_version: str = 'anti_pattern_builder_v1',
) -> list[AntiPatternCard]:
    builder = AntiPatternBuilder(builder_version=builder_version)
    return builder.build_candidates(
        support_route_states,
        alternative_route_states=alternative_route_states,
        comparison_cases=comparison_cases,
        reviewer_ids=reviewer_ids,
        built_at=built_at,
        anti_pattern_prefix=anti_pattern_prefix,
    )


__all__ = ['AntiPatternBuilder', 'build_anti_pattern_candidates']
