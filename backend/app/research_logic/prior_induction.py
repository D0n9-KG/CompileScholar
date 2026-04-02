from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import re
from typing import Any

from pydantic import ConfigDict, Field

from .anti_pattern_builder import AntiPatternBuilder
from .decision_prior_builder import DecisionPriorBuilder
from .models import ContractModel, AntiPatternCard, DecisionPriorCard, RouteComparisonCase, RouteState
from .route_comparison_builder import build_route_comparison_case
from .route_state_package import LoadedRouteStatePackage, RouteStatePackageCompilation


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _slug(value: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', str(value or '').strip().lower())
    return slug.strip('_') or 'cluster'


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


def _score_bucket(value: float | None) -> str:
    if value is None:
        return 'unknown'
    if value >= 0.75:
        return 'high'
    if value >= 0.55:
        return 'medium'
    return 'low'


def _readiness_bucket(route_state: RouteState) -> str:
    readiness = route_state.readiness_scores
    return ':'.join(
        [
            f'method-{_score_bucket(readiness.method)}',
            f'measurement-{_score_bucket(readiness.measurement)}',
            f'data-{_score_bucket(readiness.data_resource)}',
            f'overall-{_score_bucket(readiness.overall)}',
        ]
    )


def _method_labels(route_state: RouteState) -> list[str]:
    labels = [
        _normalize(method.label) or _normalize(method.family)
        for method in route_state.route_landscape.dominant_methods[:2]
        if _normalize(method.label) or _normalize(method.family)
    ]
    return sorted(_unique(labels))


def _method_families(route_state: RouteState) -> list[str]:
    families = [
        _normalize(method.family)
        for method in route_state.route_landscape.dominant_methods[:2]
        if _normalize(method.family)
    ]
    return sorted(_unique(families))


def _bottleneck_types(route_state: RouteState) -> list[str]:
    values = [
        _normalize(bottleneck.bottleneck_type)
        for bottleneck in route_state.route_landscape.known_bottlenecks[:2]
        if _normalize(bottleneck.bottleneck_type)
    ]
    return sorted(_unique(values))


def _cluster_id(
    *,
    scope_label: str,
    method_labels: list[str],
    method_families: list[str],
    bottleneck_types: list[str],
    readiness_bucket: str,
) -> str:
    return _slug(
        '::'.join(
            [
                scope_label,
                '|'.join(method_labels),
                '|'.join(method_families),
                '|'.join(bottleneck_types),
                readiness_bucket,
            ]
        )
    )


def _comparison_cases_for_cluster(
    support_route_states: list[RouteState],
    *,
    alternative_route_states: list[RouteState],
    comparison_cases: list[RouteComparisonCase] | None,
    built_at: str | None,
) -> list[RouteComparisonCase]:
    support_route_ids = {route_state.route_state_id for route_state in support_route_states}
    if comparison_cases:
        return [
            comparison_case
            for comparison_case in comparison_cases
            if comparison_case.route_a_state_id in support_route_ids or comparison_case.route_b_state_id in support_route_ids
        ]
    return [
        build_route_comparison_case(route_state, alternative_route_state, built_at=built_at)
        for route_state in support_route_states
        for alternative_route_state in alternative_route_states
    ]


def _group_route_states(route_states: list[RouteState]) -> list[tuple['PriorSupportCluster', list[RouteState]]]:
    grouped: dict[str, list[RouteState]] = defaultdict(list)
    cluster_lookup: dict[str, PriorSupportCluster] = {}

    for route_state in route_states:
        scope_label = route_state.scope_resolution.accepted_scope_label or route_state.topic_scope
        method_labels = _method_labels(route_state)
        method_families = _method_families(route_state)
        bottleneck_types = _bottleneck_types(route_state)
        readiness_bucket = _readiness_bucket(route_state)
        cluster_id = _cluster_id(
            scope_label=scope_label,
            method_labels=method_labels,
            method_families=method_families,
            bottleneck_types=bottleneck_types,
            readiness_bucket=readiness_bucket,
        )
        grouped[cluster_id].append(route_state)
        cluster_lookup[cluster_id] = PriorSupportCluster(
            cluster_id=cluster_id,
            scope_label=scope_label,
            dominant_method_labels=method_labels,
            dominant_method_families=method_families,
            bottleneck_types=bottleneck_types,
            readiness_bucket=readiness_bucket,
            support_route_state_ids=[],
            support_count=0,
        )

    clusters: list[tuple[PriorSupportCluster, list[RouteState]]] = []
    for cluster_id, cluster_route_states in grouped.items():
        cluster = cluster_lookup[cluster_id].model_copy(
            update={
                'support_route_state_ids': [route_state.route_state_id for route_state in cluster_route_states],
                'support_count': len(cluster_route_states),
            }
        )
        clusters.append((cluster, cluster_route_states))
    return sorted(clusters, key=lambda item: item[0].cluster_id)


def _quality_flag_counts(prior_candidates: list[DecisionPriorCard], anti_pattern_candidates: list[AntiPatternCard]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for card in [*prior_candidates, *anti_pattern_candidates]:
        for flag in card.quality.quality_flags:
            counts[flag] += 1
    return dict(sorted(counts.items()))


class PriorSupportCluster(ContractModel):
    cluster_id: str
    scope_label: str
    dominant_method_labels: list[str] = Field(default_factory=list)
    dominant_method_families: list[str] = Field(default_factory=list)
    bottleneck_types: list[str] = Field(default_factory=list)
    readiness_bucket: str
    support_route_state_ids: list[str] = Field(default_factory=list)
    support_count: int = 0


class PriorCandidateRegistry(ContractModel):
    model_config = ConfigDict(extra='forbid')

    built_at: str
    package_id: str | None = None
    clusters: list[PriorSupportCluster] = Field(default_factory=list)
    prior_candidates: list[DecisionPriorCard] = Field(default_factory=list)
    anti_pattern_candidates: list[AntiPatternCard] = Field(default_factory=list)
    accepted_prior_ids: list[str] = Field(default_factory=list)
    accepted_anti_pattern_ids: list[str] = Field(default_factory=list)
    quality_flag_counts: dict[str, int] = Field(default_factory=dict)

    def accepted_prior_cards(self) -> list[DecisionPriorCard]:
        accepted_ids = set(self.accepted_prior_ids)
        return [card for card in self.prior_candidates if card.prior_id in accepted_ids]

    def accepted_anti_pattern_cards(self) -> list[AntiPatternCard]:
        accepted_ids = set(self.accepted_anti_pattern_ids)
        return [card for card in self.anti_pattern_candidates if card.anti_pattern_id in accepted_ids]


class PriorInductionEngine:
    def __init__(
        self,
        *,
        prior_builder_version: str = 'decision_prior_builder_v1',
        anti_pattern_builder_version: str = 'anti_pattern_builder_v1',
    ) -> None:
        self.prior_builder = DecisionPriorBuilder(builder_version=prior_builder_version)
        self.anti_pattern_builder = AntiPatternBuilder(builder_version=anti_pattern_builder_version)

    def build_registry(
        self,
        *,
        support_route_states: list[RouteState],
        alternative_route_states: list[RouteState] | None = None,
        held_out_route_states: list[RouteState] | None = None,
        comparison_cases: list[RouteComparisonCase] | None = None,
        reviewer_ids: list[str] | None = None,
        built_at: str | None = None,
        package_id: str | None = None,
    ) -> PriorCandidateRegistry:
        support_route_states = list(support_route_states)
        alternative_route_states = list(alternative_route_states or [])
        held_out_route_states = list(held_out_route_states or [])
        reviewer_ids = _unique(list(reviewer_ids or []))
        built_at_value = built_at or _utc_now_iso()

        clusters_with_routes = _group_route_states(support_route_states)
        clusters = [cluster for cluster, _ in clusters_with_routes]
        prior_candidates: list[DecisionPriorCard] = []
        anti_pattern_candidates: list[AntiPatternCard] = []

        for cluster, cluster_route_states in clusters_with_routes:
            cluster_comparisons = _comparison_cases_for_cluster(
                cluster_route_states,
                alternative_route_states=alternative_route_states,
                comparison_cases=comparison_cases,
                built_at=built_at_value,
            )
            if len({route_state.route_state_id for route_state in cluster_route_states}) >= 2:
                prior_candidates.append(
                    self.prior_builder.build(
                        cluster_route_states,
                        comparison_cases=cluster_comparisons,
                        held_out_route_states=held_out_route_states,
                        reviewer_ids=reviewer_ids,
                        built_at=built_at_value,
                        prior_id=f'prior:{cluster.cluster_id}',
                    )
                )
            anti_pattern_candidates.extend(
                self.anti_pattern_builder.build_candidates(
                    cluster_route_states,
                    alternative_route_states=alternative_route_states,
                    comparison_cases=cluster_comparisons,
                    reviewer_ids=reviewer_ids,
                    built_at=built_at_value,
                )
            )

        return PriorCandidateRegistry(
            built_at=built_at_value,
            package_id=package_id,
            clusters=clusters,
            prior_candidates=prior_candidates,
            anti_pattern_candidates=anti_pattern_candidates,
            accepted_prior_ids=[card.prior_id for card in prior_candidates if card.quality.quality_tier == 'green'],
            accepted_anti_pattern_ids=[
                card.anti_pattern_id for card in anti_pattern_candidates if card.quality.quality_tier == 'green'
            ],
            quality_flag_counts=_quality_flag_counts(prior_candidates, anti_pattern_candidates),
        )


def build_prior_candidate_registry(
    *,
    support_route_states: list[RouteState],
    alternative_route_states: list[RouteState] | None = None,
    held_out_route_states: list[RouteState] | None = None,
    comparison_cases: list[RouteComparisonCase] | None = None,
    reviewer_ids: list[str] | None = None,
    built_at: str | None = None,
    package_id: str | None = None,
    prior_builder_version: str = 'decision_prior_builder_v1',
    anti_pattern_builder_version: str = 'anti_pattern_builder_v1',
) -> PriorCandidateRegistry:
    engine = PriorInductionEngine(
        prior_builder_version=prior_builder_version,
        anti_pattern_builder_version=anti_pattern_builder_version,
    )
    return engine.build_registry(
        support_route_states=support_route_states,
        alternative_route_states=alternative_route_states,
        held_out_route_states=held_out_route_states,
        comparison_cases=comparison_cases,
        reviewer_ids=reviewer_ids,
        built_at=built_at,
        package_id=package_id,
    )


def _package_induction_inputs(
    package: LoadedRouteStatePackage | RouteStatePackageCompilation,
) -> tuple[str | None, dict[str, list[RouteState]]]:
    if isinstance(package, LoadedRouteStatePackage):
        return package.manifest.package_id, package.induction_inputs()
    return package.package_id, package.induction_inputs()


def build_prior_candidate_registry_from_package(
    package: LoadedRouteStatePackage | RouteStatePackageCompilation,
    *,
    reviewer_ids: list[str] | None = None,
    built_at: str | None = None,
) -> PriorCandidateRegistry:
    package_id, grouped_inputs = _package_induction_inputs(package)
    return build_prior_candidate_registry(
        support_route_states=grouped_inputs['support_route_states'],
        alternative_route_states=grouped_inputs['alternative_route_states'],
        held_out_route_states=grouped_inputs['held_out_route_states'],
        reviewer_ids=reviewer_ids,
        built_at=built_at,
        package_id=package_id,
    )


__all__ = [
    'PriorCandidateRegistry',
    'PriorInductionEngine',
    'PriorSupportCluster',
    'build_prior_candidate_registry',
    'build_prior_candidate_registry_from_package',
]
