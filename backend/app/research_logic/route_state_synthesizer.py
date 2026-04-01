from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import re
from typing import Any, Iterable

from app.paper_logic_trace.derived_views import build_derived_views
from app.paper_logic_trace.models import PaperLogicTrace

from .models import (
    AlternativeRouteState,
    AvailabilityLevel,
    BenchmarkState,
    BottleneckState,
    CapabilityState,
    ConditionState,
    InfrastructureState,
    MethodState,
    NotNowFeatures,
    ProtocolState,
    ReadinessScores,
    RouteCompilerMetadata,
    RouteEvidenceBundle,
    RouteFeature,
    RouteLandscape,
    RoutePacket,
    RouteState,
    RouteStateQuality,
    RouteStateSourcePacket,
    RouteUncertaintyPoints,
    ScopeResolution,
    WhyNowFeatures,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value or '').strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _slug(value: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', str(value or '').strip().lower())
    return slug.strip('_') or 'route'


def _bounded_score(count: int, max_count: int) -> float | None:
    if count <= 0:
        return None
    if max_count <= 0:
        return 1.0
    return round(min(float(count) / float(max_count), 1.0), 2)


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        normalized = str(value or '').strip()
        if normalized:
            return normalized
    return None


def _derived_views(trace: PaperLogicTrace) -> dict[str, Any]:
    if not trace.derived_views:
        trace.derived_views = build_derived_views(trace)
        return trace.derived_views
    required = {'route_state_seed', 'route_compiler_contract', 'l1_bridge_hints'}
    if required.issubset(trace.derived_views):
        return trace.derived_views
    rebuilt = build_derived_views(trace)
    merged = dict(rebuilt)
    merged.update(trace.derived_views)
    trace.derived_views = merged
    return trace.derived_views


def _topic_scope_resolution(
    packet: RoutePacket,
    seeds: list[dict[str, Any]],
) -> ScopeResolution:
    counts: Counter[str] = Counter()
    for seed in seeds:
        for label in list(seed.get('topic_scope_candidates') or []):
            normalized = str(label or '').strip()
            if normalized:
                counts[normalized] += 1
    preferred = _first_non_empty(packet.compiler_hints.preferred_scope_label, packet.topic_scope_candidate)
    accepted = preferred or (counts.most_common(1)[0][0] if counts else '')
    candidates = _unique(
        [
            *(counts.keys()),
            str(packet.topic_scope_candidate or '').strip(),
            str(packet.compiler_hints.preferred_scope_label or '').strip(),
        ]
    )
    return ScopeResolution(
        topic_scope_candidates=candidates,
        accepted_scope_label=accepted,
        rejected_scope_labels=[label for label in candidates if label != accepted],
        resolution_rationale='Scope resolved from packet preference plus aggregated route_state_seed topic signals.',
        resolution_evidence_ids=_unique(
            evidence_id
            for seed in seeds
            for evidence_id in list(seed.get('supporting_evidence_ids') or [])[:2]
        ),
    )


def _aggregate_method_states(
    packet: RoutePacket,
    traces: list[PaperLogicTrace],
    seeds: list[dict[str, Any]],
) -> list[MethodState]:
    buckets: dict[str, dict[str, Any]] = {}
    preferred_labels = {str(label or '').strip().lower() for label in list(packet.compiler_hints.preferred_method_labels or []) if str(label or '').strip()}
    for trace, seed in zip(traces, seeds):
        for label in list(seed.get('dominant_method_candidates') or []):
            normalized = str(label or '').strip().lower()
            if not normalized:
                continue
            bucket = buckets.setdefault(
                normalized,
                {
                    'label': normalized,
                    'paper_ids': set(),
                    'move_ids': set(),
                    'evidence_ids': set(),
                },
            )
            bucket['paper_ids'].add(trace.paper_metadata.paper_id)
            bucket['move_ids'].update(str(move_id).strip() for move_id in list(seed.get('source_move_ids') or []) if str(move_id).strip())
            bucket['evidence_ids'].update(str(anchor_id).strip() for anchor_id in list(seed.get('supporting_evidence_ids') or []) if str(anchor_id).strip())

    states: list[MethodState] = []
    for label, bucket in sorted(
        buckets.items(),
        key=lambda item: (
            item[0] not in preferred_labels,
            -len(item[1]['paper_ids']),
            -len(item[1]['evidence_ids']),
            item[0],
        ),
    ):
        paper_count = len(bucket['paper_ids'])
        if paper_count >= 3:
            adoption = 'dominant'
        elif paper_count >= 2:
            adoption = 'established'
        else:
            adoption = 'workable'
        states.append(
            MethodState(
                label=label,
                family=None,
                maturity_score=_bounded_score(paper_count, 3),
                adoption_level=adoption,
                source_move_ids=sorted(bucket['move_ids']),
                source_paper_ids=sorted(bucket['paper_ids']),
                evidence_ids=sorted(bucket['evidence_ids']),
            )
        )
    return states


def _aggregate_benchmark_states(
    traces: list[PaperLogicTrace],
    seeds: list[dict[str, Any]],
) -> list[BenchmarkState]:
    buckets: dict[str, dict[str, set[str]]] = {}
    for trace, seed in zip(traces, seeds):
        for label in list(seed.get('active_benchmark_candidates') or []):
            normalized = str(label or '').strip().lower()
            if not normalized:
                continue
            bucket = buckets.setdefault(normalized, {'paper_ids': set(), 'evidence_ids': set()})
            bucket['paper_ids'].add(trace.paper_metadata.paper_id)
            bucket['evidence_ids'].update(str(anchor_id).strip() for anchor_id in list(seed.get('supporting_evidence_ids') or []) if str(anchor_id).strip())

    states: list[BenchmarkState] = []
    for label, bucket in sorted(buckets.items(), key=lambda item: (-len(item[1]['paper_ids']), item[0])):
        paper_count = len(bucket['paper_ids'])
        if paper_count >= 3:
            adoption = 'dominant'
        elif paper_count >= 1:
            adoption = 'active'
        else:
            adoption = 'unknown'
        states.append(
            BenchmarkState(
                label=label,
                benchmark_type='benchmark',
                adoption_level=adoption,
                source_paper_ids=sorted(bucket['paper_ids']),
                evidence_ids=sorted(bucket['evidence_ids']),
            )
        )
    return states


def _aggregate_protocol_states(
    traces: list[PaperLogicTrace],
    seeds: list[dict[str, Any]],
) -> list[ProtocolState]:
    buckets: dict[str, dict[str, set[str]]] = {}
    for trace, seed in zip(traces, seeds):
        for entry in list(seed.get('measurement_protocol_candidates') or []):
            label = _first_non_empty(
                ', '.join(str(token).strip() for token in list(entry.get('metric_tokens') or []) if str(token).strip()),
                str(entry.get('summary') or '').strip(),
            )
            normalized = str(label or '').strip().lower()
            if not normalized:
                continue
            bucket = buckets.setdefault(normalized, {'paper_ids': set(), 'evidence_ids': set()})
            bucket['paper_ids'].add(trace.paper_metadata.paper_id)
            bucket['evidence_ids'].update(str(anchor_id).strip() for anchor_id in list(entry.get('anchor_ids') or []) if str(anchor_id).strip())

    return [
        ProtocolState(
            label=label,
            protocol_type='measurement',
            maturity_score=_bounded_score(len(bucket['paper_ids']), 2),
            source_paper_ids=sorted(bucket['paper_ids']),
            evidence_ids=sorted(bucket['evidence_ids']),
        )
        for label, bucket in sorted(buckets.items(), key=lambda item: (-len(item[1]['paper_ids']), item[0]))
    ]


def _aggregate_infrastructure_states(
    traces: list[PaperLogicTrace],
    seeds: list[dict[str, Any]],
) -> list[InfrastructureState]:
    buckets: dict[str, dict[str, Any]] = {}
    for trace, seed in zip(traces, seeds):
        for entry in list(seed.get('toolchain_candidates') or []):
            resource_types = [str(token).strip().lower() for token in list(entry.get('resource_types') or []) if str(token).strip()]
            for token in list(entry.get('resource_tokens') or []):
                label = str(token or '').strip().lower()
                if not label:
                    continue
                bucket = buckets.setdefault(
                    label,
                    {'paper_ids': set(), 'evidence_ids': set(), 'resource_types': set()},
                )
                bucket['paper_ids'].add(trace.paper_metadata.paper_id)
                bucket['evidence_ids'].update(str(anchor_id).strip() for anchor_id in list(entry.get('anchor_ids') or []) if str(anchor_id).strip())
                bucket['resource_types'].update(resource_types)

    states: list[InfrastructureState] = []
    for label, bucket in sorted(buckets.items(), key=lambda item: (-len(item[1]['paper_ids']), item[0])):
        paper_count = len(bucket['paper_ids'])
        availability: AvailabilityLevel = 'abundant' if paper_count >= 3 else 'usable' if paper_count >= 1 else 'unknown'
        infra_type = sorted(bucket['resource_types'])[0] if bucket['resource_types'] else 'unknown'
        states.append(
            InfrastructureState(
                label=label,
                infra_type=infra_type,
                availability_level=availability,
                source_paper_ids=sorted(bucket['paper_ids']),
                evidence_ids=sorted(bucket['evidence_ids']),
            )
        )
    return states


def _aggregate_capability_states(
    traces: list[PaperLogicTrace],
    contracts: list[dict[str, Any]],
) -> list[CapabilityState]:
    states: list[CapabilityState] = []
    for trace, contract in zip(traces, contracts):
        for entry in list(contract.get('outcome_signals') or []):
            label = _first_non_empty(
                ', '.join(str(token).strip() for token in list(entry.get('metric_tokens') or []) if str(token).strip()),
                str(entry.get('summary') or '').strip(),
            )
            normalized = str(label or '').strip()
            if not normalized:
                continue
            states.append(
                CapabilityState(
                    label=normalized,
                    capability_type='prediction',
                    status='repeatable' if entry.get('comparator_tokens') else 'tentative',
                    metric_signals=_unique(str(token).strip() for token in list(entry.get('metric_tokens') or [])),
                    condition_signals=_unique(str(token).strip() for token in list(entry.get('condition_tokens') or [])),
                    source_paper_ids=[trace.paper_metadata.paper_id],
                    evidence_ids=_unique(str(anchor_id).strip() for anchor_id in list(entry.get('anchor_ids') or [])),
                )
            )
    return states[:10]


def _infer_bottleneck_type(label: str) -> str:
    lowered = str(label or '').lower()
    if any(token in lowered for token in ('compute', 'gpu', 'training')):
        return 'compute'
    if any(token in lowered for token in ('data', 'label', 'benchmark')):
        return 'data'
    if any(token in lowered for token in ('measurement', 'metric', 'evaluation')):
        return 'measurement'
    if any(token in lowered for token in ('resource', 'cost')):
        return 'resource'
    return 'unknown'


def _aggregate_bottleneck_states(
    traces: list[PaperLogicTrace],
    seeds: list[dict[str, Any]],
) -> list[BottleneckState]:
    buckets: dict[str, dict[str, set[str]]] = {}
    for trace, seed in zip(traces, seeds):
        for label in list(seed.get('known_bottleneck_candidates') or []):
            normalized = str(label or '').strip().lower()
            if not normalized:
                continue
            bucket = buckets.setdefault(normalized, {'paper_ids': set(), 'evidence_ids': set()})
            bucket['paper_ids'].add(trace.paper_metadata.paper_id)
            bucket['evidence_ids'].update(str(anchor_id).strip() for anchor_id in list(seed.get('challenging_evidence_ids') or []) if str(anchor_id).strip())

    states: list[BottleneckState] = []
    for label, bucket in sorted(buckets.items(), key=lambda item: (-len(item[1]['paper_ids']), item[0])):
        paper_count = len(bucket['paper_ids'])
        severity = 'blocking' if paper_count >= 3 else 'high' if paper_count >= 2 else 'medium'
        states.append(
            BottleneckState(
                label=label,
                bottleneck_type=_infer_bottleneck_type(label),
                severity=severity,
                blocking_scope='route_level',
                source_paper_ids=sorted(bucket['paper_ids']),
                evidence_ids=sorted(bucket['evidence_ids']),
                counterevidence_ids=[],
            )
        )
    return states


def _aggregate_condition_states(
    traces: list[PaperLogicTrace],
    seeds: list[dict[str, Any]],
) -> list[ConditionState]:
    buckets: dict[str, dict[str, set[str]]] = {}
    for trace, seed in zip(traces, seeds):
        for label in list(seed.get('enabling_condition_candidates') or []):
            normalized = str(label or '').strip().lower()
            if not normalized:
                continue
            bucket = buckets.setdefault(normalized, {'paper_ids': set(), 'evidence_ids': set()})
            bucket['paper_ids'].add(trace.paper_metadata.paper_id)
            bucket['evidence_ids'].update(str(anchor_id).strip() for anchor_id in list(seed.get('supporting_evidence_ids') or []) if str(anchor_id).strip())

    return [
        ConditionState(
            label=label,
            condition_type='resource',
            status='met',
            source_paper_ids=sorted(bucket['paper_ids']),
            evidence_ids=sorted(bucket['evidence_ids']),
        )
        for label, bucket in sorted(buckets.items(), key=lambda item: (-len(item[1]['paper_ids']), item[0]))
    ]


def _aggregate_alternative_routes(
    traces: list[PaperLogicTrace],
    seeds: list[dict[str, Any]],
    packet: RoutePacket,
) -> list[AlternativeRouteState]:
    buckets: dict[str, dict[str, set[str]]] = {}
    for trace, seed in zip(traces, seeds):
        for label in list(seed.get('alternative_route_candidates') or []):
            normalized = str(label or '').strip().lower()
            if not normalized:
                continue
            bucket = buckets.setdefault(normalized, {'paper_ids': set(), 'evidence_ids': set()})
            bucket['paper_ids'].add(trace.paper_metadata.paper_id)
            bucket['evidence_ids'].update(str(anchor_id).strip() for anchor_id in list(seed.get('supporting_evidence_ids') or []) if str(anchor_id).strip())
    for label in list(packet.compiler_hints.expected_alternative_routes or []):
        normalized = str(label or '').strip().lower()
        if not normalized:
            continue
        buckets.setdefault(normalized, {'paper_ids': set(), 'evidence_ids': set()})

    return [
        AlternativeRouteState(
            label=label,
            route_family=None,
            relation_to_main_route='competing',
            distinguishing_features=[],
            source_paper_ids=sorted(bucket['paper_ids']),
            evidence_ids=sorted(bucket['evidence_ids']),
        )
        for label, bucket in sorted(buckets.items(), key=lambda item: (-len(item[1]['paper_ids']), item[0]))
    ]


def _build_readiness_scores(
    packet: RoutePacket,
    landscape: RouteLandscape,
    seeds: list[dict[str, Any]],
) -> ReadinessScores:
    method_support = max((len(method.source_paper_ids) for method in landscape.dominant_methods), default=0)
    measurement_signal_count = max(
        (len(list((seed.get('readiness_feature_inputs') or {}).get('measurement_maturity_signals') or [])) for seed in seeds),
        default=0,
    )
    data_signal_count = max(
        (len(list((seed.get('readiness_feature_inputs') or {}).get('data_resource_signals') or [])) for seed in seeds),
        default=0,
    )
    infra_signal_count = max(
        (len(list((seed.get('readiness_feature_inputs') or {}).get('infrastructure_signals') or [])) for seed in seeds),
        default=0,
    )
    theory = _bounded_score(len({candidate for seed in seeds for candidate in list(seed.get('topic_scope_candidates') or [])}), 3)
    method = _bounded_score(method_support, 3)
    measurement = _bounded_score(max(len(landscape.measurement_protocols), measurement_signal_count), 3)
    data_resource = _bounded_score(
        max(
            len(landscape.active_benchmarks) + int(bool(packet.l1_snapshot_ref and packet.l1_snapshot_ref.benchmark_timeline_ref)),
            data_signal_count,
        ),
        3,
    )
    infrastructure = _bounded_score(
        max(
            len(landscape.toolchains_and_infrastructure) + int(bool(packet.l1_snapshot_ref and packet.l1_snapshot_ref.toolchain_timeline_ref)),
            infra_signal_count,
        ),
        3,
    )
    community = _bounded_score(len({seed.get('paper_id') for seed in seeds if str(seed.get('paper_id') or '').strip()}), max(2, len(seeds)))
    highest_severity = max(
        (
            {'low': 0.2, 'medium': 0.4, 'high': 0.7, 'blocking': 0.9, 'unknown': 0.3}.get(bottleneck.severity, 0.3)
            for bottleneck in landscape.known_bottlenecks
        ),
        default=0.25,
    )
    cost_cycle = round(max(0.0, 1.0 - highest_severity), 2)
    numeric_scores = [score for score in [theory, method, measurement, data_resource, infrastructure, community, cost_cycle] if score is not None]
    overall = round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else None

    strengths: list[str] = []
    if data_resource and data_resource >= 0.66:
        strengths.append('benchmark/data support is strong')
    if method and method >= 0.66:
        strengths.append('method evidence spans multiple papers')
    if measurement and measurement >= 0.5:
        strengths.append('measurement signals are present')
    constraints: list[str] = []
    if landscape.known_bottlenecks:
        constraints.append('known bottlenecks remain visible')
    if infrastructure is None or infrastructure < 0.5:
        constraints.append('infrastructure evidence is still thin')
    rationale_parts = []
    if strengths:
        rationale_parts.append(', '.join(strengths))
    if constraints:
        rationale_parts.append(', '.join(constraints))

    return ReadinessScores(
        theory=theory,
        method=method,
        measurement=measurement,
        data_resource=data_resource,
        infrastructure=infrastructure,
        community=community,
        cost_cycle=cost_cycle,
        overall=overall,
        score_rationale='; '.join(rationale_parts) if rationale_parts else None,
    )


def _build_why_now_features(
    packet: RoutePacket,
    landscape: RouteLandscape,
) -> WhyNowFeatures:
    l1_benchmark_ref = packet.l1_snapshot_ref.benchmark_timeline_ref if packet.l1_snapshot_ref else None
    l1_toolchain_ref = packet.l1_snapshot_ref.toolchain_timeline_ref if packet.l1_snapshot_ref else None
    unlocking_factors = [
        RouteFeature(
            label=benchmark.label,
            feature_type='benchmark_availability',
            direction='unlock',
            source_paper_ids=benchmark.source_paper_ids,
            evidence_ids=benchmark.evidence_ids,
            l1_refs=_unique([l1_benchmark_ref]),
            confidence=0.75,
        )
        for benchmark in landscape.active_benchmarks
    ]
    unlocking_factors.extend(
        RouteFeature(
            label=condition.label,
            feature_type='new_resource',
            direction='unlock',
            source_paper_ids=condition.source_paper_ids,
            evidence_ids=condition.evidence_ids,
            l1_refs=[],
            confidence=0.7,
        )
        for condition in landscape.enabling_conditions
    )
    acceleration_factors = [
        RouteFeature(
            label=method.label,
            feature_type='method_maturity',
            direction='accelerate',
            source_paper_ids=method.source_paper_ids,
            evidence_ids=method.evidence_ids,
            l1_refs=[],
            confidence=method.maturity_score,
        )
        for method in landscape.dominant_methods
        if method.maturity_score is not None and method.maturity_score >= 0.5
    ]
    if l1_toolchain_ref and landscape.toolchains_and_infrastructure:
        acceleration_factors.extend(
            RouteFeature(
                label=infra.label,
                feature_type='infrastructure',
                direction='accelerate',
                source_paper_ids=infra.source_paper_ids,
                evidence_ids=infra.evidence_ids,
                l1_refs=[l1_toolchain_ref],
                confidence=0.62,
            )
            for infra in landscape.toolchains_and_infrastructure
        )
    positive_comparison_signals = [
        RouteFeature(
            label=alternative.label,
            feature_type='comparative_gain',
            direction='accelerate',
            source_paper_ids=alternative.source_paper_ids,
            evidence_ids=alternative.evidence_ids,
            l1_refs=[],
            confidence=0.55,
        )
        for alternative in landscape.alternative_routes
    ]
    return WhyNowFeatures(
        unlocking_factors=unlocking_factors[:8],
        acceleration_factors=acceleration_factors[:8],
        positive_comparison_signals=positive_comparison_signals[:8],
    )


def _build_not_now_features(
    packet: RoutePacket,
    landscape: RouteLandscape,
) -> NotNowFeatures:
    l1_constraint_refs = _unique(
        [
            packet.l1_snapshot_ref.toolchain_timeline_ref if packet.l1_snapshot_ref else None,
            packet.l1_snapshot_ref.resource_timeline_ref if packet.l1_snapshot_ref else None,
        ]
    )
    blocking_factors = [
        RouteFeature(
            label=bottleneck.label,
            feature_type='blocker',
            direction='block',
            source_paper_ids=bottleneck.source_paper_ids,
            evidence_ids=bottleneck.evidence_ids,
            l1_refs=l1_constraint_refs,
            confidence=0.8 if bottleneck.severity in {'high', 'blocking'} else 0.6,
        )
        for bottleneck in landscape.known_bottlenecks
    ]
    missing_prerequisites: list[RouteFeature] = []
    if not landscape.active_benchmarks:
        missing_prerequisites.append(
            RouteFeature(
                label='benchmark coverage is missing',
                feature_type='missing_prerequisite',
                direction='warn',
                source_paper_ids=[],
                evidence_ids=[],
                l1_refs=_unique([packet.l1_snapshot_ref.benchmark_timeline_ref if packet.l1_snapshot_ref else None]),
                confidence=0.7,
            )
        )
    if not landscape.measurement_protocols:
        missing_prerequisites.append(
            RouteFeature(
                label='measurement protocol evidence is missing',
                feature_type='missing_prerequisite',
                direction='warn',
                source_paper_ids=[],
                evidence_ids=[],
                l1_refs=_unique([packet.l1_snapshot_ref.protocol_registry_ref if packet.l1_snapshot_ref else None]),
                confidence=0.7,
            )
        )
    return NotNowFeatures(
        blocking_factors=blocking_factors[:8],
        fragility_factors=[],
        missing_prerequisites=missing_prerequisites[:8],
    )


def _build_evidence_bundle(
    packet: RoutePacket,
    traces: list[PaperLogicTrace],
    seeds: list[dict[str, Any]],
) -> RouteEvidenceBundle:
    return RouteEvidenceBundle(
        supporting_evidence_ids=_unique(
            anchor_id
            for seed in seeds
            for anchor_id in list(seed.get('supporting_evidence_ids') or [])
        ),
        challenging_evidence_ids=_unique(
            anchor_id
            for seed in seeds
            for anchor_id in list(seed.get('challenging_evidence_ids') or [])
        ),
        representative_move_ids=_unique(
            move_id
            for seed in seeds
            for move_id in list(seed.get('source_move_ids') or [])[:4]
        )[:12],
        representative_paper_ids=_unique(trace.paper_metadata.paper_id for trace in traces),
        l1_support_refs=_unique(
            [
                packet.l1_snapshot_ref.resource_registry_ref if packet.l1_snapshot_ref else None,
                packet.l1_snapshot_ref.benchmark_timeline_ref if packet.l1_snapshot_ref else None,
                packet.l1_snapshot_ref.protocol_registry_ref if packet.l1_snapshot_ref else None,
            ]
        ),
        l1_constraint_refs=_unique(
            [
                packet.l1_snapshot_ref.resource_timeline_ref if packet.l1_snapshot_ref else None,
                packet.l1_snapshot_ref.toolchain_timeline_ref if packet.l1_snapshot_ref else None,
            ]
        ),
    )


def _build_uncertainty_points(
    landscape: RouteLandscape,
    evidence_bundle: RouteEvidenceBundle,
) -> RouteUncertaintyPoints:
    weak_fields: list[str] = []
    if not landscape.measurement_protocols:
        weak_fields.append('measurement_protocols')
    if not landscape.toolchains_and_infrastructure:
        weak_fields.append('toolchains_and_infrastructure')
    if not landscape.alternative_routes:
        weak_fields.append('alternative_routes')
    unresolved_conflicts = []
    if not evidence_bundle.challenging_evidence_ids:
        unresolved_conflicts.append('challenging evidence is missing at route level')
    return RouteUncertaintyPoints(
        open_questions=['packet-level route synthesis still needs richer L1 joins for broader replay coverage'],
        unresolved_conflicts=unresolved_conflicts,
        weak_fields=weak_fields,
        low_confidence_clusters=[],
    )


def _build_quality(
    packet: RoutePacket,
    scope_resolution: ScopeResolution,
    landscape: RouteLandscape,
    evidence_bundle: RouteEvidenceBundle,
    why_now_features: WhyNowFeatures,
) -> RouteStateQuality:
    flags: list[str] = []
    has_scope = bool(scope_resolution.accepted_scope_label)
    method_multi_paper = any(len(method.source_paper_ids) >= 2 for method in landscape.dominant_methods)
    has_any_method = bool(landscape.dominant_methods)
    has_bottleneck = bool(landscape.known_bottlenecks)
    has_enabling = bool(landscape.enabling_conditions or why_now_features.unlocking_factors)
    has_alternative = bool(landscape.alternative_routes)
    has_support = bool(evidence_bundle.supporting_evidence_ids)
    has_challenge = bool(evidence_bundle.challenging_evidence_ids)
    has_l1 = bool(evidence_bundle.l1_support_refs or packet.l1_snapshot_ref)

    if not has_scope:
        flags.append('missing_topic_scope')
    if not has_any_method:
        flags.append('missing_dominant_method')
    if has_any_method and not method_multi_paper:
        flags.append('dominant_method_not_multi_paper')
    if not has_bottleneck:
        flags.append('missing_bottleneck_signal')
    if not has_enabling:
        flags.append('missing_enabling_signal')
    if not has_alternative:
        flags.append('missing_alternative_route')
    if not has_support:
        flags.append('missing_supporting_evidence')
    if not has_challenge:
        flags.append('missing_challenging_evidence')
    if not has_l1:
        flags.append('missing_l1_support')

    green = all(
        [
            has_scope,
            method_multi_paper,
            has_bottleneck,
            has_enabling,
            has_alternative,
            has_support,
            has_challenge,
            has_l1,
        ]
    )
    yellow = has_scope and has_any_method and has_support
    quality_tier = 'green' if green else 'yellow' if yellow else 'red'
    ready_for_why_now = quality_tier != 'red' and has_support
    ready_for_route_comparison = quality_tier == 'green'
    ready_for_prior_selection = quality_tier == 'green'
    if quality_tier == 'green' and packet.packet_quality.manual_review_status == 'completed':
        audit_status = 'reviewed'
    elif quality_tier != 'red':
        audit_status = 'eligible'
    else:
        audit_status = 'hot_path'
    return RouteStateQuality(
        quality_tier=quality_tier,
        ready_for_why_now=ready_for_why_now,
        ready_for_route_comparison=ready_for_route_comparison,
        ready_for_prior_selection=ready_for_prior_selection,
        quality_flags=flags,
        audit_status=audit_status,
    )


class RouteStateSynthesizer:
    def __init__(
        self,
        *,
        compiler_version: str = 'route_state_synthesizer_v1',
        compile_mode: str = 'rule_only',
    ) -> None:
        self.compiler_version = compiler_version
        self.compile_mode = compile_mode

    def synthesize(
        self,
        packet: RoutePacket | dict[str, Any],
        traces: list[PaperLogicTrace],
        *,
        built_at: str | None = None,
        route_state_id: str | None = None,
    ) -> RouteState:
        packet_model = packet if isinstance(packet, RoutePacket) else RoutePacket.model_validate(packet)
        if not packet_model.included_items:
            raise ValueError('RouteStateSynthesizer requires packet included_items')

        trace_by_id = {trace.trace_id: trace for trace in traces}
        included_items = [item for item in packet_model.included_items if str(item.trace_id or '').strip()]
        missing_trace_ids = [str(item.trace_id) for item in included_items if str(item.trace_id) not in trace_by_id]
        if missing_trace_ids:
            raise ValueError(f'missing traces for packet items: {", ".join(missing_trace_ids)}')

        selected_traces: list[PaperLogicTrace] = []
        for item in included_items:
            trace = trace_by_id[str(item.trace_id)]
            if str(item.paper_id).strip() and str(item.paper_id).strip() != str(trace.paper_metadata.paper_id).strip():
                raise ValueError(f'packet paper_id mismatch for trace {trace.trace_id}')
            paper_year = trace.paper_metadata.year if trace.paper_metadata.year is not None else item.paper_year
            if paper_year is not None and int(paper_year) > int(packet_model.cutoff_year):
                raise ValueError(f'trace {trace.trace_id} exceeds packet cutoff_year {packet_model.cutoff_year}')
            selected_traces.append(trace)

        derived_views = [_derived_views(trace) for trace in selected_traces]
        seeds = [dict(views.get('route_state_seed') or {}) for views in derived_views]
        contracts = [dict(views.get('route_compiler_contract') or {}) for views in derived_views]

        scope_resolution = _topic_scope_resolution(packet_model, seeds)
        landscape = RouteLandscape(
            dominant_methods=_aggregate_method_states(packet_model, selected_traces, seeds),
            active_benchmarks=_aggregate_benchmark_states(selected_traces, seeds),
            measurement_protocols=_aggregate_protocol_states(selected_traces, seeds),
            toolchains_and_infrastructure=_aggregate_infrastructure_states(selected_traces, seeds),
            known_capabilities=_aggregate_capability_states(selected_traces, contracts),
            known_bottlenecks=_aggregate_bottleneck_states(selected_traces, seeds),
            enabling_conditions=_aggregate_condition_states(selected_traces, seeds),
            alternative_routes=_aggregate_alternative_routes(selected_traces, seeds, packet_model),
        )
        readiness_scores = _build_readiness_scores(packet_model, landscape, seeds)
        why_now_features = _build_why_now_features(packet_model, landscape)
        not_now_features = _build_not_now_features(packet_model, landscape)
        evidence_bundle = _build_evidence_bundle(packet_model, selected_traces, seeds)
        uncertainty_points = _build_uncertainty_points(landscape, evidence_bundle)
        quality = _build_quality(packet_model, scope_resolution, landscape, evidence_bundle, why_now_features)

        built_at_value = built_at or _utc_now_iso()
        accepted_scope = scope_resolution.accepted_scope_label
        route_state_id_value = route_state_id or f'{_slug(accepted_scope)}_{packet_model.cutoff_year}_{_slug(packet_model.packet_id)}_{packet_model.schema_version}'

        return RouteState(
            route_state_id=route_state_id_value,
            built_at=built_at_value,
            topic_scope=accepted_scope,
            cutoff_year=packet_model.cutoff_year,
            source_packet=RouteStateSourcePacket(
                packet_id=packet_model.packet_id,
                included_trace_ids=[str(item.trace_id) for item in included_items],
                included_paper_ids=[item.paper_id for item in included_items],
                packet_role_counts=packet_model.packet_composition.role_counts,
                l1_snapshot_ref=packet_model.l1_snapshot_ref.snapshot_id if packet_model.l1_snapshot_ref else None,
            ),
            scope_resolution=scope_resolution,
            route_landscape=landscape,
            readiness_scores=readiness_scores,
            why_now_features=why_now_features,
            not_now_features=not_now_features,
            evidence_bundle=evidence_bundle,
            uncertainty_points=uncertainty_points,
            compiler_metadata=RouteCompilerMetadata(
                compiler_version=self.compiler_version,
                packet_builder_version=f'packet_builder_{packet_model.schema_version}',
                l1_snapshot_version=packet_model.l1_snapshot_ref.snapshot_id if packet_model.l1_snapshot_ref else None,
                trace_versions={trace.trace_id: trace.schema_version for trace in selected_traces},
                compile_mode=self.compile_mode,
                llm_usage_notes=None if self.compile_mode == 'rule_only' else 'LLM usage, if any, is constrained to rationale phrasing.',
            ),
            quality=quality,
        )


def synthesize_route_state(
    packet: RoutePacket | dict[str, Any],
    traces: list[PaperLogicTrace],
    *,
    built_at: str | None = None,
    route_state_id: str | None = None,
    compiler_version: str = 'route_state_synthesizer_v1',
    compile_mode: str = 'rule_only',
) -> RouteState:
    synthesizer = RouteStateSynthesizer(
        compiler_version=compiler_version,
        compile_mode=compile_mode,
    )
    return synthesizer.synthesize(
        packet,
        traces,
        built_at=built_at,
        route_state_id=route_state_id,
    )


__all__ = ['RouteStateSynthesizer', 'synthesize_route_state']
