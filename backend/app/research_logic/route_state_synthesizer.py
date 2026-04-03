from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import re
from typing import Any, Iterable

from app.paper_logic_trace.derived_views import build_derived_views
from app.paper_logic_trace.models import PaperLogicTrace

from .historical_environment import HistoricalEnvironmentSnapshot, build_l1_snapshot_ref
from .models import (
    AlternativeRouteState,
    AvailabilityLevel,
    BenchmarkState,
    BottleneckState,
    CapabilityState,
    ConditionState,
    InfrastructureState,
    L1SnapshotRef,
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


def _build_route_family_id(
    scope_label: str,
    cutoff_year: int,
    dominant_methods: list[MethodState],
    *,
    preferred_method_labels: Iterable[str] | None = None,
) -> str:
    primary_method_label = _first_non_empty(
        next((str(label).strip() for label in preferred_method_labels or [] if str(label).strip()), None),
        dominant_methods[0].label if dominant_methods else None,
        dominant_methods[0].family if dominant_methods else None,
    )
    primary_method_key = _slug(primary_method_label or 'route')
    return f'route_family:{_slug(scope_label)}:{cutoff_year}:{primary_method_key}'


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


def _normalized_label(value: str | None) -> str:
    return str(value or '').strip().lower()


def _status_label(value: str | None) -> str:
    return str(value or 'unknown').strip().lower() or 'unknown'


def _is_l1_support_status(status: str | None) -> bool:
    return _status_label(status) in {'available', 'emerging'}


def _is_l1_constraint_status(status: str | None) -> bool:
    return _status_label(status) in {'contested', 'missing', 'unknown'}


def _resource_condition_type(resource_types: Iterable[str]) -> ConditionState.__annotations__['condition_type']:
    normalized_types = {_normalized_label(token) for token in resource_types if _normalized_label(token)}
    if normalized_types & {'benchmark', 'dataset', 'corpus', 'data'}:
        return 'data'
    if normalized_types & {'software', 'hardware', 'platform', 'instrument', 'compute', 'analysis_tool', 'protocol'}:
        return 'tool'
    return 'resource'


def _resource_infra_type(resource_types: Iterable[str]) -> InfrastructureState.__annotations__['infra_type']:
    normalized_types = [_normalized_label(token) for token in resource_types if _normalized_label(token)]
    for token in normalized_types:
        if token in {'software', 'hardware', 'platform', 'instrument', 'compute'}:
            return token
        if token == 'analysis_tool':
            return 'software'
        if token == 'protocol':
            return 'platform'
    return 'unknown'


def _condition_status_from_l1(status: str | None) -> ConditionState.__annotations__['status']:
    normalized = _status_label(status)
    if normalized == 'available':
        return 'met'
    if normalized == 'emerging':
        return 'partially_met'
    if normalized == 'contested':
        return 'unstable'
    if normalized == 'missing':
        return 'unmet'
    return 'unknown'


def _availability_from_l1(status: str | None, source_count: int) -> AvailabilityLevel:
    normalized = _status_label(status)
    if normalized == 'available':
        return 'abundant' if source_count >= 2 else 'usable'
    if normalized == 'emerging':
        return 'limited'
    if normalized == 'missing':
        return 'unavailable'
    return 'unknown'


def _benchmark_adoption_from_l1(status: str | None, source_count: int) -> BenchmarkState.__annotations__['adoption_level']:
    normalized = _status_label(status)
    if normalized == 'available':
        return 'dominant' if source_count >= 2 else 'active'
    if normalized == 'emerging':
        return 'emerging'
    if normalized == 'missing':
        return 'absent'
    return 'unknown'


def _l1_confidence(entry: Any, fallback: float) -> float:
    confidence = getattr(entry, 'confidence', None)
    if confidence is not None:
        return float(confidence)
    return fallback


def _effective_l1_snapshot(
    packet: RoutePacket,
    l1_snapshot: HistoricalEnvironmentSnapshot | dict[str, Any] | None,
) -> tuple[HistoricalEnvironmentSnapshot | None, L1SnapshotRef | None]:
    if l1_snapshot is None:
        return None, packet.l1_snapshot_ref

    snapshot_model = (
        l1_snapshot
        if isinstance(l1_snapshot, HistoricalEnvironmentSnapshot)
        else HistoricalEnvironmentSnapshot.model_validate(l1_snapshot)
    )
    if snapshot_model.cutoff_year != packet.cutoff_year:
        raise ValueError('l1_snapshot cutoff_year must match RoutePacket cutoff_year')
    return snapshot_model, build_l1_snapshot_ref(snapshot_model)


def _merge_route_features(features: list[RouteFeature]) -> list[RouteFeature]:
    merged: dict[tuple[str, str, str], RouteFeature] = {}
    for feature in features:
        key = (_normalized_label(feature.label), feature.feature_type, feature.direction)
        existing = merged.get(key)
        if existing is None:
            merged[key] = feature
            continue
        merged[key] = existing.model_copy(
            update={
                'source_paper_ids': _unique([*existing.source_paper_ids, *feature.source_paper_ids]),
                'evidence_ids': _unique([*existing.evidence_ids, *feature.evidence_ids]),
                'l1_refs': _unique([*existing.l1_refs, *feature.l1_refs]),
                'confidence': max(
                    confidence
                    for confidence in [existing.confidence, feature.confidence]
                    if confidence is not None
                )
                if any(confidence is not None for confidence in [existing.confidence, feature.confidence])
                else None,
            }
        )
    return list(merged.values())


def _merge_benchmark_states(
    states: list[BenchmarkState],
    l1_snapshot: HistoricalEnvironmentSnapshot | None,
) -> list[BenchmarkState]:
    merged: dict[str, BenchmarkState] = {_normalized_label(state.label): state for state in states}
    adoption_rank = {'absent': 0, 'unknown': 0, 'emerging': 1, 'active': 2, 'dominant': 3}

    for entry in list(l1_snapshot.benchmark_timeline if l1_snapshot else []):
        if not _is_l1_support_status(entry.status):
            continue
        key = _normalized_label(entry.label)
        if not key:
            continue
        l1_state = BenchmarkState(
            label=entry.label,
            benchmark_type=entry.benchmark_type if entry.benchmark_type in {'dataset', 'benchmark', 'task_suite'} else 'unknown',
            adoption_level=_benchmark_adoption_from_l1(entry.status, len(entry.source_paper_ids)),
            source_paper_ids=list(entry.source_paper_ids),
            evidence_ids=list(entry.evidence_ids),
        )
        existing = merged.get(key)
        if existing is None:
            merged[key] = l1_state
            continue
        merged[key] = existing.model_copy(
            update={
                'benchmark_type': existing.benchmark_type if existing.benchmark_type != 'unknown' else l1_state.benchmark_type,
                'adoption_level': existing.adoption_level
                if adoption_rank.get(existing.adoption_level, 0) >= adoption_rank.get(l1_state.adoption_level, 0)
                else l1_state.adoption_level,
                'source_paper_ids': _unique([*existing.source_paper_ids, *l1_state.source_paper_ids]),
                'evidence_ids': _unique([*existing.evidence_ids, *l1_state.evidence_ids]),
            }
        )

    return list(merged.values())


def _merge_protocol_states(
    states: list[ProtocolState],
    l1_snapshot: HistoricalEnvironmentSnapshot | None,
) -> list[ProtocolState]:
    merged: dict[str, ProtocolState] = {_normalized_label(state.label): state for state in states}

    for entry in list(l1_snapshot.protocol_registry if l1_snapshot else []):
        if not _is_l1_support_status(entry.status):
            continue
        key = _normalized_label(entry.label)
        if not key:
            continue
        l1_state = ProtocolState(
            label=entry.label,
            protocol_type='measurement',
            maturity_score=_l1_confidence(entry, 0.7 if _status_label(entry.status) == 'available' else 0.55),
            source_paper_ids=list(entry.source_paper_ids),
            evidence_ids=list(entry.evidence_ids),
        )
        existing = merged.get(key)
        if existing is None:
            merged[key] = l1_state
            continue
        merged[key] = existing.model_copy(
            update={
                'protocol_type': existing.protocol_type if existing.protocol_type != 'unknown' else l1_state.protocol_type,
                'maturity_score': max(
                    score
                    for score in [existing.maturity_score, l1_state.maturity_score]
                    if score is not None
                )
                if any(score is not None for score in [existing.maturity_score, l1_state.maturity_score])
                else None,
                'source_paper_ids': _unique([*existing.source_paper_ids, *l1_state.source_paper_ids]),
                'evidence_ids': _unique([*existing.evidence_ids, *l1_state.evidence_ids]),
            }
        )

    return list(merged.values())


def _merge_infrastructure_states(
    states: list[InfrastructureState],
    l1_snapshot: HistoricalEnvironmentSnapshot | None,
) -> list[InfrastructureState]:
    merged: dict[str, InfrastructureState] = {_normalized_label(state.label): state for state in states}

    if l1_snapshot is None:
        return list(merged.values())

    infra_entries: list[tuple[str, list[str], list[str], list[str], str | None]] = [
        (entry.label, list(entry.source_paper_ids), list(entry.evidence_ids), list(entry.resource_types), entry.status)
        for entry in l1_snapshot.toolchain_timeline
        if _is_l1_support_status(entry.status)
    ]
    infra_entries.extend(
        (entry.label, list(entry.source_paper_ids), list(entry.evidence_ids), list(entry.resource_types), entry.status)
        for entry in l1_snapshot.resource_registry
        if _is_l1_support_status(entry.status) and _resource_infra_type(entry.resource_types) != 'unknown'
    )

    availability_rank = {'unknown': 0, 'limited': 1, 'usable': 2, 'abundant': 3}
    for label, source_paper_ids, evidence_ids, resource_types, status in infra_entries:
        key = _normalized_label(label)
        if not key:
            continue
        l1_state = InfrastructureState(
            label=label,
            infra_type=_resource_infra_type(resource_types),
            availability_level=_availability_from_l1(status, len(source_paper_ids)),
            source_paper_ids=source_paper_ids,
            evidence_ids=evidence_ids,
        )
        existing = merged.get(key)
        if existing is None:
            merged[key] = l1_state
            continue
        merged[key] = existing.model_copy(
            update={
                'infra_type': existing.infra_type if existing.infra_type != 'unknown' else l1_state.infra_type,
                'availability_level': existing.availability_level
                if availability_rank.get(existing.availability_level, 0) >= availability_rank.get(l1_state.availability_level, 0)
                else l1_state.availability_level,
                'source_paper_ids': _unique([*existing.source_paper_ids, *l1_state.source_paper_ids]),
                'evidence_ids': _unique([*existing.evidence_ids, *l1_state.evidence_ids]),
            }
        )

    return list(merged.values())


def _merge_condition_states(
    states: list[ConditionState],
    l1_snapshot: HistoricalEnvironmentSnapshot | None,
) -> list[ConditionState]:
    merged: dict[str, ConditionState] = {_normalized_label(state.label): state for state in states}
    status_rank = {'unknown': 0, 'unmet': 0, 'partially_met': 1, 'unstable': 1, 'met': 2}

    for entry in list(l1_snapshot.resource_registry if l1_snapshot else []):
        if not _is_l1_support_status(entry.status):
            continue
        key = _normalized_label(entry.label)
        if not key:
            continue
        l1_state = ConditionState(
            label=entry.label,
            condition_type=_resource_condition_type(entry.resource_types),
            status=_condition_status_from_l1(entry.status),
            source_paper_ids=list(entry.source_paper_ids),
            evidence_ids=list(entry.evidence_ids),
        )
        existing = merged.get(key)
        if existing is None:
            merged[key] = l1_state
            continue
        merged[key] = existing.model_copy(
            update={
                'condition_type': existing.condition_type if existing.condition_type != 'unknown' else l1_state.condition_type,
                'status': existing.status
                if status_rank.get(existing.status, 0) >= status_rank.get(l1_state.status, 0)
                else l1_state.status,
                'source_paper_ids': _unique([*existing.source_paper_ids, *l1_state.source_paper_ids]),
                'evidence_ids': _unique([*existing.evidence_ids, *l1_state.evidence_ids]),
            }
        )

    return list(merged.values())


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
        comparison_evidence_ids = _unique(
            str(anchor_id).strip()
            for entry in list(seed.get('comparison_signal_entries') or [])
            for anchor_id in list(entry.get('evidence_ids') or entry.get('anchor_ids') or [])
            if str(anchor_id).strip()
        )
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
            bucket['evidence_ids'].update(comparison_evidence_ids)

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
        infra_type = _resource_infra_type(bucket['resource_types'])
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
    seeds: list[dict[str, Any]],
) -> list[CapabilityState]:
    states: list[CapabilityState] = []
    for trace, contract, seed in zip(traces, contracts, seeds):
        comparison_entries = list(seed.get('comparison_signal_entries') or contract.get('comparison_signals') or [])
        comparison_evidence_ids = _unique(
            str(anchor_id).strip()
            for entry in comparison_entries
            for anchor_id in list(entry.get('evidence_ids') or entry.get('anchor_ids') or [])
            if str(anchor_id).strip()
        )
        has_comparison_support = bool(comparison_entries)
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
                    status='repeatable' if entry.get('comparator_tokens') or has_comparison_support else 'tentative',
                    metric_signals=_unique(str(token).strip() for token in list(entry.get('metric_tokens') or [])),
                    condition_signals=_unique(str(token).strip() for token in list(entry.get('condition_tokens') or [])),
                    source_paper_ids=[trace.paper_metadata.paper_id],
                    evidence_ids=_unique(
                        [
                            *[
                                str(anchor_id).strip()
                                for anchor_id in list(entry.get('anchor_ids') or [])
                                if str(anchor_id).strip()
                            ],
                            *comparison_evidence_ids,
                        ]
                    ),
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

    def _signal_label(entry: dict[str, Any]) -> str:
        derived_label = str(entry.get('derived_label') or '').strip().lower()
        if derived_label:
            return derived_label
        methods = [str(token).strip().lower() for token in list(entry.get('method_tokens') or []) if str(token).strip()]
        targets = [str(token).strip().lower() for token in list(entry.get('target_object_tokens') or []) if str(token).strip()]
        resources = [str(token).strip().lower() for token in list(entry.get('resource_tokens') or []) if str(token).strip()]
        conditions = [str(token).strip().lower() for token in list(entry.get('condition_tokens') or []) if str(token).strip()]
        if methods and targets:
            return f'{methods[0]} -> {targets[0]}'
        if targets and resources:
            return f'{targets[0]} via {resources[0]}'
        if methods and resources:
            return f'{methods[0]} with {resources[0]}'
        if methods:
            return methods[0]
        if targets:
            return targets[0]
        if resources:
            return resources[0]
        if conditions:
            return conditions[0]
        return str(entry.get('summary') or '').strip().lower()

    for trace, seed in zip(traces, seeds):
        signal_entries = list(seed.get('alternative_route_signal_entries') or [])
        for entry in signal_entries:
            normalized = _signal_label(entry)
            if not normalized:
                continue
            bucket = buckets.setdefault(normalized, {'paper_ids': set(), 'evidence_ids': set(), 'features': set()})
            bucket['paper_ids'].add(trace.paper_metadata.paper_id)
            bucket['evidence_ids'].update(
                str(anchor_id).strip()
                for anchor_id in list(entry.get('evidence_ids') or entry.get('anchor_ids') or [])
                if str(anchor_id).strip()
            )
            bucket['features'].update(
                str(token).strip().lower()
                for field in ('condition_tokens', 'limitation_tokens', 'resource_tokens')
                for token in list(entry.get(field) or [])
                if str(token).strip()
            )
        if signal_entries:
            continue
        for label in list(seed.get('alternative_route_candidates') or []):
            normalized = str(label or '').strip().lower()
            if not normalized:
                continue
            bucket = buckets.setdefault(normalized, {'paper_ids': set(), 'evidence_ids': set(), 'features': set()})
            bucket['paper_ids'].add(trace.paper_metadata.paper_id)
            bucket['evidence_ids'].update(str(anchor_id).strip() for anchor_id in list(seed.get('supporting_evidence_ids') or []) if str(anchor_id).strip())
    for label in list(packet.compiler_hints.expected_alternative_routes or []):
        normalized = str(label or '').strip().lower()
        if not normalized:
            continue
        buckets.setdefault(normalized, {'paper_ids': set(), 'evidence_ids': set(), 'features': set()})

    return [
        AlternativeRouteState(
            label=label,
            route_family=None,
            relation_to_main_route='competing',
            distinguishing_features=sorted(bucket['features'])[:6],
            source_paper_ids=sorted(bucket['paper_ids']),
            evidence_ids=sorted(bucket['evidence_ids']),
        )
        for label, bucket in sorted(buckets.items(), key=lambda item: (-len(item[1]['paper_ids']), item[0]))
    ]


def _build_readiness_scores(
    packet: RoutePacket,
    landscape: RouteLandscape,
    seeds: list[dict[str, Any]],
    *,
    l1_snapshot: HistoricalEnvironmentSnapshot | None = None,
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
    l1_measurement_signal_count = (
        sum(1 for entry in l1_snapshot.protocol_registry if _is_l1_support_status(entry.status))
        if l1_snapshot
        else 0
    )
    measurement = _bounded_score(max(len(landscape.measurement_protocols), measurement_signal_count, l1_measurement_signal_count), 3)
    l1_data_signal_count = (
        max(
            sum(1 for entry in l1_snapshot.benchmark_timeline if _is_l1_support_status(entry.status)),
            sum(
                1
                for entry in l1_snapshot.resource_registry
                if _is_l1_support_status(entry.status) and _resource_condition_type(entry.resource_types) == 'data'
            ),
        )
        if l1_snapshot
        else int(bool(packet.l1_snapshot_ref and packet.l1_snapshot_ref.benchmark_timeline_ref))
    )
    l1_data_baseline = (
        int(
            bool(packet.l1_snapshot_ref and packet.l1_snapshot_ref.benchmark_timeline_ref)
            and bool(
                (l1_snapshot.benchmark_timeline if l1_snapshot else [])
                or [
                    entry
                    for entry in (l1_snapshot.resource_registry if l1_snapshot else [])
                    if _resource_condition_type(entry.resource_types) == 'data'
                ]
            )
        )
        if l1_snapshot
        else 0
    )
    data_resource = _bounded_score(
        (
            max(
                len(landscape.active_benchmarks),
                data_signal_count,
                l1_data_signal_count,
                l1_data_baseline,
            )
            if l1_snapshot
            else max(
                len(landscape.active_benchmarks) + l1_data_signal_count,
                data_signal_count,
            )
        ),
        3,
    )
    l1_infra_signal_count = (
        max(
            sum(1 for entry in l1_snapshot.toolchain_timeline if _is_l1_support_status(entry.status)),
            sum(
                1
                for entry in l1_snapshot.resource_registry
                if _is_l1_support_status(entry.status) and _resource_infra_type(entry.resource_types) != 'unknown'
            ),
        )
        if l1_snapshot
        else int(bool(packet.l1_snapshot_ref and packet.l1_snapshot_ref.toolchain_timeline_ref))
    )
    l1_infra_baseline = (
        int(
            bool(packet.l1_snapshot_ref and packet.l1_snapshot_ref.toolchain_timeline_ref)
            and bool(
                (l1_snapshot.toolchain_timeline if l1_snapshot else [])
                or [
                    entry
                    for entry in (l1_snapshot.resource_registry if l1_snapshot else [])
                    if _resource_infra_type(entry.resource_types) != 'unknown'
                ]
            )
        )
        if l1_snapshot
        else 0
    )
    infrastructure = _bounded_score(
        (
            max(
                len(landscape.toolchains_and_infrastructure),
                infra_signal_count,
                l1_infra_signal_count,
                l1_infra_baseline,
            )
            if l1_snapshot
            else max(
                len(landscape.toolchains_and_infrastructure) + l1_infra_signal_count,
                infra_signal_count,
            )
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
    if l1_snapshot and l1_infra_signal_count > 0:
        strengths.append('historical L1 environment adds concrete infrastructure evidence')
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
    *,
    l1_snapshot: HistoricalEnvironmentSnapshot | None = None,
) -> WhyNowFeatures:
    l1_benchmark_ref = packet.l1_snapshot_ref.benchmark_timeline_ref if packet.l1_snapshot_ref else None
    l1_toolchain_ref = packet.l1_snapshot_ref.toolchain_timeline_ref if packet.l1_snapshot_ref else None
    l1_protocol_ref = packet.l1_snapshot_ref.protocol_registry_ref if packet.l1_snapshot_ref else None
    l1_resource_ref = packet.l1_snapshot_ref.resource_registry_ref if packet.l1_snapshot_ref else None
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
            l1_refs=_unique([l1_resource_ref]) if l1_snapshot else [],
            confidence=0.7 if condition.status == 'met' else 0.58,
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
    if l1_snapshot is not None and l1_protocol_ref and landscape.measurement_protocols:
        acceleration_factors.extend(
            RouteFeature(
                label=protocol.label,
                feature_type='protocol',
                direction='accelerate',
                source_paper_ids=protocol.source_paper_ids,
                evidence_ids=protocol.evidence_ids,
                l1_refs=[l1_protocol_ref],
                confidence=protocol.maturity_score or 0.6,
            )
            for protocol in landscape.measurement_protocols
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
        unlocking_factors=_merge_route_features(unlocking_factors)[:8],
        acceleration_factors=_merge_route_features(acceleration_factors)[:8],
        positive_comparison_signals=_merge_route_features(positive_comparison_signals)[:8],
    )


def _build_not_now_features(
    packet: RoutePacket,
    landscape: RouteLandscape,
    *,
    l1_snapshot: HistoricalEnvironmentSnapshot | None = None,
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
    fragility_factors: list[RouteFeature] = []
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
    if l1_snapshot:
        for entry in l1_snapshot.benchmark_timeline:
            if _status_label(entry.status) == 'missing':
                missing_prerequisites.append(
                    RouteFeature(
                        label=f'benchmark {entry.label} is still missing before the cutoff',
                        feature_type='missing_prerequisite',
                        direction='warn',
                        source_paper_ids=list(entry.source_paper_ids),
                        evidence_ids=list(entry.evidence_ids),
                        l1_refs=_unique([packet.l1_snapshot_ref.benchmark_timeline_ref if packet.l1_snapshot_ref else None]),
                        confidence=_l1_confidence(entry, 0.72),
                    )
                )
            elif _status_label(entry.status) == 'contested':
                fragility_factors.append(
                    RouteFeature(
                        label=f'benchmark availability around {entry.label} is contested',
                        feature_type='fragility',
                        direction='warn',
                        source_paper_ids=list(entry.source_paper_ids),
                        evidence_ids=list(entry.evidence_ids),
                        l1_refs=_unique([packet.l1_snapshot_ref.benchmark_timeline_ref if packet.l1_snapshot_ref else None]),
                        confidence=_l1_confidence(entry, 0.58),
                    )
                )
        for entry in [*l1_snapshot.toolchain_timeline, *l1_snapshot.resource_registry]:
            if _status_label(entry.status) == 'missing':
                missing_prerequisites.append(
                    RouteFeature(
                        label=f'required resource {entry.label} is not yet available before the cutoff',
                        feature_type='missing_prerequisite',
                        direction='warn',
                        source_paper_ids=list(entry.source_paper_ids),
                        evidence_ids=list(entry.evidence_ids),
                        l1_refs=l1_constraint_refs,
                        confidence=_l1_confidence(entry, 0.7),
                    )
                )
            elif _status_label(entry.status) == 'contested':
                fragility_factors.append(
                    RouteFeature(
                        label=f'resource readiness around {entry.label} is contested',
                        feature_type='fragility',
                        direction='warn',
                        source_paper_ids=list(entry.source_paper_ids),
                        evidence_ids=list(entry.evidence_ids),
                        l1_refs=l1_constraint_refs,
                        confidence=_l1_confidence(entry, 0.56),
                    )
                )
        for entry in l1_snapshot.protocol_registry:
            if _status_label(entry.status) == 'missing':
                missing_prerequisites.append(
                    RouteFeature(
                        label=f'measurement protocol {entry.label} is still missing before the cutoff',
                        feature_type='missing_prerequisite',
                        direction='warn',
                        source_paper_ids=list(entry.source_paper_ids),
                        evidence_ids=list(entry.evidence_ids),
                        l1_refs=_unique([packet.l1_snapshot_ref.protocol_registry_ref if packet.l1_snapshot_ref else None]),
                        confidence=_l1_confidence(entry, 0.7),
                    )
                )
            elif _status_label(entry.status) == 'contested':
                fragility_factors.append(
                    RouteFeature(
                        label=f'measurement protocol {entry.label} remains contested',
                        feature_type='fragility',
                        direction='warn',
                        source_paper_ids=list(entry.source_paper_ids),
                        evidence_ids=list(entry.evidence_ids),
                        l1_refs=_unique([packet.l1_snapshot_ref.protocol_registry_ref if packet.l1_snapshot_ref else None]),
                        confidence=_l1_confidence(entry, 0.56),
                    )
                )
    return NotNowFeatures(
        blocking_factors=blocking_factors[:8],
        fragility_factors=_merge_route_features(fragility_factors)[:8],
        missing_prerequisites=_merge_route_features(missing_prerequisites)[:8],
    )


def _build_evidence_bundle(
    packet: RoutePacket,
    traces: list[PaperLogicTrace],
    seeds: list[dict[str, Any]],
    *,
    l1_snapshot: HistoricalEnvironmentSnapshot | None = None,
) -> RouteEvidenceBundle:
    l1_supporting_evidence_ids = _unique(
        evidence_id
        for entry in [
            *(l1_snapshot.benchmark_timeline if l1_snapshot else []),
            *(l1_snapshot.toolchain_timeline if l1_snapshot else []),
            *(l1_snapshot.protocol_registry if l1_snapshot else []),
            *(l1_snapshot.resource_registry if l1_snapshot else []),
        ]
        if _is_l1_support_status(getattr(entry, 'status', None))
        for evidence_id in list(getattr(entry, 'evidence_ids', []) or [])
    )
    l1_challenging_evidence_ids = _unique(
        evidence_id
        for entry in [
            *(l1_snapshot.benchmark_timeline if l1_snapshot else []),
            *(l1_snapshot.toolchain_timeline if l1_snapshot else []),
            *(l1_snapshot.protocol_registry if l1_snapshot else []),
            *(l1_snapshot.resource_registry if l1_snapshot else []),
        ]
        if _is_l1_constraint_status(getattr(entry, 'status', None))
        for evidence_id in list(getattr(entry, 'evidence_ids', []) or [])
    )
    return RouteEvidenceBundle(
        supporting_evidence_ids=_unique(
            [
                *[
                    anchor_id
                    for seed in seeds
                    for anchor_id in list(seed.get('supporting_evidence_ids') or [])
                ],
                *l1_supporting_evidence_ids,
            ]
        ),
        challenging_evidence_ids=_unique(
            [
                *[
                    anchor_id
                    for seed in seeds
                    for anchor_id in list(seed.get('challenging_evidence_ids') or [])
                ],
                *l1_challenging_evidence_ids,
            ]
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
    *,
    l1_snapshot: HistoricalEnvironmentSnapshot | None = None,
) -> RouteUncertaintyPoints:
    weak_fields: list[str] = []
    if not landscape.measurement_protocols:
        weak_fields.append('measurement_protocols')
    if not landscape.toolchains_and_infrastructure:
        weak_fields.append('toolchains_and_infrastructure')
    if not landscape.alternative_routes:
        weak_fields.append('alternative_routes')
    unresolved_conflicts: list[str] = []
    if not evidence_bundle.challenging_evidence_ids:
        unresolved_conflicts.append('challenging evidence is missing at route level')
    if l1_snapshot:
        unresolved_conflicts.extend(
            f'L1 snapshot marks {entry.label} as contested'
            for entry in [
                *l1_snapshot.benchmark_timeline,
                *l1_snapshot.toolchain_timeline,
                *l1_snapshot.protocol_registry,
                *l1_snapshot.resource_registry,
            ]
            if _status_label(getattr(entry, 'status', None)) == 'contested'
        )
    open_questions = (
        list(l1_snapshot.unresolved_questions)
        if l1_snapshot
        else ['packet-level route synthesis still needs richer L1 joins for broader replay coverage']
    )
    if l1_snapshot and l1_snapshot.quality.paper_only_snapshot:
        open_questions.append('paper-grounded L1-lite may still miss external benchmark, toolchain, or protocol history')
    return RouteUncertaintyPoints(
        open_questions=_unique(open_questions),
        unresolved_conflicts=_unique(unresolved_conflicts),
        weak_fields=weak_fields,
        low_confidence_clusters=_unique(
            entry.label
            for entry in [
                *(l1_snapshot.benchmark_timeline if l1_snapshot else []),
                *(l1_snapshot.toolchain_timeline if l1_snapshot else []),
                *(l1_snapshot.protocol_registry if l1_snapshot else []),
                *(l1_snapshot.resource_registry if l1_snapshot else []),
            ]
            if getattr(entry, 'confidence', None) is not None and float(entry.confidence) < 0.65
        ),
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
    method_paper_support = _unique(
        paper_id
        for method in landscape.dominant_methods
        for paper_id in list(method.source_paper_ids or [])
    )
    method_landscape_multi_paper = len(method_paper_support) >= 2 and len(landscape.dominant_methods) >= 2
    method_grounded = method_multi_paper or method_landscape_multi_paper
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
    if has_any_method and not method_grounded:
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
            method_grounded,
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
        l1_snapshot: HistoricalEnvironmentSnapshot | dict[str, Any] | None = None,
        built_at: str | None = None,
        route_state_id: str | None = None,
    ) -> RouteState:
        packet_model = packet if isinstance(packet, RoutePacket) else RoutePacket.model_validate(packet)
        if not packet_model.included_items:
            raise ValueError('RouteStateSynthesizer requires packet included_items')
        l1_snapshot_model, effective_l1_snapshot_ref = _effective_l1_snapshot(packet_model, l1_snapshot)
        if effective_l1_snapshot_ref is not None and packet_model.l1_snapshot_ref != effective_l1_snapshot_ref:
            packet_model = packet_model.model_copy(update={'l1_snapshot_ref': effective_l1_snapshot_ref})

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
            active_benchmarks=_merge_benchmark_states(_aggregate_benchmark_states(selected_traces, seeds), l1_snapshot_model),
            measurement_protocols=_merge_protocol_states(_aggregate_protocol_states(selected_traces, seeds), l1_snapshot_model),
            toolchains_and_infrastructure=_merge_infrastructure_states(_aggregate_infrastructure_states(selected_traces, seeds), l1_snapshot_model),
            known_capabilities=_aggregate_capability_states(selected_traces, contracts, seeds),
            known_bottlenecks=_aggregate_bottleneck_states(selected_traces, seeds),
            enabling_conditions=_merge_condition_states(_aggregate_condition_states(selected_traces, seeds), l1_snapshot_model),
            alternative_routes=_aggregate_alternative_routes(selected_traces, seeds, packet_model),
        )
        readiness_scores = _build_readiness_scores(packet_model, landscape, seeds, l1_snapshot=l1_snapshot_model)
        why_now_features = _build_why_now_features(packet_model, landscape, l1_snapshot=l1_snapshot_model)
        not_now_features = _build_not_now_features(packet_model, landscape, l1_snapshot=l1_snapshot_model)
        evidence_bundle = _build_evidence_bundle(packet_model, selected_traces, seeds, l1_snapshot=l1_snapshot_model)
        uncertainty_points = _build_uncertainty_points(landscape, evidence_bundle, l1_snapshot=l1_snapshot_model)
        quality = _build_quality(packet_model, scope_resolution, landscape, evidence_bundle, why_now_features)

        built_at_value = built_at or _utc_now_iso()
        accepted_scope = scope_resolution.accepted_scope_label
        route_family_id_value = _build_route_family_id(
            accepted_scope,
            packet_model.cutoff_year,
            landscape.dominant_methods,
            preferred_method_labels=packet_model.compiler_hints.preferred_method_labels,
        )
        route_state_id_value = route_state_id or f'{_slug(accepted_scope)}_{packet_model.cutoff_year}_{_slug(packet_model.packet_id)}_{packet_model.schema_version}'

        return RouteState(
            route_state_id=route_state_id_value,
            route_family_id=route_family_id_value,
            built_at=built_at_value,
            topic_scope=accepted_scope,
            cutoff_year=packet_model.cutoff_year,
            source_packet=RouteStateSourcePacket(
                packet_id=packet_model.packet_id,
                included_trace_ids=[str(item.trace_id) for item in included_items],
                included_paper_ids=[item.paper_id for item in included_items],
                packet_role_counts=packet_model.packet_composition.role_counts,
                l1_snapshot_ref=l1_snapshot_model.snapshot_id if l1_snapshot_model else packet_model.l1_snapshot_ref.snapshot_id if packet_model.l1_snapshot_ref else None,
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
                l1_snapshot_version=l1_snapshot_model.snapshot_id if l1_snapshot_model else packet_model.l1_snapshot_ref.snapshot_id if packet_model.l1_snapshot_ref else None,
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
    l1_snapshot: HistoricalEnvironmentSnapshot | dict[str, Any] | None = None,
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
        l1_snapshot=l1_snapshot,
        built_at=built_at,
        route_state_id=route_state_id,
    )


__all__ = ['RouteStateSynthesizer', 'synthesize_route_state']
