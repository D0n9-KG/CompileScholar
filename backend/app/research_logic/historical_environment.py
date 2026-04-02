from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import Field

from app.paper_logic_trace.derived_views import build_derived_views
from app.paper_logic_trace.models import PaperLogicTrace

from .models import AuditStatus, ContractModel, L1SnapshotRef, QualityTier, RoutePacket


L1GroundingMode = Literal['paper_grounded_l1_lite']
L1EntryStatus = Literal['available', 'emerging', 'contested', 'missing', 'unknown']


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
    return slug.strip('_') or 'l1'


def _derived_views(trace: PaperLogicTrace) -> dict[str, Any]:
    if not trace.derived_views:
        trace.derived_views = build_derived_views(trace)
        return trace.derived_views
    required = {'l1_bridge_hints', 'route_compiler_contract'}
    if required.issubset(trace.derived_views):
        return trace.derived_views
    rebuilt = build_derived_views(trace)
    merged = dict(rebuilt)
    merged.update(trace.derived_views)
    trace.derived_views = merged
    return trace.derived_views


def _candidate_label(candidate: dict[str, Any]) -> str:
    normalized = str(candidate.get('normalized') or candidate.get('surface') or '').strip().lower()
    if normalized:
        return normalized
    metric_tokens = [str(token).strip().lower() for token in list(candidate.get('metric_tokens') or []) if str(token).strip()]
    if metric_tokens:
        return ', '.join(metric_tokens[:2])
    resource_tokens = [str(token).strip().lower() for token in list(candidate.get('resource_tokens') or []) if str(token).strip()]
    if resource_tokens:
        return resource_tokens[0]
    return str(candidate.get('summary') or '').strip().lower()


def _entry_status(*, support_count: int, challenge_count: int = 0, missing: bool = False) -> L1EntryStatus:
    if missing:
        return 'missing'
    if support_count <= 0:
        return 'unknown'
    if challenge_count > 0 and challenge_count >= support_count:
        return 'contested'
    if support_count >= 2:
        return 'available'
    return 'emerging'


def _entry_confidence(*, support_count: int, challenge_count: int = 0, missing: bool = False) -> float:
    if missing:
        return 0.2
    if support_count >= 3:
        base = 0.9
    elif support_count == 2:
        base = 0.78
    elif support_count == 1:
        base = 0.62
    else:
        base = 0.35
    if challenge_count > 0:
        base -= 0.15
    return round(max(0.2, min(base, 0.95)), 2)


def _challenge_blob(row: dict[str, Any]) -> str:
    parts = [
        str(row.get('normalized') or '').lower(),
        str(row.get('surface') or '').lower(),
        ' '.join(str(token).strip().lower() for token in list(row.get('resource_tokens') or []) if str(token).strip()),
        ' '.join(str(token).strip().lower() for token in list(row.get('metric_tokens') or []) if str(token).strip()),
        ' '.join(str(token).strip().lower() for token in list(row.get('comparator_tokens') or []) if str(token).strip()),
        ' '.join(str(token).strip().lower() for token in list(row.get('condition_tokens') or []) if str(token).strip()),
        ' '.join(str(token).strip().lower() for token in list(row.get('method_tokens') or []) if str(token).strip()),
    ]
    return ' '.join(part for part in parts if part).strip()


def _matching_challenge_count(label: str, challenge_rows: list[dict[str, Any]]) -> int:
    normalized = str(label or '').strip().lower()
    if not normalized:
        return 0
    paper_ids = {
        str(row.get('_paper_id') or '').strip()
        for row in challenge_rows
        if normalized in _challenge_blob(row)
    }
    paper_ids.discard('')
    return len(paper_ids)


def _normalize_text(value: str | None) -> str:
    text = str(value or '').strip().lower()
    replacements = {
        'φ': 'phi',
        'ϕ': 'phi',
        '_': ' ',
        '-': ' ',
        '/': ' ',
        '\\': ' ',
        ':': ' ',
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r'\s+', ' ', text).strip()


def _benchmark_aliases(label: str) -> list[str]:
    normalized = _normalize_text(label)
    aliases: set[str] = {normalized}
    if 'maximally random jammed' in normalized:
        aliases.update(
            {
                'maximally random jammed',
                'mrj',
                'random close packing',
                'random close packed',
                'rcp',
            }
        )
    if 'packing fraction phi c' in normalized or 'phi c' in normalized:
        aliases.update(
            {
                'packing fraction phi c',
                'phi c',
                'critical packing fraction',
                'packing fraction at jamming transition',
                'jamming transition packing fraction',
                'jamming threshold packing fraction',
                'packing fraction at the jamming transition',
                'packing fraction at the jamming onset',
            }
        )
    return sorted(alias for alias in aliases if alias)


def _register_benchmark_candidate(
    buckets: dict[str, dict[str, Any]],
    *,
    label: str,
    trace: PaperLogicTrace,
    move_id: str | None = None,
    evidence_ids: Iterable[str] | None = None,
    metric_tokens: Iterable[str] | None = None,
    comparator_tokens: Iterable[str] | None = None,
    benchmark_type: str | None = None,
) -> None:
    normalized_label = _normalize_text(label)
    if not normalized_label:
        return
    bucket = buckets.setdefault(
        normalized_label,
        {
            'paper_ids': set(),
            'trace_ids': set(),
            'move_ids': set(),
            'evidence_ids': set(),
            'metric_tokens': set(),
            'comparator_tokens': set(),
            'years': [],
            'types': set(),
        },
    )
    bucket['paper_ids'].add(trace.paper_metadata.paper_id)
    bucket['trace_ids'].add(trace.trace_id)
    if move_id:
        bucket['move_ids'].add(str(move_id).strip())
    bucket['evidence_ids'].update(str(anchor_id).strip() for anchor_id in list(evidence_ids or []) if str(anchor_id).strip())
    bucket['metric_tokens'].update(_normalize_text(token) for token in list(metric_tokens or []) if _normalize_text(token))
    bucket['comparator_tokens'].update(_normalize_text(token) for token in list(comparator_tokens or []) if _normalize_text(token))
    if benchmark_type:
        bucket['types'].add(_normalize_text(benchmark_type))
    if trace.paper_metadata.year is not None:
        bucket['years'].append(int(trace.paper_metadata.year))


def _candidate_fragments(candidate: dict[str, Any]) -> list[str]:
    return [
        _candidate_label(candidate),
        str(candidate.get('summary') or ''),
        *(str(token) for token in list(candidate.get('metric_tokens') or [])),
        *(str(token) for token in list(candidate.get('comparator_tokens') or [])),
        *(str(token) for token in list(candidate.get('resource_tokens') or [])),
        *(str(token) for token in list(candidate.get('condition_tokens') or [])),
        *(str(token) for token in list(candidate.get('method_tokens') or [])),
    ]


def _move_fragments(move: Any) -> list[str]:
    fragments = [str(getattr(move, 'summary', '') or '')]
    for field in [
        'research_objects',
        'methods',
        'observed_variables',
        'metrics',
        'comparators',
        'conditions',
        'limitation_types',
        'resource_mentions',
    ]:
        for mention in list(getattr(move, field, []) or []):
            fragments.append(str(getattr(mention, 'normalized', None) or getattr(mention, 'surface', '') or ''))
    for effect in list(getattr(move, 'effects', []) or []):
        fragments.append(str(getattr(effect, 'comparator_surface', '') or ''))
        fragments.append(str(getattr(effect, 'magnitude_text', '') or ''))
    return fragments


def _contains_alias(fragments: Iterable[str], aliases: Iterable[str]) -> bool:
    normalized_fragments = [_normalize_text(fragment) for fragment in fragments if _normalize_text(fragment)]
    alias_list = [_normalize_text(alias) for alias in aliases if _normalize_text(alias)]
    return any(alias in fragment for alias in alias_list for fragment in normalized_fragments)


def _fallback_preferred_benchmark_candidates(
    packet: RoutePacket | None,
    trace: PaperLogicTrace,
    views: dict[str, Any],
) -> list[dict[str, Any]]:
    if packet is None:
        return []

    preferred_labels = [
        str(label or '').strip()
        for label in list(packet.compiler_hints.preferred_benchmark_labels or [])
        if str(label or '').strip()
    ]
    if not preferred_labels:
        return []

    hints = dict(views.get('l1_bridge_hints') or {})
    candidate_rows = [
        *list(hints.get('protocol_candidates') or []),
        *list(hints.get('resource_candidates') or []),
        *list(hints.get('toolchain_candidates') or []),
    ]
    fallback_candidates: list[dict[str, Any]] = []
    for preferred_label in preferred_labels:
        aliases = _benchmark_aliases(preferred_label)
        seen_move_ids: set[str] = set()
        seen_evidence_ids: set[str] = set()
        metric_tokens: set[str] = set()
        comparator_tokens: set[str] = set()

        for row in candidate_rows:
            if not _contains_alias(_candidate_fragments(row), aliases):
                continue
            seen_move_ids.add(str(row.get('move_id') or '').strip())
            seen_evidence_ids.update(str(anchor_id).strip() for anchor_id in list(row.get('anchor_ids') or []) if str(anchor_id).strip())
            metric_tokens.update(_normalize_text(token) for token in list(row.get('metric_tokens') or []) if _normalize_text(token))
            comparator_tokens.update(_normalize_text(token) for token in list(row.get('comparator_tokens') or []) if _normalize_text(token))

        for move in list(trace.canonical_core.moves or []):
            if not _contains_alias(_move_fragments(move), aliases):
                continue
            seen_move_ids.add(str(move.move_id).strip())
            seen_evidence_ids.update(str(anchor_id).strip() for anchor_id in list(move.anchor_ids or []) if str(anchor_id).strip())
            metric_tokens.update(
                _normalize_text(getattr(metric, 'normalized', None) or getattr(metric, 'surface', '') or '')
                for metric in list(getattr(move, 'metrics', []) or [])
                if _normalize_text(getattr(metric, 'normalized', None) or getattr(metric, 'surface', '') or '')
            )
            comparator_tokens.update(
                _normalize_text(getattr(comparator, 'normalized', None) or getattr(comparator, 'surface', '') or '')
                for comparator in list(getattr(move, 'comparators', []) or [])
                if _normalize_text(getattr(comparator, 'normalized', None) or getattr(comparator, 'surface', '') or '')
            )

        if not seen_move_ids and not seen_evidence_ids:
            continue
        fallback_candidates.append(
            {
                'normalized': preferred_label,
                'move_id': next((move_id for move_id in sorted(seen_move_ids) if move_id), None),
                'anchor_ids': sorted(seen_evidence_ids),
                'metric_tokens': sorted(metric_tokens),
                'comparator_tokens': sorted(comparator_tokens),
                'type': 'benchmark',
            }
        )
    return fallback_candidates


class HistoricalEnvironmentEntry(ContractModel):
    label: str
    status: L1EntryStatus = 'unknown'
    confidence: float | None = None
    first_seen_year: int | None = None
    latest_seen_year: int | None = None
    source_paper_ids: list[str] = Field(default_factory=list)
    source_trace_ids: list[str] = Field(default_factory=list)
    source_move_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class ResourceRegistryEntry(HistoricalEnvironmentEntry):
    resource_types: list[str] = Field(default_factory=list)


class BenchmarkTimelineEntry(HistoricalEnvironmentEntry):
    benchmark_type: str = 'unknown'
    metric_tokens: list[str] = Field(default_factory=list)
    comparator_tokens: list[str] = Field(default_factory=list)


class ToolchainTimelineEntry(HistoricalEnvironmentEntry):
    resource_types: list[str] = Field(default_factory=list)
    method_tokens: list[str] = Field(default_factory=list)


class ProtocolRegistryEntry(HistoricalEnvironmentEntry):
    metric_tokens: list[str] = Field(default_factory=list)
    comparator_tokens: list[str] = Field(default_factory=list)
    condition_tokens: list[str] = Field(default_factory=list)
    method_tokens: list[str] = Field(default_factory=list)
    resource_tokens: list[str] = Field(default_factory=list)


class HistoricalEnvironmentQuality(ContractModel):
    quality_tier: QualityTier = 'red'
    quality_flags: list[str] = Field(default_factory=list)
    audit_status: AuditStatus = 'hot_path'
    paper_only_snapshot: bool = True
    completeness_score: float | None = None


class HistoricalEnvironmentSnapshot(ContractModel):
    snapshot_id: str
    schema_version: str = 'v1'
    built_at: str
    topic_scope: str
    cutoff_year: int
    grounding_mode: L1GroundingMode = 'paper_grounded_l1_lite'
    source_packet_id: str | None = None
    source_trace_ids: list[str] = Field(default_factory=list)
    source_paper_ids: list[str] = Field(default_factory=list)
    resource_registry: list[ResourceRegistryEntry] = Field(default_factory=list)
    benchmark_timeline: list[BenchmarkTimelineEntry] = Field(default_factory=list)
    toolchain_timeline: list[ToolchainTimelineEntry] = Field(default_factory=list)
    protocol_registry: list[ProtocolRegistryEntry] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    quality: HistoricalEnvironmentQuality = Field(default_factory=HistoricalEnvironmentQuality)


def _resource_entries(
    traces: list[PaperLogicTrace],
    derived_views: list[dict[str, Any]],
    challenge_rows: list[dict[str, Any]],
) -> list[ResourceRegistryEntry]:
    buckets: dict[str, dict[str, Any]] = {}
    for trace, views in zip(traces, derived_views):
        hints = dict(views.get('l1_bridge_hints') or {})
        for candidate in list(hints.get('resource_candidates') or []):
            label = _candidate_label(candidate)
            if not label:
                continue
            bucket = buckets.setdefault(
                label,
                {
                    'paper_ids': set(),
                    'trace_ids': set(),
                    'move_ids': set(),
                    'evidence_ids': set(),
                    'resource_types': set(),
                    'years': [],
                },
            )
            bucket['paper_ids'].add(trace.paper_metadata.paper_id)
            bucket['trace_ids'].add(trace.trace_id)
            bucket['move_ids'].add(str(candidate.get('move_id') or '').strip())
            bucket['evidence_ids'].update(str(anchor_id).strip() for anchor_id in list(candidate.get('anchor_ids') or []) if str(anchor_id).strip())
            bucket['resource_types'].add(str(candidate.get('type') or '').strip().lower())
            if trace.paper_metadata.year is not None:
                bucket['years'].append(int(trace.paper_metadata.year))

    entries: list[ResourceRegistryEntry] = []
    for label, bucket in sorted(buckets.items(), key=lambda item: (-len(item[1]['paper_ids']), item[0])):
        challenge_count = _matching_challenge_count(label, challenge_rows)
        entries.append(
            ResourceRegistryEntry(
                label=label,
                status=_entry_status(support_count=len(bucket['paper_ids']), challenge_count=challenge_count),
                confidence=_entry_confidence(support_count=len(bucket['paper_ids']), challenge_count=challenge_count),
                first_seen_year=min(bucket['years']) if bucket['years'] else None,
                latest_seen_year=max(bucket['years']) if bucket['years'] else None,
                source_paper_ids=sorted(bucket['paper_ids']),
                source_trace_ids=sorted(bucket['trace_ids']),
                source_move_ids=sorted(move_id for move_id in bucket['move_ids'] if move_id),
                evidence_ids=sorted(bucket['evidence_ids']),
                resource_types=sorted(token for token in bucket['resource_types'] if token),
            )
        )
    return entries


def _benchmark_entries(
    packet: RoutePacket | None,
    traces: list[PaperLogicTrace],
    derived_views: list[dict[str, Any]],
    challenge_rows: list[dict[str, Any]],
) -> tuple[list[BenchmarkTimelineEntry], list[str]]:
    buckets: dict[str, dict[str, Any]] = {}
    for trace, views in zip(traces, derived_views):
        hints = dict(views.get('l1_bridge_hints') or {})
        candidate_rows = [
            *list(hints.get('benchmark_candidates') or []),
            *_fallback_preferred_benchmark_candidates(packet, trace, views),
        ]
        for candidate in candidate_rows:
            label = _candidate_label(candidate)
            if not label:
                continue
            _register_benchmark_candidate(
                buckets,
                label=label,
                trace=trace,
                move_id=str(candidate.get('move_id') or '').strip() or None,
                evidence_ids=list(candidate.get('anchor_ids') or []),
                metric_tokens=list(candidate.get('metric_tokens') or []),
                comparator_tokens=list(candidate.get('comparator_tokens') or []),
                benchmark_type=str(candidate.get('type') or '').strip().lower() or 'benchmark',
            )

    quality_flags: list[str] = []
    preferred_labels = {
        _normalize_text(label)
        for label in list((packet.compiler_hints.preferred_benchmark_labels if packet else []) or [])
        if _normalize_text(label)
    }
    missing_labels = sorted(label for label in preferred_labels if label not in buckets)

    entries: list[BenchmarkTimelineEntry] = []
    for label, bucket in sorted(buckets.items(), key=lambda item: (-len(item[1]['paper_ids']), item[0])):
        challenge_count = _matching_challenge_count(label, challenge_rows)
        entries.append(
            BenchmarkTimelineEntry(
                label=label,
                status=_entry_status(support_count=len(bucket['paper_ids']), challenge_count=challenge_count),
                confidence=_entry_confidence(support_count=len(bucket['paper_ids']), challenge_count=challenge_count),
                first_seen_year=min(bucket['years']) if bucket['years'] else None,
                latest_seen_year=max(bucket['years']) if bucket['years'] else None,
                source_paper_ids=sorted(bucket['paper_ids']),
                source_trace_ids=sorted(bucket['trace_ids']),
                source_move_ids=sorted(move_id for move_id in bucket['move_ids'] if move_id),
                evidence_ids=sorted(bucket['evidence_ids']),
                benchmark_type=next((token for token in sorted(bucket['types']) if token), 'benchmark'),
                metric_tokens=sorted(bucket['metric_tokens']),
                comparator_tokens=sorted(bucket['comparator_tokens']),
            )
        )
    for label in missing_labels:
        quality_flags.append('preferred_benchmark_missing')
        entries.append(
            BenchmarkTimelineEntry(
                label=label,
                status='missing',
                confidence=_entry_confidence(support_count=0, missing=True),
                benchmark_type='benchmark',
                notes='Preferred benchmark label was not recovered from paper-derived L1 evidence.',
            )
        )
    return entries, _unique(quality_flags)


def _toolchain_entries(
    traces: list[PaperLogicTrace],
    derived_views: list[dict[str, Any]],
    challenge_rows: list[dict[str, Any]],
) -> list[ToolchainTimelineEntry]:
    buckets: dict[str, dict[str, Any]] = {}
    for trace, views in zip(traces, derived_views):
        hints = dict(views.get('l1_bridge_hints') or {})
        for candidate in list(hints.get('toolchain_candidates') or []):
            resource_types = [str(token).strip().lower() for token in list(candidate.get('resource_types') or []) if str(token).strip()]
            method_tokens = [str(token).strip().lower() for token in list(candidate.get('method_tokens') or []) if str(token).strip()]
            for token in [str(resource).strip().lower() for resource in list(candidate.get('resource_tokens') or []) if str(resource).strip()]:
                bucket = buckets.setdefault(
                    token,
                    {
                        'paper_ids': set(),
                        'trace_ids': set(),
                        'move_ids': set(),
                        'evidence_ids': set(),
                        'resource_types': set(),
                        'method_tokens': set(),
                        'years': [],
                    },
                )
                bucket['paper_ids'].add(trace.paper_metadata.paper_id)
                bucket['trace_ids'].add(trace.trace_id)
                bucket['move_ids'].add(str(candidate.get('move_id') or '').strip())
                bucket['evidence_ids'].update(str(anchor_id).strip() for anchor_id in list(candidate.get('anchor_ids') or []) if str(anchor_id).strip())
                bucket['resource_types'].update(resource_types)
                bucket['method_tokens'].update(method_tokens)
                if trace.paper_metadata.year is not None:
                    bucket['years'].append(int(trace.paper_metadata.year))

    entries: list[ToolchainTimelineEntry] = []
    for label, bucket in sorted(buckets.items(), key=lambda item: (-len(item[1]['paper_ids']), item[0])):
        challenge_count = _matching_challenge_count(label, challenge_rows)
        entries.append(
            ToolchainTimelineEntry(
                label=label,
                status=_entry_status(support_count=len(bucket['paper_ids']), challenge_count=challenge_count),
                confidence=_entry_confidence(support_count=len(bucket['paper_ids']), challenge_count=challenge_count),
                first_seen_year=min(bucket['years']) if bucket['years'] else None,
                latest_seen_year=max(bucket['years']) if bucket['years'] else None,
                source_paper_ids=sorted(bucket['paper_ids']),
                source_trace_ids=sorted(bucket['trace_ids']),
                source_move_ids=sorted(move_id for move_id in bucket['move_ids'] if move_id),
                evidence_ids=sorted(bucket['evidence_ids']),
                resource_types=sorted(bucket['resource_types']),
                method_tokens=sorted(bucket['method_tokens']),
            )
        )
    return entries


def _protocol_entries(
    traces: list[PaperLogicTrace],
    derived_views: list[dict[str, Any]],
    challenge_rows: list[dict[str, Any]],
) -> list[ProtocolRegistryEntry]:
    buckets: dict[str, dict[str, Any]] = {}
    for trace, views in zip(traces, derived_views):
        hints = dict(views.get('l1_bridge_hints') or {})
        for candidate in list(hints.get('protocol_candidates') or []):
            label = _candidate_label(candidate)
            if not label:
                continue
            bucket = buckets.setdefault(
                label,
                {
                    'paper_ids': set(),
                    'trace_ids': set(),
                    'move_ids': set(),
                    'evidence_ids': set(),
                    'metric_tokens': set(),
                    'comparator_tokens': set(),
                    'condition_tokens': set(),
                    'method_tokens': set(),
                    'resource_tokens': set(),
                    'years': [],
                },
            )
            bucket['paper_ids'].add(trace.paper_metadata.paper_id)
            bucket['trace_ids'].add(trace.trace_id)
            bucket['move_ids'].add(str(candidate.get('move_id') or '').strip())
            bucket['evidence_ids'].update(str(anchor_id).strip() for anchor_id in list(candidate.get('anchor_ids') or []) if str(anchor_id).strip())
            bucket['metric_tokens'].update(str(token).strip().lower() for token in list(candidate.get('metric_tokens') or []) if str(token).strip())
            bucket['comparator_tokens'].update(str(token).strip().lower() for token in list(candidate.get('comparator_tokens') or []) if str(token).strip())
            bucket['condition_tokens'].update(str(token).strip().lower() for token in list(candidate.get('condition_tokens') or []) if str(token).strip())
            bucket['method_tokens'].update(str(token).strip().lower() for token in list(candidate.get('method_tokens') or []) if str(token).strip())
            bucket['resource_tokens'].update(str(token).strip().lower() for token in list(candidate.get('resource_tokens') or []) if str(token).strip())
            if trace.paper_metadata.year is not None:
                bucket['years'].append(int(trace.paper_metadata.year))

    entries: list[ProtocolRegistryEntry] = []
    for label, bucket in sorted(buckets.items(), key=lambda item: (-len(item[1]['paper_ids']), item[0])):
        challenge_count = _matching_challenge_count(label, challenge_rows)
        entries.append(
            ProtocolRegistryEntry(
                label=label,
                status=_entry_status(support_count=len(bucket['paper_ids']), challenge_count=challenge_count),
                confidence=_entry_confidence(support_count=len(bucket['paper_ids']), challenge_count=challenge_count),
                first_seen_year=min(bucket['years']) if bucket['years'] else None,
                latest_seen_year=max(bucket['years']) if bucket['years'] else None,
                source_paper_ids=sorted(bucket['paper_ids']),
                source_trace_ids=sorted(bucket['trace_ids']),
                source_move_ids=sorted(move_id for move_id in bucket['move_ids'] if move_id),
                evidence_ids=sorted(bucket['evidence_ids']),
                metric_tokens=sorted(bucket['metric_tokens']),
                comparator_tokens=sorted(bucket['comparator_tokens']),
                condition_tokens=sorted(bucket['condition_tokens']),
                method_tokens=sorted(bucket['method_tokens']),
                resource_tokens=sorted(bucket['resource_tokens']),
            )
        )
    return entries


def _quality_for_snapshot(
    *,
    resource_registry: list[ResourceRegistryEntry],
    benchmark_timeline: list[BenchmarkTimelineEntry],
    toolchain_timeline: list[ToolchainTimelineEntry],
    protocol_registry: list[ProtocolRegistryEntry],
    extra_flags: list[str] | None = None,
) -> HistoricalEnvironmentQuality:
    flags = ['paper_only_snapshot', *(extra_flags or [])]
    if not resource_registry:
        flags.append('resource_registry_empty')
    if not benchmark_timeline:
        flags.append('benchmark_timeline_empty')
    if not toolchain_timeline:
        flags.append('toolchain_timeline_empty')
    if not protocol_registry:
        flags.append('protocol_registry_empty')

    present_categories = sum(bool(entries) for entries in [resource_registry, benchmark_timeline, toolchain_timeline, protocol_registry])
    multi_paper_categories = sum(
        any(entry.status == 'available' for entry in entries)
        for entries in [resource_registry, benchmark_timeline, toolchain_timeline, protocol_registry]
    )
    completeness_score = round((present_categories + multi_paper_categories) / 8.0, 2)
    quality_tier: QualityTier = 'yellow' if present_categories >= 2 else 'red'
    audit_status: AuditStatus = 'eligible' if quality_tier == 'yellow' else 'hot_path'
    return HistoricalEnvironmentQuality(
        quality_tier=quality_tier,
        quality_flags=_unique(flags),
        audit_status=audit_status,
        paper_only_snapshot=True,
        completeness_score=completeness_score,
    )


class HistoricalEnvironmentBuilder:
    def __init__(
        self,
        *,
        builder_version: str = 'historical_environment_builder_v1',
        grounding_mode: L1GroundingMode = 'paper_grounded_l1_lite',
    ) -> None:
        self.builder_version = builder_version
        self.grounding_mode = grounding_mode

    def build(
        self,
        packet: RoutePacket | dict[str, Any] | None,
        traces: list[PaperLogicTrace],
        *,
        built_at: str | None = None,
        snapshot_id: str | None = None,
    ) -> HistoricalEnvironmentSnapshot:
        packet_model = packet if isinstance(packet, RoutePacket) else RoutePacket.model_validate(packet) if packet is not None else None
        if not traces:
            raise ValueError('HistoricalEnvironmentBuilder requires at least one trace')

        if packet_model is not None:
            trace_ids = {trace.trace_id for trace in traces}
            for item in packet_model.included_items:
                if str(item.trace_id or '').strip() and str(item.trace_id) not in trace_ids:
                    raise ValueError(f'missing traces for packet item: {item.trace_id}')

        derived_views = [_derived_views(trace) for trace in traces]
        challenge_rows: list[dict[str, Any]] = []
        for trace, views in zip(traces, derived_views):
            contract = dict(views.get('route_compiler_contract') or {})
            constraint_signals = dict(contract.get('constraint_signals') or {})
            for row in list(constraint_signals.get('limitations') or []):
                enriched = dict(row)
                enriched['_paper_id'] = trace.paper_metadata.paper_id
                challenge_rows.append(enriched)

        cutoff_year = packet_model.cutoff_year if packet_model is not None else max(
            int(trace.paper_metadata.year or 0) for trace in traces
        )
        for trace in traces:
            paper_year = trace.paper_metadata.year
            if paper_year is not None and int(paper_year) > int(cutoff_year):
                raise ValueError(f'trace {trace.trace_id} exceeds L1 snapshot cutoff_year {cutoff_year}')

        topic_scope = (
            packet_model.topic_scope_candidate
            if packet_model is not None
            else next(
                (
                    str((views.get('route_state_seed') or {}).get('topic_scope_candidates', [''])[0]).strip()
                    for views in derived_views
                    if (views.get('route_state_seed') or {}).get('topic_scope_candidates')
                ),
                'historical environment snapshot',
            )
        )

        resource_registry = _resource_entries(traces, derived_views, challenge_rows)
        benchmark_timeline, quality_flags = _benchmark_entries(packet_model, traces, derived_views, challenge_rows)
        toolchain_timeline = _toolchain_entries(traces, derived_views, challenge_rows)
        protocol_registry = _protocol_entries(traces, derived_views, challenge_rows)
        unresolved_questions: list[str] = []
        if not benchmark_timeline:
            unresolved_questions.append('No explicit pre-cutoff benchmark signal was recovered from paper evidence.')
        if not protocol_registry:
            unresolved_questions.append('Measurement/evaluation protocol evidence remains thin in the current paper-only L1 snapshot.')
        if not toolchain_timeline:
            unresolved_questions.append('Toolchain and infrastructure availability are still under-described by the current paper set.')

        built_at_value = built_at or _utc_now_iso()
        snapshot_id_value = snapshot_id or f'{_slug(topic_scope)}_{cutoff_year}_l1_snapshot_v1'
        quality = _quality_for_snapshot(
            resource_registry=resource_registry,
            benchmark_timeline=benchmark_timeline,
            toolchain_timeline=toolchain_timeline,
            protocol_registry=protocol_registry,
            extra_flags=quality_flags,
        )

        return HistoricalEnvironmentSnapshot(
            snapshot_id=snapshot_id_value,
            built_at=built_at_value,
            topic_scope=topic_scope,
            cutoff_year=cutoff_year,
            grounding_mode=self.grounding_mode,
            source_packet_id=packet_model.packet_id if packet_model is not None else None,
            source_trace_ids=_unique(trace.trace_id for trace in traces),
            source_paper_ids=_unique(trace.paper_metadata.paper_id for trace in traces),
            resource_registry=resource_registry,
            benchmark_timeline=benchmark_timeline,
            toolchain_timeline=toolchain_timeline,
            protocol_registry=protocol_registry,
            unresolved_questions=unresolved_questions,
            quality=quality,
        )


def build_historical_environment_snapshot(
    packet: RoutePacket | dict[str, Any] | None,
    traces: list[PaperLogicTrace],
    *,
    built_at: str | None = None,
    snapshot_id: str | None = None,
    builder_version: str = 'historical_environment_builder_v1',
    grounding_mode: L1GroundingMode = 'paper_grounded_l1_lite',
) -> HistoricalEnvironmentSnapshot:
    builder = HistoricalEnvironmentBuilder(
        builder_version=builder_version,
        grounding_mode=grounding_mode,
    )
    return builder.build(
        packet,
        traces,
        built_at=built_at,
        snapshot_id=snapshot_id,
    )


def build_l1_snapshot_ref(snapshot: HistoricalEnvironmentSnapshot) -> L1SnapshotRef:
    base = f'l1:{snapshot.snapshot_id}'
    return L1SnapshotRef(
        snapshot_id=snapshot.snapshot_id,
        resource_registry_ref=f'{base}:resource-registry',
        resource_timeline_ref=f'{base}:resource-timeline',
        benchmark_timeline_ref=f'{base}:benchmark-timeline',
        toolchain_timeline_ref=f'{base}:toolchain-timeline',
        protocol_registry_ref=f'{base}:protocol-registry',
    )


def write_historical_environment_snapshot(
    output_path: str | Path,
    snapshot: HistoricalEnvironmentSnapshot,
) -> Path:
    path = output_path if isinstance(output_path, Path) else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot.model_dump(mode='json', exclude_none=True), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return path


__all__ = [
    'BenchmarkTimelineEntry',
    'HistoricalEnvironmentBuilder',
    'HistoricalEnvironmentQuality',
    'HistoricalEnvironmentSnapshot',
    'ProtocolRegistryEntry',
    'ResourceRegistryEntry',
    'ToolchainTimelineEntry',
    'build_historical_environment_snapshot',
    'build_l1_snapshot_ref',
    'write_historical_environment_snapshot',
]
