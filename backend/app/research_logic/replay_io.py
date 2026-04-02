from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from app.paper_logic_trace.models import PaperLogicTrace

from .historical_environment import HistoricalEnvironmentSnapshot
from .historical_replay_compiler import HistoricalReplayCompilation
from .models import RoutePacket, RouteState

if TYPE_CHECKING:
    from .prior_induction import PriorCandidateRegistry
    from .route_state_package import RouteStatePackageValidation


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _read_json(path_like: str | Path) -> Any:
    path = _as_path(path_like)
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'replay artifact not found: {path}') from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid JSON in replay artifact: {path}') from exc


def _write_json(path_like: str | Path, payload: Any) -> Path:
    path = _as_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return path


def _replay_quality_tier(compilation: HistoricalReplayCompilation) -> str:
    if any(record.blocking for record in compilation.failure_records):
        return 'red'
    if not compilation.quality_flags:
        return 'green'
    return 'yellow' if compilation.ready_for_pilot else 'red'


def _failure_count_map(compilation: HistoricalReplayCompilation, key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in compilation.failure_records:
        value = str(getattr(record, key)).strip()
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _failure_counts_by_blocking(compilation: HistoricalReplayCompilation) -> dict[str, int]:
    blocking_count = sum(1 for record in compilation.failure_records if record.blocking)
    return {
        'blocking': blocking_count,
        'nonblocking': len(compilation.failure_records) - blocking_count,
    }


def _review_status_counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        normalized = str(value or '').strip()
        if not normalized:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1
    return counts


def load_route_packet(path_like: str | Path) -> RoutePacket:
    return RoutePacket.model_validate(_read_json(path_like))


def load_paper_logic_trace(path_like: str | Path) -> PaperLogicTrace:
    return PaperLogicTrace.model_validate(_read_json(path_like))


def load_route_state(path_like: str | Path) -> RouteState:
    return RouteState.model_validate(_read_json(path_like))


def load_historical_environment_snapshot(path_like: str | Path) -> HistoricalEnvironmentSnapshot:
    return HistoricalEnvironmentSnapshot.model_validate(_read_json(path_like))


def _collect_trace_paths(
    *,
    trace_dir: str | Path | None = None,
    trace_files: Sequence[str | Path] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()

    if trace_dir is not None:
        directory = _as_path(trace_dir)
        if not directory.exists():
            raise FileNotFoundError(f'trace directory not found: {directory}')
        if not directory.is_dir():
            raise ValueError(f'trace directory is not a directory: {directory}')
        for path in sorted(directory.glob('*.json')):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)

    for trace_file in trace_files or []:
        path = _as_path(trace_file)
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(path)

    if not paths:
        raise ValueError('replay requires at least one trace source via --trace-dir or --trace-file')

    return paths


def load_paper_logic_traces(
    *,
    trace_dir: str | Path | None = None,
    trace_files: Sequence[str | Path] | None = None,
) -> list[PaperLogicTrace]:
    traces: list[PaperLogicTrace] = []
    seen_ids: set[str] = set()
    for path in _collect_trace_paths(trace_dir=trace_dir, trace_files=trace_files):
        trace = load_paper_logic_trace(path)
        if trace.trace_id in seen_ids:
            raise ValueError(f'duplicate trace_id in replay inputs: {trace.trace_id}')
        seen_ids.add(trace.trace_id)
        traces.append(trace)
    return traces


def load_route_states(paths: Sequence[str | Path] | None = None) -> list[RouteState]:
    route_states: list[RouteState] = []
    seen_ids: set[str] = set()
    for path_like in paths or []:
        route_state = load_route_state(path_like)
        if route_state.route_state_id in seen_ids:
            raise ValueError(f'duplicate route_state_id in replay inputs: {route_state.route_state_id}')
        seen_ids.add(route_state.route_state_id)
        route_states.append(route_state)
    return route_states


def ensure_packet_trace_coverage(packet: RoutePacket, traces: Sequence[PaperLogicTrace]) -> None:
    if not packet.included_items:
        raise ValueError('RoutePacket requires included_items for replay')

    missing_trace_id_papers = [
        item.paper_id
        for item in packet.included_items
        if not str(item.trace_id or '').strip()
    ]
    if missing_trace_id_papers:
        raise ValueError(
            'RoutePacket included_items missing trace_id for papers: '
            + ', '.join(sorted(dict.fromkeys(missing_trace_id_papers)))
        )

    trace_by_id = {trace.trace_id: trace for trace in traces}
    missing_trace_ids = [
        str(item.trace_id)
        for item in packet.included_items
        if str(item.trace_id) not in trace_by_id
    ]
    if missing_trace_ids:
        raise ValueError(f'missing traces for packet items: {", ".join(missing_trace_ids)}')


def build_replay_summary(
    *,
    route_packet: RoutePacket,
    traces: Sequence[PaperLogicTrace],
    compilation: HistoricalReplayCompilation,
    historical_environment_snapshot: HistoricalEnvironmentSnapshot | None = None,
    support_route_states: Sequence[RouteState] | None = None,
    alternative_route_states: Sequence[RouteState] | None = None,
    held_out_route_states: Sequence[RouteState] | None = None,
    reviewer_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        'packet_id': route_packet.packet_id,
        'topic_scope_candidate': route_packet.topic_scope_candidate,
        'cutoff_year': route_packet.cutoff_year,
        'trace_count': len(traces),
        'support_route_state_count': len(support_route_states or []),
        'alternative_route_state_count': len(alternative_route_states or []),
        'held_out_route_state_count': len(held_out_route_states or []),
        'reviewer_ids': [str(reviewer_id) for reviewer_id in reviewer_ids or [] if str(reviewer_id).strip()],
        'l1_snapshot_id': (
            historical_environment_snapshot.snapshot_id
            if historical_environment_snapshot is not None
            else compilation.primary_route_state.compiler_metadata.l1_snapshot_version
        ),
        'primary_route_state_id': compilation.primary_route_state.route_state_id,
        'selected_comparison_case_id': compilation.selected_comparison_case_id,
        'decision_prior_id': compilation.decision_prior_card.prior_id,
        'decision_episode_id': compilation.decision_episode.episode_id,
        'replay_quality_tier': _replay_quality_tier(compilation),
        'ready_for_pilot': compilation.ready_for_pilot,
        'quality_flags': list(compilation.quality_flags),
        'failure_record_count': len(compilation.failure_records),
        'failure_counts_by_layer': _failure_count_map(compilation, 'layer'),
        'failure_counts_by_stage': _failure_count_map(compilation, 'stage'),
        'failure_counts_by_blocking': _failure_counts_by_blocking(compilation),
    }


def build_replay_inspection(
    *,
    route_packet: RoutePacket,
    compilation: HistoricalReplayCompilation,
    traces: Sequence[PaperLogicTrace],
    route_state_package_validation: 'RouteStatePackageValidation | None' = None,
) -> dict[str, Any]:
    selected_comparison_case = next(
        (
            case
            for case in compilation.route_comparison_cases
            if case.route_comparison_case_id == compilation.selected_comparison_case_id
        ),
        None,
    )
    return {
        'packet_id': route_packet.packet_id,
        'topic_scope_candidate': route_packet.topic_scope_candidate,
        'cutoff_year': route_packet.cutoff_year,
        'trace_count': len(traces),
        'replay_quality_tier': _replay_quality_tier(compilation),
        'replay_quality_flags': list(compilation.quality_flags),
        'ready_for_pilot': compilation.ready_for_pilot,
        'failure_records': [record.model_dump(mode='json') for record in compilation.failure_records],
        'stage_quality': {
            'route_state': {
                'quality_tier': compilation.primary_route_state.quality.quality_tier,
                'quality_flags': list(compilation.primary_route_state.quality.quality_flags),
            },
            'why_now_case': {
                'quality_tier': compilation.why_now_case.quality.quality_tier,
                'quality_flags': list(compilation.why_now_case.quality.quality_flags),
            },
            'route_comparison': {
                'case_count': len(compilation.route_comparison_cases),
                'selected_case_id': compilation.selected_comparison_case_id,
                'selected_quality_tier': selected_comparison_case.quality.quality_tier if selected_comparison_case else None,
                'selected_quality_flags': list(selected_comparison_case.quality.quality_flags) if selected_comparison_case else [],
            },
            'decision_prior_card': {
                'quality_tier': compilation.decision_prior_card.quality.quality_tier,
                'quality_flags': list(compilation.decision_prior_card.quality.quality_flags),
                'supporting_route_state_count': len(compilation.decision_prior_card.supporting_route_state_ids),
                'held_out_pass_rate': compilation.decision_prior_card.held_out_consistency.pass_rate,
            },
            'decision_episode': {
                'quality_tier': compilation.decision_episode.quality.quality_tier,
                'quality_flags': list(compilation.decision_episode.quality.quality_flags),
                'ready_for_training': compilation.decision_episode.quality.ready_for_training,
                'ready_for_eval': compilation.decision_episode.quality.ready_for_eval,
            },
        },
        'minimal_attack_path': {
            'required_resources': list(compilation.decision_episode.minimal_attack_path.required_resources),
            'required_measurements': list(compilation.decision_episode.minimal_attack_path.required_measurements),
            'expected_checkpoints': list(compilation.decision_episode.minimal_attack_path.expected_checkpoints),
        },
        'route_state_package_validation': (
            route_state_package_validation.model_dump(mode='json', exclude_none=True)
            if route_state_package_validation is not None
            else None
        ),
    }


def build_prior_candidate_review_summary(
    *,
    registry: 'PriorCandidateRegistry',
) -> dict[str, Any]:
    return {
        'package_id': registry.package_id,
        'built_at': registry.built_at,
        'cluster_count': len(registry.clusters),
        'cluster_ids': [cluster.cluster_id for cluster in registry.clusters],
        'cluster_support_counts': {
            cluster.cluster_id: cluster.support_count for cluster in registry.clusters
        },
        'prior_candidate_count': len(registry.prior_candidates),
        'anti_pattern_candidate_count': len(registry.anti_pattern_candidates),
        'accepted_prior_ids': list(registry.accepted_prior_ids),
        'accepted_anti_pattern_ids': list(registry.accepted_anti_pattern_ids),
        'prior_review_status_counts': _review_status_counts(
            [candidate.review.review_status for candidate in registry.prior_candidates]
        ),
        'anti_pattern_review_status_counts': _review_status_counts(
            [candidate.review.review_status for candidate in registry.anti_pattern_candidates]
        ),
        'quality_flag_counts': dict(registry.quality_flag_counts),
        'prior_candidates': [
            {
                'prior_id': candidate.prior_id,
                'support_count': len(candidate.supporting_route_state_ids),
                'held_out_pass_rate': candidate.held_out_consistency.pass_rate,
                'review_status': candidate.review.review_status,
                'quality_tier': candidate.quality.quality_tier,
                'accepted': candidate.prior_id in registry.accepted_prior_ids,
            }
            for candidate in registry.prior_candidates
        ],
        'anti_pattern_candidates': [
            {
                'anti_pattern_id': candidate.anti_pattern_id,
                'failure_route_state_count': len(candidate.failure_examples.route_state_ids),
                'review_status': candidate.review.review_status,
                'quality_tier': candidate.quality.quality_tier,
                'accepted': candidate.anti_pattern_id in registry.accepted_anti_pattern_ids,
            }
            for candidate in registry.anti_pattern_candidates
        ],
    }


def _model_payload(model: Any) -> Any:
    if hasattr(model, 'model_dump'):
        return model.model_dump(mode='json', exclude_none=True)
    return model


def write_replay_bundle(
    output_dir: str | Path,
    *,
    route_packet: RoutePacket,
    traces: Sequence[PaperLogicTrace],
    compilation: HistoricalReplayCompilation,
    historical_environment_snapshot: HistoricalEnvironmentSnapshot | None = None,
    support_route_states: Sequence[RouteState] | None = None,
    alternative_route_states: Sequence[RouteState] | None = None,
    held_out_route_states: Sequence[RouteState] | None = None,
    reviewer_ids: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
    route_state_package_validation: 'RouteStatePackageValidation | None' = None,
) -> dict[str, Path]:
    bundle_dir = _as_path(output_dir)
    inputs_dir = bundle_dir / 'inputs'
    outputs_dir = bundle_dir / 'outputs'
    built_at = _utc_now_iso()

    summary_payload = build_replay_summary(
        route_packet=route_packet,
        traces=traces,
        compilation=compilation,
        historical_environment_snapshot=historical_environment_snapshot,
        support_route_states=support_route_states,
        alternative_route_states=alternative_route_states,
        held_out_route_states=held_out_route_states,
        reviewer_ids=reviewer_ids,
    )
    inspection_payload = build_replay_inspection(
        route_packet=route_packet,
        compilation=compilation,
        traces=traces,
        route_state_package_validation=route_state_package_validation,
    )

    written_files = {
        'route_packet': _write_json(inputs_dir / 'route_packet.json', _model_payload(route_packet)),
        'paper_logic_traces': _write_json(
            inputs_dir / 'paper_logic_traces.json',
            [_model_payload(trace) for trace in traces],
        ),
        'support_route_states': _write_json(
            inputs_dir / 'support_route_states.json',
            [_model_payload(route_state) for route_state in support_route_states or []],
        ),
        'alternative_route_states': _write_json(
            inputs_dir / 'alternative_route_states.json',
            [_model_payload(route_state) for route_state in alternative_route_states or []],
        ),
        'held_out_route_states': _write_json(
            inputs_dir / 'held_out_route_states.json',
            [_model_payload(route_state) for route_state in held_out_route_states or []],
        ),
        'primary_route_state': _write_json(outputs_dir / 'primary_route_state.json', _model_payload(compilation.primary_route_state)),
        'why_now_case': _write_json(outputs_dir / 'why_now_case.json', _model_payload(compilation.why_now_case)),
        'route_comparison_cases': _write_json(
            outputs_dir / 'route_comparison_cases.json',
            [_model_payload(case) for case in compilation.route_comparison_cases],
        ),
        'decision_prior_card': _write_json(outputs_dir / 'decision_prior_card.json', _model_payload(compilation.decision_prior_card)),
        'decision_episode': _write_json(outputs_dir / 'decision_episode.json', _model_payload(compilation.decision_episode)),
        'replay_summary': _write_json(bundle_dir / 'replay_summary.json', summary_payload),
        'replay_inspection': _write_json(bundle_dir / 'replay_inspection.json', inspection_payload),
    }
    if historical_environment_snapshot is not None:
        written_files['historical_environment_snapshot'] = _write_json(
            inputs_dir / 'historical_environment_snapshot.json',
            _model_payload(historical_environment_snapshot),
        )

    manifest_payload = {
        'schema_version': 'v1',
        'built_at': built_at,
        'packet_id': route_packet.packet_id,
        'cutoff_year': route_packet.cutoff_year,
        'ready_for_pilot': compilation.ready_for_pilot,
        'quality_flags': list(compilation.quality_flags),
        'metadata': dict(metadata or {}),
        'files': {
            name: str(path.relative_to(bundle_dir)).replace('\\', '/')
            for name, path in written_files.items()
        },
    }
    written_files['bundle_manifest'] = _write_json(bundle_dir / 'bundle_manifest.json', manifest_payload)
    return written_files


def write_prior_candidate_review_bundle(
    output_dir: str | Path,
    *,
    registry: 'PriorCandidateRegistry',
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    bundle_dir = _as_path(output_dir)
    summary_payload = build_prior_candidate_review_summary(registry=registry)
    written_files = {
        'prior_candidates': _write_json(
            bundle_dir / 'prior_candidates.json',
            [candidate.model_dump(mode='json', exclude_none=True) for candidate in registry.prior_candidates],
        ),
        'anti_pattern_candidates': _write_json(
            bundle_dir / 'anti_pattern_candidates.json',
            [candidate.model_dump(mode='json', exclude_none=True) for candidate in registry.anti_pattern_candidates],
        ),
        'clusters': _write_json(
            bundle_dir / 'clusters.json',
            [cluster.model_dump(mode='json', exclude_none=True) for cluster in registry.clusters],
        ),
        'candidate_review_summary': _write_json(
            bundle_dir / 'candidate_review_summary.json',
            summary_payload,
        ),
    }
    written_files['bundle_manifest'] = _write_json(
        bundle_dir / 'bundle_manifest.json',
        {
            'schema_version': 'v1',
            'built_at': registry.built_at,
            'package_id': registry.package_id,
            'accepted_prior_ids': list(registry.accepted_prior_ids),
            'accepted_anti_pattern_ids': list(registry.accepted_anti_pattern_ids),
            'metadata': dict(metadata or {}),
            'files': {
                name: str(path.relative_to(bundle_dir)).replace('\\', '/')
                for name, path in written_files.items()
            },
        },
    )
    return written_files


__all__ = [
    'build_prior_candidate_review_summary',
    'build_replay_summary',
    'build_replay_inspection',
    'ensure_packet_trace_coverage',
    'load_historical_environment_snapshot',
    'load_paper_logic_trace',
    'load_paper_logic_traces',
    'load_route_packet',
    'load_route_state',
    'load_route_states',
    'write_prior_candidate_review_bundle',
    'write_replay_bundle',
]
