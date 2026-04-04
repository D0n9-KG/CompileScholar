from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from .bounded_packet_audit import (
    BoundedPacketAssemblyManifest,
    audit_bounded_packet_assembly,
    load_bounded_packet_assembly_manifest,
)
from .decision_episode_export import build_decision_episode_audit_export
from .historical_environment import HistoricalEnvironmentSnapshot, build_historical_environment_snapshot, build_l1_snapshot_ref, write_historical_environment_snapshot
from .historical_replay_compiler import compile_historical_replay
from .models import AntiPatternCard, DecisionEpisode, DecisionPriorCard, RouteComparisonCase, RoutePacket, RouteState, WhyNowCase
from .replay_io import (
    build_replay_summary,
    ensure_packet_trace_coverage,
    load_paper_logic_trace,
    load_paper_logic_traces,
    load_route_packet,
    write_decision_episode_export_bundle,
    write_prior_candidate_review_bundle,
    write_replay_bundle,
)
from .route_state_package import (
    RouteStatePackageEntry,
    RouteStatePackageManifest,
    build_route_state_package_summary,
    compile_route_state_package,
    load_route_state_package_bundle,
    write_route_state_package_bundle,
)
from .prior_induction import build_prior_candidate_registry_from_package

Phase10Role = Literal['support', 'alternative', 'held_out']
ModelT = TypeVar('ModelT', bound=BaseModel)

DEFAULT_PHASE10_PACKET_PATH = Path('docs/replay/pilot_packets/phase9-route-packet.json')
DEFAULT_PHASE10_ASSEMBLY_MANIFEST_PATH = Path('docs/replay/pilot_packets/phase9-assembly-manifest.json')
DEFAULT_PHASE10_OUTPUT_ROOT = Path('tmp/phase10_multi_paper_validation')
DEFAULT_PHASE10_PACKAGE_MANIFEST_PATH = DEFAULT_PHASE10_OUTPUT_ROOT / 'generated' / 'phase10-route-state-package-manifest.json'
DEFAULT_PHASE10_L1_SNAPSHOT_PATH = DEFAULT_PHASE10_OUTPUT_ROOT / 'shared' / 'phase9-comp-mech-l1-snapshot.json'
DEFAULT_PHASE10_BASELINE_REPLAY_BUNDLE = Path('tmp/phase3_route_state_package/replay_with_package')
DEFAULT_PHASE10_BASELINE_EXPORT_BUNDLE = Path('tmp/phase6_decision_episode_audit_export')


@dataclass(frozen=True)
class Phase10CanonicalInputs:
    repo_root: Path
    packet_path: Path
    assembly_manifest_path: Path
    route_packet: RoutePacket
    assembly_manifest: BoundedPacketAssemblyManifest


@dataclass(frozen=True)
class Phase10TraceSource:
    role: Phase10Role
    paper_id: str
    trace_id: str
    runtime_paper_id: str
    trace_ref: str
    resolved_path: Path


@dataclass(frozen=True)
class Phase10RuntimePaths:
    repo_root: Path
    output_root: Path
    generated_dir: Path
    generated_packets_dir: Path
    package_manifest_path: Path
    shared_dir: Path
    l1_snapshot_path: Path


@dataclass(frozen=True)
class Phase10RolePacketArtifact:
    entry_id: str
    role: Phase10Role
    packet_path: Path
    route_state_id: str
    paper_ids: tuple[str, ...]
    trace_files: tuple[Path, ...]
    notes: str | None = None
    distinctness_rationale: str | None = None


@dataclass(frozen=True)
class Phase10RuntimeRolePacketSpec:
    entry_id: str
    role: Phase10Role
    route_state_id: str
    paper_ids: tuple[str, ...]
    notes: str | None = None
    distinctness_rationale: str | None = None


@dataclass(frozen=True)
class Phase10RuntimeBridge:
    canonical_inputs: Phase10CanonicalInputs
    runtime_paths: Phase10RuntimePaths
    trace_sources: tuple[Phase10TraceSource, ...]
    l1_snapshot: HistoricalEnvironmentSnapshot
    l1_snapshot_path: Path
    package_manifest: RouteStatePackageManifest
    package_manifest_path: Path
    role_packets: dict[Phase10Role, tuple[Phase10RolePacketArtifact, ...]]


@dataclass(frozen=True)
class Phase10ValidationRun:
    bridge: Phase10RuntimeBridge
    route_state_package_output_dir: Path
    route_state_package_bundle_files: dict[str, Path]
    replay_output_dir: Path
    replay_bundle_files: dict[str, Path]
    prior_review_output_dir: Path
    prior_review_bundle_files: dict[str, Path]
    export_output_dir: Path
    export_bundle_files: dict[str, Path]
    comparison_summary_path: Path
    report_markdown_path: Path | None
    summary: dict[str, object]


def _repo_root(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    return root.resolve()


def _resolve_repo_path(path_like: str | Path, *, repo_root: Path) -> Path:
    path = path_like if isinstance(path_like, Path) else Path(path_like)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'Phase 10 artifact not found: {path}') from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid JSON in Phase 10 artifact: {path}') from exc


def _relative_manifest_path(target: Path, *, base_dir: Path) -> str:
    try:
        return os.path.relpath(target, base_dir).replace('\\', '/')
    except ValueError:
        return str(target)


def _role_slug(role: Phase10Role) -> str:
    return role.replace('_', '-')


def _phase10_runtime_role_packets(
    canonical_inputs: Phase10CanonicalInputs,
) -> tuple[Phase10RuntimeRolePacketSpec, ...]:
    role_groups = canonical_inputs.assembly_manifest.role_groups()
    runtime_packets = (
        Phase10RuntimeRolePacketSpec(
            entry_id='phase10-support-core',
            role='support',
            route_state_id='phase10-support-core-route-state',
            paper_ids=('1000', '1001'),
            notes='Runtime support-core subgroup for the constitutive-route anchors from the committed Phase 9 support role.',
        ),
        Phase10RuntimeRolePacketSpec(
            entry_id='phase10-support-context',
            role='support',
            route_state_id='phase10-support-context-route-state',
            paper_ids=('1002', '1005'),
            notes='Runtime support-context subgroup for the Phase 9 support papers that carry data-conditioning and boundary-framing context.',
        ),
        Phase10RuntimeRolePacketSpec(
            entry_id='phase10-support-clustering',
            role='support',
            route_state_id='phase10-support-clustering-route-state',
            paper_ids=('1017',),
            notes='Runtime support-clustering subgroup for the clustering-analysis anchor inside the committed Phase 9 support role.',
        ),
        Phase10RuntimeRolePacketSpec(
            entry_id='phase10-alternative',
            role='alternative',
            route_state_id='phase10-alternative-route-state',
            paper_ids=('1007',),
            notes=role_groups['alternative'].group_reason,
            distinctness_rationale=role_groups['alternative'].distinctness_rationale,
        ),
        Phase10RuntimeRolePacketSpec(
            entry_id='phase10-held-out',
            role='held_out',
            route_state_id='phase10-held-out-route-state',
            paper_ids=('1023',),
            notes=role_groups['held_out'].group_reason,
        ),
    )

    expected_by_role = {
        role: sorted(group.paper_ids())
        for role, group in role_groups.items()
    }
    actual_by_role = {
        role: sorted(
            paper_id
            for artifact in runtime_packets
            if artifact.role == role
            for paper_id in artifact.paper_ids
        )
        for role in expected_by_role
    }
    for role, expected_paper_ids in expected_by_role.items():
        if actual_by_role[role] != expected_paper_ids:
            raise ValueError(
                f'Phase 10 runtime bridge paper_ids for {role} must preserve the committed assembly manifest membership'
            )

    return runtime_packets


def _package_built_at(route_packet: RoutePacket, assembly_manifest: BoundedPacketAssemblyManifest, built_at: str | None) -> str:
    return built_at or assembly_manifest.built_at or route_packet.built_at


def _load_model(path: Path, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(_load_json(path))


def _load_models(path: Path, model_type: type[ModelT]) -> list[ModelT]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f'Phase 10 artifact must decode to a JSON list: {path}')
    return [model_type.model_validate(item) for item in payload]


def _json_object(payload: Any, *, label: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f'{label} must decode to a JSON object')
    return payload


def _bundle_string_list(manifest_payload: dict[str, object], *, key: str, label: str) -> list[str]:
    value = manifest_payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f'{label} manifest missing {key} list')
    return [str(item).strip() for item in value if str(item).strip()]


def _bundle_source_refs(bundle_manifest_path: Path, manifest_payload: dict[str, object]) -> dict[str, object]:
    files = manifest_payload.get('files')
    if not isinstance(files, dict):
        raise ValueError(f'Phase 10 bundle manifest missing files map: {bundle_manifest_path}')

    bundle_dir = bundle_manifest_path.parent.resolve()
    return {
        'bundle_ref': str(bundle_dir),
        'manifest_ref': str(bundle_manifest_path.resolve()),
        'files': {
            key: str((bundle_dir / relative_path).resolve())
            for key, relative_path in files.items()
            if isinstance(key, str) and isinstance(relative_path, str) and str(relative_path).strip()
        },
    }


def _string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or '').strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _int_map(values: object) -> dict[str, int]:
    if not isinstance(values, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in values.items():
        try:
            normalized[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _count_delta(current: object, baseline: object) -> int | None:
    try:
        return int(current) - int(baseline)
    except (TypeError, ValueError):
        return None


def _count_delta_map(current: object, baseline: object) -> dict[str, int]:
    current_map = _int_map(current)
    baseline_map = _int_map(baseline)
    return {
        key: current_map.get(key, 0) - baseline_map.get(key, 0)
        for key in sorted(set(current_map) | set(baseline_map))
    }


def _quality_trend(current: object, baseline: object) -> str:
    ranks = {'red': 0, 'yellow': 1, 'green': 2}
    current_rank = ranks.get(str(current or '').strip(), -1)
    baseline_rank = ranks.get(str(baseline or '').strip(), -1)
    if current_rank > baseline_rank:
        return 'improved'
    if current_rank < baseline_rank:
        return 'regressed'
    return 'unchanged'


def _flag_delta(current: object, baseline: object) -> dict[str, list[str]]:
    current_flags = set(_string_list(current))
    baseline_flags = set(_string_list(baseline))
    return {
        'added': sorted(current_flags - baseline_flags),
        'removed': sorted(baseline_flags - current_flags),
    }


def _count_trend(current: int, baseline: int) -> str:
    if current > baseline:
        return 'higher'
    if current < baseline:
        return 'lower'
    return 'unchanged'


def _resolve_bundle_dir(path_like: str | Path, *, repo_root: Path) -> Path:
    path = _resolve_repo_path(path_like, repo_root=repo_root)
    return path.parent if path.name == 'bundle_manifest.json' else path


def _comparison_stage_summary(
    *,
    quality_tier: object,
    baseline_quality_tier: object,
    quality_flags: object,
    baseline_quality_flags: object,
    extra_delta: dict[str, object] | None = None,
) -> dict[str, object]:
    delta: dict[str, object] = {
        'quality_tier_trend': _quality_trend(quality_tier, baseline_quality_tier),
        'quality_flag_delta': _flag_delta(quality_flags, baseline_quality_flags),
    }
    if extra_delta:
        delta.update(extra_delta)
    return delta


def _blocker_entry(
    *,
    code: str,
    message: str,
    current: object,
    baseline: object | None = None,
    vs_baseline: str | None = None,
) -> dict[str, object]:
    entry: dict[str, object] = {
        'code': code,
        'message': message,
        'current': current,
    }
    if baseline is not None:
        entry['baseline'] = baseline
    if vs_baseline is not None:
        entry['vs_baseline'] = vs_baseline
    return entry


def build_phase10_comparison_summary(
    *,
    route_state_package_bundle: str | Path,
    replay_bundle: str | Path,
    prior_review_bundle: str | Path,
    export_bundle: str | Path,
    baseline_replay_bundle: str | Path = DEFAULT_PHASE10_BASELINE_REPLAY_BUNDLE,
    baseline_export_bundle: str | Path = DEFAULT_PHASE10_BASELINE_EXPORT_BUNDLE,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    resolved_repo_root = _repo_root(repo_root)
    route_state_package_dir = _resolve_bundle_dir(route_state_package_bundle, repo_root=resolved_repo_root)
    replay_bundle_dir = _resolve_bundle_dir(replay_bundle, repo_root=resolved_repo_root)
    prior_review_bundle_dir = _resolve_bundle_dir(prior_review_bundle, repo_root=resolved_repo_root)
    export_bundle_dir = _resolve_bundle_dir(export_bundle, repo_root=resolved_repo_root)
    baseline_replay_bundle_dir = _resolve_bundle_dir(baseline_replay_bundle, repo_root=resolved_repo_root)
    baseline_export_bundle_dir = _resolve_bundle_dir(baseline_export_bundle, repo_root=resolved_repo_root)

    package_validation = _json_object(
        _load_json(route_state_package_dir / 'validation.json'),
        label='Phase 10 route-state package validation',
    )
    replay_summary = _json_object(
        _load_json(replay_bundle_dir / 'replay_summary.json'),
        label='Phase 10 replay summary',
    )
    replay_inspection = _json_object(
        _load_json(replay_bundle_dir / 'replay_inspection.json'),
        label='Phase 10 replay inspection',
    )
    prior_review_summary = _json_object(
        _load_json(prior_review_bundle_dir / 'candidate_review_summary.json'),
        label='Phase 10 prior review summary',
    )
    prior_review_manifest = _json_object(
        _load_json(prior_review_bundle_dir / 'bundle_manifest.json'),
        label='Phase 10 prior review bundle manifest',
    )
    export_summary = _json_object(
        _load_json(export_bundle_dir / 'export_summary.json'),
        label='Phase 10 export summary',
    )
    export_inspection = _json_object(
        _load_json(export_bundle_dir / 'export_inspection.json'),
        label='Phase 10 export inspection',
    )
    export_manifest = _json_object(
        _load_json(export_bundle_dir / 'bundle_manifest.json'),
        label='Phase 10 export bundle manifest',
    )
    training_view = _json_object(
        _load_json(export_bundle_dir / 'outputs' / 'training_view.json'),
        label='Phase 10 training view',
    )
    best_cycle_selection = _json_object(
        _load_json(export_bundle_dir / 'best_cycle_selection.json'),
        label='Phase 10 best-cycle selection',
    )
    baseline_replay_summary = _json_object(
        _load_json(baseline_replay_bundle_dir / 'replay_summary.json'),
        label='Baseline replay summary',
    )
    baseline_replay_inspection = _json_object(
        _load_json(baseline_replay_bundle_dir / 'replay_inspection.json'),
        label='Baseline replay inspection',
    )
    baseline_export_summary = _json_object(
        _load_json(baseline_export_bundle_dir / 'export_summary.json'),
        label='Baseline export summary',
    )
    baseline_export_inspection = _json_object(
        _load_json(baseline_export_bundle_dir / 'export_inspection.json'),
        label='Baseline export inspection',
    )

    baseline_package_validation = _json_object(
        baseline_replay_inspection.get('route_state_package_validation') or {},
        label='Baseline route-state package validation',
    )

    package = {
        'current': {
            'quality_tier': package_validation.get('quality_tier'),
            'ready_for_replay': package_validation.get('ready_for_replay'),
            'quality_flags': _string_list(package_validation.get('quality_flags')),
            'role_counts': _int_map(package_validation.get('role_counts')),
            'bundle_manifest': str((route_state_package_dir / 'bundle_manifest.json').resolve()),
        },
        'baseline': {
            'quality_tier': baseline_package_validation.get('quality_tier'),
            'ready_for_replay': baseline_package_validation.get('ready_for_replay'),
            'quality_flags': _string_list(baseline_package_validation.get('quality_flags')),
            'role_counts': _int_map(
                baseline_package_validation.get('role_counts')
                or {
                    'support': baseline_replay_summary.get('support_route_state_count', 0),
                    'alternative': baseline_replay_summary.get('alternative_route_state_count', 0),
                    'held_out': baseline_replay_summary.get('held_out_route_state_count', 0),
                }
            ),
            'bundle_manifest': str((baseline_replay_bundle_dir / 'bundle_manifest.json').resolve()),
        },
    }
    package['delta'] = _comparison_stage_summary(
        quality_tier=package['current']['quality_tier'],
        baseline_quality_tier=package['baseline']['quality_tier'],
        quality_flags=package['current']['quality_flags'],
        baseline_quality_flags=package['baseline']['quality_flags'],
        extra_delta={
            'ready_for_replay_changed': package['current']['ready_for_replay'] != package['baseline']['ready_for_replay'],
            'role_count_delta': _count_delta_map(package['current']['role_counts'], package['baseline']['role_counts']),
        },
    )

    replay = {
        'current': {
            'quality_tier': replay_summary.get('replay_quality_tier'),
            'ready_for_pilot': replay_summary.get('ready_for_pilot'),
            'quality_flags': _string_list(replay_summary.get('quality_flags')),
            'failure_record_count': int(replay_summary.get('failure_record_count') or 0),
            'failure_counts_by_stage': _int_map(replay_summary.get('failure_counts_by_stage')),
            'failure_counts_by_layer': _int_map(replay_summary.get('failure_counts_by_layer')),
            'failure_counts_by_blocking': _int_map(replay_summary.get('failure_counts_by_blocking')),
            'bundle_manifest': str((replay_bundle_dir / 'bundle_manifest.json').resolve()),
            'selected_comparison_case_id': replay_summary.get('selected_comparison_case_id'),
        },
        'baseline': {
            'quality_tier': baseline_replay_summary.get('replay_quality_tier'),
            'ready_for_pilot': baseline_replay_summary.get('ready_for_pilot'),
            'quality_flags': _string_list(baseline_replay_summary.get('quality_flags')),
            'failure_record_count': int(baseline_replay_summary.get('failure_record_count') or 0),
            'failure_counts_by_stage': _int_map(baseline_replay_summary.get('failure_counts_by_stage')),
            'failure_counts_by_layer': _int_map(baseline_replay_summary.get('failure_counts_by_layer')),
            'failure_counts_by_blocking': _int_map(baseline_replay_summary.get('failure_counts_by_blocking')),
            'bundle_manifest': str((baseline_replay_bundle_dir / 'bundle_manifest.json').resolve()),
            'selected_comparison_case_id': baseline_replay_summary.get('selected_comparison_case_id'),
        },
    }
    replay['delta'] = _comparison_stage_summary(
        quality_tier=replay['current']['quality_tier'],
        baseline_quality_tier=replay['baseline']['quality_tier'],
        quality_flags=replay['current']['quality_flags'],
        baseline_quality_flags=replay['baseline']['quality_flags'],
        extra_delta={
            'ready_for_pilot_changed': replay['current']['ready_for_pilot'] != replay['baseline']['ready_for_pilot'],
            'failure_record_count_delta': _count_delta(
                replay['current']['failure_record_count'],
                replay['baseline']['failure_record_count'],
            ),
            'failure_counts_by_stage_delta': _count_delta_map(
                replay['current']['failure_counts_by_stage'],
                replay['baseline']['failure_counts_by_stage'],
            ),
            'failure_counts_by_layer_delta': _count_delta_map(
                replay['current']['failure_counts_by_layer'],
                replay['baseline']['failure_counts_by_layer'],
            ),
        },
    )

    prior_review = {
        'current': {
            'cluster_strategy': str(prior_review_summary.get('cluster_strategy') or 'default'),
            'cluster_count': int(prior_review_summary.get('cluster_count') or 0),
            'fallback_reason': prior_review_summary.get('fallback_reason'),
            'prior_candidate_count': int(prior_review_summary.get('prior_candidate_count') or 0),
            'anti_pattern_candidate_count': int(prior_review_summary.get('anti_pattern_candidate_count') or 0),
            'accepted_prior_ids': _bundle_string_list(
                prior_review_manifest,
                key='accepted_prior_ids',
                label='Phase 10 prior review bundle',
            ),
            'accepted_anti_pattern_ids': _bundle_string_list(
                prior_review_manifest,
                key='accepted_anti_pattern_ids',
                label='Phase 10 prior review bundle',
            ),
            'quality_flag_counts': _int_map(prior_review_summary.get('quality_flag_counts')),
            'bundle_manifest': str((prior_review_bundle_dir / 'bundle_manifest.json').resolve()),
            'prior_review_summary_path': str((prior_review_bundle_dir / 'candidate_review_summary.json').resolve()),
        },
        'baseline': {
            'accepted_prior_count': int(baseline_export_summary.get('accepted_prior_count') or 0),
            'accepted_anti_pattern_count': int(baseline_export_summary.get('accepted_anti_pattern_count') or 0),
            'selected_prior_count': int(baseline_export_summary.get('selected_prior_count') or 0),
            'selected_antipattern_count': int(baseline_export_summary.get('selected_antipattern_count') or 0),
            'source': str((baseline_export_bundle_dir / 'export_summary.json').resolve()),
        },
    }
    prior_review['delta'] = {
        'accepted_prior_count_delta': _count_delta(
            len(prior_review['current']['accepted_prior_ids']),
            prior_review['baseline']['accepted_prior_count'],
        ),
        'accepted_anti_pattern_count_delta': _count_delta(
            len(prior_review['current']['accepted_anti_pattern_ids']),
            prior_review['baseline']['accepted_anti_pattern_count'],
        ),
        'prior_candidate_count_vs_baseline_selected_prior_count': _count_delta(
            prior_review['current']['prior_candidate_count'],
            prior_review['baseline']['selected_prior_count'],
        ),
        'anti_pattern_candidate_count_vs_baseline_selected_antipattern_count': _count_delta(
            prior_review['current']['anti_pattern_candidate_count'],
            prior_review['baseline']['selected_antipattern_count'],
        ),
    }

    export = {
        'current': {
            'quality_tier': export_summary.get('quality_tier'),
            'ready_for_training': export_summary.get('ready_for_training'),
            'ready_for_eval': export_summary.get('ready_for_eval'),
            'quality_flags': _string_list(export_summary.get('quality_flags')),
            'accepted_prior_count': int(export_summary.get('accepted_prior_count') or 0),
            'accepted_anti_pattern_count': int(export_summary.get('accepted_anti_pattern_count') or 0),
            'selected_prior_count': int(export_summary.get('selected_prior_count') or 0),
            'selected_antipattern_count': int(export_summary.get('selected_antipattern_count') or 0),
            'accepted_but_unselected_prior_count': int(export_summary.get('accepted_but_unselected_prior_count') or 0),
            'accepted_but_unselected_antipattern_count': int(
                export_summary.get('accepted_but_unselected_antipattern_count') or 0
            ),
            'visibility_bucket_counts': _int_map(export_summary.get('visibility_bucket_counts')),
            'review_status': export_summary.get('review_status'),
            'training_acceptance_verdict': export_summary.get('training_acceptance_verdict'),
            'selected_iteration_label': best_cycle_selection.get('selected_iteration_label'),
            'primary_recommendation_id': best_cycle_selection.get('primary_recommendation_id'),
            'recommendation_evidence_refs': _string_list(best_cycle_selection.get('recommendation_evidence_refs')),
            'training_view_sections': sorted(
                str(key)
                for key in _json_object(training_view.get('sections') or {}, label='Phase 10 training view sections')
            ),
            'bundle_manifest': str((export_bundle_dir / 'bundle_manifest.json').resolve()),
            'training_view': str((export_bundle_dir / 'outputs' / 'training_view.json').resolve()),
            'best_cycle_selection': str((export_bundle_dir / 'best_cycle_selection.json').resolve()),
        },
        'baseline': {
            'quality_tier': baseline_export_summary.get('quality_tier'),
            'ready_for_training': baseline_export_summary.get('ready_for_training'),
            'ready_for_eval': baseline_export_summary.get('ready_for_eval'),
            'quality_flags': _string_list(baseline_export_summary.get('quality_flags')),
            'accepted_prior_count': int(baseline_export_summary.get('accepted_prior_count') or 0),
            'accepted_anti_pattern_count': int(baseline_export_summary.get('accepted_anti_pattern_count') or 0),
            'selected_prior_count': int(baseline_export_summary.get('selected_prior_count') or 0),
            'selected_antipattern_count': int(baseline_export_summary.get('selected_antipattern_count') or 0),
            'accepted_but_unselected_prior_count': int(baseline_export_summary.get('accepted_but_unselected_prior_count') or 0),
            'accepted_but_unselected_antipattern_count': int(
                baseline_export_summary.get('accepted_but_unselected_antipattern_count') or 0
            ),
            'visibility_bucket_counts': _int_map(baseline_export_summary.get('visibility_bucket_counts')),
            'bundle_manifest': str((baseline_export_bundle_dir / 'bundle_manifest.json').resolve()),
        },
    }
    export['delta'] = _comparison_stage_summary(
        quality_tier=export['current']['quality_tier'],
        baseline_quality_tier=export['baseline']['quality_tier'],
        quality_flags=export['current']['quality_flags'],
        baseline_quality_flags=export['baseline']['quality_flags'],
        extra_delta={
            'ready_for_training_changed': export['current']['ready_for_training'] != export['baseline']['ready_for_training'],
            'ready_for_eval_changed': export['current']['ready_for_eval'] != export['baseline']['ready_for_eval'],
            'accepted_prior_count_delta': _count_delta(
                export['current']['accepted_prior_count'],
                export['baseline']['accepted_prior_count'],
            ),
            'accepted_anti_pattern_count_delta': _count_delta(
                export['current']['accepted_anti_pattern_count'],
                export['baseline']['accepted_anti_pattern_count'],
            ),
            'selected_prior_count_delta': _count_delta(
                export['current']['selected_prior_count'],
                export['baseline']['selected_prior_count'],
            ),
            'selected_antipattern_count_delta': _count_delta(
                export['current']['selected_antipattern_count'],
                export['baseline']['selected_antipattern_count'],
            ),
            'accepted_but_unselected_prior_count_delta': _count_delta(
                export['current']['accepted_but_unselected_prior_count'],
                export['baseline']['accepted_but_unselected_prior_count'],
            ),
            'accepted_but_unselected_antipattern_count_delta': _count_delta(
                export['current']['accepted_but_unselected_antipattern_count'],
                export['baseline']['accepted_but_unselected_antipattern_count'],
            ),
            'visibility_bucket_count_delta': _count_delta_map(
                export['current']['visibility_bucket_counts'],
                export['baseline']['visibility_bucket_counts'],
            ),
        },
    )

    package_blockers = [
        _blocker_entry(
            code=flag,
            message=f'Package validation reports `{flag}`.',
            current=True,
            baseline=flag in set(package['baseline']['quality_flags']),
            vs_baseline='new' if flag not in set(package['baseline']['quality_flags']) else 'carried_forward',
        )
        for flag in package['current']['quality_flags']
    ]
    if not package['current']['ready_for_replay'] and not package_blockers:
        package_blockers.append(
            _blocker_entry(
                code='ready_for_replay_false',
                message='Package validation did not mark the bundle ready for replay.',
                current=package['current']['ready_for_replay'],
                baseline=package['baseline']['ready_for_replay'],
                vs_baseline='regressed'
                if package['baseline']['ready_for_replay'] and not package['current']['ready_for_replay']
                else 'unchanged',
            )
        )

    replay_blockers = [
        _blocker_entry(
            code=flag,
            message=f'Replay summary reports `{flag}`.',
            current=True,
            baseline=flag in set(replay['baseline']['quality_flags']),
            vs_baseline='new' if flag not in set(replay['baseline']['quality_flags']) else 'carried_forward',
        )
        for flag in replay['current']['quality_flags']
    ]
    for stage_name, count in replay['current']['failure_counts_by_stage'].items():
        if count <= 0:
            continue
        baseline_count = replay['baseline']['failure_counts_by_stage'].get(stage_name, 0)
        replay_blockers.append(
            _blocker_entry(
                code=f'failure_stage:{stage_name}',
                message=f'Replay recorded {count} failure record(s) at the `{stage_name}` stage.',
                current=count,
                baseline=baseline_count,
                vs_baseline=_count_trend(count, baseline_count),
            )
        )

    prior_blockers: list[dict[str, object]] = []
    if prior_review['current']['prior_candidate_count'] == 0:
        prior_blockers.append(
            _blocker_entry(
                code='no_prior_candidates',
                message='Prior induction produced no DecisionPriorCard candidates for review.',
                current=0,
                baseline=prior_review['baseline']['selected_prior_count'],
                vs_baseline=_count_trend(0, prior_review['baseline']['selected_prior_count']),
            )
        )
    if not prior_review['current']['accepted_prior_ids']:
        prior_blockers.append(
            _blocker_entry(
                code='accepted_prior_ids_empty',
                message='Prior review accepted no prior ids, so export must keep selected_prior_ids empty.',
                current=0,
                baseline=prior_review['baseline']['accepted_prior_count'],
                vs_baseline=_count_trend(0, prior_review['baseline']['accepted_prior_count']),
            )
        )
    for flag, count in prior_review['current']['quality_flag_counts'].items():
        if count <= 0:
            continue
        prior_blockers.append(
            _blocker_entry(
                code=f'quality_flag:{flag}',
                message=f'Prior induction candidates carry `{flag}` on {count} card(s).',
                current=count,
            )
        )

    export_blockers = [
        _blocker_entry(
            code=flag,
            message=f'Export summary reports `{flag}`.',
            current=True,
            baseline=flag in set(export['baseline']['quality_flags']),
            vs_baseline='new' if flag not in set(export['baseline']['quality_flags']) else 'carried_forward',
        )
        for flag in export['current']['quality_flags']
    ]
    if not export['current']['ready_for_training']:
        export_blockers.append(
            _blocker_entry(
                code='not_ready_for_training',
                message='Export remains unavailable for training.',
                current=export['current']['ready_for_training'],
                baseline=export['baseline']['ready_for_training'],
                vs_baseline='regressed'
                if export['baseline']['ready_for_training'] and not export['current']['ready_for_training']
                else 'unchanged',
            )
        )
    if not export['current']['ready_for_eval']:
        export_blockers.append(
            _blocker_entry(
                code='not_ready_for_eval',
                message='Export is not ready for evaluation.',
                current=export['current']['ready_for_eval'],
                baseline=export['baseline']['ready_for_eval'],
                vs_baseline='regressed'
                if export['baseline']['ready_for_eval'] and not export['current']['ready_for_eval']
                else 'unchanged',
            )
        )
    if export['current']['accepted_but_unselected_prior_count'] > 0:
        baseline_unselected_prior_count = int(export['baseline']['accepted_but_unselected_prior_count'] or 0)
        export_blockers.append(
            _blocker_entry(
                code='accepted_priors_unselected',
                message=(
                    'Export preserved accepted prior ids only as explicit exclusion records because they do not support '
                    'the exported primary route.'
                ),
                current=export['current']['accepted_but_unselected_prior_count'],
                baseline=baseline_unselected_prior_count,
                vs_baseline=_count_trend(
                    export['current']['accepted_but_unselected_prior_count'],
                    baseline_unselected_prior_count,
                ),
            )
        )
    if export['current']['accepted_but_unselected_antipattern_count'] > 0:
        baseline_unselected_antipattern_count = int(export['baseline']['accepted_but_unselected_antipattern_count'] or 0)
        export_blockers.append(
            _blocker_entry(
                code='accepted_antipatterns_unselected',
                message=(
                    'Export preserved accepted anti-pattern ids only as explicit exclusion records because they do not '
                    'match the exported primary route.'
                ),
                current=export['current']['accepted_but_unselected_antipattern_count'],
                baseline=baseline_unselected_antipattern_count,
                vs_baseline=_count_trend(
                    export['current']['accepted_but_unselected_antipattern_count'],
                    baseline_unselected_antipattern_count,
                ),
            )
        )

    return {
        'packet_id': replay_summary.get('packet_id'),
        'cutoff_year': replay_summary.get('cutoff_year'),
        'baseline_replay_bundle': str(baseline_replay_bundle_dir.resolve()),
        'baseline_export_bundle': str(baseline_export_bundle_dir.resolve()),
        'current_recommendation': best_cycle_selection.get('primary_recommendation_id'),
        'package': package,
        'replay': replay,
        'prior_review': prior_review,
        'export': export,
        'best_cycle_selection': {
            'selected_iteration_label': best_cycle_selection.get('selected_iteration_label'),
            'review_status': best_cycle_selection.get('review_status'),
            'training_acceptance_verdict': best_cycle_selection.get('training_acceptance_verdict'),
            'primary_recommendation_id': best_cycle_selection.get('primary_recommendation_id'),
            'recommendation_evidence_refs': _string_list(best_cycle_selection.get('recommendation_evidence_refs')),
            'reviewed_candidate_cycles': (
                list(best_cycle_selection.get('reviewed_candidate_cycles'))
                if isinstance(best_cycle_selection.get('reviewed_candidate_cycles'), list)
                else []
            ),
            'bundle_manifest': str((export_bundle_dir / 'bundle_manifest.json').resolve()),
            'source_file': str((export_bundle_dir / 'best_cycle_selection.json').resolve()),
        },
        'blocker_queue': {
            'package_validation': package_blockers,
            'replay': replay_blockers,
            'prior_induction': prior_blockers,
            'export': export_blockers,
        },
        'source_artifacts': {
            'route_state_package_bundle': str(route_state_package_dir.resolve()),
            'replay_bundle': str(replay_bundle_dir.resolve()),
            'prior_review_bundle': str(prior_review_bundle_dir.resolve()),
            'export_bundle': str(export_bundle_dir.resolve()),
            'baseline_replay_inspection': str((baseline_replay_bundle_dir / 'replay_inspection.json').resolve()),
            'baseline_export_inspection': str((baseline_export_bundle_dir / 'export_inspection.json').resolve()),
            'current_export_inspection': str((export_bundle_dir / 'export_inspection.json').resolve()),
            'current_export_manifest': str((export_bundle_dir / 'bundle_manifest.json').resolve()),
            'current_training_view': str((export_bundle_dir / 'outputs' / 'training_view.json').resolve()),
            'current_best_cycle_selection': str((export_bundle_dir / 'best_cycle_selection.json').resolve()),
            'current_replay_inspection': str((replay_bundle_dir / 'replay_inspection.json').resolve()),
            'current_prior_review_summary': str((prior_review_bundle_dir / 'candidate_review_summary.json').resolve()),
        },
        'notes': {
            'export_visibility_policy': export_inspection.get('policy'),
            'baseline_visibility_policy': baseline_export_inspection.get('policy'),
            'route_state_package_validation': replay_inspection.get('route_state_package_validation'),
            'export_manifest': export_manifest,
            'training_view_review': training_view.get('review'),
        },
    }


def render_phase10_validation_report(comparison_summary: dict[str, object]) -> str:
    package = _json_object(comparison_summary.get('package') or {}, label='Phase 10 comparison package')
    replay = _json_object(comparison_summary.get('replay') or {}, label='Phase 10 comparison replay')
    prior_review = _json_object(comparison_summary.get('prior_review') or {}, label='Phase 10 comparison prior review')
    export = _json_object(comparison_summary.get('export') or {}, label='Phase 10 comparison export')
    best_cycle_selection = _json_object(
        comparison_summary.get('best_cycle_selection') or {},
        label='Phase 10 best-cycle selection',
    )
    source_artifacts = _json_object(comparison_summary.get('source_artifacts') or {}, label='Phase 10 source artifacts')
    blocker_queue = _json_object(comparison_summary.get('blocker_queue') or {}, label='Phase 10 blocker queue')

    package_current = _json_object(package.get('current') or {}, label='Phase 10 package current')
    package_baseline = _json_object(package.get('baseline') or {}, label='Phase 10 package baseline')
    replay_current = _json_object(replay.get('current') or {}, label='Phase 10 replay current')
    replay_baseline = _json_object(replay.get('baseline') or {}, label='Phase 10 replay baseline')
    prior_current = _json_object(prior_review.get('current') or {}, label='Phase 10 prior current')
    prior_baseline = _json_object(prior_review.get('baseline') or {}, label='Phase 10 prior baseline')
    export_current = _json_object(export.get('current') or {}, label='Phase 10 export current')
    export_baseline = _json_object(export.get('baseline') or {}, label='Phase 10 export baseline')

    lines = [
        '# Phase 10 Multi-Paper Validation Report',
        '',
        f"- Packet id: `{comparison_summary.get('packet_id')}`",
        f"- Cutoff year: `{comparison_summary.get('cutoff_year')}`",
        f"- Baseline replay bundle: `{comparison_summary.get('baseline_replay_bundle')}`",
        f"- Baseline export bundle: `{comparison_summary.get('baseline_export_bundle')}`",
        '',
        '## Stage Comparison',
        '',
        '### Package',
        f"- Current quality: `{package_current.get('quality_tier')}` vs baseline `{package_baseline.get('quality_tier')}`",
        f"- Ready for replay: `{package_current.get('ready_for_replay')}` vs baseline `{package_baseline.get('ready_for_replay')}`",
        f"- Quality flags: `{', '.join(_string_list(package_current.get('quality_flags'))) or 'none'}`",
        f"- Role counts: `{package_current.get('role_counts')}`",
        '',
        '### Replay',
        f"- Current quality: `{replay_current.get('quality_tier')}` vs baseline `{replay_baseline.get('quality_tier')}`",
        f"- Ready for pilot: `{replay_current.get('ready_for_pilot')}` vs baseline `{replay_baseline.get('ready_for_pilot')}`",
        f"- Failure record count: `{replay_current.get('failure_record_count')}` vs baseline `{replay_baseline.get('failure_record_count')}`",
        f"- Failure counts by stage: `{replay_current.get('failure_counts_by_stage')}`",
        f"- Quality flags: `{', '.join(_string_list(replay_current.get('quality_flags'))) or 'none'}`",
        '',
        '### Prior Review',
        f"- Prior candidates: `{prior_current.get('prior_candidate_count')}`",
        f"- Anti-pattern candidates: `{prior_current.get('anti_pattern_candidate_count')}`",
        f"- Accepted prior ids: `{len(_string_list(prior_current.get('accepted_prior_ids')))}` vs baseline accepted prior count `{prior_baseline.get('accepted_prior_count')}`",
        f"- Accepted anti-pattern ids: `{len(_string_list(prior_current.get('accepted_anti_pattern_ids')))}` vs baseline accepted anti-pattern count `{prior_baseline.get('accepted_anti_pattern_count')}`",
        f"- Quality flag counts: `{prior_current.get('quality_flag_counts')}`",
        '',
        '### Export',
        f"- Current quality: `{export_current.get('quality_tier')}` vs baseline `{export_baseline.get('quality_tier')}`",
        f"- Ready for training: `{export_current.get('ready_for_training')}` vs baseline `{export_baseline.get('ready_for_training')}`",
        f"- Ready for eval: `{export_current.get('ready_for_eval')}` vs baseline `{export_baseline.get('ready_for_eval')}`",
        f"- Selected prior count: `{export_current.get('selected_prior_count')}` vs baseline `{export_baseline.get('selected_prior_count')}`",
        f"- Selected anti-pattern count: `{export_current.get('selected_antipattern_count')}` vs baseline `{export_baseline.get('selected_antipattern_count')}`",
        (
            f"- Accepted but unselected prior count: `{export_current.get('accepted_but_unselected_prior_count')}` "
            f"vs baseline `{export_baseline.get('accepted_but_unselected_prior_count')}`"
        ),
        (
            f"- Accepted but unselected anti-pattern count: `{export_current.get('accepted_but_unselected_antipattern_count')}` "
            f"vs baseline `{export_baseline.get('accepted_but_unselected_antipattern_count')}`"
        ),
        f"- Visibility buckets: `{export_current.get('visibility_bucket_counts')}`",
        f"- Training view sections: `{export_current.get('training_view_sections')}`",
        f"- Training view file: `{source_artifacts.get('current_training_view')}`",
        f"- Best-cycle selection file: `{source_artifacts.get('current_best_cycle_selection')}`",
        f"- Current recommendation: `{comparison_summary.get('current_recommendation') or 'not stated'}`",
        (
            f"- Recommendation evidence refs: "
            f"`{best_cycle_selection.get('recommendation_evidence_refs') or []}`"
        ),
        f"- Quality flags: `{', '.join(_string_list(export_current.get('quality_flags'))) or 'none'}`",
        '',
        '## Blocker Queue',
        '',
    ]

    for stage_name in ('package_validation', 'replay', 'prior_induction', 'export'):
        lines.append(f'### {stage_name.replace("_", " ").title()}')
        blockers = blocker_queue.get(stage_name)
        if not isinstance(blockers, list) or not blockers:
            lines.append('- None.')
            lines.append('')
            continue
        for blocker in blockers:
            if not isinstance(blocker, dict):
                continue
            lines.append(
                f"- `{blocker.get('code')}`: {blocker.get('message')} "
                f"(current=`{blocker.get('current')}`, baseline=`{blocker.get('baseline', 'n/a')}`, vs_baseline=`{blocker.get('vs_baseline', 'n/a')}`)"
            )
        lines.append('')

    lines.extend(
        [
            '## Source Of Truth',
            '',
            '- This report is rendered from `comparison_summary.json`, not from bundle prose.',
        ]
    )
    return '\n'.join(lines) + '\n'


def _write_phase10_export_bundle(
    *,
    replay_bundle_files: dict[str, Path],
    prior_review_bundle_files: dict[str, Path],
    output_dir: Path,
    built_at: str | None = None,
) -> tuple[dict[str, Path], dict[str, object]]:
    replay_manifest_path = replay_bundle_files['bundle_manifest']
    review_manifest_path = prior_review_bundle_files['bundle_manifest']
    replay_manifest_payload = _load_json(replay_manifest_path)
    review_manifest_payload = _load_json(review_manifest_path)
    if not isinstance(replay_manifest_payload, dict):
        raise ValueError(f'Phase 10 replay bundle manifest must decode to a JSON object: {replay_manifest_path}')
    if not isinstance(review_manifest_payload, dict):
        raise ValueError(f'Phase 10 prior review bundle manifest must decode to a JSON object: {review_manifest_path}')

    replay_summary_payload = _load_json(replay_bundle_files['replay_summary'])
    if not isinstance(replay_summary_payload, dict):
        raise ValueError(f'Phase 10 replay summary must decode to a JSON object: {replay_bundle_files["replay_summary"]}')

    route_packet = load_route_packet(replay_bundle_files['route_packet'])
    route_state = _load_model(replay_bundle_files['primary_route_state'], RouteState)
    why_now_case = _load_model(replay_bundle_files['why_now_case'], WhyNowCase)
    replay_episode = _load_model(replay_bundle_files['decision_episode'], DecisionEpisode)
    comparison_cases = _load_models(replay_bundle_files['route_comparison_cases'], RouteComparisonCase)
    prior_candidates = _load_models(prior_review_bundle_files['prior_candidates'], DecisionPriorCard)
    anti_pattern_candidates = _load_models(prior_review_bundle_files['anti_pattern_candidates'], AntiPatternCard)
    accepted_prior_ids = _bundle_string_list(
        review_manifest_payload,
        key='accepted_prior_ids',
        label='Phase 10 prior review bundle',
    )
    accepted_anti_pattern_ids = _bundle_string_list(
        review_manifest_payload,
        key='accepted_anti_pattern_ids',
        label='Phase 10 prior review bundle',
    )

    selected_comparison_case_id = str(replay_summary_payload.get('selected_comparison_case_id') or '').strip()
    comparison_case = next(
        (
            case
            for case in comparison_cases
            if case.route_comparison_case_id == selected_comparison_case_id
        ),
        None,
    )

    export = build_decision_episode_audit_export(
        route_packet=route_packet,
        route_state=route_state,
        why_now_case=why_now_case,
        comparison_case=comparison_case,
        hindsight_outcome=replay_episode.hindsight_outcome,
        prior_cards=prior_candidates,
        anti_pattern_cards=anti_pattern_candidates,
        accepted_prior_ids=accepted_prior_ids,
        accepted_anti_pattern_ids=accepted_anti_pattern_ids,
        route_state_ref=str(replay_bundle_files['primary_route_state'].resolve()),
        source_replay_bundle_refs=_bundle_source_refs(replay_manifest_path, replay_manifest_payload),
        source_review_bundle_refs=_bundle_source_refs(review_manifest_path, review_manifest_payload),
        built_at=built_at,
        episode_id=replay_episode.episode_id,
    )

    written_files = write_decision_episode_export_bundle(
        output_dir,
        export=export,
        metadata={
            'runner': 'backend/scripts/run_phase10_multi_paper_validation.py',
            'source_replay_bundle': str(replay_manifest_path.parent.resolve()),
            'source_prior_review_bundle': str(review_manifest_path.parent.resolve()),
        },
        selected_iteration_label=output_dir.parent.name,
        selection_source_artifacts={
            'replay_bundle': str(replay_manifest_path.parent.resolve()),
            'prior_review_bundle': str(review_manifest_path.parent.resolve()),
            'primary_route_state': str(replay_bundle_files['primary_route_state'].resolve()),
            'candidate_review_summary': str(prior_review_bundle_files['candidate_review_summary'].resolve()),
        },
    )
    export_summary_payload = _load_json(written_files['export_summary'])
    if not isinstance(export_summary_payload, dict):
        raise ValueError(f'Phase 10 export summary must decode to a JSON object: {written_files["export_summary"]}')
    return written_files, export_summary_payload


def resolve_phase10_runtime_paths(
    *,
    output_root: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> Phase10RuntimePaths:
    resolved_repo_root = _repo_root(repo_root)
    resolved_output_root = _resolve_repo_path(output_root or DEFAULT_PHASE10_OUTPUT_ROOT, repo_root=resolved_repo_root)
    generated_dir = resolved_output_root / 'generated'
    return Phase10RuntimePaths(
        repo_root=resolved_repo_root,
        output_root=resolved_output_root,
        generated_dir=generated_dir,
        generated_packets_dir=generated_dir / 'packets',
        package_manifest_path=generated_dir / DEFAULT_PHASE10_PACKAGE_MANIFEST_PATH.name,
        shared_dir=resolved_output_root / 'shared',
        l1_snapshot_path=resolved_output_root / 'shared' / DEFAULT_PHASE10_L1_SNAPSHOT_PATH.name,
    )


def load_phase10_canonical_inputs(
    *,
    packet_path: str | Path = DEFAULT_PHASE10_PACKET_PATH,
    assembly_manifest_path: str | Path = DEFAULT_PHASE10_ASSEMBLY_MANIFEST_PATH,
    repo_root: str | Path | None = None,
) -> Phase10CanonicalInputs:
    resolved_repo_root = _repo_root(repo_root)
    resolved_packet_path = _resolve_repo_path(packet_path, repo_root=resolved_repo_root)
    resolved_assembly_manifest_path = _resolve_repo_path(assembly_manifest_path, repo_root=resolved_repo_root)

    route_packet = load_route_packet(resolved_packet_path)
    assembly_manifest = load_bounded_packet_assembly_manifest(resolved_assembly_manifest_path)
    audit_result = audit_bounded_packet_assembly(route_packet, assembly_manifest)
    if audit_result.structural_errors:
        raise ValueError('; '.join(audit_result.structural_errors))

    return Phase10CanonicalInputs(
        repo_root=resolved_repo_root,
        packet_path=resolved_packet_path,
        assembly_manifest_path=resolved_assembly_manifest_path,
        route_packet=route_packet,
        assembly_manifest=assembly_manifest,
    )


def collect_phase10_trace_sources(canonical_inputs: Phase10CanonicalInputs) -> tuple[Phase10TraceSource, ...]:
    collected: list[Phase10TraceSource] = []
    seen_paper_ids: set[str] = set()
    packet_items_by_paper_id = {item.paper_id: item for item in canonical_inputs.route_packet.included_items}

    for role, group in canonical_inputs.assembly_manifest.role_groups().items():
        for member in group.members:
            trace_ref = str(member.trace_ref or '').strip()
            if not trace_ref:
                raise ValueError(f'Phase 10 assembly member {member.paper_id} is missing trace_ref')

            trace_ref_path = Path(trace_ref)
            if trace_ref_path.is_absolute():
                raise ValueError(f'Phase 10 assembly member {member.paper_id} trace_ref must stay repo-relative')

            resolved_trace_path = _resolve_repo_path(trace_ref_path, repo_root=canonical_inputs.repo_root)
            if not resolved_trace_path.is_file():
                raise FileNotFoundError(f'Phase 10 trace_ref not found for paper {member.paper_id}: {resolved_trace_path}')

            if member.paper_id in seen_paper_ids:
                raise ValueError(f'Phase 10 assembly paper_id reused across role groups: {member.paper_id}')

            packet_item = packet_items_by_paper_id.get(member.paper_id)
            if packet_item is None:
                raise ValueError(f'Phase 10 assembly paper_id missing from canonical packet: {member.paper_id}')

            trace = load_paper_logic_trace(resolved_trace_path)
            trace_id = str(trace.trace_id or '').strip()
            if not trace_id:
                raise ValueError(f'Phase 10 trace_ref missing trace_id for paper {member.paper_id}: {resolved_trace_path}')

            packet_trace_id = str(packet_item.trace_id or '').strip()
            if packet_trace_id and packet_trace_id != trace_id:
                raise ValueError(
                    f'Phase 10 trace_ref trace_id mismatch for paper {member.paper_id}: expected {packet_trace_id}, got {trace_id}'
                )

            runtime_paper_id = str(trace.paper_metadata.paper_id or '').strip()
            if not runtime_paper_id:
                raise ValueError(
                    f'Phase 10 trace_ref missing paper_metadata.paper_id for paper {member.paper_id}: {resolved_trace_path}'
                )

            seen_paper_ids.add(member.paper_id)
            collected.append(
                Phase10TraceSource(
                    role=role,
                    paper_id=member.paper_id,
                    trace_id=trace_id,
                    runtime_paper_id=runtime_paper_id,
                    trace_ref=trace_ref,
                    resolved_path=resolved_trace_path,
                )
            )

    return tuple(collected)


def build_phase10_l1_snapshot(
    canonical_inputs: Phase10CanonicalInputs,
    *,
    trace_sources: tuple[Phase10TraceSource, ...],
    output_path: str | Path | None = None,
    built_at: str | None = None,
) -> tuple[HistoricalEnvironmentSnapshot, Path]:
    snapshot_path = _resolve_repo_path(
        output_path or DEFAULT_PHASE10_L1_SNAPSHOT_PATH,
        repo_root=canonical_inputs.repo_root,
    )
    traces = load_paper_logic_traces(trace_files=[source.resolved_path for source in trace_sources])
    ensure_packet_trace_coverage(canonical_inputs.route_packet, traces)

    snapshot = build_historical_environment_snapshot(
        canonical_inputs.route_packet,
        traces,
        built_at=built_at,
        snapshot_id=snapshot_path.stem,
    )
    written_snapshot_path = write_historical_environment_snapshot(snapshot_path, snapshot)
    return snapshot, written_snapshot_path


def _subset_role_counts(route_packet: RoutePacket, *, paper_ids: set[str]) -> dict[str, int]:
    counts = {key: 0 for key in route_packet.packet_composition.role_counts.model_dump(mode='json').keys()}
    for item in route_packet.included_items:
        if item.paper_id in paper_ids:
            counts[item.item_role] += 1
    return counts


def _subset_packet_quality_flags(route_packet: RoutePacket, *, role: Phase10Role) -> list[str]:
    flags = [flag for flag in route_packet.packet_quality.quality_flags if flag != 'l1_snapshot_placeholder']
    flags.append(f'phase10_{role}_runtime_subset')
    deduped: list[str] = []
    for flag in flags:
        normalized = str(flag or '').strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _runtime_paper_id_map(trace_sources: tuple[Phase10TraceSource, ...]) -> dict[str, str]:
    return {source.paper_id: source.runtime_paper_id for source in trace_sources}


def _runtime_packet_items_payload(
    items: list[object],
    *,
    runtime_paper_ids: dict[str, str],
) -> list[dict[str, object]]:
    packet_items_payload: list[dict[str, object]] = []
    for item in items:
        item_payload = item.model_dump(mode='json', exclude_none=True)
        runtime_paper_id = runtime_paper_ids.get(item.paper_id)
        if not runtime_paper_id:
            raise ValueError(f'Phase 10 runtime subset missing trace-native paper_id for canonical paper {item.paper_id}')
        item_payload['paper_id'] = runtime_paper_id
        packet_items_payload.append(item_payload)
    return packet_items_payload


def _build_role_subset_packet(
    canonical_inputs: Phase10CanonicalInputs,
    *,
    role: Phase10Role,
    paper_ids: list[str],
    runtime_paper_ids: dict[str, str],
    built_at: str,
    snapshot: HistoricalEnvironmentSnapshot,
) -> RoutePacket:
    included_lookup = {item.paper_id: item for item in canonical_inputs.route_packet.included_items}
    missing_paper_ids = [paper_id for paper_id in paper_ids if paper_id not in included_lookup]
    if missing_paper_ids:
        raise ValueError(f'Phase 10 runtime subset contains paper_ids missing from the canonical packet: {", ".join(missing_paper_ids)}')

    subset_items = [included_lookup[paper_id] for paper_id in paper_ids]
    subset_role_counts = _subset_role_counts(canonical_inputs.route_packet, paper_ids=set(paper_ids))
    packet_items_payload = _runtime_packet_items_payload(subset_items, runtime_paper_ids=runtime_paper_ids)

    subset_payload = canonical_inputs.route_packet.model_dump(mode='json', exclude_none=True)
    subset_payload['packet_id'] = f'phase10-{_role_slug(role)}-{canonical_inputs.route_packet.packet_id}'
    subset_payload['built_at'] = built_at
    subset_payload['l1_snapshot_ref'] = build_l1_snapshot_ref(snapshot).model_dump(mode='json', exclude_none=True)
    subset_payload['included_items'] = packet_items_payload
    subset_payload['packet_composition'] = {
        'target_size': len(subset_items),
        'actual_size': len(subset_items),
        'role_counts': subset_role_counts,
        'coverage_ok': False,
        'missing_roles': [role_name for role_name, count in subset_role_counts.items() if count == 0],
    }
    subset_payload['packet_quality'] = {
        **canonical_inputs.route_packet.packet_quality.model_dump(mode='json', exclude_none=True),
        'quality_tier': 'yellow',
        'ready_for_route_state': False,
        'quality_flags': _subset_packet_quality_flags(canonical_inputs.route_packet, role=role),
    }
    notes = str(canonical_inputs.route_packet.compiler_hints.notes_for_route_state_compiler or '').strip()
    subset_payload['compiler_hints'] = {
        **canonical_inputs.route_packet.compiler_hints.model_dump(mode='json', exclude_none=True),
        'notes_for_route_state_compiler': (
            f'{notes} Runtime subset generated from the committed Phase 9 assembly manifest for the {role} role. '
            f'Canonical Phase 9 paper_ids: {", ".join(paper_ids)}. Runtime packet paper_ids are trace-native for synthesis compatibility.'
            if notes
            else (
                f'Runtime subset generated from the committed Phase 9 assembly manifest for the {role} role. '
                f'Canonical Phase 9 paper_ids: {", ".join(paper_ids)}. Runtime packet paper_ids are trace-native for synthesis compatibility.'
            )
        ),
    }

    return RoutePacket.model_validate(subset_payload)


def _build_replay_runtime_packet(
    canonical_inputs: Phase10CanonicalInputs,
    *,
    trace_sources: tuple[Phase10TraceSource, ...],
    built_at: str,
    snapshot: HistoricalEnvironmentSnapshot,
) -> RoutePacket:
    packet_payload = canonical_inputs.route_packet.model_dump(mode='json', exclude_none=True)
    packet_payload['built_at'] = built_at
    packet_payload['l1_snapshot_ref'] = build_l1_snapshot_ref(snapshot).model_dump(mode='json', exclude_none=True)
    packet_payload['included_items'] = _runtime_packet_items_payload(
        list(canonical_inputs.route_packet.included_items),
        runtime_paper_ids=_runtime_paper_id_map(trace_sources),
    )
    notes = str(canonical_inputs.route_packet.compiler_hints.notes_for_route_state_compiler or '').strip()
    packet_payload['compiler_hints'] = {
        **canonical_inputs.route_packet.compiler_hints.model_dump(mode='json', exclude_none=True),
        'notes_for_route_state_compiler': (
            f'{notes} Runtime replay packet preserves the committed Phase 9 boundary while translating included paper_ids '
            'to trace-native ids for synthesis compatibility.'
            if notes
            else (
                'Runtime replay packet preserves the committed Phase 9 boundary while translating included paper_ids '
                'to trace-native ids for synthesis compatibility.'
            )
        ),
    }
    return RoutePacket.model_validate(packet_payload)


def _write_phase10_role_packets(
    canonical_inputs: Phase10CanonicalInputs,
    *,
    trace_sources: tuple[Phase10TraceSource, ...],
    runtime_paths: Phase10RuntimePaths,
    snapshot: HistoricalEnvironmentSnapshot,
    built_at: str,
) -> dict[Phase10Role, tuple[Phase10RolePacketArtifact, ...]]:
    trace_lookup = {source.paper_id: source for source in trace_sources}
    role_packets: dict[Phase10Role, list[Phase10RolePacketArtifact]] = {
        'support': [],
        'alternative': [],
        'held_out': [],
    }
    runtime_paper_ids = _runtime_paper_id_map(trace_sources)

    for spec in _phase10_runtime_role_packets(canonical_inputs):
        trace_files = tuple(trace_lookup[paper_id].resolved_path for paper_id in spec.paper_ids)
        role_packet = _build_role_subset_packet(
            canonical_inputs,
            role=spec.role,
            paper_ids=list(spec.paper_ids),
            runtime_paper_ids=runtime_paper_ids,
            built_at=built_at,
            snapshot=snapshot,
        )
        packet_slug = spec.entry_id.removeprefix('phase10-')
        packet_path = runtime_paths.generated_packets_dir / f'phase10-{packet_slug}-route-packet.json'
        _write_json(packet_path, role_packet.model_dump(mode='json', exclude_none=True))
        role_packets[spec.role].append(
            Phase10RolePacketArtifact(
                entry_id=spec.entry_id,
                role=spec.role,
                packet_path=packet_path,
                route_state_id=spec.route_state_id,
                paper_ids=spec.paper_ids,
                trace_files=trace_files,
                notes=spec.notes,
                distinctness_rationale=spec.distinctness_rationale,
            )
        )

    return {
        role: tuple(artifacts)
        for role, artifacts in role_packets.items()
    }


def _build_phase10_package_manifest(
    canonical_inputs: Phase10CanonicalInputs,
    *,
    runtime_paths: Phase10RuntimePaths,
    snapshot_path: Path,
    role_packets: dict[Phase10Role, tuple[Phase10RolePacketArtifact, ...]],
    built_at: str,
) -> RouteStatePackageManifest:
    role_groups = canonical_inputs.assembly_manifest.role_groups()
    entries: list[RouteStatePackageEntry] = []
    for role in ('support', 'alternative', 'held_out'):
        for artifact in role_packets[role]:
            entries.append(
                RouteStatePackageEntry(
                    entry_id=artifact.entry_id,
                    role=role,
                    packet_path=_relative_manifest_path(artifact.packet_path, base_dir=runtime_paths.generated_dir),
                    trace_files=[
                        _relative_manifest_path(trace_path, base_dir=runtime_paths.generated_dir)
                        for trace_path in artifact.trace_files
                    ],
                    l1_snapshot_path=_relative_manifest_path(snapshot_path, base_dir=runtime_paths.generated_dir),
                    route_state_id=artifact.route_state_id,
                    built_at=built_at,
                    notes=artifact.notes or role_groups[role].group_reason,
                    distinctness_rationale=artifact.distinctness_rationale,
                )
            )
    return RouteStatePackageManifest(
        package_id=f'phase10-{canonical_inputs.route_packet.packet_id}-route-state-package',
        built_at=built_at,
        topic_scope=canonical_inputs.assembly_manifest.topic_scope,
        cutoff_year=canonical_inputs.assembly_manifest.cutoff_year,
        entries=entries,
    )


def prepare_phase10_runtime_bridge(
    *,
    packet_path: str | Path = DEFAULT_PHASE10_PACKET_PATH,
    assembly_manifest_path: str | Path = DEFAULT_PHASE10_ASSEMBLY_MANIFEST_PATH,
    output_root: str | Path | None = None,
    l1_snapshot_output_path: str | Path | None = None,
    built_at: str | None = None,
    repo_root: str | Path | None = None,
) -> Phase10RuntimeBridge:
    canonical_inputs = load_phase10_canonical_inputs(
        packet_path=packet_path,
        assembly_manifest_path=assembly_manifest_path,
        repo_root=repo_root,
    )
    runtime_paths = resolve_phase10_runtime_paths(output_root=output_root, repo_root=canonical_inputs.repo_root)
    trace_sources = collect_phase10_trace_sources(canonical_inputs)
    package_built_at = _package_built_at(canonical_inputs.route_packet, canonical_inputs.assembly_manifest, built_at)
    l1_snapshot, snapshot_path = build_phase10_l1_snapshot(
        canonical_inputs,
        trace_sources=trace_sources,
        output_path=l1_snapshot_output_path or runtime_paths.l1_snapshot_path,
        built_at=package_built_at,
    )
    role_packets = _write_phase10_role_packets(
        canonical_inputs,
        trace_sources=trace_sources,
        runtime_paths=runtime_paths,
        snapshot=l1_snapshot,
        built_at=package_built_at,
    )
    package_manifest = _build_phase10_package_manifest(
        canonical_inputs,
        runtime_paths=runtime_paths,
        snapshot_path=snapshot_path,
        role_packets=role_packets,
        built_at=package_built_at,
    )
    package_manifest_path = _write_json(
        runtime_paths.package_manifest_path,
        package_manifest.model_dump(mode='json', exclude_none=True),
    )
    return Phase10RuntimeBridge(
        canonical_inputs=canonical_inputs,
        runtime_paths=runtime_paths,
        trace_sources=trace_sources,
        l1_snapshot=l1_snapshot,
        l1_snapshot_path=snapshot_path,
        package_manifest=package_manifest,
        package_manifest_path=package_manifest_path,
        role_packets=role_packets,
    )


def run_phase10_package_and_replay(
    *,
    packet_path: str | Path = DEFAULT_PHASE10_PACKET_PATH,
    assembly_manifest_path: str | Path = DEFAULT_PHASE10_ASSEMBLY_MANIFEST_PATH,
    l1_snapshot_output_path: str | Path = DEFAULT_PHASE10_L1_SNAPSHOT_PATH,
    output_dir: str | Path,
    baseline_replay_bundle: str | Path = DEFAULT_PHASE10_BASELINE_REPLAY_BUNDLE,
    baseline_export_bundle: str | Path = DEFAULT_PHASE10_BASELINE_EXPORT_BUNDLE,
    reviewer_ids: list[str] | None = None,
    allow_scope_fallback_merge: bool = False,
    report_md: str | Path | None = None,
    built_at: str | None = None,
    repo_root: str | Path | None = None,
) -> Phase10ValidationRun:
    bridge = prepare_phase10_runtime_bridge(
        packet_path=packet_path,
        assembly_manifest_path=assembly_manifest_path,
        l1_snapshot_output_path=l1_snapshot_output_path,
        built_at=built_at,
        repo_root=repo_root,
    )
    resolved_output_dir = _resolve_repo_path(output_dir, repo_root=bridge.canonical_inputs.repo_root)
    route_state_package_output_dir = resolved_output_dir / 'route_state_package'
    replay_output_dir = resolved_output_dir / 'replay_bundle'
    prior_review_output_dir = resolved_output_dir / 'prior_review_bundle'
    export_output_dir = resolved_output_dir / 'export_bundle'
    comparison_summary_path = resolved_output_dir / 'comparison_summary.json'

    route_state_package_compilation = compile_route_state_package(
        bridge.package_manifest,
        manifest_base_dir=bridge.package_manifest_path.parent,
    )
    route_state_package_bundle_files = write_route_state_package_bundle(
        route_state_package_output_dir,
        compilation=route_state_package_compilation,
        metadata={
            'runner': 'backend/scripts/run_phase10_multi_paper_validation.py',
            'manifest_path': str(bridge.package_manifest_path.resolve()),
            'packet_path': str(bridge.canonical_inputs.packet_path.resolve()),
            'assembly_manifest_path': str(bridge.canonical_inputs.assembly_manifest_path.resolve()),
        },
    )
    loaded_route_state_package = load_route_state_package_bundle(route_state_package_output_dir)
    traces = load_paper_logic_traces(trace_files=[source.resolved_path for source in bridge.trace_sources])
    ensure_packet_trace_coverage(bridge.canonical_inputs.route_packet, traces)
    runtime_replay_packet = _build_replay_runtime_packet(
        bridge.canonical_inputs,
        trace_sources=bridge.trace_sources,
        built_at=built_at or bridge.l1_snapshot.built_at,
        snapshot=bridge.l1_snapshot,
    )

    replay_compilation = compile_historical_replay(
        runtime_replay_packet,
        traces,
        l1_snapshot=bridge.l1_snapshot,
        support_route_states=loaded_route_state_package.support_route_states,
        alternative_route_states=loaded_route_state_package.alternative_route_states,
        held_out_route_states=loaded_route_state_package.held_out_route_states,
        reviewer_ids=reviewer_ids,
        built_at=built_at,
    )
    replay_bundle_files = write_replay_bundle(
        replay_output_dir,
        route_packet=bridge.canonical_inputs.route_packet,
        traces=traces,
        compilation=replay_compilation,
        historical_environment_snapshot=bridge.l1_snapshot,
        support_route_states=loaded_route_state_package.support_route_states,
        alternative_route_states=loaded_route_state_package.alternative_route_states,
        held_out_route_states=loaded_route_state_package.held_out_route_states,
        reviewer_ids=reviewer_ids,
        metadata={
            'runner': 'backend/scripts/run_phase10_multi_paper_validation.py',
            'l1_snapshot_id': bridge.l1_snapshot.snapshot_id,
            'route_state_package_id': loaded_route_state_package.manifest.package_id,
            'generated_manifest_path': str(bridge.package_manifest_path.resolve()),
        },
        route_state_package_validation=loaded_route_state_package.validation,
    )
    prior_candidate_registry = build_prior_candidate_registry_from_package(
        loaded_route_state_package,
        reviewer_ids=reviewer_ids,
        allow_scope_fallback_merge=allow_scope_fallback_merge,
        built_at=built_at,
    )
    prior_review_bundle_files = write_prior_candidate_review_bundle(
        prior_review_output_dir,
        registry=prior_candidate_registry,
        metadata={
            'runner': 'backend/scripts/run_phase10_multi_paper_validation.py',
            'route_state_package_id': loaded_route_state_package.manifest.package_id,
        },
    )
    export_bundle_files, export_summary_payload = _write_phase10_export_bundle(
        replay_bundle_files=replay_bundle_files,
        prior_review_bundle_files=prior_review_bundle_files,
        output_dir=export_output_dir,
        built_at=built_at,
    )
    comparison_summary = build_phase10_comparison_summary(
        route_state_package_bundle=route_state_package_output_dir,
        replay_bundle=replay_output_dir,
        prior_review_bundle=prior_review_output_dir,
        export_bundle=export_output_dir,
        baseline_replay_bundle=baseline_replay_bundle,
        baseline_export_bundle=baseline_export_bundle,
        repo_root=bridge.canonical_inputs.repo_root,
    )
    comparison_summary_path = _write_json(comparison_summary_path, comparison_summary)
    report_markdown_path = (
        _write_text(
            _resolve_repo_path(report_md, repo_root=bridge.canonical_inputs.repo_root),
            render_phase10_validation_report(comparison_summary),
        )
        if report_md is not None
        else None
    )

    route_state_package_summary = build_route_state_package_summary(
        route_state_package_compilation,
        output_dir=route_state_package_output_dir,
        bundle_manifest=route_state_package_bundle_files['bundle_manifest'],
    )
    replay_summary = build_replay_summary(
        route_packet=bridge.canonical_inputs.route_packet,
        traces=traces,
        compilation=replay_compilation,
        historical_environment_snapshot=bridge.l1_snapshot,
        support_route_states=loaded_route_state_package.support_route_states,
        alternative_route_states=loaded_route_state_package.alternative_route_states,
        held_out_route_states=loaded_route_state_package.held_out_route_states,
        reviewer_ids=reviewer_ids,
    )
    prior_review_manifest_payload = _load_json(prior_review_bundle_files['bundle_manifest'])
    if not isinstance(prior_review_manifest_payload, dict):
        raise ValueError(
            f'Phase 10 prior review bundle manifest must decode to a JSON object: {prior_review_bundle_files["bundle_manifest"]}'
        )

    summary = {
        'packet_id': bridge.canonical_inputs.route_packet.packet_id,
        'cutoff_year': bridge.canonical_inputs.route_packet.cutoff_year,
        'l1_snapshot_output': str(bridge.l1_snapshot_path.resolve()),
        'generated_package_manifest': str(bridge.package_manifest_path.resolve()),
        'route_state_package_output_dir': str(route_state_package_output_dir.resolve()),
        'route_state_package_bundle_manifest': str(route_state_package_bundle_files['bundle_manifest'].resolve()),
        'route_state_package_validation_quality_tier': route_state_package_summary['validation_quality_tier'],
        'route_state_package_validation_flags': list(route_state_package_summary['validation_quality_flags']),
        'replay_output_dir': str(replay_output_dir.resolve()),
        'replay_bundle_manifest': str(replay_bundle_files['bundle_manifest'].resolve()),
        'replay_summary_path': str(replay_bundle_files['replay_summary'].resolve()),
        'replay_inspection_path': str(replay_bundle_files['replay_inspection'].resolve()),
        'replay_quality_tier': replay_summary['replay_quality_tier'],
        'replay_quality_flags': list(replay_summary['quality_flags']),
        'ready_for_pilot': replay_summary['ready_for_pilot'],
        'support_route_state_count': replay_summary['support_route_state_count'],
        'alternative_route_state_count': replay_summary['alternative_route_state_count'],
        'held_out_route_state_count': replay_summary['held_out_route_state_count'],
        'reviewer_ids': list(replay_summary['reviewer_ids']),
        'prior_review_output_dir': str(prior_review_output_dir.resolve()),
        'prior_review_bundle_manifest': str(prior_review_bundle_files['bundle_manifest'].resolve()),
        'prior_review_summary_path': str(prior_review_bundle_files['candidate_review_summary'].resolve()),
        'prior_review_cluster_strategy': prior_candidate_registry.cluster_strategy,
        'prior_review_cluster_count': len(prior_candidate_registry.clusters),
        'prior_review_fallback_reason': prior_candidate_registry.fallback_reason,
        'prior_candidate_count': len(prior_candidate_registry.prior_candidates),
        'accepted_prior_ids': _bundle_string_list(
            prior_review_manifest_payload,
            key='accepted_prior_ids',
            label='Phase 10 prior review bundle',
        ),
        'accepted_anti_pattern_ids': _bundle_string_list(
            prior_review_manifest_payload,
            key='accepted_anti_pattern_ids',
            label='Phase 10 prior review bundle',
        ),
        'export_output_dir': str(export_output_dir.resolve()),
        'export_bundle_manifest': str(export_bundle_files['bundle_manifest'].resolve()),
        'export_summary_path': str(export_bundle_files['export_summary'].resolve()),
        'export_inspection_path': str(export_bundle_files['export_inspection'].resolve()),
        'training_view_path': str(export_bundle_files['training_view'].resolve()),
        'best_cycle_selection_path': str(export_bundle_files['best_cycle_selection'].resolve()),
        'selected_prior_ids': list(export_summary_payload['selected_prior_ids']),
        'selected_antipattern_ids': list(export_summary_payload['selected_antipattern_ids']),
        'accepted_but_unselected_prior_count': int(export_summary_payload.get('accepted_but_unselected_prior_count') or 0),
        'accepted_but_unselected_antipattern_count': int(
            export_summary_payload.get('accepted_but_unselected_antipattern_count') or 0
        ),
        'visibility_bucket_counts': dict(export_summary_payload['visibility_bucket_counts']),
        'baseline_replay_bundle': str(_resolve_bundle_dir(baseline_replay_bundle, repo_root=bridge.canonical_inputs.repo_root).resolve()),
        'baseline_export_bundle': str(_resolve_bundle_dir(baseline_export_bundle, repo_root=bridge.canonical_inputs.repo_root).resolve()),
        'allow_scope_fallback_merge': allow_scope_fallback_merge,
        'comparison_summary_path': str(comparison_summary_path.resolve()),
        'report_markdown_path': str(report_markdown_path.resolve()) if report_markdown_path is not None else None,
    }
    return Phase10ValidationRun(
        bridge=bridge,
        route_state_package_output_dir=route_state_package_output_dir,
        route_state_package_bundle_files=route_state_package_bundle_files,
        replay_output_dir=replay_output_dir,
        replay_bundle_files=replay_bundle_files,
        prior_review_output_dir=prior_review_output_dir,
        prior_review_bundle_files=prior_review_bundle_files,
        export_output_dir=export_output_dir,
        export_bundle_files=export_bundle_files,
        comparison_summary_path=comparison_summary_path,
        report_markdown_path=report_markdown_path,
        summary=summary,
    )


__all__ = [
    'DEFAULT_PHASE10_ASSEMBLY_MANIFEST_PATH',
    'DEFAULT_PHASE10_BASELINE_EXPORT_BUNDLE',
    'DEFAULT_PHASE10_BASELINE_REPLAY_BUNDLE',
    'DEFAULT_PHASE10_L1_SNAPSHOT_PATH',
    'DEFAULT_PHASE10_OUTPUT_ROOT',
    'DEFAULT_PHASE10_PACKAGE_MANIFEST_PATH',
    'DEFAULT_PHASE10_PACKET_PATH',
    'Phase10CanonicalInputs',
    'Phase10RolePacketArtifact',
    'Phase10RuntimeBridge',
    'Phase10RuntimePaths',
    'Phase10TraceSource',
    'Phase10ValidationRun',
    'build_phase10_comparison_summary',
    'build_phase10_l1_snapshot',
    'collect_phase10_trace_sources',
    'load_phase10_canonical_inputs',
    'prepare_phase10_runtime_bridge',
    'render_phase10_validation_report',
    'resolve_phase10_runtime_paths',
    'run_phase10_package_and_replay',
]
