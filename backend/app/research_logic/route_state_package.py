from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import RouteState
from .replay_io import (
    ensure_packet_trace_coverage,
    load_historical_environment_snapshot,
    load_paper_logic_traces,
    load_route_packet,
    load_route_state,
)
from .route_state_synthesizer import synthesize_route_state


RouteStatePackageRole = Literal['primary', 'support', 'alternative', 'held_out']
QualityTier = Literal['red', 'yellow', 'green']


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _read_json(path_like: str | Path) -> Any:
    path = _as_path(path_like)
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'route-state package artifact not found: {path}') from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid JSON in route-state package artifact: {path}') from exc


def _write_json(path_like: str | Path, payload: Any) -> Path:
    path = _as_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


def _resolve_path(path_like: str | Path, base_dir: str | Path) -> Path:
    path = _as_path(path_like)
    return path if path.is_absolute() else _as_path(base_dir) / path


def _normalize(value: str | None) -> str:
    return str(value or '').strip().lower()


def _scope_tokens(value: str | None) -> set[str]:
    normalized = _normalize(value)
    tokens = {
        token
        for token in ''.join(char if char.isalnum() else ' ' for char in normalized).split()
        if token
    }
    return tokens


def _scope_similarity(a: str | None, b: str | None) -> float:
    tokens_a = _scope_tokens(a)
    tokens_b = _scope_tokens(b)
    if not tokens_a or not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class RouteStatePackageEntry(ContractModel):
    entry_id: str
    role: RouteStatePackageRole
    packet_path: str
    trace_dir: str | None = None
    trace_files: list[str] = Field(default_factory=list)
    l1_snapshot_path: str | None = None
    route_state_id: str | None = None
    built_at: str | None = None
    notes: str | None = None

    @model_validator(mode='after')
    def validate_trace_source(self) -> 'RouteStatePackageEntry':
        if self.trace_dir is None and not self.trace_files:
            raise ValueError('RouteStatePackageEntry requires trace_dir or trace_files')
        return self


class RouteStatePackageManifest(ContractModel):
    package_id: str
    schema_version: str = 'v1'
    built_at: str
    topic_scope: str
    cutoff_year: int
    entries: list[RouteStatePackageEntry] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_entries(self) -> 'RouteStatePackageManifest':
        if not self.entries:
            raise ValueError('RouteStatePackageManifest requires at least one entry')

        seen_entry_ids: set[str] = set()
        primary_count = 0
        for entry in self.entries:
            if entry.entry_id in seen_entry_ids:
                raise ValueError(f'duplicate RouteStatePackageEntry entry_id: {entry.entry_id}')
            seen_entry_ids.add(entry.entry_id)
            if entry.role == 'primary':
                primary_count += 1
        if primary_count > 1:
            raise ValueError('RouteStatePackageManifest allows at most one primary entry')
        return self


class RouteStatePackageArtifact(ContractModel):
    entry_id: str
    role: RouteStatePackageRole
    packet_id: str
    packet_path: str
    trace_sources: list[str] = Field(default_factory=list)
    l1_snapshot_path: str | None = None
    notes: str | None = None
    route_state: RouteState


class RouteStatePackageCompilation(ContractModel):
    package_id: str
    schema_version: str = 'v1'
    built_at: str
    topic_scope: str
    cutoff_year: int
    entries: list[RouteStatePackageArtifact] = Field(default_factory=list)

    def primary_route_state(self) -> RouteState | None:
        for artifact in self.entries:
            if artifact.role == 'primary':
                return artifact.route_state
        return None

    def grouped_route_states(self) -> dict[RouteStatePackageRole, list[RouteState]]:
        grouped: dict[RouteStatePackageRole, list[RouteState]] = {
            'primary': [],
            'support': [],
            'alternative': [],
            'held_out': [],
        }
        for artifact in self.entries:
            grouped[artifact.role].append(artifact.route_state)
        return grouped

    def induction_inputs(self) -> dict[str, list[RouteState]]:
        grouped = self.grouped_route_states()
        return {
            'support_route_states': list(grouped['support']),
            'alternative_route_states': list(grouped['alternative']),
            'held_out_route_states': list(grouped['held_out']),
        }


class RouteStatePackageBundleEntryRef(ContractModel):
    entry_id: str
    role: RouteStatePackageRole
    packet_id: str
    route_state_id: str
    file: str


class RouteStatePackageBundleManifest(ContractModel):
    package_id: str
    schema_version: str = 'v1'
    built_at: str
    topic_scope: str
    cutoff_year: int
    entry_count: int
    primary_route_state_file: str | None = None
    support_route_states_file: str
    alternative_route_states_file: str
    held_out_route_states_file: str
    validation_file: str | None = None
    entries: list[RouteStatePackageBundleEntryRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouteStatePackageValidation(ContractModel):
    quality_tier: QualityTier = 'red'
    ready_for_replay: bool = False
    quality_flags: list[str] = Field(default_factory=list)
    role_counts: dict[str, int] = Field(default_factory=dict)
    role_quality_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    support_route_state_ids: list[str] = Field(default_factory=list)
    alternative_route_state_ids: list[str] = Field(default_factory=list)
    held_out_route_state_ids: list[str] = Field(default_factory=list)
    scope_mismatch_entry_ids: list[str] = Field(default_factory=list)
    indistinct_alternative_entry_ids: list[str] = Field(default_factory=list)
    reused_packet_ids_across_roles: list[str] = Field(default_factory=list)


class LoadedRouteStatePackage(ContractModel):
    manifest: RouteStatePackageBundleManifest
    primary_route_state: RouteState | None = None
    support_route_states: list[RouteState] = Field(default_factory=list)
    alternative_route_states: list[RouteState] = Field(default_factory=list)
    held_out_route_states: list[RouteState] = Field(default_factory=list)
    validation: RouteStatePackageValidation | None = None

    def induction_inputs(self) -> dict[str, list[RouteState]]:
        return {
            'support_route_states': list(self.support_route_states),
            'alternative_route_states': list(self.alternative_route_states),
            'held_out_route_states': list(self.held_out_route_states),
        }


def load_route_state_package_manifest(path_like: str | Path) -> RouteStatePackageManifest:
    return RouteStatePackageManifest.model_validate(_read_json(path_like))


def compile_route_state_package(
    manifest: RouteStatePackageManifest | dict[str, Any],
    *,
    manifest_base_dir: str | Path | None = None,
) -> RouteStatePackageCompilation:
    manifest_model = (
        manifest if isinstance(manifest, RouteStatePackageManifest) else RouteStatePackageManifest.model_validate(manifest)
    )
    base_dir = _as_path(manifest_base_dir or Path.cwd())

    artifacts: list[RouteStatePackageArtifact] = []
    seen_route_state_ids: set[str] = set()

    for entry in manifest_model.entries:
        packet_path = _resolve_path(entry.packet_path, base_dir).resolve()
        route_packet = load_route_packet(packet_path)
        if route_packet.cutoff_year != manifest_model.cutoff_year:
            raise ValueError(
                f'RouteStatePackageEntry {entry.entry_id} cutoff_year {route_packet.cutoff_year} does not match package cutoff_year {manifest_model.cutoff_year}'
            )

        trace_dir = _resolve_path(entry.trace_dir, base_dir).resolve() if entry.trace_dir else None
        trace_files = [_resolve_path(path, base_dir).resolve() for path in entry.trace_files]
        traces = load_paper_logic_traces(trace_dir=trace_dir, trace_files=trace_files)
        ensure_packet_trace_coverage(route_packet, traces)

        historical_environment_snapshot = None
        resolved_snapshot_path: Path | None = None
        if entry.l1_snapshot_path:
            resolved_snapshot_path = _resolve_path(entry.l1_snapshot_path, base_dir).resolve()
            historical_environment_snapshot = load_historical_environment_snapshot(resolved_snapshot_path)
            if historical_environment_snapshot.cutoff_year != manifest_model.cutoff_year:
                raise ValueError(
                    f'RouteStatePackageEntry {entry.entry_id} l1_snapshot cutoff_year {historical_environment_snapshot.cutoff_year} does not match package cutoff_year {manifest_model.cutoff_year}'
                )

        route_state = synthesize_route_state(
            route_packet,
            traces,
            l1_snapshot=historical_environment_snapshot,
            built_at=entry.built_at,
            route_state_id=entry.route_state_id,
        )
        if route_state.route_state_id in seen_route_state_ids:
            raise ValueError(f'duplicate route_state_id in route-state package: {route_state.route_state_id}')
        seen_route_state_ids.add(route_state.route_state_id)

        artifacts.append(
            RouteStatePackageArtifact(
                entry_id=entry.entry_id,
                role=entry.role,
                packet_id=route_packet.packet_id,
                packet_path=str(packet_path),
                trace_sources=[str(path) for path in ([trace_dir] if trace_dir is not None else []) + trace_files],
                l1_snapshot_path=str(resolved_snapshot_path) if resolved_snapshot_path is not None else None,
                notes=entry.notes,
                route_state=route_state,
            )
        )

    return RouteStatePackageCompilation(
        package_id=manifest_model.package_id,
        schema_version=manifest_model.schema_version,
        built_at=manifest_model.built_at,
        topic_scope=manifest_model.topic_scope,
        cutoff_year=manifest_model.cutoff_year,
        entries=artifacts,
    )


def write_route_state_package_bundle(
    output_dir: str | Path,
    *,
    compilation: RouteStatePackageCompilation,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    bundle_dir = _as_path(output_dir)
    entries_dir = bundle_dir / 'entries'
    grouped_dir = bundle_dir / 'grouped'

    grouped = compilation.grouped_route_states()
    entry_refs: list[RouteStatePackageBundleEntryRef] = []
    validation = validate_route_state_package(compilation)

    written_files: dict[str, Path] = {}
    primary_route_state = compilation.primary_route_state()
    if primary_route_state is not None:
        written_files['primary_route_state'] = _write_json(
            grouped_dir / 'primary_route_state.json',
            primary_route_state.model_dump(mode='json', exclude_none=True),
        )

    written_files['support_route_states'] = _write_json(
        grouped_dir / 'support_route_states.json',
        [route_state.model_dump(mode='json', exclude_none=True) for route_state in grouped['support']],
    )
    written_files['alternative_route_states'] = _write_json(
        grouped_dir / 'alternative_route_states.json',
        [route_state.model_dump(mode='json', exclude_none=True) for route_state in grouped['alternative']],
    )
    written_files['held_out_route_states'] = _write_json(
        grouped_dir / 'held_out_route_states.json',
        [route_state.model_dump(mode='json', exclude_none=True) for route_state in grouped['held_out']],
    )
    written_files['validation'] = _write_json(
        bundle_dir / 'validation.json',
        validation.model_dump(mode='json', exclude_none=True),
    )

    for artifact in compilation.entries:
        entry_path = _write_json(
            entries_dir / f'{artifact.entry_id}.json',
            artifact.route_state.model_dump(mode='json', exclude_none=True),
        )
        entry_refs.append(
            RouteStatePackageBundleEntryRef(
                entry_id=artifact.entry_id,
                role=artifact.role,
                packet_id=artifact.packet_id,
                route_state_id=artifact.route_state.route_state_id,
                file=str(entry_path.relative_to(bundle_dir)).replace('\\', '/'),
            )
        )

    manifest_payload = RouteStatePackageBundleManifest(
        package_id=compilation.package_id,
        schema_version=compilation.schema_version,
        built_at=_utc_now_iso(),
        topic_scope=compilation.topic_scope,
        cutoff_year=compilation.cutoff_year,
        entry_count=len(compilation.entries),
        primary_route_state_file=(
            str(written_files['primary_route_state'].relative_to(bundle_dir)).replace('\\', '/')
            if 'primary_route_state' in written_files
            else None
        ),
        support_route_states_file=str(written_files['support_route_states'].relative_to(bundle_dir)).replace('\\', '/'),
        alternative_route_states_file=str(written_files['alternative_route_states'].relative_to(bundle_dir)).replace('\\', '/'),
        held_out_route_states_file=str(written_files['held_out_route_states'].relative_to(bundle_dir)).replace('\\', '/'),
        validation_file=str(written_files['validation'].relative_to(bundle_dir)).replace('\\', '/'),
        entries=entry_refs,
        metadata=dict(metadata or {}),
    )
    written_files['bundle_manifest'] = _write_json(
        bundle_dir / 'bundle_manifest.json',
        manifest_payload.model_dump(mode='json', exclude_none=True),
    )
    return written_files


def _load_route_state_list_file(path_like: str | Path) -> list[RouteState]:
    payload = _read_json(path_like)
    if not isinstance(payload, list):
        raise ValueError(f'route-state package list must be a JSON array: {path_like}')
    return [RouteState.model_validate(item) for item in payload]


def load_route_state_package_bundle(path_like: str | Path) -> LoadedRouteStatePackage:
    input_path = _as_path(path_like)
    manifest_path = input_path / 'bundle_manifest.json' if input_path.is_dir() else input_path
    manifest = RouteStatePackageBundleManifest.model_validate(_read_json(manifest_path))
    bundle_dir = manifest_path.parent

    primary_route_state = (
        load_route_state(bundle_dir / manifest.primary_route_state_file)
        if manifest.primary_route_state_file is not None
        else None
    )
    validation = (
        RouteStatePackageValidation.model_validate(_read_json(bundle_dir / manifest.validation_file))
        if manifest.validation_file is not None
        else None
    )
    return LoadedRouteStatePackage(
        manifest=manifest,
        primary_route_state=primary_route_state,
        support_route_states=_load_route_state_list_file(bundle_dir / manifest.support_route_states_file),
        alternative_route_states=_load_route_state_list_file(bundle_dir / manifest.alternative_route_states_file),
        held_out_route_states=_load_route_state_list_file(bundle_dir / manifest.held_out_route_states_file),
        validation=validation,
    )


def validate_route_state_package(compilation: RouteStatePackageCompilation) -> RouteStatePackageValidation:
    grouped = compilation.grouped_route_states()
    role_counts = {
        'primary': len(grouped['primary']),
        'support': len(grouped['support']),
        'alternative': len(grouped['alternative']),
        'held_out': len(grouped['held_out']),
    }

    role_quality_counts: dict[str, dict[str, int]] = {}
    for role, route_states in grouped.items():
        counts: Counter[str] = Counter(route_state.quality.quality_tier for route_state in route_states)
        role_quality_counts[role] = {
            'green': counts.get('green', 0),
            'yellow': counts.get('yellow', 0),
            'red': counts.get('red', 0),
        }

    quality_flags: list[str] = []
    if role_counts['support'] < 2:
        quality_flags.append('support_cluster_too_small')
    if role_counts['alternative'] < 1:
        quality_flags.append('alternative_route_states_missing')
    if role_counts['held_out'] < 1:
        quality_flags.append('held_out_route_states_missing')

    scope_mismatch_entry_ids: list[str] = []
    indistinct_alternative_entry_ids: list[str] = []
    packet_ids_by_role: dict[str, set[str]] = {role: set() for role in role_counts}
    for artifact in compilation.entries:
        packet_ids_by_role[artifact.role].add(artifact.packet_id)
        if artifact.role == 'support':
            if _scope_similarity(artifact.route_state.topic_scope, compilation.topic_scope) < 0.5:
                scope_mismatch_entry_ids.append(artifact.entry_id)
        elif artifact.role == 'alternative':
            if _scope_similarity(artifact.route_state.topic_scope, compilation.topic_scope) >= 0.8:
                indistinct_alternative_entry_ids.append(artifact.entry_id)

    if scope_mismatch_entry_ids:
        quality_flags.append('support_scope_drift')
    if indistinct_alternative_entry_ids:
        quality_flags.append('alternative_scope_not_distinct')

    reused_packet_ids_across_roles = sorted(
        packet_ids_by_role['support'] & packet_ids_by_role['held_out']
    )
    if reused_packet_ids_across_roles:
        quality_flags.append('held_out_packet_reused_in_support')

    if any(counts['red'] > 0 for counts in role_quality_counts.values()):
        quality_flags.append('red_route_state_present')
    elif any(counts['yellow'] > 0 for counts in role_quality_counts.values()):
        quality_flags.append('yellow_route_state_present')

    blocking_flags = {
        'support_cluster_too_small',
        'alternative_route_states_missing',
        'held_out_route_states_missing',
        'red_route_state_present',
    }
    has_blocking = any(flag in blocking_flags for flag in quality_flags)
    if has_blocking:
        quality_tier: QualityTier = 'red'
    elif quality_flags:
        quality_tier = 'yellow'
    else:
        quality_tier = 'green'

    return RouteStatePackageValidation(
        quality_tier=quality_tier,
        ready_for_replay=not has_blocking,
        quality_flags=quality_flags,
        role_counts=role_counts,
        role_quality_counts=role_quality_counts,
        support_route_state_ids=[artifact.route_state.route_state_id for artifact in compilation.entries if artifact.role == 'support'],
        alternative_route_state_ids=[artifact.route_state.route_state_id for artifact in compilation.entries if artifact.role == 'alternative'],
        held_out_route_state_ids=[artifact.route_state.route_state_id for artifact in compilation.entries if artifact.role == 'held_out'],
        scope_mismatch_entry_ids=scope_mismatch_entry_ids,
        indistinct_alternative_entry_ids=indistinct_alternative_entry_ids,
        reused_packet_ids_across_roles=reused_packet_ids_across_roles,
    )


def build_route_state_package_summary(
    compilation: RouteStatePackageCompilation,
    *,
    output_dir: str | Path | None = None,
    bundle_manifest: str | Path | None = None,
) -> dict[str, Any]:
    grouped = compilation.grouped_route_states()
    primary_route_state = compilation.primary_route_state()
    validation = validate_route_state_package(compilation)
    summary: dict[str, Any] = {
        'package_id': compilation.package_id,
        'topic_scope': compilation.topic_scope,
        'cutoff_year': compilation.cutoff_year,
        'entry_count': len(compilation.entries),
        'primary_route_state_id': primary_route_state.route_state_id if primary_route_state is not None else None,
        'support_route_state_count': len(grouped['support']),
        'alternative_route_state_count': len(grouped['alternative']),
        'held_out_route_state_count': len(grouped['held_out']),
        'validation_quality_tier': validation.quality_tier,
        'validation_ready_for_replay': validation.ready_for_replay,
        'validation_quality_flags': list(validation.quality_flags),
    }
    if output_dir is not None:
        summary['output_dir'] = str(_as_path(output_dir).resolve())
    if bundle_manifest is not None:
        summary['bundle_manifest'] = str(_as_path(bundle_manifest).resolve())
    return summary


__all__ = [
    'LoadedRouteStatePackage',
    'RouteStatePackageArtifact',
    'RouteStatePackageBundleEntryRef',
    'RouteStatePackageBundleManifest',
    'RouteStatePackageCompilation',
    'RouteStatePackageEntry',
    'RouteStatePackageManifest',
    'RouteStatePackageValidation',
    'build_route_state_package_summary',
    'compile_route_state_package',
    'load_route_state_package_bundle',
    'load_route_state_package_manifest',
    'validate_route_state_package',
    'write_route_state_package_bundle',
]
