from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Literal

from .bounded_packet_audit import (
    BoundedPacketAssemblyManifest,
    audit_bounded_packet_assembly,
    load_bounded_packet_assembly_manifest,
)
from .historical_environment import HistoricalEnvironmentSnapshot, build_historical_environment_snapshot, build_l1_snapshot_ref, write_historical_environment_snapshot
from .models import RoutePacket
from .replay_io import ensure_packet_trace_coverage, load_paper_logic_traces, load_route_packet
from .route_state_package import RouteStatePackageEntry, RouteStatePackageManifest

Phase10Role = Literal['support', 'alternative', 'held_out']

DEFAULT_PHASE10_PACKET_PATH = Path('docs/replay/pilot_packets/phase9-route-packet.json')
DEFAULT_PHASE10_ASSEMBLY_MANIFEST_PATH = Path('docs/replay/pilot_packets/phase9-assembly-manifest.json')
DEFAULT_PHASE10_OUTPUT_ROOT = Path('tmp/phase10_multi_paper_validation')
DEFAULT_PHASE10_PACKAGE_MANIFEST_PATH = DEFAULT_PHASE10_OUTPUT_ROOT / 'generated' / 'phase10-route-state-package-manifest.json'
DEFAULT_PHASE10_L1_SNAPSHOT_PATH = DEFAULT_PHASE10_OUTPUT_ROOT / 'shared' / 'phase9-comp-mech-l1-snapshot.json'


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
    role: Phase10Role
    packet_path: Path
    route_state_id: str
    paper_ids: tuple[str, ...]
    trace_files: tuple[Path, ...]


@dataclass(frozen=True)
class Phase10RuntimeBridge:
    canonical_inputs: Phase10CanonicalInputs
    runtime_paths: Phase10RuntimePaths
    trace_sources: tuple[Phase10TraceSource, ...]
    l1_snapshot: HistoricalEnvironmentSnapshot
    l1_snapshot_path: Path
    package_manifest: RouteStatePackageManifest
    package_manifest_path: Path
    role_packets: dict[Phase10Role, Phase10RolePacketArtifact]


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


def _relative_manifest_path(target: Path, *, base_dir: Path) -> str:
    try:
        return os.path.relpath(target, base_dir).replace('\\', '/')
    except ValueError:
        return str(target)


def _role_slug(role: Phase10Role) -> str:
    return role.replace('_', '-')


def _package_built_at(route_packet: RoutePacket, assembly_manifest: BoundedPacketAssemblyManifest, built_at: str | None) -> str:
    return built_at or assembly_manifest.built_at or route_packet.built_at


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

            seen_paper_ids.add(member.paper_id)
            collected.append(
                Phase10TraceSource(
                    role=role,
                    paper_id=member.paper_id,
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


def _build_role_subset_packet(
    canonical_inputs: Phase10CanonicalInputs,
    *,
    role: Phase10Role,
    paper_ids: list[str],
    built_at: str,
    snapshot: HistoricalEnvironmentSnapshot,
) -> RoutePacket:
    included_lookup = {item.paper_id: item for item in canonical_inputs.route_packet.included_items}
    missing_paper_ids = [paper_id for paper_id in paper_ids if paper_id not in included_lookup]
    if missing_paper_ids:
        raise ValueError(f'Phase 10 runtime subset contains paper_ids missing from the canonical packet: {", ".join(missing_paper_ids)}')

    subset_items = [included_lookup[paper_id] for paper_id in paper_ids]
    subset_role_counts = _subset_role_counts(canonical_inputs.route_packet, paper_ids=set(paper_ids))

    subset_payload = canonical_inputs.route_packet.model_dump(mode='json', exclude_none=True)
    subset_payload['packet_id'] = f'phase10-{_role_slug(role)}-{canonical_inputs.route_packet.packet_id}'
    subset_payload['built_at'] = built_at
    subset_payload['l1_snapshot_ref'] = build_l1_snapshot_ref(snapshot).model_dump(mode='json', exclude_none=True)
    subset_payload['included_items'] = [item.model_dump(mode='json', exclude_none=True) for item in subset_items]
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
            f'{notes} Runtime subset generated from the committed Phase 9 assembly manifest for the {role} role.'
            if notes
            else f'Runtime subset generated from the committed Phase 9 assembly manifest for the {role} role.'
        ),
    }

    return RoutePacket.model_validate(subset_payload)


def _write_phase10_role_packets(
    canonical_inputs: Phase10CanonicalInputs,
    *,
    trace_sources: tuple[Phase10TraceSource, ...],
    runtime_paths: Phase10RuntimePaths,
    snapshot: HistoricalEnvironmentSnapshot,
    built_at: str,
) -> dict[Phase10Role, Phase10RolePacketArtifact]:
    trace_lookup = {source.paper_id: source for source in trace_sources}
    role_packets: dict[Phase10Role, Phase10RolePacketArtifact] = {}

    for role, group in canonical_inputs.assembly_manifest.role_groups().items():
        paper_ids = tuple(group.paper_ids())
        trace_files = tuple(trace_lookup[paper_id].resolved_path for paper_id in paper_ids)
        role_packet = _build_role_subset_packet(
            canonical_inputs,
            role=role,
            paper_ids=list(paper_ids),
            built_at=built_at,
            snapshot=snapshot,
        )
        packet_path = runtime_paths.generated_packets_dir / f'phase10-{_role_slug(role)}-route-packet.json'
        _write_json(packet_path, role_packet.model_dump(mode='json', exclude_none=True))
        role_packets[role] = Phase10RolePacketArtifact(
            role=role,
            packet_path=packet_path,
            route_state_id=f'phase10-{_role_slug(role)}-route-state',
            paper_ids=paper_ids,
            trace_files=trace_files,
        )

    return role_packets


def _build_phase10_package_manifest(
    canonical_inputs: Phase10CanonicalInputs,
    *,
    runtime_paths: Phase10RuntimePaths,
    snapshot_path: Path,
    role_packets: dict[Phase10Role, Phase10RolePacketArtifact],
    built_at: str,
) -> RouteStatePackageManifest:
    role_groups = canonical_inputs.assembly_manifest.role_groups()
    entries = [
        RouteStatePackageEntry(
            entry_id=f'phase10-{_role_slug(role)}',
            role=role,
            packet_path=_relative_manifest_path(role_packets[role].packet_path, base_dir=runtime_paths.generated_dir),
            trace_files=[
                _relative_manifest_path(trace_path, base_dir=runtime_paths.generated_dir)
                for trace_path in role_packets[role].trace_files
            ],
            l1_snapshot_path=_relative_manifest_path(snapshot_path, base_dir=runtime_paths.generated_dir),
            route_state_id=role_packets[role].route_state_id,
            built_at=built_at,
            notes=role_groups[role].group_reason,
        )
        for role in ('support', 'alternative', 'held_out')
    ]
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


__all__ = [
    'DEFAULT_PHASE10_ASSEMBLY_MANIFEST_PATH',
    'DEFAULT_PHASE10_L1_SNAPSHOT_PATH',
    'DEFAULT_PHASE10_OUTPUT_ROOT',
    'DEFAULT_PHASE10_PACKAGE_MANIFEST_PATH',
    'DEFAULT_PHASE10_PACKET_PATH',
    'Phase10CanonicalInputs',
    'Phase10RolePacketArtifact',
    'Phase10RuntimeBridge',
    'Phase10RuntimePaths',
    'Phase10TraceSource',
    'build_phase10_l1_snapshot',
    'collect_phase10_trace_sources',
    'load_phase10_canonical_inputs',
    'prepare_phase10_runtime_bridge',
    'resolve_phase10_runtime_paths',
]
