from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from app.research_logic import (
    DEFAULT_PHASE10_L1_SNAPSHOT_PATH,
    DEFAULT_PHASE10_PACKAGE_MANIFEST_PATH,
    load_historical_environment_snapshot,
    load_phase10_canonical_inputs,
    load_route_packet,
    prepare_phase10_runtime_bridge,
    resolve_phase10_runtime_paths,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE9_PACKET_PATH = REPO_ROOT / 'docs' / 'replay' / 'pilot_packets' / 'phase9-route-packet.json'
PHASE9_ASSEMBLY_MANIFEST_PATH = REPO_ROOT / 'docs' / 'replay' / 'pilot_packets' / 'phase9-assembly-manifest.json'


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


def test_phase10_runtime_paths_point_to_expected_manifest_and_snapshot_defaults() -> None:
    runtime_paths = resolve_phase10_runtime_paths(repo_root=REPO_ROOT)

    assert runtime_paths.package_manifest_path == (REPO_ROOT / DEFAULT_PHASE10_PACKAGE_MANIFEST_PATH)
    assert runtime_paths.l1_snapshot_path == (REPO_ROOT / DEFAULT_PHASE10_L1_SNAPSHOT_PATH)


def test_phase10_bridge_manifest_preserves_phase9_role_membership(tmp_path: Path) -> None:
    output_root = tmp_path / 'phase10_multi_paper_validation'
    bridge = prepare_phase10_runtime_bridge(
        packet_path=PHASE9_PACKET_PATH,
        assembly_manifest_path=PHASE9_ASSEMBLY_MANIFEST_PATH,
        output_root=output_root,
        built_at='2026-04-03T12:30:00Z',
        repo_root=REPO_ROOT,
    )

    assert bridge.package_manifest_path == (output_root / 'generated' / 'phase10-route-state-package-manifest.json')
    assert bridge.package_manifest.cutoff_year == 2021
    assert [entry.role for entry in bridge.package_manifest.entries] == ['support', 'alternative', 'held_out']

    generated_dir = bridge.package_manifest_path.parent
    support_entry, alternative_entry, held_out_entry = bridge.package_manifest.entries
    support_packet = load_route_packet(generated_dir / Path(support_entry.packet_path))
    alternative_packet = load_route_packet(generated_dir / Path(alternative_entry.packet_path))
    held_out_packet = load_route_packet(generated_dir / Path(held_out_entry.packet_path))

    assert [item.paper_id for item in support_packet.included_items] == ['1000', '1001', '1002', '1005', '1017']
    assert [item.paper_id for item in alternative_packet.included_items] == ['1007']
    assert [item.paper_id for item in held_out_packet.included_items] == ['1023']
    assert support_packet.cutoff_year == 2021
    assert alternative_packet.cutoff_year == 2021
    assert held_out_packet.cutoff_year == 2021


def test_phase10_bridge_snapshot_writes_expected_default_path_and_cutoff() -> None:
    runtime_paths = resolve_phase10_runtime_paths(repo_root=REPO_ROOT)
    output_root = runtime_paths.output_root
    shutil.rmtree(output_root, ignore_errors=True)

    try:
        bridge = prepare_phase10_runtime_bridge(
            packet_path=PHASE9_PACKET_PATH,
            assembly_manifest_path=PHASE9_ASSEMBLY_MANIFEST_PATH,
            built_at='2026-04-03T12:45:00Z',
            repo_root=REPO_ROOT,
        )

        assert bridge.package_manifest_path == runtime_paths.package_manifest_path
        assert bridge.l1_snapshot_path == runtime_paths.l1_snapshot_path
        assert bridge.package_manifest_path.is_file()
        assert bridge.l1_snapshot_path.is_file()

        snapshot = load_historical_environment_snapshot(bridge.l1_snapshot_path)
        assert snapshot.snapshot_id == runtime_paths.l1_snapshot_path.stem
        assert snapshot.cutoff_year == 2021
        assert snapshot.source_packet_id == 'phase9_comp_mech_2021_packet_01'
    finally:
        shutil.rmtree(output_root, ignore_errors=True)


def test_phase10_bridge_rejects_cutoff_mismatch(tmp_path: Path) -> None:
    packet_payload = json.loads(PHASE9_PACKET_PATH.read_text(encoding='utf-8'))
    packet_payload['cutoff_year'] = 2020
    mismatched_packet_path = _write_json(tmp_path / 'phase9-route-packet.json', packet_payload)

    with pytest.raises(ValueError, match='cutoff_year mismatch'):
        load_phase10_canonical_inputs(
            packet_path=mismatched_packet_path,
            assembly_manifest_path=PHASE9_ASSEMBLY_MANIFEST_PATH,
            repo_root=REPO_ROOT,
        )
