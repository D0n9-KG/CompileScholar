from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from app.research_logic import (
    DEFAULT_PHASE10_L1_SNAPSHOT_PATH,
    DEFAULT_PHASE10_PACKAGE_MANIFEST_PATH,
    load_historical_environment_snapshot,
    load_paper_logic_trace,
    load_phase10_canonical_inputs,
    load_route_packet,
    prepare_phase10_runtime_bridge,
    resolve_phase10_runtime_paths,
)
from app.research_logic.phase10_multi_paper_validation import run_phase10_package_and_replay


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

    assert list(bridge.role_packets['support'].paper_ids) == ['1000', '1001', '1002', '1005', '1017']
    assert list(bridge.role_packets['alternative'].paper_ids) == ['1007']
    assert list(bridge.role_packets['held_out'].paper_ids) == ['1023']
    assert [item.paper_id for item in support_packet.included_items] == [
        load_paper_logic_trace(path).paper_metadata.paper_id for path in bridge.role_packets['support'].trace_files
    ]
    assert [item.paper_id for item in alternative_packet.included_items] == [
        load_paper_logic_trace(path).paper_metadata.paper_id for path in bridge.role_packets['alternative'].trace_files
    ]
    assert [item.paper_id for item in held_out_packet.included_items] == [
        load_paper_logic_trace(path).paper_metadata.paper_id for path in bridge.role_packets['held_out'].trace_files
    ]
    assert support_packet.cutoff_year == 2021
    assert alternative_packet.cutoff_year == 2021
    assert held_out_packet.cutoff_year == 2021
    assert 'Canonical Phase 9 paper_ids: 1000, 1001, 1002, 1005, 1017' in (
        support_packet.compiler_hints.notes_for_route_state_compiler or ''
    )


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


def test_phase10_workflow_writes_prior_review_and_export_bundles(tmp_path: Path) -> None:
    runtime_paths = resolve_phase10_runtime_paths(repo_root=REPO_ROOT)
    shutil.rmtree(runtime_paths.output_root, ignore_errors=True)

    try:
        l1_snapshot_output_path = tmp_path / 'phase9-comp-mech-l1-snapshot.json'
        output_dir = tmp_path / 'phase10-workflow-run'
        result = run_phase10_package_and_replay(
            packet_path=PHASE9_PACKET_PATH,
            assembly_manifest_path=PHASE9_ASSEMBLY_MANIFEST_PATH,
            l1_snapshot_output_path=l1_snapshot_output_path,
            output_dir=output_dir,
            built_at='2026-04-03T13:00:00Z',
            repo_root=REPO_ROOT,
        )

        review_manifest_payload = json.loads((output_dir / 'prior_review_bundle' / 'bundle_manifest.json').read_text(encoding='utf-8'))
        comparison_summary_payload = json.loads((output_dir / 'comparison_summary.json').read_text(encoding='utf-8'))
        export_summary_payload = json.loads((output_dir / 'export_bundle' / 'export_summary.json').read_text(encoding='utf-8'))
        export_inspection_payload = json.loads((output_dir / 'export_bundle' / 'export_inspection.json').read_text(encoding='utf-8'))
        decision_episode_payload = json.loads((output_dir / 'export_bundle' / 'outputs' / 'decision_episode.json').read_text(encoding='utf-8'))

        assert result.summary['prior_review_bundle_manifest'] == str((output_dir / 'prior_review_bundle' / 'bundle_manifest.json').resolve())
        assert result.summary['export_summary_path'] == str((output_dir / 'export_bundle' / 'export_summary.json').resolve())
        assert result.summary['export_inspection_path'] == str((output_dir / 'export_bundle' / 'export_inspection.json').resolve())
        assert result.summary['comparison_summary_path'] == str((output_dir / 'comparison_summary.json').resolve())
        assert Path(comparison_summary_payload['baseline_replay_bundle']).name == 'replay_with_package'
        assert Path(comparison_summary_payload['baseline_export_bundle']).name == 'phase6_decision_episode_audit_export'
        assert set(export_summary_payload['selected_prior_ids']).issubset(set(review_manifest_payload['accepted_prior_ids']))
        assert decision_episode_payload['relevant_priors']['selected_prior_ids'] == export_summary_payload['selected_prior_ids']
        assert 'visible_input_refs' in export_inspection_payload['visibility_buckets']
        assert 'audit_only_refs' in export_inspection_payload['visibility_buckets']
        assert 'label_eval_only_refs' in export_inspection_payload['visibility_buckets']
        assert export_summary_payload['visibility_bucket_counts']['visible_input_refs'] >= 1
    finally:
        shutil.rmtree(runtime_paths.output_root, ignore_errors=True)


def test_phase10_cli_help_lists_required_arguments() -> None:
    script_path = REPO_ROOT / 'backend' / 'scripts' / 'run_phase10_multi_paper_validation.py'
    result = subprocess.run(
        [sys.executable, str(script_path), '--help'],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT / 'backend'),
    )

    assert result.returncode == 0
    assert '--packet' in result.stdout
    assert '--assembly-manifest' in result.stdout
    assert '--l1-snapshot-output' in result.stdout
    assert '--output-dir' in result.stdout


def test_phase10_cli_writes_package_and_replay_bundles(tmp_path: Path) -> None:
    runtime_paths = resolve_phase10_runtime_paths(repo_root=REPO_ROOT)
    shutil.rmtree(runtime_paths.output_root, ignore_errors=True)

    try:
        script_path = REPO_ROOT / 'backend' / 'scripts' / 'run_phase10_multi_paper_validation.py'
        l1_snapshot_output_path = tmp_path / 'phase9-comp-mech-l1-snapshot.json'
        output_dir = tmp_path / 'phase10-cli-run'

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                '--packet',
                str(PHASE9_PACKET_PATH),
                '--assembly-manifest',
                str(PHASE9_ASSEMBLY_MANIFEST_PATH),
                '--l1-snapshot-output',
                str(l1_snapshot_output_path),
                '--output-dir',
                str(output_dir),
                '--built-at',
                '2026-04-03T13:15:00Z',
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO_ROOT / 'backend'),
        )

        assert result.returncode == 0, result.stderr
        summary_payload = json.loads(result.stdout)
        replay_inspection_payload = json.loads((output_dir / 'replay_bundle' / 'replay_inspection.json').read_text(encoding='utf-8'))

        assert summary_payload['generated_package_manifest'] == str(runtime_paths.package_manifest_path.resolve())
        assert summary_payload['l1_snapshot_output'] == str(l1_snapshot_output_path.resolve())
        assert (output_dir / 'route_state_package' / 'bundle_manifest.json').is_file()
        assert (output_dir / 'route_state_package' / 'validation.json').is_file()
        assert (output_dir / 'replay_bundle' / 'replay_summary.json').is_file()
        assert (output_dir / 'replay_bundle' / 'replay_inspection.json').is_file()
        assert (output_dir / 'prior_review_bundle' / 'bundle_manifest.json').is_file()
        assert (output_dir / 'export_bundle' / 'export_summary.json').is_file()
        assert (output_dir / 'export_bundle' / 'export_inspection.json').is_file()
        assert (output_dir / 'comparison_summary.json').is_file()
        assert replay_inspection_payload['route_state_package_validation'] is not None
        assert replay_inspection_payload['route_state_package_validation']['quality_tier'] == summary_payload[
            'route_state_package_validation_quality_tier'
        ]
        assert 'route_state_package_validation' in replay_inspection_payload
        assert set(summary_payload['selected_prior_ids']).issubset(set(summary_payload['accepted_prior_ids']))
        assert 'visibility_bucket_counts' in summary_payload
        assert summary_payload['comparison_summary_path'] == str((output_dir / 'comparison_summary.json').resolve())
    finally:
        shutil.rmtree(runtime_paths.output_root, ignore_errors=True)
