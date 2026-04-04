from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
SCRIPT_PATH = BACKEND_DIR / 'scripts' / 'run_phase10_multi_paper_validation.py'
PHASE9_PACKET_PATH = REPO_ROOT / 'docs' / 'replay' / 'pilot_packets' / 'phase9-route-packet.json'
PHASE9_ASSEMBLY_MANIFEST_PATH = REPO_ROOT / 'docs' / 'replay' / 'pilot_packets' / 'phase9-assembly-manifest.json'
BASELINE_REPLAY_BUNDLE = REPO_ROOT / 'tmp' / 'phase3_route_state_package' / 'replay_with_package'
BASELINE_EXPORT_BUNDLE = REPO_ROOT / 'tmp' / 'phase6_decision_episode_audit_export'


def _assert_baseline_bundles_exist() -> None:
    assert BASELINE_REPLAY_BUNDLE.is_dir(), f'missing baseline replay bundle fixture: {BASELINE_REPLAY_BUNDLE}'
    assert BASELINE_EXPORT_BUNDLE.is_dir(), f'missing baseline export bundle fixture: {BASELINE_EXPORT_BUNDLE}'


def test_phase10_cli_help_lists_comparison_arguments() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), '--help'],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(BACKEND_DIR),
    )

    assert result.returncode == 0
    assert '--baseline-replay-bundle' in result.stdout
    assert '--baseline-export-bundle' in result.stdout
    assert '--report-md' in result.stdout
    assert '--reviewer' in result.stdout


def test_phase10_cli_writes_comparison_summary_and_report(tmp_path: Path) -> None:
    _assert_baseline_bundles_exist()
    output_dir = tmp_path / 'phase10-cli-run'
    report_path = tmp_path / 'phase10-validation-report.md'
    l1_snapshot_output = tmp_path / 'phase10-l1-snapshot.json'

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            '--packet',
            str(PHASE9_PACKET_PATH),
            '--assembly-manifest',
            str(PHASE9_ASSEMBLY_MANIFEST_PATH),
            '--l1-snapshot-output',
            str(l1_snapshot_output),
            '--output-dir',
            str(output_dir),
            '--baseline-replay-bundle',
            str(BASELINE_REPLAY_BUNDLE),
            '--baseline-export-bundle',
            str(BASELINE_EXPORT_BUNDLE),
            '--report-md',
            str(report_path),
            '--built-at',
            '2026-04-03T14:15:00Z',
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(BACKEND_DIR),
    )

    assert result.returncode == 0, result.stderr

    summary_payload = json.loads(result.stdout)
    comparison_payload = json.loads((output_dir / 'comparison_summary.json').read_text(encoding='utf-8'))
    report_text = report_path.read_text(encoding='utf-8')

    assert summary_payload['comparison_summary_path'] == str((output_dir / 'comparison_summary.json').resolve())
    assert summary_payload['baseline_replay_bundle'] == str(BASELINE_REPLAY_BUNDLE.resolve())
    assert summary_payload['baseline_export_bundle'] == str(BASELINE_EXPORT_BUNDLE.resolve())
    assert summary_payload['report_markdown_path'] == str(report_path.resolve())
    assert comparison_payload['baseline_replay_bundle'] == str(BASELINE_REPLAY_BUNDLE.resolve())
    assert comparison_payload['baseline_export_bundle'] == str(BASELINE_EXPORT_BUNDLE.resolve())
    assert 'package' in comparison_payload
    assert 'replay' in comparison_payload
    assert 'prior_review' in comparison_payload
    assert 'export' in comparison_payload
    assert set(comparison_payload['blocker_queue']) == {
        'package_validation',
        'replay',
        'prior_induction',
        'export',
    }
    assert report_path.is_file()
    assert '# Phase 10 Multi-Paper Validation Report' in report_text
    assert '## Blocker Queue' in report_text


def test_phase10_cli_accepts_backend_relative_paths(tmp_path: Path) -> None:
    _assert_baseline_bundles_exist()
    output_dir = tmp_path / 'phase10-cli-relative-run'
    report_path = tmp_path / 'phase10-relative-report.md'
    l1_snapshot_output = tmp_path / 'phase10-relative-l1-snapshot.json'

    def rel(path: Path) -> str:
        return os.path.relpath(path, BACKEND_DIR)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            '--packet',
            rel(PHASE9_PACKET_PATH),
            '--assembly-manifest',
            rel(PHASE9_ASSEMBLY_MANIFEST_PATH),
            '--l1-snapshot-output',
            rel(l1_snapshot_output),
            '--output-dir',
            rel(output_dir),
            '--baseline-replay-bundle',
            rel(BASELINE_REPLAY_BUNDLE),
            '--baseline-export-bundle',
            rel(BASELINE_EXPORT_BUNDLE),
            '--report-md',
            rel(report_path),
            '--built-at',
            '2026-04-03T14:25:00Z',
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(BACKEND_DIR),
    )

    assert result.returncode == 0, result.stderr

    summary_payload = json.loads(result.stdout)

    assert summary_payload['route_state_package_bundle_manifest'] == str((output_dir / 'route_state_package' / 'bundle_manifest.json').resolve())
    assert summary_payload['replay_summary_path'] == str((output_dir / 'replay_bundle' / 'replay_summary.json').resolve())
    assert summary_payload['comparison_summary_path'] == str((output_dir / 'comparison_summary.json').resolve())
    assert summary_payload['report_markdown_path'] == str(report_path.resolve())


def test_phase10_cli_threads_reviewer_ids_into_replay_summary(tmp_path: Path) -> None:
    _assert_baseline_bundles_exist()
    output_dir = tmp_path / 'phase10-cli-reviewer-run'
    l1_snapshot_output = tmp_path / 'phase10-cli-reviewer-l1-snapshot.json'

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            '--packet',
            str(PHASE9_PACKET_PATH),
            '--assembly-manifest',
            str(PHASE9_ASSEMBLY_MANIFEST_PATH),
            '--l1-snapshot-output',
            str(l1_snapshot_output),
            '--output-dir',
            str(output_dir),
            '--baseline-replay-bundle',
            str(BASELINE_REPLAY_BUNDLE),
            '--baseline-export-bundle',
            str(BASELINE_EXPORT_BUNDLE),
            '--reviewer',
            'reviewer-1',
            '--built-at',
            '2026-04-04T04:35:00Z',
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(BACKEND_DIR),
    )

    assert result.returncode == 0, result.stderr

    summary_payload = json.loads(result.stdout)
    replay_summary_payload = json.loads((output_dir / 'replay_bundle' / 'replay_summary.json').read_text(encoding='utf-8'))

    assert summary_payload['reviewer_ids'] == ['reviewer-1']
    assert replay_summary_payload['reviewer_ids'] == ['reviewer-1']
    assert 'reviewer_missing' not in replay_summary_payload['quality_flags']
