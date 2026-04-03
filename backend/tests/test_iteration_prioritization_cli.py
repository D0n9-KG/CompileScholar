from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / 'backend' / 'scripts' / 'run_phase11_iteration_prioritization.py'
PHASE8_SUMMARY_PATH = REPO_ROOT / 'tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/comparison_summary.json'
PHASE8_INSPECTION_PATH = REPO_ROOT / 'tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/comparison_inspection.json'
PHASE10_VERIFICATION_PATH = REPO_ROOT / '.planning/phases/10-multi-paper-l3-and-l4-validation/10-VERIFICATION.md'
PHASE10_REPORT_PATH = REPO_ROOT / 'docs/replay/reports/phase10-multi-paper-l3-l4-validation.md'


def _phase10_summary_fixture(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                'packet_id': 'phase9_comp_mech_2021_packet_01',
                'cutoff_year': 2021,
                'current_recommendation': 'packet_construction',
                'package': {'current': {'quality_tier': 'red'}},
                'replay': {'delta': {'failure_counts_by_layer_delta': {'l2': 0, 'l3_l4': 2}}},
                'prior_review': {'current': {'prior_candidate_count': 0}},
                'export': {'current': {'quality_tier': 'yellow'}},
                'blocker_queue': {
                    'package_validation': [
                        {
                            'code': 'support_cluster_too_small',
                            'message': 'Package validation reports `support_cluster_too_small`.',
                            'current': True,
                            'baseline': False,
                            'vs_baseline': 'new',
                        }
                    ],
                    'replay': [
                        {
                            'code': 'failure_stage:decision_prior_card',
                            'message': 'Replay keeps decision prior pressure visible.',
                            'current': True,
                            'baseline': False,
                            'vs_baseline': 'higher',
                        }
                    ],
                },
                'source_artifacts': {'current_replay_inspection': 'tmp/replay/replay_inspection.json'},
                'notes': {'source': 'fixture'},
            },
            ensure_ascii=False,
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )
    return path


def test_phase11_cli_help_lists_required_arguments() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), '--help'],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT / 'backend'),
    )

    assert result.returncode == 0
    assert '--phase8-summary' in result.stdout
    assert '--phase8-inspection' in result.stdout
    assert '--phase10-summary' in result.stdout
    assert '--phase10-verification' in result.stdout
    assert '--phase10-report' in result.stdout
    assert '--output-dir' in result.stdout
    assert '--report-md' in result.stdout


def test_phase11_cli_fallback_output_writes_bundle_manifest_and_summary_files(tmp_path: Path) -> None:
    output_dir = tmp_path / 'phase11-output'
    report_md = tmp_path / 'phase11-report.md'

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            '--phase8-summary',
            str(PHASE8_SUMMARY_PATH),
            '--phase8-inspection',
            str(PHASE8_INSPECTION_PATH),
            '--phase10-verification',
            str(PHASE10_VERIFICATION_PATH),
            '--phase10-report',
            str(PHASE10_REPORT_PATH),
            '--output-dir',
            str(output_dir),
            '--report-md',
            str(report_md),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT / 'backend'),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload['fallback_used'] is True
    assert payload['primary_recommendation'] == 'packet_construction'
    assert payload['source_refs']['phase10_verification_path'] == str(PHASE10_VERIFICATION_PATH.resolve())
    assert payload['source_refs']['phase10_report_path'] == str(PHASE10_REPORT_PATH.resolve())
    assert (output_dir / 'bundle_manifest.json').is_file()
    assert (output_dir / 'outputs' / 'prioritization_summary.json').is_file()
    assert (output_dir / 'outputs' / 'prioritization_inspection.json').is_file()


def test_phase11_cli_output_records_supplemental_phase10_provenance_when_json_is_present(tmp_path: Path) -> None:
    output_dir = tmp_path / 'phase11-output'
    report_md = tmp_path / 'phase11-report.md'
    phase10_summary_path = _phase10_summary_fixture(tmp_path / 'phase10' / 'comparison_summary.json')

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            '--phase8-summary',
            str(PHASE8_SUMMARY_PATH),
            '--phase8-inspection',
            str(PHASE8_INSPECTION_PATH),
            '--phase10-summary',
            str(phase10_summary_path),
            '--phase10-verification',
            str(PHASE10_VERIFICATION_PATH),
            '--phase10-report',
            str(PHASE10_REPORT_PATH),
            '--output-dir',
            str(output_dir),
            '--report-md',
            str(report_md),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT / 'backend'),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload['fallback_used'] is False
    assert payload['source_refs']['phase10_summary_path'] == str(phase10_summary_path.resolve())
    assert payload['source_refs']['phase10_verification_path'] == str(PHASE10_VERIFICATION_PATH.resolve())
    assert payload['source_refs']['phase10_report_path'] == str(PHASE10_REPORT_PATH.resolve())
