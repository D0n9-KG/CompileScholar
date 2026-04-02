from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
SCRIPT_PATH = BACKEND_DIR / 'scripts' / 'export_decision_episode_pilot.py'
PHASE5_ROOT = REPO_ROOT / 'tmp' / 'phase5_multi_route_prior_induction'
REPLAY_BUNDLE = PHASE5_ROOT / 'replay_bundle'
REVIEW_BUNDLE = PHASE5_ROOT / 'review_bundle'


def _load_script_module():
    spec = importlib.util.spec_from_file_location('export_decision_episode_pilot', SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPORT_CLI = _load_script_module()


def _assert_phase5_bundles_exist() -> None:
    assert REPLAY_BUNDLE.is_dir(), f'missing replay bundle fixture: {REPLAY_BUNDLE}'
    assert REVIEW_BUNDLE.is_dir(), f'missing review bundle fixture: {REVIEW_BUNDLE}'


def test_build_parser_defaults_output_dir_to_phase6_export_tmp() -> None:
    parser = EXPORT_CLI.build_parser()

    args = parser.parse_args(
        [
            '--replay-bundle',
            str(REPLAY_BUNDLE),
            '--prior-review-bundle',
            str(REVIEW_BUNDLE),
        ]
    )

    assert Path(args.output_dir) == EXPORT_CLI._default_output_dir()


def test_run_export_decision_episode_pilot_creates_dedicated_bundle_from_phase5_inputs(tmp_path: Path) -> None:
    _assert_phase5_bundles_exist()
    replay_manifest_path = REPLAY_BUNDLE / 'bundle_manifest.json'
    review_manifest_path = REVIEW_BUNDLE / 'bundle_manifest.json'
    replay_manifest_before = replay_manifest_path.read_text(encoding='utf-8')
    review_manifest_before = review_manifest_path.read_text(encoding='utf-8')
    output_dir = tmp_path / 'phase6-export'

    summary = EXPORT_CLI.run_export_decision_episode_pilot(
        EXPORT_CLI.build_parser().parse_args(
            [
                '--replay-bundle',
                str(REPLAY_BUNDLE),
                '--prior-review-bundle',
                str(REVIEW_BUNDLE),
                '--output-dir',
                str(output_dir),
            ]
        )
    )

    manifest_payload = json.loads((output_dir / 'bundle_manifest.json').read_text(encoding='utf-8'))
    export_summary_payload = json.loads((output_dir / 'export_summary.json').read_text(encoding='utf-8'))
    inspection_payload = json.loads((output_dir / 'export_inspection.json').read_text(encoding='utf-8'))
    decision_episode_payload = json.loads((output_dir / 'outputs' / 'decision_episode.json').read_text(encoding='utf-8'))

    assert summary['output_dir'] == str(output_dir.resolve())
    assert summary['bundle_manifest'] == str((output_dir / 'bundle_manifest.json').resolve())
    assert summary['export_summary'] == str((output_dir / 'export_summary.json').resolve())
    assert summary['export_inspection'] == str((output_dir / 'export_inspection.json').resolve())
    assert manifest_payload['files']['decision_episode'] == 'outputs/decision_episode.json'
    assert manifest_payload['source_replay_bundle_refs']['manifest_ref'] == str(replay_manifest_path.resolve())
    assert manifest_payload['source_review_bundle_refs']['manifest_ref'] == str(review_manifest_path.resolve())
    assert export_summary_payload['audit_posture'] == 'audit_grade_pilot'
    assert decision_episode_payload['hindsight_outcome']['input_visible'] is False
    assert 'visible_input_refs' in inspection_payload['visibility_buckets']
    assert 'audit_only_refs' in inspection_payload['visibility_buckets']
    assert 'label_eval_only_refs' in inspection_payload['visibility_buckets']
    assert replay_manifest_before == replay_manifest_path.read_text(encoding='utf-8')
    assert review_manifest_before == review_manifest_path.read_text(encoding='utf-8')


def test_run_export_decision_episode_pilot_preserves_empty_accepted_prior_truth_from_phase5_review(tmp_path: Path) -> None:
    _assert_phase5_bundles_exist()
    output_dir = tmp_path / 'phase6-export'

    summary = EXPORT_CLI.run_export_decision_episode_pilot(
        EXPORT_CLI.build_parser().parse_args(
            [
                '--replay-bundle',
                str(REPLAY_BUNDLE),
                '--prior-review-bundle',
                str(REVIEW_BUNDLE),
                '--output-dir',
                str(output_dir),
            ]
        )
    )

    export_summary_payload = json.loads((output_dir / 'export_summary.json').read_text(encoding='utf-8'))
    decision_episode_payload = json.loads((output_dir / 'outputs' / 'decision_episode.json').read_text(encoding='utf-8'))

    assert summary['accepted_prior_count'] == 0
    assert summary['accepted_anti_pattern_count'] == 5
    assert export_summary_payload['accepted_prior_ids'] == []
    assert export_summary_payload['selected_prior_ids'] == []
    assert len(export_summary_payload['accepted_anti_pattern_ids']) == 5
    assert set(export_summary_payload['selected_antipattern_ids']).issubset(
        set(export_summary_payload['accepted_anti_pattern_ids'])
    )
    assert decision_episode_payload['relevant_priors']['selected_prior_ids'] == []


def test_export_decision_episode_pilot_cli_outputs_summary_and_reports_missing_bundle(tmp_path: Path) -> None:
    _assert_phase5_bundles_exist()
    output_dir = tmp_path / 'phase6-export'

    success = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            '--replay-bundle',
            str(REPLAY_BUNDLE),
            '--prior-review-bundle',
            str(REVIEW_BUNDLE),
            '--output-dir',
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(BACKEND_DIR),
    )

    assert success.returncode == 0, success.stderr
    success_payload = json.loads(success.stdout)

    assert success_payload['output_dir'] == str(output_dir.resolve())
    assert success_payload['exported_episode_id'].startswith('episode:')
    assert success_payload['accepted_prior_count'] == 0
    assert success_payload['accepted_anti_pattern_count'] == 5
    assert success_payload['visibility_bucket_counts']['visible_input_refs'] >= 1

    missing = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            '--replay-bundle',
            str(tmp_path / 'missing-replay-bundle'),
            '--prior-review-bundle',
            str(REVIEW_BUNDLE),
            '--output-dir',
            str(tmp_path / 'should-not-exist'),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(BACKEND_DIR),
    )

    assert missing.returncode == 1
    missing_payload = json.loads(missing.stderr)
    assert 'error' in missing_payload
    assert 'replay bundle' in missing_payload['error']
