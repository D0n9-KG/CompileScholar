from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
SCRIPT_PATH = BACKEND_DIR / 'scripts' / 'run_phase13_cycle_refinement.py'


def _load_phase13_module():
    spec = importlib.util.spec_from_file_location('run_phase13_cycle_refinement', SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase13_cli_help_lists_required_flags() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), '--help'],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(BACKEND_DIR),
    )

    assert result.returncode == 0
    assert '--iteration-label' in result.stdout
    assert '--output-root' in result.stdout
    assert '--baseline-replay-bundle' in result.stdout
    assert '--baseline-export-bundle' in result.stdout


def test_phase13_defaults_lock_cycle1_baseline_and_iteration_layout() -> None:
    phase13_script = _load_phase13_module()

    resolved_paths = phase13_script.resolve_phase13_paths(iteration_label='iter-01')

    assert str(resolved_paths.baseline_replay_bundle).endswith('tmp\\phase12_direct_fix_cycle\\cycle1\\replay_bundle')
    assert str(resolved_paths.baseline_export_bundle).endswith('tmp\\phase12_direct_fix_cycle\\cycle1\\export_bundle')
    assert str(resolved_paths.output_dir).endswith('tmp\\phase13_cycle2_refine\\iter-01')


def test_phase13_cli_smoke_run_writes_iteration_scoped_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / 'phase13-runs'

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            '--iteration-label',
            'dev-check',
            '--output-root',
            str(output_root),
            '--built-at',
            '2026-04-04T04:15:00Z',
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(BACKEND_DIR),
    )

    assert result.returncode == 0, result.stderr

    summary_payload = json.loads(result.stdout)
    iteration_dir = output_root / 'dev-check'

    assert summary_payload['iteration_label'] == 'dev-check'
    assert summary_payload['output_dir'] == str(iteration_dir.resolve())
    assert summary_payload['baseline_replay_bundle'].endswith('tmp\\phase12_direct_fix_cycle\\cycle1\\replay_bundle')
    assert summary_payload['baseline_export_bundle'].endswith('tmp\\phase12_direct_fix_cycle\\cycle1\\export_bundle')
    assert 'comparison_summary_path' in summary_payload
    assert 'report_markdown_path' in summary_payload
    assert (iteration_dir / 'comparison_summary.json').is_file()
    assert (iteration_dir / 'replay_bundle' / 'replay_summary.json').is_file()
