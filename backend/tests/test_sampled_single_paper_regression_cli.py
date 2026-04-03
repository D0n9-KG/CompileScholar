from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from app.research_logic import (
    FixedRegressionSampledPaperResult,
    RandomExplorationSampledPaperResult,
    SampledPaperAvailabilityIssue,
    SampledSinglePaperIterationResult,
    write_sampled_l2_iteration_bundle,
)


BACKEND_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = BACKEND_DIR / 'scripts' / 'run_sampled_single_paper_l2_regression.py'


def _load_script_module():
    spec = importlib.util.spec_from_file_location('run_sampled_single_paper_l2_regression', SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SAMPLED_L2_CLI = _load_script_module()


def _iteration(iteration_label: str, *, fixed_tier: str, random_tier: str) -> SampledSinglePaperIterationResult:
    return SampledSinglePaperIterationResult(
        iteration_label=iteration_label,
        sampling_bundle_dir='tmp/phase7_corpus_sampling_baseline',
        sampling_bundle_manifest_ref='tmp/phase7_corpus_sampling_baseline/bundle_manifest.json',
        fixed_selected_count=2,
        random_selected_count=2,
        fixed_results=[
            FixedRegressionSampledPaperResult(
                corpus_paper_id='1001',
                display_title='Fixed Alpha',
                corpus_relative_ref='txt/1001.txt',
                preferred_source_path='C:/corpus/1001.txt',
                preferred_source_kind='txt',
                iteration_label=iteration_label,
                paper_id='doi:10.1000/1001',
                trace_id='trace:1001',
                source_path='C:/corpus/1001.txt',
                source_kind='txt',
                quality_report={'quality_tier': fixed_tier, 'gate_passed': fixed_tier != 'red'},
                trace_quality={'quality_tier': fixed_tier, 'audit_status': 'eligible'},
                citations={'refs': 1, 'cites_resolved': 1, 'cites_unresolved': 0},
                citation_semantic={'citation_acts': 1, 'citation_mentions': 1},
                llm={'purposes': 1, 'moves': 1, 'gate_passed': fixed_tier != 'red', 'quality_tier': fixed_tier},
            )
        ],
        random_results=[
            RandomExplorationSampledPaperResult(
                corpus_paper_id='2001',
                display_title='Random Beta',
                corpus_relative_ref='txt/2001.txt',
                preferred_source_path='C:/corpus/2001.txt',
                preferred_source_kind='txt',
                iteration_label=iteration_label,
                paper_id='doi:10.1000/2001',
                trace_id='trace:2001',
                source_path='C:/corpus/2001.txt',
                source_kind='txt',
                quality_report={'quality_tier': random_tier, 'gate_passed': random_tier != 'red'},
                trace_quality={'quality_tier': random_tier, 'audit_status': 'eligible'},
                citations={'refs': 1, 'cites_resolved': 1, 'cites_unresolved': 0},
                citation_semantic={'citation_acts': 1, 'citation_mentions': 1},
                llm={'purposes': 1, 'moves': 1, 'gate_passed': random_tier != 'red', 'quality_tier': random_tier},
            )
        ],
        availability_issues=[
            SampledPaperAvailabilityIssue(
                corpus_paper_id='2999',
                display_title='Missing Gamma',
                cohort='random_exploration',
                selection_mode='random_exploration',
                corpus_relative_ref='txt/2999.txt',
                preferred_source_path='C:/corpus/2999.txt',
                preferred_source_kind='txt',
                iteration_label=iteration_label,
                execution_status='source_missing',
                error_message='missing source',
                error_type='FileNotFoundError',
            )
        ],
    )


def test_run_sampled_single_paper_l2_regression_uses_default_output_dir_and_writes_baseline_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_output_dir = tmp_path / 'phase8-default'
    expected_report = tmp_path / 'phase8-report.md'
    monkeypatch.setattr(SAMPLED_L2_CLI, '_default_output_dir', lambda iteration_label: expected_output_dir / iteration_label)
    monkeypatch.setattr(SAMPLED_L2_CLI, '_default_docs_report', lambda: expected_report)
    monkeypatch.setattr(
        SAMPLED_L2_CLI,
        'run_sampled_single_paper_iteration',
        lambda sampling_bundle, iteration_label, artifacts_dir, allow_graph_write=False: _iteration(
            iteration_label,
            fixed_tier='red',
            random_tier='yellow',
        ),
    )

    summary = SAMPLED_L2_CLI.run_sampled_single_paper_l2_regression(
        SAMPLED_L2_CLI.build_parser().parse_args(['--sampling-bundle', str(tmp_path / 'phase7')])
    )

    report_text = expected_report.read_text(encoding='utf-8')
    comparison_summary = json.loads((Path(summary['output_dir']) / 'comparison_summary.json').read_text(encoding='utf-8'))

    assert Path(summary['output_dir']) == (expected_output_dir / 'baseline-cycle-01').resolve()
    assert summary['baseline_only'] is True
    assert '## Iteration Inputs' in report_text
    assert '## Fixed Regression Outcomes' in report_text
    assert '## Random Exploration Outcomes' in report_text
    assert '## Owner Buckets' in report_text
    assert '## Recommended L2 Queue' in report_text
    assert '## Execution Limits' in report_text
    assert 'baseline-only' in report_text
    assert comparison_summary['fixed_verdict_counts']['recurring_failure'] == 1
    assert comparison_summary['random_verdict_counts']['new_edge_case'] == 1


def test_run_sampled_single_paper_l2_regression_writes_comparison_mode_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_dir = tmp_path / 'previous-iteration'
    write_sampled_l2_iteration_bundle(previous_dir, iteration=_iteration('baseline-cycle-00', fixed_tier='green', random_tier='green'))
    monkeypatch.setattr(
        SAMPLED_L2_CLI,
        'run_sampled_single_paper_iteration',
        lambda sampling_bundle, iteration_label, artifacts_dir, allow_graph_write=False: _iteration(
            iteration_label,
            fixed_tier='red',
            random_tier='yellow',
        ),
    )

    output_dir = tmp_path / 'current-iteration'
    docs_report = tmp_path / 'comparison-report.md'
    summary = SAMPLED_L2_CLI.run_sampled_single_paper_l2_regression(
        SAMPLED_L2_CLI.build_parser().parse_args(
            [
                '--sampling-bundle',
                str(tmp_path / 'phase7'),
                '--output-dir',
                str(output_dir),
                '--iteration-label',
                'baseline-cycle-01',
                '--previous-iteration-dir',
                str(previous_dir),
                '--docs-report',
                str(docs_report),
            ]
        )
    )

    report_text = docs_report.read_text(encoding='utf-8')

    assert summary['baseline_only'] is False
    assert summary['fixed_verdict_counts']['new_regression'] == 1
    assert (output_dir / 'bundle_manifest.json').is_file()
    assert (output_dir / 'comparison_summary.json').is_file()
    assert (output_dir / 'comparison_inspection.json').is_file()
    assert 'compared against `baseline-cycle-00`' in report_text
