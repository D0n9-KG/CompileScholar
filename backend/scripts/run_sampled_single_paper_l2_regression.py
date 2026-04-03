from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script: `python scripts/run_sampled_single_paper_l2_regression.py ...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.research_logic import (  # noqa: E402
    build_sampled_l2_comparison_summary,
    build_sampled_l2_iteration_summary,
    compare_sampled_l2_iterations,
    load_sampled_l2_iteration_bundle,
    run_sampled_single_paper_iteration,
    write_sampled_l2_comparison_bundle,
    write_sampled_l2_iteration_bundle,
)


def _default_sampling_bundle() -> Path:
    return Path(__file__).resolve().parents[2] / 'tmp' / 'phase7_corpus_sampling_baseline'


def _default_output_dir(iteration_label: str) -> Path:
    return Path(__file__).resolve().parents[2] / 'tmp' / 'phase8_sampled_single_paper_l2' / iteration_label


def _default_docs_report() -> Path:
    return Path(__file__).resolve().parents[2] / 'docs' / 'replay' / 'reports' / 'phase8-sampled-single-paper-l2-regression-baseline.md'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run Phase 8 sampled single-paper L2 regression, compare iterations, and emit an auditable report.'
    )
    parser.add_argument(
        '--sampling-bundle',
        default=str(_default_sampling_bundle()),
        help='Phase 7 sampling bundle directory used as the fixed/random source of truth.',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Directory where the Phase 8 runtime bundle will be written. Defaults to tmp/phase8_sampled_single_paper_l2/<iteration_label>.',
    )
    parser.add_argument(
        '--iteration-label',
        default='baseline-cycle-01',
        help='Stable label used for the Phase 8 iteration directory and report metadata.',
    )
    parser.add_argument(
        '--previous-iteration-dir',
        default=None,
        help='Optional prior Phase 8 iteration bundle directory used for comparison.',
    )
    parser.add_argument(
        '--docs-report',
        default=str(_default_docs_report()),
        help='Markdown report path written for the operator-facing Phase 8 summary.',
    )
    parser.add_argument(
        '--allow-graph-write',
        action='store_true',
        help='Allow sampled-paper evaluation to write graph updates while it runs.',
    )
    return parser


def _render_verdict_lines(counts: dict[str, int]) -> list[str]:
    return [f'- `{name}`: {value}' for name, value in counts.items()]


def _render_owner_queue(owner_buckets: list[dict[str, object]]) -> list[str]:
    if not owner_buckets:
        return ['1. No non-green owner buckets were identified in this iteration.']
    lines: list[str] = []
    for index, bucket in enumerate(owner_buckets[:5], start=1):
        exemplars = ', '.join(str(item) for item in bucket.get('exemplar_corpus_paper_ids') or [])
        lines.append(
            f"{index}. `{bucket['bucket']}` - fixed={bucket['fixed_count']}, random={bucket['random_count']}, exemplars={exemplars or 'none'}"
        )
    return lines


def _render_report(
    *,
    iteration_summary: dict[str, object],
    comparison_summary: dict[str, object],
    output_dir: Path,
    docs_report_path: Path,
    allow_graph_write: bool,
) -> str:
    fixed_counts = dict(comparison_summary.get('fixed_verdict_counts') or {})
    random_counts = dict(comparison_summary.get('random_verdict_counts') or {})
    owner_buckets = list(comparison_summary.get('owner_buckets') or [])
    baseline_only = bool(comparison_summary.get('baseline_only'))
    previous_iteration_label = str(comparison_summary.get('previous_iteration_label') or '').strip() or None

    lines = [
        '# Phase 8 Sampled Single-Paper L2 Regression',
        '',
        f"Built from runtime bundle `{output_dir}` and written to `{docs_report_path}`.",
        '',
        '## Iteration Inputs',
        '',
        f"- Iteration label: `{iteration_summary['iteration_label']}`",
        f"- Sampling bundle: `{iteration_summary['sampling_bundle_dir']}`",
        f"- Sampling manifest: `{iteration_summary['sampling_bundle_manifest_ref']}`",
        f"- Output directory: `{output_dir}`",
        f"- Comparison mode: {'baseline-only' if baseline_only else f'compared against `{previous_iteration_label}`'}",
        f"- Graph writes enabled: `{str(bool(allow_graph_write)).lower()}`",
        '',
        '## Fixed Regression Outcomes',
        '',
        ('This run is baseline-only, so fixed-set outcomes are classified against the current baseline surface.'
         if baseline_only
         else f"This run compares the fixed regression set against `{previous_iteration_label}`."),
        f"- Fixed executed count: `{len([]) + int(iteration_summary['fixed_selected_count']) - int(fixed_counts.get('availability_only', 0))}`",
        *(_render_verdict_lines(fixed_counts)),
        '',
        '## Random Exploration Outcomes',
        '',
        ('This run is baseline-only, so random failures are treated as new edge cases unless they are availability-only.'
         if baseline_only
         else f"This run compares the random exploration set against `{previous_iteration_label}`."),
        f"- Random executed count: `{len([]) + int(iteration_summary['random_selected_count']) - int(random_counts.get('availability_only', 0))}`",
        *(_render_verdict_lines(random_counts)),
        '',
        '## Owner Buckets',
        '',
    ]
    if owner_buckets:
        lines.extend(
            [
                f"- `{bucket['bucket']}`: fixed={bucket['fixed_count']}, random={bucket['random_count']}, exemplars={', '.join(bucket.get('exemplar_corpus_paper_ids') or []) or 'none'}"
                for bucket in owner_buckets
            ]
        )
    else:
        lines.append('- None - no non-green executed rows produced owner-bucket signals.')

    lines.extend(
        [
            '',
            '## Recommended L2 Queue',
            '',
            *_render_owner_queue(owner_buckets),
            '',
            '## Execution Limits',
            '',
            f"- Availability-only issues remained separate from regression counts: `{iteration_summary['availability_issue_count']}` issue(s).",
            '- This report only covers sampled single-paper L2 behavior; it does not make packet-level L3/L4 claims.',
            '- Source-path availability still depends on the Phase 7 bundle and the current machine being able to reach each preferred source path.',
            f"- This run {'did not compare against an earlier Phase 8 bundle' if baseline_only else f'used `{previous_iteration_label}` as the previous comparison surface'}.",
        ]
    )
    return '\n'.join(lines) + '\n'


def run_sampled_single_paper_l2_regression(args: argparse.Namespace) -> dict[str, object]:
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_dir(str(args.iteration_label)).resolve()
    )
    docs_report_path = Path(args.docs_report).expanduser().resolve()

    iteration = run_sampled_single_paper_iteration(
        args.sampling_bundle,
        iteration_label=str(args.iteration_label),
        artifacts_dir=output_dir,
        allow_graph_write=bool(args.allow_graph_write),
    )
    previous_iteration = (
        load_sampled_l2_iteration_bundle(args.previous_iteration_dir)
        if args.previous_iteration_dir
        else None
    )
    comparison = compare_sampled_l2_iterations(iteration, previous_iteration)

    iteration_files = write_sampled_l2_iteration_bundle(
        output_dir,
        iteration=iteration,
        metadata={
            'runner': 'backend/scripts/run_sampled_single_paper_l2_regression.py',
            'allow_graph_write': bool(args.allow_graph_write),
        },
    )
    comparison_files = write_sampled_l2_comparison_bundle(output_dir, comparison=comparison)

    iteration_summary = build_sampled_l2_iteration_summary(iteration=iteration)
    comparison_summary = build_sampled_l2_comparison_summary(comparison=comparison)
    docs_report_path.parent.mkdir(parents=True, exist_ok=True)
    docs_report_path.write_text(
        _render_report(
            iteration_summary=iteration_summary,
            comparison_summary=comparison_summary,
            output_dir=output_dir,
            docs_report_path=docs_report_path,
            allow_graph_write=bool(args.allow_graph_write),
        ),
        encoding='utf-8',
    )

    return {
        'iteration_label': iteration.iteration_label,
        'output_dir': str(output_dir),
        'docs_report': str(docs_report_path),
        'baseline_only': comparison.baseline_only,
        'fixed_evaluated_count': len(iteration.fixed_results),
        'random_evaluated_count': len(iteration.random_results),
        'availability_issue_count': len(iteration.availability_issues),
        'fixed_verdict_counts': comparison_summary['fixed_verdict_counts'],
        'top_owner_buckets': [bucket['bucket'] for bucket in comparison_summary['owner_buckets'][:3]],
        'bundle_manifest': str(iteration_files['bundle_manifest'].resolve()),
        'comparison_summary': str(comparison_files['comparison_summary'].resolve()),
        'comparison_inspection': str(comparison_files['comparison_inspection'].resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_sampled_single_paper_l2_regression(args)
    except Exception as exc:  # noqa: BLE001 - CLI should report explicit failure details
        print(json.dumps({'error': str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
