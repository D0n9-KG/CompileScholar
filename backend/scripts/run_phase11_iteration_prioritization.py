from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.research_logic import (  # noqa: E402
    IterationPriorityPreflightError,
    build_iteration_priority_inspection,
    build_iteration_priority_summary,
    build_iteration_priority_inspection_payload,
    build_iteration_priority_summary_payload,
    load_phase8_comparison_inspection,
    load_phase8_comparison_summary,
    load_phase10_evidence,
    write_iteration_priority_bundle,
)
from app.research_logic.iteration_prioritization import render_iteration_priority_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Read explicit Phase 8 and Phase 10 evidence, write a Phase 11 bundle, and render a markdown report.'
    )
    parser.add_argument('--phase8-summary', required=True, help='Path to the Phase 8 comparison_summary.json file.')
    parser.add_argument('--phase8-inspection', required=True, help='Path to the Phase 8 comparison_inspection.json file.')
    parser.add_argument(
        '--phase10-summary',
        help='Optional path to the Phase 10 comparison_summary.json file. If omitted, fallback markdown inputs are required.',
    )
    parser.add_argument(
        '--phase10-verification',
        help='Path to the Phase 10 verification markdown. Required when --phase10-summary is omitted; otherwise recorded as supplemental provenance.',
    )
    parser.add_argument(
        '--phase10-report',
        help='Path to the Phase 10 report markdown. Required when --phase10-summary is omitted; otherwise recorded as supplemental provenance.',
    )
    parser.add_argument('--output-dir', required=True, help='Directory where the Phase 11 bundle will be written.')
    parser.add_argument('--report-md', required=True, help='Markdown path where the Phase 11 operator report will be written.')
    return parser


def _resolve_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    return Path(path_value).expanduser().resolve()


def _require_file(path: Path | None, *, label: str) -> Path:
    if path is None:
        raise IterationPriorityPreflightError(f'Missing required {label}.')
    if not path.is_file():
        raise IterationPriorityPreflightError(f'{label} not found: {path}')
    return path


def _validate_phase10_inputs(
    *,
    phase10_summary_path: Path | None,
    phase10_verification_path: Path | None,
    phase10_report_path: Path | None,
) -> None:
    if phase10_summary_path is not None:
        _require_file(phase10_summary_path, label='Phase 10 comparison summary')
        if phase10_verification_path is not None:
            _require_file(phase10_verification_path, label='Phase 10 verification note')
        if phase10_report_path is not None:
            _require_file(phase10_report_path, label='Phase 10 validation report')
        return

    if phase10_verification_path is None or phase10_report_path is None:
        raise IterationPriorityPreflightError(
            'When --phase10-summary is omitted, both --phase10-verification and --phase10-report are required.'
        )
    _require_file(phase10_verification_path, label='Phase 10 verification note')
    _require_file(phase10_report_path, label='Phase 10 validation report')


def run_phase11_iteration_prioritization(args: argparse.Namespace) -> dict[str, object]:
    phase8_summary_path = _require_file(_resolve_path(args.phase8_summary), label='Phase 8 comparison summary')
    phase8_inspection_path = _require_file(_resolve_path(args.phase8_inspection), label='Phase 8 comparison inspection')
    phase10_summary_path = _resolve_path(args.phase10_summary)
    phase10_verification_path = _resolve_path(args.phase10_verification)
    phase10_report_path = _resolve_path(args.phase10_report)
    output_dir = _resolve_path(args.output_dir)
    report_md = _resolve_path(args.report_md)

    if output_dir is None or report_md is None:
        raise IterationPriorityPreflightError('Both --output-dir and --report-md are required.')

    _validate_phase10_inputs(
        phase10_summary_path=phase10_summary_path,
        phase10_verification_path=phase10_verification_path,
        phase10_report_path=phase10_report_path,
    )

    phase8_summary = load_phase8_comparison_summary(phase8_summary_path)
    phase8_inspection = load_phase8_comparison_inspection(phase8_inspection_path)
    phase10_surface = load_phase10_evidence(
        summary_path=phase10_summary_path,
        verification_path=phase10_verification_path,
        report_path=phase10_report_path,
    )
    if phase10_summary_path is not None:
        phase10_surface.source_refs.phase10_summary_path = str(phase10_summary_path)
        phase10_surface.source_refs.phase10_mode = 'json'
        phase10_surface.source_refs.fallback_used = False
        if phase10_verification_path is not None:
            phase10_surface.source_refs.phase10_verification_path = str(phase10_verification_path)
        if phase10_report_path is not None:
            phase10_surface.source_refs.phase10_report_path = str(phase10_report_path)

    summary = build_iteration_priority_summary(
        phase8_summary=phase8_summary,
        phase8_inspection=phase8_inspection,
        phase10_surface=phase10_surface,
    )
    inspection = build_iteration_priority_inspection(
        phase8_summary=phase8_summary,
        phase8_inspection=phase8_inspection,
        phase10_surface=phase10_surface,
    )
    summary_payload = build_iteration_priority_summary_payload(summary=summary)
    inspection_payload = build_iteration_priority_inspection_payload(inspection=inspection)

    written_files = write_iteration_priority_bundle(
        output_dir,
        summary=summary,
        inspection=inspection,
        metadata={
            'runner': 'backend/scripts/run_phase11_iteration_prioritization.py',
            'report_md': str(report_md),
        },
    )

    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(
        render_iteration_priority_report(summary_payload, inspection_payload),
        encoding='utf-8',
    )

    return {
        'output_dir': str(output_dir),
        'report_md': str(report_md),
        'primary_recommendation': summary.primary_recommendation_id,
        'fallback_used': summary.source_refs.fallback_used,
        'source_refs': summary.source_refs.model_dump(mode='json', exclude_none=True),
        'bundle_manifest': str(written_files['bundle_manifest'].resolve()),
        'prioritization_summary': str(written_files['prioritization_summary'].resolve()),
        'prioritization_inspection': str(written_files['prioritization_inspection'].resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run_phase11_iteration_prioritization(args)
    except Exception as exc:  # noqa: BLE001 - CLI should report explicit failure details
        print(json.dumps({'error': str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
