from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.research_logic.phase10_multi_paper_validation import run_phase10_package_and_replay  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent

DEFAULT_PACKET_PATH = Path(r'..\docs\replay\pilot_packets\phase9-route-packet.json')
DEFAULT_ASSEMBLY_MANIFEST_PATH = Path(r'..\docs\replay\pilot_packets\phase9-assembly-manifest.json')
DEFAULT_BASELINE_REPLAY_BUNDLE = Path(r'..\tmp\phase12_direct_fix_cycle\cycle1\replay_bundle')
DEFAULT_BASELINE_EXPORT_BUNDLE = Path(r'..\tmp\phase12_direct_fix_cycle\cycle1\export_bundle')
DEFAULT_OUTPUT_ROOT = Path(r'..\tmp\phase13_cycle2_refine')
DEFAULT_REPORT_NAME = 'phase13-cycle-report.md'
DEFAULT_L1_SNAPSHOT_NAME = 'phase9-comp-mech-l1-snapshot.json'


@dataclass(frozen=True)
class Phase13ResolvedPaths:
    packet_path: Path
    assembly_manifest_path: Path
    baseline_replay_bundle: Path
    baseline_export_bundle: Path
    output_root: Path
    output_dir: Path
    l1_snapshot_output_path: Path
    report_md: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run a bounded Phase 13 refinement iteration on the frozen Phase 9 packet against the Phase 12 cycle-1 baseline.'
    )
    parser.add_argument('--iteration-label', required=True, help='Iteration label written under the Phase 13 output root.')
    parser.add_argument(
        '--output-root',
        default=str(DEFAULT_OUTPUT_ROOT),
        help='Root directory for iteration-scoped Phase 13 outputs.',
    )
    parser.add_argument(
        '--packet',
        default=str(DEFAULT_PACKET_PATH),
        help='Path to the canonical Phase 9 route packet JSON file.',
    )
    parser.add_argument(
        '--assembly-manifest',
        default=str(DEFAULT_ASSEMBLY_MANIFEST_PATH),
        help='Path to the canonical Phase 9 assembly manifest JSON file.',
    )
    parser.add_argument(
        '--baseline-replay-bundle',
        default=str(DEFAULT_BASELINE_REPLAY_BUNDLE),
        help='Path to the Phase 12 cycle-1 replay bundle used as the default comparison baseline.',
    )
    parser.add_argument(
        '--baseline-export-bundle',
        default=str(DEFAULT_BASELINE_EXPORT_BUNDLE),
        help='Path to the Phase 12 cycle-1 export bundle used as the default comparison baseline.',
    )
    parser.add_argument(
        '--report-md',
        help='Optional markdown report path. Defaults to <iteration-dir>\\phase13-cycle-report.md.',
    )
    parser.add_argument(
        '--reviewer',
        action='append',
        default=[],
        help='Reviewer id to attach to replay and prior-review generation. Repeat for multiple reviewers.',
    )
    parser.add_argument('--built-at', help='Override the build timestamp used for generated Phase 13 artifacts.')
    return parser


def _flag_present(raw_args: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f'{flag}=') for token in raw_args)


def _resolve_cli_path(path_value: str) -> Path:
    return Path(path_value).expanduser().resolve()


def _resolve_backend_default(path_value: Path) -> Path:
    return (BACKEND_DIR / path_value).resolve()


def resolve_phase13_paths(
    *,
    iteration_label: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    packet_path: str | Path = DEFAULT_PACKET_PATH,
    assembly_manifest_path: str | Path = DEFAULT_ASSEMBLY_MANIFEST_PATH,
    baseline_replay_bundle: str | Path = DEFAULT_BASELINE_REPLAY_BUNDLE,
    baseline_export_bundle: str | Path = DEFAULT_BASELINE_EXPORT_BUNDLE,
    report_md: str | Path | None = None,
) -> Phase13ResolvedPaths:
    resolved_output_root = (
        _resolve_backend_default(Path(output_root))
        if output_root == DEFAULT_OUTPUT_ROOT
        else _resolve_cli_path(str(output_root))
    )
    resolved_iteration_dir = resolved_output_root / iteration_label
    resolved_report_md = (
        resolved_iteration_dir / DEFAULT_REPORT_NAME
        if report_md is None
        else _resolve_cli_path(str(report_md))
    )
    return Phase13ResolvedPaths(
        packet_path=_resolve_backend_default(Path(packet_path)) if packet_path == DEFAULT_PACKET_PATH else _resolve_cli_path(str(packet_path)),
        assembly_manifest_path=(
            _resolve_backend_default(Path(assembly_manifest_path))
            if assembly_manifest_path == DEFAULT_ASSEMBLY_MANIFEST_PATH
            else _resolve_cli_path(str(assembly_manifest_path))
        ),
        baseline_replay_bundle=(
            _resolve_backend_default(Path(baseline_replay_bundle))
            if baseline_replay_bundle == DEFAULT_BASELINE_REPLAY_BUNDLE
            else _resolve_cli_path(str(baseline_replay_bundle))
        ),
        baseline_export_bundle=(
            _resolve_backend_default(Path(baseline_export_bundle))
            if baseline_export_bundle == DEFAULT_BASELINE_EXPORT_BUNDLE
            else _resolve_cli_path(str(baseline_export_bundle))
        ),
        output_root=resolved_output_root,
        output_dir=resolved_iteration_dir,
        l1_snapshot_output_path=resolved_iteration_dir / 'shared' / DEFAULT_L1_SNAPSHOT_NAME,
        report_md=resolved_report_md,
    )


def main(argv: list[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(raw_args)

    resolved_paths = resolve_phase13_paths(
        iteration_label=args.iteration_label,
        output_root=args.output_root,
        packet_path=args.packet,
        assembly_manifest_path=args.assembly_manifest,
        baseline_replay_bundle=args.baseline_replay_bundle,
        baseline_export_bundle=args.baseline_export_bundle,
        report_md=args.report_md if _flag_present(raw_args, '--report-md') else None,
    )

    try:
        result = run_phase10_package_and_replay(
            packet_path=resolved_paths.packet_path,
            assembly_manifest_path=resolved_paths.assembly_manifest_path,
            l1_snapshot_output_path=resolved_paths.l1_snapshot_output_path,
            output_dir=resolved_paths.output_dir,
            baseline_replay_bundle=resolved_paths.baseline_replay_bundle,
            baseline_export_bundle=resolved_paths.baseline_export_bundle,
            reviewer_ids=args.reviewer,
            report_md=resolved_paths.report_md,
            built_at=args.built_at,
            repo_root=REPO_ROOT,
        )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({'error': str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    summary = {
        **result.summary,
        'iteration_label': args.iteration_label,
        'output_dir': str(resolved_paths.output_dir.resolve()),
        'baseline_replay_bundle': str(resolved_paths.baseline_replay_bundle.resolve()),
        'baseline_export_bundle': str(resolved_paths.baseline_export_bundle.resolve()),
        'reviewer_ids': [reviewer_id for reviewer_id in args.reviewer if str(reviewer_id).strip()],
        'comparison_summary_path': str(result.comparison_summary_path.resolve()),
        'report_markdown_path': str(resolved_paths.report_md.resolve()),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
