from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.research_logic.phase10_multi_paper_validation import (  # noqa: E402
    DEFAULT_PHASE10_BASELINE_EXPORT_BUNDLE,
    DEFAULT_PHASE10_BASELINE_REPLAY_BUNDLE,
    run_phase10_package_and_replay,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Bridge the committed Phase 9 packet into package, replay, prior-review, and audited export bundles.'
    )
    parser.add_argument('--packet', required=True, help='Path to the canonical Phase 9 route packet JSON file.')
    parser.add_argument('--assembly-manifest', required=True, help='Path to the committed Phase 9 assembly manifest JSON file.')
    parser.add_argument('--l1-snapshot-output', required=True, help='Path where the generated HistoricalEnvironmentSnapshot JSON will be written.')
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory where the Phase 10 route_state_package/, replay_bundle/, prior_review_bundle/, and export_bundle/ folders will be written.',
    )
    parser.add_argument(
        '--runtime-output-root',
        help='Optional root for the generated Phase 10 runtime bridge packets, manifest, and shared snapshot.',
    )
    parser.add_argument(
        '--baseline-replay-bundle',
        default=str(DEFAULT_PHASE10_BASELINE_REPLAY_BUNDLE),
        help='Path to the baseline replay bundle directory or bundle_manifest.json used for comparison.',
    )
    parser.add_argument(
        '--baseline-export-bundle',
        default=str(DEFAULT_PHASE10_BASELINE_EXPORT_BUNDLE),
        help='Path to the baseline export bundle directory or bundle_manifest.json used for comparison.',
    )
    parser.add_argument(
        '--report-md',
        help='Optional markdown report path rendered from comparison_summary.json.',
    )
    parser.add_argument(
        '--reviewer',
        action='append',
        default=[],
        help='Reviewer id to attach to replay and prior-review generation. Repeat for multiple reviewers.',
    )
    parser.add_argument('--built-at', help='Override the build timestamp used for generated Phase 10 artifacts.')
    return parser


def _flag_present(raw_args: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f'{flag}=') for token in raw_args)


def _resolve_cli_path(path_value: str) -> str:
    return str(Path(path_value).expanduser().resolve())


def main(argv: list[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(raw_args)

    # Required CLI paths should honor the caller's current working directory,
    # while omitted optional paths should continue using repo-relative defaults.
    args.packet = _resolve_cli_path(args.packet)
    args.assembly_manifest = _resolve_cli_path(args.assembly_manifest)
    args.l1_snapshot_output = _resolve_cli_path(args.l1_snapshot_output)
    args.output_dir = _resolve_cli_path(args.output_dir)
    if args.runtime_output_root is not None and _flag_present(raw_args, '--runtime-output-root'):
        args.runtime_output_root = _resolve_cli_path(args.runtime_output_root)
    if args.report_md is not None and _flag_present(raw_args, '--report-md'):
        args.report_md = _resolve_cli_path(args.report_md)
    if _flag_present(raw_args, '--baseline-replay-bundle'):
        args.baseline_replay_bundle = _resolve_cli_path(args.baseline_replay_bundle)
    if _flag_present(raw_args, '--baseline-export-bundle'):
        args.baseline_export_bundle = _resolve_cli_path(args.baseline_export_bundle)

    try:
        result = run_phase10_package_and_replay(
            packet_path=args.packet,
            assembly_manifest_path=args.assembly_manifest,
            l1_snapshot_output_path=args.l1_snapshot_output,
            runtime_output_root=args.runtime_output_root,
            output_dir=args.output_dir,
            baseline_replay_bundle=args.baseline_replay_bundle,
            baseline_export_bundle=args.baseline_export_bundle,
            reviewer_ids=args.reviewer,
            report_md=args.report_md,
            built_at=args.built_at,
            repo_root=REPO_ROOT,
        )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({'error': str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
