from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.research_logic.phase10_multi_paper_validation import (  # noqa: E402
    run_phase10_package_and_replay,
)


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
    parser.add_argument('--built-at', help='Override the build timestamp used for generated Phase 10 artifacts.')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_phase10_package_and_replay(
            packet_path=args.packet,
            assembly_manifest_path=args.assembly_manifest,
            l1_snapshot_output_path=args.l1_snapshot_output,
            output_dir=args.output_dir,
            built_at=args.built_at,
        )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({'error': str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
