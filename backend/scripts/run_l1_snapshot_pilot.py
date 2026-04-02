from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.research_logic import (  # noqa: E402
    build_historical_environment_snapshot,
    build_l1_snapshot_ref,
    ensure_packet_trace_coverage,
    load_paper_logic_traces,
    load_route_packet,
    write_historical_environment_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build a paper-grounded L1 historical environment snapshot.')
    parser.add_argument('--packet', required=True, help='Path to a route packet JSON manifest.')
    parser.add_argument('--trace-dir', help='Directory containing PaperLogicTrace JSON files.')
    parser.add_argument('--trace-file', action='append', default=[], help='Explicit PaperLogicTrace JSON file. Repeatable.')
    parser.add_argument('--output-path', required=True, help='Path where the L1 snapshot JSON will be written.')
    parser.add_argument('--built-at', help='Override the snapshot build timestamp.')
    parser.add_argument('--snapshot-id', help='Override the generated snapshot id.')
    return parser


def run_l1_snapshot_pilot(args: argparse.Namespace) -> dict[str, object]:
    route_packet = load_route_packet(args.packet)
    traces = load_paper_logic_traces(trace_dir=args.trace_dir, trace_files=args.trace_file)
    ensure_packet_trace_coverage(route_packet, traces)

    snapshot = build_historical_environment_snapshot(
        route_packet,
        traces,
        built_at=args.built_at,
        snapshot_id=args.snapshot_id,
    )
    output_path = write_historical_environment_snapshot(args.output_path, snapshot)
    snapshot_ref = build_l1_snapshot_ref(snapshot)
    return {
        'snapshot_id': snapshot.snapshot_id,
        'topic_scope': snapshot.topic_scope,
        'cutoff_year': snapshot.cutoff_year,
        'trace_count': len(traces),
        'quality_tier': snapshot.quality.quality_tier,
        'quality_flags': list(snapshot.quality.quality_flags),
        'output_path': str(output_path.resolve()),
        'snapshot_ref': snapshot_ref.model_dump(mode='json', exclude_none=True),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_l1_snapshot_pilot(args)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({'error': str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
