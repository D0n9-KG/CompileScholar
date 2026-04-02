from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.research_logic import (  # noqa: E402
    ensure_packet_trace_coverage,
    load_historical_environment_snapshot,
    load_paper_logic_traces,
    load_route_packet,
    synthesize_route_state,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compile a route-state artifact from packetized traces and an optional L1 snapshot.')
    parser.add_argument('--packet', required=True, help='Path to a route packet JSON manifest.')
    parser.add_argument('--trace-dir', help='Directory containing PaperLogicTrace JSON files.')
    parser.add_argument('--trace-file', action='append', default=[], help='Explicit PaperLogicTrace JSON file. Repeatable.')
    parser.add_argument('--l1-snapshot', help='Optional HistoricalEnvironmentSnapshot JSON file.')
    parser.add_argument('--output-path', required=True, help='Path where the route state JSON will be written.')
    parser.add_argument('--built-at', help='Override the route-state build timestamp.')
    parser.add_argument('--route-state-id', help='Override the generated route_state_id.')
    return parser


def run_route_state_pilot(args: argparse.Namespace) -> dict[str, object]:
    route_packet = load_route_packet(args.packet)
    traces = load_paper_logic_traces(trace_dir=args.trace_dir, trace_files=args.trace_file)
    ensure_packet_trace_coverage(route_packet, traces)
    historical_environment_snapshot = (
        load_historical_environment_snapshot(args.l1_snapshot)
        if str(args.l1_snapshot or '').strip()
        else None
    )

    route_state = synthesize_route_state(
        route_packet,
        traces,
        l1_snapshot=historical_environment_snapshot,
        built_at=args.built_at,
        route_state_id=args.route_state_id,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(route_state.model_dump(mode='json', exclude_none=True), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )

    return {
        'route_state_id': route_state.route_state_id,
        'topic_scope': route_state.topic_scope,
        'cutoff_year': route_state.cutoff_year,
        'trace_count': len(traces),
        'l1_snapshot_id': route_state.compiler_metadata.l1_snapshot_version,
        'quality_tier': route_state.quality.quality_tier,
        'quality_flags': list(route_state.quality.quality_flags),
        'ready_for_why_now': route_state.quality.ready_for_why_now,
        'ready_for_route_comparison': route_state.quality.ready_for_route_comparison,
        'output_path': str(output_path.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_route_state_pilot(args)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({'error': str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
