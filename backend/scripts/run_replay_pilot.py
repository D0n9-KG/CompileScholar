from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script: `python scripts/run_replay_pilot.py ...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.research_logic import (
    build_prior_candidate_registry_from_package,
    build_replay_summary,
    compile_historical_replay,
    ensure_packet_trace_coverage,
    load_historical_environment_snapshot,
    load_paper_logic_traces,
    load_route_packet,
    load_route_state_package_bundle,
    load_route_states,
    write_prior_candidate_review_bundle,
    write_replay_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run a local historical replay pilot from packetized artifacts.')
    parser.add_argument('--packet', required=True, help='Path to a route packet JSON manifest.')
    parser.add_argument('--trace-dir', help='Directory containing PaperLogicTrace JSON files.')
    parser.add_argument('--trace-file', action='append', default=[], help='Explicit PaperLogicTrace JSON file. Repeatable.')
    parser.add_argument('--l1-snapshot', help='Optional HistoricalEnvironmentSnapshot JSON file.')
    parser.add_argument('--output-dir', required=True, help='Directory where the replay bundle will be written.')
    parser.add_argument(
        '--route-state-package',
        help='Optional route-state package bundle directory or bundle_manifest.json. Supplies grouped support / alternative / held-out route states.',
    )
    parser.add_argument(
        '--prior-review-output-dir',
        help='Optional output directory for the Phase 5 prior review bundle. Requires --route-state-package.',
    )
    parser.add_argument(
        '--support-route-state',
        action='append',
        default=[],
        help='Serialized RouteState JSON used as support cluster input. Repeatable.',
    )
    parser.add_argument(
        '--alternative-route-state',
        action='append',
        default=[],
        help='Serialized RouteState JSON used as alternative-route input. Repeatable.',
    )
    parser.add_argument(
        '--held-out-route-state',
        action='append',
        default=[],
        help='Serialized RouteState JSON used as held-out evaluation input. Repeatable.',
    )
    parser.add_argument('--reviewer', action='append', default=[], help='Reviewer id for replay metadata. Repeatable.')
    parser.add_argument('--built-at', help='Override the replay build timestamp.')
    parser.add_argument('--route-state-ref', help='Optional reference string stored in the DecisionEpisode.')
    parser.add_argument('--historical-cutoff-time', help='Override the historical cutoff timestamp in the DecisionEpisode.')
    return parser


def run_replay_pilot(args: argparse.Namespace) -> dict[str, object]:
    route_packet = load_route_packet(args.packet)
    traces = load_paper_logic_traces(trace_dir=args.trace_dir, trace_files=args.trace_file)
    ensure_packet_trace_coverage(route_packet, traces)
    historical_environment_snapshot = (
        load_historical_environment_snapshot(args.l1_snapshot)
        if str(args.l1_snapshot or '').strip()
        else None
    )

    route_state_package = (
        load_route_state_package_bundle(args.route_state_package)
        if str(args.route_state_package or '').strip()
        else None
    )
    support_route_states = list(route_state_package.support_route_states if route_state_package else [])
    support_route_states.extend(load_route_states(args.support_route_state))
    alternative_route_states = list(route_state_package.alternative_route_states if route_state_package else [])
    alternative_route_states.extend(load_route_states(args.alternative_route_state))
    held_out_route_states = list(route_state_package.held_out_route_states if route_state_package else [])
    held_out_route_states.extend(load_route_states(args.held_out_route_state))
    reviewer_ids = [reviewer_id for reviewer_id in args.reviewer if str(reviewer_id).strip()]
    prior_review_bundle = None
    if str(args.prior_review_output_dir or '').strip():
        if route_state_package is None:
            raise ValueError('--prior-review-output-dir requires --route-state-package')
        prior_candidate_registry = build_prior_candidate_registry_from_package(
            route_state_package,
            reviewer_ids=reviewer_ids or None,
            built_at=args.built_at,
        )
        prior_review_bundle = write_prior_candidate_review_bundle(
            args.prior_review_output_dir,
            registry=prior_candidate_registry,
            metadata={
                'runner': 'backend/scripts/run_replay_pilot.py',
                'route_state_package_id': route_state_package.manifest.package_id,
            },
        )

    compilation = compile_historical_replay(
        route_packet,
        traces,
        l1_snapshot=historical_environment_snapshot,
        support_route_states=support_route_states,
        alternative_route_states=alternative_route_states,
        held_out_route_states=held_out_route_states,
        reviewer_ids=reviewer_ids or None,
        route_state_ref=args.route_state_ref,
        historical_cutoff_time=args.historical_cutoff_time,
        built_at=args.built_at,
    )

    written_files = write_replay_bundle(
        args.output_dir,
        route_packet=route_packet,
        traces=traces,
        compilation=compilation,
        historical_environment_snapshot=historical_environment_snapshot,
        support_route_states=support_route_states,
        alternative_route_states=alternative_route_states,
        held_out_route_states=held_out_route_states,
        reviewer_ids=reviewer_ids,
        metadata={
            'runner': 'backend/scripts/run_replay_pilot.py',
            'l1_snapshot_id': historical_environment_snapshot.snapshot_id if historical_environment_snapshot else None,
            'route_state_package_id': route_state_package.manifest.package_id if route_state_package else None,
        },
        route_state_package_validation=route_state_package.validation if route_state_package else None,
    )

    summary = build_replay_summary(
        route_packet=route_packet,
        traces=traces,
        compilation=compilation,
        historical_environment_snapshot=historical_environment_snapshot,
        support_route_states=support_route_states,
        alternative_route_states=alternative_route_states,
        held_out_route_states=held_out_route_states,
        reviewer_ids=reviewer_ids,
    )
    summary['output_dir'] = str(Path(args.output_dir).resolve())
    summary['bundle_manifest'] = str(written_files['bundle_manifest'].resolve())
    if route_state_package is not None:
        summary['route_state_package_id'] = route_state_package.manifest.package_id
        summary['route_state_package_validation_quality_tier'] = (
            route_state_package.validation.quality_tier if route_state_package.validation is not None else None
        )
        summary['route_state_package_validation_flags'] = (
            list(route_state_package.validation.quality_flags) if route_state_package.validation is not None else []
        )
    if prior_review_bundle is not None:
        summary['prior_review_output_dir'] = str(Path(args.prior_review_output_dir).resolve())
        summary['prior_review_bundle_manifest'] = str(prior_review_bundle['bundle_manifest'].resolve())
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_replay_pilot(args)
    except Exception as exc:  # noqa: BLE001 - CLI should return explicit failure details
        print(
            json.dumps(
                {'error': str(exc)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
