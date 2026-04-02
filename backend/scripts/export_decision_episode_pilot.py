from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script: `python scripts/export_decision_episode_pilot.py ...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.research_logic import (  # noqa: E402
    build_decision_episode_audit_export,
    build_decision_episode_export_summary,
    load_route_packet,
    load_route_state,
    write_decision_episode_export_bundle,
)
from app.research_logic.models import (  # noqa: E402
    AntiPatternCard,
    DecisionEpisode,
    DecisionPriorCard,
    RouteComparisonCase,
    WhyNowCase,
)


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / 'tmp' / 'phase6_decision_episode_audit_export'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Build a bounded Phase 6 audited DecisionEpisode export from replay and review bundles.'
    )
    parser.add_argument(
        '--replay-bundle',
        required=True,
        help='Path to the replay bundle directory or its bundle_manifest.json.',
    )
    parser.add_argument(
        '--prior-review-bundle',
        required=True,
        help='Path to the prior review bundle directory or its bundle_manifest.json.',
    )
    parser.add_argument(
        '--output-dir',
        default=str(_default_output_dir()),
        help='Directory where the Phase 6 export bundle will be written.',
    )
    parser.add_argument('--built-at', help='Override the export build timestamp.')
    return parser


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'export artifact not found: {path}') from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid JSON in export artifact: {path}') from exc


def _resolve_bundle(path_like: str, *, label: str) -> tuple[Path, dict[str, object], Path]:
    path = Path(path_like).expanduser()
    manifest_path = path if path.name == 'bundle_manifest.json' else path / 'bundle_manifest.json'
    bundle_dir = manifest_path.parent

    if not bundle_dir.exists():
        raise FileNotFoundError(f'{label} not found: {bundle_dir}')
    if not manifest_path.is_file():
        raise FileNotFoundError(f'{label} bundle_manifest.json not found: {manifest_path}')

    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f'{label} manifest must decode to a JSON object: {manifest_path}')
    return bundle_dir.resolve(), manifest, manifest_path.resolve()


def _bundle_file(bundle_dir: Path, manifest: dict[str, object], *, key: str, label: str) -> Path:
    files = manifest.get('files')
    if not isinstance(files, dict) or not isinstance(files.get(key), str) or not str(files[key]).strip():
        raise ValueError(f'{label} manifest missing files.{key}')
    path = (bundle_dir / str(files[key])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{label} artifact missing for files.{key}: {path}')
    return path


def _load_model(path: Path, model_type):
    return model_type.model_validate(_read_json(path))


def _load_models(path: Path, model_type) -> list:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f'expected a JSON list at {path}')
    return [model_type.model_validate(item) for item in payload]


def _source_refs(bundle_dir: Path, manifest_path: Path, manifest: dict[str, object]) -> dict[str, object]:
    files = manifest.get('files')
    if not isinstance(files, dict):
        raise ValueError(f'Bundle manifest missing files map: {manifest_path}')
    return {
        'bundle_ref': str(bundle_dir),
        'manifest_ref': str(manifest_path),
        'files': {
            key: str((bundle_dir / relative_path).resolve())
            for key, relative_path in files.items()
            if isinstance(relative_path, str) and str(relative_path).strip()
        },
    }


def run_export_decision_episode_pilot(args: argparse.Namespace) -> dict[str, object]:
    replay_bundle_dir, replay_manifest, replay_manifest_path = _resolve_bundle(
        args.replay_bundle,
        label='replay bundle',
    )
    review_bundle_dir, review_manifest, review_manifest_path = _resolve_bundle(
        args.prior_review_bundle,
        label='prior review bundle',
    )

    route_packet = load_route_packet(
        _bundle_file(replay_bundle_dir, replay_manifest, key='route_packet', label='replay bundle')
    )
    route_state = load_route_state(
        _bundle_file(replay_bundle_dir, replay_manifest, key='primary_route_state', label='replay bundle')
    )
    why_now_case = _load_model(
        _bundle_file(replay_bundle_dir, replay_manifest, key='why_now_case', label='replay bundle'),
        WhyNowCase,
    )
    replay_summary = _read_json(
        _bundle_file(replay_bundle_dir, replay_manifest, key='replay_summary', label='replay bundle')
    )
    comparison_cases = _load_models(
        _bundle_file(replay_bundle_dir, replay_manifest, key='route_comparison_cases', label='replay bundle'),
        RouteComparisonCase,
    )
    replay_decision_episode = _load_model(
        _bundle_file(replay_bundle_dir, replay_manifest, key='decision_episode', label='replay bundle'),
        DecisionEpisode,
    )
    prior_candidates = _load_models(
        _bundle_file(review_bundle_dir, review_manifest, key='prior_candidates', label='prior review bundle'),
        DecisionPriorCard,
    )
    anti_pattern_candidates = _load_models(
        _bundle_file(review_bundle_dir, review_manifest, key='anti_pattern_candidates', label='prior review bundle'),
        AntiPatternCard,
    )

    selected_comparison_case_id = None
    if isinstance(replay_summary, dict):
        selected_comparison_case_id = replay_summary.get('selected_comparison_case_id')
    comparison_case = next(
        (
            case
            for case in comparison_cases
            if case.route_comparison_case_id == selected_comparison_case_id
        ),
        None,
    )

    accepted_prior_ids = review_manifest.get('accepted_prior_ids')
    accepted_anti_pattern_ids = review_manifest.get('accepted_anti_pattern_ids')
    if not isinstance(accepted_prior_ids, list):
        raise ValueError('prior review bundle manifest missing accepted_prior_ids list')
    if not isinstance(accepted_anti_pattern_ids, list):
        raise ValueError('prior review bundle manifest missing accepted_anti_pattern_ids list')

    export = build_decision_episode_audit_export(
        route_packet=route_packet,
        route_state=route_state,
        why_now_case=why_now_case,
        comparison_case=comparison_case,
        hindsight_outcome=replay_decision_episode.hindsight_outcome,
        prior_cards=prior_candidates,
        anti_pattern_cards=anti_pattern_candidates,
        accepted_prior_ids=accepted_prior_ids,
        accepted_anti_pattern_ids=accepted_anti_pattern_ids,
        route_state_ref=str(
            _bundle_file(replay_bundle_dir, replay_manifest, key='primary_route_state', label='replay bundle')
        ),
        source_replay_bundle_refs=_source_refs(replay_bundle_dir, replay_manifest_path, replay_manifest),
        source_review_bundle_refs=_source_refs(review_bundle_dir, review_manifest_path, review_manifest),
        built_at=args.built_at,
        episode_id=replay_decision_episode.episode_id,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    written_files = write_decision_episode_export_bundle(
        output_dir,
        export=export,
        metadata={
            'runner': 'backend/scripts/export_decision_episode_pilot.py',
            'source_replay_bundle': str(replay_bundle_dir),
            'source_prior_review_bundle': str(review_bundle_dir),
        },
    )

    summary = build_decision_episode_export_summary(export=export)
    summary['exported_episode_id'] = export.decision_episode.episode_id
    summary['output_dir'] = str(output_dir)
    summary['bundle_manifest'] = str(written_files['bundle_manifest'].resolve())
    summary['export_summary'] = str(written_files['export_summary'].resolve())
    summary['export_inspection'] = str(written_files['export_inspection'].resolve())
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_export_decision_episode_pilot(args)
    except Exception as exc:  # noqa: BLE001 - CLI should report explicit failure details
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
