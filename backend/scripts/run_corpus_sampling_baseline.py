from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

# Allow running as a script: `python scripts/run_corpus_sampling_baseline.py ...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph.neo4j_client import Neo4jClient  # noqa: E402
from app.research_logic import (  # noqa: E402
    CorpusSamplingBundle,
    build_corpus_sampling_summary,
    build_fixed_regression_batch,
    build_random_exploration_batch,
    load_fixed_regression_manifest,
    scan_corpus_inventory,
    write_corpus_sampling_bundle,
)
from app.settings import settings  # noqa: E402


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / 'tmp' / 'phase7_corpus_sampling_baseline'


def _default_fixed_manifest() -> Path:
    return Path(__file__).resolve().parents[2] / 'docs' / 'replay' / 'corpus_sampling' / 'phase7-fixed-regression-set.json'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Scan the shared corpus, freeze the Phase 7 fixed set, and emit a reproducible sampling baseline bundle.'
    )
    parser.add_argument('--corpus-root', required=True, help='Shared corpus root used as the sampling inventory source.')
    parser.add_argument(
        '--output-dir',
        default=str(_default_output_dir()),
        help='Directory where the Phase 7 sampling bundle will be written.',
    )
    parser.add_argument(
        '--fixed-regression-manifest',
        default=str(_default_fixed_manifest()),
        help='JSON manifest describing the fixed regression set.',
    )
    parser.add_argument('--fixed-count', type=int, default=10, help='Expected fixed regression set size.')
    parser.add_argument('--random-count', type=int, default=5, help='Random exploration set size.')
    parser.add_argument('--seed', type=int, default=7, help='Seed used for random exploration sampling.')
    parser.add_argument(
        '--skip-neo4j',
        action='store_true',
        help='Skip best-effort Neo4j enrichment and mark graph metadata as skipped.',
    )
    return parser


def _apply_neo4j_rows(bundle: CorpusSamplingBundle, rows: list[dict[str, object]]) -> None:
    rows_by_source_md_path = {
        str(row.get('source_md_path') or '').strip(): row
        for row in rows
        if str(row.get('source_md_path') or '').strip()
    }
    for entry in bundle.inventory_entries:
        if not entry.md_path:
            continue
        row = rows_by_source_md_path.get(entry.md_path)
        if row is None:
            continue
        entry.neo4j_paper_id = str(row.get('paper_id') or '').strip() or None
        entry.neo4j_paper_source = str(row.get('paper_source') or '').strip() or None
        ingested = row.get('ingested')
        entry.neo4j_ingested = bool(ingested) if ingested is not None else None


def run_corpus_sampling_baseline(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = Path(args.fixed_regression_manifest).expanduser().resolve()
    manifest_entries = load_fixed_regression_manifest(manifest_path)
    if len(manifest_entries) != int(args.fixed_count):
        raise ValueError(
            f'fixed regression manifest has {len(manifest_entries)} entries but --fixed-count requested {int(args.fixed_count)}'
        )

    initial_bundle = scan_corpus_inventory(args.corpus_root)
    neo4j_lookup_status = 'skipped' if args.skip_neo4j else 'ready'
    neo4j_lookup_error = None

    if not args.skip_neo4j:
        md_paths = [entry.md_path for entry in initial_bundle.inventory_entries if entry.md_path]
        try:
            with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
                rows = client.list_paper_ingestion_rows_by_source_md_paths(md_paths)
            _apply_neo4j_rows(initial_bundle, rows)
        except Exception as exc:  # noqa: BLE001 - best-effort enrichment must not abort sampling
            neo4j_lookup_status = 'unavailable'
            neo4j_lookup_error = str(exc)

    built_at = _utc_now_iso()
    fixed_batch = build_fixed_regression_batch(
        initial_bundle.inventory_entries,
        manifest_entries,
        fixed_manifest_ref=str(manifest_path),
        built_at=built_at,
    )
    random_batch = build_random_exploration_batch(
        initial_bundle.inventory_entries,
        random_count=int(args.random_count),
        seed=int(args.seed),
        fixed_ids=fixed_batch.selected_ids,
        built_at=built_at,
    )
    bundle = CorpusSamplingBundle(
        built_at=built_at,
        corpus_root=initial_bundle.corpus_root,
        inventory_entries=initial_bundle.inventory_entries,
        corpus_health_failures=initial_bundle.corpus_health_failures,
        fixed_regression_batch=fixed_batch,
        random_exploration_batch=random_batch,
        fixed_manifest_ref=str(manifest_path),
        seed=int(args.seed),
        neo4j_lookup_status=neo4j_lookup_status,
        neo4j_lookup_error=neo4j_lookup_error,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    written_files = write_corpus_sampling_bundle(
        output_dir,
        bundle=bundle,
        metadata={
            'runner': 'backend/scripts/run_corpus_sampling_baseline.py',
            'fixed_count': int(args.fixed_count),
            'random_count': int(args.random_count),
        },
    )
    summary = build_corpus_sampling_summary(bundle=bundle)
    summary['output_dir'] = str(output_dir)
    summary['bundle_manifest'] = str(written_files['bundle_manifest'].resolve())
    summary['fixed_batch_path'] = str(written_files['fixed_regression_batch'].resolve())
    summary['random_batch_path'] = str(written_files['random_exploration_batch'].resolve())
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_corpus_sampling_baseline(args)
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
