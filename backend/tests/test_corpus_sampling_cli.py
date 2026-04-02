from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = BACKEND_DIR / 'scripts' / 'run_corpus_sampling_baseline.py'


def _load_script_module():
    spec = importlib.util.spec_from_file_location('run_corpus_sampling_baseline', SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORPUS_CLI = _load_script_module()


def _write_text(path: Path, text: str = 'content') -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


def _fixture_corpus(root: Path) -> Path:
    _write_text(root / '1001_Alpha_Paper' / '1001_Alpha_Paper.md', '# alpha')
    _write_text(root / 'txt' / '1001_Alpha_Paper.txt', 'alpha text')
    _write_text(root / '1002_Beta_Result' / '1002_Beta_Result.md', '# beta')
    _write_text(root / 'txt' / '1003_Gamma_Scan.txt', 'gamma text')
    _write_text(root / '1004_Delta_Study' / '1004_Delta_Study.md', '# delta')
    _write_text(root / 'txt' / '1005_Epsilon_Test.txt', 'epsilon text')
    return root


def _manifest(path: Path, ids: list[str]) -> Path:
    title_map = {
        '1001': 'Alpha Paper',
        '1002': 'Beta Result',
        '9999': 'Missing Paper',
    }
    ref_map = {
        '1001': '1001_Alpha_Paper/1001_Alpha_Paper.md',
        '1002': '1002_Beta_Result/1002_Beta_Result.md',
        '9999': 'missing/9999_Missing_Paper.md',
    }
    return _write_json(
        path,
        [
            {
                'corpus_paper_id': corpus_paper_id,
                'display_title': title_map[corpus_paper_id],
                'corpus_relative_ref': ref_map[corpus_paper_id],
                'selection_reason': 'fixture regression set entry',
            }
            for corpus_paper_id in ids
        ],
    )


class _HappyNeo4jClient:
    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return None

    def list_paper_ingestion_rows_by_source_md_paths(self, source_md_paths: list[str]) -> list[dict]:
        rows: list[dict] = []
        for path in source_md_paths:
            if path.endswith('1001_Alpha_Paper.md'):
                rows.append(
                    {
                        'source_md_path': path,
                        'paper_id': 'paper:1001',
                        'paper_source': 'fixture-md',
                        'ingested': True,
                    }
                )
            if path.endswith('1002_Beta_Result.md'):
                rows.append(
                    {
                        'source_md_path': path,
                        'paper_id': 'paper:1002',
                        'paper_source': 'fixture-md',
                        'ingested': True,
                    }
                )
        return rows


class _FailingNeo4jClient(_HappyNeo4jClient):
    def list_paper_ingestion_rows_by_source_md_paths(self, source_md_paths: list[str]) -> list[dict]:
        raise RuntimeError('neo4j unavailable')


def test_build_parser_defaults_output_dir_to_phase7_tmp() -> None:
    parser = CORPUS_CLI.build_parser()

    args = parser.parse_args(['--corpus-root', 'C:/demo'])

    assert Path(args.output_dir) == CORPUS_CLI._default_output_dir()


def test_run_corpus_sampling_baseline_writes_bundle_from_fixture_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_root = _fixture_corpus(tmp_path / 'corpus')
    manifest_path = _manifest(tmp_path / 'manifest.json', ['1001', '1002'])
    output_dir = tmp_path / 'phase7-output'
    monkeypatch.setattr(CORPUS_CLI, 'Neo4jClient', _HappyNeo4jClient)

    summary = CORPUS_CLI.run_corpus_sampling_baseline(
        CORPUS_CLI.build_parser().parse_args(
            [
                '--corpus-root',
                str(corpus_root),
                '--fixed-regression-manifest',
                str(manifest_path),
                '--fixed-count',
                '2',
                '--random-count',
                '2',
                '--seed',
                '17',
                '--output-dir',
                str(output_dir),
            ]
        )
    )

    fixed_batch_payload = json.loads((output_dir / 'outputs' / 'fixed_regression_batch.json').read_text(encoding='utf-8'))
    random_batch_payload = json.loads((output_dir / 'outputs' / 'random_exploration_batch.json').read_text(encoding='utf-8'))
    manifest_payload = json.loads((output_dir / 'bundle_manifest.json').read_text(encoding='utf-8'))

    assert summary['output_dir'] == str(output_dir.resolve())
    assert summary['fixed_selected_count'] == 2
    assert summary['random_selected_count'] == 2
    assert summary['corpus_health_failure_count'] == 0
    assert summary['neo4j_lookup_status'] == 'ready'
    assert set(random_batch_payload['selected_ids']).isdisjoint(fixed_batch_payload['selected_ids'])
    assert manifest_payload['neo4j_lookup_status'] == 'ready'
    assert Path(summary['bundle_manifest']).is_file()


def test_run_corpus_sampling_baseline_rejects_missing_fixed_ids(tmp_path: Path) -> None:
    corpus_root = _fixture_corpus(tmp_path / 'corpus')
    manifest_path = _manifest(tmp_path / 'manifest.json', ['1001', '9999'])

    with pytest.raises(ValueError, match='fixed regression ids not found'):
        CORPUS_CLI.run_corpus_sampling_baseline(
            CORPUS_CLI.build_parser().parse_args(
                [
                    '--corpus-root',
                    str(corpus_root),
                    '--fixed-regression-manifest',
                    str(manifest_path),
                    '--fixed-count',
                    '2',
                    '--random-count',
                    '2',
                    '--skip-neo4j',
                ]
            )
        )


def test_run_corpus_sampling_baseline_continues_when_neo4j_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_root = _fixture_corpus(tmp_path / 'corpus')
    manifest_path = _manifest(tmp_path / 'manifest.json', ['1001', '1002'])
    output_dir = tmp_path / 'phase7-output'
    monkeypatch.setattr(CORPUS_CLI, 'Neo4jClient', _FailingNeo4jClient)

    summary = CORPUS_CLI.run_corpus_sampling_baseline(
        CORPUS_CLI.build_parser().parse_args(
            [
                '--corpus-root',
                str(corpus_root),
                '--fixed-regression-manifest',
                str(manifest_path),
                '--fixed-count',
                '2',
                '--random-count',
                '2',
                '--seed',
                '17',
                '--output-dir',
                str(output_dir),
            ]
        )
    )

    manifest_payload = json.loads((output_dir / 'bundle_manifest.json').read_text(encoding='utf-8'))

    assert summary['neo4j_lookup_status'] == 'unavailable'
    assert summary['selected_without_neo4j_metadata_count'] == 4
    assert manifest_payload['neo4j_lookup_status'] == 'unavailable'
    assert 'neo4j unavailable' in str(manifest_payload['neo4j_lookup_error'])
