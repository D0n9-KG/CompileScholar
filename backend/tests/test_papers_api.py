from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers.papers import _bib_escape, _canonical_dir_for_paper_id, _doi_sanitized, _export_bibtex, _export_csv, _safe_rel
from app.paper_logic_trace.models import CanonicalCore, PaperLogicTrace, PaperMetadata, ResearchMove


def _sample_trace() -> PaperLogicTrace:
    return PaperLogicTrace(
        trace_id='doi:10.1000/test:paper_logic_trace',
        schema_version='v2',
        built_at='2026-03-22T00:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id='doi:10.1000/test',
            canonical_doi='10.1000/test',
            title='A Test Paper',
            year=2024,
            authors=['Alice', 'Bob'],
            source_refs=['chunk:1'],
        ),
        canonical_core=CanonicalCore(
            evidence_anchors=[],
            moves=[
                ResearchMove(
                    move_id='m-1',
                    sequence_no=1,
                    role='method',
                    act_type='propose_method',
                    summary='Finding one',
                    anchor_ids=['a-1'],
                    confidence=0.9,
                ),
                ResearchMove(
                    move_id='m-2',
                    sequence_no=2,
                    role='result',
                    act_type='report_effect',
                    summary='Finding two',
                    anchor_ids=['a-2'],
                    confidence=0.8,
                ),
            ],
            move_relations=[],
            citation_acts=[],
            figure_refs=[],
            table_refs=[],
        ),
        derived_views={},
        quality={'quality_tier': 'green', 'audit_status': 'not_needed', 'hot_path_gate_report': {'passed': True}},
    )


def test_doi_sanitized_basic():
    assert _doi_sanitized('10.1000/abc') == '10.1000_abc'


def test_doi_sanitized_strips_and_lowercases():
    assert _doi_sanitized('  10.ABC/XYZ  ') == '10.abc_xyz'


def test_doi_sanitized_special_chars():
    assert _doi_sanitized('10.1000/a(b)c') == '10.1000_a_b_c'


def test_safe_rel_normal():
    assert _safe_rel('foo/bar.png') == 'foo/bar.png'


def test_safe_rel_backslash():
    assert _safe_rel('foo\\bar.png') == 'foo/bar.png'


def test_safe_rel_rejects_absolute():
    with pytest.raises(ValueError):
        _safe_rel('/etc/passwd')


def test_safe_rel_rejects_dotdot():
    with pytest.raises(ValueError):
        _safe_rel('../secret')


def test_safe_rel_rejects_empty():
    with pytest.raises(ValueError):
        _safe_rel('')


def test_safe_rel_rejects_drive_letter():
    with pytest.raises(ValueError):
        _safe_rel('C:/Windows')


def test_safe_rel_rejects_special_chars():
    with pytest.raises(ValueError):
        _safe_rel('foo bar.png')


def test_canonical_dir_rejects_non_doi():
    with pytest.raises(FileNotFoundError, match='not found'):
        _canonical_dir_for_paper_id('sha256:abc123')


def test_canonical_dir_not_found():
    with pytest.raises(FileNotFoundError, match='not found'):
        _canonical_dir_for_paper_id('doi:10.9999/nonexistent')


def test_get_paper_content_finds_paper_md():
    with tempfile.TemporaryDirectory() as tmpdir:
        paper_dir = Path(tmpdir) / 'papers' / 'doi' / '10.1000_test'
        paper_dir.mkdir(parents=True)
        md_file = paper_dir / 'paper.md'
        md_file.write_text('# Test Paper\n\nContent here.', encoding='utf-8')

        with patch('app.api.routers.papers._canonical_dir_for_paper_id', return_value=paper_dir):
            from app.api.routers.papers import get_paper_content

            resp = get_paper_content('doi:10.1000/test')
            assert resp.status_code == 200
            assert b'# Test Paper' in resp.body


def test_get_paper_content_fallback_to_source_md():
    with tempfile.TemporaryDirectory() as tmpdir:
        paper_dir = Path(tmpdir) / 'papers' / 'doi' / '10.1000_test'
        paper_dir.mkdir(parents=True)
        md_file = paper_dir / 'source.md'
        md_file.write_text('# Source', encoding='utf-8')

        with patch('app.api.routers.papers._canonical_dir_for_paper_id', return_value=paper_dir):
            from app.api.routers.papers import get_paper_content

            resp = get_paper_content('doi:10.1000/test')
            assert resp.status_code == 200
            assert b'# Source' in resp.body


def test_get_paper_content_prefers_source_md_when_both_exist():
    with tempfile.TemporaryDirectory() as tmpdir:
        paper_dir = Path(tmpdir) / 'papers' / 'doi' / '10.1000_test'
        paper_dir.mkdir(parents=True)
        (paper_dir / 'paper.md').write_text('# Paper Version', encoding='utf-8')
        (paper_dir / 'source.md').write_text('# Source Version', encoding='utf-8')

        with patch('app.api.routers.papers._canonical_dir_for_paper_id', return_value=paper_dir):
            from app.api.routers.papers import get_paper_content

            resp = get_paper_content('doi:10.1000/test')
            assert resp.status_code == 200
            assert b'# Source Version' in resp.body
            assert b'# Paper Version' not in resp.body


def test_get_paper_content_prefers_exact_indexed_source_path_when_available():
    with tempfile.TemporaryDirectory() as tmpdir:
        paper_dir = Path(tmpdir) / 'papers' / 'doi' / '10.1000_test'
        paper_dir.mkdir(parents=True)
        indexed_md = paper_dir / 'indexed-source.md'
        indexed_md.write_text('# Indexed Source Version', encoding='utf-8')
        (paper_dir / 'source.md').write_text('# Generic Source Version', encoding='utf-8')

        with patch('app.api.routers.papers._canonical_dir_for_paper_id', return_value=paper_dir):
            with patch('app.api.routers.papers._source_md_file_for_paper_id', return_value=indexed_md, create=True):
                from app.api.routers.papers import get_paper_content

                resp = get_paper_content('doi:10.1000/test')
                assert resp.status_code == 200
                assert b'# Indexed Source Version' in resp.body
                assert b'# Generic Source Version' not in resp.body


def test_get_paper_content_no_md_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        paper_dir = Path(tmpdir) / 'papers' / 'doi' / '10.1000_test'
        paper_dir.mkdir(parents=True)

        with patch('app.api.routers.papers._canonical_dir_for_paper_id', return_value=paper_dir):
            from fastapi import HTTPException
            from app.api.routers.papers import get_paper_content

            with pytest.raises(HTTPException) as exc_info:
                get_paper_content('doi:10.1000/test')
            assert exc_info.value.status_code == 404


def test_export_bibtex_basic():
    bib = _export_bibtex(_sample_trace())
    assert '@article{' in bib
    assert 'title = {A Test Paper}' in bib
    assert 'year = {2024}' in bib
    assert 'doi = {10.1000/test}' in bib
    assert 'Alice and Bob' in bib


def test_export_bibtex_no_authors():
    trace = _sample_trace().model_copy(
        update={
            'paper_metadata': _sample_trace().paper_metadata.model_copy(
                update={'canonical_doi': '10.1000/x', 'title': 'T', 'authors': []},
            ),
        },
    )
    bib = _export_bibtex(trace)
    assert 'author' not in bib
    assert 'title = {T}' in bib


def test_export_bibtex_empty_paper():
    trace = _sample_trace().model_copy(
        update={
            'paper_metadata': _sample_trace().paper_metadata.model_copy(
                update={'canonical_doi': None, 'title': '', 'authors': [], 'year': None},
            ),
        },
    )
    bib = _export_bibtex(trace)
    assert '@article{unknown,' in bib
    assert 'title = {Untitled}' in bib


def test_export_csv_basic():
    text = _export_csv(_sample_trace())
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    assert rows[0] == ['move_id', 'sequence_no', 'role', 'act_type', 'summary', 'anchor_count', 'confidence']
    assert len(rows) == 3
    assert rows[1][0] == 'm-1'
    assert rows[1][2] == 'method'
    assert rows[2][0] == 'm-2'
    assert rows[2][2] == 'result'


def test_export_csv_no_moves():
    trace = _sample_trace().model_copy(
        update={'canonical_core': _sample_trace().canonical_core.model_copy(update={'moves': []})},
    )
    text = _export_csv(trace)
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    assert len(rows) == 1


def test_bib_escape_braces_and_backslash():
    assert _bib_escape('a{b}c') == 'a\\{b\\}c'
    assert _bib_escape('x\\y') == 'x\\\\y'


def test_bib_escape_newline():
    assert _bib_escape('line1\nline2') == 'line1 line2'


def test_bib_escape_plain_text():
    assert _bib_escape('Hello World') == 'Hello World'


def test_export_bibtex_escapes_special_chars():
    trace = _sample_trace().model_copy(
        update={
            'paper_metadata': _sample_trace().paper_metadata.model_copy(
                update={'canonical_doi': '10.1/x', 'title': 'A {B} Title\nMore', 'authors': ["O'Brien"], 'year': None},
            ),
        },
    )
    bib = _export_bibtex(trace)
    assert 'A \\{B\\} Title More' in bib
    assert "O'Brien" in bib


def test_logic_trace_route_returns_canonical_payload(monkeypatch):
    import app.api.routers.papers as papers_router

    class _FakeNeo4jClient:
        def __init__(self, uri: str, user: str, password: str) -> None:
            self.uri = uri
            self.user = user
            self.password = password

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return None

    def _fake_export(_client, paper_id: str):
        return {
            'paper_metadata': {'paper_id': paper_id, 'title': 'Demo'},
            'canonical_core': {
                'moves': [{'move_id': 'm-1'}],
                'move_relations': [],
                'evidence_anchors': [],
                'citation_acts': [],
                'figure_refs': [],
                'table_refs': [],
            },
            'derived_views': {'community_signatures': [{'move_id': 'm-1'}]},
            'quality': {'audit_status': 'eligible'},
        }

    monkeypatch.setattr(papers_router, 'Neo4jClient', _FakeNeo4jClient)
    monkeypatch.setattr(papers_router, 'export_paper_logic_trace', _fake_export)

    app = FastAPI()
    app.include_router(papers_router.router)
    client = TestClient(app)

    res = client.get('/papers/paper-1/logic-trace')

    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload['paper_metadata']['paper_id'] == 'paper-1'
    assert payload['canonical_core']['moves'][0]['move_id'] == 'm-1'
    assert payload['derived_views']['community_signatures'][0]['move_id'] == 'm-1'
    assert payload['quality']['audit_status'] == 'eligible'
