from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.research_logic import (
    CorpusInventoryEntry,
    CorpusSamplingBatch,
    CorpusSamplingBundle,
    run_sampled_single_paper_iteration,
    write_corpus_sampling_bundle,
)


def _sample_entry(
    *,
    corpus_paper_id: str,
    display_title: str,
    cohort: str,
    preferred_source_path: str,
    preferred_source_kind: str,
) -> CorpusInventoryEntry:
    return CorpusInventoryEntry(
        corpus_paper_id=corpus_paper_id,
        display_title=display_title,
        corpus_relative_ref=f'{cohort}/{corpus_paper_id}',
        preferred_source_path=preferred_source_path,
        preferred_source_kind=preferred_source_kind,  # type: ignore[arg-type]
        md_path=preferred_source_path if preferred_source_kind == 'md' else None,
        txt_path=preferred_source_path if preferred_source_kind == 'txt' else None,
        eligibility_status='eligible',
    )


def _fixture_sampling_bundle(bundle_dir: Path) -> Path:
    fixed_entry = _sample_entry(
        corpus_paper_id='1001',
        display_title='Fixed Alpha',
        cohort='fixed',
        preferred_source_path=str(bundle_dir / 'sources' / 'fixed-alpha.txt'),
        preferred_source_kind='txt',
    )
    random_entry = _sample_entry(
        corpus_paper_id='2001',
        display_title='Random Beta',
        cohort='random',
        preferred_source_path=str(bundle_dir / 'sources' / 'random-beta.md'),
        preferred_source_kind='md',
    )
    bundle = CorpusSamplingBundle(
        built_at='2026-04-03T04:00:00Z',
        corpus_root=str(bundle_dir / 'corpus-root'),
        inventory_entries=[fixed_entry, random_entry],
        fixed_regression_batch=CorpusSamplingBatch(
            batch_id='phase7-fixed-regression',
            sampling_mode='fixed_regression',
            built_at='2026-04-03T04:00:00Z',
            requested_count=1,
            selected=[fixed_entry],
            selected_ids=['1001'],
        ),
        random_exploration_batch=CorpusSamplingBatch(
            batch_id='phase7-random-exploration',
            sampling_mode='random_exploration',
            built_at='2026-04-03T04:00:00Z',
            requested_count=1,
            selected=[random_entry],
            selected_ids=['2001'],
            seed=7,
        ),
        seed=7,
        neo4j_lookup_status='unavailable',
    )
    write_corpus_sampling_bundle(bundle_dir, bundle=bundle, metadata={'runner': 'pytest'})
    return bundle_dir


def test_evaluate_sampled_paper_from_source_returns_source_missing_for_unreachable_path(tmp_path: Path) -> None:
    from app.ingest.rebuild import evaluate_sampled_paper_from_source

    result = evaluate_sampled_paper_from_source(
        corpus_paper_id='1001',
        cohort='fixed_regression',
        corpus_relative_ref='txt/1001_missing.txt',
        preferred_source_path=str(tmp_path / 'missing.txt'),
        preferred_source_kind='txt',
        iteration_label='cycle-01',
        artifacts_dir=tmp_path / 'artifacts',
    )

    assert result['execution_status'] == 'source_missing'
    assert result['corpus_paper_id'] == '1001'
    assert result['paper_id'] is None
    assert result['quality_report'] is None
    assert result['error_type'] == 'FileNotFoundError'


def test_evaluate_sampled_paper_from_source_preserves_corpus_identity_for_txt_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from app.ingest import rebuild as rebuild_mod
    from app.ingest.models import Chunk, CitationEvent, DocumentIR, MdSpan, PaperDraft, ReferenceEntry

    txt_path = tmp_path / '1001_alpha.txt'
    txt_path.write_text('# Demo\n\nBody [1]\n\n# References\n[1] Demo ref\n', encoding='utf-8')
    doc = DocumentIR(
        paper=PaperDraft(
            paper_source='txt-demo',
            md_path=str(txt_path),
            title='Txt Demo',
            title_alt=None,
            authors=['Alice Smith'],
            doi='10.1000/txt-demo',
            year=2024,
        ),
        chunks=[
            Chunk(
                chunk_id='chunk-1',
                paper_source='txt-demo',
                md_path=str(txt_path),
                span=MdSpan(start_line=3, end_line=3),
                section='Intro',
                kind='block',
                text='Body [1]',
            )
        ],
        references=[
            ReferenceEntry(
                paper_source='txt-demo',
                md_path=str(txt_path),
                ref_num=1,
                raw='Demo reference',
            )
        ],
        citations=[
            CitationEvent(
                paper_source='txt-demo',
                md_path=str(txt_path),
                cited_ref_num=1,
                chunk_id='chunk-1',
                span=MdSpan(start_line=3, end_line=3),
                context='Body [1]',
            )
        ],
    )

    monkeypatch.setattr(rebuild_mod, 'Neo4jClient', lambda *args, **kwargs: pytest.fail('graph should not be used'))
    monkeypatch.setattr(rebuild_mod, 'parse_mineru_markdown', lambda _: doc)
    monkeypatch.setattr(rebuild_mod, 'recover_references_with_agent', lambda doc, **kwargs: (doc, {'status': 'ok'}))
    monkeypatch.setattr(rebuild_mod, 'recover_citation_events_from_references', lambda doc, **kwargs: (doc, {'status': 'ok'}))
    monkeypatch.setattr(rebuild_mod, 'CrossrefClient', lambda: object())
    monkeypatch.setattr(
        rebuild_mod,
        'build_reference_and_cite_records',
        lambda doc, **kwargs: {
            'paper_id': 'doi:10.1000/txt-demo',
            'refs': [{'ref_num': 1, 'raw': 'Demo reference'}],
            'cited_papers': [{'paper_id': 'doi:10.1000/ref', 'title': 'Demo cited paper'}],
            'cites_resolved': [
                {
                    'cited_paper_id': 'doi:10.1000/ref',
                    'ref_nums': [1],
                    'evidence_chunk_ids': ['chunk-1'],
                    'evidence_spans': ['3-3'],
                }
            ],
            'cites_unresolved': [],
        },
    )
    monkeypatch.setattr(rebuild_mod, '_schema_for_md', lambda *_: {'prompts': {}, 'rules': {}, 'version': 1, 'paper_type': 'research'})
    monkeypatch.setattr(rebuild_mod, 'load_canonical_meta', lambda *_: {'paper_type': 'research'})
    monkeypatch.setattr(rebuild_mod, 'load_active', lambda *_: {'version': 1, 'paper_type': 'research', 'prompts': {}, 'rules': {}})
    monkeypatch.setattr(
        rebuild_mod,
        'run_phase1_paper_logic_trace',
        lambda **kwargs: {
            'quality_report': {
                'gate_passed': True,
                'quality_tier': 'green',
                'quality_tier_score': 0.92,
                'quality_flags': ['slot_recovery'],
                'l2_completeness_audit': {'missing_expected_slots': []},
            },
            'paper_logic_trace': {
                'trace_id': 'trace:txt-demo',
                'quality': {
                    'quality_tier': 'green',
                    'audit_status': 'reviewed',
                    'hot_path_gate_report': {'passed': True},
                },
                'canonical_core': {'moves': [{'move_id': 'm1'}]},
            },
        },
    )
    monkeypatch.setattr(
        rebuild_mod,
        'classify_citation_purposes_batch',
        lambda **kwargs: {
            'by_id': {
                'doi:10.1000/ref': {
                    'labels': ['Background'],
                    'scores': [0.2],
                }
            }
        },
    )

    result = rebuild_mod.evaluate_sampled_paper_from_source(
        corpus_paper_id='1001',
        cohort='fixed_regression',
        corpus_relative_ref='txt/1001_alpha.txt',
        preferred_source_path=str(txt_path),
        preferred_source_kind='txt',
        iteration_label='cycle-01',
        artifacts_dir=tmp_path / 'artifacts',
    )

    assert result['execution_status'] == 'executed'
    assert result['corpus_paper_id'] == '1001'
    assert result['preferred_source_kind'] == 'txt'
    assert result['paper_id'] == 'doi:10.1000/txt-demo'
    assert result['trace_id'] == 'trace:txt-demo'
    assert result['quality_report']['quality_tier'] == 'green'
    assert result['trace_quality']['quality_tier'] == 'green'
    assert result['reference_recovery'] == {'status': 'ok'}
    assert result['citation_event_recovery']['paper_id'] == 'doi:10.1000/txt-demo'
    assert Path(result['artifact_refs']['paper_logic_trace']).is_file()
    assert Path(result['artifact_refs']['citation_mentions']).is_file()


def test_run_sampled_single_paper_iteration_separates_availability_issues_from_executed_results(tmp_path: Path) -> None:
    bundle_dir = _fixture_sampling_bundle(tmp_path / 'phase7-bundle')
    calls: list[tuple[str, str]] = []

    def fake_evaluator(**kwargs):
        calls.append((kwargs['corpus_paper_id'], str(kwargs['artifacts_dir'])))
        if kwargs['corpus_paper_id'] == '1001':
            artifact_dir = Path(kwargs['artifacts_dir'])
            artifact_dir.mkdir(parents=True, exist_ok=True)
            trace_path = artifact_dir / 'paper_logic_trace.json'
            trace_path.write_text('{}\n', encoding='utf-8')
            return {
                'execution_status': 'executed',
                'paper_id': 'doi:10.1000/fixed-alpha',
                'trace_id': 'trace:fixed-alpha',
                'source_path': kwargs['preferred_source_path'],
                'source_kind': kwargs['preferred_source_kind'],
                'quality_report': {'quality_tier': 'green', 'gate_passed': True},
                'trace_quality': {'quality_tier': 'green', 'audit_status': 'reviewed'},
                'reference_recovery': {'status': 'ok'},
                'citation_event_recovery': {'status': 'ok'},
                'artifacts_dir': str(artifact_dir),
                'artifact_refs': {'paper_logic_trace': str(trace_path)},
                'citations': {'refs': 1, 'cites_resolved': 1, 'cites_unresolved': 0},
                'llm': {'purposes': 1, 'moves': 1, 'gate_passed': True, 'quality_tier': 'green'},
                'skipped_canonical_write': True,
            }
        return {
            'execution_status': 'source_missing',
            'error_message': 'missing source',
            'error_type': 'FileNotFoundError',
        }

    iteration = run_sampled_single_paper_iteration(
        bundle_dir,
        iteration_label='cycle-01',
        artifacts_dir=tmp_path / 'phase8-output',
        evaluator=fake_evaluator,
    )

    assert iteration.fixed_selected_count == 1
    assert iteration.random_selected_count == 1
    assert len(iteration.fixed_results) == 1
    assert len(iteration.random_results) == 0
    assert len(iteration.availability_issues) == 1
    assert iteration.fixed_results[0].corpus_paper_id == '1001'
    assert iteration.fixed_results[0].selection_mode == 'fixed_regression'
    assert iteration.availability_issues[0].corpus_paper_id == '2001'
    assert iteration.availability_issues[0].execution_status == 'source_missing'
    assert all(result.corpus_paper_id != '2001' for result in iteration.fixed_results)
    assert calls[0][1].endswith(str(Path('paper_artifacts') / 'fixed_regression' / '1001'))
