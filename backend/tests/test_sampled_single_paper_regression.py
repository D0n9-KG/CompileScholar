from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.research_logic import (
    CorpusInventoryEntry,
    CorpusSamplingBatch,
    CorpusSamplingBundle,
    FixedRegressionSampledPaperResult,
    RandomExplorationSampledPaperResult,
    SampledPaperAvailabilityIssue,
    SampledSinglePaperIterationResult,
    compare_sampled_l2_iterations,
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


def _executed_result(
    *,
    cohort: str,
    corpus_paper_id: str,
    display_title: str,
    quality_tier: str,
    quality_flags: list[str] | None = None,
    completeness_audit: dict | None = None,
    reference_status: str = 'recovered',
    citation_event_status: str = 'recovered',
    purposes: int = 1,
    citation_acts: int = 1,
    citation_mentions: int = 1,
):
    payload = {
        'corpus_paper_id': corpus_paper_id,
        'display_title': display_title,
        'corpus_relative_ref': f'txt/{corpus_paper_id}.txt',
        'preferred_source_path': f'C:/corpus/{corpus_paper_id}.txt',
        'preferred_source_kind': 'txt',
        'iteration_label': 'cycle-01',
        'paper_id': f'doi:10.1000/{corpus_paper_id}',
        'trace_id': f'trace:{corpus_paper_id}',
        'source_path': f'C:/corpus/{corpus_paper_id}.txt',
        'source_kind': 'txt',
        'quality_report': {
            'quality_tier': quality_tier,
            'gate_passed': quality_tier != 'red',
            'quality_flags': list(quality_flags or []),
            'l2_completeness_audit': dict(completeness_audit or {}),
            'hot_path_gate_report': {
                'relation_coverage_ratio': (completeness_audit or {}).get('relation_coverage_ratio', 1.0),
            },
        },
        'trace_quality': {'quality_tier': quality_tier, 'audit_status': 'eligible'},
        'reference_recovery': {'status': reference_status},
        'citation_event_recovery': {'status': citation_event_status},
        'artifact_refs': {'paper_logic_trace': f'tmp/{corpus_paper_id}/paper_logic_trace.json'},
        'citations': {'refs': 1, 'cites_resolved': 1, 'cites_unresolved': 0},
        'citation_semantic': {'citation_acts': citation_acts, 'citation_mentions': citation_mentions},
        'llm': {'purposes': purposes, 'moves': 1, 'gate_passed': quality_tier != 'red', 'quality_tier': quality_tier},
    }
    if cohort == 'fixed_regression':
        return FixedRegressionSampledPaperResult(**payload)
    return RandomExplorationSampledPaperResult(**payload)


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


def test_compare_sampled_l2_iterations_classifies_fixed_random_and_availability_rows() -> None:
    previous = SampledSinglePaperIterationResult(
        iteration_label='cycle-00',
        sampling_bundle_dir='tmp/phase7',
        sampling_bundle_manifest_ref='tmp/phase7/bundle_manifest.json',
        fixed_selected_count=4,
        random_selected_count=4,
        fixed_results=[
            _executed_result(cohort='fixed_regression', corpus_paper_id='1001', display_title='Fixed One', quality_tier='green'),
            _executed_result(cohort='fixed_regression', corpus_paper_id='1002', display_title='Fixed Two', quality_tier='red'),
            _executed_result(cohort='fixed_regression', corpus_paper_id='1003', display_title='Fixed Three', quality_tier='green'),
            _executed_result(cohort='fixed_regression', corpus_paper_id='1004', display_title='Fixed Four', quality_tier='yellow'),
        ],
        random_results=[
            _executed_result(cohort='random_exploration', corpus_paper_id='2001', display_title='Random One', quality_tier='green'),
            _executed_result(cohort='random_exploration', corpus_paper_id='2003', display_title='Random Three', quality_tier='red'),
            _executed_result(cohort='random_exploration', corpus_paper_id='2004', display_title='Random Four', quality_tier='yellow'),
            _executed_result(cohort='random_exploration', corpus_paper_id='2005', display_title='Random Five', quality_tier='green'),
        ],
    )
    current = SampledSinglePaperIterationResult(
        iteration_label='cycle-01',
        sampling_bundle_dir='tmp/phase7',
        sampling_bundle_manifest_ref='tmp/phase7/bundle_manifest.json',
        fixed_selected_count=4,
        random_selected_count=5,
        fixed_results=[
            _executed_result(cohort='fixed_regression', corpus_paper_id='1001', display_title='Fixed One', quality_tier='red'),
            _executed_result(cohort='fixed_regression', corpus_paper_id='1002', display_title='Fixed Two', quality_tier='green'),
            _executed_result(cohort='fixed_regression', corpus_paper_id='1003', display_title='Fixed Three', quality_tier='green'),
            _executed_result(cohort='fixed_regression', corpus_paper_id='1004', display_title='Fixed Four', quality_tier='yellow'),
        ],
        random_results=[
            _executed_result(cohort='random_exploration', corpus_paper_id='2001', display_title='Random One', quality_tier='yellow'),
            _executed_result(cohort='random_exploration', corpus_paper_id='2003', display_title='Random Three', quality_tier='red'),
            _executed_result(cohort='random_exploration', corpus_paper_id='2004', display_title='Random Four', quality_tier='green'),
            _executed_result(cohort='random_exploration', corpus_paper_id='2005', display_title='Random Five', quality_tier='green'),
        ],
        availability_issues=[
            SampledPaperAvailabilityIssue(
                corpus_paper_id='2002',
                display_title='Random Missing',
                cohort='random_exploration',
                selection_mode='random_exploration',
                corpus_relative_ref='txt/2002.txt',
                preferred_source_path='C:/corpus/2002.txt',
                preferred_source_kind='txt',
                iteration_label='cycle-01',
                execution_status='source_missing',
                error_message='missing',
                error_type='FileNotFoundError',
            )
        ],
    )

    comparison = compare_sampled_l2_iterations(current, previous)

    fixed_verdicts = {row.corpus_paper_id: row.verdict for row in comparison.fixed_comparisons}
    random_verdicts = {row.corpus_paper_id: row.verdict for row in comparison.random_comparisons}

    assert fixed_verdicts == {
        '1001': 'new_regression',
        '1002': 'improved',
        '1003': 'stable_pass',
        '1004': 'recurring_failure',
    }
    assert random_verdicts == {
        '2001': 'new_edge_case',
        '2002': 'availability_only',
        '2003': 'repeated_random_failure',
        '2004': 'random_improved',
        '2005': 'stable_random_pass',
    }


def test_compare_sampled_l2_iterations_builds_all_owner_buckets_and_prioritizes_fixed_failures() -> None:
    current = SampledSinglePaperIterationResult(
        iteration_label='cycle-01',
        sampling_bundle_dir='tmp/phase7',
        sampling_bundle_manifest_ref='tmp/phase7/bundle_manifest.json',
        fixed_selected_count=1,
        random_selected_count=1,
        fixed_results=[
            _executed_result(
                cohort='fixed_regression',
                corpus_paper_id='1001',
                display_title='Fixed Owner Case',
                quality_tier='red',
                quality_flags=['metadata_summary_mismatch', 'route_state_seed_thin', 'residual_noise_moves'],
                completeness_audit={
                    'missing_expected_roles': ['result'],
                    'missing_expected_slot_fields': ['methods'],
                    'sparse_expected_slot_fields': ['effects'],
                    'relation_coverage_ratio': 0.9,
                    'noise_move_ids': ['noise-1'],
                },
                reference_status='recovered_heuristic_after_agent_error',
                citation_event_status='empty_result',
                purposes=0,
                citation_acts=0,
                citation_mentions=0,
            )
        ],
        random_results=[
            _executed_result(
                cohort='random_exploration',
                corpus_paper_id='2001',
                display_title='Random Relation Case',
                quality_tier='yellow',
                quality_flags=['weak_relation_stitching'],
                completeness_audit={'relation_coverage_ratio': 0.2},
            )
        ],
    )

    comparison = compare_sampled_l2_iterations(current, None)
    bucket_names = [bucket.bucket for bucket in comparison.owner_buckets]

    assert {
        'slot_recovery',
        'relation_assembly',
        'route_seed_richness',
        'metadata_repair',
        'noise_cleanup',
        'reference_recovery',
        'citation_semantics',
    }.issubset(set(bucket_names))
    assert comparison.owner_buckets[0].bucket != 'relation_assembly'
