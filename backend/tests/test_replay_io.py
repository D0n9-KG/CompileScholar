from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from app.paper_logic_trace.derived_views import build_derived_views
from app.paper_logic_trace.models import CanonicalCore, MentionValue, PaperLogicTrace, PaperMetadata, ResearchMove, SlotProvenance
from app.research_logic import (
    CorpusHealthIssue,
    CorpusInventoryEntry,
    CorpusSamplingBatch,
    CorpusSamplingBundle,
    FixedRegressionSampledPaperResult,
    IterationPriorityInspection,
    IterationPriorityRecommendation,
    IterationPrioritySourceRefs,
    IterationPrioritySummary,
    RandomExplorationSampledPaperResult,
    SampledL2ComparisonResult,
    SampledPaperAvailabilityIssue,
    SampledSinglePaperIterationResult,
    build_corpus_sampling_inspection,
    build_corpus_sampling_summary,
    build_decision_episode_audit_export,
    build_decision_episode_export_inspection,
    build_decision_episode_export_summary,
    build_iteration_priority_inspection_payload,
    build_iteration_priority_summary_payload,
    build_prior_candidate_registry,
    compile_historical_replay,
    compare_sampled_l2_iterations,
    ensure_packet_trace_coverage,
    load_historical_environment_snapshot,
    load_paper_logic_traces,
    load_route_packet,
    load_route_states,
    build_sampled_l2_comparison_inspection,
    build_sampled_l2_comparison_summary,
    build_sampled_l2_iteration_inspection,
    build_sampled_l2_iteration_summary,
    write_corpus_sampling_bundle,
    write_decision_episode_export_bundle,
    write_final_training_dataset_bundle,
    write_iteration_priority_bundle,
    write_prior_candidate_review_bundle,
    write_replay_bundle,
    write_sampled_l2_comparison_bundle,
    write_sampled_l2_iteration_bundle,
)
from app.research_logic.models import AntiPatternCard


def _mention(surface: str, normalized: str, anchor_id: str, *, mention_type: str | None = None) -> MentionValue:
    return MentionValue(
        surface=surface,
        normalized=normalized,
        type=mention_type,
        anchor_ids=[anchor_id],
    )


def _prov(field: str, anchor_id: str, value_index: int) -> SlotProvenance:
    return SlotProvenance(
        field=field,
        value_index=value_index,
        anchor_ids=[anchor_id],
        extraction_mode='direct',
        support_strength='strong',
    )


def _thin_method_move(prefix: str) -> ResearchMove:
    return ResearchMove(
        move_id=f'{prefix}-m1',
        sequence_no=1,
        role='method',
        act_type='propose_method',
        summary='Uses graph neural network modeling for retrieval.',
        research_objects=[_mention('entity relation graph', 'entity relation graph', f'{prefix}-a1')],
        methods=[_mention('graph neural network', 'graph neural network', f'{prefix}-a2')],
        anchor_ids=[f'{prefix}-a1', f'{prefix}-a2'],
        slot_provenance=[
            _prov('research_objects', f'{prefix}-a1', 0),
            _prov('methods', f'{prefix}-a2', 0),
        ],
        confidence=0.86,
    )


def _trace(paper_id: str, year: int, prefix: str) -> PaperLogicTrace:
    trace = PaperLogicTrace(
        trace_id=f'{paper_id}:paper_logic_trace',
        schema_version='v2',
        built_at='2026-04-02T01:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id=paper_id,
            title=f'Demo {paper_id}',
            year=year,
            paper_type='empirical',
            source_refs=[f'{prefix}-src'],
        ),
        canonical_core=CanonicalCore(moves=[_thin_method_move(prefix)]),
        quality={},
    )
    trace.derived_views = build_derived_views(trace)
    return trace


def _packet(
    traces: list[PaperLogicTrace],
    *,
    cutoff_year: int,
    missing_trace_id: bool = False,
) -> dict[str, object]:
    packet_quality = {
        'quality_tier': 'yellow' if missing_trace_id else 'green',
        'ready_for_route_state': not missing_trace_id,
        'quality_flags': ['missing_trace_ids'] if missing_trace_id else [],
        'topic_boundary_confidence': 0.82,
        'leakage_risk': 'low',
        'manual_review_status': 'completed' if not missing_trace_id else 'partial',
    }
    return {
        'packet_id': f'packet-{cutoff_year}',
        'built_at': '2026-04-02T01:05:00Z',
        'topic_scope_candidate': 'large-scale image recognition with deep neural networks',
        'cutoff_year': cutoff_year,
        'packet_status': 'frozen',
        'packet_composition': {
            'target_size': len(traces),
            'actual_size': len(traces),
            'role_counts': {
                'core_method': max(1, len(traces)),
                'resource_or_benchmark': 0,
                'limitation_or_critique': 0,
                'survey_or_review': 0,
                'alternative_route': 0,
            },
            'coverage_ok': True,
            'missing_roles': [],
        },
        'l1_snapshot_ref': {
            'snapshot_id': 'imagenet_2011_snapshot',
            'resource_registry_ref': 'l1:resource-registry',
            'resource_timeline_ref': 'l1:resource-timeline',
            'benchmark_timeline_ref': 'l1:benchmark-timeline',
            'toolchain_timeline_ref': 'l1:toolchain-timeline',
            'protocol_registry_ref': 'l1:protocol-registry',
        },
        'inclusion_rules': {
            'scope_definition': 'Compile a bounded route state from packetized traces.',
            'scope_aliases': ['cnn image recognition'],
            'accepted_year_range': {'min_year': 2005, 'max_year': cutoff_year},
            'hard_exclusion_rules': ['exclude papers after cutoff'],
            'role_assignment_rules': ['one primary role per packet item'],
            'leakage_policy': 'Papers after cutoff are excluded and hindsight labels never enter packet compilation.',
        },
        'included_items': [
            {
                'paper_id': trace.paper_metadata.paper_id,
                'trace_id': None if missing_trace_id else trace.trace_id,
                'paper_year': trace.paper_metadata.year,
                'item_role': 'core_method',
                'inclusion_reason': 'Supports packetized route reconstruction.',
                'source_selector': 'manual',
                'evidence_for_inclusion': [trace.paper_metadata.paper_id],
                'title': trace.paper_metadata.title,
            }
            for trace in traces
        ],
        'excluded_items': [],
        'packet_quality': packet_quality,
        'compiler_hints': {
            'preferred_scope_label': 'large-scale image recognition with deep neural networks',
            'preferred_method_labels': ['graph neural network'],
            'preferred_benchmark_labels': [],
            'preferred_bottleneck_labels': [],
            'expected_alternative_routes': [],
            'notes_for_route_state_compiler': 'Prefer packet-level aggregation over single-paper restatement.',
        },
    }


def _l1_snapshot(*, cutoff_year: int, snapshot_id: str = 'imagenet_2011_snapshot') -> dict[str, object]:
    return {
        'snapshot_id': snapshot_id,
        'built_at': '2026-04-02T09:00:00Z',
        'topic_scope': 'large-scale image recognition with deep neural networks',
        'cutoff_year': cutoff_year,
        'grounding_mode': 'paper_grounded_l1_lite',
        'source_packet_id': f'packet-{cutoff_year}',
        'source_trace_ids': ['paper-a:paper_logic_trace'],
        'source_paper_ids': ['paper-a'],
        'resource_registry': [
            {
                'label': 'imagenet-1k',
                'status': 'available',
                'confidence': 0.82,
                'source_paper_ids': ['paper-a'],
                'source_trace_ids': ['paper-a:paper_logic_trace'],
                'source_move_ids': ['pa-m1'],
                'evidence_ids': ['l1-r1'],
                'resource_types': ['dataset'],
            }
        ],
        'benchmark_timeline': [
            {
                'label': 'wn18rr',
                'status': 'available',
                'confidence': 0.8,
                'source_paper_ids': ['paper-a'],
                'source_trace_ids': ['paper-a:paper_logic_trace'],
                'source_move_ids': ['pa-m1'],
                'evidence_ids': ['l1-b1'],
                'benchmark_type': 'benchmark',
                'metric_tokens': ['mrr'],
                'comparator_tokens': ['baseline retriever'],
            }
        ],
        'toolchain_timeline': [
            {
                'label': 'cuda stack',
                'status': 'available',
                'confidence': 0.78,
                'source_paper_ids': ['paper-a'],
                'source_trace_ids': ['paper-a:paper_logic_trace'],
                'source_move_ids': ['pa-m1'],
                'evidence_ids': ['l1-t1'],
                'resource_types': ['compute'],
                'method_tokens': ['graph neural network'],
            }
        ],
        'protocol_registry': [
            {
                'label': 'top-1 accuracy',
                'status': 'available',
                'confidence': 0.79,
                'source_paper_ids': ['paper-a'],
                'source_trace_ids': ['paper-a:paper_logic_trace'],
                'source_move_ids': ['pa-m1'],
                'evidence_ids': ['l1-p1'],
                'metric_tokens': ['top-1 accuracy'],
                'comparator_tokens': ['baseline retriever'],
                'condition_tokens': ['standard split'],
                'method_tokens': ['evaluation'],
                'resource_tokens': ['imagenet-1k'],
            }
        ],
        'unresolved_questions': ['External benchmark governance remains unmodeled.'],
        'quality': {
            'quality_tier': 'yellow',
            'quality_flags': ['paper_only_snapshot'],
            'audit_status': 'eligible',
            'paper_only_snapshot': True,
            'completeness_score': 0.74,
        },
    }


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


def _anti_pattern_card(*, anti_pattern_id: str, route_state_ids: list[str]) -> AntiPatternCard:
    return AntiPatternCard(
        anti_pattern_id=anti_pattern_id,
        built_at='2026-04-02T03:20:00Z',
        anti_pattern_text='Repeated blocker pressure should trigger caution.',
        warning_signal_pattern={
            'signals': [
                {
                    'label': 'anisotropic packing',
                    'signal_type': 'bottleneck',
                    'severity': 'high',
                }
            ],
            'trigger_logic': 'all',
        },
        failure_examples={
            'route_state_ids': route_state_ids,
            'decision_episode_ids': [],
            'notes': 'Synthetic replay-io export test.',
        },
        corrective_checklist=['Reduce blocker pressure before reuse.'],
        counterexamples={
            'route_state_ids': ['route-alt'],
            'notes': 'Synthetic counterexample.',
        },
        review={
            'review_status': 'reviewed',
            'reviewer_notes': 'Reviewed for export IO tests.',
            'reviewer_ids': ['reviewer-1'],
        },
        quality={
            'quality_tier': 'green',
            'quality_flags': [],
            'audit_status': 'reviewed',
        },
    )


def _decision_episode_audit_export(tmp_path: Path, **export_overrides: object):
    traces = [_trace('paper-a', 2011, 'pa')]
    route_packet = load_route_packet(_write_json(tmp_path / 'route_packet.json', _packet(traces, cutoff_year=2011)))
    compilation = compile_historical_replay(route_packet, traces, built_at='2026-04-02T04:00:00Z')
    anti_pattern = _anti_pattern_card(
        anti_pattern_id='anti:route-main:matching',
        route_state_ids=[compilation.primary_route_state.route_state_id],
    )
    return build_decision_episode_audit_export(
        route_packet=route_packet,
        route_state=compilation.primary_route_state,
        why_now_case=compilation.why_now_case,
        anti_pattern_cards=[anti_pattern],
        accepted_prior_ids=[],
        accepted_anti_pattern_ids=[anti_pattern.anti_pattern_id],
        route_state_ref='outputs/primary_route_state.json',
        source_replay_bundle_refs={
            'bundle_ref': 'tmp/source-replay',
            'manifest_ref': 'tmp/source-replay/bundle_manifest.json',
            'files': {
                'route_packet': 'inputs/route_packet.json',
                'decision_episode': 'outputs/decision_episode.json',
            },
        },
        source_review_bundle_refs={
            'bundle_ref': 'tmp/source-review',
            'manifest_ref': 'tmp/source-review/bundle_manifest.json',
            'files': {
                'anti_pattern_candidates': 'anti_pattern_candidates.json',
                'candidate_review_summary': 'candidate_review_summary.json',
            },
        },
        hindsight_outcome={
            'outcome_label': 'success',
            'later_evidence_refs': ['future-ref-1'],
            'input_visible': False,
        },
        built_at='2026-04-02T04:05:00Z',
        **export_overrides,
    )


def _corpus_sampling_bundle() -> CorpusSamplingBundle:
    fixed_entry = CorpusInventoryEntry(
        corpus_paper_id='1001',
        display_title='Alpha Paper',
        corpus_relative_ref='1001_Alpha_Paper/1001_Alpha_Paper.md',
        md_path='C:/corpus/1001_Alpha_Paper/1001_Alpha_Paper.md',
        txt_path='C:/corpus/txt/1001_Alpha_Paper.txt',
        preferred_source_path='C:/corpus/1001_Alpha_Paper/1001_Alpha_Paper.md',
        preferred_source_kind='md',
        eligibility_status='eligible',
        neo4j_paper_id='paper:1001',
        neo4j_ingested=True,
    )
    fixed_partner = CorpusInventoryEntry(
        corpus_paper_id='1002',
        display_title='Beta Result',
        corpus_relative_ref='1002_Beta_Result/1002_Beta_Result.md',
        md_path='C:/corpus/1002_Beta_Result/1002_Beta_Result.md',
        preferred_source_path='C:/corpus/1002_Beta_Result/1002_Beta_Result.md',
        preferred_source_kind='md',
        eligibility_status='eligible',
        neo4j_paper_id='paper:1002',
        neo4j_ingested=True,
    )
    random_entry = CorpusInventoryEntry(
        corpus_paper_id='1003',
        display_title='Gamma Scan',
        corpus_relative_ref='txt/1003_Gamma_Scan.txt',
        txt_path='C:/corpus/txt/1003_Gamma_Scan.txt',
        preferred_source_path='C:/corpus/txt/1003_Gamma_Scan.txt',
        preferred_source_kind='txt',
        eligibility_status='eligible',
    )
    ineligible_entry = CorpusInventoryEntry(
        corpus_paper_id='1004',
        display_title='Broken Delta',
        corpus_relative_ref='txt/1004_Broken_Delta.txt',
        txt_path='C:/corpus/txt/1004_Broken_Delta.txt',
        eligibility_status='corpus_health_failure',
    )
    return CorpusSamplingBundle(
        built_at='2026-04-03T01:00:00Z',
        corpus_root='\\\\server\\share\\output',
        inventory_entries=[fixed_entry, fixed_partner, random_entry, ineligible_entry],
        corpus_health_failures=[
            CorpusHealthIssue(
                issue_type='unreadable_file',
                source_path='C:/corpus/txt/1004_Broken_Delta.txt',
                corpus_relative_ref='txt/1004_Broken_Delta.txt',
                detail='permission denied',
                exception_type='PermissionError',
            )
        ],
        fixed_regression_batch=CorpusSamplingBatch(
            batch_id='phase7-fixed-regression',
            sampling_mode='fixed_regression',
            built_at='2026-04-03T01:00:00Z',
            requested_count=2,
            selected=[fixed_entry, fixed_partner],
            selected_ids=['1001', '1002'],
            fixed_manifest_ref='docs/replay/corpus_sampling/phase7-fixed-regression-set.json',
        ),
        random_exploration_batch=CorpusSamplingBatch(
            batch_id='phase7-random-exploration',
            sampling_mode='random_exploration',
            built_at='2026-04-03T01:00:00Z',
            requested_count=1,
            selected=[random_entry],
            selected_ids=['1003'],
            seed=13,
            exclusions=[{'corpus_paper_id': '1001', 'reason': 'fixed_regression_exclusion'}],
        ),
        fixed_manifest_ref='docs/replay/corpus_sampling/phase7-fixed-regression-set.json',
        seed=13,
        neo4j_lookup_status='unavailable',
    )


def _sampled_iteration() -> SampledSinglePaperIterationResult:
    return SampledSinglePaperIterationResult(
        built_at='2026-04-03T05:00:00Z',
        iteration_label='baseline-cycle-01',
        sampling_bundle_dir='tmp/phase7_corpus_sampling_baseline',
        sampling_bundle_manifest_ref='tmp/phase7_corpus_sampling_baseline/bundle_manifest.json',
        fixed_manifest_ref='docs/replay/corpus_sampling/phase7-fixed-regression-set.json',
        seed=7,
        neo4j_lookup_status='unavailable',
        fixed_selected_count=2,
        random_selected_count=1,
        fixed_results=[
            FixedRegressionSampledPaperResult(
                corpus_paper_id='1001',
                display_title='Alpha',
                corpus_relative_ref='txt/1001_alpha.txt',
                preferred_source_path='C:/corpus/1001_alpha.txt',
                preferred_source_kind='txt',
                iteration_label='baseline-cycle-01',
                paper_id='doi:10.1000/alpha',
                trace_id='trace:alpha',
                source_path='C:/corpus/1001_alpha.txt',
                source_kind='txt',
                quality_report={'quality_tier': 'green', 'gate_passed': True},
                trace_quality={'quality_tier': 'green', 'audit_status': 'reviewed'},
                artifact_refs={'paper_logic_trace': 'tmp/phase8/paper_artifacts/fixed/1001/paper_logic_trace.json'},
                citations={'refs': 1, 'cites_resolved': 1, 'cites_unresolved': 0},
                llm={'purposes': 1, 'moves': 1, 'gate_passed': True, 'quality_tier': 'green'},
            ),
            FixedRegressionSampledPaperResult(
                corpus_paper_id='1002',
                display_title='Beta',
                corpus_relative_ref='txt/1002_beta.txt',
                preferred_source_path='C:/corpus/1002_beta.txt',
                preferred_source_kind='txt',
                iteration_label='baseline-cycle-01',
                paper_id='doi:10.1000/beta',
                trace_id='trace:beta',
                source_path='C:/corpus/1002_beta.txt',
                source_kind='txt',
                quality_report={'quality_tier': 'red', 'gate_passed': False},
                trace_quality={'quality_tier': 'red', 'audit_status': 'eligible'},
                artifact_refs={'paper_logic_trace': 'tmp/phase8/paper_artifacts/fixed/1002/paper_logic_trace.json'},
                citations={'refs': 1, 'cites_resolved': 1, 'cites_unresolved': 0},
                llm={'purposes': 1, 'moves': 1, 'gate_passed': False, 'quality_tier': 'red'},
                skipped_canonical_write=True,
            ),
        ],
        random_results=[
            RandomExplorationSampledPaperResult(
                corpus_paper_id='2001',
                display_title='Gamma',
                corpus_relative_ref='txt/2001_gamma.txt',
                preferred_source_path='C:/corpus/2001_gamma.txt',
                preferred_source_kind='txt',
                iteration_label='baseline-cycle-01',
                paper_id='doi:10.1000/gamma',
                trace_id='trace:gamma',
                source_path='C:/corpus/2001_gamma.txt',
                source_kind='txt',
                quality_report={'quality_tier': 'yellow', 'gate_passed': True},
                trace_quality={'quality_tier': 'yellow', 'audit_status': 'eligible'},
                artifact_refs={'paper_logic_trace': 'tmp/phase8/paper_artifacts/random/2001/paper_logic_trace.json'},
                citations={'refs': 1, 'cites_resolved': 1, 'cites_unresolved': 0},
                llm={'purposes': 1, 'moves': 1, 'gate_passed': True, 'quality_tier': 'yellow'},
            )
        ],
        availability_issues=[
            SampledPaperAvailabilityIssue(
                corpus_paper_id='2002',
                display_title='Delta',
                cohort='random_exploration',
                selection_mode='random_exploration',
                corpus_relative_ref='txt/2002_delta.txt',
                preferred_source_path='C:/corpus/2002_delta.txt',
                preferred_source_kind='txt',
                iteration_label='baseline-cycle-01',
                execution_status='source_missing',
                error_message='missing source',
                error_type='FileNotFoundError',
            )
        ],
    )


def test_load_route_packet_and_trace_dir_round_trip(tmp_path: Path) -> None:
    traces = [_trace('paper-a', 2011, 'pa')]
    trace_dir = tmp_path / 'traces'
    trace_dir.mkdir()
    for trace in traces:
        _write_json(trace_dir / f'{trace.paper_metadata.paper_id}.json', trace.model_dump(mode='json'))

    packet_path = _write_json(tmp_path / 'route_packet.json', _packet(traces, cutoff_year=2011))

    loaded_packet = load_route_packet(packet_path)
    loaded_traces = load_paper_logic_traces(trace_dir=trace_dir)

    ensure_packet_trace_coverage(loaded_packet, loaded_traces)

    assert loaded_packet.packet_id == 'packet-2011'
    assert [trace.trace_id for trace in loaded_traces] == [traces[0].trace_id]


def test_ensure_packet_trace_coverage_rejects_missing_trace_id(tmp_path: Path) -> None:
    traces = [_trace('paper-a', 2011, 'pa')]
    trace_path = _write_json(tmp_path / 'paper-a.json', traces[0].model_dump(mode='json'))
    packet_path = _write_json(tmp_path / 'route_packet.json', _packet(traces, cutoff_year=2011, missing_trace_id=True))

    loaded_packet = load_route_packet(packet_path)
    loaded_traces = load_paper_logic_traces(trace_files=[trace_path])

    with pytest.raises(ValueError, match='missing trace_id'):
        ensure_packet_trace_coverage(loaded_packet, loaded_traces)


def test_load_route_states_round_trip(tmp_path: Path) -> None:
    traces = [_trace('paper-a', 2011, 'pa')]
    route_packet = load_route_packet(_write_json(tmp_path / 'route_packet.json', _packet(traces, cutoff_year=2011)))
    compilation = compile_historical_replay(route_packet, traces, built_at='2026-04-02T01:20:00Z')

    route_state_path = _write_json(tmp_path / 'route_state.json', compilation.primary_route_state.model_dump(mode='json'))
    loaded_route_states = load_route_states([route_state_path])

    assert [route_state.route_state_id for route_state in loaded_route_states] == [compilation.primary_route_state.route_state_id]


def test_load_historical_environment_snapshot_round_trip(tmp_path: Path) -> None:
    snapshot_path = _write_json(tmp_path / 'historical_environment_snapshot.json', _l1_snapshot(cutoff_year=2011))

    snapshot = load_historical_environment_snapshot(snapshot_path)

    assert snapshot.snapshot_id == 'imagenet_2011_snapshot'
    assert snapshot.benchmark_timeline[0].label == 'wn18rr'


def test_write_replay_bundle_writes_expected_json_files(tmp_path: Path) -> None:
    traces = [_trace('paper-a', 2011, 'pa')]
    route_packet = load_route_packet(_write_json(tmp_path / 'route_packet.json', _packet(traces, cutoff_year=2011)))
    compilation = compile_historical_replay(route_packet, traces, built_at='2026-04-02T01:20:00Z')

    written_files = write_replay_bundle(
        tmp_path / 'bundle',
        route_packet=route_packet,
        traces=traces,
        compilation=compilation,
        reviewer_ids=['reviewer-1'],
        metadata={'runner': 'pytest'},
    )

    expected_names = {
        'bundle_manifest',
        'replay_summary',
        'replay_inspection',
        'route_packet',
        'paper_logic_traces',
        'support_route_states',
        'alternative_route_states',
        'held_out_route_states',
        'primary_route_state',
        'why_now_case',
        'route_comparison_cases',
        'decision_prior_card',
        'decision_episode',
    }
    assert expected_names == set(written_files)
    assert written_files['bundle_manifest'].is_file()
    assert written_files['replay_summary'].is_file()
    assert written_files['replay_inspection'].is_file()
    assert written_files['route_packet'].is_file()
    assert written_files['decision_episode'].is_file()

    summary_payload = json.loads(written_files['replay_summary'].read_text(encoding='utf-8'))
    inspection_payload = json.loads(written_files['replay_inspection'].read_text(encoding='utf-8'))
    manifest_payload = json.loads(written_files['bundle_manifest'].read_text(encoding='utf-8'))

    assert summary_payload['ready_for_pilot'] is False
    assert summary_payload['replay_quality_tier'] == 'red'
    assert 'support_cluster_too_small' in summary_payload['quality_flags']
    assert summary_payload['failure_record_count'] >= 4
    assert 'l2' in summary_payload['failure_counts_by_layer']
    assert summary_payload['failure_counts_by_blocking']['blocking'] >= 1
    assert inspection_payload['stage_quality']['decision_episode']['quality_tier'] in {'yellow', 'red', 'green'}
    assert inspection_payload['failure_records']
    assert {'failure_id', 'layer', 'stage', 'failure_code', 'blocking', 'repair_target'}.issubset(
        inspection_payload['failure_records'][0]
    )
    assert inspection_payload['route_state_package_validation'] is None
    assert manifest_payload['files']['decision_episode'] == 'outputs/decision_episode.json'


def test_write_replay_bundle_writes_l1_snapshot_when_provided(tmp_path: Path) -> None:
    traces = [_trace('paper-a', 2011, 'pa')]
    route_packet = load_route_packet(_write_json(tmp_path / 'route_packet.json', _packet(traces, cutoff_year=2011)))
    historical_environment_snapshot = load_historical_environment_snapshot(
        _write_json(tmp_path / 'historical_environment_snapshot.json', _l1_snapshot(cutoff_year=2011))
    )
    compilation = compile_historical_replay(
        route_packet,
        traces,
        l1_snapshot=historical_environment_snapshot,
        built_at='2026-04-02T01:20:00Z',
    )

    written_files = write_replay_bundle(
        tmp_path / 'bundle',
        route_packet=route_packet,
        traces=traces,
        compilation=compilation,
        historical_environment_snapshot=historical_environment_snapshot,
        reviewer_ids=['reviewer-1'],
        metadata={'runner': 'pytest'},
    )

    summary_payload = json.loads(written_files['replay_summary'].read_text(encoding='utf-8'))
    manifest_payload = json.loads(written_files['bundle_manifest'].read_text(encoding='utf-8'))

    assert written_files['historical_environment_snapshot'].is_file()
    assert summary_payload['l1_snapshot_id'] == 'imagenet_2011_snapshot'
    assert manifest_payload['files']['historical_environment_snapshot'] == 'inputs/historical_environment_snapshot.json'


def test_run_replay_pilot_cli_outputs_summary(tmp_path: Path) -> None:
    traces = [_trace('paper-a', 2011, 'pa')]
    trace_path = _write_json(tmp_path / 'paper-a.json', traces[0].model_dump(mode='json'))
    packet_path = _write_json(tmp_path / 'route_packet.json', _packet(traces, cutoff_year=2011))
    l1_snapshot_path = _write_json(tmp_path / 'historical_environment_snapshot.json', _l1_snapshot(cutoff_year=2011))
    output_dir = tmp_path / 'replay-output'
    script_path = Path(__file__).resolve().parents[1] / 'scripts' / 'run_replay_pilot.py'

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            '--packet',
            str(packet_path),
            '--trace-file',
            str(trace_path),
            '--l1-snapshot',
            str(l1_snapshot_path),
            '--output-dir',
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    assert result.returncode == 0, result.stderr
    summary_payload = json.loads(result.stdout)

    assert summary_payload['ready_for_pilot'] is False
    assert summary_payload['l1_snapshot_id'] == 'imagenet_2011_snapshot'
    assert summary_payload['selected_comparison_case_id'] is None
    assert 'quality_flags' in summary_payload
    assert 'failure_counts_by_layer' in summary_payload
    assert 'failure_counts_by_blocking' in summary_payload
    assert (output_dir / 'bundle_manifest.json').is_file()
    assert (output_dir / 'replay_inspection.json').is_file()


def test_write_prior_candidate_review_bundle_writes_expected_files(tmp_path: Path) -> None:
    support_route_states = load_route_states(
        [
            _write_json(
                tmp_path / 'support-a.json',
                compile_historical_replay(
                    load_route_packet(_write_json(tmp_path / 'route_packet.json', _packet([_trace('paper-a', 2011, 'pa')], cutoff_year=2011))),
                    [_trace('paper-a', 2011, 'pa')],
                    route_state_id='support-a',
                    built_at='2026-04-02T03:20:00Z',
                ).primary_route_state.model_dump(mode='json'),
            ),
            _write_json(
                tmp_path / 'support-b.json',
                compile_historical_replay(
                    load_route_packet(_write_json(tmp_path / 'route_packet-b.json', _packet([_trace('paper-b', 2011, 'pb')], cutoff_year=2011))),
                    [_trace('paper-b', 2011, 'pb')],
                    route_state_id='support-b',
                    built_at='2026-04-02T03:25:00Z',
                ).primary_route_state.model_dump(mode='json'),
            ),
        ]
    )
    registry = build_prior_candidate_registry(
        support_route_states=support_route_states,
        alternative_route_states=[],
        held_out_route_states=[],
        built_at='2026-04-02T03:30:00Z',
        package_id='test-package',
    )

    written_files = write_prior_candidate_review_bundle(
        tmp_path / 'prior-review',
        registry=registry,
        metadata={'runner': 'pytest'},
    )
    summary_payload = json.loads(written_files['candidate_review_summary'].read_text(encoding='utf-8'))

    assert {'bundle_manifest', 'prior_candidates', 'anti_pattern_candidates', 'clusters', 'candidate_review_summary'} == set(
        written_files
    )
    assert summary_payload['package_id'] == 'test-package'
    assert 'cluster_ids' in summary_payload
    assert 'accepted_prior_ids' in summary_payload


def test_build_decision_episode_export_summary_and_inspection_expose_visibility_buckets(tmp_path: Path) -> None:
    export = _decision_episode_audit_export(tmp_path)

    summary_payload = build_decision_episode_export_summary(export=export)
    inspection_payload = build_decision_episode_export_inspection(export=export)

    assert summary_payload['accepted_prior_ids'] == []
    assert summary_payload['accepted_anti_pattern_count'] == 1
    assert summary_payload['accepted_but_unselected_antipattern_count'] == 0
    assert summary_payload['visibility_bucket_counts']['label_eval_only_refs'] == 1
    assert inspection_payload['source_bundles']['replay_bundle']['manifest_ref'] == 'tmp/source-replay/bundle_manifest.json'
    assert inspection_payload['anti_pattern_selection']['selected_antipattern_ids'] == ['anti:route-main:matching']
    assert inspection_payload['anti_pattern_selection']['accepted_but_unselected_antipatterns'] == []
    assert inspection_payload['visibility_buckets']['visible_input_refs']
    assert inspection_payload['visibility_buckets']['label_eval_only_refs'] == ['hindsight_evidence:future-ref-1']


def test_build_decision_episode_export_summary_and_inspection_include_review_metadata(tmp_path: Path) -> None:
    export = _decision_episode_audit_export(
        tmp_path,
        review_status='reviewed',
        training_acceptance_verdict='needs_revision',
        reviewer_ids=['reviewer-1', 'reviewer-2'],
        reviewed_at='2026-04-02T04:06:00Z',
        rationale='Training-facing review found residual prior grounding defects.',
        residual_defects=['weak_prior_support'],
        section_reviews={
            'evidence_pack': {
                'review_status': 'reviewed',
                'training_acceptance_verdict': 'accepted',
                'reviewer_ids': ['reviewer-1'],
                'reviewed_at': '2026-04-02T04:06:00Z',
                'rationale': 'Evidence pack is explicit.',
                'residual_defects': [],
            },
            'review_labels': {
                'review_status': 'reviewed',
                'training_acceptance_verdict': 'needs_revision',
                'reviewer_ids': ['reviewer-1', 'reviewer-2'],
                'reviewed_at': '2026-04-02T04:06:00Z',
                'rationale': 'Verdict labels are present but still conservative.',
                'residual_defects': ['weak_prior_support'],
            },
        },
    )

    summary_payload = build_decision_episode_export_summary(export=export)
    inspection_payload = build_decision_episode_export_inspection(export=export)

    assert summary_payload['training_acceptance_verdict'] == 'needs_revision'
    assert summary_payload['reviewer_ids'] == ['reviewer-1', 'reviewer-2']
    assert summary_payload['residual_defects'] == ['weak_prior_support']
    assert summary_payload['accepted_but_unselected_prior_count'] == 0
    assert summary_payload['accepted_but_unselected_antipattern_count'] == 0
    assert set(summary_payload['section_reviews']) == {
        'evidence_pack',
        'route_synthesis',
        'why_now',
        'route_comparison',
        'priors_antipatterns',
        'minimal_attack_path',
        'final_decision',
        'review_labels',
    }
    assert summary_payload['section_reviews']['evidence_pack']['training_acceptance_verdict'] == 'accepted'
    assert inspection_payload['review']['review_status'] == 'reviewed'
    assert inspection_payload['review']['section_reviews']['review_labels']['residual_defects'] == ['weak_prior_support']
    assert inspection_payload['training_sections']['route_synthesis']['route_state_id'] == export.route_state_snapshot.route_state_id
    assert inspection_payload['training_sections']['why_now']['route_state_id'] == export.route_state_snapshot.route_state_id


def test_write_decision_episode_export_bundle_writes_expected_files(tmp_path: Path) -> None:
    export = _decision_episode_audit_export(tmp_path)

    written_files = write_decision_episode_export_bundle(
        tmp_path / 'export-bundle',
        export=export,
        metadata={'runner': 'pytest'},
    )

    assert {
        'best_cycle_selection',
        'bundle_manifest',
        'decision_episode',
        'export_inspection',
        'export_summary',
        'final_decision_view',
        'prior_antipattern_view',
        'route_comparison_view',
        'route_synthesis_view',
        'training_view',
        'why_now_view',
    } == set(written_files)
    assert written_files['bundle_manifest'].is_file()
    assert written_files['export_summary'].is_file()
    assert written_files['export_inspection'].is_file()
    assert written_files['decision_episode'].is_file()
    assert written_files['training_view'].is_file()
    assert written_files['route_synthesis_view'].is_file()
    assert written_files['why_now_view'].is_file()
    assert written_files['route_comparison_view'].is_file()
    assert written_files['prior_antipattern_view'].is_file()
    assert written_files['final_decision_view'].is_file()
    assert written_files['best_cycle_selection'].is_file()
    assert all(str(path).startswith(str((tmp_path / 'export-bundle').resolve())) for path in written_files.values())

    manifest_payload = json.loads(written_files['bundle_manifest'].read_text(encoding='utf-8'))
    summary_payload = json.loads(written_files['export_summary'].read_text(encoding='utf-8'))
    inspection_payload = json.loads(written_files['export_inspection'].read_text(encoding='utf-8'))
    training_view_payload = json.loads(written_files['training_view'].read_text(encoding='utf-8'))
    route_synthesis_view_payload = json.loads(written_files['route_synthesis_view'].read_text(encoding='utf-8'))
    why_now_view_payload = json.loads(written_files['why_now_view'].read_text(encoding='utf-8'))
    route_comparison_view_payload = json.loads(written_files['route_comparison_view'].read_text(encoding='utf-8'))
    prior_antipattern_view_payload = json.loads(written_files['prior_antipattern_view'].read_text(encoding='utf-8'))
    final_decision_view_payload = json.loads(written_files['final_decision_view'].read_text(encoding='utf-8'))
    best_cycle_selection_payload = json.loads(written_files['best_cycle_selection'].read_text(encoding='utf-8'))

    assert manifest_payload['files']['decision_episode'] == 'outputs/decision_episode.json'
    assert manifest_payload['files']['training_view'] == 'outputs/training_view.json'
    assert manifest_payload['files']['route_synthesis_view'] == 'outputs/route_synthesis_view.json'
    assert manifest_payload['files']['why_now_view'] == 'outputs/why_now_view.json'
    assert manifest_payload['files']['route_comparison_view'] == 'outputs/route_comparison_view.json'
    assert manifest_payload['files']['prior_antipattern_view'] == 'outputs/prior_antipattern_view.json'
    assert manifest_payload['files']['final_decision_view'] == 'outputs/final_decision_view.json'
    assert manifest_payload['files']['best_cycle_selection'] == 'best_cycle_selection.json'
    assert manifest_payload['source_replay_bundle_refs']['manifest_ref'] == 'tmp/source-replay/bundle_manifest.json'
    assert manifest_payload['accepted_prior_ids'] == []
    assert manifest_payload['accepted_anti_pattern_ids'] == ['anti:route-main:matching']
    assert manifest_payload['selected_iteration_label'] == 'export-bundle'
    assert summary_payload['audit_posture'] == 'audit_grade_pilot'
    assert summary_payload['visibility_bucket_counts']['visible_input_refs'] == len(export.visible_input_refs)
    assert summary_payload['accepted_but_unselected_antipatterns'] == []
    assert inspection_payload['visibility_buckets']['audit_only_refs']
    assert inspection_payload['visibility_buckets']['label_eval_only_refs'] == ['hindsight_evidence:future-ref-1']
    assert training_view_payload['sections']['route_synthesis']['route_state']['route_state_id'] == export.route_state_snapshot.route_state_id
    assert training_view_payload['sections']['priors_antipatterns']['accepted_but_unselected_priors'] == []
    assert training_view_payload['sections']['priors_antipatterns']['accepted_but_unselected_antipatterns'] == []
    assert route_synthesis_view_payload['sections'] == {
        'route_synthesis': training_view_payload['sections']['route_synthesis']
    }
    assert why_now_view_payload['sections'] == {'why_now': training_view_payload['sections']['why_now']}
    assert route_comparison_view_payload['sections'] == {
        'route_comparison': training_view_payload['sections']['route_comparison']
    }
    assert prior_antipattern_view_payload['sections'] == {
        'priors_antipatterns': training_view_payload['sections']['priors_antipatterns']
    }
    assert final_decision_view_payload['sections'] == {
        'final_decision': training_view_payload['sections']['final_decision']
    }
    for task_view_payload in (
        route_synthesis_view_payload,
        why_now_view_payload,
        route_comparison_view_payload,
        prior_antipattern_view_payload,
        final_decision_view_payload,
    ):
        assert task_view_payload['visibility_buckets'] == training_view_payload['visibility_buckets']
        assert task_view_payload['selected_iteration_label'] == training_view_payload['selected_iteration_label']
        assert task_view_payload['review']['review_status'] == training_view_payload['review']['review_status']
        assert task_view_payload['review']['training_acceptance_verdict'] == training_view_payload['review'][
            'training_acceptance_verdict'
        ]
        assert task_view_payload['source_bundle_refs'] == training_view_payload['source_bundle_refs']
    assert best_cycle_selection_payload['selected_iteration_label'] == 'export-bundle'
    assert best_cycle_selection_payload['recommendation_evidence_refs']


def test_write_final_training_dataset_bundle_writes_expected_scaffold(tmp_path: Path) -> None:
    export = _decision_episode_audit_export(tmp_path)
    export_bundle_files = write_decision_episode_export_bundle(
        tmp_path / 'cycle4-stable' / 'export_bundle',
        export=export,
        metadata={'runner': 'pytest'},
        selected_iteration_label='cycle4-stable',
    )
    primary_cycle_root = tmp_path / 'cycle4-stable'
    supporting_cycle_root = tmp_path / 'cycle3-best'
    supporting_cycle_root.mkdir(parents=True)

    written_files = write_final_training_dataset_bundle(
        tmp_path / 'final-dataset',
        primary_cycle_label='cycle4-stable',
        supporting_cycle_labels=['cycle3-best'],
        source_cycle_roots={
            'cycle4-stable': primary_cycle_root,
            'cycle3-best': supporting_cycle_root,
        },
        source_export_bundles={
            'cycle4-stable': primary_cycle_root / 'export_bundle',
        },
        task_training_views={
            'training_view': export_bundle_files['training_view'],
            'route_synthesis_view': export_bundle_files['route_synthesis_view'],
            'why_now_view': export_bundle_files['why_now_view'],
            'route_comparison_view': export_bundle_files['route_comparison_view'],
            'prior_antipattern_view': export_bundle_files['prior_antipattern_view'],
            'final_decision_view': export_bundle_files['final_decision_view'],
        },
        schema_refs={
            'decision_episode_schema': Path('docs/superpowers/specs/2026-04-01-logickg-decision-prior-and-episode-schema.md'),
            'route_comparison_schema': Path('docs/superpowers/specs/2026-04-01-logickg-why-now-and-route-comparison-schema.md'),
        },
        primary_recommendation_id='packet_construction',
        recommendation_evidence_refs=['comparison_summary:C:/tmp/phase16/comparison_summary.json'],
        residual_risk_source=export_bundle_files['best_cycle_selection'],
        metadata={'runner': 'pytest'},
    )

    assert {
        'bundle_manifest',
        'dataset_manifest',
        'dataset_summary',
        'training_views_index',
    } == set(written_files)

    dataset_manifest_payload = json.loads(written_files['dataset_manifest'].read_text(encoding='utf-8'))
    dataset_summary_payload = json.loads(written_files['dataset_summary'].read_text(encoding='utf-8'))
    training_views_index_payload = json.loads(written_files['training_views_index'].read_text(encoding='utf-8'))
    bundle_manifest_payload = json.loads(written_files['bundle_manifest'].read_text(encoding='utf-8'))

    assert written_files['dataset_manifest'].is_file()
    assert written_files['dataset_summary'].is_file()
    assert written_files['training_views_index'].is_file()
    assert written_files['bundle_manifest'].is_file()
    assert dataset_manifest_payload['primary_cycle_label'] == 'cycle4-stable'
    assert dataset_manifest_payload['supporting_cycle_labels'] == ['cycle3-best']
    assert dataset_manifest_payload['primary_recommendation_id'] == 'packet_construction'
    assert dataset_manifest_payload['source_cycle_roots']['cycle4-stable'] == '../cycle4-stable'
    assert dataset_manifest_payload['source_export_bundles']['cycle4-stable'] == '../cycle4-stable/export_bundle'
    assert dataset_manifest_payload['schema_refs']['decision_episode_schema'].endswith(
        'docs/superpowers/specs/2026-04-01-logickg-decision-prior-and-episode-schema.md'
    )
    assert dataset_manifest_payload['residual_risk_source'] == '../cycle4-stable/export_bundle/best_cycle_selection.json'
    assert set(training_views_index_payload) == {
        'training_view',
        'route_synthesis_view',
        'why_now_view',
        'route_comparison_view',
        'prior_antipattern_view',
        'final_decision_view',
    }
    assert training_views_index_payload['training_view'] == '../cycle4-stable/export_bundle/outputs/training_view.json'
    assert dataset_manifest_payload['task_training_views'] == training_views_index_payload
    assert dataset_summary_payload['task_training_view_count'] == 6
    assert dataset_summary_payload['training_views_index_ref'] == 'outputs/training_views_index.json'
    assert bundle_manifest_payload['files']['dataset_manifest'] == 'dataset_manifest.json'
    assert bundle_manifest_payload['files']['dataset_summary'] == 'dataset_summary.json'
    assert bundle_manifest_payload['files']['training_views_index'] == 'outputs/training_views_index.json'


def test_write_corpus_sampling_bundle_writes_expected_files_and_separates_buckets(tmp_path: Path) -> None:
    bundle = _corpus_sampling_bundle()

    summary_payload = build_corpus_sampling_summary(bundle=bundle)
    inspection_payload = build_corpus_sampling_inspection(bundle=bundle)
    written_files = write_corpus_sampling_bundle(
        tmp_path / 'sampling-bundle',
        bundle=bundle,
        metadata={'runner': 'pytest'},
    )

    assert summary_payload['inventory_entry_count'] == 4
    assert summary_payload['eligible_entry_count'] == 3
    assert summary_payload['selected_with_neo4j_metadata_count'] == 2
    assert summary_payload['selected_without_neo4j_metadata_count'] == 1
    assert inspection_payload['fixed_regression_ids'] == ['1001', '1002']
    assert inspection_payload['random_exploration_ids'] == ['1003']
    assert inspection_payload['corpus_health_failures'][0]['issue_type'] == 'unreadable_file'
    assert inspection_payload['selected_but_downstream_unavailable'][0]['reason'] == 'neo4j_lookup_unavailable'

    assert {
        'bundle_manifest',
        'sampling_summary',
        'sampling_inspection',
        'fixed_regression_batch',
        'random_exploration_batch',
        'corpus_health_failures',
    } == set(written_files)

    manifest_payload = json.loads(written_files['bundle_manifest'].read_text(encoding='utf-8'))
    assert manifest_payload['seed'] == 13
    assert manifest_payload['fixed_manifest_ref'] == 'docs/replay/corpus_sampling/phase7-fixed-regression-set.json'
    assert manifest_payload['neo4j_lookup_status'] == 'unavailable'
    assert manifest_payload['files']['fixed_regression_batch'] == 'outputs/fixed_regression_batch.json'


def test_write_sampled_l2_iteration_bundle_writes_expected_files_and_separates_availability_issues(tmp_path: Path) -> None:
    iteration = _sampled_iteration()

    summary_payload = build_sampled_l2_iteration_summary(iteration=iteration)
    inspection_payload = build_sampled_l2_iteration_inspection(iteration=iteration)
    written_files = write_sampled_l2_iteration_bundle(
        tmp_path / 'sampled-l2-bundle',
        iteration=iteration,
        metadata={'runner': 'pytest'},
    )

    assert summary_payload['fixed_selected_count'] == 2
    assert summary_payload['random_selected_count'] == 1
    assert summary_payload['executed_count'] == 3
    assert summary_payload['availability_issue_count'] == 1
    assert summary_payload['fixed_green_count'] == 1
    assert summary_payload['fixed_red_count'] == 1
    assert summary_payload['random_yellow_count'] == 1
    assert inspection_payload['availability_issues'][0]['execution_status'] == 'source_missing'

    assert {
        'bundle_manifest',
        'iteration_summary',
        'iteration_inspection',
        'fixed_regression_results',
        'random_exploration_results',
        'availability_issues',
    } == set(written_files)

    manifest_payload = json.loads(written_files['bundle_manifest'].read_text(encoding='utf-8'))
    fixed_results_payload = json.loads(written_files['fixed_regression_results'].read_text(encoding='utf-8'))
    random_results_payload = json.loads(written_files['random_exploration_results'].read_text(encoding='utf-8'))
    availability_payload = json.loads(written_files['availability_issues'].read_text(encoding='utf-8'))

    assert manifest_payload['files']['fixed_regression_results'] == 'outputs/fixed_regression_results.json'
    assert manifest_payload['files']['availability_issues'] == 'outputs/availability_issues.json'
    assert [row['corpus_paper_id'] for row in fixed_results_payload] == ['1001', '1002']
    assert [row['corpus_paper_id'] for row in random_results_payload] == ['2001']
    assert [row['corpus_paper_id'] for row in availability_payload] == ['2002']


def test_write_sampled_l2_comparison_bundle_writes_expected_counts_and_owner_buckets(tmp_path: Path) -> None:
    comparison = compare_sampled_l2_iterations(_sampled_iteration(), None)

    summary_payload = build_sampled_l2_comparison_summary(comparison=comparison)
    inspection_payload = build_sampled_l2_comparison_inspection(comparison=comparison)
    written_files = write_sampled_l2_comparison_bundle(
        tmp_path / 'sampled-l2-bundle',
        comparison=comparison,
    )

    assert summary_payload['baseline_only'] is True
    assert summary_payload['fixed_verdict_counts'] == {
        'stable_pass': 1,
        'recurring_failure': 1,
        'new_regression': 0,
        'improved': 0,
        'availability_only': 0,
    }
    assert summary_payload['random_verdict_counts'] == {
        'new_edge_case': 1,
        'repeated_random_failure': 0,
        'random_improved': 0,
        'stable_random_pass': 0,
        'availability_only': 1,
    }
    assert inspection_payload['fixed_comparisons'][0]['verdict'] == 'stable_pass'

    comparison_summary_payload = json.loads(written_files['comparison_summary'].read_text(encoding='utf-8'))
    comparison_inspection_payload = json.loads(written_files['comparison_inspection'].read_text(encoding='utf-8'))

    assert comparison_summary_payload['fixed_verdict_counts']['recurring_failure'] == 1
    assert comparison_inspection_payload['random_comparisons'][1]['verdict'] == 'availability_only'


def _iteration_priority_bundle_models() -> tuple[IterationPrioritySummary, IterationPriorityInspection]:
    source_refs = IterationPrioritySourceRefs(
        phase8_summary_path='tmp/phase8/comparison_summary.json',
        phase8_inspection_path='tmp/phase8/comparison_inspection.json',
        phase10_verification_path='.planning/phases/10/10-VERIFICATION.md',
        phase10_report_path='docs/replay/reports/phase10.md',
        phase10_mode='fallback',
        fallback_used=True,
    )
    recommendations = [
        IterationPriorityRecommendation(
            id='packet_construction',
            rank=1,
            title='Deepen bounded packet construction',
            why_now='New package blockers appear before any new L2 regression.',
            score=180,
            supporting_owner_buckets=['relation_assembly', 'slot_recovery'],
            supporting_blocker_stages=['package_validation', 'replay'],
            evidence=['package blockers: support_cluster_too_small', 'replay l2 delta: 0'],
        ),
        IterationPriorityRecommendation(
            id='l4_aggregation',
            rank=2,
            title='Tune downstream L4 aggregation',
            why_now='Decision-prior failures remain downstream follow-up work.',
            score=110,
            supporting_owner_buckets=['relation_assembly', 'slot_recovery'],
            supporting_blocker_stages=['replay', 'prior_induction'],
            evidence=['replay l3_l4 delta: 2'],
        ),
        IterationPriorityRecommendation(
            id='l2_extraction',
            rank=3,
            title='Run a targeted L2 extraction pass',
            why_now='Phase 8 still exposes recurring owner buckets.',
            score=72,
            supporting_owner_buckets=['relation_assembly', 'slot_recovery'],
            supporting_blocker_stages=['phase8_owner_queue'],
            evidence=['phase8 lead owners: relation_assembly, slot_recovery'],
        ),
    ]
    summary = IterationPrioritySummary(
        source_refs=source_refs,
        phase8_iteration_label='baseline-cycle-01',
        baseline_only=True,
        packet_id='phase9_comp_mech_2021_packet_01',
        cutoff_year=2021,
        current_recommendation='packet_construction',
        primary_recommendation_id='packet_construction',
        phase8_fixed_verdict_counts={'recurring_failure': 7, 'stable_pass': 3},
        phase8_random_verdict_counts={'new_edge_case': 2, 'stable_random_pass': 3},
        phase10_stage_surfaces={
            'package': {'current': {'quality_tier': 'red'}},
            'replay': {'delta': {'failure_counts_by_layer_delta': {'l2': 0, 'l3_l4': 2}}},
        },
        recommendations=recommendations,
        notes=['fallback used'],
    )
    inspection = IterationPriorityInspection(
        source_refs=source_refs,
        phase8_summary={'iteration_label': 'baseline-cycle-01'},
        phase8_inspection={'owner_buckets': ['relation_assembly', 'slot_recovery']},
        phase10_surface={'current_recommendation': 'packet_construction'},
        ranking_signals={
            'replay_l2_delta': 0,
            'replay_l3_l4_delta': 2,
            'supporting_owner_buckets': ['relation_assembly', 'slot_recovery'],
        },
        recommendations=recommendations,
    )
    return summary, inspection


def test_write_iteration_priority_bundle_writes_expected_manifest_refs(tmp_path: Path) -> None:
    summary, inspection = _iteration_priority_bundle_models()

    summary_payload = build_iteration_priority_summary_payload(summary=summary)
    inspection_payload = build_iteration_priority_inspection_payload(inspection=inspection)
    written_files = write_iteration_priority_bundle(
        tmp_path / 'iteration-priority',
        summary=summary,
        inspection=inspection,
        metadata={'phase': '11'},
    )

    assert summary_payload['primary_recommendation_id'] == 'packet_construction'
    assert inspection_payload['ranking_signals']['replay_l2_delta'] == 0
    assert set(written_files) == {'prioritization_summary', 'prioritization_inspection', 'bundle_manifest'}
    assert written_files['prioritization_summary'].is_file()
    assert written_files['prioritization_inspection'].is_file()
    assert written_files['bundle_manifest'].is_file()

    manifest_payload = json.loads(written_files['bundle_manifest'].read_text(encoding='utf-8'))

    assert manifest_payload['files']['prioritization_summary'] == 'outputs/prioritization_summary.json'
    assert manifest_payload['files']['prioritization_inspection'] == 'outputs/prioritization_inspection.json'
    assert manifest_payload['source_refs']['phase10_mode'] == 'fallback'
