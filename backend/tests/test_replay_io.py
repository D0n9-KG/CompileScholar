from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from app.paper_logic_trace.derived_views import build_derived_views
from app.paper_logic_trace.models import CanonicalCore, MentionValue, PaperLogicTrace, PaperMetadata, ResearchMove, SlotProvenance
from app.research_logic import (
    build_prior_candidate_registry,
    compile_historical_replay,
    ensure_packet_trace_coverage,
    load_historical_environment_snapshot,
    load_paper_logic_traces,
    load_route_packet,
    load_route_states,
    write_prior_candidate_review_bundle,
    write_replay_bundle,
)


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
