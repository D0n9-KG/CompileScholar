from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from app.paper_logic_trace.derived_views import build_derived_views
from app.paper_logic_trace.models import CanonicalCore, MentionValue, PaperLogicTrace, PaperMetadata, ResearchMove, SlotProvenance
from app.research_logic import (
    build_prior_candidate_registry_from_package,
    compile_route_state_package,
    load_route_state_package_bundle,
    load_route_state_package_manifest,
    validate_route_state_package,
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


def _method_move(prefix: str) -> ResearchMove:
    return ResearchMove(
        move_id=f'{prefix}-m1',
        sequence_no=1,
        role='method',
        act_type='propose_method',
        summary='Uses graph neural network modeling for retrieval.',
        research_objects=[_mention('entity relation graph', 'entity relation graph', f'{prefix}-a1')],
        methods=[_mention('graph neural network', 'graph neural network', f'{prefix}-a2')],
        metrics=[_mention('MRR', 'mean reciprocal rank', f'{prefix}-a3')],
        conditions=[_mention('under high compression', 'under high compression', f'{prefix}-a4')],
        comparators=[_mention('baseline retriever', 'baseline retriever', f'{prefix}-a5')],
        limitation_types=[_mention('computational cost', 'computational cost', f'{prefix}-a6')],
        resource_mentions=[_mention('WN18RR', 'wn18rr', f'{prefix}-a7', mention_type='benchmark')],
        anchor_ids=[f'{prefix}-a1', f'{prefix}-a2', f'{prefix}-a3', f'{prefix}-a4', f'{prefix}-a5', f'{prefix}-a6', f'{prefix}-a7'],
        slot_provenance=[
            _prov('research_objects', f'{prefix}-a1', 0),
            _prov('methods', f'{prefix}-a2', 0),
            _prov('metrics', f'{prefix}-a3', 0),
            _prov('conditions', f'{prefix}-a4', 0),
            _prov('comparators', f'{prefix}-a5', 0),
            _prov('limitation_types', f'{prefix}-a6', 0),
            _prov('resource_mentions', f'{prefix}-a7', 0),
        ],
        confidence=0.9,
    )


def _result_move(prefix: str) -> ResearchMove:
    return ResearchMove(
        move_id=f'{prefix}-m2',
        sequence_no=2,
        role='result',
        act_type='report_effect',
        summary='The graph neural network improves MRR compared to the baseline retriever under high compression.',
        methods=[_mention('graph neural network', 'graph neural network', f'{prefix}-a10')],
        metrics=[_mention('MRR', 'mean reciprocal rank', f'{prefix}-a11')],
        comparators=[_mention('baseline retriever', 'baseline retriever', f'{prefix}-a12')],
        conditions=[_mention('under high compression', 'under high compression', f'{prefix}-a13')],
        limitation_types=[_mention('computational cost', 'computational cost', f'{prefix}-a14')],
        anchor_ids=[f'{prefix}-a10', f'{prefix}-a11', f'{prefix}-a12', f'{prefix}-a13', f'{prefix}-a14'],
        slot_provenance=[
            _prov('methods', f'{prefix}-a10', 0),
            _prov('metrics', f'{prefix}-a11', 0),
            _prov('comparators', f'{prefix}-a12', 0),
            _prov('conditions', f'{prefix}-a13', 0),
            _prov('limitation_types', f'{prefix}-a14', 0),
        ],
        confidence=0.92,
    )


def _bridge_move(prefix: str) -> ResearchMove:
    return ResearchMove(
        move_id=f'{prefix}-m3',
        sequence_no=3,
        role='experiment',
        act_type='set_condition',
        summary='Uses X-ray microtomography and YADE software to measure packing density against the dry baseline under high compression.',
        methods=[_mention('discrete element simulation', 'discrete element simulation', f'{prefix}-a20')],
        metrics=[_mention('packing density', 'packing density', f'{prefix}-a21')],
        comparators=[_mention('dry baseline', 'dry baseline', f'{prefix}-a22')],
        conditions=[_mention('under high compression', 'under high compression', f'{prefix}-a23')],
        resource_mentions=[
            _mention('X-ray microtomography', 'x-ray microtomography', f'{prefix}-a24', mention_type='instrument'),
            _mention('YADE', 'yade', f'{prefix}-a25', mention_type='software'),
        ],
        anchor_ids=[f'{prefix}-a20', f'{prefix}-a21', f'{prefix}-a22', f'{prefix}-a23', f'{prefix}-a24', f'{prefix}-a25'],
        slot_provenance=[
            _prov('methods', f'{prefix}-a20', 0),
            _prov('metrics', f'{prefix}-a21', 0),
            _prov('comparators', f'{prefix}-a22', 0),
            _prov('conditions', f'{prefix}-a23', 0),
            _prov('resource_mentions', f'{prefix}-a24', 0),
            _prov('resource_mentions', f'{prefix}-a25', 1),
        ],
        confidence=0.88,
    )


def _future_work_move(prefix: str) -> ResearchMove:
    return ResearchMove(
        move_id=f'{prefix}-m4',
        sequence_no=4,
        role='future_work',
        act_type='suggest_extension',
        summary='Future work should extend graph neural network retrieval to temporal knowledge graphs under noisy labels using web-scale corpora.',
        research_objects=[_mention('temporal knowledge graphs', 'temporal knowledge graph', f'{prefix}-a30')],
        methods=[_mention('graph neural network retrieval', 'graph neural network retrieval', f'{prefix}-a31')],
        conditions=[_mention('under noisy labels', 'under noisy labels', f'{prefix}-a32')],
        limitation_types=[_mention('limited labeled data', 'limited labeled data', f'{prefix}-a33')],
        resource_mentions=[_mention('web-scale corpora', 'web-scale corpora', f'{prefix}-a34', mention_type='dataset')],
        anchor_ids=[f'{prefix}-a30', f'{prefix}-a31', f'{prefix}-a32', f'{prefix}-a33', f'{prefix}-a34'],
        slot_provenance=[
            _prov('research_objects', f'{prefix}-a30', 0),
            _prov('methods', f'{prefix}-a31', 0),
            _prov('conditions', f'{prefix}-a32', 0),
            _prov('limitation_types', f'{prefix}-a33', 0),
            _prov('resource_mentions', f'{prefix}-a34', 0),
        ],
        confidence=0.81,
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
        canonical_core=CanonicalCore(moves=[_method_move(prefix), _result_move(prefix), _bridge_move(prefix), _future_work_move(prefix)]),
        quality={},
    )
    trace.derived_views = build_derived_views(trace)
    return trace


def _packet(
    packet_id: str,
    traces: list[PaperLogicTrace],
    *,
    cutoff_year: int,
    preferred_scope_label: str,
) -> dict[str, object]:
    return {
        'packet_id': packet_id,
        'built_at': '2026-04-02T01:05:00Z',
        'topic_scope_candidate': preferred_scope_label,
        'cutoff_year': cutoff_year,
        'packet_status': 'frozen',
        'packet_composition': {
            'target_size': len(traces),
            'actual_size': len(traces),
            'role_counts': {
                'core_method': max(1, len(traces)),
                'resource_or_benchmark': 1,
                'limitation_or_critique': 1,
                'survey_or_review': 1,
                'alternative_route': 1,
            },
            'coverage_ok': True,
            'missing_roles': [],
        },
        'l1_snapshot_ref': {
            'snapshot_id': f'{packet_id}:snapshot',
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
                'trace_id': trace.trace_id,
                'paper_year': trace.paper_metadata.year,
                'item_role': 'core_method' if index == 0 else 'resource_or_benchmark',
                'inclusion_reason': 'Supports packetized route reconstruction.',
                'source_selector': 'manual',
                'evidence_for_inclusion': [trace.paper_metadata.paper_id],
                'title': trace.paper_metadata.title,
            }
            for index, trace in enumerate(traces)
        ],
        'excluded_items': [],
        'packet_quality': {
            'quality_tier': 'green',
            'ready_for_route_state': True,
            'quality_flags': [],
            'topic_boundary_confidence': 0.82,
            'leakage_risk': 'low',
            'manual_review_status': 'completed',
        },
        'compiler_hints': {
            'preferred_scope_label': preferred_scope_label,
            'preferred_method_labels': ['graph neural network'],
            'preferred_benchmark_labels': ['wn18rr'],
            'preferred_bottleneck_labels': ['computational cost'],
            'expected_alternative_routes': ['temporal knowledge graph route'],
            'notes_for_route_state_compiler': 'Prefer packet-level aggregation over single-paper restatement.',
        },
    }


def _l1_snapshot(snapshot_id: str, *, cutoff_year: int) -> dict[str, object]:
    return {
        'snapshot_id': snapshot_id,
        'built_at': '2026-04-02T09:00:00Z',
        'topic_scope': 'large-scale image recognition with deep neural networks',
        'cutoff_year': cutoff_year,
        'grounding_mode': 'paper_grounded_l1_lite',
        'source_packet_id': snapshot_id.replace(':snapshot', ''),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


def _build_fixture_paths(tmp_path: Path) -> dict[str, Path]:
    cutoff_year = 2011
    traces = {
        'paper-a': _trace('paper-a', cutoff_year, 'pa'),
        'paper-b': _trace('paper-b', cutoff_year, 'pb'),
        'paper-c': _trace('paper-c', cutoff_year, 'pc'),
        'paper-d': _trace('paper-d', cutoff_year, 'pd'),
        'paper-e': _trace('paper-e', cutoff_year, 'pe'),
        'paper-f': _trace('paper-f', cutoff_year, 'pf'),
        'paper-g': _trace('paper-g', cutoff_year, 'pg'),
    }

    trace_dir = tmp_path / 'traces'
    trace_dir.mkdir()
    trace_paths: dict[str, Path] = {}
    for paper_id, trace in traces.items():
        trace_paths[paper_id] = _write_json(trace_dir / f'{paper_id}.json', trace.model_dump(mode='json'))

    packet_dir = tmp_path / 'packets'
    packet_dir.mkdir()
    packet_paths = {
        'primary': _write_json(
            packet_dir / 'primary.json',
            _packet(
                'packet-primary',
                [traces['paper-a'], traces['paper-b']],
                cutoff_year=cutoff_year,
                preferred_scope_label='large-scale image recognition with deep neural networks',
            ),
        ),
        'support-1': _write_json(
            packet_dir / 'support-1.json',
            _packet(
                'packet-support-1',
                [traces['paper-a'], traces['paper-c']],
                cutoff_year=cutoff_year,
                preferred_scope_label='large-scale image recognition with deep neural networks',
            ),
        ),
        'support-2': _write_json(
            packet_dir / 'support-2.json',
            _packet(
                'packet-support-2',
                [traces['paper-b'], traces['paper-d']],
                cutoff_year=cutoff_year,
                preferred_scope_label='large-scale image recognition with deep neural networks',
            ),
        ),
        'alternative-1': _write_json(
            packet_dir / 'alternative-1.json',
            _packet(
                'packet-alternative-1',
                [traces['paper-e'], traces['paper-f']],
                cutoff_year=cutoff_year,
                preferred_scope_label='temporal knowledge graph retrieval with graph neural networks',
            ),
        ),
        'held-out-1': _write_json(
            packet_dir / 'held-out-1.json',
            _packet(
                'packet-held-out-1',
                [traces['paper-f'], traces['paper-g']],
                cutoff_year=cutoff_year,
                preferred_scope_label='evaluation-heavy retrieval route under noisy labels',
            ),
        ),
    }

    l1_dir = tmp_path / 'l1'
    l1_dir.mkdir()
    l1_paths = {
        key: _write_json(l1_dir / f'{key}.json', _l1_snapshot(f'{key}:snapshot', cutoff_year=cutoff_year))
        for key in packet_paths
    }

    return {
        **{f'trace:{key}': value for key, value in trace_paths.items()},
        **{f'packet:{key}': value for key, value in packet_paths.items()},
        **{f'l1:{key}': value for key, value in l1_paths.items()},
    }


def test_run_route_state_package_cli_writes_bundle(tmp_path: Path) -> None:
    fixture_paths = _build_fixture_paths(tmp_path)
    manifest_path = _write_json(
        tmp_path / 'route_state_package_manifest.json',
        {
            'package_id': 'demo-route-state-package',
            'built_at': '2026-04-02T10:00:00Z',
            'topic_scope': 'large-scale image recognition with deep neural networks',
            'cutoff_year': 2011,
            'entries': [
                {
                    'entry_id': 'primary-main',
                    'role': 'primary',
                    'packet_path': 'packets/primary.json',
                    'trace_files': ['traces/paper-a.json', 'traces/paper-b.json'],
                    'l1_snapshot_path': 'l1/primary.json',
                },
                {
                    'entry_id': 'support-01',
                    'role': 'support',
                    'packet_path': 'packets/support-1.json',
                    'trace_files': ['traces/paper-a.json', 'traces/paper-c.json'],
                    'l1_snapshot_path': 'l1/support-1.json',
                },
                {
                    'entry_id': 'support-02',
                    'role': 'support',
                    'packet_path': 'packets/support-2.json',
                    'trace_files': ['traces/paper-b.json', 'traces/paper-d.json'],
                    'l1_snapshot_path': 'l1/support-2.json',
                },
                {
                    'entry_id': 'alternative-01',
                    'role': 'alternative',
                    'packet_path': 'packets/alternative-1.json',
                    'trace_files': ['traces/paper-e.json', 'traces/paper-f.json'],
                    'l1_snapshot_path': 'l1/alternative-1.json',
                },
                {
                    'entry_id': 'held-out-01',
                    'role': 'held_out',
                    'packet_path': 'packets/held-out-1.json',
                    'trace_files': ['traces/paper-f.json', 'traces/paper-g.json'],
                    'l1_snapshot_path': 'l1/held-out-1.json',
                },
            ],
        },
    )
    output_dir = tmp_path / 'route_state_package_bundle'
    script_path = Path(__file__).resolve().parents[1] / 'scripts' / 'run_route_state_package.py'

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            '--manifest',
            str(manifest_path),
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
    bundle = load_route_state_package_bundle(output_dir)

    assert summary_payload['package_id'] == 'demo-route-state-package'
    assert summary_payload['entry_count'] == 5
    assert summary_payload['support_route_state_count'] == 2
    assert summary_payload['alternative_route_state_count'] == 1
    assert summary_payload['held_out_route_state_count'] == 1
    assert summary_payload['validation_quality_tier'] == 'green'
    assert summary_payload['validation_ready_for_replay'] is True
    assert summary_payload['validation_quality_flags'] == []
    assert bundle.manifest.package_id == 'demo-route-state-package'
    assert bundle.primary_route_state is not None
    assert len(bundle.support_route_states) == 2
    assert len(bundle.alternative_route_states) == 1
    assert len(bundle.held_out_route_states) == 1
    assert bundle.validation is not None
    assert bundle.validation.quality_tier == 'green'
    assert (output_dir / 'bundle_manifest.json').is_file()
    assert (output_dir / 'validation.json').is_file()
    assert (output_dir / 'grouped' / 'support_route_states.json').is_file()
    assert (output_dir / 'entries' / 'support-01.json').is_file()


def test_run_replay_pilot_accepts_route_state_package(tmp_path: Path) -> None:
    _build_fixture_paths(tmp_path)
    manifest_path = _write_json(
        tmp_path / 'route_state_package_manifest.json',
        {
            'package_id': 'demo-route-state-package',
            'built_at': '2026-04-02T10:00:00Z',
            'topic_scope': 'large-scale image recognition with deep neural networks',
            'cutoff_year': 2011,
            'entries': [
                {
                    'entry_id': 'support-01',
                    'role': 'support',
                    'packet_path': 'packets/support-1.json',
                    'trace_files': ['traces/paper-a.json', 'traces/paper-c.json'],
                    'l1_snapshot_path': 'l1/support-1.json',
                },
                {
                    'entry_id': 'support-02',
                    'role': 'support',
                    'packet_path': 'packets/support-2.json',
                    'trace_files': ['traces/paper-b.json', 'traces/paper-d.json'],
                    'l1_snapshot_path': 'l1/support-2.json',
                },
                {
                    'entry_id': 'alternative-01',
                    'role': 'alternative',
                    'packet_path': 'packets/alternative-1.json',
                    'trace_files': ['traces/paper-e.json', 'traces/paper-f.json'],
                    'l1_snapshot_path': 'l1/alternative-1.json',
                },
                {
                    'entry_id': 'held-out-01',
                    'role': 'held_out',
                    'packet_path': 'packets/held-out-1.json',
                    'trace_files': ['traces/paper-f.json', 'traces/paper-g.json'],
                    'l1_snapshot_path': 'l1/held-out-1.json',
                },
            ],
        },
    )
    package_output_dir = tmp_path / 'route_state_package_bundle'
    package_script = Path(__file__).resolve().parents[1] / 'scripts' / 'run_route_state_package.py'
    package_result = subprocess.run(
        [
            sys.executable,
            str(package_script),
            '--manifest',
            str(manifest_path),
            '--output-dir',
            str(package_output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert package_result.returncode == 0, package_result.stderr

    replay_script = Path(__file__).resolve().parents[1] / 'scripts' / 'run_replay_pilot.py'
    replay_output_dir = tmp_path / 'replay_bundle'
    replay_result = subprocess.run(
        [
            sys.executable,
            str(replay_script),
            '--packet',
            str(tmp_path / 'packets' / 'primary.json'),
            '--trace-file',
            str(tmp_path / 'traces' / 'paper-a.json'),
            '--trace-file',
            str(tmp_path / 'traces' / 'paper-b.json'),
            '--l1-snapshot',
            str(tmp_path / 'l1' / 'primary.json'),
            '--route-state-package',
            str(package_output_dir),
            '--reviewer',
            'reviewer-1',
            '--output-dir',
            str(replay_output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    assert replay_result.returncode == 0, replay_result.stderr
    summary_payload = json.loads(replay_result.stdout)

    assert summary_payload['support_route_state_count'] == 2
    assert summary_payload['alternative_route_state_count'] == 1
    assert summary_payload['held_out_route_state_count'] == 1
    assert summary_payload['route_state_package_id'] == 'demo-route-state-package'
    assert summary_payload['route_state_package_validation_quality_tier'] == 'green'
    assert summary_payload['route_state_package_validation_flags'] == []
    assert 'support_cluster_too_small' not in summary_payload['quality_flags']
    assert 'no_alternative_route_states' not in summary_payload['quality_flags']
    assert 'held_out_routes_missing' not in summary_payload['quality_flags']


def test_validate_route_state_package_flags_missing_roles(tmp_path: Path) -> None:
    _build_fixture_paths(tmp_path)
    manifest_path = _write_json(
        tmp_path / 'route_state_package_manifest_sparse.json',
        {
            'package_id': 'sparse-route-state-package',
            'built_at': '2026-04-02T10:00:00Z',
            'topic_scope': 'large-scale image recognition with deep neural networks',
            'cutoff_year': 2011,
            'entries': [
                {
                    'entry_id': 'support-01',
                    'role': 'support',
                    'packet_path': 'packets/support-1.json',
                    'trace_files': ['traces/paper-a.json', 'traces/paper-c.json'],
                    'l1_snapshot_path': 'l1/support-1.json',
                }
            ],
        },
    )

    manifest = load_route_state_package_manifest(manifest_path)
    compilation = compile_route_state_package(manifest, manifest_base_dir=manifest_path.parent)
    validation = validate_route_state_package(compilation)

    assert validation.quality_tier == 'red'
    assert validation.ready_for_replay is False
    assert 'support_cluster_too_small' in validation.quality_flags
    assert 'alternative_route_states_missing' in validation.quality_flags
    assert 'held_out_route_states_missing' in validation.quality_flags


def test_validate_route_state_package_preserves_distinctness_rationale_for_overlapping_alternative_scope(
    tmp_path: Path,
) -> None:
    _build_fixture_paths(tmp_path)
    manifest_path = _write_json(
        tmp_path / 'route_state_package_manifest_distinct_alternative.json',
        {
            'package_id': 'distinct-route-state-package',
            'built_at': '2026-04-02T10:00:00Z',
            'topic_scope': 'large-scale image recognition with deep neural networks',
            'cutoff_year': 2011,
            'entries': [
                {
                    'entry_id': 'support-01',
                    'role': 'support',
                    'packet_path': 'packets/support-1.json',
                    'trace_files': ['traces/paper-a.json', 'traces/paper-c.json'],
                    'l1_snapshot_path': 'l1/support-1.json',
                },
                {
                    'entry_id': 'support-02',
                    'role': 'support',
                    'packet_path': 'packets/support-2.json',
                    'trace_files': ['traces/paper-b.json', 'traces/paper-d.json'],
                    'l1_snapshot_path': 'l1/support-2.json',
                },
                {
                    'entry_id': 'alternative-01',
                    'role': 'alternative',
                    'packet_path': 'packets/support-1.json',
                    'trace_files': ['traces/paper-a.json', 'traces/paper-c.json'],
                    'l1_snapshot_path': 'l1/support-1.json',
                    'route_state_id': 'alternative-01-route-state',
                    'distinctness_rationale': 'Alternative route stays because the learned synthesis path differs from the packet mainline even when topical overlap is high.',
                },
                {
                    'entry_id': 'held-out-01',
                    'role': 'held_out',
                    'packet_path': 'packets/held-out-1.json',
                    'trace_files': ['traces/paper-f.json', 'traces/paper-g.json'],
                    'l1_snapshot_path': 'l1/held-out-1.json',
                },
            ],
        },
    )

    manifest = load_route_state_package_manifest(manifest_path)
    compilation = compile_route_state_package(manifest, manifest_base_dir=manifest_path.parent)
    validation = validate_route_state_package(compilation)
    alternative_artifact = next(artifact for artifact in compilation.entries if artifact.role == 'alternative')

    assert alternative_artifact.distinctness_rationale == (
        'Alternative route stays because the learned synthesis path differs from the packet mainline even when topical overlap is high.'
    )
    assert 'alternative_scope_not_distinct' not in validation.quality_flags
    assert validation.indistinct_alternative_entry_ids == []


def test_route_state_package_bundle_feeds_prior_induction_without_reserialization(tmp_path: Path) -> None:
    _build_fixture_paths(tmp_path)
    manifest_path = _write_json(
        tmp_path / 'route_state_package_manifest.json',
        {
            'package_id': 'demo-route-state-package',
            'built_at': '2026-04-02T10:00:00Z',
            'topic_scope': 'large-scale image recognition with deep neural networks',
            'cutoff_year': 2011,
            'entries': [
                {
                    'entry_id': 'support-01',
                    'role': 'support',
                    'packet_path': 'packets/support-1.json',
                    'trace_files': ['traces/paper-a.json', 'traces/paper-c.json'],
                    'l1_snapshot_path': 'l1/support-1.json',
                },
                {
                    'entry_id': 'support-02',
                    'role': 'support',
                    'packet_path': 'packets/support-2.json',
                    'trace_files': ['traces/paper-b.json', 'traces/paper-d.json'],
                    'l1_snapshot_path': 'l1/support-2.json',
                },
                {
                    'entry_id': 'alternative-01',
                    'role': 'alternative',
                    'packet_path': 'packets/alternative-1.json',
                    'trace_files': ['traces/paper-e.json', 'traces/paper-f.json'],
                    'l1_snapshot_path': 'l1/alternative-1.json',
                },
                {
                    'entry_id': 'held-out-01',
                    'role': 'held_out',
                    'packet_path': 'packets/held-out-1.json',
                    'trace_files': ['traces/paper-f.json', 'traces/paper-g.json'],
                    'l1_snapshot_path': 'l1/held-out-1.json',
                },
            ],
        },
    )
    manifest = load_route_state_package_manifest(manifest_path)
    compilation = compile_route_state_package(manifest, manifest_base_dir=manifest_path.parent)

    registry = build_prior_candidate_registry_from_package(
        compilation,
        reviewer_ids=['reviewer-1'],
        built_at='2026-04-02T10:30:00Z',
    )

    assert registry.package_id == 'demo-route-state-package'
    assert registry.prior_candidates
    assert registry.clusters[0].support_count == 2
    assert registry.prior_candidates[0].supporting_route_state_ids == [
        artifact.route_state.route_state_id for artifact in compilation.entries if artifact.role == 'support'
    ]


def test_run_replay_pilot_can_emit_prior_review_bundle_from_route_state_package(tmp_path: Path) -> None:
    _build_fixture_paths(tmp_path)
    manifest_path = _write_json(
        tmp_path / 'route_state_package_manifest.json',
        {
            'package_id': 'demo-route-state-package',
            'built_at': '2026-04-02T10:00:00Z',
            'topic_scope': 'large-scale image recognition with deep neural networks',
            'cutoff_year': 2011,
            'entries': [
                {
                    'entry_id': 'support-01',
                    'role': 'support',
                    'packet_path': 'packets/support-1.json',
                    'trace_files': ['traces/paper-a.json', 'traces/paper-c.json'],
                    'l1_snapshot_path': 'l1/support-1.json',
                },
                {
                    'entry_id': 'support-02',
                    'role': 'support',
                    'packet_path': 'packets/support-2.json',
                    'trace_files': ['traces/paper-b.json', 'traces/paper-d.json'],
                    'l1_snapshot_path': 'l1/support-2.json',
                },
                {
                    'entry_id': 'alternative-01',
                    'role': 'alternative',
                    'packet_path': 'packets/alternative-1.json',
                    'trace_files': ['traces/paper-e.json', 'traces/paper-f.json'],
                    'l1_snapshot_path': 'l1/alternative-1.json',
                },
                {
                    'entry_id': 'held-out-01',
                    'role': 'held_out',
                    'packet_path': 'packets/held-out-1.json',
                    'trace_files': ['traces/paper-f.json', 'traces/paper-g.json'],
                    'l1_snapshot_path': 'l1/held-out-1.json',
                },
            ],
        },
    )
    package_output_dir = tmp_path / 'route_state_package_bundle'
    package_script = Path(__file__).resolve().parents[1] / 'scripts' / 'run_route_state_package.py'
    package_result = subprocess.run(
        [
            sys.executable,
            str(package_script),
            '--manifest',
            str(manifest_path),
            '--output-dir',
            str(package_output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert package_result.returncode == 0, package_result.stderr

    replay_script = Path(__file__).resolve().parents[1] / 'scripts' / 'run_replay_pilot.py'
    replay_output_dir = tmp_path / 'replay_bundle'
    prior_review_output_dir = tmp_path / 'prior_review_bundle'
    replay_result = subprocess.run(
        [
            sys.executable,
            str(replay_script),
            '--packet',
            str(tmp_path / 'packets' / 'primary.json'),
            '--trace-file',
            str(tmp_path / 'traces' / 'paper-a.json'),
            '--trace-file',
            str(tmp_path / 'traces' / 'paper-b.json'),
            '--l1-snapshot',
            str(tmp_path / 'l1' / 'primary.json'),
            '--route-state-package',
            str(package_output_dir),
            '--reviewer',
            'reviewer-1',
            '--output-dir',
            str(replay_output_dir),
            '--prior-review-output-dir',
            str(prior_review_output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    assert replay_result.returncode == 0, replay_result.stderr
    summary_payload = json.loads(replay_result.stdout)

    assert (replay_output_dir / 'bundle_manifest.json').is_file()
    assert (prior_review_output_dir / 'bundle_manifest.json').is_file()
    assert (prior_review_output_dir / 'prior_candidates.json').is_file()
    assert (prior_review_output_dir / 'anti_pattern_candidates.json').is_file()
    assert summary_payload['prior_review_output_dir'] == str(prior_review_output_dir.resolve())
    assert 'prior_review_bundle_manifest' in summary_payload
