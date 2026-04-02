from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from app.paper_logic_trace.derived_views import build_derived_views
from app.paper_logic_trace.models import CanonicalCore, MentionValue, PaperLogicTrace, PaperMetadata, ResearchMove, SlotProvenance


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


def _packet(traces: list[PaperLogicTrace], *, cutoff_year: int) -> dict[str, object]:
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
                'trace_id': trace.trace_id,
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
        'packet_quality': {
            'quality_tier': 'green',
            'ready_for_route_state': True,
            'quality_flags': [],
            'topic_boundary_confidence': 0.82,
            'leakage_risk': 'low',
            'manual_review_status': 'completed',
        },
        'compiler_hints': {
            'preferred_scope_label': 'large-scale image recognition with deep neural networks',
            'preferred_method_labels': ['graph neural network'],
            'preferred_benchmark_labels': ['wn18rr'],
            'preferred_bottleneck_labels': [],
            'expected_alternative_routes': [],
            'notes_for_route_state_compiler': 'Prefer packet-level aggregation over single-paper restatement.',
        },
    }


def _l1_snapshot(*, cutoff_year: int) -> dict[str, object]:
    return {
        'snapshot_id': 'imagenet_2011_snapshot',
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
        'toolchain_timeline': [],
        'protocol_registry': [],
        'unresolved_questions': [],
        'quality': {
            'quality_tier': 'yellow',
            'quality_flags': ['paper_only_snapshot'],
            'audit_status': 'eligible',
            'paper_only_snapshot': True,
            'completeness_score': 0.5,
        },
    }


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


def test_run_route_state_pilot_cli_outputs_summary(tmp_path: Path) -> None:
    traces = [_trace('paper-a', 2011, 'pa')]
    trace_path = _write_json(tmp_path / 'paper-a.json', traces[0].model_dump(mode='json'))
    packet_path = _write_json(tmp_path / 'route_packet.json', _packet(traces, cutoff_year=2011))
    l1_snapshot_path = _write_json(tmp_path / 'historical_environment_snapshot.json', _l1_snapshot(cutoff_year=2011))
    output_path = tmp_path / 'route_state.json'
    script_path = Path(__file__).resolve().parents[1] / 'scripts' / 'run_route_state_pilot.py'

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
            '--output-path',
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    assert result.returncode == 0, result.stderr
    summary_payload = json.loads(result.stdout)
    route_state_payload = json.loads(output_path.read_text(encoding='utf-8'))

    assert summary_payload['l1_snapshot_id'] == 'imagenet_2011_snapshot'
    assert summary_payload['output_path'] == str(output_path.resolve())
    assert route_state_payload['compiler_metadata']['l1_snapshot_version'] == 'imagenet_2011_snapshot'
