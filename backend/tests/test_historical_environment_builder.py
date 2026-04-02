from __future__ import annotations

import pytest

from app.paper_logic_trace.derived_views import build_derived_views
from app.paper_logic_trace.models import CanonicalCore, MentionValue, PaperLogicTrace, PaperMetadata, ResearchMove, SlotProvenance
from app.research_logic.historical_environment import (
    HistoricalEnvironmentBuilder,
    build_historical_environment_snapshot,
    build_l1_snapshot_ref,
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
        summary='Uses graph neural network modeling for retrieval with WN18RR.',
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
        confidence=0.91,
    )


def _bridge_move(prefix: str) -> ResearchMove:
    return ResearchMove(
        move_id=f'{prefix}-m2',
        sequence_no=2,
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
        move_id=f'{prefix}-m3',
        sequence_no=3,
        role='future_work',
        act_type='suggest_extension',
        summary='Future work should extend graph neural network retrieval to temporal knowledge graphs using web-scale corpora.',
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
        confidence=0.82,
    )


def _jamming_benchmark_move(prefix: str) -> ResearchMove:
    return ResearchMove(
        move_id=f'{prefix}-m4',
        sequence_no=4,
        role='result',
        act_type='report_effect',
        summary='The maximally random jammed state replaces random close packing, and the onset is tracked at packing fraction phi_c.',
        metrics=[_mention('packing fraction phi_c', 'packing fraction phi_c', f'{prefix}-a40')],
        comparators=[_mention('random close packing', 'random close packing', f'{prefix}-a41')],
        anchor_ids=[f'{prefix}-a40', f'{prefix}-a41'],
        slot_provenance=[
            _prov('metrics', f'{prefix}-a40', 0),
            _prov('comparators', f'{prefix}-a41', 0),
        ],
        confidence=0.84,
    )


def _trace(paper_id: str, year: int, prefix: str) -> PaperLogicTrace:
    trace = PaperLogicTrace(
        trace_id=f'{paper_id}:paper_logic_trace',
        schema_version='v2',
        built_at='2026-04-02T10:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id=paper_id,
            title=f'Demo {paper_id}',
            year=year,
            paper_type='empirical',
            source_refs=[f'{prefix}-src'],
        ),
        canonical_core=CanonicalCore(moves=[_method_move(prefix), _bridge_move(prefix), _future_work_move(prefix)]),
        quality={},
    )
    trace.derived_views = build_derived_views(trace)
    return trace


def _jamming_trace(paper_id: str, year: int, prefix: str) -> PaperLogicTrace:
    trace = PaperLogicTrace(
        trace_id=f'{paper_id}:paper_logic_trace',
        schema_version='v2',
        built_at='2026-04-02T10:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id=paper_id,
            title=f'Jamming {paper_id}',
            year=year,
            paper_type='empirical',
            source_refs=[f'{prefix}-src'],
        ),
        canonical_core=CanonicalCore(moves=[_jamming_benchmark_move(prefix)]),
        quality={},
    )
    trace.derived_views = build_derived_views(trace)
    return trace


def _packet(traces: list[PaperLogicTrace], *, cutoff_year: int, preferred_benchmarks: list[str] | None = None) -> dict:
    return {
        'packet_id': f'packet-{cutoff_year}',
        'built_at': '2026-04-02T10:05:00Z',
        'topic_scope_candidate': 'large-scale image retrieval with graph neural networks',
        'cutoff_year': cutoff_year,
        'packet_status': 'frozen',
        'packet_composition': {
            'target_size': len(traces),
            'actual_size': len(traces),
            'role_counts': {
                'core_method': len(traces),
                'resource_or_benchmark': 1,
                'limitation_or_critique': 1,
                'survey_or_review': 0,
                'alternative_route': 1,
            },
            'coverage_ok': True,
            'missing_roles': [],
        },
        'l1_snapshot_ref': {
            'snapshot_id': 'demo_snapshot',
            'resource_registry_ref': 'l1:demo:resource-registry',
            'resource_timeline_ref': 'l1:demo:resource-timeline',
            'benchmark_timeline_ref': 'l1:demo:benchmark-timeline',
            'toolchain_timeline_ref': 'l1:demo:toolchain-timeline',
            'protocol_registry_ref': 'l1:demo:protocol-registry',
        },
        'inclusion_rules': {
            'scope_definition': 'Build a bounded L1 snapshot from packetized traces.',
            'scope_aliases': ['graph retrieval'],
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
                'inclusion_reason': 'Supports bounded L1 reconstruction.',
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
            'topic_boundary_confidence': 0.83,
            'leakage_risk': 'low',
            'manual_review_status': 'completed',
        },
        'compiler_hints': {
            'preferred_scope_label': 'large-scale image retrieval with graph neural networks',
            'preferred_method_labels': ['graph neural network'],
            'preferred_benchmark_labels': preferred_benchmarks or ['wn18rr'],
            'preferred_bottleneck_labels': ['computational cost'],
            'expected_alternative_routes': ['graph neural network retrieval -> temporal knowledge graph'],
        },
    }


def test_historical_environment_builder_compiles_paper_grounded_l1_snapshot() -> None:
    traces = [_trace('paper-a', 2011, 'pa'), _trace('paper-b', 2011, 'pb')]

    snapshot = build_historical_environment_snapshot(
        _packet(traces, cutoff_year=2011),
        traces,
        built_at='2026-04-02T10:10:00Z',
    )

    assert snapshot.grounding_mode == 'paper_grounded_l1_lite'
    assert snapshot.quality.quality_tier == 'yellow'
    assert 'paper_only_snapshot' in snapshot.quality.quality_flags
    assert snapshot.source_paper_ids == ['paper-a', 'paper-b']
    assert any(entry.label == 'wn18rr' and entry.status == 'available' for entry in snapshot.benchmark_timeline)
    assert any(entry.label == 'x-ray microtomography' and entry.status == 'available' for entry in snapshot.toolchain_timeline)
    assert any(entry.label == 'yade' and entry.status == 'available' for entry in snapshot.toolchain_timeline)
    assert any(entry.label == 'packing density' and entry.status == 'available' for entry in snapshot.protocol_registry)

    snapshot_ref = build_l1_snapshot_ref(snapshot)
    assert snapshot_ref.snapshot_id == snapshot.snapshot_id
    assert snapshot_ref.benchmark_timeline_ref.endswith(':benchmark-timeline')


def test_historical_environment_builder_marks_missing_preferred_benchmark() -> None:
    traces = [_trace('paper-a', 2011, 'pa'), _trace('paper-b', 2011, 'pb')]

    snapshot = build_historical_environment_snapshot(
        _packet(traces, cutoff_year=2011, preferred_benchmarks=['imagenet']),
        traces,
        built_at='2026-04-02T10:10:00Z',
    )

    assert 'preferred_benchmark_missing' in snapshot.quality.quality_flags
    assert any(entry.label == 'imagenet' and entry.status == 'missing' for entry in snapshot.benchmark_timeline)


def test_historical_environment_builder_recovers_preferred_benchmark_from_alias_fallback() -> None:
    trace = _jamming_trace('paper-j', 2010, 'pj')

    snapshot = build_historical_environment_snapshot(
        _packet(
            [trace],
            cutoff_year=2010,
            preferred_benchmarks=['maximally random jammed state', 'packing fraction phi_c'],
        ),
        [trace],
        built_at='2026-04-02T10:10:00Z',
    )

    benchmark_status = {entry.label: entry.status for entry in snapshot.benchmark_timeline}

    assert 'preferred_benchmark_missing' not in snapshot.quality.quality_flags
    assert benchmark_status['maximally random jammed state'] == 'emerging'
    assert benchmark_status['packing fraction phi c'] == 'emerging'


def test_historical_environment_builder_rejects_trace_after_cutoff() -> None:
    traces = [_trace('paper-a', 2011, 'pa'), _trace('paper-b', 2012, 'pb')]
    builder = HistoricalEnvironmentBuilder()

    with pytest.raises(ValueError, match='exceeds L1 snapshot cutoff_year'):
        builder.build(_packet(traces, cutoff_year=2011), traces)
