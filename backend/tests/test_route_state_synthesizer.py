from __future__ import annotations

import pytest

from app.paper_logic_trace.derived_views import build_derived_views
from app.paper_logic_trace.models import CanonicalCore, MentionValue, PaperLogicTrace, PaperMetadata, ResearchMove, SlotProvenance
from app.research_logic.route_state_synthesizer import RouteStateSynthesizer, synthesize_route_state


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


def _trace(paper_id: str, year: int, prefix: str, *, thin: bool = False) -> PaperLogicTrace:
    moves = [_thin_method_move(prefix)] if thin else [_method_move(prefix), _result_move(prefix), _bridge_move(prefix), _future_work_move(prefix)]
    trace = PaperLogicTrace(
        trace_id=f'{paper_id}:paper_logic_trace',
        schema_version='v2',
        built_at='2026-04-01T21:00:00Z',
        paper_metadata=PaperMetadata(
            paper_id=paper_id,
            title=f'Demo {paper_id}',
            year=year,
            paper_type='empirical',
            source_refs=[f'{prefix}-src'],
        ),
        canonical_core=CanonicalCore(moves=moves),
        quality={},
    )
    trace.derived_views = build_derived_views(trace)
    return trace


def _rename_methods(trace: PaperLogicTrace, prefix: str) -> PaperLogicTrace:
    for move_index, move in enumerate(trace.canonical_core.moves, start=1):
        renamed_methods: list[MentionValue] = []
        for method_index, method in enumerate(move.methods, start=1):
            renamed_methods.append(
                MentionValue(
                    surface=f'{prefix} method {move_index}-{method_index}',
                    normalized=f'{prefix} method {move_index}-{method_index}',
                    type=method.type,
                    anchor_ids=list(method.anchor_ids),
                    confidence=method.confidence,
                )
            )
        move.methods = renamed_methods
    trace.derived_views = build_derived_views(trace)
    return trace


def _packet(traces: list[PaperLogicTrace], *, cutoff_year: int) -> dict:
    return {
        'packet_id': f'packet-{cutoff_year}',
        'built_at': '2026-04-01T21:05:00Z',
        'topic_scope_candidate': 'large-scale image recognition with deep neural networks',
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
                'item_role': 'core_method' if index == 0 else 'resource_or_benchmark',
                'inclusion_reason': 'Supports packetized route reconstruction.',
                'source_selector': 'manual',
                'evidence_for_inclusion': [trace.paper_metadata.paper_id],
                'title': trace.paper_metadata.title,
            }
            for index, trace in enumerate(traces)
        ],
        'excluded_items': [
            {
                'paper_id': 'future-paper',
                'paper_year': cutoff_year + 1,
                'exclusion_reason': 'after_cutoff',
                'notes': 'Audit only',
            }
        ],
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
            'preferred_bottleneck_labels': ['computational cost'],
            'expected_alternative_routes': ['graph neural network retrieval -> temporal knowledge graph'],
            'notes_for_route_state_compiler': 'Prefer packet-level aggregation over single-paper restatement.',
        },
    }


def _l1_snapshot(*, cutoff_year: int, snapshot_id: str = 'imagenet_2011_snapshot') -> dict:
    return {
        'snapshot_id': snapshot_id,
        'built_at': '2026-04-02T09:00:00Z',
        'topic_scope': 'large-scale image recognition with deep neural networks',
        'cutoff_year': cutoff_year,
        'grounding_mode': 'paper_grounded_l1_lite',
        'source_packet_id': f'packet-{cutoff_year}',
        'source_trace_ids': ['paper-thin:paper_logic_trace'],
        'source_paper_ids': ['paper-thin'],
        'resource_registry': [
            {
                'label': 'imagenet-1k',
                'status': 'available',
                'confidence': 0.82,
                'source_paper_ids': ['paper-thin'],
                'source_trace_ids': ['paper-thin:paper_logic_trace'],
                'source_move_ids': ['pt-m1'],
                'evidence_ids': ['l1-r1'],
                'resource_types': ['dataset'],
            }
        ],
        'benchmark_timeline': [
            {
                'label': 'wn18rr',
                'status': 'available',
                'confidence': 0.8,
                'source_paper_ids': ['paper-thin'],
                'source_trace_ids': ['paper-thin:paper_logic_trace'],
                'source_move_ids': ['pt-m1'],
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
                'source_paper_ids': ['paper-thin'],
                'source_trace_ids': ['paper-thin:paper_logic_trace'],
                'source_move_ids': ['pt-m1'],
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
                'source_paper_ids': ['paper-thin'],
                'source_trace_ids': ['paper-thin:paper_logic_trace'],
                'source_move_ids': ['pt-m1'],
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


def test_route_state_synthesizer_compiles_green_route_state_from_packetized_traces() -> None:
    traces = [_trace('paper-a', 2011, 'pa'), _trace('paper-b', 2011, 'pb')]

    route_state = synthesize_route_state(_packet(traces, cutoff_year=2011), traces, built_at='2026-04-01T21:10:00Z')

    assert route_state.topic_scope == 'large-scale image recognition with deep neural networks'
    assert route_state.route_family_id
    assert route_state.quality.quality_tier == 'green'
    assert route_state.source_packet.included_trace_ids == [trace.trace_id for trace in traces]
    assert route_state.route_landscape.dominant_methods[0].label == 'graph neural network'
    assert len(route_state.route_landscape.dominant_methods[0].source_paper_ids) == 2
    assert route_state.route_landscape.active_benchmarks[0].label == 'wn18rr'
    assert route_state.route_landscape.active_benchmarks[0].raw_source_phrases == ['WN18RR']
    assert route_state.route_landscape.known_bottlenecks
    assert route_state.route_landscape.alternative_routes
    assert route_state.why_now_features.unlocking_factors
    assert route_state.why_now_features.unlocking_factors[0].label == 'wn18rr'
    assert route_state.why_now_features.unlocking_factors[0].raw_source_phrases == ['WN18RR']
    assert route_state.evidence_bundle.supporting_evidence_ids
    assert route_state.evidence_bundle.challenging_evidence_ids
    assert route_state.compiler_metadata.trace_versions == {trace.trace_id: 'v2' for trace in traces}


def test_route_state_synthesizer_derives_route_family_id_independent_of_packet_id() -> None:
    traces = [_trace('paper-a', 2011, 'pa'), _trace('paper-b', 2011, 'pb')]
    primary_packet = _packet(traces, cutoff_year=2011)
    runtime_subset_packet = _packet(traces, cutoff_year=2011)
    runtime_subset_packet['packet_id'] = 'packet-2011-runtime-subset'

    primary_route_state = synthesize_route_state(primary_packet, traces, built_at='2026-04-01T21:12:00Z')
    runtime_subset_route_state = synthesize_route_state(runtime_subset_packet, traces, built_at='2026-04-01T21:13:00Z')

    assert primary_route_state.route_state_id != runtime_subset_route_state.route_state_id
    assert primary_route_state.route_family_id == runtime_subset_route_state.route_family_id


def test_route_state_synthesizer_raises_when_packet_trace_is_missing() -> None:
    traces = [_trace('paper-a', 2011, 'pa'), _trace('paper-b', 2011, 'pb')]
    packet = _packet(traces, cutoff_year=2011)

    with pytest.raises(ValueError, match='missing traces'):
        RouteStateSynthesizer().synthesize(packet, traces[:1])


def test_route_state_synthesizer_raises_when_trace_exceeds_cutoff() -> None:
    late_trace = _trace('paper-late', 2012, 'pl')
    packet = _packet([late_trace], cutoff_year=2011)

    with pytest.raises(ValueError, match='exceeds packet cutoff_year'):
        RouteStateSynthesizer().synthesize(packet, [late_trace])


def test_route_state_synthesizer_marks_thin_single_paper_state_yellow() -> None:
    trace = _trace('paper-thin', 2011, 'pt', thin=True)

    route_state = synthesize_route_state(_packet([trace], cutoff_year=2011), [trace], built_at='2026-04-01T21:20:00Z')

    assert route_state.quality.quality_tier == 'yellow'
    assert route_state.quality.ready_for_prior_selection is False
    assert 'dominant_method_not_multi_paper' in route_state.quality.quality_flags
    assert 'missing_challenging_evidence' in route_state.quality.quality_flags


def test_route_state_synthesizer_allows_multi_paper_method_landscape_without_single_repeated_method() -> None:
    trace_a = _trace('paper-a', 2011, 'pa')
    trace_b = _rename_methods(_trace('paper-b', 2011, 'pb'), 'route-b')

    route_state = synthesize_route_state(
        _packet([trace_a, trace_b], cutoff_year=2011),
        [trace_a, trace_b],
        built_at='2026-04-02T23:10:00Z',
    )

    assert route_state.quality.quality_tier == 'green'
    assert route_state.quality.ready_for_route_comparison is True
    assert 'dominant_method_not_multi_paper' not in route_state.quality.quality_flags


def test_route_state_synthesizer_merges_l1_snapshot_into_route_landscape() -> None:
    trace = _trace('paper-thin', 2011, 'pt', thin=True)

    route_state = synthesize_route_state(
        _packet([trace], cutoff_year=2011),
        [trace],
        l1_snapshot=_l1_snapshot(cutoff_year=2011),
        built_at='2026-04-02T09:10:00Z',
    )

    assert [benchmark.label for benchmark in route_state.route_landscape.active_benchmarks] == ['wn18rr']
    assert [infra.label for infra in route_state.route_landscape.toolchains_and_infrastructure] == ['cuda stack']
    assert [protocol.label for protocol in route_state.route_landscape.measurement_protocols] == ['top-1 accuracy']
    assert any(feature.feature_type == 'protocol' for feature in route_state.why_now_features.acceleration_factors)
    assert 'l1-b1' in route_state.evidence_bundle.supporting_evidence_ids
    assert route_state.source_packet.l1_snapshot_ref == 'imagenet_2011_snapshot'
    assert route_state.compiler_metadata.l1_snapshot_version == 'imagenet_2011_snapshot'
    assert route_state.uncertainty_points.open_questions == [
        'External benchmark governance remains unmodeled.',
        'paper-grounded L1-lite may still miss external benchmark, toolchain, or protocol history',
    ]


def test_route_state_synthesizer_rejects_l1_snapshot_cutoff_mismatch() -> None:
    trace = _trace('paper-thin', 2011, 'pt', thin=True)

    with pytest.raises(ValueError, match='l1_snapshot cutoff_year must match RoutePacket cutoff_year'):
        synthesize_route_state(
            _packet([trace], cutoff_year=2011),
            [trace],
            l1_snapshot=_l1_snapshot(cutoff_year=2012),
            built_at='2026-04-02T09:20:00Z',
        )


def test_route_state_synthesizer_prefers_explicit_l1_snapshot_over_packet_placeholder_ref() -> None:
    trace = _trace('paper-thin', 2011, 'pt', thin=True)
    packet = _packet([trace], cutoff_year=2011)
    packet['l1_snapshot_ref']['snapshot_id'] = 'packet_placeholder_snapshot'

    route_state = synthesize_route_state(
        packet,
        [trace],
        l1_snapshot=_l1_snapshot(cutoff_year=2011, snapshot_id='actual_l1_snapshot_v1'),
        built_at='2026-04-02T09:30:00Z',
    )

    assert route_state.source_packet.l1_snapshot_ref == 'actual_l1_snapshot_v1'
    assert route_state.compiler_metadata.l1_snapshot_version == 'actual_l1_snapshot_v1'
