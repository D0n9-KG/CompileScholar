from __future__ import annotations

from app.paper_logic_trace.derived_views import (
    build_community_signatures,
    build_l1_bridge_hints,
    build_route_feature_candidates,
)
from app.paper_logic_trace.models import MentionValue, ResearchMove, SlotProvenance


def _build_move() -> ResearchMove:
    return ResearchMove(
        move_id='m-1',
        sequence_no=1,
        role='method',
        act_type='propose_method',
        summary='Uses graph neural network modeling for retrieval.',
        research_objects=[
            MentionValue(
                surface='entity relation graph',
                normalized='entity relation graph',
                anchor_ids=['a-1'],
            )
        ],
        methods=[
            MentionValue(
                surface='graph neural network',
                normalized='graph neural network',
                anchor_ids=['a-1'],
            )
        ],
        metrics=[
            MentionValue(
                surface='MRR',
                normalized='mean reciprocal rank',
                anchor_ids=['a-2'],
            )
        ],
        conditions=[
            MentionValue(
                surface='low-resource setting',
                normalized='low-resource setting',
                anchor_ids=['a-3'],
            )
        ],
        comparators=[
            MentionValue(
                surface='baseline retriever',
                normalized='baseline retriever',
                anchor_ids=['a-4'],
            )
        ],
        limitation_types=[
            MentionValue(
                surface='computational cost',
                normalized='computational cost',
                anchor_ids=['a-6'],
            )
        ],
        resource_mentions=[
            MentionValue(
                surface='WN18RR',
                normalized='wn18rr',
                type='benchmark',
                anchor_ids=['a-5'],
            )
        ],
        anchor_ids=['a-1', 'a-2', 'a-3', 'a-4', 'a-5'],
        slot_provenance=[
            SlotProvenance(
                field='resource_mentions',
                value_index=0,
                anchor_ids=['a-5'],
                extraction_mode='direct',
                support_strength='exact',
                confidence=0.95,
            )
        ],
        confidence=0.9,
    )


def test_build_community_signatures_uses_move_slots_not_raw_summary() -> None:
    signatures = build_community_signatures('paper-1', [_build_move()])

    assert signatures[0]['move_id'] == 'm-1'
    assert 'graph neural network' in signatures[0]['method_tokens']
    assert 'entity relation graph' in signatures[0]['object_tokens']


def test_build_route_feature_candidates_points_back_to_canonical_moves() -> None:
    candidates = build_route_feature_candidates([_build_move()])
    candidate_types = {candidate['candidate_type'] for candidate in candidates}

    assert candidates[0]['move_id'] == 'm-1'
    assert 'method_candidate' in candidate_types
    assert 'metric_candidate' in candidate_types
    assert 'condition_candidate' in candidate_types
    assert 'comparison_candidate' in candidate_types
    assert 'benchmark_candidate' in candidate_types
    assert 'limitation_candidate' in candidate_types
    assert any(
        candidate['candidate_type'] == 'benchmark_candidate' and 'wn18rr' in candidate['tokens']
        for candidate in candidates
    )


def test_build_l1_bridge_hints_preserves_provenance_refs() -> None:
    hints = build_l1_bridge_hints([_build_move()])

    assert hints['resource_candidates'][0]['move_id'] == 'm-1'
    assert hints['resource_candidates'][0]['anchor_ids'] == ['a-5']
    assert hints['resource_candidates'][0]['provenance'][0]['field'] == 'resource_mentions'
