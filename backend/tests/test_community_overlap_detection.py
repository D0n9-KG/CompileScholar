from __future__ import annotations

from app.community.overlap_detection import detect_overlapping_communities, merge_high_overlap_communities


def test_detector_allows_research_move_to_belong_to_multiple_communities() -> None:
    result = detect_overlapping_communities(
        nodes=['a', 'b', 'c'],
        edges=[
            {'source': 'a', 'target': 'b', 'weight': 0.9},
            {'source': 'a', 'target': 'c', 'weight': 0.88},
        ],
        max_memberships_per_node=2,
        min_community_size=2,
    )

    memberships = result['memberships']['a']

    assert len(memberships) == 2
    assert result['communities'][0]['member_ids']


def test_detector_can_expand_dense_triangle_into_single_three_node_community() -> None:
    result = detect_overlapping_communities(
        nodes=['a', 'b', 'c'],
        edges=[
            {'source': 'a', 'target': 'b', 'weight': 0.94},
            {'source': 'a', 'target': 'c', 'weight': 0.92},
            {'source': 'b', 'target': 'c', 'weight': 0.9},
        ],
        max_memberships_per_node=2,
        min_community_size=2,
    )

    community_sizes = sorted(len(row['member_ids']) for row in result['communities'])

    assert 3 in community_sizes


def test_detector_can_expand_sparse_but_consistent_three_node_community() -> None:
    result = detect_overlapping_communities(
        nodes=['a', 'b', 'c'],
        edges=[
            {'source': 'a', 'target': 'b', 'weight': 0.40},
            {'source': 'a', 'target': 'c', 'weight': 0.39},
            {'source': 'b', 'target': 'c', 'weight': 0.38},
        ],
        max_memberships_per_node=2,
        min_community_size=2,
    )

    community_sizes = sorted(len(row['member_ids']) for row in result['communities'])

    assert 3 in community_sizes


def test_detector_does_not_expand_pair_with_single_sided_attachment() -> None:
    result = detect_overlapping_communities(
        nodes=['a', 'b', 'c'],
        edges=[
            {'source': 'a', 'target': 'b', 'weight': 0.92},
            {'source': 'a', 'target': 'c', 'weight': 0.78},
        ],
        max_memberships_per_node=2,
        min_community_size=2,
    )

    community_sizes = sorted(len(row['member_ids']) for row in result['communities'])

    assert 3 not in community_sizes


def test_merge_high_overlap_communities_collapses_near_duplicate_clusters() -> None:
    edge_weight = {}
    dense_group = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    for idx, left in enumerate(dense_group):
        for right in dense_group[idx + 1 :]:
            edge_weight[frozenset((left, right))] = 0.86
    for extra in ('h', 'i'):
        for node in ('a', 'b', 'c', 'd', 'e', 'f', 'g'):
            edge_weight[frozenset((extra, node))] = 0.74

    communities = [
        {
            'community_id': 'gc-1',
            'member_ids': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
            'core_member_ids': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
            'confidence': 0.81,
        },
        {
            'community_id': 'gc-2',
            'member_ids': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'i'],
            'core_member_ids': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'i'],
            'confidence': 0.8,
        },
    ]

    merged = merge_high_overlap_communities(communities=communities, edge_weight=edge_weight, min_community_size=2)

    assert len(merged) == 1
    assert set(merged[0]['member_ids']) == {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'}
