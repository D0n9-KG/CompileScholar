from __future__ import annotations

from app.community.refinement import disambiguate_community_labels, merge_labeled_communities


def test_merge_labeled_communities_merges_same_title_high_paper_overlap_clusters() -> None:
    communities = [
        {
            'community_id': 'gc-1',
            'member_ids': ['m1', 'm2', 'm3', 'm4'],
            'core_member_ids': ['m1', 'm2', 'm3', 'm4'],
            'confidence': 0.82,
        },
        {
            'community_id': 'gc-2',
            'member_ids': ['m1', 'm2', 'm3', 'm5'],
            'core_member_ids': ['m1', 'm2', 'm3', 'm5'],
            'confidence': 0.81,
        },
    ]
    labels = {
        'gc-1': {
            'title': 'discrete element method',
            'summary': '',
            'keywords': ['discrete element method', 'particle crushing'],
        },
        'gc-2': {
            'title': 'discrete element method',
            'summary': '',
            'keywords': ['discrete element method', 'particle crushing'],
        },
    }
    moves_by_id = {
        'm1': {'paper_id': 'p1', 'role': 'method'},
        'm2': {'paper_id': 'p2', 'role': 'method'},
        'm3': {'paper_id': 'p3', 'role': 'method'},
        'm4': {'paper_id': 'p4', 'role': 'method'},
        'm5': {'paper_id': 'p4', 'role': 'method'},
    }

    refined = merge_labeled_communities(
        communities=communities,
        labels=labels,
        moves_by_id=moves_by_id,
        max_memberships_per_node=2,
        min_community_size=2,
    )

    assert len(refined['communities']) == 1
    assert set(refined['communities'][0]['member_ids']) == {'m1', 'm2', 'm3', 'm4', 'm5'}


def test_disambiguate_community_labels_uses_secondary_keyword_for_duplicate_titles() -> None:
    communities = [
        {'community_id': 'gc-1', 'member_ids': ['m1', 'm2'], 'confidence': 0.9, 'member_count': 2},
        {'community_id': 'gc-2', 'member_ids': ['m3', 'm4'], 'confidence': 0.8, 'member_count': 2},
    ]
    labels = {
        'gc-1': {
            'title': 'discrete element method',
            'summary': '',
            'keywords': ['discrete element method', 'particle crushing', 'granular soils'],
        },
        'gc-2': {
            'title': 'discrete element method',
            'summary': '',
            'keywords': ['discrete element method', 'mixing performance', 'pharmaceutical powders'],
        },
    }
    moves_by_id = {
        'm1': {'role': 'method'},
        'm2': {'role': 'method'},
        'm3': {'role': 'method'},
        'm4': {'role': 'method'},
    }

    updated = disambiguate_community_labels(
        communities=communities,
        labels=labels,
        moves_by_id=moves_by_id,
    )

    assert updated['gc-1']['title'] == 'discrete element method'
    assert updated['gc-2']['title'] == 'mixing performance'


def test_disambiguate_community_labels_promotes_more_specific_future_work_keyword() -> None:
    communities = [
        {'community_id': 'gc-1', 'member_ids': ['m1', 'm2', 'm3'], 'confidence': 0.77, 'member_count': 3},
    ]
    labels = {
        'gc-1': {
            'title': 'future work',
            'summary': '',
            'keywords': ['future work', 'granular media', 'larger polydispersity'],
        },
    }
    moves_by_id = {
        'm1': {'role': 'future_work'},
        'm2': {'role': 'future_work'},
        'm3': {'role': 'future_work'},
    }

    updated = disambiguate_community_labels(
        communities=communities,
        labels=labels,
        moves_by_id=moves_by_id,
    )

    assert updated['gc-1']['title'] == 'granular media'


def test_disambiguate_community_labels_promotes_more_specific_experimental_setup_keyword() -> None:
    communities = [
        {'community_id': 'gc-1', 'member_ids': ['m1', 'm2'], 'confidence': 0.83, 'member_count': 2},
    ]
    labels = {
        'gc-1': {
            'title': 'experimental setup',
            'summary': '',
            'keywords': ['experimental setup', 'simulation setup', 'u-shaped ribbon mixer'],
        },
    }
    moves_by_id = {
        'm1': {'role': 'method'},
        'm2': {'role': 'experiment'},
    }

    updated = disambiguate_community_labels(
        communities=communities,
        labels=labels,
        moves_by_id=moves_by_id,
    )

    assert updated['gc-1']['title'] == 'simulation setup'
