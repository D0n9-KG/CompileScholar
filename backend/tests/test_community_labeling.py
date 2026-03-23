from __future__ import annotations

from app.community.labeling import label_community


def test_labeler_prefers_distinctive_method_phrase_over_generic_tokens() -> None:
    label = label_community(
        core_members=[
            {'summary': 'Uses relation-aware graph encoding for reasoning', 'paper_title': 'A'},
            {'summary': 'Proposes relation-aware graph representations', 'paper_title': 'B'},
        ],
        evidence_rows=[],
    )

    assert 'relation-aware graph' in label['title'].lower()
    assert label['keywords']


def test_labeler_prefers_structured_method_and_object_tokens_over_generic_summary_phrases() -> None:
    label = label_community(
        core_members=[
            {
                'summary': 'This work uses the method to investigate the effect of particle crushing.',
                'paper_title': 'A',
                'method_tokens': ['discrete element method'],
                'object_tokens': ['particle crushing'],
            },
            {
                'summary': 'To investigate the same effect, the method is calibrated against experiments.',
                'paper_title': 'B',
                'method_tokens': ['discrete element method'],
                'object_tokens': ['particle crushing'],
            },
        ],
        evidence_rows=[],
    )

    assert label['title'] == 'discrete element method'
    assert 'particle crushing' in label['keywords']


def test_labeler_merges_overlapping_signal_phrases_without_repeating_middle_token() -> None:
    label = label_community(
        core_members=[
            {
                'summary': 'Reports good agreement between simulations and experiments.',
                'paper_title': 'A',
                'method_tokens': ['good agreement', 'agreement found'],
            },
            {
                'summary': 'Good agreement is found for the validation results.',
                'paper_title': 'B',
                'method_tokens': ['good agreement', 'agreement found'],
            },
        ],
        evidence_rows=[],
    )

    assert label['title'] == 'good agreement found'
