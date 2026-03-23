from __future__ import annotations

from app.community.candidate_graph import build_move_candidate_graph


def test_candidate_graph_ignores_same_paper_neighbors_and_keeps_cross_paper_edges() -> None:
    graph = build_move_candidate_graph(
        moves=[
            {'move_id': 'a', 'paper_id': 'p1', 'summary': 'graph encoding'},
            {'move_id': 'b', 'paper_id': 'p2', 'summary': 'relation-aware graph encoding'},
            {'move_id': 'c', 'paper_id': 'p1', 'summary': 'intra paper detail'},
        ],
        similar_move_edges=[
            {'source': 'a', 'target': 'b', 'score': 0.91},
            {'source': 'a', 'target': 'c', 'score': 0.97},
        ],
        shared_signal_edges=[],
        citation_boosts=[],
    )

    assert graph['nodes'] == ['a', 'b', 'c']
    assert graph['edges'] == [{'source': 'a', 'target': 'b', 'weight': 0.91}]
