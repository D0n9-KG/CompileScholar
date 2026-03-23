from __future__ import annotations

import app.community.service as community_service
from app.community.service_v2 import rebuild_global_communities_v2


class _FakeClient:
    def __init__(self) -> None:
        self.communities: list[dict] = []
        self.keywords: list[dict] = []
        self.memberships: list[dict] = []
        self.cleared = False

    def ensure_schema(self) -> None:
        return None

    def list_research_moves(self, paper_id: str | None = None, limit: int = 50000) -> list[dict]:
        del paper_id, limit
        return [
            {
                'move_id': 'p1:method',
                'paper_id': 'p1',
                'paper_source': 'paper-1',
                'role': 'method',
                'act_type': 'propose_method',
                'summary': 'Uses relation-aware graph encoding for reasoning.',
                'sequence_no': 1,
                'method_tokens': ['relation-aware', 'graph', 'encoding'],
                'object_tokens': ['reasoning'],
            },
            {
                'move_id': 'p2:method',
                'paper_id': 'p2',
                'paper_source': 'paper-2',
                'role': 'method',
                'act_type': 'propose_method',
                'summary': 'Proposes relation-aware graph representations.',
                'sequence_no': 1,
                'method_tokens': ['relation-aware', 'graph', 'representations'],
                'object_tokens': ['reasoning'],
            },
            {
                'move_id': 'p3:method',
                'paper_id': 'p3',
                'paper_source': 'paper-3',
                'role': 'method',
                'act_type': 'propose_method',
                'summary': 'Applies relation-aware graph encoding to reasoning tasks.',
                'sequence_no': 1,
                'method_tokens': ['relation-aware', 'graph', 'encoding'],
                'object_tokens': ['reasoning'],
            },
        ]

    def list_paper_citation_pairs(self, paper_ids: list[str], limit: int = 50000) -> list[dict]:
        del paper_ids, limit
        return []

    def clear_global_communities(self) -> dict:
        self.cleared = True
        return {'deleted_communities': 0, 'deleted_keywords': 0, 'deleted_memberships': 0, 'deleted_keyword_edges': 0}

    def upsert_global_communities(self, items: list[dict]) -> int:
        self.communities = list(items)
        return len(items)

    def upsert_global_keywords(self, items: list[dict]) -> int:
        self.keywords = list(items)
        return len(items)

    def replace_global_memberships(self, items: list[dict]) -> int:
        self.memberships = list(items)
        return len(items)


def test_rebuild_global_communities_v2_materializes_cross_paper_move_clusters() -> None:
    client = _FakeClient()

    result = rebuild_global_communities_v2(client=client)

    assert result['communities'] == 1
    assert client.cleared is True
    assert client.communities[0]['paper_count'] == 3
    assert 'relation-aware graph' in client.communities[0]['title'].lower()
    assert any(row['member_id'] == 'p1:method' for row in client.memberships)
    assert all(row['member_kind'] == 'ResearchMove' for row in client.memberships)


class _PairOnlyClient(_FakeClient):
    def list_research_moves(self, paper_id: str | None = None, limit: int = 50000) -> list[dict]:
        del paper_id, limit
        return [
            {
                'move_id': 'p1:method',
                'paper_id': 'p1',
                'paper_source': 'paper-1',
                'role': 'method',
                'act_type': 'propose_method',
                'summary': 'Uses relation-aware graph encoding for reasoning.',
                'sequence_no': 1,
                'method_tokens': ['relation-aware', 'graph', 'encoding'],
                'object_tokens': ['reasoning'],
            },
            {
                'move_id': 'p2:method',
                'paper_id': 'p2',
                'paper_source': 'paper-2',
                'role': 'method',
                'act_type': 'propose_method',
                'summary': 'Proposes relation-aware graph representations.',
                'sequence_no': 1,
                'method_tokens': ['relation-aware', 'graph', 'representations'],
                'object_tokens': ['reasoning'],
            },
        ]


def test_rebuild_global_communities_v2_does_not_materialize_pair_only_clusters() -> None:
    client = _PairOnlyClient()

    result = rebuild_global_communities_v2(client=client)

    assert result['communities'] == 0
    assert client.communities == []
    assert client.memberships == []


def test_rebuild_global_communities_always_uses_v2_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(
        community_service,
        'rebuild_global_communities_v2',
        lambda **_kwargs: {'ok': True, 'version': 'v2'},
    )

    result = community_service.rebuild_global_communities()

    assert result['version'] == 'v2'
