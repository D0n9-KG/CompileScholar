from __future__ import annotations

import json

from app.graph.neo4j_client import Neo4jClient


class _Result:
    def __init__(self, row: dict | None = None) -> None:
        self._row = row or {"cnt": 0}

    def single(self):
        return self._row

    def __iter__(self):
        return iter(())


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.community_members: dict[str, list[dict]] = {}
        self.membership_edge_rows: list[dict] = []

    def run(self, query: str, **params):
        self.calls.append((str(query), dict(params)))
        if "RETURN count(DISTINCT gc) AS cnt" in str(query):
            return _Result({"cnt": 1})
        if "RETURN count(hk) AS cnt" in str(query):
            return _Result({"cnt": 2})
        if "SET gc.member_rows_json" in str(query):
            rows = list(params.get("rows") or [])
            self.community_members = {
                str(row.get("community_id") or "").strip(): json.loads(str(row.get("member_rows_json") or "[]"))
                for row in rows
                if str(row.get("community_id") or "").strip()
            }
            return _Result({"cnt": sum(len(members) for members in self.community_members.values())})
        if "IN_GLOBAL_COMMUNITY" in str(query) and "MERGE (member)-[ig:IN_GLOBAL_COMMUNITY]->(gc)" in str(query):
            rows = list(params.get("rows") or [])
            self.membership_edge_rows.extend(rows)
            return _Result({"cnt": len(rows)})
        if "count(gc) AS deleted_communities" in str(query):
            return _Result(
                {
                    "deleted_communities": 1,
                    "deleted_keywords": 2,
                    "deleted_memberships": 2,
                    "deleted_keyword_edges": 2,
                }
            )
        if "RETURN gc.community_id AS community_id" in str(query):
            return [
                {
                    "community_id": "gc:demo",
                    "title": "Finite element stability",
                    "summary": "Research moves and evidence anchors about FEM stability.",
                    "member_count": 2,
                    "keywords": ["finite element", "stability"],
                }
            ]
        if "RETURN gc.member_rows_json AS member_rows_json" in str(query):
            community_id = str(params.get("community_id") or "").strip()
            return _Result({"member_rows_json": json.dumps(self.community_members.get(community_id, []))})
        return _Result({"cnt": 0})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeDriver:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    def session(self):
        return self._session

    def close(self):
        return None


def _client_with_fake_driver(fake_session: _FakeSession) -> Neo4jClient:
    client = object.__new__(Neo4jClient)
    client._driver = _FakeDriver(fake_session)
    return client


def test_global_community_writer_helpers_use_global_labels_and_edges() -> None:
    fake_session = _FakeSession()
    client = _client_with_fake_driver(fake_session)

    assert hasattr(client, "clear_global_communities"), "Expected clear_global_communities() to exist."
    assert hasattr(client, "upsert_global_communities"), "Expected upsert_global_communities() to exist."
    assert hasattr(client, "upsert_global_keywords"), "Expected upsert_global_keywords() to exist."
    assert hasattr(client, "replace_global_memberships"), "Expected replace_global_memberships() to exist."

    deleted = client.clear_global_communities()
    written = client.upsert_global_communities(
        [
            {
                "community_id": "gc:demo",
                "title": "Finite element stability",
                "summary": "Research moves and evidence anchors about FEM stability.",
                "confidence": 0.88,
                "member_count": 2,
                "version": "v1",
            }
        ]
    )
    keyword_edges = client.upsert_global_keywords(
        [
            {
                "community_id": "gc:demo",
                "keyword_id": "gk:demo:1",
                "keyword": "finite element",
                "rank": 1,
                "weight": 0.82,
            },
            {
                "community_id": "gc:demo",
                "keyword_id": "gk:demo:2",
                "keyword": "stability",
                "rank": 2,
                "weight": 0.71,
            },
        ]
    )
    membership_edges = client.replace_global_memberships(
        [
            {
                "community_id": "gc:demo",
                "member_id": "move-1",
                "member_kind": "ResearchMove",
                "weight": 0.91,
                "text": "FEM improves stability.",
                "paper_id": "doi:10.1000/demo",
                "paper_source": "paper-A",
                "paper_title": "Finite element paper",
                "role": "result",
            },
            {
                "community_id": "gc:demo",
                "member_id": "anchor-1",
                "member_kind": "EvidenceAnchor",
                "weight": 0.73,
                "text": "Finite Element Method",
                "paper_id": "doi:10.1000/demo",
                "paper_source": "paper-A",
                "paper_title": "Finite element paper",
                "role": "result",
            },
        ]
    )

    assert deleted["deleted_communities"] == 1
    assert written == 1
    assert keyword_edges == 2
    assert membership_edges == 2

    queries = "\n".join(query for query, _ in fake_session.calls)
    assert "GlobalCommunity" in queries
    assert "GlobalKeyword" in queries
    assert "HAS_GLOBAL_KEYWORD" in queries
    assert "member_rows_json" in queries
    assert "MERGE (member)-[ig:IN_GLOBAL_COMMUNITY]->(gc)" in queries
    assert [row["member_id"] for row in fake_session.membership_edge_rows] == ["move-1", "anchor-1"]


def test_global_community_read_helpers_return_keywords_and_members() -> None:
    fake_session = _FakeSession()
    fake_session.community_members = {
        "gc:demo": [
            {
                "member_id": "move-1",
                "member_kind": "ResearchMove",
                "text": "FEM improves stability.",
            },
            {
                "member_id": "anchor-1",
                "member_kind": "EvidenceAnchor",
                "text": "Finite Element Method",
            },
        ]
    }
    client = _client_with_fake_driver(fake_session)

    assert hasattr(client, "list_global_community_rows"), "Expected list_global_community_rows() to exist."
    assert hasattr(client, "list_global_community_members"), "Expected list_global_community_members() to exist."

    rows = client.list_global_community_rows(limit=20)
    members = client.list_global_community_members("gc:demo", limit=10)

    assert rows == [
        {
            "community_id": "gc:demo",
            "title": "Finite element stability",
            "summary": "Research moves and evidence anchors about FEM stability.",
            "member_count": 2,
            "keywords": ["finite element", "stability"],
        }
    ]
    assert members == [
        {"member_id": "move-1", "member_kind": "ResearchMove", "text": "FEM improves stability."},
        {"member_id": "anchor-1", "member_kind": "EvidenceAnchor", "text": "Finite Element Method"},
    ]


def test_global_community_members_reader_uses_embedded_membership_rows_json() -> None:
    fake_session = _FakeSession()
    client = _client_with_fake_driver(fake_session)

    client.list_global_community_members("gc:demo", limit=10)

    queries = "\n".join(query for query, _ in fake_session.calls)
    assert "member_rows_json" in queries


def test_legacy_proposition_cleanup_helper_deletes_groups_nodes_and_relation_edges() -> None:
    class _CleanupSession(_FakeSession):
        def run(self, query: str, **params):
            self.calls.append((str(query), dict(params)))
            if "deleted_proposition_groups" in str(query):
                return _Result(
                    {
                        "deleted_proposition_groups": 2,
                        "deleted_propositions": 3,
                        "deleted_relation_edges": 4,
                    }
                )
            return _Result({"cnt": 0})

    fake_session = _CleanupSession()
    client = _client_with_fake_driver(fake_session)

    assert hasattr(
        client,
        "clear_legacy_proposition_artifacts",
    ), "Expected clear_legacy_proposition_artifacts() to exist."

    deleted = client.clear_legacy_proposition_artifacts()

    assert deleted == {
        "deleted_proposition_groups": 2,
        "deleted_propositions": 3,
        "deleted_relation_edges": 4,
    }

    queries = "\n".join(query for query, _ in fake_session.calls)
    assert "PropositionGroup" in queries
    assert "Proposition" in queries
    assert "SUPPORTS" in queries
    assert "CHALLENGES" in queries
    assert "SUPERSEDES" in queries


def test_legacy_proposition_schema_cleanup_helper_drops_constraints_and_indexes() -> None:
    fake_session = _FakeSession()
    client = _client_with_fake_driver(fake_session)

    dropped = client.drop_legacy_proposition_schema()

    assert dropped == {
        "dropped_constraints": 3,
        "dropped_indexes": 2,
    }

    queries = "\n".join(query for query, _ in fake_session.calls)
    assert "DROP CONSTRAINT proposition_id_unique IF EXISTS" in queries
    assert "DROP CONSTRAINT proposition_key_unique IF EXISTS" in queries
    assert "DROP CONSTRAINT proposition_group_id_unique IF EXISTS" in queries
    assert "DROP INDEX proposition_state IF EXISTS" in queries
    assert "DROP INDEX proposition_score IF EXISTS" in queries


