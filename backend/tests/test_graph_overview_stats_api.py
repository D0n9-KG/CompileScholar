from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.routers.graph as graph_router


class _FakeNeo4jClient:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self.uri = uri
        self.user = user
        self.password = password

    def __enter__(self) -> "_FakeNeo4jClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def get_overview_stats(self) -> dict:
        return {
            "paper_count": 19,
            "research_move_count": 553,
            "global_community_count": 33,
            "ready_for_l3_count": 19,
            "ready_for_l4_count": 18,
        }


def test_graph_overview_stats_endpoint_returns_l2_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(graph_router, "Neo4jClient", _FakeNeo4jClient)

    app = FastAPI()
    app.include_router(graph_router.router)
    client = TestClient(app)

    res = client.get("/graph/overview-stats")

    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload == {
        "paper_count": 19,
        "research_move_count": 553,
        "global_community_count": 33,
        "ready_for_l3_count": 19,
        "ready_for_l4_count": 18,
    }
