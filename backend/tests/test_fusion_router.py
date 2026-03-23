from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import app.api.routers.fusion as fusion_router

def test_fusion_graph_returns_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        fusion_router,
        "get_fusion_graph",
        lambda limit_nodes=1000, limit_edges=3000: {"nodes": [{"id": "n1"}], "edges": [{"source": "a", "target": "b"}]},
    )
    out = fusion_router.fusion_graph()
    assert "nodes" in out
    assert "edges" in out


def test_fusion_router_maps_internal_errors_to_http_500(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(fusion_router, "list_fusion_sections_for_paper", _boom)
    try:
        fusion_router.fusion_sections("doi:1")
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 500


def test_fusion_sections_query_endpoint_accepts_doi_with_slash(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def _fake_sections(paper_id: str):
        seen["paper_id"] = paper_id
        return {"paper_id": paper_id, "sections": []}

    monkeypatch.setattr(fusion_router, "list_fusion_sections_for_paper", _fake_sections)

    app = FastAPI()
    app.include_router(fusion_router.router)
    client = TestClient(app)

    doi = "doi:10.1021/acs.iecr.7b04833"
    res = client.get("/fusion/sections", params={"paper_id": doi})
    assert res.status_code == 200
    assert res.json()["paper_id"] == doi
    assert seen["paper_id"] == doi


def test_fusion_basics_query_endpoint_accepts_doi_with_slash(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def _fake_basics(paper_id: str, role: str, limit: int = 50):
        seen["paper_id"] = paper_id
        seen["role"] = role
        seen["limit"] = str(limit)
        return {"paper_id": paper_id, "role": role, "basics": []}

    monkeypatch.setattr(fusion_router, "list_fusion_basics_for_role", _fake_basics)

    app = FastAPI()
    app.include_router(fusion_router.router)
    client = TestClient(app)

    doi = "doi:10.1021/acs.iecr.7b04833"
    res = client.get("/fusion/basics", params={"paper_id": doi, "role": "method", "limit": 7})
    assert res.status_code == 200
    payload = res.json()
    assert payload["paper_id"] == doi
    assert payload["role"] == "method"
    assert seen["paper_id"] == doi
    assert seen["role"] == "method"
    assert seen["limit"] == "7"
