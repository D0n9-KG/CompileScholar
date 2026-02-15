from __future__ import annotations

import json

from app.similarity import service as similarity_service


def test_embedding_502_degradation_is_explicit(monkeypatch, tmp_path):
    captured: dict[str, str | None] = {"claim_mode": None}

    class _FakeNeo4jClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def list_claim_similarity_rows(self, paper_id: str | None = None):
            return [
                {"node_id": "c1", "paper_id": "p1", "text": "Granular flow increases with vibration."},
                {"node_id": "c2", "paper_id": "p2", "text": "Vibration increases granular flow rate."},
            ]

        def list_logic_step_similarity_rows(self, paper_id: str | None = None):
            return []

        def replace_similar_claim_edges_batch(self, items, model, built_at, mode="embedding"):
            captured["claim_mode"] = mode

        def replace_similar_logic_edges_batch(self, items, model, built_at):
            return None

    class _FailingEmbeddingClient:
        def embed_documents(self, texts):
            raise RuntimeError("Embedding API error 502: Bad Gateway")

    claim_meta_path = tmp_path / "claim_meta.json"
    logic_meta_path = tmp_path / "logic_meta.json"

    monkeypatch.setattr(similarity_service, "Neo4jClient", _FakeNeo4jClient)
    monkeypatch.setattr(similarity_service, "faiss", object())  # ensure degradation is from embedding, not faiss missing
    monkeypatch.setattr(similarity_service, "_embedding_client", lambda: _FailingEmbeddingClient())
    monkeypatch.setattr(similarity_service, "_write_items", lambda kind, items: None)
    monkeypatch.setattr(similarity_service, "_save_embeddings", lambda kind, x: None)
    monkeypatch.setattr(
        similarity_service,
        "_meta_path",
        lambda kind: claim_meta_path if kind == "claim" else logic_meta_path,
    )

    logs: list[str] = []
    out = similarity_service.rebuild_similarity_global(log=logs.append)

    assert out["mode"] == "lexical"
    assert out["embedding_degraded"] is True
    assert "502" in str(out["degradation_reason"])
    assert captured["claim_mode"] == "lexical"
    assert any("falling back to lexical mode" in line for line in logs)

    claim_meta = json.loads(claim_meta_path.read_text(encoding="utf-8"))
    assert claim_meta["mode"] == "lexical"
    assert claim_meta["embedding_degraded"] is True
    assert "502" in str(claim_meta["degradation_reason"])
