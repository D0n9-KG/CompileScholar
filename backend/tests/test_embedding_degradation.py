from __future__ import annotations

import pytest

from app.similarity import service as similarity_service


def test_embedding_502_degradation_is_explicit(monkeypatch, tmp_path):
    """
    When embedding API returns 502, rebuild_similarity_global raises RuntimeError
    after exhausting all 3 retries.  No lexical fallback exists in current code.
    """

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

    class _FailingEmbeddingClient:
        def embed_documents(self, texts):
            raise RuntimeError("Embedding API error 502: Bad Gateway")

    # Track sleep calls so the test doesn't actually pause for 10 seconds.
    sleep_calls: list[float] = []
    monkeypatch.setattr("time.sleep", lambda seconds: sleep_calls.append(seconds))

    monkeypatch.setattr(similarity_service, "Neo4jClient", _FakeNeo4jClient)
    # Ensure faiss is non-None so the failure comes from embedding, not from missing faiss.
    monkeypatch.setattr(similarity_service, "faiss", object())
    monkeypatch.setattr(similarity_service, "_embedding_client", lambda: _FailingEmbeddingClient())
    monkeypatch.setattr(similarity_service, "_write_items", lambda kind, items: None)
    monkeypatch.setattr(similarity_service, "_save_embeddings", lambda kind, x: None)
    monkeypatch.setattr(
        similarity_service,
        "_meta_path",
        lambda kind: tmp_path / f"{kind}_meta.json",
    )

    logs: list[str] = []

    with pytest.raises(RuntimeError) as ctx:
        similarity_service.rebuild_similarity_global(log=logs.append)

    error_msg = str(ctx.value)
    # Error message must mention retries exhausted and the original 502.
    assert "3 attempts" in error_msg
    assert "embedding unavailable" in error_msg.lower()
    assert "502" in error_msg

    # Two sleeps: after attempt 1 and after attempt 2 (not after the final failure).
    assert sleep_calls == [5, 5]

    # No meta files written – exception occurred before any successful embedding.
    assert not (tmp_path / "claim_meta.json").exists()
    assert not (tmp_path / "logic_meta.json").exists()

    # Retry progress was logged.
    assert any("attempt 1/3" in line.lower() for line in logs)
    assert any("failed after 3 attempts" in line.lower() for line in logs)
