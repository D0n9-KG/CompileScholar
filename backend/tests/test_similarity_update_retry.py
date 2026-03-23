"""
Test retry logic for update_similarity_for_paper on the ResearchMove-only path.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.similarity import service as similarity_service


def _setup_update_similarity_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    move_rows: list[dict[str, str]],
) -> None:
    move_items_path = tmp_path / 'move_items.jsonl'
    move_meta_path = tmp_path / 'move_meta.json'
    move_emb_path = tmp_path / 'move_embeddings.npy'
    move_neighbors_path = tmp_path / 'move_neighbors.json'

    move_items_path.write_text('', encoding='utf-8')
    move_meta_path.write_text(json.dumps({'mode': 'embedding'}), encoding='utf-8')
    np.save(str(move_emb_path), np.zeros((0, 3), dtype=np.float32))

    class _FakeNeo4jClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def list_research_moves(self, paper_id: str | None = None, limit: int = 50000):
            return move_rows

    monkeypatch.setattr(similarity_service, 'Neo4jClient', _FakeNeo4jClient)
    monkeypatch.setattr(similarity_service, '_items_path', lambda kind: move_items_path)
    monkeypatch.setattr(similarity_service, '_meta_path', lambda kind: move_meta_path)
    monkeypatch.setattr(similarity_service, '_emb_path', lambda kind: move_emb_path)
    monkeypatch.setattr(similarity_service, '_neighbors_path', lambda kind: move_neighbors_path)
    monkeypatch.setattr(similarity_service, 'faiss', object())
    monkeypatch.setattr(similarity_service, '_build_index', lambda _x: object())
    monkeypatch.setattr(similarity_service, '_topk_pairs', lambda *args, **kwargs: [])
    monkeypatch.setattr(similarity_service, '_write_items', lambda kind, items: None)
    monkeypatch.setattr(similarity_service, '_save_embeddings', lambda kind, x: None)
    monkeypatch.setattr(similarity_service, '_write_neighbors', lambda kind, rows: None)


def test_embedding_retry_success_on_third_attempt(monkeypatch, tmp_path):
    paper_id = 'test_paper_001'
    _setup_update_similarity_env(
        monkeypatch,
        tmp_path,
        move_rows=[{'move_id': 'move_001', 'paper_id': paper_id, 'summary': 'Test move 1', 'role': 'method', 'act_type': 'propose_method'}],
    )

    sleep_calls: list[float] = []
    monkeypatch.setattr('time.sleep', lambda seconds: sleep_calls.append(seconds))

    call_count = {'count': 0}

    class _FlakyEmbeddingClient:
        def embed_documents(self, texts):
            call_count['count'] += 1
            if call_count['count'] < 3:
                raise RuntimeError('Error code: 502')
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(similarity_service, '_embedding_client', lambda: _FlakyEmbeddingClient())

    result = similarity_service.update_similarity_for_paper(paper_id)

    assert result.get('ok') is True
    assert result.get('mode') == 'embedding'
    assert result.get('research_moves_updated') == 1
    assert call_count['count'] == 3
    assert sleep_calls == [similarity_service._backoff_delay(i) for i in range(2)]


def test_embedding_retry_fails_after_three_attempts(monkeypatch, tmp_path):
    paper_id = 'test_paper_002'
    _setup_update_similarity_env(
        monkeypatch,
        tmp_path,
        move_rows=[{'move_id': 'move_002', 'paper_id': paper_id, 'summary': 'Test move 2', 'role': 'method', 'act_type': 'propose_method'}],
    )

    sleep_calls: list[float] = []
    monkeypatch.setattr('time.sleep', lambda seconds: sleep_calls.append(seconds))

    call_count = {'count': 0}

    class _FailingEmbeddingClient:
        def embed_documents(self, texts):
            call_count['count'] += 1
            raise RuntimeError('Error code: 400')

    monkeypatch.setattr(similarity_service, '_embedding_client', lambda: _FailingEmbeddingClient())

    with pytest.raises(RuntimeError) as ctx:
        similarity_service.update_similarity_for_paper(paper_id)

    error_msg = str(ctx.value)
    assert '3 attempts' in error_msg
    assert 'embedding unavailable' in error_msg.lower()
    assert call_count['count'] == 3
    assert sleep_calls == [similarity_service._STABLE_DELAY, similarity_service._STABLE_DELAY]


def test_embedding_never_falls_back_to_lexical(monkeypatch, tmp_path):
    paper_id = 'test_paper_003'
    _setup_update_similarity_env(
        monkeypatch,
        tmp_path,
        move_rows=[{'move_id': 'move_003', 'paper_id': paper_id, 'summary': 'Test move 3', 'role': 'method', 'act_type': 'propose_method'}],
    )

    def _unexpected_rebuild(*args, **kwargs):
        raise AssertionError('rebuild_similarity_global should not be called - hot path failed')

    class _StableEmbeddingClient:
        def embed_documents(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(similarity_service, 'rebuild_similarity_global', _unexpected_rebuild)
    monkeypatch.setattr(similarity_service, '_embedding_client', lambda: _StableEmbeddingClient())
    monkeypatch.setattr('time.sleep', lambda seconds: None)

    result = similarity_service.update_similarity_for_paper(paper_id)

    assert result.get('ok') is True
    assert result.get('mode') == 'embedding'
    assert result.get('mode') != 'lexical'
    assert result.get('mode') != 'mixed'
