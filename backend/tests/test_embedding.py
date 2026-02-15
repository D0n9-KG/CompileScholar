"""Tests for embedding generation utilities."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.similarity.embedding import cosine_similarity, get_embeddings_batch


class _FakeEmbeddingsResource:
    """Mock OpenAI embeddings resource for testing."""

    def __init__(self, calls: dict[str, object]) -> None:
        self._calls = calls

    def create(self, *, input: list[str], model: str) -> SimpleNamespace:
        self._calls["input"] = input
        self._calls["model"] = model
        payload = [SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in input]
        return SimpleNamespace(data=payload)


class _FakeOpenAIClient:
    """Mock OpenAI client for testing."""

    def __init__(self, calls: dict[str, object], **kwargs: object) -> None:
        self.embeddings = _FakeEmbeddingsResource(calls)
        calls["client_kwargs"] = kwargs


def test_embedding_generation_uses_settings_credentials(monkeypatch: pytest.MonkeyPatch):
    """Test that embedding generation uses Settings abstraction for credentials."""
    calls: dict[str, object] = {}

    # Mock the Settings class methods before import
    mock_settings = Mock()
    mock_settings.effective_embedding_api_key.return_value = "embed-key"
    mock_settings.effective_embedding_base_url.return_value = "http://192.168.199.73/v1"
    mock_settings.effective_embedding_model.return_value = "qwen3-embedding-8b-local"

    monkeypatch.setattr("app.similarity.embedding.settings", mock_settings)
    monkeypatch.setattr(
        "app.similarity.embedding.openai.OpenAI",
        lambda **kwargs: _FakeOpenAIClient(calls, **kwargs),
    )

    embeddings = get_embeddings_batch(["Particle friction affects flow", "Flow is influenced by friction"])

    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    assert isinstance(embeddings[0][0], float)
    assert calls["client_kwargs"] == {"api_key": "embed-key", "base_url": "http://192.168.199.73/v1"}
    assert calls["model"] == "qwen3-embedding-8b-local"


def test_embedding_generation_prefers_explicit_model(monkeypatch: pytest.MonkeyPatch):
    """Test that explicit model parameter overrides settings default."""
    calls: dict[str, object] = {}

    mock_settings = Mock()
    mock_settings.effective_embedding_api_key.return_value = "embed-key"
    mock_settings.effective_embedding_base_url.return_value = "http://192.168.199.73/v1"
    mock_settings.effective_embedding_model.return_value = "default-model"

    monkeypatch.setattr("app.similarity.embedding.settings", mock_settings)
    monkeypatch.setattr(
        "app.similarity.embedding.openai.OpenAI",
        lambda **kwargs: _FakeOpenAIClient(calls, **kwargs),
    )

    _ = get_embeddings_batch(["text"], model="override-model")
    assert calls["model"] == "override-model"


def test_embedding_generation_requires_embedding_api_key(monkeypatch: pytest.MonkeyPatch):
    """Test that missing API key raises clear error."""
    mock_settings = Mock()
    mock_settings.effective_embedding_api_key.return_value = None

    monkeypatch.setattr("app.similarity.embedding.settings", mock_settings)

    with pytest.raises(ValueError, match="Embedding API key is not configured"):
        _ = get_embeddings_batch(["text"])


def test_embedding_generation_handles_empty_input():
    """Test that empty input returns empty list."""
    result = get_embeddings_batch([])
    assert result == []


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0]
    vec_c = [1.0, 0.0, 0.0]

    sim_ab = cosine_similarity(vec_a, vec_b)
    sim_ac = cosine_similarity(vec_a, vec_c)

    assert abs(sim_ab - 0.0) < 0.01  # Orthogonal vectors
    assert abs(sim_ac - 1.0) < 0.01  # Identical vectors


def test_cosine_similarity_zero_vector():
    """Test cosine similarity with zero vector."""
    vec_a = [1.0, 2.0, 3.0]
    vec_zero = [0.0, 0.0, 0.0]

    sim = cosine_similarity(vec_a, vec_zero)
    assert sim == 0.0


def test_cosine_similarity_dimension_mismatch():
    """Test that dimension mismatch raises error."""
    vec_a = [1.0, 2.0]
    vec_b = [1.0, 2.0, 3.0]

    with pytest.raises(ValueError, match="Vectors must have same dimension"):
        _ = cosine_similarity(vec_a, vec_b)
