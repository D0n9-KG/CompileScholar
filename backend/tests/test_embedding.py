"""Tests for embedding generation utilities."""
import pytest
from app.similarity.embedding import get_embeddings_batch, cosine_similarity


def test_embedding_generation():
    """Test embedding generation (requires API key)."""
    texts = ["Particle friction affects flow", "Flow is influenced by friction"]

    embeddings = get_embeddings_batch(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0  # Should have dimension > 0
    assert isinstance(embeddings[0][0], float)


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0]
    vec_c = [1.0, 0.0, 0.0]

    sim_ab = cosine_similarity(vec_a, vec_b)
    sim_ac = cosine_similarity(vec_a, vec_c)

    assert abs(sim_ab - 0.0) < 0.01  # Orthogonal vectors
    assert abs(sim_ac - 1.0) < 0.01  # Identical vectors
