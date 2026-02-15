"""Embedding generation utilities for Proposition clustering."""
from __future__ import annotations

import openai

from app.settings import settings


def get_embeddings_batch(texts: list[str], model: str | None = None) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts using OpenAI-compatible API.

    Args:
        texts: List of text strings to embed.
        model: Embedding model name (default: from settings or text-embedding-3-small)

    Returns:
        List of embedding vectors (each is list of floats)
    """
    if not texts:
        return []

    # Resolve model + credentials from shared settings abstraction
    model = model or settings.effective_embedding_model() or "text-embedding-3-small"
    api_key = settings.effective_embedding_api_key()
    if not api_key:
        raise ValueError("Embedding API key is not configured")
    base_url = settings.effective_embedding_base_url()

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # OpenAI API call
    response = client.embeddings.create(
        input=texts,
        model=model
    )

    # Extract embeddings in order
    embeddings = [item.embedding for item in response.data]

    return embeddings


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    import math

    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have same dimension")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)
