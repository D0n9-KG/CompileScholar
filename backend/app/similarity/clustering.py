"""Agglomerative clustering for Proposition grouping."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def cluster_propositions(
    embeddings: list[list[float]],
    texts: list[str],
    threshold: float = 0.85,
    min_shared_anchors: int = 1,
) -> list[dict[str, Any]]:
    """
    Cluster propositions using Agglomerative Clustering with constraints.

    Args:
        embeddings: List of embedding vectors
        texts: List of proposition texts (parallel to embeddings)
        threshold: Similarity threshold for merging (0.82-0.88 recommended)
        min_shared_anchors: Minimum shared anchor words required (currently unused)

    Returns:
        List of groups, each containing member indices and representative text
    """
    if not embeddings or len(embeddings) != len(texts):
        return []

    # Handle single item or very small datasets
    if len(embeddings) == 1:
        return [{
            "label": 0,
            "representative_text": texts[0],
            "member_indices": [0],
            "member_count": 1,
            "avg_similarity": 1.0,
        }]

    # Convert to numpy array
    X = np.array(embeddings)

    # Agglomerative clustering with distance threshold
    # distance_threshold = 1 - cosine similarity threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - threshold,
        metric="cosine",
        linkage="average"
    )

    labels = clustering.fit_predict(X)

    # Group by cluster labels
    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(idx)

    # Build group structures
    result = []
    for label, member_indices in groups.items():
        # Representative text: earliest/most common
        representative_text = texts[member_indices[0]]

        # Calculate avg similarity within group
        group_embeddings = [embeddings[i] for i in member_indices]
        centroid = np.mean(group_embeddings, axis=0)

        from app.similarity.embedding import cosine_similarity
        scores = [cosine_similarity(embeddings[i], centroid.tolist()) for i in member_indices]

        result.append({
            "label": label,
            "representative_text": representative_text,
            "member_indices": member_indices,
            "member_count": len(member_indices),
            "avg_similarity": float(np.mean(scores)),
        })

    return result
