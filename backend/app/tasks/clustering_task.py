"""Async task for proposition clustering into semantic groups."""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from app.graph.neo4j_client import Neo4jClient
from app.settings import Settings
from app.similarity.clustering import cluster_propositions
from app.similarity.embedding import get_embeddings_batch

logger = logging.getLogger(__name__)


def run_proposition_clustering(task_id: str | None = None) -> dict[str, Any]:
    """
    Async task: Cluster propositions into semantic groups.

    Args:
        task_id: Optional task ID for status tracking

    Returns:
        Result summary dict with status, groups_created, propositions_clustered
    """
    try:
        logger.info(f"Starting proposition clustering task: {task_id}")

        settings = Settings()

        # 1. Fetch all propositions from Neo4j
        with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
            query = """
            MATCH (p:Proposition)
            RETURN p.prop_id as prop_id,
                   p.canonical_text as text,
                   coalesce(p.paper_count, 0) as paper_count
            ORDER BY p.created_at
            """
            with client._driver.session() as session:
                result = session.run(query)
                propositions = [dict(record) for record in result]

        if not propositions:
            logger.warning("No propositions found for clustering")
            return {"status": "completed", "groups_created": 0, "propositions_clustered": 0}

        logger.info(f"Fetched {len(propositions)} propositions")

        # 2. Generate embeddings
        texts = [p["text"] for p in propositions]
        embedding_model = settings.effective_embedding_model() or "text-embedding-3-small"
        embeddings = get_embeddings_batch(texts, model=embedding_model)

        logger.info(f"Generated {len(embeddings)} embeddings")

        # 3. Cluster propositions
        # TODO: Get threshold from schema rules in future
        threshold = 0.85
        groups = cluster_propositions(embeddings, texts, threshold=threshold)

        logger.info(f"Found {len(groups)} proposition groups")

        # 4. Write PropositionGroups to Neo4j (clean rebuild)
        with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
            cleanup_stats = _clear_existing_proposition_groups(client)
            logger.info(
                "Cleared %s old groups and %s IN_GROUP memberships",
                cleanup_stats["groups_deleted"],
                cleanup_stats["memberships_deleted"],
            )
            for group in groups:
                # Get proposition IDs for this group
                member_prop_ids = [propositions[i]["prop_id"] for i in group["member_indices"]]

                # Use avg_similarity as score for all members (could be refined)
                similarity_scores = [group["avg_similarity"]] * len(member_prop_ids)

                group_id = _create_proposition_group(
                    client=client,
                    label_text=group["representative_text"],
                    member_prop_ids=member_prop_ids,
                    similarity_scores=similarity_scores,
                    model=embedding_model,
                    version="2024-01",
                    threshold=threshold,
                    method="agglomerative"
                )
                logger.info(f"Created group {group_id} with {group['member_count']} members")

        return {
            "status": "completed",
            "groups_created": len(groups),
            "propositions_clustered": len(propositions)
        }

    except Exception as e:
        logger.error(f"Clustering task failed: {e}", exc_info=True)
        # Re-raise to fail the task properly (don't return success-like payload)
        raise RuntimeError(f"Proposition clustering failed: {str(e)}") from e


def _clear_existing_proposition_groups(client: Neo4jClient) -> dict[str, int]:
    """Remove existing PropositionGroup nodes and IN_GROUP edges before rebuild."""
    with client._driver.session() as session:
        rel_summary = session.run(
            """
            MATCH (:Proposition)-[r:IN_GROUP]->(:PropositionGroup)
            DELETE r
            """
        ).consume()
        group_summary = session.run(
            """
            MATCH (pg:PropositionGroup)
            DETACH DELETE pg
            """
        ).consume()

    return {
        "memberships_deleted": int(rel_summary.counters.relationships_deleted or 0),
        "groups_deleted": int(group_summary.counters.nodes_deleted or 0),
    }


def _create_proposition_group(
    client: Neo4jClient,
    label_text: str,
    member_prop_ids: list[str],
    similarity_scores: list[float],
    model: str,
    version: str,
    threshold: float,
    method: str
) -> str:
    """
    Create a PropositionGroup node and IN_GROUP relationships.

    Args:
        client: Neo4j client
        label_text: Representative text for the group
        member_prop_ids: List of proposition IDs in this group
        similarity_scores: Similarity scores for each member
        model: Embedding model used
        version: Model version
        threshold: Clustering threshold
        method: Clustering method name

    Returns:
        The created group_id
    """
    # Generate group_id from label text (deterministic)
    group_id = hashlib.sha256(label_text.encode("utf-8", errors="ignore")).hexdigest()[:24]

    now = datetime.now(tz=timezone.utc).isoformat()

    # Create group node
    create_group_query = """
    MERGE (pg:PropositionGroup {group_id: $group_id})
    ON CREATE SET
        pg.label_text = $label_text,
        pg.proposition_count = $proposition_count,
        pg.paper_count = 0,
        pg.embedding_model = $model,
        pg.model_version = $version,
        pg.similarity_threshold = $threshold,
        pg.clustering_method = $method,
        pg.build_status = 'ready',
        pg.created_at = $created_at,
        pg.updated_at = $updated_at
    ON MATCH SET
        pg.updated_at = $updated_at,
        pg.proposition_count = $proposition_count
    """

    with client._driver.session() as session:
        session.run(
            create_group_query,
            group_id=group_id,
            label_text=label_text,
            proposition_count=len(member_prop_ids),
            model=model,
            version=version,
            threshold=threshold,
            method=method,
            created_at=now,
            updated_at=now
        )

        # Create IN_GROUP relationships
        for prop_id, score in zip(member_prop_ids, similarity_scores, strict=True):
            rel_query = """
            MATCH (p:Proposition {prop_id: $prop_id})
            MATCH (pg:PropositionGroup {group_id: $group_id})
            MERGE (p)-[r:IN_GROUP]->(pg)
            ON CREATE SET
                r.similarity_score = $score,
                r.model = $model,
                r.version = $version,
                r.added_at = $added_at
            ON MATCH SET
                r.similarity_score = $score,
                r.added_at = $added_at
            """
            session.run(
                rel_query,
                prop_id=prop_id,
                group_id=group_id,
                score=score,
                model=model,
                version=version,
                added_at=now
            )

    return group_id
