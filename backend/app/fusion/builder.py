from __future__ import annotations

from typing import Any

from app.fusion.linking import generate_explains_links


def build_fusion_projection(
    research_moves: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    textbook_relations: list[dict[str, Any]],
    *,
    min_link_score: float = 0.45,
    top_k_per_move: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    links = generate_explains_links(
        research_moves,
        entities,
        min_score=min_link_score,
        top_k_per_move=top_k_per_move,
    )

    nodes_by_id: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    for move in research_moves:
        move_id = str(move.get("move_id") or "").strip()
        if not move_id:
            continue
        nodes_by_id[move_id] = {
            "id": move_id,
            "label": "ResearchMove",
            "paper_id": move.get("paper_id"),
            "paper_source": move.get("paper_source"),
            "role": move.get("role"),
            "act_type": move.get("act_type"),
            "summary": move.get("summary"),
        }

    for ent in entities:
        eid = str(ent.get("entity_id") or "").strip()
        if not eid:
            continue
        nodes_by_id[eid] = {
            "id": eid,
            "label": "KnowledgeEntity",
            "name": ent.get("name"),
            "entity_type": ent.get("entity_type"),
        }

    for rel in textbook_relations:
        s = str(rel.get("start_id") or "").strip()
        t = str(rel.get("end_id") or "").strip()
        if not s or not t:
            continue
        edges.append(
            {
                "type": "RELATES_TO",
                "source": s,
                "target": t,
                "rel_type": rel.get("rel_type"),
            }
        )

    for link in links:
        edges.append(
            {
                "type": "EXPLAINS",
                "source": link["move_id"],
                "target": link["entity_id"],
                "score": link.get("score"),
                "reasons": link.get("reasons"),
                "anchor_ids": link.get("anchor_ids"),
                "evidence_quote": link.get("evidence_quote"),
                "role": link.get("role"),
                "act_type": link.get("act_type"),
            }
        )

    return {
        "nodes": list(nodes_by_id.values()),
        "edges": edges,
    }
